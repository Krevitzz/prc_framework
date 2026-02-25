"""
prc.running.hub_running

Responsabilité : Orchestration batch : compositions → benchmark → dry-run → runs → Parquet

Usage : Toujours lancer depuis prc/ (python -m ...)
"""

import time
import gc
import numpy as np

from pathlib import Path
from typing import Dict, List, Set

from utils.data_loading_lite import load_yaml, write_parquet
from running.compositions import generate_compositions
from running.runner import run_single, RunnerError
from featuring.hub_featuring import extract_features, load_all_configs


# =============================================================================
# HELPERS
# =============================================================================

def detect_unique_ranks(compositions: List[Dict]) -> Set[int]:
    """
    Détecte ranks uniques présents dans compositions.
    
    Returns:
        Set de ranks (ex: {2, 3})
    
    Notes:
        - SYM/ASY → rank 2
        - R3 → rank 3
    """
    ranks = set()
    for comp in compositions:
        rank = infer_rank_from_encoding(comp['encoding_id'])
        ranks.add(rank)
    return ranks


def infer_rank_from_encoding(encoding_id: str) -> int:
    """
    Infère rank depuis encoding_id.
    
    Args:
        encoding_id : 'SYM-001', 'ASY-002', 'R3-003', etc.
    
    Returns:
        2 (SYM/ASY) ou 3 (R3)
    """
    if encoding_id.startswith('R3-'):
        return 3
    return 2


def find_first_composition_with_rank(compositions: List[Dict], rank: int) -> Dict:
    """
    Trouve première composition avec rank donné.
    
    Raises:
        ValueError si aucune composition avec rank
    """
    for comp in compositions:
        if infer_rank_from_encoding(comp['encoding_id']) == rank:
            return comp
    raise ValueError(f"Aucune composition avec rank {rank}")


# =============================================================================
# BATCH RUNNER
# =============================================================================

def run_batch(
    yaml_path: Path,
    auto_confirm: bool = False,
    output_dir: Path = None
) -> Dict:
    """
    Orchestre batch : compositions → benchmark → confirm → runs → Parquet.
    
    Workflow:
        1. Generate compositions
        2. Benchmark samples (1 par rank)
        3. Dry-run stats (temps moyen/max/total, RAM moyen/max)
        4. Confirmation utilisateur (sauf si auto_confirm=True)
        5. Execute avec progress (accumule features, libère history)
        6. Write Parquet (traçabilité)
    
    Args:
        yaml_path    : Path vers YAML de run
        auto_confirm : Si True, skip confirmation (pour tests)
        output_dir   : Dossier Parquet (défaut: prc/data/results/)
    
    Returns:
        {
            'n_success': int,
            'n_skipped': int,
            'rows': List[Dict],  # features only
            'skipped': List[Dict],
            'stats': Dict,
            'parquet_path': Path,
        }
    """
    if output_dir is None:
        output_dir = Path('data/results')
    
    print(f"=== PRC Batch Runner ===")
    print(f"Config: {yaml_path}\n")
    
    # 1. Generate compositions
    config = load_yaml(yaml_path)
    compositions = generate_compositions(config)
    print(f"Compositions générées: {len(compositions)}\n")
    
    if len(compositions) == 0:
        print("⚠️  Aucune composition générée — vérifier config YAML")
        return {
            'n_success': 0,
            'n_skipped': 0,
            'rows': [],
            'skipped': [],
            'stats': {},
            'parquet_path': None,
        }
    
    # 1b. Load features config (découverte automatique)
    print(f"Loading features config...")
    features_config = load_all_configs()
    
    print(f"  Layers découverts: {list(features_config.keys())}")
    for layer_name, layer_config in features_config.items():
        n_functions = len(layer_config.get('functions', []))
        print(f"    {layer_name}: {n_functions} functions")
    print()
    
    # 2. Benchmark samples
    print("=== Benchmark samples ===")
    ranks_present = detect_unique_ranks(compositions)
    benchmarks = {}
    
    for rank in sorted(ranks_present):
        sample = find_first_composition_with_rank(compositions, rank)
        
        print(f"Rank {rank} sample: {sample['gamma_id']} × {sample['encoding_id']}")
        
        t_start = time.time()
        history = run_single(sample)
        t_elapsed = time.time() - t_start
        
        ram_mb = history.nbytes / 1e6
        
        benchmarks[rank] = {
            'time_s': t_elapsed,
            'ram_mb': ram_mb,
            'sample_id': f"{sample['gamma_id']} × {sample['encoding_id']}",
        }
        
        print(f"  Temps: {t_elapsed:.3f}s, RAM: {ram_mb:.1f} MB\n")
    
    # 3. Estimer stats batch
    print("=== Estimations batch ===")
    times = []
    rams = []
    
    for comp in compositions:
        rank = infer_rank_from_encoding(comp['encoding_id'])
        bench = benchmarks[rank]
        times.append(bench['time_s'])
        rams.append(bench['ram_mb'])
    
    t_mean = np.mean(times)
    t_max = np.max(times)
    t_total = np.sum(times)
    
    ram_mean = np.mean(rams)
    ram_max = np.max(rams)
    
    stats = {
        'n_compositions': len(compositions),
        'time_mean_s': t_mean,
        'time_max_s': t_max,
        'time_total_s': t_total,
        'ram_mean_mb': ram_mean,
        'ram_max_mb': ram_max,
    }
    
    print(f"Compositions: {len(compositions)}")
    print(f"  Temps moyen : {t_mean:.2f}s")
    print(f"  Temps max   : {t_max:.2f}s")
    print(f"  Temps TOTAL : {t_total/60:.1f}min ({t_total/3600:.2f}h)")
    print(f"  RAM moyenne : {ram_mean/1e3:.2f} Go")
    print(f"  RAM max     : {ram_max/1e3:.2f} Go")
    
    if ram_max > 15e3:
        print(f"\n⚠️  ALERT : RAM max dépasse 15 Go")
    
    # 4. Confirmation
    if not auto_confirm:
        print(f"\nContinuer ? (o/n) : ", end='')
        response = input().strip().lower()
        
        if response != 'o':
            print("Batch annulé")
            return {
                'n_success': 0,
                'n_skipped': 0,
                'rows': [],
                'skipped': [],
                'stats': stats,
                'parquet_path': None,
            }
    else:
        print("\n[auto_confirm=True] Skip confirmation\n")
    
    # 5. Execute batch
    print(f"=== Exécution batch ===")
    rows = []  # Features only (pas history)
    skipped = []
    
    t_batch_start = time.time()
    
    for i, comp in enumerate(compositions):
        try:
            # Run kernel → history
            history = run_single(comp)
            
            # Extract features
            result = extract_features(history, features_config)
            
            # Store features only (libération history immédiate)
            rows.append({
                'composition': comp,
                'features': result['features'],
                'layers': result['layers']  # ← Ajouter tag layers
            })
            
            # Libérer history immédiatement (RAM critique)
            del history
            
            # GC périodique (tous les 10 runs)
            if (i+1) % 10 == 0:
                gc.collect()
            
        except RunnerError as e:
            skipped.append({
                'composition': comp,
                'error': str(e),
            })
            print(f"[SKIP] {comp['gamma_id']} × {comp['encoding_id']}: {e}")
            continue
        except Exception as e:
            # Featuring errors shouldn't crash batch
            print(f"[WARNING] Featuring failed for {comp['gamma_id']} × {comp['encoding_id']}: {e}")
            rows.append({
                'composition': comp,
                'features': {},  # Empty features but keep run
            })
        
        if (i+1) % 10 == 0 or (i+1) == len(compositions):
            elapsed = time.time() - t_batch_start
            progress_pct = 100 * (i+1) / len(compositions)
            print(f"Progress: {i+1}/{len(compositions)} ({progress_pct:.1f}%) — {elapsed:.1f}s")
    
    # GC final
    gc.collect()
    
    t_batch_total = time.time() - t_batch_start
    
    # 6. Write Parquet (traçabilité)
    phase = yaml_path.stem  # poc.yaml → 'poc', poc2.yaml → 'poc2'
    parquet_path = None
    
    if len(rows) > 0:
        parquet_path = write_parquet(rows, phase, output_dir)
    
    # 7. Résultats
    print(f"\n=== Résultats ===")
    print(f"Success: {len(rows)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Temps total: {t_batch_total/60:.1f}min")
    
    # Afficher sample features
    n_features_extracted = len(rows[0]['features']) if len(rows) > 0 else 0
    print(f"\nFeatures extraites: {n_features_extracted} scalaires par run")
    
    return {
        'n_success': len(rows),
        'n_skipped': len(skipped),
        'rows': rows,
        'skipped': skipped,
        'stats': stats,
        'parquet_path': parquet_path,
    }
