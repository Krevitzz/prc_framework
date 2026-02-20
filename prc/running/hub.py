"""
prc.running.hub

Responsabilité : Orchestration batch : compositions → benchmark → dry-run → runs → Parquet

Usage : Toujours lancer depuis prc/ (python -m ...)
"""

import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Set

from utils.data_loading_lite import load_yaml
from running.compositions import generate_compositions
from running.runner import run_single, RunnerError
from featuring.hub_lite import extract_features


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

def run_batch(yaml_path: Path, auto_confirm: bool = False) -> Dict:
    """
    Orchestre batch : compositions → benchmark → confirm → runs → Parquet.
    
    Workflow:
        1. Generate compositions
        2. Benchmark samples (1 par rank)
        3. Dry-run stats (temps moyen/max/total, RAM moyen/max)
        4. Confirmation utilisateur (sauf si auto_confirm=True)
        5. Execute avec progress
        6. Write Parquet (stub)
    
    Args:
        yaml_path    : Path vers YAML de run
        auto_confirm : Si True, skip confirmation (pour tests)
    
    Returns:
        {
            'n_success': int,
            'n_skipped': int,
            'results': List[Dict],
            'skipped': List[Dict],
            'stats': Dict (estimations),
        }
    
    Examples:
        >>> result = run_batch(Path('configs/phases/poc/poc.yaml'))
        >>> result['n_success']
        252
    """
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
            'results': [],
            'skipped': [],
            'stats': {},
        }
    
    # 1b. Load features config
    features_config_path = Path('configs/features/minimal/universal.yaml')
    print(f"Loading features config: {features_config_path}")
    universal_config = load_yaml(features_config_path)
    features_config = {'universal': universal_config}
    print(f"Features config loaded: {len(universal_config.get('functions', []))} functions\n")
    
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
                'results': [],
                'skipped': [],
                'stats': stats,
            }
    else:
        print("\n[auto_confirm=True] Skip confirmation\n")
    
    # 5. Execute batch
    print(f"=== Exécution batch ===")
    results = []
    skipped = []
    
    t_batch_start = time.time()
    
    for i, comp in enumerate(compositions):
        try:
            # Run kernel
            history = run_single(comp)
            
            # Extract features
            features = extract_features(history, features_config)
            
            # Store result
            results.append({
                'composition': comp,
                'history': history,
                'features': features,
            })
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
            results.append({
                'composition': comp,
                'history': history,
                'features': {},  # Empty features but keep run
            })
        
        if (i+1) % 10 == 0 or (i+1) == len(compositions):
            elapsed = time.time() - t_batch_start
            progress_pct = 100 * (i+1) / len(compositions)
            print(f"Progress: {i+1}/{len(compositions)} ({progress_pct:.1f}%) — {elapsed:.1f}s")
    
    t_batch_total = time.time() - t_batch_start
    
    # 6. Write Parquet (stub)
    print(f"\n=== Résultats ===")
    print(f"Success: {len(results)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Temps total: {t_batch_total/60:.1f}min")
    
    # Afficher sample features
    if len(results) > 0:
        sample_features = results[0]['features']
        print(f"\nFeatures extraites (sample): {len(sample_features)}")
        for key, value in list(sample_features.items())[:3]:
            print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print(f"\n⚠️  TODO: Write Parquet")
    
    return {
        'n_success': len(results),
        'n_skipped': len(skipped),
        'results': results,
        'skipped': skipped,
        'stats': stats,
    }