"""
prc.running.hub_running

Responsabilité : Orchestration batch : compositions → dry-run → runs → Parquet

Usage : Toujours lancer depuis prc/ (python -m ...)
"""

import time
import gc

from pathlib import Path
from typing import Dict, List

from utils.data_loading_lite import load_yaml, write_parquet
from running.compositions import generate_compositions, count_compositions
from running.runner import run_single, RunnerError
from featuring.hub_featuring import load_all_configs


def run_batch(
    yaml_path   : Path,
    auto_confirm: bool = False,
    output_dir  : Path = None,
    verbose     : bool = False,
) -> Dict:
    """
    Orchestre batch : compositions → confirm → runs streaming → Parquet incrémental.

    Workflow:
        1. Load config + features config
        2. Confirmation utilisateur (sauf auto_confirm)
        3. Execute — streaming, history jamais en RAM
        4. Parquet incrémental tous les 1000 runs success + final
    """
    if output_dir is None:
        output_dir = Path('data/results')

    print(f"=== PRC Batch Runner ===")
    print(f"Config: {yaml_path}\n")

    config          = load_yaml(yaml_path)
    features_config = load_all_configs()
    phase           = yaml_path.stem

    print(f"Loading features config...")
    print(f"  Layers découverts: {list(features_config.keys())}")
    for layer_name, layer_config in features_config.items():
        n_functions = len(layer_config.get('functions', []))
        print(f"    {layer_name}: {n_functions} functions")
    print()

    if not auto_confirm:
        print(f"RAM : négligeable (streaming — history non matérialisé)")
        print(f"\nContinuer ? (o/n) : ", end='')
        response = input().strip().lower()
        if response != 'o':
            print("Batch annulé")
            return {
                'n_success': 0, 'n_skipped': 0,
                'rows': [], 'skipped': [],
                'stats': {}, 'parquet_path': None,
            }
    else:
        print("[auto_confirm=True] Skip confirmation\n")

    print(f"=== Exécution batch ===")
    total = count_compositions(config)
    print(f"Compositions : {total:,}\n")
    rows    = []
    skipped = []
    i       = 0

    t_batch_start = time.time()
    parquet_path  = None

    for comp in generate_compositions(config):
        try:
            result = run_single(comp, features_config, verbose=verbose)
            rows.append({
                'composition': comp,
                'features'   : result['features'],
                'layers'     : result['layers'],
            })

        except RunnerError as e:
            skipped.append({'composition': comp, 'error': str(e)})
            if verbose:
                print(f"[SKIP] {comp['gamma_id']} × {comp['encoding_id']}: {e}")

        except Exception as e:
            skipped.append({'composition': comp, 'error': f"Featuring failed: {e}"})
            if verbose:
                print(f"[SKIP] Featuring {comp['gamma_id']} × {comp['encoding_id']}: {e}")

        i += 1

        if i % 10 == 0:
            elapsed = time.time() - t_batch_start
            pct     = 100 * i / total
            print(f"Progress: {i}/{total} ({pct:.1f}%) — {elapsed:.1f}s", flush=True)
            gc.collect()

        # Checkpoint parquet tous les 1000 runs success
        if len(rows) > 0 and len(rows) % 1000 == 0:
            parquet_path = write_parquet(rows, phase, output_dir)
            print(f"  ✓ Checkpoint parquet : {len(rows)} runs")

    gc.collect()
    t_batch_total = time.time() - t_batch_start

    if len(rows) > 0:
        parquet_path = write_parquet(rows, phase, output_dir)

    print(f"\n=== Résultats ===")
    print(f"Success : {len(rows)}")
    print(f"Skipped : {len(skipped)}")
    print(f"Temps   : {t_batch_total/60:.1f}min")
    n_features = len(rows[0]['features']) if rows else 0
    print(f"Features: {n_features} scalaires par run")

    return {
        'n_success'   : len(rows),
        'n_skipped'   : len(skipped),
        'rows'        : rows,
        'skipped'     : skipped,
        'stats'       : {'n_success': len(rows), 'n_skipped': len(skipped), 'time_total_s': t_batch_total},
        'parquet_path': parquet_path,
    }