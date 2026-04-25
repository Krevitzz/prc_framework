"""
Point d'entrée + orchestration analysing v8.

Pipeline absorbé : data → profiling → outliers → prepare_matrix →
clustering → namer → visualisation → verdict. Routing pur vers les
3 modules spécialisés (data_v8, clustering_v8, outputs_v8).

@ROLE    Orchestration analysing : parquet → analyse → rapports
@LAYER   analysing

"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from utils.io_v8 import load_yaml
from analysing.data_v8 import load_analysing_data, prepare_matrix, AnalysingData
from analysing.clustering_v8 import (
    run_profiling, analyze_outliers, run_clustering, ClusterNamer,
)
from analysing.outputs_v8 import (
    write_verdict_report, write_verdict_report_txt,
    ClusterVisualizer,
)

try:
    from analysing.data_v8 import compute_projection
except ImportError:
    compute_projection = None


# =========================================================================
# DEFAULT CONFIG
# =========================================================================

def _default_cfg_path():
    return Path(__file__).parent / 'configs' / 'analysing_v8.yaml'


# =========================================================================
# SCAN PARQUETS
# =========================================================================

def scan_major_phases(results_dir=None):
    if results_dir is None:
        results_dir = Path('data/results')
    if not Path(results_dir).exists():
        return []
    return sorted(Path(results_dir).glob('*.parquet'))


# =========================================================================
# PIPELINE (ex pipeline_v8.py — absorbé ici)
# =========================================================================

def _empty_result():
    return {
        'n_observations': 0, 'profiling': {}, 'outliers': {},
        'clustering': None, 'named_clusters': [], 'metadata': {},
    }


def _print_naming_summary(named_clusters):
    n_named = sum(1 for nc in named_clusters if nc.get('cluster_id', -1) >= 0)
    n_res = sum(1 for nc in named_clusters if nc.get('cluster_id', -1) == -1)
    print(f"  {n_named} clusters nommés, {n_res} résidu")
    for nc in sorted(named_clusters, key=lambda x: x.get('cluster_id', 999)):
        cid = nc.get('cluster_id', '?')
        name = nc.get('name', '?')
        n = nc.get('n', 0)
        homo = nc.get('cluster_homogeneity', 0.0)
        het = ' ⚠' if nc.get('heterogeneous') else ''
        print(f"    C{cid:>3} ({n:>5} runs) homo={homo:.2f}{het:2s} → {name}")


def run_analysing_pipeline(data, cfg, output_dir=None, label='verdict',
                            plot=False, save_debug=False):
    """Pipeline analysing complet v8."""
    print(f"\n=== Analysing v8 — {label} ===")
    print(f"Observations : {data.n}")

    if data.n == 0:
        return _empty_result()

    # 1. Profiling
    print("\n--- Profiling ---")
    profiling_results = run_profiling(data)

    # 2. Outliers
    print("\n--- Outliers ---")
    contamination = cfg.get('outliers', {}).get('contamination', 0.1)
    outliers_results = analyze_outliers(data, contamination=contamination)
    print(f"  Outliers : {outliers_results['n_outliers']} "
          f"({outliers_results['outlier_fraction'] * 100:.1f}%)")

    # 3. Matrice — UNE SEULE FOIS
    print("\n--- Préparation matrice ---")
    M_ortho, feat_names, nan_mask, matrix_meta = prepare_matrix(data, cfg)

    # del data.M si gros dataset
    if data.n > cfg.get('del_M_threshold', 10000):
        data.M = None

    if M_ortho.shape[0] < 5:
        print("  WARNING: Pas assez de samples")
        return _empty_result()

    # 4. Projection (si plot)
    M_2d = None
    if plot and output_dir:
        from analysing.data_v8 import compute_projection
        proj_cache = str(Path(output_dir) / f'{label}_proj.npy')
        M_2d = compute_projection(M_ortho, cfg=cfg.get('projection', {}),
                                   cache_path=proj_cache)

    # 5. Clustering
    print("\n--- Clustering ---")
    peeling_cfg = cfg.get('peeling')
    if peeling_cfg is None:
        raise ValueError("cfg['peeling'] manquant")

    clustering_result = run_clustering(
        M_ortho, feat_names, peeling_cfg,
        verbose=peeling_cfg.get('verbose', False),
    )
    if clustering_result is None:
        return _empty_result()

    peeling_result = clustering_result['peeling_result']

    # 6. Nommage
    print("\n--- Nommage clusters ---")
    namer = ClusterNamer.from_yaml()
    named_clusters = namer.name_all(peeling_result, M_ortho, feat_names)
    _print_naming_summary(named_clusters)

    # 7. del M_ortho
    del M_ortho

    # 8. Visualisation
    if plot and M_2d is not None and output_dir:
        print("\n--- Visualisation ---")
        try:
            viz = ClusterVisualizer(
                M_2d=M_2d, named_clusters=named_clusters,
                peeling_result=peeling_result,
                run_regimes=None, gammas=data.gamma_ids.tolist(),
            )
            viz.plot_all(output_dir=str(output_dir), label=label)
        except Exception as e:
            print(f"  WARNING: Visualisation failed ({e})")
        del M_2d

    # 9. Assemblage
    metadata = {
        'n_observations': data.n,
        'n_gammas': len(profiling_results.get('gamma', {})),
        'n_encodings': len(profiling_results.get('encoding', {})),
        'n_modifiers': len(profiling_results.get('modifier', {})),
        'n_clusters': clustering_result.get('n_clusters', 0),
        'n_unresolved': clustering_result.get('n_noise', 0),
        'n_features_ortho': matrix_meta.get('n_features_ortho', 0),
        'n_features_total': matrix_meta.get('n_features_total', 0),
        'n_nan_imputed': matrix_meta.get('n_nan_imputed', 0),
        'ortho_threshold': matrix_meta.get('ortho_threshold'),
        'label': label,
    }

    print(f"\n✓ Pipeline terminé — {metadata['n_clusters']} clusters, "
          f"{metadata['n_unresolved']} résidu")

    return {
        'n_observations': data.n,
        'profiling': profiling_results,
        'outliers': outliers_results,
        'clustering': clustering_result,
        'named_clusters': named_clusters,
        'metadata': metadata,
    }


# =========================================================================
# VERDICT SINGLE-PHASE
# =========================================================================

def run_verdict_from_parquet(parquet_path, cfg_path=None, output_dir=None,
                              label=None, plot=False, save_debug=False):
    parquet_path = Path(parquet_path)
    cfg_path = Path(cfg_path) if cfg_path else _default_cfg_path()
    cfg = load_yaml(cfg_path)
    _label = label or parquet_path.stem

    print(f"\n=== Verdict v8 : {parquet_path.name} ===")

    scope = cfg.get('scope', {})
    apply_pool = cfg.get('pool_requirements', {}).get('apply', False)
    data = load_analysing_data(parquet_path, scope=scope, apply_pool=apply_pool)

    if data.n == 0:
        print("⚠ Aucune row après filtrage")
        return {'metadata': {'n_observations': 0, 'label': _label}}

    _output_dir = Path(output_dir) if output_dir else Path('data/results/reports') / _label
    _output_dir.mkdir(parents=True, exist_ok=True)

    result = run_analysing_pipeline(
        data=data, cfg=cfg, output_dir=_output_dir,
        label=_label, plot=plot, save_debug=save_debug,
    )

    write_verdict_report(result, _output_dir / f'verdict_{_label}.json')
    write_verdict_report_txt(result, _output_dir / f'verdict_{_label}.txt')
    return result


# =========================================================================
# VERDICT CROSS-PHASES
# =========================================================================

def run_verdict_cross_phases(results_dir=None, cfg_path=None,
                              output_dir=None, plot=False):
    cfg_path = Path(cfg_path) if cfg_path else _default_cfg_path()
    cfg = load_yaml(cfg_path)
    parquet_paths = scan_major_phases(results_dir)

    print(f"\n=== Verdict cross-phases v8 ===")
    if not parquet_paths:
        print("⚠ Aucun parquet trouvé")
        return {'phases': {}, 'metadata': {'n_phases': 0}}

    print(f"Phases : {[p.stem for p in parquet_paths]}")

    scope = cfg.get('scope', {})
    apply_pool = cfg.get('pool_requirements', {}).get('apply', False)

    phases_results = {}
    for pp in parquet_paths:
        phase_name = pp.stem
        print(f"\n--- Phase {phase_name} ---")
        data = load_analysing_data(pp, scope=scope, apply_pool=apply_pool)
        if data.n == 0:
            continue
        out_phase = Path(output_dir) / phase_name if output_dir else None
        phases_results[phase_name] = run_analysing_pipeline(
            data=data, cfg=cfg, output_dir=out_phase, label=phase_name, plot=plot)

    result = {
        'phases': phases_results,
        'metadata': {'n_phases': len(parquet_paths),
                     'phases_analyzed': [p.stem for p in parquet_paths]},
    }
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_verdict_report(result, out / 'verdict_cross_phases.json')
        write_verdict_report_txt(result, out / 'verdict_cross_phases.txt')
    return result
