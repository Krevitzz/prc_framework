"""
analysing/pipeline.py

Responsabilité : Orchestration pipeline analysing v7.

Reçoit AnalysingData — une seule matrice numpy, pas de List[Dict].
prepare_matrix appelée une seule fois — M_ortho partagé par clustering et namer.

Enchaînement :
    1. run_profiling(data)
    2. analyze_outliers(data, cfg)
    3. M_ortho, feat_names, matrix_meta = prepare_matrix(data, corr_threshold)
    4. del data.M  ← libération explicite si mémoire critique
    5. compute_tsne(M_ortho)  ← si plot
    6. run_clustering(M_ortho, feat_names, data.n, cfg)
    7. namer.name_all(peeling_result, M_ortho, feat_names)
    8. viz reçoit M_2d, labels, data.gamma_ids
    9. Assemblage dict résultat
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from analysing.parquet_filter import AnalysingData


def _empty_result() -> Dict:
    return {
        'n_observations': 0,
        'profiling'     : {},
        'outliers'      : {},
        'clustering'    : None,
        'named_clusters': [],
        'M_2d'          : None,
        'metadata'      : {},
    }


def run_analysing_pipeline(
    data      : AnalysingData,
    cfg       : Dict,
    output_dir: Optional[Path] = None,
    label     : str  = 'verdict',
    plot      : bool = True,
    save_debug: bool = False,
) -> Dict:
    """
    Pipeline analysing complet.

    Args:
        data       : AnalysingData depuis parquet_filter.load_analysing_data()
        cfg        : Config YAML analysing complet
        output_dir : Répertoire PNG + debug
        label      : Préfixe fichiers
        plot       : Générer PNG t-SNE
        save_debug : Sauvegarder labels + trace peeling JSON

    Returns:
        dict résultat pipeline
    """
    from analysing.profiling_lite     import run_profiling
    from analysing.outliers_lite_new  import analyze_outliers
    from analysing.clustering_lite_new import prepare_matrix, compute_tsne, run_clustering
    from analysing.cluster_namer_new  import ClusterNamer, build_layer_distribution
    from analysing.visualizer         import ClusterVisualizer

    print(f"\n=== Analysing pipeline — {label} ===")
    print(f"Observations : {data.n}")

    if data.n == 0:
        return _empty_result()

    # ── 1. Profiling ─────────────────────────────────────────────────────────
    print("\n--- Profiling ---")
    profiling_results = run_profiling(data)

    # ── 2. Outliers ──────────────────────────────────────────────────────────
    print("\n--- Outliers ---")
    contamination    = cfg.get('outliers', {}).get('contamination', 0.1)
    outliers_results = analyze_outliers(data, contamination=contamination)
    print(f"  Outliers : {outliers_results['n_outliers']} "
          f"({outliers_results['outlier_fraction']*100:.1f}%)")

    # ── 3. Matrice — appelée UNE SEULE FOIS ──────────────────────────────────
    print("\n--- Préparation matrice ---")
    corr_threshold              = cfg.get('orthogonalization', {}).get('correlation_threshold', 0.85)
    M_ortho, feat_names, matrix_meta = prepare_matrix(data, corr_threshold=corr_threshold)

    if M_ortho.shape[0] < 5:
        print("  WARNING: Pas assez de samples")
        return _empty_result()

    # ── 4. t-SNE ─────────────────────────────────────────────────────────────
    M_2d = None
    if plot and output_dir:
        tsne_cache = str(Path(output_dir) / f'{label}_tsne.npy')
        M_2d       = compute_tsne(M_ortho, cache_path=tsne_cache)

    # ── 5. Clustering peeling ─────────────────────────────────────────────────
    print("\n--- Clustering ---")
    peeling_cfg = cfg.get('peeling')
    if peeling_cfg is None:
        raise ValueError("cfg['peeling'] manquant — vérifier analysing_default.yaml")

    clustering_result = run_clustering(
        M_ortho    = M_ortho,
        feat_names = feat_names,
        n          = data.n,
        peeling_cfg= peeling_cfg,
        M_2d       = M_2d,
        output_dir = str(output_dir) if output_dir else None,
        label      = label,
        save_debug = save_debug,
    )

    if clustering_result is None:
        print("  WARNING: Clustering failed")
        return _empty_result()

    peeling_result = clustering_result['peeling_result']

    # ── 6. Nommage clusters ──────────────────────────────────────────────────
    print("\n--- Nommage clusters ---")
    namer_cfg = cfg.get('namer')
    if namer_cfg is None:
        raise ValueError("cfg['namer'] manquant — vérifier analysing_default.yaml")

    namer          = ClusterNamer(namer_cfg)
    named_clusters = namer.name_all(peeling_result, M_ortho, feat_names)

    _print_naming_summary(named_clusters)

    # ── 7. Visualisations ─────────────────────────────────────────────────────
    if plot and M_2d is not None and output_dir:
        print("\n--- Visualisation ---")
        try:
            viz = ClusterVisualizer(
                M_2d           = M_2d,
                named_clusters = named_clusters,
                peeling_result = peeling_result,
                run_regimes    = None,
                gammas         = data.gamma_ids.tolist(),
            )
            viz.plot_all(output_dir=str(output_dir), label=label)
        except Exception as e:
            print(f"  WARNING: Visualisation failed ({e})")

    # ── 8. Assemblage résultat ────────────────────────────────────────────────
    metadata = {
        'n_observations'  : data.n,
        'n_gammas'        : len(profiling_results.get('gamma', {})),
        'n_encodings'     : len(profiling_results.get('encoding', {})),
        'n_modifiers'     : len(profiling_results.get('modifier', {})),
        'n_clusters'      : clustering_result.get('n_clusters', 0),
        'n_unresolved'    : clustering_result.get('n_noise', 0),
        'n_features_ortho': matrix_meta.get('n_features_ortho', 0),
        'n_features_total': matrix_meta.get('n_features_total', 0),
        'n_named_clusters': len([
            nc for nc in named_clusters if nc.get('cluster_id', -1) >= 0
        ]),
        'label'           : label,
    }

    print(f"\n✓ Pipeline terminé — {metadata['n_clusters']} clusters, "
          f"{metadata['n_unresolved']} résidu")

    return {
        'n_observations': data.n,
        'profiling'     : profiling_results,
        'outliers'      : outliers_results,
        'clustering'    : clustering_result,
        'named_clusters': named_clusters,
        'M_2d'          : M_2d,
        'metadata'      : metadata,
    }


def _print_naming_summary(named_clusters: List[Dict]):
    n_named   = sum(1 for nc in named_clusters if nc.get('cluster_id', -1) >= 0)
    n_residue = sum(1 for nc in named_clusters if nc.get('cluster_id', -1) == -1)
    print(f"  {n_named} clusters nommés, {n_residue} résidu")
    for nc in sorted(named_clusters, key=lambda x: x.get('cluster_id', 999)):
        cid  = nc.get('cluster_id', '?')
        name = nc.get('name', '?')
        n    = nc.get('n', 0)
        homo = nc.get('cluster_homogeneity', 0.0)
        het  = ' ⚠' if nc.get('heterogeneous') else ''
        print(f"    C{cid:>3} ({n:>5} runs) homo={homo:.2f}{het:2s} → {name}")
