"""
prc.analysing.hub_analysing

Responsabilité : Orchestration analysing — peeling → namer → visualizer

Pipeline :
    1. prepare_matrix()         — sanitize, log, scale, ortho (clustering_lite)
    2. compute_tsne()           — projection 2D pour visualisations
    3. run_peeling()            — clustering résiduel progressif (clustering_peeling)
    4. ClusterNamer.name_all()  — nommage compositionnel (cluster_namer)
    5. ClusterVisualizer        — sorties PNG (visualizer)

Connexions verdict :
    run_analysing() retourne un dict complet consommé par verdict.py.
    Les régimes (issus de regimes_lite, en transition) sont passés
    en paramètre optionnel pour enrichir le namer — pas de dépendance circulaire.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from analysing.clustering_lite import (
    prepare_matrix,
    compute_tsne,
    run_clustering_stratified,
    run_clustering,
)
from analysing.clustering_peeling import load_config as load_peeling_config
from analysing.cluster_namer import ClusterNamer, build_layer_distribution
from analysing.visualizer import ClusterVisualizer


# =============================================================================
# ORCHESTRATION PRINCIPALE
# =============================================================================

def run_analysing(
    rows              : List[Dict],
    n_clusters        : int = 3,          # ignoré — HDBSCAN découvre seul (rétrocompat)
    stratified        : bool = True,
    run_regimes       : Optional[List[str]] = None,
    peeling_cfg_path  : Optional[str] = None,
    namer_cfg_path    : Optional[str] = None,
    tsne_cache_path   : Optional[str] = None,
    output_dir        : Optional[str] = None,
    label             : str = 'analysing',
    plot              : bool = True,
) -> Dict:
    """
    Analyse patterns ML complète.

    Args:
        rows             : Liste {composition, features, layers}
        n_clusters       : Ignoré — conservé pour rétrocompatibilité avec verdict.py
        stratified       : Si True, une passe peeling par layer (recommandé)
        run_regimes      : Régimes par run (List[str], optionnel) — enrichit le namer
        peeling_cfg_path : Chemin YAML config peeling
        namer_cfg_path   : Chemin YAML config cluster_namer
        tsne_cache_path  : Cache .npy t-SNE (évite recalcul)
        output_dir       : Répertoire sorties PNG + trace
        label            : Préfixe fichiers
        plot             : Générer PNG (False pour tests rapides)

    Returns:
        {
            'n_observations'        : int,
            'strategy'              : str,
            'clustering'            : Dict,         # résultat run_clustering (unified)
            'clustering_stratified' : Dict,         # {layer: result} si stratified
            'named_clusters'        : List[Dict],   # sortie ClusterNamer
            'layer_distribution'    : Dict,         # percentile base (namer)
            'M_2d'                  : np.ndarray,   # t-SNE coords (None si non calculé)
            'metadata'              : Dict,
        }
    """
    print(f"\n=== Analysing patterns ML ===")
    print(f"Observations : {len(rows)}")

    n = len(rows)
    if n == 0:
        return _empty_result()

    # ── Config ───────────────────────────────────────────────────────────────
    peeling_config = _load_peeling_cfg(peeling_cfg_path)
    namer_config   = _load_namer_cfg(namer_cfg_path)

    # ── Matrice + t-SNE ──────────────────────────────────────────────────────
    print("\nPréparation matrice...")
    M_ortho, kept_names, matrix_meta = prepare_matrix(rows)

    if M_ortho.shape[0] < 5:
        print("  WARNING: Pas assez de samples")
        return _empty_result()

    M_2d = None
    if plot or tsne_cache_path:
        M_2d = compute_tsne(M_ortho, cache_path=tsne_cache_path)

    # ── Clustering ───────────────────────────────────────────────────────────
    print(f"\nMode : {'stratified by layers' if stratified else 'unified'}")

    clustering_unified     = None
    clustering_stratified  = {}

    if stratified:
        clustering_stratified = run_clustering_stratified(
            rows             = rows,
            peeling_config   = peeling_config,
            run_regimes      = run_regimes,
            M_2d             = M_2d,
            output_dir       = output_dir,
            label            = label,
        )
        # Résultat principal = premier layer (rétrocompat verdict)
        clustering_unified = next(iter(clustering_stratified.values()), None)
    else:
        clustering_unified = run_clustering(
            rows           = rows,
            peeling_config = peeling_config,
            run_regimes    = run_regimes,
            M_2d           = M_2d,
            output_dir     = output_dir,
            label          = label,
        )

    if clustering_unified is None:
        print("  WARNING: Clustering failed")
        return _empty_result()

    peeling_result = clustering_unified.get('peeling_result', {})

    # ── Nommage clusters ─────────────────────────────────────────────────────
    print("\nNommage clusters...")
    all_features    = [row['features'] for row in rows]
    layer_dist      = build_layer_distribution(all_features)
    namer           = ClusterNamer(namer_config)
    named_clusters  = namer.name_all(peeling_result, all_features)

    _print_naming_summary(named_clusters)

    # ── Visualisations ───────────────────────────────────────────────────────
    if plot and M_2d is not None and output_dir:
        print("\nVisualisation...")
        gammas = [row['composition'].get('gamma_id', '?') for row in rows]
        viz = ClusterVisualizer(
            M_2d           = M_2d,
            named_clusters = named_clusters,
            peeling_result = peeling_result,
            run_regimes    = run_regimes,
            gammas         = gammas,
        )
        viz.plot_all(output_dir=output_dir, label=label)

    return {
        'n_observations'       : n,
        'strategy'             : 'stratified' if stratified else 'unified',
        'clustering'           : clustering_unified,
        'clustering_stratified': clustering_stratified,
        'named_clusters'       : named_clusters,
        'layer_distribution'   : layer_dist,
        'M_2d'                 : M_2d,
        'metadata'             : {
            'n_features'       : matrix_meta.get('n_features_ortho', 0),
            'n_features_total' : matrix_meta.get('n_features_total', 0),
            'n_clusters'       : peeling_result.get('n_clusters', 0),
            'n_unresolved'     : peeling_result.get('n_unresolved', 0),
            'stratified'       : stratified,
            'label'            : label,
        },
    }


# =============================================================================
# HELPERS INTERNES
# =============================================================================

def _empty_result() -> Dict:
    return {
        'n_observations'       : 0,
        'strategy'             : 'none',
        'clustering'           : None,
        'clustering_stratified': {},
        'named_clusters'       : [],
        'layer_distribution'   : {},
        'M_2d'                 : None,
        'metadata'             : {},
    }


def _load_peeling_cfg(cfg_path: Optional[str]) -> Dict:
    """Charge config peeling — chemin explicite ou défaut."""
    candidates = [
        cfg_path,
        'analysing/configs/clustering_peeling.yaml',
        'tests/clustering_peeling.yaml',
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return load_peeling_config(path)
    raise FileNotFoundError(
        "Config peeling introuvable. Placer clustering_peeling.yaml dans "
        "analysing/configs/ ou passer peeling_cfg_path."
    )


def _load_namer_cfg(cfg_path: Optional[str]) -> Dict:
    """Charge config namer — chemin explicite ou défaut."""
    import yaml
    candidates = [
        cfg_path,
        'analysing/configs/cluster_namer.yaml',
        'tests/cluster_namer.yaml',
    ]
    for path in candidates:
        if path and os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f)['namer']
    raise FileNotFoundError(
        "Config namer introuvable. Placer cluster_namer.yaml dans "
        "analysing/configs/ ou passer namer_cfg_path."
    )


def _print_naming_summary(named_clusters: List[Dict]):
    """Affiche résumé compact des clusters nommés."""
    n_named   = sum(1 for nc in named_clusters if nc.get('cluster_id', -1) >= 0)
    n_residue = sum(1 for nc in named_clusters if nc.get('cluster_id', -1) == -1)
    print(f"  {n_named} clusters nommés, {n_residue} résidu")
    for nc in sorted(named_clusters, key=lambda x: x.get('cluster_id', 999)):
        cid  = nc.get('cluster_id', '?')
        name = nc.get('name', '?')
        n    = nc.get('n', 0)
        homo = nc.get('cluster_homogeneity', 0.0)
        het  = ' ⚠' if nc.get('heterogeneous') else ''
        print(f"    C{cid:>3} ({n:>4} runs) homo={homo:.2f}{het:2s} → {name}")
