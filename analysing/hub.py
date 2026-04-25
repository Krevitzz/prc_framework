"""
Orchestrateur du layer analysing.

Point d'entrée unique : parquet(s) → analyse → rapports.
Route vers les modules spécialisés avec gestion mémoire explicite :
une seule matrice lourde en RAM à la fois.

@ROLE    Orchestration analysing : parquet(s) → strates → clusters → rapports
@LAYER   analysing

@EXPORTS
  run_analysing(source, cfg_path, ...)  → Dict | point d'entrée unique

@LIFECYCLE
  CREATES  résultats légers (dicts) agrégés cross-strates
  RECEIVES parquet(s) ou pa.Table depuis pool
  PASSES   résultats vers outputs (fichiers sur disque)

@CONFORMITY
  OK   Point d'entrée unique (SD-7)
  OK   Pool transparent si mono-parquet
  OK   Une seule matrice lourde à la fois (SD-3)
  OK   Projection 2D avant libération M_ortho (Q3)
  OK   Namer sur features brutes (point 2)
  OK   Stratification via METADATA AST (point 3)
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from utils.io_v8 import load_yaml

from analysing.data import load_analysing_data, AnalysingData
from analysing.pool import (
    load_pool_config, scan_parquets, merge_parquets,
)
from analysing.stratify import compute_strates, Strate
from analysing.prepare import materialize_and_transform, MatrixMeta
from analysing.clustering import run_clustering
from analysing.namer import ClusterNamer
from analysing.validate import compute_cluster_coherence
from analysing.profile import (
    compute_entity_profiles, detect_universality, detect_encoding_convergence,
)
from analysing.outputs import (
    write_strate_report_json, write_strate_report_txt,
    write_synthesis_json, write_synthesis_txt,
    ClusterVisualizer,
)


# =========================================================================
# CONFIG
# =========================================================================

def _default_cfg_path() -> Path:
    return Path(__file__).parent / 'configs' / 'analysing.yaml'


# =========================================================================
# TRIAGE — séparation par statut
# =========================================================================

def _triage(data: AnalysingData, verbose: bool = True) -> Dict:
    """Sépare OK/OK_TRUNCATED (analyse principale) vs EXPLOSION/COLLAPSED.

    Returns:
        {
            'principal_indices': np.ndarray,
            'pathological_stats': {
                'EXPLOSION': {n, top_gammas, top_encodings},
                'COLLAPSED': {n, top_gammas, top_encodings},
            }
        }
    """
    principal_mask = np.isin(data.run_statuses, ['OK', 'OK_TRUNCATED'])
    principal_indices = np.where(principal_mask)[0]

    pathological_stats = {}
    for status in ('EXPLOSION', 'COLLAPSED'):
        mask = data.run_statuses == status
        n = int(mask.sum())
        if n == 0:
            pathological_stats[status] = {'n': 0, 'top_gammas': [], 'top_encodings': []}
            continue

        indices = np.where(mask)[0]

        # Top gammas
        gammas = data.gamma_ids[indices]
        unique_g, counts_g = np.unique(gammas, return_counts=True)
        top_g = sorted(zip(unique_g, counts_g), key=lambda x: -x[1])[:10]
        total_g = n
        top_gammas = [{'gamma_id': str(g), 'count': int(c), 'total': total_g,
                       'fraction': float(c / total_g)}
                      for g, c in top_g]

        # Top encodings
        encodings = data.encoding_ids[indices]
        unique_e, counts_e = np.unique(encodings, return_counts=True)
        top_e = sorted(zip(unique_e, counts_e), key=lambda x: -x[1])[:10]
        top_encodings = [{'encoding_id': str(e), 'count': int(c), 'total': total_g,
                          'fraction': float(c / total_g)}
                         for e, c in top_e]

        pathological_stats[status] = {
            'n': n, 'top_gammas': top_gammas, 'top_encodings': top_encodings,
        }

    if verbose:
        n_ok = int(np.sum(data.run_statuses == 'OK'))
        n_trunc = int(np.sum(data.run_statuses == 'OK_TRUNCATED'))
        n_exp = pathological_stats.get('EXPLOSION', {}).get('n', 0)
        n_col = pathological_stats.get('COLLAPSED', {}).get('n', 0)
        print(f"\n[triage] Principal : {len(principal_indices)} "
              f"(OK={n_ok}, TRUNC={n_trunc})")
        print(f"[triage] Pathologique : EXP={n_exp}, COL={n_col}")

    return {
        'principal_indices': principal_indices,
        'pathological_stats': pathological_stats,
    }


# =========================================================================
# PROJECTION 2D (optionnel, pendant que M_ortho est en RAM)
# =========================================================================

def _compute_projection(M_ortho: np.ndarray, cfg: Dict) -> Optional[np.ndarray]:
    """Projection 2D (t-SNE ou UMAP) tant que M_ortho est en mémoire."""
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError:
        return None

    try:
        import umap
        _has_umap = True
    except ImportError:
        _has_umap = False

    n = M_ortho.shape[0]
    if n < 5:
        return None

    max_proj = cfg.get('max_projection_samples', 5000)
    umap_threshold = cfg.get('umap_threshold', 50000)

    # Subsample si nécessaire
    if n > max_proj:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_proj, replace=False)
        M_proj = M_ortho[idx]
        print(f"  [projection] Subsampled : {n} → {max_proj}")
    else:
        idx = None
        M_proj = M_ortho

    n_proj = M_proj.shape[0]
    n_pca = min(50, M_proj.shape[1], n_proj - 1)

    if n_pca < 2:
        return None

    M_pca = PCA(n_components=n_pca, random_state=42).fit_transform(M_proj)

    if n_proj >= umap_threshold and _has_umap:
        print(f"  [projection] UMAP 2D (n={n_proj})")
        M_2d_sub = umap.UMAP(n_components=2, random_state=42).fit_transform(M_pca)
    else:
        perplexity = min(50, max(5, int(np.sqrt(n_proj)) // 3))
        print(f"  [projection] t-SNE 2D (n={n_proj}, perplexity={perplexity})")
        M_2d_sub = TSNE(n_components=2, perplexity=perplexity,
                        max_iter=1000, random_state=42).fit_transform(M_pca)

    del M_pca

    if idx is not None:
        M_2d = np.full((n, 2), np.nan)
        M_2d[idx] = M_2d_sub
    else:
        M_2d = M_2d_sub

    return M_2d


# =========================================================================
# TRAITEMENT D'UNE STRATE
# =========================================================================

def _process_strate(data: AnalysingData,
                     strate: Strate,
                     cfg: Dict,
                     namer: ClusterNamer,
                     output_dir: Optional[Path],
                     plot: bool,
                     verbose: bool) -> Dict:
    """Traitement complet d'une strate : prepare → cluster → name → validate → profile.

    Gestion mémoire : M_ortho créé, utilisé, libéré dans cette fonction.
    """
    print(f"\n--- Strate {strate.strate_id} ({strate.n_runs} runs) ---")

    if strate.n_runs < 5:
        print(f"  Pas assez de runs ({strate.n_runs}) — skip")
        return {'metadata': {'strate_id': strate.strate_id,
                             'n_runs': strate.n_runs, 'skipped': True}}

    # --- 5a. Matérialiser M_ortho ---
    M_ortho, feat_names_ortho, matrix_meta = materialize_and_transform(
        data, strate, cfg
    )

    # --- 5b. Projection 2D (tant que M_ortho est en RAM) ---
    M_2d = None
    if plot:
        proj_cfg = cfg.get('projection', {})
        M_2d = _compute_projection(M_ortho, proj_cfg)

    # --- 5c. Clustering ---
    clustering_result = run_clustering(M_ortho, feat_names_ortho, cfg, verbose=verbose)

    # --- 5d. Libérer M_ortho ---
    del M_ortho

    if clustering_result is None:
        return {'metadata': {'strate_id': strate.strate_id,
                             'n_runs': strate.n_runs, 'skipped': True,
                             'reason': 'clustering_failed'}}

    peeling_result = clustering_result['peeling_result']
    labels = clustering_result['labels']

    # --- 5e. Nommer (sur features brutes via AnalysingData) ---
    print(f"\n  [namer] Nommage sur features brutes")
    # Features pour le namer : toutes les applicables (incluant health_*, meta_*)
    named_clusters = namer.name_all(
        peeling_result=peeling_result,
        data=data,
        feature_names=strate.features_applicable,
        run_indices=strate.run_indices,
    )
    _print_naming_summary(named_clusters)

    # --- 5f. Cohérence P1 ↔ clusters ---
    coherence = compute_cluster_coherence(
        data, labels, strate.run_indices, verbose=verbose
    )

    # --- 5g. Profils entités ---
    entity_profiles = compute_entity_profiles(
        data, labels, strate.run_indices, strate.strate_id
    )

    # Universalité
    universal_gammas = detect_universality(
        entity_profiles.get('gamma', []), named_clusters
    )
    convergent_encodings = detect_encoding_convergence(
        entity_profiles.get('encoding', []), named_clusters
    )

    if universal_gammas:
        print(f"  [profile] {len(universal_gammas)} gammas universels détectés")
    if convergent_encodings:
        print(f"  [profile] {len(convergent_encodings)} encodings convergents détectés")

    # --- Visualisation ---
    if plot and M_2d is not None and output_dir:
        try:
            viz = ClusterVisualizer(
                M_2d=M_2d, named_clusters=named_clusters,
                peeling_result=peeling_result, coherence=coherence,
            )
            viz.plot_all(output_dir, strate.strate_id)
        except Exception as e:
            print(f"  WARNING: Visualisation failed ({e})")
    del M_2d

    # --- Assemblage résultat strate ---
    result = {
        'metadata': {
            'strate_id': strate.strate_id,
            'rank_eff': strate.rank_eff,
            'differentiable': strate.differentiable,
            'n_runs': strate.n_runs,
            'n_features_applicable': strate.n_features,
            'n_features_input': matrix_meta.n_features_input,
            'n_features_ortho': matrix_meta.n_features_ortho,
            'n_nan_imputed': matrix_meta.n_nan_imputed,
            'ortho_threshold': matrix_meta.ortho_threshold,
            'features_excluded': strate.features_excluded,
        },
        'clustering': {
            'n_clusters': clustering_result['n_clusters'],
            'n_noise': clustering_result['n_noise'],
            'n_samples': clustering_result['n_samples'],
            'comparison': clustering_result.get('comparison', {}),
            'trace': peeling_result.get('trace', []),
        },
        'named_clusters': named_clusters,
        'coherence': coherence,
        'entity_profiles': {
            k: [{'entity_id': p.entity_id, 'n_total': p.n_runs_total,
                 'n_clustered': p.n_runs_clustered,
                 'dominant_cluster': p.dominant_cluster,
                 'concentration': p.dominant_cluster_fraction,
                 'explosion_rate': p.explosion_rate}
                for p in profiles]
            for k, profiles in entity_profiles.items()
        },
        'universal_gammas': universal_gammas,
        'convergent_encodings': convergent_encodings,
    }

    return result


# =========================================================================
# HELPERS
# =========================================================================

def _print_naming_summary(named_clusters: List[Dict]):
    """Affiche le résumé du nommage."""
    n_named = sum(1 for nc in named_clusters if nc.get('cluster_id', -1) >= 0)
    n_res = sum(1 for nc in named_clusters if nc.get('cluster_id', -1) == -1)
    print(f"  {n_named} clusters nommés, {n_res} résidu")
    for nc in sorted(named_clusters, key=lambda x: x.get('cluster_id', 999)):
        cid = nc.get('cluster_id', '?')
        name = nc.get('name', '?')
        n = nc.get('n', 0)
        homo = nc.get('cluster_homogeneity', 0.0)
        het = ' ⚠' if nc.get('heterogeneous') else ''
        comp = nc.get('composition', {})
        dom = comp.get('dominant_regime', '')
        trunc = comp.get('truncated_fraction', 0)
        trunc_s = f' T={trunc:.0%}' if trunc > 0 else ''
        print(f"    C{cid:>3} ({n:>5} runs) homo={homo:.2f}{het:2s} "
              f"P1={dom}{trunc_s} → {name}")


# =========================================================================
# POINT D'ENTRÉE UNIQUE
# =========================================================================

def run_analysing(source: Union[Path, List[Path]],
                   cfg_path: Optional[Path] = None,
                   output_dir: Optional[Path] = None,
                   pool_yaml: Optional[Path] = None,
                   label: Optional[str] = None,
                   plot: bool = False,
                   verbose: bool = False) -> Dict:
    """Point d'entrée unique du layer analysing.

    Flux :
    1. Pool (si multi-parquets) ou lecture directe
    2. Load → AnalysingData
    3. Triage → principal + pathologique
    4. Stratification → strates
    5. Pour chaque strate : cluster → name → validate → profile
    6. Rapports (par strate + synthèse)

    Args:
        source : chemin(s) parquet ou pa.Table.
        cfg_path : chemin YAML config analysing (None = défaut).
        output_dir : répertoire de sortie (None = data/results/reports).
        pool_yaml : chemin YAML pool pour fusion multi-parquets.
        label : nom du verdict (ex: "test_explosions"). Déduit de source si None.
        plot : générer les PNG.
        verbose : détails du peeling.

    Returns:
        Dict avec résultats par strate, stats pathologiques, synthèse.
    """
    # --- Label ---
    if label is None:
        if isinstance(source, list):
            label = '_'.join(p.stem for p in source[:4])
        elif isinstance(source, (str, Path)):
            p = Path(source)
            label = p.stem if p.is_file() else 'cross'
        else:
            label = 'verdict'

    # --- Config ---
    cfg_path = Path(cfg_path) if cfg_path else _default_cfg_path()
    cfg = load_yaml(cfg_path)
    print(f"\n{'='*60}")
    print(f"ANALYSING — {label}")
    print(f"{'='*60}")

    # --- 1. Pool ou lecture directe ---
    if isinstance(source, list):
        # Multi-parquets
        pool_config = load_pool_config(pool_yaml)
        table = merge_parquets(source, pool_config, verbose=True)
        data = load_analysing_data(table, scope=cfg.get('scope'), verbose=True)
        del table
    elif isinstance(source, Path) or isinstance(source, str):
        source = Path(source)
        if source.is_dir():
            # Répertoire → scan + merge
            paths = scan_parquets(source)
            if not paths:
                print("⚠ Aucun parquet trouvé")
                return {'strates': {}, 'pathological': {}, 'summary': {}}
            pool_config = load_pool_config(pool_yaml)
            table = merge_parquets(paths, pool_config, verbose=True)
            data = load_analysing_data(table, scope=cfg.get('scope'), verbose=True)
            del table
        else:
            # Fichier unique
            data = load_analysing_data(source, scope=cfg.get('scope'), verbose=True)
    else:
        raise ValueError(f"Source non supportée : {type(source)}")

    if data.n_runs == 0:
        print("⚠ Aucune donnée après filtrage")
        return {'strates': {}, 'pathological': {}, 'summary': {}}

    # --- Output dir ---
    if output_dir is None:
        output_dir = Path('data/results/reports')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 3. Triage ---
    triage = _triage(data, verbose=True)
    principal_indices = triage['principal_indices']
    pathological_stats = triage['pathological_stats']

    if len(principal_indices) == 0:
        print("⚠ Aucun run OK/OK_TRUNCATED")
        return {
            'strates': {},
            'pathological': pathological_stats,
            'summary': {'n_principal': 0},
        }

    # --- 4. Stratification ---
    print("\n--- Stratification ---")
    strates = compute_strates(data, principal_indices, verbose=True)

    # --- Namer (chargé une seule fois) ---
    namer = ClusterNamer.from_yaml()

    # --- 5. Traitement par strate ---
    all_strate_results = {}

    for strate in strates:
        strate_output = output_dir / strate.strate_id if output_dir else None
        if strate_output:
            strate_output.mkdir(parents=True, exist_ok=True)

        result = _process_strate(
            data=data,
            strate=strate,
            cfg=cfg,
            namer=namer,
            output_dir=strate_output,
            plot=plot,
            verbose=verbose,
        )
        all_strate_results[strate.strate_id] = result

        # Rapports par strate
        if strate_output:
            write_strate_report_json(
                result, strate_output / f'report_{strate.strate_id}.json'
            )
            write_strate_report_txt(
                result, strate_output / f'report_{strate.strate_id}.txt'
            )

    # --- 6. Synthèse ---
    print(f"\n{'='*60}")
    print("SYNTHÈSE")
    print(f"{'='*60}")

    total_clusters = sum(
        r.get('clustering', {}).get('n_clusters', 0)
        for r in all_strate_results.values()
    )
    total_runs = sum(
        r.get('metadata', {}).get('n_runs', 0)
        for r in all_strate_results.values()
    )
    print(f"  {len(strates)} strates, {total_runs} runs, {total_clusters} clusters")

    for status in ('EXPLOSION', 'COLLAPSED'):
        n = pathological_stats.get(status, {}).get('n', 0)
        if n > 0:
            print(f"  {status} : {n} runs")

    # Rapports synthèse
    write_synthesis_json(all_strate_results, pathological_stats,
                          output_dir / f'verdict_{label}.json')
    write_synthesis_txt(all_strate_results, pathological_stats,
                         output_dir / f'verdict_{label}.txt',
                         label=label)

    return {
        'strates': all_strate_results,
        'pathological': pathological_stats,
        'summary': {
            'n_strates': len(strates),
            'n_principal': len(principal_indices),
            'total_clusters': total_clusters,
            'total_runs': total_runs,
        },
    }
