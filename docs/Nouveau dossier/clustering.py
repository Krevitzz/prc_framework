"""
Clustering par peeling HDBSCAN multi-niveaux + IsolationForest.

Le peeling extrait les clusters par homogénéité décroissante : les clusters
les plus cohérents sont extraits d'abord, le résidu est re-clusteré à chaque
niveau avec des paramètres affinés.

IsolationForest est conservé comme outil de vérification complémentaire
(comparé au résidu du peeling).

@ROLE    Clustering : peeling HDBSCAN + IsolationForest (vérification)
@LAYER   analysing

@EXPORTS
  run_peeling(M_ortho, cfg)              → Dict | peeling multi-niveaux
  run_isolation_forest(M_ortho, cfg)     → Dict | outlier detection
  run_clustering(M_ortho, feat_names, cfg) → Dict | orchestration clustering

@LIFECYCLE
  CREATES  labels          np.ndarray (n,) int — cluster assignments
  RECEIVES M_ortho         depuis prepare.materialize_and_transform
  PASSES   clustering_result  vers hub → namer, validate, profile

@CONFORMITY
  OK   Peeling v8 conservé (SD-4)
  OK   IsolationForest = vérification, pas filtrage
  OK   Paramètres depuis cfg YAML (P4)
"""

import warnings

import numpy as np
from typing import Dict, List, Optional, Tuple

from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score


# =========================================================================
# PEELING — HELPERS
# =========================================================================

def _threshold(cfg: Dict, level: int) -> float:
    """Seuil d'homogénéité pour le niveau courant (croissant par niveau)."""
    base = cfg['homogeneity']['threshold_base']
    step = cfg['homogeneity']['threshold_step']
    maxi = cfg['homogeneity']['threshold_max']
    return min(maxi, base + level * step)


def _mcs_from_n(n: int, cfg: Dict) -> int:
    """min_cluster_size = max(floor, min(cap, n // divisor))."""
    floor = cfg.get('mcs_floor', 8)
    divisor = cfg.get('mcs_divisor', 30)
    cap = cfg.get('mcs_cap', 500)
    return max(floor, min(cap, n // divisor))


def _ms_from_n(n: int, cfg: Dict) -> int:
    """min_samples = max(floor, mcs // 4)."""
    floor = cfg.get('ms_floor', 3)
    mcs = _mcs_from_n(n, cfg)
    return max(floor, mcs // 4)


def _mcs_residual(cfg: Dict, mcs_global: int, n_total: int,
                   n_residual: int, M_residual: np.ndarray,
                   M_global: np.ndarray) -> int:
    """mcs adaptatif pour le résidu (taille + densité)."""
    floor = cfg.get('mcs_floor', 8)
    adaptive = cfg.get('mcs_adaptive', {})

    if not adaptive.get('enabled', True) or n_residual >= n_total:
        return max(floor, mcs_global)

    size_factor = np.sqrt(n_residual / n_total)

    try:
        spread_res = np.trace(np.cov(M_residual.T)) + 1e-10
        spread_all = np.trace(np.cov(M_global.T)) + 1e-10
        df_min = adaptive.get('density_factor_min', 0.5)
        df_max = adaptive.get('density_factor_max', 2.0)
        density_factor = float(np.clip(
            (n_residual / spread_res) / (n_total / spread_all),
            df_min, df_max
        ))
    except Exception:
        density_factor = 1.0

    return max(floor, int(mcs_global * size_factor / density_factor))


def _homogeneity_score(M_cluster: np.ndarray, proba_cluster: np.ndarray,
                        M_others: Optional[np.ndarray], cfg: Dict) -> float:
    """Score d'homogénéité combiné (probabilité HDBSCAN + silhouette)."""
    hcfg = cfg['homogeneity']
    wp = hcfg['weight_probability']
    ws = hcfg['weight_silhouette']

    s_proba = float(np.mean(proba_cluster)) if len(proba_cluster) > 0 else 0.0

    s_silh = 0.5
    if M_others is not None and len(M_others) >= 2 and len(M_cluster) >= 2:
        max_s = cfg.get('silhouette', {}).get('max_samples', 2000)
        M_all = np.vstack([M_cluster, M_others])
        lbs_all = np.array([0] * len(M_cluster) + [1] * len(M_others))
        if len(M_all) > max_s:
            idx = np.random.RandomState(42).choice(len(M_all), max_s, replace=False)
            M_all, lbs_all = M_all[idx], lbs_all[idx]
        try:
            s_silh = (silhouette_score(M_all, lbs_all) + 1) / 2
        except Exception:
            pass

    return float(np.clip((wp * s_proba + ws * s_silh) / (wp + ws), 0.0, 1.0))


# =========================================================================
# PEELING — UN NIVEAU
# =========================================================================

def _run_level(M_level: np.ndarray, global_idx: np.ndarray,
                M_global: np.ndarray, cfg: Dict, level: int,
                mcs: int, ms: int, configs: List[Dict],
                verbose: bool = False) -> Tuple[Dict, np.ndarray]:
    """Exécute un niveau de peeling : HDBSCAN + extraction par homogénéité."""
    n_level = len(M_level)
    threshold = _threshold(cfg, level)
    best_result = None
    best_score = -np.inf

    for hcfg in configs:
        metric = hcfg['metric']
        selection = hcfg['selection']
        pca_dims = hcfg.get('pca_dims')

        if pca_dims and pca_dims < M_level.shape[1]:
            safe = min(pca_dims, M_level.shape[1] - 1, M_level.shape[0] - 1)
            M_use = (PCA(n_components=max(2, safe), random_state=42)
                     .fit_transform(M_level) if safe >= 2 else M_level)
        else:
            M_use = M_level

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', module='sklearn')
                hdb = HDBSCAN(
                    min_cluster_size=mcs, min_samples=ms,
                    metric=metric, cluster_selection_method=selection
                )
                lbs = hdb.fit_predict(M_use)
                proba = hdb.probabilities_
        except Exception:
            continue

        n_cl = len(set(lbs) - {-1})
        pct_bruit = 100 * (lbs == -1).sum() / n_level

        # Score : favorise 3-10 clusters, pénalise le bruit
        sc = (3.0 if 3 <= n_cl <= 10
              else max(0, 3.0 - 0.5 * min(abs(n_cl - 3), abs(n_cl - 10)))
              ) - 0.03 * pct_bruit

        if sc > best_score:
            best_score = sc
            best_result = (lbs, proba, M_use, metric, selection, pca_dims)

    if best_result is None:
        return {'extracted': [], 'level': level, 'n_extracted': 0,
                'n_input': n_level,
                'n_residual_out': len(global_idx)}, global_idx

    lbs, proba, M_use, metric, selection, pca_dims = best_result
    extracted = []
    residual_local = list(np.where(lbs == -1)[0])

    for cid in sorted(set(lbs) - {-1}):
        local_mask = lbs == cid
        local_idx = np.where(local_mask)[0]
        M_cluster = M_use[local_mask]
        others_mask = (lbs != cid) & (lbs != -1)
        M_others = M_use[others_mask] if others_mask.any() else None

        h_score = _homogeneity_score(M_cluster, proba[local_mask], M_others, cfg)

        if verbose:
            print(f'    C{cid} n={local_mask.sum():3d} homo={h_score:.2f} '
                  f'(seuil={threshold:.2f}) → '
                  f'{"EXTRAIT" if h_score >= threshold else "résidu"}')

        if h_score >= threshold:
            extracted.append({
                'level': level,
                'local_cluster': int(cid),
                'global_indices': global_idx[local_idx].tolist(),
                'n': int(local_mask.sum()),
                'homogeneity': float(h_score),
                'threshold': float(threshold),
                'metric': metric,
                'selection': selection,
                'pca_dims': pca_dims,
            })
        else:
            residual_local.extend(local_idx.tolist())

    residual_global = global_idx[sorted(set(residual_local))]
    return {
        'extracted': extracted, 'level': level, 'metric': metric,
        'selection': selection, 'n_input': n_level,
        'n_extracted': sum(e['n'] for e in extracted),
        'n_residual_out': len(residual_global),
    }, residual_global


# =========================================================================
# PEELING — PRINCIPAL
# =========================================================================

def run_peeling(M_ortho: np.ndarray,
                 cfg: Dict,
                 verbose: bool = False) -> Dict:
    """Peeling HDBSCAN multi-niveaux.

    Algorithme conservé de v8 :
    1. Niveau 0 : HDBSCAN sur la population complète
    2. Extraction des clusters avec homogénéité > seuil
    3. Le résidu est re-clusteré au niveau suivant (seuil croissant)
    4. Retry intra-niveau si résidu > residual_threshold

    Returns:
        {labels, extracted, residual_idx, trace, n_levels, n_clusters, n_unresolved}
    """
    n = len(M_ortho)
    mcs_global = _mcs_from_n(n, cfg)
    ms_global = _ms_from_n(n, cfg)
    max_levels = cfg.get('max_levels', 6)
    floor = cfg.get('mcs_floor', 8)
    residual_threshold = cfg.get('residual_threshold', 0.30)
    resolution_step_factor = cfg.get('resolution_step_factor', 2)
    max_resolution_steps = cfg.get('max_resolution_steps', 3)

    print(f'  [clustering] Peeling — {n} runs × {M_ortho.shape[1]} features '
          f'(mcs={mcs_global}, ms={ms_global})')

    labels = np.full(n, -1, dtype=int)
    next_label = 0
    all_extracted = []
    trace = []
    residual_idx = np.arange(n)

    l0 = cfg.get('level_0', {'metric': 'cosine', 'selection': 'eom'})
    configs_l0 = [{'metric': l0['metric'], 'selection': l0['selection'],
                   'pca_dims': l0.get('pca_dims')}]
    configs_residual = cfg.get('residual_configs', configs_l0)

    for level in range(max_levels):
        n_res = len(residual_idx)
        if n_res < floor:
            break

        if level == 0:
            configs, mcs, ms = configs_l0, mcs_global, ms_global
        else:
            configs = configs_residual
            mcs = _mcs_residual(cfg, mcs_global, n, n_res,
                                M_ortho[residual_idx], M_ortho)
            ms = max(cfg.get('ms_floor', 3), mcs // 4)

        level_result, new_residual = _run_level(
            M_ortho[residual_idx], residual_idx, M_ortho,
            cfg, level, mcs, ms, configs, verbose
        )

        # Retry résidu
        retry = 0
        while (len(new_residual) / n > residual_threshold and
               mcs > floor and retry < max_resolution_steps):
            retry += 1
            mcs = max(floor, mcs // resolution_step_factor)
            ms = max(cfg.get('ms_floor', 3), mcs // 4)
            lr, nr = _run_level(
                M_ortho[new_residual], new_residual, M_ortho,
                cfg, level, mcs, ms, configs, verbose
            )
            level_result['extracted'].extend(lr['extracted'])
            level_result['n_extracted'] += lr['n_extracted']
            new_residual = nr

        for ci in level_result['extracted']:
            labels[np.array(ci['global_indices'])] = next_label
            ci['final_label'] = next_label
            all_extracted.append(ci)
            next_label += 1

        n_ext = level_result['n_extracted']
        trace.append({
            'level': level, 'n_input': n_res,
            'n_extracted': n_ext, 'n_residual': len(new_residual),
        })
        print(f'  [clustering] Niveau {level} → {n_ext} extraits, '
              f'{len(new_residual)} résidu')

        if level >= 1 and len(trace) >= 2:
            if trace[-1]['n_extracted'] == 0 and trace[-2]['n_extracted'] == 0:
                break
        if n_ext < cfg.get('min_delta_extracted', 1) and level > 0:
            break
        residual_idx = new_residual

    n_unresolved = int((labels == -1).sum())
    print(f'  [clustering] Peeling terminé — {next_label} clusters | '
          f'résidu {n_unresolved} ({100 * n_unresolved / max(n, 1):.0f}%)')

    return {
        'labels': labels,
        'extracted': all_extracted,
        'residual_idx': residual_idx,
        'trace': trace,
        'n_levels': len(trace),
        'n_clusters': next_label,
        'n_unresolved': n_unresolved,
    }


# =========================================================================
# ISOLATION FOREST — vérification complémentaire
# =========================================================================

def run_isolation_forest(M_ortho: np.ndarray,
                          cfg: Dict) -> Dict:
    """IsolationForest pour détection d'outliers.

    Conservé comme outil de vérification — comparé au résidu du peeling.
    Si le résidu peeling ≈ outliers IF → cohérence. Si divergence → à investiguer.

    Returns:
        {outlier_mask, scores, n_outliers, n_inliers, outlier_fraction}
    """
    contamination = cfg.get('contamination', 0.10)

    if M_ortho.shape[0] < 2:
        return {
            'outlier_mask': np.zeros(M_ortho.shape[0], dtype=bool),
            'scores': np.zeros(M_ortho.shape[0]),
            'n_outliers': 0,
            'n_inliers': M_ortho.shape[0],
            'outlier_fraction': 0.0,
        }

    M_clean = M_ortho.copy()
    M_clean[~np.isfinite(M_clean)] = 0.0

    clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    predictions = clf.fit_predict(M_clean)
    scores = clf.score_samples(M_clean)
    del M_clean

    outlier_mask = predictions == -1
    n_outliers = int(outlier_mask.sum())

    return {
        'outlier_mask': outlier_mask,
        'scores': scores,
        'n_outliers': n_outliers,
        'n_inliers': int((~outlier_mask).sum()),
        'outlier_fraction': n_outliers / len(predictions) if len(predictions) > 0 else 0.0,
    }


# =========================================================================
# COMPARAISON PEELING ↔ ISOLATION FOREST
# =========================================================================

def _compare_residual_outliers(peeling_result: Dict,
                                 if_result: Dict,
                                 n_total: int) -> Dict:
    """Compare le résidu du peeling avec les outliers IsolationForest.

    Retourne le taux de recouvrement (Jaccard) et les différences.
    """
    residual_set = set(peeling_result.get('residual_idx', []).tolist()
                       if isinstance(peeling_result.get('residual_idx'), np.ndarray)
                       else peeling_result.get('residual_idx', []))
    outlier_set = set(np.where(if_result['outlier_mask'])[0].tolist())

    intersection = residual_set & outlier_set
    union = residual_set | outlier_set
    jaccard = len(intersection) / len(union) if union else 1.0

    return {
        'n_residual': len(residual_set),
        'n_outliers': len(outlier_set),
        'n_intersection': len(intersection),
        'jaccard_overlap': jaccard,
        'only_residual': len(residual_set - outlier_set),
        'only_outliers': len(outlier_set - residual_set),
    }


# =========================================================================
# ORCHESTRATION CLUSTERING
# =========================================================================

def run_clustering(M_ortho: np.ndarray,
                    feat_names: List[str],
                    cfg: Dict,
                    verbose: bool = False) -> Optional[Dict]:
    """Orchestration clustering : peeling + isolation forest + comparaison.

    Args:
        M_ortho : matrice transformée (n, F).
        feat_names : noms des features ortho.
        cfg : section 'peeling' du YAML + 'outliers'.
        verbose : afficher les détails.

    Returns:
        Dict avec peeling_result, if_result, comparison, labels, n_clusters.
        None si pas assez de samples.
    """
    if M_ortho.shape[0] < 5:
        print(f"  [clustering] Pas assez de samples ({M_ortho.shape[0]})")
        return None

    # Peeling
    peeling_cfg = cfg.get('peeling', cfg)
    peeling_result = run_peeling(M_ortho, peeling_cfg, verbose=verbose)

    # IsolationForest
    outliers_cfg = cfg.get('outliers', {})
    if_result = run_isolation_forest(M_ortho, outliers_cfg)
    print(f"  [clustering] IsolationForest : {if_result['n_outliers']} outliers "
          f"({if_result['outlier_fraction'] * 100:.1f}%)")

    # Comparaison
    comparison = _compare_residual_outliers(
        peeling_result, if_result, M_ortho.shape[0]
    )
    print(f"  [clustering] Recouvrement résidu/outliers : "
          f"Jaccard={comparison['jaccard_overlap']:.2f}")

    return {
        'n_clusters': peeling_result['n_clusters'],
        'n_noise': peeling_result['n_unresolved'],
        'n_samples': M_ortho.shape[0],
        'n_features': len(feat_names),
        'labels': peeling_result['labels'],
        'feat_names': feat_names,
        'peeling_result': peeling_result,
        'if_result': if_result,
        'comparison': comparison,
    }
