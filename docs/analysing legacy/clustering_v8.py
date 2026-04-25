"""
Analyse ML complète et interprétation : profiling, outliers, peeling, nommage.

Profiling agrège par entité (gamma/encoding/modifier). IsolationForest détecte
les anomalies. Peeling HDBSCAN multi-niveaux extrait les clusters par homogénéité
décroissante. ClusterNamer nomme les clusters depuis YAML (data-driven).

@ROLE    Analyse : profiling + outliers + peeling résiduel + nommage compositionnel
@LAYER   analysing
"""

import warnings
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

from utils.io_v8 import load_yaml
from analysing.data_v8 import AnalysingData


# =========================================================================
# PROFILING — agrégation cross-runs par entité
# =========================================================================

def _aggregate_by_entity(M, feat_names, entity_arr):
    unique_ids = np.unique(entity_arr)
    profiles = {}
    for eid in unique_ids:
        mask = entity_arr == eid
        sub = M[mask].astype(np.float64)
        profiles[str(eid)] = {}
        for j, fname in enumerate(feat_names):
            col = sub[:, j]
            col = col[np.isfinite(col)]
            if len(col) == 0:
                continue
            profiles[str(eid)][fname] = {
                'median': float(np.median(col)),
                'q1': float(np.percentile(col, 25)),
                'q3': float(np.percentile(col, 75)),
                'n_runs': len(col),
            }
    return profiles


def run_profiling(data):
    """Profiling cross-runs par gamma, encoding, modifier."""
    if data.n == 0:
        return {'n_observations': 0, 'gamma': {}, 'encoding': {}, 'modifier': {}}
    M, feat_names = data.features_for_ml()
    result = {'n_observations': data.n}
    for entity_arr, label in [
        (data.gamma_ids, 'gamma'),
        (data.encoding_ids, 'encoding'),
        (data.modifier_ids, 'modifier'),
    ]:
        result[label] = _aggregate_by_entity(M, feat_names, entity_arr)
        print(f"  {len(result[label])} {label}s profiled")
    return result


# =========================================================================
# OUTLIERS — IsolationForest
# =========================================================================

def _compute_atomic_recurrence(entity_arr, mask):
    subset = entity_arr[mask]
    n_total = len(subset)
    if n_total == 0:
        return {}
    unique, counts = np.unique(subset, return_counts=True)
    recurrence = {
        str(uid): {'count': int(cnt), 'fraction': float(cnt / n_total),
                    'total_subset': n_total}
        for uid, cnt in zip(unique, counts)
    }
    return dict(sorted(recurrence.items(), key=lambda x: x[1]['fraction'], reverse=True))


def analyze_outliers(data, contamination=0.1):
    if data.n < 2:
        return {'n_outliers': 0, 'n_stables': data.n, 'outlier_fraction': 0.0,
                'outlier_mask': np.zeros(data.n, dtype=bool),
                'stable_mask': np.ones(data.n, dtype=bool),
                'recurrence': {}, 'n_features_used': 0}

    M_ml, feat_names = data.features_for_ml()
    M_clean = M_ml.copy()
    M_clean[~np.isfinite(M_clean)] = 0.0

    clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    predictions = clf.fit_predict(M_clean)
    outlier_mask = predictions == -1
    stable_mask = predictions == 1

    n_outliers = int(outlier_mask.sum())
    n_stables = int(stable_mask.sum())
    n_total = n_outliers + n_stables

    return {
        'n_outliers': n_outliers, 'n_stables': n_stables,
        'outlier_fraction': n_outliers / n_total if n_total > 0 else 0.0,
        'outlier_mask': outlier_mask, 'stable_mask': stable_mask,
        'recurrence': {
            'gamma': _compute_atomic_recurrence(data.gamma_ids, outlier_mask),
            'encoding': _compute_atomic_recurrence(data.encoding_ids, outlier_mask),
            'modifier': _compute_atomic_recurrence(data.modifier_ids, outlier_mask),
        },
        'n_features_used': len(feat_names),
    }


# =========================================================================
# PEELING — HELPERS
# =========================================================================

def _threshold(cfg, level):
    base = cfg['homogeneity']['threshold_base']
    step = cfg['homogeneity']['threshold_step']
    maxi = cfg['homogeneity']['threshold_max']
    return min(maxi, base + level * step)


def _mcs_from_n(n, cfg):
    """mcs = max(floor, min(cap, n // divisor)). Start large, peel reduces."""
    floor = cfg.get('mcs_floor', 8)
    divisor = cfg.get('mcs_divisor', 30)
    cap = cfg.get('mcs_cap', 500)
    return max(floor, min(cap, n // divisor))


def _ms_from_n(n, cfg):
    floor = cfg.get('ms_floor', 3)
    mcs = _mcs_from_n(n, cfg)
    return max(floor, mcs // 4)


def _mcs_residual(cfg, mcs_global, n_total, n_residual, M_residual, M_global):
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
        density_factor = float(np.clip((n_residual / spread_res) / (n_total / spread_all), df_min, df_max))
    except Exception:
        density_factor = 1.0
    return max(floor, int(mcs_global * size_factor / density_factor))


def _homogeneity_score(M_cluster, proba_cluster, M_others, cfg):
    hcfg = cfg['homogeneity']
    wp, ws = hcfg['weight_probability'], hcfg['weight_silhouette']
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

def _run_level(M_level, global_idx, M_global, cfg, level, mcs, ms,
               configs, verbose=False):
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
            M_use = PCA(n_components=max(2, safe), random_state=42).fit_transform(M_level) if safe >= 2 else M_level
        else:
            M_use = M_level

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', module='sklearn')
                hdb = HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                              metric=metric, cluster_selection_method=selection)
                lbs = hdb.fit_predict(M_use)
                proba = hdb.probabilities_
        except Exception:
            continue

        n_cl = len(set(lbs) - {-1})
        pct_b = 100 * (lbs == -1).sum() / n_level
        sc = (3.0 if 3 <= n_cl <= 10 else max(0, 3.0 - 0.5 * min(abs(n_cl - 3), abs(n_cl - 10)))) - 0.03 * pct_b

        if sc > best_score:
            best_score = sc
            best_result = (lbs, proba, M_use, metric, selection, pca_dims)

    if best_result is None:
        return {'extracted': [], 'level': level, 'n_extracted': 0,
                'n_input': n_level, 'n_residual_out': len(global_idx)}, global_idx

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
                  f'(seuil={threshold:.2f}) → {"EXTRAIT" if h_score >= threshold else "résidu"}')

        if h_score >= threshold:
            extracted.append({
                'level': level, 'local_cluster': int(cid),
                'global_indices': global_idx[local_idx].tolist(),
                'n': int(local_mask.sum()), 'homogeneity': float(h_score),
                'threshold': float(threshold), 'metric': metric,
                'selection': selection, 'pca_dims': pca_dims,
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

def run_peeling(M_ortho, cfg, verbose=False):
    n = len(M_ortho)
    mcs_global = _mcs_from_n(n, cfg)
    ms_global = _ms_from_n(n, cfg)
    max_levels = cfg.get('max_levels', 6)
    floor = cfg.get('mcs_floor', 8)
    residual_threshold = cfg.get('residual_threshold', 0.30)
    resolution_step_factor = cfg.get('resolution_step_factor', 2)
    max_resolution_steps = cfg.get('max_resolution_steps', 3)

    print(f'Peeling — {n} runs × {M_ortho.shape[1]} features (mcs={mcs_global}, ms={ms_global})')

    labels = np.full(n, -1, dtype=int)
    next_label = 0
    all_extracted = []
    trace = []
    residual_idx = np.arange(n)

    l0 = cfg.get('level_0', {'metric': 'cosine', 'selection': 'eom'})
    configs_l0 = [{'metric': l0['metric'], 'selection': l0['selection'], 'pca_dims': l0.get('pca_dims')}]
    configs_residual = cfg.get('residual_configs', configs_l0)

    for level in range(max_levels):
        n_res = len(residual_idx)
        if n_res < floor:
            break

        if level == 0:
            configs, mcs, ms = configs_l0, mcs_global, ms_global
        else:
            configs = configs_residual
            mcs = _mcs_residual(cfg, mcs_global, n, n_res, M_ortho[residual_idx], M_ortho)
            ms = max(cfg.get('ms_floor', 3), mcs // 4)

        level_result, new_residual = _run_level(
            M_ortho[residual_idx], residual_idx, M_ortho, cfg, level, mcs, ms, configs, verbose)

        # Retry résidu
        retry = 0
        while (len(new_residual) / n > residual_threshold and
               mcs > floor and retry < max_resolution_steps):
            retry += 1
            mcs = max(floor, mcs // resolution_step_factor)
            ms = max(cfg.get('ms_floor', 3), mcs // 4)
            lr, nr = _run_level(M_ortho[new_residual], new_residual, M_ortho, cfg,
                                level, mcs, ms, configs, verbose)
            level_result['extracted'].extend(lr['extracted'])
            level_result['n_extracted'] += lr['n_extracted']
            new_residual = nr

        for ci in level_result['extracted']:
            labels[np.array(ci['global_indices'])] = next_label
            ci['final_label'] = next_label
            all_extracted.append(ci)
            next_label += 1

        n_ext = level_result['n_extracted']
        trace.append({'level': level, 'n_input': n_res, 'n_extracted': n_ext, 'n_residual': len(new_residual)})
        print(f'  [Peeling] Niveau {level} → {n_ext} extraits, {len(new_residual)} résidu')

        if level >= 1 and trace[-1]['n_extracted'] == 0 and trace[-2]['n_extracted'] == 0:
            break
        if n_ext < cfg.get('min_delta_extracted', 1) and level > 0:
            break
        residual_idx = new_residual

    n_unresolved = int((labels == -1).sum())
    print(f'Peeling terminé — {next_label} clusters | résidu {n_unresolved} ({100 * n_unresolved / max(n,1):.0f}%)')

    return {
        'labels': labels, 'extracted': all_extracted, 'residual_idx': residual_idx,
        'trace': trace, 'n_levels': len(trace), 'n_clusters': next_label, 'n_unresolved': n_unresolved,
    }


def run_clustering(M_ortho, feat_names, peeling_cfg, verbose=False):
    if M_ortho.shape[0] < 5:
        return None
    result = run_peeling(M_ortho, peeling_cfg, verbose=verbose)
    return {
        'n_clusters': result['n_clusters'], 'n_noise': result['n_unresolved'],
        'n_samples': M_ortho.shape[0], 'n_features': len(feat_names),
        'labels': result['labels'].tolist(), 'feat_names': feat_names,
        'peeling_result': result,
    }


# =========================================================================
# NAMER — profil cluster + nommage compositionnel
# =========================================================================

EPS = 1e-10


def build_cluster_profile(cluster_mask, M_ortho, feat_names):
    n_cluster = int(cluster_mask.sum())
    if n_cluster == 0:
        return {}
    sub = M_ortho[cluster_mask].astype(np.float64)
    profil = {'n': n_cluster}
    for j, key in enumerate(feat_names):
        col = sub[:, j]
        valid = col[np.isfinite(col)]
        profil[f'{key}__nan_frac'] = (n_cluster - len(valid)) / n_cluster
        if len(valid) > 0:
            profil[f'{key}__median'] = float(np.median(valid))
            profil[f'{key}__iqr'] = float(np.percentile(valid, 75) - np.percentile(valid, 25))
            profil[f'{key}__mean'] = float(np.mean(valid))
            profil[f'{key}__std'] = float(np.std(valid))
        else:
            for s in ('median', 'iqr', 'mean', 'std'):
                profil[f'{key}__{s}'] = None
    return profil


def build_layer_distribution(M_ortho, feat_names):
    result = {}
    for j, key in enumerate(feat_names):
        col = M_ortho[:, j].astype(np.float64)
        valid = col[np.isfinite(col)]
        if len(valid) > 0:
            result[key] = np.sort(valid)
    return result


def _percentile_rank(value, dist):
    if len(dist) == 0:
        return 0.5
    return int(np.searchsorted(dist, value, side='right')) / len(dist)


def _conf_from_percentile(pct, direction, conf_at_edge=1.0, conf_at_center=0.5):
    if direction == 'lower':
        return float(np.clip(conf_at_edge - (conf_at_edge - conf_at_center) * 2 * pct, 0.0, 1.0))
    elif direction == 'upper':
        return float(np.clip(conf_at_center + (conf_at_edge - conf_at_center) * 2 * (pct - 0.5), 0.0, 1.0))
    return 0.5


def _get_median(profil, feature_key):
    v = profil.get(f'{feature_key}__median')
    return float(v) if v is not None and np.isfinite(v) else None


def _get_std(profil, feature_key):
    v = profil.get(f'{feature_key}__std')
    return float(v) if v is not None and np.isfinite(v) else None


# --- Handlers par mode ---

def _eval_zones(slot_name, slot_cfg, profil, layer_dist):
    feat_primary = slot_cfg.get('feature_primary')
    feat_complex = slot_cfg.get('feature_complex')
    median_val = _get_median(profil, feat_primary) if feat_primary else None
    inf_frac = profil.get('health_has_inf__median', 0.0) or 0.0

    for term_name, term_cfg in slot_cfg.get('terms', {}).items():
        token = term_cfg.get('token', term_name)
        condition = term_cfg.get('condition')
        lo, hi = term_cfg.get('lo'), term_cfg.get('hi')
        direction = term_cfg.get('percentile_direction')
        sentinel = term_cfg.get('sentinel')

        if condition == 'inf_frac_gt_0':
            if inf_frac <= 0.0: continue
            return {'term': token, 'conf': min(1.0, inf_frac * 2),
                    'feature': 'health_has_inf', 'value': inf_frac, 'rule': 'inf_frac', 'slot': slot_name}
        if condition == 'complex_pairs_gt_0':
            if not feat_complex: continue
            cv = _get_median(profil, feat_complex)
            if cv is None or cv <= 0: continue
            dist = layer_dist.get(feat_complex, np.array([cv]))
            return {'term': token, 'conf': _conf_from_percentile(_percentile_rank(cv, dist), 'upper'),
                    'feature': feat_complex, 'value': cv, 'rule': 'complex_pairs', 'slot': slot_name}
        if median_val is None: continue
        if sentinel and abs(median_val - sentinel) < 0.5:
            return {'term': token, 'conf': 1.0, 'feature': feat_primary,
                    'value': median_val, 'rule': 'sentinel', 'slot': slot_name}
        in_zone = (lo is None or median_val >= lo) and (hi is None or median_val < hi)
        if not in_zone: continue
        if direction is None:
            feat_iqr = slot_cfg.get('feature_iqr', feat_primary)
            iqr_val = profil.get(f'{feat_iqr}__iqr', 0.0) or 0.0
            conf = float(np.clip(1.0 - _percentile_rank(iqr_val, layer_dist.get(feat_iqr, np.array([iqr_val]))), 0.3, 1.0))
        else:
            conf = _conf_from_percentile(_percentile_rank(median_val, layer_dist.get(feat_primary, np.array([median_val]))), direction)
        return {'term': token, 'conf': conf, 'feature': feat_primary,
                'value': median_val, 'rule': 'zone', 'slot': slot_name}
    return None


def _eval_delta(slot_name, slot_cfg, profil, layer_dist):
    feat_primary = slot_cfg.get('feature_primary')
    feat_std_key = slot_cfg.get('feature_std')
    omit_neutral = slot_cfg.get('omit_if_neutral', True)
    neutral_th = slot_cfg.get('neutral_threshold', 0.05)
    median_val = _get_median(profil, feat_primary)
    if median_val is None: return None

    # Fix B_LYA_STD : use feat_std_key directly
    if feat_std_key and omit_neutral and abs(median_val) < neutral_th:
        for term_name, term_cfg in slot_cfg.get('terms', {}).items():
            if 'std_threshold' not in term_cfg: continue
            std_val = _get_std(profil, feat_std_key)
            if std_val is not None and std_val > term_cfg['std_threshold']:
                dist = layer_dist.get(feat_std_key, np.array([std_val]))
                conf = _conf_from_percentile(_percentile_rank(std_val, dist), term_cfg.get('percentile_direction', 'upper'))
                return {'term': term_cfg.get('token', term_name), 'conf': conf,
                        'feature': feat_std_key, 'value': std_val, 'rule': 'std_threshold', 'slot': slot_name}

    if omit_neutral and abs(median_val) < neutral_th: return None
    dist = layer_dist.get(feat_primary, np.array([median_val]))
    for term_name, term_cfg in slot_cfg.get('terms', {}).items():
        if 'std_threshold' in term_cfg: continue
        threshold = term_cfg.get('threshold', 0.0)
        direction = term_cfg.get('percentile_direction', 'upper')
        if (direction == 'upper' and median_val > threshold) or (direction == 'lower' and median_val < threshold):
            conf = _conf_from_percentile(_percentile_rank(median_val, dist), direction)
            return {'term': term_cfg.get('token', term_name), 'conf': conf, 'feature': feat_primary,
                    'value': median_val, 'rule': 'delta', 'slot': slot_name}
    return None


def _eval_threshold(slot_name, slot_cfg, profil, layer_dist):
    features = slot_cfg.get('features', {})
    for term_name, term_cfg in slot_cfg.get('terms', {}).items():
        token = term_cfg.get('token', term_name)
        condition = term_cfg.get('condition')
        threshold = term_cfg.get('threshold')
        direction = term_cfg.get('percentile_direction', 'upper')
        feat_key = features.get(condition, {}).get('key') if isinstance(features.get(condition), dict) else condition
        median_val = None
        if feat_key:
            median_val = profil.get(f'{feat_key}__median', 0.0) if f'{feat_key}__nan_frac' in profil else _get_median(profil, feat_key)
        if median_val is None:
            for fkey, fval in features.items():
                if isinstance(fval, dict) and (condition in fkey or condition == fkey):
                    k = fval.get('key', '')
                    median_val = _get_median(profil, k) or profil.get(f'{k}__median', 0.0)
                    feat_key = k
                    break
        if median_val is None or threshold is None or median_val <= threshold: continue
        dist = layer_dist.get(feat_key or '', np.array([median_val]))
        conf = _conf_from_percentile(_percentile_rank(median_val, dist), direction)
        return {'term': token, 'conf': conf, 'feature': feat_key or condition,
                'value': median_val, 'rule': 'threshold', 'slot': slot_name}
    return None


_EVAL_DISPATCH = {'zones': _eval_zones, 'delta': _eval_delta, 'threshold': _eval_threshold}


# --- ClusterNamer ---

class ClusterNamer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.conf_min_name = cfg.get('conf_min_name', 0.60)
        self.conf_min_display = cfg.get('conf_min_display', 0.50)
        self.separator = cfg.get('name_separator', '·')
        self.slot_order = cfg.get('slot_order', [])
        self.slots_cfg = cfg.get('slots', {})
        self.heterogeneous_threshold = cfg.get('heterogeneous_threshold', 0.70)

    @classmethod
    def from_yaml(cls, path=None):
        if path is None:
            path = Path(__file__).parent / 'configs' / 'cluster_namer.yaml'
        return cls(load_yaml(Path(path))['namer'])

    def name_cluster(self, profil, layer_dist, cluster_homogeneity, n, cluster_id=None):
        slots_primary, slots_secondary, slots_uncalib = [], [], []
        signature = {}

        for slot_name in self.slot_order:
            slot_cfg = self.slots_cfg.get(slot_name, {})
            if not slot_cfg: continue
            if not slot_cfg.get('calibrated', True):
                r = {'slot': slot_name, 'term': None, 'conf': None, 'calibrated': False, 'value': None, 'rule': 'uncalibrated'}
                slots_uncalib.append(r)
                signature[slot_name] = None
                continue

            handler = _EVAL_DISPATCH.get(slot_cfg.get('mode', 'delta'))
            result = handler(slot_name, slot_cfg, profil, layer_dist) if handler else None
            if result is None:
                signature[slot_name] = 0.0
                continue
            result['slot'] = slot_name
            conf = result.get('conf', 0.0) or 0.0
            signature[slot_name] = conf
            if conf >= self.conf_min_name:
                slots_primary.append(result)
            elif conf >= self.conf_min_display:
                slots_secondary.append(result)

        tokens = [s['term'] for s in slots_primary if s.get('term')]
        name = self.separator.join(tokens) if tokens else 'UNCATEGORIZED'
        parts = list(tokens)
        if slots_secondary:
            parts.append(f'({self.separator.join(s["term"] for s in slots_secondary if s.get("term"))})')
        if slots_uncalib:
            parts.append('·'.join(f'[{s["slot"]}=?]' for s in slots_uncalib))

        return {
            'cluster_id': cluster_id, 'name': name,
            'name_full': self.separator.join(parts) if parts else 'UNCATEGORIZED',
            'slots': slots_primary, 'slots_secondary': slots_secondary,
            'slots_uncalibrated': slots_uncalib,
            'signature_vector': [signature.get(s) for s in self.slot_order],
            'slot_order': self.slot_order,
            'cluster_homogeneity': cluster_homogeneity,
            'heterogeneous': cluster_homogeneity < self.heterogeneous_threshold,
            'n': n,
        }

    def name_all(self, peeling_result, M_ortho, feat_names):
        n = M_ortho.shape[0]
        layer_dist = build_layer_distribution(M_ortho, feat_names)
        results = []
        for ci in peeling_result.get('extracted', []):
            mask = np.zeros(n, dtype=bool)
            mask[np.array(ci['global_indices'])] = True
            profil = build_cluster_profile(mask, M_ortho, feat_names)
            named = self.name_cluster(profil, layer_dist, ci['homogeneity'], ci['n'], ci['final_label'])
            named['level'] = ci['level']
            named['metric'] = ci['metric']
            named['config_ari'] = ci.get('config_ari')
            results.append(named)

        residual_idx = np.array(peeling_result.get('residual_idx', []))
        if len(residual_idx) > 0:
            mask_res = np.zeros(n, dtype=bool)
            mask_res[residual_idx] = True
            profil_res = build_cluster_profile(mask_res, M_ortho, feat_names)
            named_res = self.name_cluster(profil_res, layer_dist, 0.0, len(residual_idx), -1)
            named_res.update({'name': 'RÉSIDU', 'name_full': 'RÉSIDU (non résolu)', 'level': -1})
            results.append(named_res)
        return results
