"""
Nommage compositionnel des clusters sur features brutes.

Le profiling est fait sur M_raw (features non-transformées) pour que les
seuils YAML restent interprétables en unités physiques. Le clustering opère
sur M_ortho (transformé), le namer interprète sur M_raw.

@ROLE    Profils bruts + nommage compositionnel YAML-driven
@LAYER   analysing

@EXPORTS
  build_cluster_profile_raw(data, indices, features) → Dict | profil brut
  build_cluster_composition(data, indices)            → Dict | distribution P1
  build_layer_distribution_raw(data, features, indices) → Dict | distribution globale
  ClusterNamer                                        → class | nommage YAML-driven

@LIFECYCLE
  CREATES  profils (dicts)     légers, passés au hub
  RECEIVES AnalysingData       depuis hub (matérialise à la demande)
  PASSES   named_clusters      vers hub → validate, profile, outputs

@CONFORMITY
  OK   Namer opère sur features brutes, pas transformées (point 2 validé)
  OK   Matérialise par cluster puis libère (SD-5)
  OK   ps_rank_delta → f1_effective_rank_delta corrigé (I1)
  OK   Composition P1 intégrée au profil
"""

from pathlib import Path
from typing import Dict, List, Optional, Set
import warnings

import numpy as np

from utils.io_v8 import load_yaml


# =========================================================================
# CONSTANTES
# =========================================================================

EPS = 1e-10


# =========================================================================
# PROFIL CLUSTER SUR FEATURES BRUTES
# =========================================================================

def build_cluster_profile_raw(data,  # AnalysingData
                                cluster_indices: np.ndarray,
                                feature_names: List[str]) -> Dict:
    """Profil statistique sur features BRUTES (pas transformées).

    Matérialise uniquement les colonnes nécessaires pour les runs du cluster.
    Calcule médiane, IQR, mean, std, nan_frac par feature.
    Libère la matrice après calcul.

    Returns:
        Dict avec pour chaque feature :
            {feature}__median, {feature}__iqr, {feature}__mean,
            {feature}__std, {feature}__nan_frac
        Plus : 'n' = nombre de runs dans le cluster.
    """
    n_cluster = len(cluster_indices)
    if n_cluster == 0:
        return {'n': 0}

    # Matérialisation ciblée — uniquement ce cluster, ces features
    M_raw = data.materialize_features(
        columns=feature_names,
        rows=cluster_indices
    ).astype(np.float64)

    profil = {'n': n_cluster}

    for j, key in enumerate(feature_names):
        col = M_raw[:, j]
        valid = col[np.isfinite(col)]

        profil[f'{key}__nan_frac'] = (n_cluster - len(valid)) / n_cluster

        if len(valid) > 0:
            profil[f'{key}__median'] = float(np.median(valid))
            profil[f'{key}__iqr'] = float(
                np.percentile(valid, 75) - np.percentile(valid, 25)
            )
            profil[f'{key}__mean'] = float(np.mean(valid))
            profil[f'{key}__std'] = float(np.std(valid))
        else:
            for s in ('median', 'iqr', 'mean', 'std'):
                profil[f'{key}__{s}'] = None

    # Libérer
    del M_raw

    return profil


# =========================================================================
# COMPOSITION CLUSTER (P1 + statuts)
# =========================================================================

def build_cluster_composition(data,  # AnalysingData
                                cluster_indices: np.ndarray) -> Dict:
    """Distribution p1_regime_class et fraction OK_TRUNCATED dans le cluster.

    Returns:
        {
            'regime_distribution': {FLAT: 0.6, OSCILLATING: 0.3, ...},
            'dominant_regime': 'FLAT',
            'truncated_fraction': 0.15,
            'status_distribution': {OK: 0.85, OK_TRUNCATED: 0.15},
            'n': int,
        }
    """
    n = len(cluster_indices)
    if n == 0:
        return {
            'regime_distribution': {},
            'dominant_regime': '',
            'truncated_fraction': 0.0,
            'status_distribution': {},
            'n': 0,
        }

    # P1 regime class
    regimes = data.p1_regime_class[cluster_indices]
    unique_reg, counts_reg = np.unique(regimes, return_counts=True)
    regime_dist = {str(r): int(c) / n for r, c in zip(unique_reg, counts_reg)}

    dominant = str(unique_reg[np.argmax(counts_reg)]) if len(unique_reg) > 0 else ''

    # Statuts
    statuses = data.run_statuses[cluster_indices]
    unique_st, counts_st = np.unique(statuses, return_counts=True)
    status_dist = {str(s): int(c) / n for s, c in zip(unique_st, counts_st)}

    truncated_frac = float(np.sum(statuses == 'OK_TRUNCATED') / n)

    return {
        'regime_distribution': regime_dist,
        'dominant_regime': dominant,
        'truncated_fraction': truncated_frac,
        'status_distribution': status_dist,
        'n': n,
    }


# =========================================================================
# DISTRIBUTION GLOBALE (pour percentiles namer)
# =========================================================================

def build_layer_distribution_raw(data,  # AnalysingData
                                   feature_names: List[str],
                                   run_indices: np.ndarray) -> Dict[str, np.ndarray]:
    """Distribution globale par feature sur la strate entière (features brutes).

    Pour le calcul des percentiles dans le namer.
    Matérialise la strate entière une seule fois — nécessaire pour les percentiles.

    Returns:
        {feature_name: np.ndarray trié des valeurs valides}
    """
    M_raw = data.materialize_features(
        columns=feature_names,
        rows=run_indices
    ).astype(np.float64)

    result = {}
    for j, key in enumerate(feature_names):
        col = M_raw[:, j]
        valid = col[np.isfinite(col)]
        if len(valid) > 0:
            result[key] = np.sort(valid)

    del M_raw
    return result


# =========================================================================
# HELPERS NAMER
# =========================================================================

def _percentile_rank(value: float, dist: np.ndarray) -> float:
    """Rang percentile d'une valeur dans une distribution triée."""
    if len(dist) == 0:
        return 0.5
    return int(np.searchsorted(dist, value, side='right')) / len(dist)


def _conf_from_percentile(pct: float, direction: str,
                           conf_at_edge: float = 1.0,
                           conf_at_center: float = 0.5) -> float:
    """Confiance depuis un rang percentile."""
    if direction == 'lower':
        return float(np.clip(
            conf_at_edge - (conf_at_edge - conf_at_center) * 2 * pct,
            0.0, 1.0
        ))
    elif direction == 'upper':
        return float(np.clip(
            conf_at_center + (conf_at_edge - conf_at_center) * 2 * (pct - 0.5),
            0.0, 1.0
        ))
    return 0.5


def _get_median(profil: Dict, feature_key: str) -> Optional[float]:
    """Extrait la médiane d'une feature depuis un profil."""
    v = profil.get(f'{feature_key}__median')
    return float(v) if v is not None and np.isfinite(v) else None


def _get_std(profil: Dict, feature_key: str) -> Optional[float]:
    """Extrait le std d'une feature depuis un profil."""
    v = profil.get(f'{feature_key}__std')
    return float(v) if v is not None and np.isfinite(v) else None


# =========================================================================
# ÉVALUATION PAR MODE (zones, delta, threshold)
# =========================================================================

def _eval_zones(slot_name: str, slot_cfg: Dict,
                 profil: Dict, layer_dist: Dict) -> Optional[Dict]:
    """Évalue un slot en mode 'zones' (AMP, DMD)."""
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
            if inf_frac <= 0.0:
                continue
            return {
                'term': token, 'conf': min(1.0, inf_frac * 2),
                'feature': 'health_has_inf', 'value': inf_frac,
                'rule': 'inf_frac', 'slot': slot_name,
            }

        if condition == 'complex_pairs_gt_0':
            if not feat_complex:
                continue
            cv = _get_median(profil, feat_complex)
            if cv is None or cv <= 0:
                continue
            dist = layer_dist.get(feat_complex, np.array([cv]))
            return {
                'term': token,
                'conf': _conf_from_percentile(
                    _percentile_rank(cv, dist), 'upper'
                ),
                'feature': feat_complex, 'value': cv,
                'rule': 'complex_pairs', 'slot': slot_name,
            }

        if median_val is None:
            continue

        if sentinel and abs(median_val - sentinel) < 0.5:
            return {
                'term': token, 'conf': 1.0, 'feature': feat_primary,
                'value': median_val, 'rule': 'sentinel', 'slot': slot_name,
            }

        in_zone = ((lo is None or median_val >= lo) and
                   (hi is None or median_val < hi))
        if not in_zone:
            continue

        if direction is None:
            feat_iqr = slot_cfg.get('feature_iqr', feat_primary)
            iqr_val = profil.get(f'{feat_iqr}__iqr', 0.0) or 0.0
            conf = float(np.clip(
                1.0 - _percentile_rank(
                    iqr_val,
                    layer_dist.get(feat_iqr, np.array([iqr_val]))
                ),
                0.3, 1.0
            ))
        else:
            conf = _conf_from_percentile(
                _percentile_rank(
                    median_val,
                    layer_dist.get(feat_primary, np.array([median_val]))
                ),
                direction,
            )

        return {
            'term': token, 'conf': conf, 'feature': feat_primary,
            'value': median_val, 'rule': 'zone', 'slot': slot_name,
        }

    return None


def _eval_delta(slot_name: str, slot_cfg: Dict,
                 profil: Dict, layer_dist: Dict) -> Optional[Dict]:
    """Évalue un slot en mode 'delta' (ENT, RNK, LYA, LAG, PNN)."""
    feat_primary = slot_cfg.get('feature_primary')
    feat_std_key = slot_cfg.get('feature_std')
    omit_neutral = slot_cfg.get('omit_if_neutral', True)
    neutral_th = slot_cfg.get('neutral_threshold', 0.05)
    median_val = _get_median(profil, feat_primary)

    if median_val is None:
        return None

    # LYA~ : std élevé + mean neutre → alternance
    if feat_std_key and omit_neutral and abs(median_val) < neutral_th:
        for term_name, term_cfg in slot_cfg.get('terms', {}).items():
            if 'std_threshold' not in term_cfg:
                continue
            std_val = _get_std(profil, feat_std_key)
            if std_val is not None and std_val > term_cfg['std_threshold']:
                dist = layer_dist.get(feat_std_key, np.array([std_val]))
                conf = _conf_from_percentile(
                    _percentile_rank(std_val, dist),
                    term_cfg.get('percentile_direction', 'upper'),
                )
                return {
                    'term': term_cfg.get('token', term_name), 'conf': conf,
                    'feature': feat_std_key, 'value': std_val,
                    'rule': 'std_threshold', 'slot': slot_name,
                }

    if omit_neutral and abs(median_val) < neutral_th:
        return None

    dist = layer_dist.get(feat_primary, np.array([median_val]))
    for term_name, term_cfg in slot_cfg.get('terms', {}).items():
        if 'std_threshold' in term_cfg:
            continue
        threshold = term_cfg.get('threshold', 0.0)
        direction = term_cfg.get('percentile_direction', 'upper')

        if ((direction == 'upper' and median_val > threshold) or
                (direction == 'lower' and median_val < threshold)):
            conf = _conf_from_percentile(
                _percentile_rank(median_val, dist), direction
            )
            return {
                'term': term_cfg.get('token', term_name), 'conf': conf,
                'feature': feat_primary, 'value': median_val,
                'rule': 'delta', 'slot': slot_name,
            }

    return None


def _eval_threshold(slot_name: str, slot_cfg: Dict,
                     profil: Dict, layer_dist: Dict) -> Optional[Dict]:
    """Évalue un slot en mode 'threshold' (CND)."""
    features = slot_cfg.get('features', {})

    for term_name, term_cfg in slot_cfg.get('terms', {}).items():
        token = term_cfg.get('token', term_name)
        condition = term_cfg.get('condition')
        threshold = term_cfg.get('threshold')
        direction = term_cfg.get('percentile_direction', 'upper')

        # Résoudre la feature depuis la config
        feat_key = None
        if condition and isinstance(features.get(condition), dict):
            feat_key = features[condition].get('key')
        elif condition:
            feat_key = condition

        if feat_key is None:
            # Chercher dans les features par correspondance
            for fkey, fval in features.items():
                if isinstance(fval, dict) and condition in fkey:
                    feat_key = fval.get('key', '')
                    break

        if feat_key is None or threshold is None:
            continue

        median_val = _get_median(profil, feat_key)
        if median_val is None or median_val <= threshold:
            continue

        dist = layer_dist.get(feat_key, np.array([median_val]))
        conf = _conf_from_percentile(
            _percentile_rank(median_val, dist), direction
        )
        return {
            'term': token, 'conf': conf, 'feature': feat_key,
            'value': median_val, 'rule': 'threshold', 'slot': slot_name,
        }

    return None


_EVAL_DISPATCH = {
    'zones': _eval_zones,
    'delta': _eval_delta,
    'threshold': _eval_threshold,
}


# =========================================================================
# CLUSTER NAMER
# =========================================================================

class ClusterNamer:
    """Nommage compositionnel YAML-driven.

    Chaque cluster est nommé par une combinaison de tokens issus de slots
    (CND, AMP, ENT, RNK, LYA, DMD, LAG, PNN, SEQ) avec une confiance
    par slot. Le nom final est la concaténation des tokens de haute confiance.

    Opère sur features BRUTES (pas transformées).
    Les seuils YAML sont en unités physiques interprétables.
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.conf_min_name = cfg.get('conf_min_name', 0.60)
        self.conf_min_display = cfg.get('conf_min_display', 0.50)
        self.separator = cfg.get('name_separator', '·')
        self.slot_order = cfg.get('slot_order', [])
        self.slots_cfg = cfg.get('slots', {})
        self.heterogeneous_threshold = cfg.get('heterogeneous_threshold', 0.70)

    @classmethod
    def from_yaml(cls, path: Optional[Path] = None) -> 'ClusterNamer':
        """Charge la config namer depuis YAML."""
        if path is None:
            path = Path(__file__).parent / 'configs' / 'cluster_namer.yaml'
        return cls(load_yaml(Path(path))['namer'])

    def name_cluster(self,
                      profil: Dict,
                      layer_dist: Dict,
                      composition: Dict,
                      cluster_homogeneity: float,
                      n: int,
                      cluster_id: Optional[int] = None) -> Dict:
        """Nomme un cluster depuis son profil brut et sa composition P1.

        Args:
            profil : profil features brutes (build_cluster_profile_raw).
            layer_dist : distribution globale features (build_layer_distribution_raw).
            composition : distribution P1 + fraction OK_TRUNCATED.
            cluster_homogeneity : score d'homogénéité du peeling.
            n : nombre de runs.
            cluster_id : identifiant du cluster.

        Returns:
            Dict avec name, slots, signature_vector, composition, etc.
        """
        slots_primary, slots_secondary, slots_uncalib = [], [], []
        signature = {}

        for slot_name in self.slot_order:
            slot_cfg = self.slots_cfg.get(slot_name, {})
            if not slot_cfg:
                continue

            if not slot_cfg.get('calibrated', True):
                r = {
                    'slot': slot_name, 'term': None, 'conf': None,
                    'calibrated': False, 'value': None, 'rule': 'uncalibrated',
                }
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

        # Construction du nom
        tokens = [s['term'] for s in slots_primary if s.get('term')]
        name = self.separator.join(tokens) if tokens else 'UNCATEGORIZED'

        parts = list(tokens)
        if slots_secondary:
            secondary_tokens = [s['term'] for s in slots_secondary if s.get('term')]
            if secondary_tokens:
                parts.append(f'({self.separator.join(secondary_tokens)})')
        if slots_uncalib:
            parts.append('·'.join(f'[{s["slot"]}=?]' for s in slots_uncalib))

        return {
            'cluster_id': cluster_id,
            'name': name,
            'name_full': self.separator.join(parts) if parts else 'UNCATEGORIZED',
            'slots': slots_primary,
            'slots_secondary': slots_secondary,
            'slots_uncalibrated': slots_uncalib,
            'signature_vector': [signature.get(s) for s in self.slot_order],
            'slot_order': self.slot_order,
            'cluster_homogeneity': cluster_homogeneity,
            'heterogeneous': cluster_homogeneity < self.heterogeneous_threshold,
            'composition': composition,
            'n': n,
        }

    def name_all(self,
                  peeling_result: Dict,
                  data,  # AnalysingData
                  feature_names: List[str],
                  run_indices: np.ndarray) -> List[Dict]:
        """Nomme tous les clusters extraits + le résidu.

        Matérialise les features brutes par cluster (pas de M_ortho nécessaire).

        Args:
            peeling_result : sortie de run_peeling.
            data : AnalysingData avec références lazy.
            feature_names : features applicables de la strate (incluant health_*, meta_*).
            run_indices : indices des runs de la strate dans data.
        """
        # Distribution globale (une seule matérialisation pour toute la strate)
        layer_dist = build_layer_distribution_raw(data, feature_names, run_indices)

        results = []

        for ci in peeling_result.get('extracted', []):
            # Indices globaux → indices dans data
            cluster_global_idx = np.array(ci['global_indices'])
            cluster_data_idx = run_indices[cluster_global_idx]

            # Profil brut
            profil = build_cluster_profile_raw(data, cluster_data_idx, feature_names)

            # Composition P1
            composition = build_cluster_composition(data, cluster_data_idx)

            # Nommage
            named = self.name_cluster(
                profil=profil,
                layer_dist=layer_dist,
                composition=composition,
                cluster_homogeneity=ci['homogeneity'],
                n=ci['n'],
                cluster_id=ci['final_label'],
            )
            named['level'] = ci['level']
            named['metric'] = ci['metric']
            named['config_ari'] = ci.get('config_ari')
            results.append(named)

        # Résidu
        residual_idx = peeling_result.get('residual_idx', np.array([]))
        if isinstance(residual_idx, np.ndarray) and len(residual_idx) > 0:
            residual_data_idx = run_indices[residual_idx]
            profil_res = build_cluster_profile_raw(
                data, residual_data_idx, feature_names
            )
            composition_res = build_cluster_composition(data, residual_data_idx)
            named_res = self.name_cluster(
                profil=profil_res,
                layer_dist=layer_dist,
                composition=composition_res,
                cluster_homogeneity=0.0,
                n=len(residual_idx),
                cluster_id=-1,
            )
            named_res.update({
                'name': 'RÉSIDU',
                'name_full': 'RÉSIDU (non résolu)',
                'level': -1,
            })
            results.append(named_res)

        # Libérer la distribution globale
        del layer_dist

        return results
