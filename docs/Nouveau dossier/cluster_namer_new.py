"""
analysing/cluster_namer_new.py

Registre de nommage compositionnel PRC v7 — entièrement data-driven depuis YAML.

Nomenclature :
    Familles courtes (AMP, ENT, RNK, LYA, DMD, LAG, PNN, CND, SEQ)
    + qualificatifs génériques (<<, ~0, +, >>, osc, →)
    → noms compositionnels ≤3 tokens, extensibles sans modifier le code.

Architecture :
    Zéro dispatch hardcodé par slot name — le code lit le mode depuis le YAML
    et route vers le handler correspondant.

    Modes disponibles :
        zones       — zones exclusives avec bornes lo/hi (AMP, DMD)
        delta       — bidirectionnel autour de 0, omit_if_neutral (ENT, RNK, LYA)
        threshold   — seuil unique, omit_if_clean (CND)
        uncalibrated — placeholder, retourne toujours None (SEQ)

    Ajouter un slot = ajouter une entrée dans le YAML.
    Ajouter un mode = ajouter un handler dans _EVAL_DISPATCH.

Dette technique documentée :
    SEQ calibrated: false — nécessite F8 features phasiques (R1, max_it≥200)
    Quand F8 disponible : migrer FEATURE_SEMANTICS dans jax_features_new.py,
    le namer lira les métadonnées depuis la source.
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

EPS = 1e-10


# =============================================================================
# CHARGEMENT CONFIG
# =============================================================================

def load_namer_config(path) -> Dict:
    """Charge config namer depuis YAML."""
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg['namer']


# =============================================================================
# PROFIL CLUSTER — agrégation features
# =============================================================================

def build_cluster_profile(
    cluster_mask: np.ndarray,
    M_ortho     : np.ndarray,
    feat_names  : List[str],
) -> Dict:
    """
    Agrège features d'un cluster en statistiques de distribution.

    Calcule median, IQR, mean, nan_frac pour chaque feature.
    Profil = seul input du namer — isolation totale des runs individuels.

    Args:
        cluster_mask : (n,) booléen — runs appartenant au cluster
        M_ortho      : (n, F) matrice features
        feat_names   : noms colonnes de M_ortho

    Returns:
        dict {feature__median, feature__iqr, feature__mean, feature__nan_frac, n}
    """
    n_cluster = int(cluster_mask.sum())
    if n_cluster == 0:
        return {}

    sub    = M_ortho[cluster_mask].astype(np.float64)   # (n_cluster, F)
    profil = {'n': n_cluster}

    for j, key in enumerate(feat_names):
        col   = sub[:, j]
        valid = col[np.isfinite(col)]
        n_nan = n_cluster - len(valid)

        profil[f'{key}__nan_frac'] = n_nan / n_cluster

        if len(valid) > 0:
            profil[f'{key}__median'] = float(np.median(valid))
            profil[f'{key}__iqr']    = float(np.percentile(valid, 75) - np.percentile(valid, 25))
            profil[f'{key}__mean']   = float(np.mean(valid))
            profil[f'{key}__std']    = float(np.std(valid))
        else:
            for s in ('median', 'iqr', 'mean', 'std'):
                profil[f'{key}__{s}'] = None

    return profil


def build_layer_distribution(
    M_ortho   : np.ndarray,
    feat_names: List[str],
) -> Dict:
    """
    Distribution layer_local pour normalisation percentile.

    Args:
        M_ortho    : (n, F) matrice features
        feat_names : noms colonnes de M_ortho

    Returns:
        {feature_key: np.ndarray sorted} — base du percentile rank
    """
    result = {}
    for j, key in enumerate(feat_names):
        col   = M_ortho[:, j].astype(np.float64)
        valid = col[np.isfinite(col)]
        if len(valid) > 0:
            result[key] = np.sort(valid)
    return result


# =============================================================================
# HELPERS PERCENTILE
# =============================================================================

def _percentile_rank(value: float, dist: np.ndarray) -> float:
    """Rang percentile de value dans dist. 0=min, 1=max."""
    if len(dist) == 0:
        return 0.5
    return int(np.searchsorted(dist, value, side='right')) / len(dist)


def _conf_from_percentile(
    pct          : float,
    direction    : str,
    conf_at_edge : float = 1.0,
    conf_at_center: float = 0.5,
) -> float:
    """
    Percentile → confiance [0, 1].

    direction='lower' : valeur basse confirme le terme
    direction='upper' : valeur haute confirme le terme
    """
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
    """Lit feature__median depuis profil — None si absent ou non fini."""
    v = profil.get(f'{feature_key}__median')
    if v is None or not np.isfinite(v):
        return None
    return float(v)


def _get_std(profil: Dict, feature_key: str) -> Optional[float]:
    """Lit feature__std depuis profil."""
    v = profil.get(f'{feature_key}__std')
    if v is None or not np.isfinite(v):
        return None
    return float(v)


# =============================================================================
# HANDLERS PAR MODE
# =============================================================================

def _eval_zones(
    slot_name : str,
    slot_cfg  : Dict,
    profil    : Dict,
    layer_dist: Dict,
) -> Optional[Dict]:
    """
    Mode zones — zones exclusives avec bornes lo/hi.

    Parcourt les terms dans l'ordre YAML.
    Retourne le premier term dont la condition est satisfaite.

    Gère conditions spéciales :
        inf_frac_gt_0    : median de health_has_inf > 0
        complex_pairs_gt_0 : median de f7_dmd_n_complex_pairs > 0
    """
    feat_primary  = slot_cfg.get('feature_primary')
    feat_complex  = slot_cfg.get('feature_complex')
    median_val    = _get_median(profil, feat_primary) if feat_primary else None

    # fraction Inf dans le cluster (depuis health_has_inf)
    inf_frac = profil.get('health_has_inf__median', 0.0) or 0.0

    for term_name, term_cfg in slot_cfg.get('terms', {}).items():
        token     = term_cfg.get('token', term_name)
        condition = term_cfg.get('condition')
        lo        = term_cfg.get('lo')
        hi        = term_cfg.get('hi')
        direction = term_cfg.get('percentile_direction')
        sentinel  = term_cfg.get('sentinel')

        # Conditions spéciales
        if condition == 'inf_frac_gt_0':
            if inf_frac <= 0.0:
                continue
            conf = min(1.0, inf_frac * 2)
            return {'term': token, 'conf': conf,
                    'feature': 'health_has_inf', 'value': inf_frac,
                    'rule': 'inf_frac', 'slot': slot_name}

        if condition == 'complex_pairs_gt_0':
            if feat_complex is None:
                continue
            complex_val = _get_median(profil, feat_complex)
            if complex_val is None or complex_val <= 0:
                continue
            dist = layer_dist.get(feat_complex, np.array([complex_val]))
            pct  = _percentile_rank(complex_val, dist)
            conf = _conf_from_percentile(pct, 'upper')
            return {'term': token, 'conf': conf,
                    'feature': feat_complex, 'value': complex_val,
                    'rule': 'complex_pairs', 'slot': slot_name}

        if median_val is None:
            continue

        # Sentinel check (saturation)
        if sentinel and abs(median_val - sentinel) < 0.5:
            return {'term': token, 'conf': 1.0,
                    'feature': feat_primary, 'value': median_val,
                    'rule': 'sentinel', 'slot': slot_name}

        # Zone [lo, hi)
        in_zone = True
        if lo is not None and median_val < lo:
            in_zone = False
        if hi is not None and median_val >= hi:
            in_zone = False
        if not in_zone:
            continue

        # Confiance
        if direction is None:
            # Zone médiane — conf basée IQR
            feat_iqr  = slot_cfg.get('feature_iqr', feat_primary)
            iqr_val   = profil.get(f'{feat_iqr}__iqr', 0.0) or 0.0
            dist_iqr  = layer_dist.get(feat_iqr, np.array([iqr_val]))
            iqr_pct   = _percentile_rank(iqr_val, dist_iqr)
            conf      = float(np.clip(1.0 - iqr_pct, 0.3, 1.0))
        else:
            dist = layer_dist.get(feat_primary, np.array([median_val]))
            pct  = _percentile_rank(median_val, dist)
            conf = _conf_from_percentile(pct, direction)

        return {'term': token, 'conf': conf,
                'feature': feat_primary, 'value': median_val,
                'rule': 'zone', 'slot': slot_name}

    return None


def _eval_delta(
    slot_name : str,
    slot_cfg  : Dict,
    profil    : Dict,
    layer_dist: Dict,
) -> Optional[Dict]:
    """
    Mode delta — bidirectionnel autour de 0, omit_if_neutral.

    Retourne None si median dans la zone neutre.
    Gère le cas LYA~ (std élevé + mean neutre) si feature_std défini.
    """
    feat_primary   = slot_cfg.get('feature_primary')
    feat_std_key   = slot_cfg.get('feature_std')
    omit_neutral   = slot_cfg.get('omit_if_neutral', True)
    neutral_th     = slot_cfg.get('neutral_threshold', 0.05)

    median_val = _get_median(profil, feat_primary)
    if median_val is None:
        return None

    # Cas spécial LYA~ : std élevé + mean neutre
    if feat_std_key and omit_neutral and abs(median_val) < neutral_th:
        std_th  = None
        lya_tilde = None
        for term_name, term_cfg in slot_cfg.get('terms', {}).items():
            if 'std_threshold' in term_cfg:
                std_th     = term_cfg['std_threshold']
                lya_tilde  = term_cfg
                break
        if std_th is not None:
            std_val = _get_std(profil, feat_primary)
            if std_val is not None and std_val > std_th:
                dist = layer_dist.get(feat_primary + '_std', np.array([std_val]))
                pct  = _percentile_rank(std_val, dist)
                conf = _conf_from_percentile(pct, lya_tilde.get('percentile_direction', 'upper'))
                return {
                    'term'   : lya_tilde.get('token', term_name),
                    'conf'   : conf,
                    'feature': feat_std_key,
                    'value'  : std_val,
                    'rule'   : 'std_threshold',
                    'slot'   : slot_name,
                }

    if omit_neutral and abs(median_val) < neutral_th:
        return None

    dist = layer_dist.get(feat_primary, np.array([median_val]))

    for term_name, term_cfg in slot_cfg.get('terms', {}).items():
        if 'std_threshold' in term_cfg:
            continue   # LYA~ déjà géré ci-dessus
        threshold = term_cfg.get('threshold', 0.0)
        direction = term_cfg.get('percentile_direction', 'upper')
        token     = term_cfg.get('token', term_name)

        in_zone = False
        if direction == 'upper' and median_val > threshold:
            in_zone = True
        elif direction == 'lower' and median_val < threshold:
            in_zone = True

        if not in_zone:
            continue

        pct  = _percentile_rank(median_val, dist)
        conf = _conf_from_percentile(pct, direction)
        return {'term': token, 'conf': conf,
                'feature': feat_primary, 'value': median_val,
                'rule': 'delta', 'slot': slot_name}

    return None


def _eval_threshold(
    slot_name : str,
    slot_cfg  : Dict,
    profil    : Dict,
    layer_dist: Dict,
) -> Optional[Dict]:
    """
    Mode threshold — seuil unique, omit_if_clean.

    Parcourt les terms dans l'ordre.
    Retourne None si aucun seuil franchi (cluster sain).
    """
    omit_clean = slot_cfg.get('omit_if_clean', True)
    features   = slot_cfg.get('features', {})

    for term_name, term_cfg in slot_cfg.get('terms', {}).items():
        token     = term_cfg.get('token', term_name)
        condition = term_cfg.get('condition')
        threshold = term_cfg.get('threshold')
        direction = term_cfg.get('percentile_direction', 'upper')

        feat_key = features.get(condition, {}).get('key') if isinstance(
            features.get(condition), dict) else condition

        # Résoudre la valeur
        median_val = None
        if feat_key:
            # Chercher dans profil via median ou nan_frac
            if f'{feat_key}__nan_frac' in profil:
                # health_has_inf est une valeur 0/1 — median = fraction runs EXPLOSION
                median_val = profil.get(f'{feat_key}__median', 0.0) or 0.0
            else:
                median_val = _get_median(profil, feat_key)

        # Fallback : chercher directement le nom de la condition
        if median_val is None:
            for fkey, fval in features.items():
                if isinstance(fval, dict):
                    k = fval.get('key', '')
                    if condition in fkey or condition == fkey:
                        median_val = _get_median(profil, k)
                        if median_val is None:
                            median_val = profil.get(f'{k}__median', 0.0) or 0.0
                        feat_key = k
                        break

        if median_val is None or threshold is None:
            continue

        if median_val <= threshold:
            continue

        dist = layer_dist.get(feat_key or '', np.array([median_val]))
        pct  = _percentile_rank(median_val, dist)
        conf = _conf_from_percentile(pct, direction)

        return {'term': token, 'conf': conf,
                'feature': feat_key or condition,
                'value': median_val,
                'rule': 'threshold', 'slot': slot_name}

    return None if omit_clean else None


def _eval_uncalibrated(slot_name: str, slot_cfg: Dict, profil: Dict) -> Dict:
    """Mode uncalibrated — placeholder, retourne structure vide."""
    return {
        'slot'      : slot_name,
        'term'      : None,
        'conf'      : None,
        'calibrated': False,
        'value'     : None,
        'rule'      : 'uncalibrated',
    }


# Dispatch modes → handlers
_EVAL_DISPATCH = {
    'zones'       : _eval_zones,
    'delta'       : _eval_delta,
    'threshold'   : _eval_threshold,
}


# =============================================================================
# CLUSTER NAMER
# =============================================================================

class ClusterNamer:
    """
    Registre de nommage compositionnel data-driven.

    Usage :
        namer = ClusterNamer.from_yaml('analysing/configs/cluster_namer_new.yaml')
        layer_dist = build_layer_distribution(all_features)
        profile    = build_cluster_profile(cluster_indices, all_features)
        result     = namer.name_cluster(profile, layer_dist,
                                        cluster_homogeneity=0.87, n=36)
    """

    def __init__(self, cfg: Dict):
        self.cfg                    = cfg
        self.conf_min_name          = cfg.get('conf_min_name', 0.60)
        self.conf_min_display       = cfg.get('conf_min_display', 0.50)
        self.separator              = cfg.get('name_separator', '·')
        self.slot_order             = cfg.get('slot_order', [])
        self.slots_cfg              = cfg.get('slots', {})
        self.heterogeneous_threshold = cfg.get('heterogeneous_threshold', 0.70)

    @classmethod
    def from_yaml(cls, path) -> 'ClusterNamer':
        cfg = load_namer_config(path)
        return cls(cfg)

    def name_cluster(
        self,
        profil              : Dict,
        layer_dist          : Dict,
        cluster_homogeneity : float,
        n                   : int,
        cluster_id          : Optional[int] = None,
    ) -> Dict:
        """
        Produit le nom composé et la structure complète pour un cluster.

        Args:
            profil              : sortie build_cluster_profile()
            layer_dist          : sortie build_layer_distribution()
            cluster_homogeneity : score peeling
            n                   : taille cluster
            cluster_id          : label numérique optionnel

        Returns:
            {
                'cluster_id', 'name', 'name_full',
                'slots', 'slots_secondary', 'slots_uncalibrated',
                'signature_vector', 'slot_order',
                'cluster_homogeneity', 'heterogeneous', 'n'
            }
        """
        slots_primary   = []
        slots_secondary = []
        slots_uncalib   = []
        signature       = {}

        for slot_name in self.slot_order:
            slot_cfg = self.slots_cfg.get(slot_name, {})
            if not slot_cfg:
                continue

            # Slot non calibré
            if not slot_cfg.get('calibrated', True):
                result = _eval_uncalibrated(slot_name, slot_cfg, profil)
                result['slot'] = slot_name
                slots_uncalib.append(result)
                signature[slot_name] = None
                continue

            # Dispatch par mode
            mode    = slot_cfg.get('mode', 'delta')
            handler = _EVAL_DISPATCH.get(mode)
            result  = None

            if handler is not None:
                result = handler(slot_name, slot_cfg, profil, layer_dist)

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

        # Nom composé
        name_tokens = [s['term'] for s in slots_primary if s.get('term')]
        sec_tokens  = [s['term'] for s in slots_secondary if s.get('term')]

        name = self.separator.join(name_tokens) if name_tokens else 'UNCATEGORIZED'

        name_full_parts = list(name_tokens)
        if sec_tokens:
            name_full_parts.append(f'({self.separator.join(sec_tokens)})')
        if slots_uncalib:
            name_full_parts.append(
                '·'.join(f'[{s["slot"]}=?]' for s in slots_uncalib)
            )
        name_full = self.separator.join(name_full_parts) if name_full_parts else 'UNCATEGORIZED'

        sig_vector = [signature.get(s) for s in self.slot_order]

        return {
            'cluster_id'         : cluster_id,
            'name'               : name,
            'name_full'          : name_full,
            'slots'              : slots_primary,
            'slots_secondary'    : slots_secondary,
            'slots_uncalibrated' : slots_uncalib,
            'signature_vector'   : sig_vector,
            'slot_order'         : self.slot_order,
            'cluster_homogeneity': cluster_homogeneity,
            'heterogeneous'      : cluster_homogeneity < self.heterogeneous_threshold,
            'n'                  : n,
        }

    def name_all(
        self,
        peeling_result: Dict,
        M_ortho       : np.ndarray,
        feat_names    : List[str],
    ) -> List[Dict]:
        """
        Nomme tous les clusters d'un résultat peeling + le résidu.

        Args:
            peeling_result : sortie run_peeling()
            M_ortho        : (n, F) matrice features depuis prepare_matrix
            feat_names     : noms colonnes de M_ortho

        Returns:
            List[Dict] — un dict par cluster extrait + un pour le résidu
        """
        n          = M_ortho.shape[0]
        layer_dist = build_layer_distribution(M_ortho, feat_names)
        extracted  = peeling_result.get('extracted', [])
        results    = []

        for cluster_info in extracted:
            cid  = cluster_info['final_label']
            gidx = np.array(cluster_info['global_indices'])
            homo = cluster_info['homogeneity']
            nc   = cluster_info['n']

            mask   = np.zeros(n, dtype=bool)
            mask[gidx] = True
            profil = build_cluster_profile(mask, M_ortho, feat_names)
            named  = self.name_cluster(profil, layer_dist,
                                       cluster_homogeneity=homo,
                                       n=nc, cluster_id=cid)
            named['level']      = cluster_info['level']
            named['metric']     = cluster_info['metric']
            named['config_ari'] = cluster_info.get('config_ari')
            results.append(named)

        # Résidu
        residual_idx = np.array(peeling_result.get('residual_idx', []))
        if len(residual_idx) > 0:
            mask_res = np.zeros(n, dtype=bool)
            mask_res[residual_idx] = True
            profil_res = build_cluster_profile(mask_res, M_ortho, feat_names)
            named_res  = self.name_cluster(profil_res, layer_dist,
                                           cluster_homogeneity=0.0,
                                           n=len(residual_idx),
                                           cluster_id=-1)
            named_res['name']      = 'RÉSIDU'
            named_res['name_full'] = 'RÉSIDU (non résolu)'
            named_res['level']     = -1
            results.append(named_res)

        return results


# =============================================================================
# RAPPORT TEXTE
# =============================================================================

def print_naming_report(named_clusters: List[Dict]):
    """Rapport lisible des clusters nommés."""
    print('\n' + '='*65)
    print('RAPPORT NOMMAGE CLUSTERS v7')
    print('='*65)

    for nc in sorted(named_clusters, key=lambda x: x.get('cluster_id', 999)):
        cid   = nc.get('cluster_id', '?')
        label = 'RÉSIDU' if cid == -1 else f'Cluster {cid:2d}'
        homo  = nc.get('cluster_homogeneity', 0.0)
        het   = ' ⚠ hétérogène' if nc.get('heterogeneous') else ''

        print(f'\n{label} (n={nc["n"]})  homo={homo:.2f}{het}')
        print(f'  Nom      : {nc["name"]}')
        if nc['name_full'] != nc['name']:
            print(f'  Complet  : {nc["name_full"]}')

        for s in nc.get('slots', []):
            conf_str = f'{s["conf"]:.2f}' if s.get('conf') is not None else '—'
            print(f'  [{s["slot"]:10s}] {s["term"]:20s} conf={conf_str}'
                  f'  ({s.get("feature","?")}={s.get("value","?")})')

        if nc.get('slots_secondary'):
            print('  Secondaires :')
            for s in nc['slots_secondary']:
                conf_str = f'{s["conf"]:.2f}' if s.get('conf') is not None else '—'
                print(f'    [{s["slot"]:10s}] ({s["term"]})  conf={conf_str}')

        uncalib = [s['slot'] for s in nc.get('slots_uncalibrated', [])]
        if uncalib:
            print(f'  Non calibrés : {", ".join(uncalib)}')

        sig      = nc.get('signature_vector', [])
        slot_ord = nc.get('slot_order', [])
        if sig and slot_ord:
            sig_str = '  '.join(
                f'{o}={f"{v:.2f}" if v is not None else "?"}'
                for o, v in zip(slot_ord, sig)
            )
            print(f'  Signature : {sig_str}')
