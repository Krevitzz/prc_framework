"""
cluster_namer.py

Registre de nommage compositionnel pour clusters HDBSCAN peeling.

Principe :
    Chaque cluster est décrit par un nom composé de slots indépendants.
    Chaque slot capture un axe sémantique (amplitude, vitesse, texture...).
    La confiance de chaque terme est calculée par percentile rank
    sur la distribution layer_local (non paramétrique, robuste).

Format de sortie par cluster :
    {
      "name"               : "COLLAPSE·INSTABLE",          # slots conf >= conf_min_name
      "name_full"          : "COLLAPSE·INSTABLE·(ENTROPIE_DÉCROISSANTE)",  # + secondary
      "slots"              : [...],                         # tous les slots calibrés
      "slots_secondary"    : [...],                         # conf < conf_min_name
      "slots_uncalibrated" : [...],                         # calibrated=false → conf=null
      "signature_vector"   : [0.97, 0.0, null, null, 0.71, 0.0],  # conf par slot_order
      "cluster_homogeneity": float,                         # couche 1 — du peeling
      "heterogeneous"      : bool,
      "n"                  : int,
    }

Usage :
    from cluster_namer import ClusterNamer
    namer = ClusterNamer.from_yaml('analysing/configs/cluster_namer.yaml')
    names = namer.name_all(clusters_profiles, layer_distribution)
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter



# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def load_namer_config(path: str) -> Dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg['namer']


# ─────────────────────────────────────────────────────────────────────────────
# CALCUL PROFIL AGRÉGÉ D'UN CLUSTER
# ─────────────────────────────────────────────────────────────────────────────

def build_cluster_profile(
    cluster_indices : List[int],
    all_features    : List[Dict],  # all_features[i] = features du run i
) -> Dict:
    """
    Agrège les features d'un cluster en statistiques de distribution.

    Calcule median, IQR, min, max, fraction NaN/inf pour chaque feature.
    Le profil est le seul input du namer — isolation totale des runs individuels.

    Args:
        cluster_indices : indices globaux des runs dans ce cluster
        all_features    : liste features par run (longueur = n_total_runs)

    Returns:
        profil dict avec clés de la forme :
            {feature}_median, {feature}_iqr, {feature}_min, {feature}_max,
            {feature}_nan_frac
    """
    if not cluster_indices:
        return {}

    # Collecter toutes les features disponibles
    feature_keys = set()
    for idx in cluster_indices:
        feature_keys.update(all_features[idx].keys())
    feature_keys -= {'has_nan_inf', 'is_collapsed'}

    profil = {'n': len(cluster_indices)}

    for key in sorted(feature_keys):
        vals = []
        n_nan = 0
        for idx in cluster_indices:
            v = all_features[idx].get(key)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                n_nan += 1
            else:
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    n_nan += 1

        nan_frac = n_nan / len(cluster_indices)
        profil[f'{key}__nan_frac'] = nan_frac

        if vals:
            arr = np.array(vals)
            profil[f'{key}__median'] = float(np.median(arr))
            profil[f'{key}__iqr']    = float(np.percentile(arr, 75) - np.percentile(arr, 25))
            profil[f'{key}__min']    = float(np.min(arr))
            profil[f'{key}__max']    = float(np.max(arr))
            profil[f'{key}__mean']   = float(np.mean(arr))
        else:
            for suffix in ('median', 'iqr', 'min', 'max', 'mean'):
                profil[f'{key}__{suffix}'] = None

    # Flags agrégés
    profil['has_nan_frac'] = sum(
        1 for i in cluster_indices if all_features[i].get('has_nan_inf', False)
    ) / len(cluster_indices)

    profil['is_collapsed_frac'] = sum(
        1 for i in cluster_indices if all_features[i].get('is_collapsed', False)
    ) / len(cluster_indices)

    return profil


def build_layer_distribution(all_features: List[Dict]) -> Dict:
    """
    Calcule la distribution layer_local pour la normalisation percentile.

    Pour chaque feature : liste triée de valeurs finies → base du percentile rank.

    Args:
        all_features : features de TOUS les runs du layer (pas seulement un cluster)

    Returns:
        dict {feature_key: sorted_array}
    """
    from collections import defaultdict
    accum = defaultdict(list)

    for feat_dict in all_features:
        for key, val in feat_dict.items():
            if key in ('has_nan_inf', 'is_collapsed'):
                continue
            if val is not None and isinstance(val, (int, float)) and np.isfinite(val):
                accum[key].append(float(val))

    return {k: np.sort(np.array(v)) for k, v in accum.items()}


# ─────────────────────────────────────────────────────────────────────────────
# CALCUL CONFIANCE PERCENTILE
# ─────────────────────────────────────────────────────────────────────────────

def _percentile_rank(value: float, sorted_distribution: np.ndarray) -> float:
    """
    Rang percentile de value dans sorted_distribution.
    0.0 = minimum absolu, 1.0 = maximum absolu.
    Non paramétrique — pas d'hypothèse sur la distribution.
    """
    if len(sorted_distribution) == 0:
        return 0.5
    n = len(sorted_distribution)
    rank = int(np.searchsorted(sorted_distribution, value, side='right'))
    return rank / n


def _conf_from_percentile(
    percentile    : float,
    direction     : str,         # 'lower' | 'upper'
    conf_at_edge  : float = 1.0,
    conf_at_center: float = 0.5,
) -> float:
    """
    Convertit un percentile en confiance [0,1].

    direction='lower' : valeur basse confirme le terme (ex: COLLAPSE)
        percentile=0.0 → conf=1.0, percentile=0.5 → conf=0.5

    direction='upper' : valeur haute confirme le terme (ex: SATURATION)
        percentile=1.0 → conf=1.0, percentile=0.5 → conf=0.5

    Linéaire entre center et edge pour simplicité et traçabilité.
    """
    if direction == 'lower':
        # conf décroît de 1.0 (percentile=0) à 0.5 (percentile=0.5)
        # puis continue à décroître en dessous de 0.5 (zone adverse)
        return float(np.clip(conf_at_edge - (conf_at_edge - conf_at_center) * 2 * percentile, 0.0, 1.0))
    elif direction == 'upper':
        return float(np.clip(conf_at_center + (conf_at_edge - conf_at_center) * 2 * (percentile - 0.5), 0.0, 1.0))
    return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# ÉVALUATION D'UN SLOT
# ─────────────────────────────────────────────────────────────────────────────

def _eval_slot_amplitude(profil: Dict, layer_dist: Dict, slot_cfg: Dict) -> Dict:
    """
    Slot AMPLITUDE — logique spéciale car bidirectionnel avec zones disjointes.
    Retourne le terme et sa confiance.
    """
    nr_median  = profil.get('norm_ratio__median')
    eu_min     = profil.get('euclidean_norm__signal_finite_ratio__min', 1.0)
    nr_is_nan  = (nr_median is None)

    terms_cfg  = slot_cfg['terms']
    sep        = '·'

    # EXPLOSION — règle booléenne
    if eu_min is not None and eu_min < 1.0 and nr_is_nan:
        conf = 1.0 if eu_min < 0.5 else 0.75
        return {'term': 'EXPLOSION', 'conf': conf,
                'feature': 'eu_finite_ratio_min', 'value': eu_min,
                'rule': 'boolean'}

    if nr_is_nan:
        return {'term': 'UNCATEGORIZED', 'conf': 0.5,
                'feature': 'norm_ratio_median', 'value': None, 'rule': 'nan'}

    # Sentinel SATURATION
    sentinel = terms_cfg.get('SATURATION', {}).get('sentinel', 148.413)
    if abs(nr_median - sentinel) < 0.5:
        return {'term': 'SATURATION', 'conf': 1.0,
                'feature': 'norm_ratio_median', 'value': nr_median, 'rule': 'sentinel'}

    dist_nr = layer_dist.get('norm_ratio', np.array([]))

    # Trouver la zone
    thresholds = {
        'COLLAPSE'        : (None, 0.10,  'lower'),
        'CONSERVE'        : (0.10, 1.30,  None),
        'CROISSANCE_FAIBLE': (1.30, 3.0,  'upper'),
        'CROISSANCE_FORTE' : (3.0, 10.0,  'upper'),
        'SATURATION'      : (10.0, None,  'upper'),
    }

    for term, (lo, hi, direction) in thresholds.items():
        in_zone = True
        if lo is not None and nr_median < lo: in_zone = False
        if hi is not None and nr_median >= hi: in_zone = False
        if not in_zone:
            continue

        # Confiance
        if direction is None:
            # Zone médiane (CONSERVE) : confiance basée sur IQR faible
            nr_iqr = profil.get('norm_ratio__iqr', 0.0) or 0.0
            iqr_pct = _percentile_rank(nr_iqr, layer_dist.get('norm_ratio_iqr_proxy',
                                                               np.array([nr_iqr])))
            conf = float(np.clip(1.0 - iqr_pct, 0.3, 1.0))
        elif len(dist_nr) > 0:
            pct  = _percentile_rank(nr_median, dist_nr)
            conf = _conf_from_percentile(pct, direction)
        else:
            conf = 0.5

        return {'term': term, 'conf': conf,
                'feature': 'norm_ratio_median', 'value': nr_median,
                'rule': 'percentile', 'direction': direction}

    return {'term': 'UNCATEGORIZED', 'conf': 0.5,
            'feature': 'norm_ratio_median', 'value': nr_median, 'rule': 'out_of_range'}


def _eval_slot_sante_num(profil: Dict, layer_dist: Dict, slot_cfg: Dict) -> Optional[Dict]:
    """
    Slot SANTÉ_NUM — retourne None si cluster sain (omit_if_clean).
    """
    nan_frac   = profil.get('has_nan_frac', 0.0) or 0.0
    cond_med   = profil.get('condition_number_svd_final__median')

    terms = slot_cfg['terms']
    omit  = slot_cfg.get('omit_if_clean', True)

    # INSTABLE
    th_nan = terms.get('INSTABLE', {}).get('threshold', 0.30)
    if nan_frac > th_nan:
        dist = layer_dist.get('has_nan_frac_proxy', np.array([nan_frac]))
        pct  = _percentile_rank(nan_frac, dist)
        conf = _conf_from_percentile(pct, 'upper')
        return {'term': 'INSTABLE', 'conf': conf,
                'feature': 'has_nan_frac', 'value': nan_frac, 'rule': 'percentile'}

    # CONDITIONNÉ
    if cond_med is not None and np.isfinite(cond_med):
        th_cond = terms.get('CONDITIONNÉ', {}).get('threshold', 1e6)
        if cond_med > th_cond:
            dist = layer_dist.get('condition_number_svd_final', np.array([cond_med]))
            pct  = _percentile_rank(cond_med, dist)
            conf = _conf_from_percentile(pct, 'upper')
            return {'term': 'CONDITIONNÉ', 'conf': conf,
                    'feature': 'condition_number_svd_final_median', 'value': cond_med,
                    'rule': 'percentile'}

    # Cluster sain → omis
    if omit:
        return None

    return {'term': 'SAIN', 'conf': 0.8, 'feature': 'has_nan_frac',
            'value': nan_frac, 'rule': 'clean'}


def _eval_slot_uncalibrated(slot_name: str, profil: Dict,
                             slot_cfg: Dict) -> Dict:
    """
    Slot non calibré — retourne structure vide avec calibrated=False.
    Les valeurs sont présentes pour traçabilité future.
    """
    feat_primary = slot_cfg.get('feature_primary', '')
    key = f'{feat_primary}__median'.replace('__', '__') if feat_primary else None
    value = profil.get(key) if key else None

    return {
        'slot'       : slot_name,
        'term'       : None,
        'conf'       : None,
        'calibrated' : False,
        'feature'    : feat_primary,
        'value'      : value,
        'rule'       : 'uncalibrated',
    }


def _eval_slot_delta(slot_name: str, feature_key: str,
                     profil: Dict, layer_dist: Dict,
                     slot_cfg: Dict) -> Optional[Dict]:
    """
    Slot générique pour features delta (ENTROPIE, RANG) — bidirectionnel.
    Retourne None si neutre (omit_if_neutral).
    """
    median_val = profil.get(f'{feature_key}__median')
    if median_val is None:
        return None

    neutral_th = slot_cfg.get('neutral_threshold', 0.05)
    if slot_cfg.get('omit_if_neutral') and abs(median_val) < neutral_th:
        return None

    terms = slot_cfg['terms']
    dist  = layer_dist.get(feature_key, np.array([median_val]))

    for term_name, term_cfg in terms.items():
        th = term_cfg.get('threshold', 0.0)
        direction = term_cfg.get('percentile_direction', 'upper')

        in_zone = False
        if direction == 'lower' and median_val < th:
            in_zone = True
        elif direction == 'upper' and median_val > th:
            in_zone = True

        if in_zone:
            pct  = _percentile_rank(median_val, dist)
            conf = _conf_from_percentile(pct, direction)
            return {
                'term'     : term_name,
                'conf'     : conf,
                'feature'  : f'{feature_key}_median',
                'value'    : median_val,
                'rule'     : 'percentile',
                'direction': direction,
            }

    return None


# ─────────────────────────────────────────────────────────────────────────────
# NAMER PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class ClusterNamer:
    """
    Registre de nommage compositionnel.

    Usage :
        namer = ClusterNamer.from_yaml('cluster_namer.yaml')
        layer_dist = build_layer_distribution(all_features)
        profile    = build_cluster_profile(cluster_indices, all_features)
        result     = namer.name_cluster(profile, layer_dist,
                                         cluster_homogeneity=0.87, n=36)
    """

    def __init__(self, cfg: Dict):
        self.cfg           = cfg
        self.conf_min_name = cfg.get('conf_min_name', 0.65)
        self.conf_min_disp = cfg.get('conf_min_display', 0.50)
        self.separator     = cfg.get('name_separator', '·')
        self.slot_order    = cfg.get('slot_order', [])
        self.slots_cfg     = cfg.get('slots', {})
        self.heterogeneous_threshold = cfg.get('heterogeneous_threshold', 0.70)

    @classmethod
    def from_yaml(cls, path: str) -> 'ClusterNamer':
        cfg = load_namer_config(path)
        return cls(cfg)

    # ── Évaluation d'un cluster ──────────────────────────────────────────────

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
            profil              : sortie de build_cluster_profile()
            layer_dist          : sortie de build_layer_distribution()
            cluster_homogeneity : score peeling (couche 1)
            n                   : taille du cluster
            cluster_id          : label numérique optionnel

        Returns:
            dict complet avec name, slots, signature_vector, etc.
        """
        slot_results   = {}
        slots_primary  = []   # conf >= conf_min_name
        slots_secondary = []  # conf_min_display <= conf < conf_min_name
        slots_uncalib  = []   # calibrated=False
        signature      = {}   # slot_name → conf

        for slot_name in self.slot_order:
            slot_cfg = self.slots_cfg.get(slot_name, {})

            if not slot_cfg.get('calibrated', True):
                result = _eval_slot_uncalibrated(slot_name, profil, slot_cfg)
                result['slot'] = slot_name
                slots_uncalib.append(result)
                signature[slot_name] = None
                continue

            # Dispatch par slot
            result = None
            if slot_name == 'AMPLITUDE':
                result = _eval_slot_amplitude(profil, layer_dist, slot_cfg)
            elif slot_name == 'SANTÉ_NUM':
                result = _eval_slot_sante_num(profil, layer_dist, slot_cfg)
            elif slot_name == 'ENTROPIE':
                result = _eval_slot_delta('ENTROPIE', 'entropy_delta',
                                          profil, layer_dist, slot_cfg)
            elif slot_name == 'RANG':
                result = _eval_slot_delta('RANG', 'effective_rank_delta',
                                          profil, layer_dist, slot_cfg)
            elif slot_name == 'VITESSE':
                result = _eval_slot_delta('VITESSE', 'euclidean_norm__CO_FirstMin_ac',
                                          profil, layer_dist, slot_cfg)
            elif slot_name == 'TEXTURE':
                result = _eval_slot_delta('TEXTURE', 'entropy__MD_hrv_classic_pnn40',
                                          profil, layer_dist, slot_cfg)

            # Slot omis (None = slot neutre ou sain)
            if result is None:
                signature[slot_name] = 0.0
                continue

            result['slot'] = slot_name
            conf = result.get('conf', 0.0) or 0.0
            signature[slot_name] = conf

            if conf >= self.conf_min_name:
                slots_primary.append(result)
            elif conf >= self.conf_min_disp:
                slots_secondary.append(result)
            # En dessous de conf_min_display → slot ignoré à l'affichage

        # Nom composé
        name_terms = [s['term'] for s in slots_primary if s.get('term')]
        sec_terms  = [s['term'] for s in slots_secondary if s.get('term')]
        uncalib_present = len(slots_uncalib) > 0

        name = self.separator.join(name_terms) if name_terms else 'UNCATEGORIZED'

        # Nom complet avec secondaires entre parenthèses
        name_full_parts = list(name_terms)
        if sec_terms:
            name_full_parts.append(f'({self.separator.join(sec_terms)})')
        if uncalib_present:
            name_full_parts.append('·'.join(
                f'[{s["slot"]}=?]' for s in slots_uncalib
            ))
        name_full = self.separator.join(name_full_parts) if name_full_parts else 'UNCATEGORIZED'

        # Signature vector ordonnée
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

    # ── Nommage batch ────────────────────────────────────────────────────────

    def name_all(
        self,
        peeling_result  : Dict,
        all_features    : List[Dict],
    ) -> List[Dict]:
        """
        Nomme tous les clusters d'un résultat de peeling.

        Args:
            peeling_result : sortie de run_peeling()
            all_features   : features[i] pour tous les runs (pour layer_dist)

        Returns:
            Liste de dicts nom, un par cluster extrait + un pour le résidu.
        """
        layer_dist = build_layer_distribution(all_features)
        extracted  = peeling_result['extracted']
        labels     = peeling_result['labels']

        results = []

        for cluster_info in extracted:
            cid    = cluster_info['final_label']
            gidx   = cluster_info['global_indices']
            homo   = cluster_info['homogeneity']
            n      = cluster_info['n']

            profil = build_cluster_profile(gidx, all_features)
            named  = self.name_cluster(profil, layer_dist,
                                       cluster_homogeneity=homo,
                                       n=n, cluster_id=cid)
            named['level']       = cluster_info['level']
            named['metric']      = cluster_info['metric']
            named['config_ari']  = cluster_info.get('config_ari')
            results.append(named)

        # Résidu
        residual_idx = peeling_result['residual_idx'].tolist()
        if residual_idx:
            profil_res = build_cluster_profile(residual_idx, all_features)
            named_res  = self.name_cluster(profil_res, layer_dist,
                                           cluster_homogeneity=0.0,
                                           n=len(residual_idx),
                                           cluster_id=-1)
            named_res['name']     = 'RÉSIDU'
            named_res['name_full'] = 'RÉSIDU (non résolu)'
            named_res['level']    = -1
            results.append(named_res)

        return results


# ─────────────────────────────────────────────────────────────────────────────
# RAPPORT TEXTE
# ─────────────────────────────────────────────────────────────────────────────

def print_naming_report(named_clusters: List[Dict]):
    """Affiche un rapport lisible des clusters nommés."""
    print('\n' + '='*65)
    print('RAPPORT NOMMAGE CLUSTERS')
    print('='*65)

    for nc in sorted(named_clusters, key=lambda x: x.get('cluster_id', 999)):
        cid   = nc.get('cluster_id', '?')
        label = 'RÉSIDU' if cid == -1 else f'Cluster {cid:2d}'
        homo  = nc['cluster_homogeneity']
        het   = ' ⚠ hétérogène' if nc['heterogeneous'] else ''

        print(f'\n{label} (n={nc["n"]})  homo={homo:.2f}{het}')
        print(f'  Nom      : {nc["name"]}')
        if nc['name_full'] != nc['name']:
            print(f'  Complet  : {nc["name_full"]}')

        for s in nc['slots']:
            conf_str = f'{s["conf"]:.2f}' if s.get('conf') is not None else '—'
            print(f'  [{s["slot"]:15s}] {s["term"]:25s} conf={conf_str}'
                  f'  ({s.get("feature","?")}={s.get("value","?")})')

        if nc['slots_secondary']:
            print('  Secondaires :')
            for s in nc['slots_secondary']:
                conf_str = f'{s["conf"]:.2f}' if s.get('conf') is not None else '—'
                print(f'    [{s["slot"]:15s}] ({s["term"]})  conf={conf_str}')

        if nc['slots_uncalibrated']:
            slots_u = [s['slot'] for s in nc['slots_uncalibrated']]
            print(f'  Non calibrés : {", ".join(slots_u)}')

        sig = nc['signature_vector']
        sig_str = '  '.join(
            f'{o}={f"{v:.2f}" if v is not None else "?"}' 
            for o, v in zip(nc['slot_order'], sig)
        )
        print(f'  Signature : {sig_str}')

