"""
clustering_peeling.py

Residual peeling HDBSCAN — extraction progressive par homogénéité.

Principe :
    Niveau 0 : cosine/eom sur l'espace complet.
    Chaque cluster détecté est évalué par un score d'homogénéité composite.
    Si homogénéité >= seuil(niveau) → extrait définitivement.
    Sinon → réintègre le résidu avec le bruit.
    Niveaux suivants : batterie de configs sur le résidu uniquement.
    Arrêt si résidu trop petit ou rien de nouveau trouvé.

Usage depuis clustering_lite.py :
    from clustering_peeling import run_peeling
    result = run_peeling(M_ortho, config, M_2d=M_2d, run_regimes=run_regimes)

Usage standalone (test) :
    python clustering_peeling.py --parquet poc3.parquet --config clustering_peeling.yaml
"""

import warnings
import numpy as np
import yaml
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> Dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg['peeling']


def _threshold(cfg: Dict, level: int) -> float:
    """Seuil homogénéité pour un niveau donné."""
    base = cfg['homogeneity']['threshold_base']
    step = cfg['homogeneity']['threshold_step']
    maxi = cfg['homogeneity']['threshold_max']
    return min(maxi, base + level * step)


# ─────────────────────────────────────────────────────────────────────────────
# mcs ADAPTATIF SUR RÉSIDU
# ─────────────────────────────────────────────────────────────────────────────

def _mcs_residual(cfg: Dict, mcs_global: int,
                  n_total: int, n_residual: int,
                  M_residual: np.ndarray,
                  M_global: np.ndarray) -> int:
    """
    Calcule mcs adaptatif pour le résidu.

    mcs = max(floor, int(mcs_global × sqrt(n_résidu/n_total) / density_factor))

    density_factor = densité résidu / densité globale
    densité = n / trace(covariance)  — proxy "points par unité de volume"
    """
    floor = cfg['mcs_floor']

    if not cfg['mcs_adaptive']['enabled'] or n_residual >= n_total:
        return max(floor, mcs_global)

    # Fraction taille
    size_factor = np.sqrt(n_residual / n_total)

    # Densité
    try:
        spread_res = np.trace(np.cov(M_residual.T)) + 1e-10
        spread_all = np.trace(np.cov(M_global.T))   + 1e-10
        dens_res   = n_residual / spread_res
        dens_all   = n_total    / spread_all
        df_min     = cfg['mcs_adaptive']['density_factor_min']
        df_max     = cfg['mcs_adaptive']['density_factor_max']
        density_factor = float(np.clip(dens_res / dens_all, df_min, df_max))
    except Exception:
        density_factor = 1.0

    mcs = max(floor, int(mcs_global * size_factor / density_factor))
    return mcs


# ─────────────────────────────────────────────────────────────────────────────
# SCORE HOMOGÉNÉITÉ
# ─────────────────────────────────────────────────────────────────────────────

def _homogeneity_score(
    M_cluster       : np.ndarray,
    proba_cluster   : np.ndarray,
    M_others        : Optional[np.ndarray],
    labels_cluster  : np.ndarray,
    run_regimes     : Optional[List[str]],
    cfg             : Dict,
) -> float:
    """
    Score composite [0, 1] :
        mean_probability    × weight_proba
      + silhouette_cluster  × weight_silh   (normalisé 0-1)
      + regime_purity       × weight_regime (si régimes dispo)
    """
    hcfg  = cfg['homogeneity']
    wp    = hcfg['weight_probability']
    ws    = hcfg['weight_silhouette']
    wr    = hcfg['weight_regime']

    # mean_probability
    s_proba = float(np.mean(proba_cluster)) if len(proba_cluster) > 0 else 0.0

    # silhouette : nécessite au moins un "autre" cluster ou groupe
    s_silh = 0.5  # Neutre si non calculable
    if M_others is not None and len(M_others) >= 2 and len(M_cluster) >= 2:
        max_s = cfg['silhouette']['max_samples']
        M_all   = np.vstack([M_cluster, M_others])
        lbs_all = np.array([0] * len(M_cluster) + [1] * len(M_others))
        if len(M_all) > max_s:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(M_all), max_s, replace=False)
            M_all, lbs_all = M_all[idx], lbs_all[idx]
        try:
            raw_sil = silhouette_score(M_all, lbs_all)
            # Normaliser [-1,1] → [0,1]
            s_silh  = (raw_sil + 1) / 2
        except Exception:
            pass

    # regime_purity (optionnel)
    s_regime = None
    if run_regimes and len(run_regimes) == len(labels_cluster):
        cnt     = Counter(run_regimes)
        dominant = cnt.most_common(1)[0][1]
        s_regime = dominant / len(run_regimes)

    # Composition pondérée
    if s_regime is not None:
        total_w = wp + ws + wr
        score   = (wp * s_proba + ws * s_silh + wr * s_regime) / total_w
    else:
        total_w = wp + ws
        score   = (wp * s_proba + ws * s_silh) / total_w

    return float(np.clip(score, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP ARI
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_ari(M: np.ndarray, hdb_kwargs: Dict, cfg: Dict) -> float:
    if not cfg['bootstrap']['enabled']:
        return float('nan')
    n_iter = cfg['bootstrap']['n_iter']
    frac   = cfg['bootstrap']['subsample']
    n      = len(M)
    rng    = np.random.RandomState(42)
    try:
        ref = HDBSCAN(**hdb_kwargs).fit_predict(M)
    except Exception:
        return float('nan')
    aris = []
    for _ in range(n_iter):
        idx = rng.choice(n, max(10, int(n * frac)), replace=False)
        try:
            sub = HDBSCAN(**hdb_kwargs).fit_predict(M[idx])
            rv  = ref[idx]
            ok  = (sub != -1) & (rv != -1)
            if ok.sum() >= 5:
                aris.append(adjusted_rand_score(rv[ok], sub[ok]))
        except Exception:
            continue
    return float(np.mean(aris)) if aris else float('nan')


# ─────────────────────────────────────────────────────────────────────────────
# SCORE COMPOSITE CONFIG (sélection config résidu)
# ─────────────────────────────────────────────────────────────────────────────

def _config_score(n_clusters: int, pct_bruit: float,
                  ari: float, target: Tuple = (3, 10)) -> float:
    s = 0.0
    if target[0] <= n_clusters <= target[1]:
        s += 3.0
    elif n_clusters > 0:
        s += max(0, 3.0 - 0.5 * min(abs(n_clusters - target[0]),
                                     abs(n_clusters - target[1])))
    s -= 0.03 * pct_bruit
    if not np.isnan(ari):
        s += 2.0 * ari
    return s


# ─────────────────────────────────────────────────────────────────────────────
# UN NIVEAU DE PEELING
# ─────────────────────────────────────────────────────────────────────────────

def _run_level(
    M_level         : np.ndarray,
    global_idx      : np.ndarray,   # indices originaux des points du résidu
    M_global        : np.ndarray,
    cfg             : Dict,
    level           : int,
    mcs             : int,
    ms              : int,
    configs         : List[Dict],
    run_regimes_all : Optional[List[str]],
    selection_mode  : str,
) -> Tuple[Dict, np.ndarray]:
    """
    Applique la batterie de configs sur M_level.
    Choisit la meilleure config selon selection_mode.
    Retourne (résultat_niveau, indices_résidu_global).

    résultat_niveau :
        extracted : List[Dict] — clusters extraits avec métadonnées
        residual_local_idx : np.ndarray — indices locaux restant dans le résidu
    """
    n_level = len(M_level)
    threshold = _threshold(cfg, level)

    best_config_result = None
    best_score         = -np.inf

    for hcfg in configs:
        metric    = hcfg['metric']
        selection = hcfg['selection']
        pca_dims  = hcfg.get('pca_dims')

        # Réduction optionnelle
        if pca_dims and pca_dims < M_level.shape[1]:
            M_use = PCA(n_components=min(pca_dims, M_level.shape[1]-1),
                        random_state=42).fit_transform(M_level)
        else:
            M_use = M_level

        kw = dict(min_cluster_size=mcs, min_samples=ms,
                  metric=metric, cluster_selection_method=selection)
        try:
            hdb   = HDBSCAN(**kw)
            lbs   = hdb.fit_predict(M_use)
            proba = hdb.probabilities_
        except Exception:
            continue

        n_cl   = len(set(lbs) - {-1})
        pct_b  = 100 * (lbs == -1).sum() / n_level
        ari    = _bootstrap_ari(M_use, kw, cfg)
        sc     = _config_score(n_cl, pct_b, ari)

        if sc > best_score:
            best_score         = sc
            best_config_result = (lbs, proba, M_use, metric, selection, pca_dims, ari)

    if best_config_result is None:
        return {'extracted': [], 'level': level}, global_idx

    lbs, proba, M_use, metric, selection, pca_dims, ari = best_config_result
    print(f'  Niveau {level} — config retenue : {metric}/{selection}'
          f'{"PCA"+str(pca_dims) if pca_dims else ""}'
          f'  n_cl={len(set(lbs)-{-1})}  bruit={100*(lbs==-1).sum()/n_level:.0f}%'
          f'  ARI={ari:.2f}')

    # Évaluation homogénéité par cluster
    extracted        = []
    residual_local   = list(np.where(lbs == -1)[0])  # bruit → résidu direct

    cluster_ids = sorted(set(lbs) - {-1})
    for cid in cluster_ids:
        local_mask  = lbs == cid
        local_idx   = np.where(local_mask)[0]
        M_cluster   = M_use[local_mask]
        proba_c     = proba[local_mask]

        # Contexte "autres points" pour silhouette
        others_mask = (lbs != cid) & (lbs != -1)
        M_others    = M_use[others_mask] if others_mask.any() else None

        # Régimes locaux si disponibles
        regimes_c = None
        if run_regimes_all is not None:
            regimes_c = [run_regimes_all[global_idx[i]] for i in local_idx]

        h_score = _homogeneity_score(M_cluster, proba_c, M_others,
                                      local_mask, regimes_c, cfg)

        regime_dist = None
        if regimes_c:
            regime_dist = dict(Counter(regimes_c).most_common())

        print(f'    C{cid} n={local_mask.sum():3d}  '
              f'homogénéité={h_score:.2f}  '
              f'(seuil={threshold:.2f})  '
              f'→ {"EXTRAIT" if h_score >= threshold else "résidu"}')

        if h_score >= threshold:
            extracted.append({
                'level'         : level,
                'local_cluster' : int(cid),
                'global_indices': global_idx[local_idx].tolist(),
                'n'             : int(local_mask.sum()),
                'homogeneity'   : float(h_score),
                'threshold'     : float(threshold),
                'metric'        : metric,
                'selection'     : selection,
                'pca_dims'      : pca_dims,
                'config_ari'    : float(ari) if not np.isnan(ari) else None,
                'regime_dist'   : regime_dist,
            })
        else:
            residual_local.extend(local_idx.tolist())

    residual_global = global_idx[sorted(set(residual_local))]

    return {
        'extracted'      : extracted,
        'level'          : level,
        'metric'         : metric,
        'selection'      : selection,
        'pca_dims'       : pca_dims,
        'config_ari'     : float(ari) if not np.isnan(ari) else None,
        'threshold'      : float(threshold),
        'n_input'        : n_level,
        'n_extracted'    : sum(e['n'] for e in extracted),
        'n_residual_out' : len(residual_global),
    }, residual_global


# ─────────────────────────────────────────────────────────────────────────────
# PEELING PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run_peeling(
    M_ortho         : np.ndarray,
    cfg             : Dict,
    mcs_global      : Optional[int]  = None,
    ms_global       : Optional[int]  = None,
    M_2d            : Optional[np.ndarray] = None,
    run_regimes     : Optional[List[str]]  = None,
    output_dir      : Optional[str]        = None,
    label           : str = 'peeling',
) -> Dict:
    """
    Residual peeling complet.

    Args:
        M_ortho     : matrice features orthogonales (n × f)
        cfg         : config chargée depuis clustering_peeling.yaml
        mcs_global  : min_cluster_size niveau 0 (défaut n//30)
        ms_global   : min_samples (défaut n//100)
        M_2d        : projection t-SNE pour visualisations (optionnel)
        run_regimes : régimes par run pour homogénéité (optionnel)
        output_dir  : répertoire sorties (optionnel)
        label       : préfixe fichiers

    Returns:
        {
          'labels'        : np.ndarray (n,) — label final, -1 si résidu non résolu
          'extracted'     : List[Dict] — tous les clusters extraits avec trace
          'residual_idx'  : np.ndarray — indices runs non résolus
          'trace'         : List[Dict] — état de chaque niveau
          'n_levels'      : int
        }
    """
    n          = len(M_ortho)
    mcs_global = mcs_global or max(cfg['mcs_floor'], n // 30)
    ms_global  = ms_global  or max(cfg['ms_floor'],  n // 100)
    max_levels = cfg['max_levels']
    floor      = cfg['mcs_floor']

    print(f'\nResidual Peeling — {n} runs × {M_ortho.shape[1]} features')
    print(f'mcs_global={mcs_global}, ms={ms_global}, max_levels={max_levels}')
    print(f'Seuil homogénéité : base={cfg["homogeneity"]["threshold_base"]}'
          f'  step={cfg["homogeneity"]["threshold_step"]}'
          f'  max={cfg["homogeneity"]["threshold_max"]}')

    # État global
    labels          = np.full(n, -1, dtype=int)
    next_label      = 0
    all_extracted   = []
    trace           = []
    residual_idx    = np.arange(n)   # Commence avec tous les points

    # Configs niveau 0 (une seule)
    l0 = cfg['level_0']
    configs_l0 = [{'metric': l0['metric'], 'selection': l0['selection'],
                   'pca_dims': l0.get('pca_dims')}]

    # Configs résidu
    configs_residual = cfg['residual_configs']

    for level in range(max_levels):
        n_res = len(residual_idx)
        print(f'\n{"="*55}')
        print(f'Niveau {level} — {n_res} points dans le résidu')
        print(f'{"="*55}')

        if n_res < floor:
            print(f'  Résidu trop petit ({n_res} < floor={floor}) — arrêt')
            break

        # Sélection configs et mcs pour ce niveau
        if level == 0:
            configs = configs_l0
            mcs     = mcs_global
            ms      = ms_global
        else:
            configs = configs_residual
            M_res   = M_ortho[residual_idx]
            mcs     = _mcs_residual(cfg, mcs_global, n, n_res, M_res, M_ortho)
            ms      = max(cfg['ms_floor'], mcs // 4)
            print(f'  mcs adaptatif={mcs}, ms={ms}')

        # Appliquer le niveau
        M_level = M_ortho[residual_idx]
        level_result, new_residual_idx = _run_level(
            M_level        = M_level,
            global_idx     = residual_idx,
            M_global       = M_ortho,
            cfg            = cfg,
            level          = level,
            mcs            = mcs,
            ms             = ms,
            configs        = configs,
            run_regimes_all= run_regimes,
            selection_mode = cfg['residual_selection'],
        )

        # Assigner labels permanents aux clusters extraits
        for cluster_info in level_result['extracted']:
            gidx = np.array(cluster_info['global_indices'])
            labels[gidx] = next_label
            cluster_info['final_label'] = next_label
            all_extracted.append(cluster_info)
            next_label += 1

        n_extracted_this_level = level_result['n_extracted']
        trace.append({
            'level'       : level,
            'n_input'     : n_res,
            'n_extracted' : n_extracted_this_level,
            'n_residual'  : len(new_residual_idx),
            'metric'      : level_result.get('metric'),
            'selection'   : level_result.get('selection'),
            'threshold'   : level_result.get('threshold'),
        })

        print(f'  → Extraits ce niveau : {n_extracted_this_level}  '
              f'Résidu restant : {len(new_residual_idx)}')

        # Critère d'arrêt : rien extrait depuis 2 niveaux
        if (level >= 1 and
            trace[-1]['n_extracted'] == 0 and
            trace[-2]['n_extracted'] == 0):
            print('  Rien extrait depuis 2 niveaux — arrêt')
            break

        if n_extracted_this_level < cfg['min_delta_extracted'] and level > 0:
            print(f'  Delta extrait ({n_extracted_this_level}) < '
                  f'min_delta ({cfg["min_delta_extracted"]}) — arrêt')
            break

        residual_idx = new_residual_idx

    # Résidu final
    n_unresolved = int((labels == -1).sum())
    print(f'\n{"="*55}')
    print(f'Peeling terminé — {next_label} clusters extraits')
    print(f'Résidu non résolu : {n_unresolved} runs ({100*n_unresolved/n:.1f}%)')

    for ce in all_extracted:
        rd = ce.get('regime_dist', {})
        top = list(rd.items())[:2] if rd else []
        print(f'  Cluster {ce["final_label"]:2d} '
              f'(niveau {ce["level"]}) '
              f'n={ce["n"]:3d}  '
              f'homo={ce["homogeneity"]:.2f}  '
              f'{ce["metric"]}/{ce["selection"]}  '
              f'régimes={top}')

    result = {
        'labels'       : labels,
        'extracted'    : all_extracted,
        'residual_idx' : residual_idx,
        'trace'        : trace,
        'n_levels'     : len(trace),
        'n_clusters'   : next_label,
        'n_unresolved' : n_unresolved,
    }

    # Sorties optionnelles
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if cfg['output']['save_labels']:
            np.save(str(out / f'{label}_labels.npy'), labels)

        if cfg['output']['save_trace']:
            import json
            with open(str(out / f'{label}_trace.json'), 'w') as f:
                # Rendre sérialisable
                trace_serial = []
                for ce in all_extracted:
                    c = {k: v for k, v in ce.items() if k != 'global_indices'}
                    c['global_indices_count'] = ce['n']
                    trace_serial.append(c)
                json.dump({'trace': trace, 'extracted': trace_serial,
                           'n_clusters': next_label,
                           'n_unresolved': n_unresolved}, f, indent=2)
            print(f'Trace → {out / f"{label}_trace.json"}')

        if cfg['output']['plot_summary'] and M_2d is not None:
            _plot_summary(result, M_2d, run_regimes, out, label)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG    = '#0d1117'
PANEL_BG   = '#161b22'
GRID_COLOR = '#21262d'
TEXT_COLOR = '#c9d1d9'
DIM_COLOR  = '#444455'

REGIME_COLORS = {
    'CONSERVES_NORM':    '#3498db', 'EFFONDREMENT':       '#9b59b6',
    'CROISSANCE_FAIBLE': '#2ecc71', 'CROISSANCE_FORTE':   '#f39c12',
    'SATURATION':        '#e67e22', 'NUMERIC_INSTABILITY':'#e74c3c',
    'UNCATEGORIZED':     '#95a5a6', 'UNKNOWN':            '#2d3142',
}


def _plot_summary(result: Dict, M_2d: np.ndarray,
                  run_regimes: Optional[List[str]],
                  out: Path, label: str):
    """t-SNE final coloré par cluster peeling + niveaux d'extraction."""
    labels      = result['labels']
    n_clusters  = result['n_clusters']
    extracted   = result['extracted']

    # Grouper extraits par niveau pour les annotations
    by_level = {}
    for ce in extracted:
        by_level.setdefault(ce['level'], []).append(ce)

    n_levels = len(by_level)
    fig, axes = plt.subplots(1, min(n_levels + 1, 3), figsize=(7 * min(n_levels+1, 3), 7))
    if not hasattr(axes, '__len__'):
        axes = [axes]
    fig.patch.set_facecolor(DARK_BG)

    cmap = plt.cm.get_cmap('tab20', max(n_clusters, 1))

    def _style(ax, title):
        ax.set_facecolor(PANEL_BG)
        ax.set_title(title, color=TEXT_COLOR, fontsize=10, fontweight='bold', pad=7)
        ax.tick_params(colors=DIM_COLOR, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_COLOR)

    # Panel 1 — clusters finaux
    ax = axes[0]
    _style(ax, f'Peeling final — {n_clusters} clusters')
    noise_mask = labels == -1
    ax.scatter(M_2d[noise_mask, 0], M_2d[noise_mask, 1],
               c='#2d3142', s=5, alpha=0.4, edgecolors='none',
               label=f'Résidu ({noise_mask.sum()})', zorder=1)
    for cid in range(n_clusters):
        mask = labels == cid
        if not mask.any():
            continue
        lv = next((ce['level'] for ce in extracted if ce['final_label'] == cid), 0)
        ax.scatter(M_2d[mask, 0], M_2d[mask, 1], c=[cmap(cid)],
                   s=15, alpha=0.85, edgecolors='none', zorder=3,
                   label=f'C{cid} n={mask.sum()} L{lv}')
    ax.legend(framealpha=0.4, facecolor=DARK_BG, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=6.5, markerscale=1.3,
              ncol=2, loc='best')

    # Panel 2 — niveau d'extraction
    if len(axes) > 1:
        ax2 = axes[1]
        _style(ax2, 'Niveau d\'extraction')
        level_colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']
        ax2.scatter(M_2d[noise_mask, 0], M_2d[noise_mask, 1],
                    c='#2d3142', s=5, alpha=0.4, edgecolors='none', zorder=1)
        for lv, ces in by_level.items():
            col  = level_colors[lv % len(level_colors)]
            gidx = []
            for ce in ces:
                gidx.extend(ce['global_indices'])
            gidx = np.array(gidx)
            ax2.scatter(M_2d[gidx, 0], M_2d[gidx, 1],
                        c=col, s=14, alpha=0.8, edgecolors='none', zorder=3,
                        label=f'Niveau {lv} ({len(gidx)} pts)')
        ax2.legend(framealpha=0.4, facecolor=DARK_BG, edgecolor=GRID_COLOR,
                   labelcolor=TEXT_COLOR, fontsize=8)

    # Panel 3 — régimes si disponibles
    if len(axes) > 2 and run_regimes:
        ax3 = axes[2]
        _style(ax3, 'Régimes')
        for rname in ['UNKNOWN', 'UNCATEGORIZED', 'NUMERIC_INSTABILITY', 'SATURATION',
                       'CROISSANCE_FORTE', 'CROISSANCE_FAIBLE', 'EFFONDREMENT', 'CONSERVES_NORM']:
            mask = np.array([r == rname for r in run_regimes])
            if not mask.any():
                continue
            c = REGIME_COLORS.get(rname, '#999')
            ax3.scatter(M_2d[mask, 0], M_2d[mask, 1], c=c,
                        s=6 if rname in ('UNKNOWN', 'UNCATEGORIZED') else 14,
                        alpha=0.25 if rname in ('UNKNOWN', 'UNCATEGORIZED') else 0.8,
                        edgecolors='none', zorder=3, label=f'{rname} ({mask.sum()})')
        ax3.legend(framealpha=0.4, facecolor=DARK_BG, edgecolor=GRID_COLOR,
                   labelcolor=TEXT_COLOR, fontsize=7)

    n_unresolved = result['n_unresolved']
    n_total      = len(labels)
    plt.suptitle(
        f'Residual Peeling — {n_clusters} clusters extraits '
        f'| résidu {n_unresolved} ({100*n_unresolved/n_total:.0f}%)',
        color=TEXT_COLOR, fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    p = str(out / f'{label}_peeling_summary.png')
    plt.savefig(p, dpi=140, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f'Plot → {p}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI / TEST STANDALONE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    import pandas as pd

    # Constantes pipeline (reproduction clustering_lite)
    COMPOSITION_COLS = {'phase', 'gamma_id', 'encoding_id', 'modifier_id',
                        'n_dof', 'max_iterations', 'regime'}
    FLAG_PREFIXES    = ('has_', 'is_')
    PROTECTED        = {'norm_ratio', 'norm_final', 'entropy_delta',
                        'effective_rank_delta', 'log_condition_delta',
                        'spread_ratio', 'condition_number_svd_final'}
    PROT_SUFFIXES    = ('signal_finite_ratio', 'signal_norm_absolute')

    def _prepare(df):
        fcols = sorted([c for c in df.columns
                        if c not in COMPOSITION_COLS
                        and not any(c.startswith(p) for p in FLAG_PREFIXES)])

        def sanitize(v):
            if pd.isna(v):           return 0.0
            if v == float('inf'):    return  1e15
            if v == float('-inf'):   return -1e15
            return float(v)

        try:
            M_raw = df[fcols].map(sanitize).values.astype(float)
        except AttributeError:
            M_raw = df[fcols].applymap(sanitize).values.astype(float)

        M = M_raw.copy()
        for j in range(M.shape[1]):
            col = M[:, j]
            nz  = np.abs(col[np.abs(col) > 0])
            if len(nz) and np.max(np.abs(col)) / (np.min(nz) + 1e-10) > 1e6:
                M[:, j] = np.sign(col) * np.log1p(np.abs(col))

        M_sc = RobustScaler().fit_transform(M)

        # Sélection orthogonale avec protected
        protected = set(PROTECTED)
        for name in fcols:
            if any(name.endswith(s) for s in PROT_SUFFIXES):
                protected.add(name)

        prot_idx  = [i for i, k in enumerate(fcols) if k in protected]
        unprot_idx = [i for i, k in enumerate(fcols) if k not in protected]
        variances  = np.var(M_sc, axis=0)
        order_up   = sorted(unprot_idx, key=lambda i: variances[i], reverse=True)

        with np.errstate(invalid='ignore'):
            corr = np.abs(np.corrcoef(M_sc.T))
        corr = np.nan_to_num(corr, nan=1.0)

        kept = [i for i in prot_idx if variances[i] > 0]
        for idx in order_up:
            if variances[idx] == 0:
                continue
            if not kept or np.max(corr[idx, kept]) < 0.85:
                kept.append(idx)

        kept = sorted(kept)
        return M_sc[:, kept], [fcols[i] for i in kept]

    parser = argparse.ArgumentParser(description='Residual Peeling HDBSCAN')
    parser.add_argument('--parquet', required=True)
    parser.add_argument('--config',  required=True)
    parser.add_argument('--phase',   default=None)
    parser.add_argument('--out',     default='reports/peeling')
    parser.add_argument('--label',   default='peeling')
    parser.add_argument('--mcs',     type=int, default=None)
    parser.add_argument('--ms',      type=int, default=None)
    parser.add_argument('--tsne-cache', default=None)
    args = parser.parse_args()

    # Chargement
    df = pd.read_parquet(args.parquet)
    if args.phase and 'phase' in df.columns:
        df = df[df['phase'] == args.phase].reset_index(drop=True)
    print(f'Parquet : {len(df)} runs')

    run_regimes = df['regime'].tolist() if 'regime' in df.columns else None

    M_ortho, kept_names = _prepare(df)
    print(f'M_ortho : {M_ortho.shape}')

    # t-SNE
    if args.tsne_cache and Path(args.tsne_cache).exists():
        M_2d = np.load(args.tsne_cache)
        print(f't-SNE chargé depuis {args.tsne_cache}')
    else:
        from sklearn.manifold import TSNE
        print('Calcul t-SNE...')
        M_pca = PCA(n_components=min(50, M_ortho.shape[1]),
                    random_state=42).fit_transform(M_ortho)
        M_2d  = TSNE(n_components=2, perplexity=30,
                     max_iter=1000, random_state=42).fit_transform(M_pca)
        if args.tsne_cache:
            np.save(args.tsne_cache, M_2d)

    cfg = load_config(args.config)

    result = run_peeling(
        M_ortho     = M_ortho,
        cfg         = cfg,
        mcs_global  = args.mcs,
        ms_global   = args.ms,
        M_2d        = M_2d,
        run_regimes = run_regimes,
        output_dir  = args.out,
        label       = args.label,
    )
