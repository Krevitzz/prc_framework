"""
prc.analysing.visualizer

Responsabilité : Toutes les sorties visuelles du pipeline analysing.

Principe :
    Reçoit des données (labels, noms, M_2d, métadonnées).
    Produit des PNG. Zéro calcul, zéro clustering, zéro features.

Vues disponibles :
    plot_peeling_summary()   — clusters colorés + niveaux d'extraction
    plot_layer()             — une vue t-SNE par layer avec noms annotés
    plot_signature_heatmap() — matrice slots × clusters (conf par slot)
    plot_regimes()           — overlay régimes sur t-SNE (si dispo)
    plot_all()               — toutes les vues d'un coup

Migration :
    _plot_summary() de clustering_peeling.py → ici.
    clustering_peeling.py ne génère plus de PNG directement.

Usage :
    viz = ClusterVisualizer(M_2d, named_clusters, peeling_result,
                            run_regimes, gammas)
    viz.plot_all(output_dir='reports/analysing', label='poc3')
"""

import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings('ignore')

# ── Palette ──────────────────────────────────────────────────────────────────
DARK_BG    = '#0d1117'
PANEL_BG   = '#161b22'
GRID_COLOR = '#21262d'
TEXT_COLOR = '#c9d1d9'
DIM_COLOR  = '#444455'
ACCENT     = '#58a6ff'

REGIME_COLORS = {
    'CONSERVES_NORM'    : '#3498db',
    'EFFONDREMENT'      : '#9b59b6',
    'CROISSANCE_FAIBLE' : '#2ecc71',
    'CROISSANCE_FORTE'  : '#f39c12',
    'SATURATION'        : '#e67e22',
    'NUMERIC_INSTABILITY': '#e74c3c',
    'EXPLOSION_PROGRESSIVE': '#ff6b6b',
    'UNCATEGORIZED'     : '#95a5a6',
    'UNKNOWN'           : '#2d3142',
    'OUTLIER'           : '#1a1a2e',
}

LEVEL_COLORS = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']


# =============================================================================
# HELPERS STYLE
# =============================================================================

def _style(ax, title: str = ''):
    ax.set_facecolor(PANEL_BG)
    if title:
        ax.set_title(title, color=TEXT_COLOR, fontsize=10,
                     fontweight='bold', pad=7)
    ax.tick_params(colors=DIM_COLOR, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.5)


def _save(fig, path: str, dpi: int = 140):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f"  → {path}")


def _cluster_colormap(n_clusters: int):
    return plt.cm.get_cmap('tab20', max(n_clusters, 1))


# =============================================================================
# VISUALIZER
# =============================================================================

class ClusterVisualizer:
    """
    Générateur de vues pour le pipeline analysing.

    Args:
        M_2d           : np.ndarray (n, 2) — projection t-SNE
        named_clusters : List[Dict] — sortie ClusterNamer.name_all()
        peeling_result : Dict — sortie run_peeling()
        run_regimes    : List[str] (optionnel) — régime par run
        gammas         : List[str] (optionnel) — gamma_id par run
    """

    def __init__(
        self,
        M_2d           : np.ndarray,
        named_clusters : List[Dict],
        peeling_result : Dict,
        run_regimes    : Optional[List[str]] = None,
        gammas         : Optional[List[str]] = None,
    ):
        self.M_2d           = M_2d
        self.named_clusters = named_clusters
        self.peeling_result = peeling_result
        self.run_regimes    = run_regimes
        self.gammas         = gammas

        self.labels    = peeling_result.get('labels', np.full(len(M_2d), -1))
        self.extracted = peeling_result.get('extracted', [])
        self.n_clusters = peeling_result.get('n_clusters', 0)
        self.cmap       = _cluster_colormap(self.n_clusters)

        # Index: cluster_id → named_cluster dict
        self.nc_by_id = {nc.get('cluster_id'): nc for nc in named_clusters}

    # ── Vue 1 : Peeling summary ──────────────────────────────────────────────

    def plot_peeling_summary(self, output_dir: str, label: str):
        """
        Deux panels : clusters colorés + niveaux d'extraction.
        Annotation du nom composé au centroïde de chaque cluster.
        """
        out    = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        fig.patch.set_facecolor(DARK_BG)

        # Panel gauche — clusters + noms
        ax = axes[0]
        _style(ax, f'Peeling — {self.n_clusters} clusters')
        ax.set_xlabel('t-SNE 1', color=DIM_COLOR, fontsize=8)
        ax.set_ylabel('t-SNE 2', color=DIM_COLOR, fontsize=8)

        noise_mask = self.labels == -1
        ax.scatter(self.M_2d[noise_mask, 0], self.M_2d[noise_mask, 1],
                   c='#2d3142', s=5, alpha=0.4, edgecolors='none',
                   label=f'Résidu ({noise_mask.sum()})', zorder=1)

        for cid in range(self.n_clusters):
            mask = self.labels == cid
            if not mask.any():
                continue
            nc   = self.nc_by_id.get(cid, {})
            name = nc.get('name', f'C{cid}')
            lv   = nc.get('level', 0)

            ax.scatter(self.M_2d[mask, 0], self.M_2d[mask, 1],
                       c=[self.cmap(cid)], s=15, alpha=0.85,
                       edgecolors='none', zorder=3,
                       label=f'C{cid} n={mask.sum()} L{lv}')

            # Annotation nom au centroïde
            cx = float(np.median(self.M_2d[mask, 0]))
            cy = float(np.median(self.M_2d[mask, 1]))
            short = name[:22] + '…' if len(name) > 22 else name
            ax.annotate(short, (cx, cy), fontsize=5.5, color=TEXT_COLOR,
                        ha='center', va='center', alpha=0.85,
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor=DARK_BG, alpha=0.5,
                                  edgecolor='none'))

        ax.legend(framealpha=0.4, facecolor=DARK_BG, edgecolor=GRID_COLOR,
                  labelcolor=TEXT_COLOR, fontsize=6, markerscale=1.3,
                  ncol=2, loc='best')

        # Panel droit — niveaux d'extraction
        ax2 = axes[1]
        _style(ax2, 'Niveaux d\'extraction')
        ax2.set_xlabel('t-SNE 1', color=DIM_COLOR, fontsize=8)

        ax2.scatter(self.M_2d[noise_mask, 0], self.M_2d[noise_mask, 1],
                    c='#2d3142', s=5, alpha=0.4, edgecolors='none', zorder=1)

        by_level: Dict[int, List] = {}
        for ce in self.extracted:
            by_level.setdefault(ce['level'], []).append(ce)

        for lv, ces in sorted(by_level.items()):
            col   = LEVEL_COLORS[lv % len(LEVEL_COLORS)]
            gidx  = []
            for ce in ces:
                gidx.extend(ce['global_indices'])
            gidx = np.array(gidx)
            n_pts = len(gidx)
            ax2.scatter(self.M_2d[gidx, 0], self.M_2d[gidx, 1],
                        c=col, s=14, alpha=0.8, edgecolors='none', zorder=3,
                        label=f'Niveau {lv} ({n_pts} pts)')

        ax2.legend(framealpha=0.4, facecolor=DARK_BG, edgecolor=GRID_COLOR,
                   labelcolor=TEXT_COLOR, fontsize=8)

        n_unresolved = int((self.labels == -1).sum())
        n_total      = len(self.labels)
        fig.suptitle(
            f'{label} — {self.n_clusters} clusters | '
            f'résidu {n_unresolved} ({100*n_unresolved/max(n_total,1):.0f}%)',
            color=TEXT_COLOR, fontsize=11, fontweight='bold'
        )

        _save(fig, str(out / f'{label}_peeling_summary.png'))

    # ── Vue 2 : Par layer ────────────────────────────────────────────────────

    def plot_layer(self, layer_name: str, output_dir: str, label: str):
        """
        Vue t-SNE dédiée à un layer, avec noms annotés + signature slots.

        Utile pour comparer la structure détectée par chaque layer quand
        le pipeline multi-layer sera actif.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 9))
        fig.patch.set_facecolor(DARK_BG)
        _style(ax, f'Layer : {layer_name} — {self.n_clusters} clusters')
        ax.set_xlabel('t-SNE 1', color=DIM_COLOR, fontsize=8)
        ax.set_ylabel('t-SNE 2', color=DIM_COLOR, fontsize=8)

        noise_mask = self.labels == -1
        ax.scatter(self.M_2d[noise_mask, 0], self.M_2d[noise_mask, 1],
                   c='#2d3142', s=5, alpha=0.3, edgecolors='none', zorder=1)

        for cid in range(self.n_clusters):
            mask = self.labels == cid
            if not mask.any():
                continue
            nc   = self.nc_by_id.get(cid, {})
            name = nc.get('name', f'C{cid}')
            homo = nc.get('cluster_homogeneity', 0.0)

            ax.scatter(self.M_2d[mask, 0], self.M_2d[mask, 1],
                       c=[self.cmap(cid)], s=16, alpha=0.85,
                       edgecolors='none', zorder=3)

            cx = float(np.median(self.M_2d[mask, 0]))
            cy = float(np.median(self.M_2d[mask, 1]))
            ax.annotate(f'{name}\n(n={mask.sum()}, h={homo:.2f})',
                        (cx, cy), fontsize=6, color=TEXT_COLOR,
                        ha='center', va='center', alpha=0.9,
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor=DARK_BG, alpha=0.6,
                                  edgecolor=self.cmap(cid), linewidth=0.8))

        _save(fig, str(out / f'{label}_{layer_name}.png'))

    # ── Vue 3 : Signature heatmap ────────────────────────────────────────────

    def plot_signature_heatmap(self, output_dir: str, label: str):
        """
        Matrice slots × clusters — confiance par slot, couleur = valeur.

        Permet de comparer visuellement deux clusters au même nom
        mais d'intensité différente.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Filtrer clusters nommés (pas résidu)
        ncs = [nc for nc in self.named_clusters
               if nc.get('cluster_id', -1) >= 0]
        if not ncs:
            return

        slot_order = ncs[0].get('slot_order', [])
        if not slot_order:
            return

        n_slots    = len(slot_order)
        n_clusters = len(ncs)

        # Matrice conf [n_clusters × n_slots]
        data  = np.zeros((n_clusters, n_slots))
        masks = np.zeros((n_clusters, n_slots), dtype=bool)  # True = non calibré

        for i, nc in enumerate(ncs):
            sig = nc.get('signature_vector', [])
            for j, val in enumerate(sig[:n_slots]):
                if val is None:
                    masks[i, j] = True
                    data[i, j]  = 0.0
                else:
                    data[i, j]  = float(val)

        fig, ax = plt.subplots(figsize=(max(8, n_slots * 1.3),
                                        max(5, n_clusters * 0.6 + 1.5)))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(PANEL_BG)

        im = ax.imshow(data, aspect='auto', cmap='RdYlGn',
                       vmin=0, vmax=1, interpolation='nearest')

        # Annotations valeurs
        for i in range(n_clusters):
            for j in range(n_slots):
                if masks[i, j]:
                    txt = '?'
                    col = DIM_COLOR
                else:
                    v   = data[i, j]
                    txt = f'{v:.2f}'
                    col = 'black' if v > 0.45 else 'white'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=7.5, color=col, fontweight='bold')

        # Étiquettes
        cluster_labels = []
        for nc in ncs:
            cid  = nc.get('cluster_id', '?')
            name = nc.get('name', '?')
            n    = nc.get('n', 0)
            cluster_labels.append(f'C{cid} ({n}) {name[:18]}')

        ax.set_xticks(range(n_slots))
        ax.set_xticklabels(slot_order, color=TEXT_COLOR, fontsize=9, rotation=20)
        ax.set_yticks(range(n_clusters))
        ax.set_yticklabels(cluster_labels, color=TEXT_COLOR, fontsize=7.5)

        plt.colorbar(im, ax=ax, label='Confiance slot',
                     fraction=0.04, pad=0.02)
        ax.set_title(f'Signature heatmap — {label}',
                     color=TEXT_COLOR, fontsize=11, fontweight='bold', pad=10)

        _save(fig, str(out / f'{label}_signature_heatmap.png'), dpi=130)

    # ── Vue 4 : Overlay régimes ──────────────────────────────────────────────

    def plot_regimes(self, output_dir: str, label: str):
        """
        Overlay régimes sur t-SNE — comparaison avec clusters peeling.
        Uniquement si run_regimes est fourni.
        """
        if not self.run_regimes:
            return

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 9))
        fig.patch.set_facecolor(DARK_BG)
        _style(ax, f'Régimes overlay — {label}')
        ax.set_xlabel('t-SNE 1', color=DIM_COLOR, fontsize=8)
        ax.set_ylabel('t-SNE 2', color=DIM_COLOR, fontsize=8)

        regime_order = [
            'UNKNOWN', 'UNCATEGORIZED', 'OUTLIER',
            'NUMERIC_INSTABILITY', 'EXPLOSION_PROGRESSIVE',
            'SATURATION', 'CROISSANCE_FORTE', 'CROISSANCE_FAIBLE',
            'EFFONDREMENT', 'CONSERVES_NORM',
        ]

        for rname in regime_order:
            mask = np.array([r == rname for r in self.run_regimes])
            if not mask.any():
                continue
            c      = REGIME_COLORS.get(rname, '#999')
            is_dim = rname in ('UNKNOWN', 'UNCATEGORIZED', 'OUTLIER')
            ax.scatter(self.M_2d[mask, 0], self.M_2d[mask, 1],
                       c=c, s=5 if is_dim else 15,
                       alpha=0.2 if is_dim else 0.8,
                       edgecolors='none', zorder=3,
                       label=f'{rname} ({mask.sum()})')

        ax.legend(framealpha=0.4, facecolor=DARK_BG, edgecolor=GRID_COLOR,
                  labelcolor=TEXT_COLOR, fontsize=7, markerscale=1.3)

        _save(fig, str(out / f'{label}_regimes.png'))

    # ── Plot all ─────────────────────────────────────────────────────────────

    def plot_all(self, output_dir: str, label: str):
        """Génère toutes les vues disponibles."""
        self.plot_peeling_summary(output_dir, label)
        self.plot_signature_heatmap(output_dir, label)

        if self.run_regimes:
            self.plot_regimes(output_dir, label)

        # Vues par layer — pour l'instant une seule vue "universal"
        # sera étendu quand multi-layer actif
        self.plot_layer('universal', output_dir, label)
