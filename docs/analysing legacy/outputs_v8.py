"""
Sorties fichiers : verdict JSON/TXT + visualisations PNG.

Zéro calcul. Reçoit des données, sérialise, écrit. M_2d exclu du JSON.
get_cmap corrigé (matplotlib.colormaps au lieu de plt.cm.get_cmap déprécié).

@ROLE    Sorties : sérialisation verdict + génération PNG
@LAYER   analysing

"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional


# =========================================================================
# JSON
# =========================================================================

def write_verdict_report(verdict_results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _serial(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        raise TypeError(f"Non-sérialisable : {type(obj)}")

    result = {k: v for k, v in verdict_results.items() if k != 'M_2d'}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=_serial)
    print(f"✓ JSON : {output_path}")


# =========================================================================
# TXT
# =========================================================================

def write_verdict_report_txt(verdict_results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    meta = verdict_results.get('metadata', {})
    phase = meta.get('label', output_path.stem).upper()

    lines.append(f"=== VERDICT {phase} ===")
    lines.append(f"Observations : {meta.get('n_observations', 0)}")
    lines.append(f"Gammas : {meta.get('n_gammas', 0)}  |  Encodings : {meta.get('n_encodings', 0)}")
    lines.append(f"Features : {meta.get('n_features_ortho', 0)}/{meta.get('n_features_total', 0)} orthogonales")
    lines.append(f"Ortho seuil : {meta.get('ortho_threshold', '?')}  |  NaN imputés : {meta.get('n_nan_imputed', 0)}")
    lines.append("")

    outliers = verdict_results.get('outliers', {})
    n_out = outliers.get('n_outliers', 0)
    lines.append(f"OUTLIERS ({n_out} runs, {outliers.get('outlier_fraction', 0) * 100:.1f}%)")
    if n_out > 0:
        gamma_rec = outliers.get('recurrence', {}).get('gamma', {})
        if gamma_rec:
            lines.append("  Récurrence gammas :")
            for gid, rec in list(gamma_rec.items())[:5]:
                lines.append(f"    {gid} : {rec['count']}/{rec['total_subset']} ({rec['fraction'] * 100:.0f}%)")
    lines.append("")

    named = verdict_results.get('named_clusters', [])
    n_cl = meta.get('n_clusters', 0)
    n_unres = meta.get('n_unresolved', 0)
    lines.append(f"CLUSTERS ({n_cl} extraits, {n_unres} résidu)")
    lines.append("")

    for nc in sorted(named, key=lambda x: x.get('cluster_id', 999)):
        cid = nc.get('cluster_id', '?')
        if cid == -1:
            lines.append(f"  RÉSIDU ({nc.get('n', 0)} runs — non résolu)")
            lines.append("")
            continue
        name = nc.get('name', '?')
        homo = nc.get('cluster_homogeneity', 0.0)
        het = ' ⚠ hétérogène' if nc.get('heterogeneous') else ''
        lines.append(f"  Cluster {cid} ({nc.get('n', 0)} runs, niveau {nc.get('level', '?')}) homo={homo:.2f}{het}")
        lines.append(f"    Nom : {name}")
        for s in nc.get('slots', []):
            conf_s = f'{s["conf"]:.2f}' if s.get('conf') is not None else '—'
            lines.append(f"    [{s.get('slot', ''):18s}] {s.get('term', '?'):28s} conf={conf_s}")
        sig = nc.get('signature_vector', [])
        slot_ord = nc.get('slot_order', [])
        if sig and slot_ord:
            lines.append(f"    Signature : {'  '.join(f'{o}={f'{v:.2f}' if v is not None else '?'}' for o, v in zip(slot_ord, sig))}")
        lines.append("")

    lines.append("INSIGHTS")
    pure = [nc for nc in named if nc.get('cluster_id', -1) >= 0 and not nc.get('heterogeneous')]
    if pure:
        best = max(pure, key=lambda x: x.get('cluster_homogeneity', 0))
        lines.append(f"  - Meilleur : C{best['cluster_id']} '{best['name']}' (homo={best['cluster_homogeneity']:.2f})")
    if n_unres > 0:
        lines.append(f"  - Résidu : {n_unres} runs ({100 * n_unres / max(meta.get('n_observations', 1), 1):.0f}%)")
    lines.append("")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"✓ TXT : {output_path}")


# =========================================================================
# VISUALIZER
# =========================================================================

DARK_BG = '#0d1117'
PANEL_BG = '#161b22'
GRID_COLOR = '#21262d'
TEXT_COLOR = '#c9d1d9'
DIM_COLOR = '#444455'
LEVEL_COLORS = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']


def _style(ax, title=''):
    ax.set_facecolor(PANEL_BG)
    if title:
        ax.set_title(title, color=TEXT_COLOR, fontsize=10, fontweight='bold', pad=7)
    ax.tick_params(colors=DIM_COLOR, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.5)


def _save(fig, path, dpi=140):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f"  → {path}")


def _cluster_colormap(n):
    return matplotlib.colormaps['tab20'].resampled(max(n, 1))


class ClusterVisualizer:
    def __init__(self, M_2d, named_clusters, peeling_result,
                 run_regimes=None, gammas=None):
        self.M_2d = M_2d
        self.named_clusters = named_clusters
        self.peeling_result = peeling_result
        self.labels = peeling_result.get('labels', np.full(len(M_2d), -1))
        self.extracted = peeling_result.get('extracted', [])
        self.n_clusters = peeling_result.get('n_clusters', 0)
        self.cmap = _cluster_colormap(self.n_clusters)
        self.nc_by_id = {nc.get('cluster_id'): nc for nc in named_clusters}

    def plot_peeling_summary(self, output_dir, label):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        fig.patch.set_facecolor(DARK_BG)

        ax = axes[0]
        _style(ax, f'Peeling — {self.n_clusters} clusters')
        noise = self.labels == -1
        ax.scatter(self.M_2d[noise, 0], self.M_2d[noise, 1],
                   c='#2d3142', s=5, alpha=0.4, label=f'Résidu ({noise.sum()})', zorder=1)
        for cid in range(self.n_clusters):
            m = self.labels == cid
            if not m.any(): continue
            nc = self.nc_by_id.get(cid, {})
            ax.scatter(self.M_2d[m, 0], self.M_2d[m, 1], c=[self.cmap(cid)], s=15, alpha=0.85,
                       label=f'C{cid} n={m.sum()} L{nc.get("level", 0)}', zorder=3)
            cx, cy = float(np.median(self.M_2d[m, 0])), float(np.median(self.M_2d[m, 1]))
            name = nc.get('name', f'C{cid}')[:22]
            ax.annotate(name, (cx, cy), fontsize=5.5, color=TEXT_COLOR, ha='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=DARK_BG, alpha=0.5, edgecolor='none'))
        ax.legend(framealpha=0.4, facecolor=DARK_BG, labelcolor=TEXT_COLOR, fontsize=6, loc='upper right')

        ax2 = axes[1]
        _style(ax2, "Niveaux d'extraction")
        ax2.scatter(self.M_2d[noise, 0], self.M_2d[noise, 1], c='#2d3142', s=5, alpha=0.3, zorder=1)
        by_level = {}
        for ce in self.extracted:
            by_level.setdefault(ce['level'], []).append(ce)
        for lv, ces in sorted(by_level.items()):
            gidx = np.concatenate([np.array(ce['global_indices']) for ce in ces])
            ax2.scatter(self.M_2d[gidx, 0], self.M_2d[gidx, 1],
                        c=LEVEL_COLORS[lv % len(LEVEL_COLORS)], s=14, alpha=0.8,
                        label=f'Niveau {lv} ({len(gidx)} pts)', zorder=3)
        ax2.legend(framealpha=0.4, facecolor=DARK_BG, labelcolor=TEXT_COLOR, fontsize=8)

        n_unres = int((self.labels == -1).sum())
        fig.suptitle(f'{label} — {self.n_clusters} clusters | résidu {n_unres} '
                     f'({100 * n_unres / max(len(self.labels), 1):.0f}%)',
                     color=TEXT_COLOR, fontsize=11, fontweight='bold')
        _save(fig, str(out / f'{label}_peeling_summary.png'))

    def plot_signature_heatmap(self, output_dir, label):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ncs = [nc for nc in self.named_clusters if nc.get('cluster_id', -1) >= 0]
        if not ncs: return
        slot_order = ncs[0].get('slot_order', [])
        if not slot_order: return

        data = np.zeros((len(ncs), len(slot_order)))
        masks = np.zeros_like(data, dtype=bool)
        for i, nc in enumerate(ncs):
            for j, val in enumerate(nc.get('signature_vector', [])[:len(slot_order)]):
                if val is None: masks[i, j] = True
                else: data[i, j] = float(val)

        fig, ax = plt.subplots(figsize=(max(8, len(slot_order) * 1.3), max(5, len(ncs) * 0.6 + 1.5)))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(PANEL_BG)
        ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
        for i in range(len(ncs)):
            for j in range(len(slot_order)):
                txt = '?' if masks[i, j] else f'{data[i, j]:.2f}'
                col = DIM_COLOR if masks[i, j] else ('black' if data[i, j] > 0.45 else 'white')
                ax.text(j, i, txt, ha='center', va='center', fontsize=7.5, color=col, fontweight='bold')
        ax.set_xticks(range(len(slot_order)))
        ax.set_xticklabels(slot_order, color=TEXT_COLOR, fontsize=9, rotation=20)
        ax.set_yticks(range(len(ncs)))
        ax.set_yticklabels([f'C{nc.get("cluster_id","?")} ({nc.get("n",0)}) {nc.get("name","?")[:18]}' for nc in ncs],
                           color=TEXT_COLOR, fontsize=7.5)
        ax.set_title(f'Signature heatmap — {label}', color=TEXT_COLOR, fontsize=11, fontweight='bold', pad=10)
        _save(fig, str(out / f'{label}_signature_heatmap.png'), dpi=130)

    def plot_layer(self, layer_name, output_dir, label):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 9))
        fig.patch.set_facecolor(DARK_BG)
        _style(ax, f'Layer : {layer_name} — {self.n_clusters} clusters')
        noise = self.labels == -1
        ax.scatter(self.M_2d[noise, 0], self.M_2d[noise, 1], c='#2d3142', s=5, alpha=0.3, zorder=1)
        for cid in range(self.n_clusters):
            m = self.labels == cid
            if not m.any(): continue
            nc = self.nc_by_id.get(cid, {})
            ax.scatter(self.M_2d[m, 0], self.M_2d[m, 1], c=[self.cmap(cid)], s=16, alpha=0.85, zorder=3)
            cx, cy = float(np.median(self.M_2d[m, 0])), float(np.median(self.M_2d[m, 1]))
            ax.annotate(f'{nc.get("name", f"C{cid}")}\n(n={m.sum()}, h={nc.get("cluster_homogeneity", 0):.2f})',
                        (cx, cy), fontsize=6, color=TEXT_COLOR, ha='center', alpha=0.9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=DARK_BG, alpha=0.6,
                                  edgecolor=self.cmap(cid), linewidth=0.8))
        _save(fig, str(out / f'{label}_{layer_name}.png'))

    def plot_all(self, output_dir, label):
        self.plot_peeling_summary(output_dir, label)
        self.plot_signature_heatmap(output_dir, label)
        self.plot_layer('universal', output_dir, label)
