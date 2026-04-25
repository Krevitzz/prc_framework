"""
Sorties fichiers : verdict JSON/TXT + visualisations PNG.

Zéro calcul. Reçoit des données, sérialise, écrit.
Le TXT répond aux 4 questions computationnelles du charter (§1.3).

@ROLE    Sorties : sérialisation verdict + génération PNG + interprétation scientifique
@LAYER   analysing
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _serial(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_): return bool(obj)
    if hasattr(obj, '__dataclass_fields__'):
        return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
    raise TypeError(f"Non-sérialisable : {type(obj)}")

def _write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_serial)
    print(f"  ✓ JSON : {path}")

def write_strate_report_json(strate_result, output_path):
    clean = {k: v for k, v in strate_result.items()
             if k not in ('M_2d', 'M_ortho', 'labels_array')}
    _write_json(clean, Path(output_path))


# =========================================================================
# INTERPRÉTATION SCIENTIFIQUE
# =========================================================================

def _interpret_cluster(nc):
    lines = []
    slots = {s.get('slot'): s for s in nc.get('slots', [])}
    comp = nc.get('composition', {})
    dom_regime = comp.get('dominant_regime', '')
    trunc_frac = comp.get('truncated_fraction', 0)

    # Q1
    rnk, ent = slots.get('RNK'), slots.get('ENT')
    if rnk or ent:
        q1 = []
        if rnk:
            t = rnk.get('term', '')
            if 'RNK+' in t: q1.append("rang effectif croissant (D gagne en structure)")
            elif 'RNK-' in t: q1.append("rang effectif décroissant (D perd de la structure)")
        if ent:
            t = ent.get('term', '')
            if 'ENT+' in t: q1.append("entropie croissante (désordre)")
            elif 'ENT-' in t: q1.append("entropie décroissante (organisation émergente)")
        if q1: lines.append(f"    Q1 (information) : {', '.join(q1)}")

    # Q2
    lya, dmd, amp = slots.get('LYA'), slots.get('DMD'), slots.get('AMP')
    if lya or dmd or amp:
        q2 = []
        if lya:
            t = lya.get('term', '')
            if 'LYA+' in t: q2.append("Lyapunov+ → Γ amplifie")
            elif 'LYA-' in t: q2.append("Lyapunov- → Γ contracte")
            elif 'LYA~' in t: q2.append("Lyapunov oscillant → alternance")
        if dmd:
            t = dmd.get('term', '')
            if 'DMD>>' in t: q2.append("modes DMD amplifiés → instable")
            elif 'DMDosc' in t: q2.append("modes DMD oscillatoires")
            elif 'DMD<1' in t: q2.append("modes DMD amortis → convergent")
        if amp:
            t = amp.get('term', '')
            if 'AMP<<' in t: q2.append("norme collapse → Γ dissipe D")
            elif 'AMP>>>' in t or 'AMP>>' in t: q2.append("norme explosion → Γ amplifie sans borne")
            elif 'AMP~0' in t: q2.append("norme conservée → Γ préserve ||D||")
        if q2: lines.append(f"    Q2 (action Γ) : {', '.join(q2)}")

    # Q3
    q3 = []
    regime_map = {'FLAT': 'attracteur candidat', 'OSCILLATING': 'cycle limite',
                  'TRANSITIONAL': 'transition entre régimes', 'EXPLOSIVE': 'divergence',
                  'MIXED': 'régime mixte'}
    if dom_regime in regime_map:
        q3.append(regime_map[dom_regime])
    if trunc_frac > 0.5: q3.append(f"{trunc_frac:.0%} tronqués → instabilité fréquente")
    elif trunc_frac > 0: q3.append(f"{trunc_frac:.0%} tronqués")
    if q3: lines.append(f"    Q3 (émergence) : {', '.join(q3)}")

    # Q4
    cnd = slots.get('CND')
    if cnd:
        lines.append("    Q4 (pré-émergence) : pathologie numérique → non évaluable")
    else:
        has_ent_neg = ent and 'ENT-' in ent.get('term', '')
        has_dmd_stable = dmd and 'DMD<1' in dmd.get('term', '')
        if has_ent_neg and has_dmd_stable:
            lines.append("    Q4 (pré-émergence) : ENT- + DMD stable → attracteur holographique candidat")
    return lines


def _entity_verdict(profile):
    exp = profile.get('explosion_rate', 0)
    conc = profile.get('concentration', 0)
    n_cl = profile.get('n_clustered', 0)
    if exp > 0.8: return 'EXCLURE — >80% pathologique'
    elif exp > 0.3: return 'EXPLORER — taux pathologique élevé'
    elif conc > 0.9 and n_cl >= 3: return 'CONSERVER — stable et concentré'
    elif conc > 0.7: return 'CONSERVER — comportement cohérent'
    else: return 'EXPLORER — comportement dispersé'


# =========================================================================
# RAPPORT STRATE TXT
# =========================================================================

def write_strate_report_txt(strate_result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    meta = strate_result.get('metadata', {})
    strate_id = meta.get('strate_id', '?')
    clustering = strate_result.get('clustering', {})

    lines.append(f"{'='*70}")
    lines.append(f"RAPPORT STRATE : {strate_id.upper()}")
    lines.append(f"{'='*70}")
    lines.append("")
    lines.append(f"Runs : {meta.get('n_runs', 0)}")
    lines.append(f"Features : {meta.get('n_features_applicable', 0)} applicables, "
                 f"{meta.get('n_features_input', 0)} ML → "
                 f"{meta.get('n_features_ortho', 0)} après ortho")
    lines.append(f"NaN runtime imputés : {meta.get('n_nan_imputed', 0)}")
    excluded = meta.get('features_excluded', {})
    if excluded:
        lines.append(f"Features exclues (structurel) : {len(excluded)}")
    lines.append("")

    # Clustering
    n_cl = clustering.get('n_clusters', 0)
    n_noise = clustering.get('n_noise', 0)
    n_samples = clustering.get('n_samples', 0)
    lines.append(f"CLUSTERING : {n_cl} clusters, {n_noise} résidu ({100*n_noise/max(n_samples,1):.0f}%)")
    comp_if = clustering.get('comparison', {})
    if comp_if:
        lines.append(f"IsolationForest : {comp_if.get('n_outliers',0)} outliers, "
                     f"Jaccard résidu/IF={comp_if.get('jaccard_overlap',0):.2f}")
    lines.append("")

    # Clusters
    named = strate_result.get('named_clusters', [])
    lines.append("--- RÉGIMES ---")
    lines.append("")

    for nc in sorted(named, key=lambda x: x.get('cluster_id', 999)):
        cid = nc.get('cluster_id', '?')
        if cid == -1:
            n_res = nc.get('n', 0)
            comp = nc.get('composition', {})
            rd = comp.get('regime_distribution', {})
            lines.append(f"  RÉSIDU ({n_res} runs)")
            if rd:
                lines.append(f"    P1 : {', '.join(f'{r}={v:.0%}' for r,v in sorted(rd.items(), key=lambda x:-x[1]))}")
            lines.append("")
            continue

        name = nc.get('name', '?')
        homo = nc.get('cluster_homogeneity', 0)
        het = ' ⚠ HÉTÉROGÈNE' if nc.get('heterogeneous') else ''
        n = nc.get('n', 0)

        lines.append(f"  C{cid} — {name}{het}")
        lines.append(f"    {n} runs, niveau {nc.get('level','?')}, homogénéité {homo:.2f}")

        for s in nc.get('slots', []):
            conf = f'{s["conf"]:.2f}' if s.get('conf') is not None else '—'
            feat = s.get('feature', '')
            val = s.get('value')
            val_s = f'{val:.4f}' if val is not None else '—'
            lines.append(f"    {s.get('slot',''):>5} {s.get('term','?'):12s} "
                         f"conf={conf}  ({feat}={val_s})")

        comp = nc.get('composition', {})
        rd = comp.get('regime_distribution', {})
        if rd:
            lines.append(f"    P1 : {', '.join(f'{r}={v:.0%}' for r,v in sorted(rd.items(), key=lambda x:-x[1]))}")
        trunc = comp.get('truncated_fraction', 0)
        if trunc > 0:
            lines.append(f"    Tronqués : {trunc:.0%}")

        interp = _interpret_cluster(nc)
        for line in interp:
            lines.append(line)
        lines.append("")

    # Cohérence
    coherence = strate_result.get('coherence', [])
    if coherence:
        lines.append("--- COHÉRENCE P1 ↔ CLUSTERS ---")
        for coh in coherence:
            if hasattr(coh, 'cluster_id'):
                tag = 'RÉSIDU' if coh.cluster_id == -1 else f'C{coh.cluster_id}'
                purity = 'PUR' if coh.is_pure else ('MIXTE' if coh.is_mixed else 'INTER')
                lines.append(f"  {tag:>8} ({coh.n_runs:>3}) "
                             f"H={coh.regime_entropy:.2f} {purity} "
                             f"dom={coh.dominant_regime} ({coh.dominant_fraction:.0%})")
        lines.append("")

    # Verdicts entités
    entity_profiles = strate_result.get('entity_profiles', {})
    for etype in ('gamma', 'encoding'):
        profiles = entity_profiles.get(etype, [])
        if not profiles: continue
        lines.append(f"--- VERDICTS {etype.upper()} ---")
        for p in sorted(profiles, key=lambda x: -x.get('concentration', 0)):
            eid = p.get('entity_id', '?')
            dom = p.get('dominant_cluster')
            conc = p.get('concentration', 0)
            exp_rate = p.get('explosion_rate', 0)
            dom_name = '?'
            if dom is not None:
                for nc in named:
                    if nc.get('cluster_id') == dom:
                        dom_name = nc.get('name', '?'); break
            verdict = _entity_verdict(p)
            lines.append(f"  {eid}")
            lines.append(f"    concentration={conc:.0%} dans C{dom} ({dom_name})"
                         + (f", pathologique={exp_rate:.0%}" if exp_rate > 0 else ""))
            lines.append(f"    → {verdict}")
        lines.append("")

    # Universalité
    for key, title in [('universal_gammas', 'UNIVERSALITÉ Γ'),
                       ('convergent_encodings', 'CONVERGENCE ENCODINGS')]:
        items = strate_result.get(key, [])
        if items:
            lines.append(f"--- {title} ---")
            for u in items[:10]:
                eid = u.get('gamma_id', u.get('encoding_id', '?'))
                lines.append(f"  {eid} → {u['cluster_name']} ({u['concentration']:.0%})")
            lines.append("")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  ✓ TXT : {output_path}")


# =========================================================================
# SYNTHÈSE JSON
# =========================================================================

def write_synthesis_json(all_strate_results, pathological_stats, output_path):
    synthesis = {
        'strates': {k: {kk: vv for kk, vv in v.items()
                         if kk not in ('M_2d', 'M_ortho', 'labels_array')}
                    for k, v in all_strate_results.items()},
        'pathological': pathological_stats,
        'summary': {
            'n_strates': len(all_strate_results),
            'total_runs_clustered': sum(v.get('metadata',{}).get('n_runs',0) for v in all_strate_results.values()),
            'total_clusters': sum(v.get('clustering',{}).get('n_clusters',0) for v in all_strate_results.values()),
        },
    }
    _write_json(synthesis, Path(output_path))


# =========================================================================
# SYNTHÈSE TXT
# =========================================================================

def write_synthesis_txt(all_strate_results, pathological_stats,
                         output_path, label=''):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    title = f"SYNTHÈSE — {label}" if label else "SYNTHÈSE ANALYSING"
    lines.append(f"{'='*70}")
    lines.append(title)
    lines.append(f"{'='*70}")
    lines.append("")

    total_runs = sum(r.get('metadata',{}).get('n_runs',0) for r in all_strate_results.values())
    total_cl = sum(r.get('clustering',{}).get('n_clusters',0) for r in all_strate_results.values())
    total_noise = sum(r.get('clustering',{}).get('n_noise',0) for r in all_strate_results.values())
    total_patho = sum(s.get('n',0) for s in pathological_stats.values())

    lines.append(f"Population : {total_runs} runs analysés + {total_patho} pathologiques")
    lines.append(f"Résultat : {len(all_strate_results)} strate(s), "
                 f"{total_cl} clusters, {total_noise} résidu "
                 f"({100*total_noise/max(total_runs,1):.0f}%)")
    lines.append("")

    for sid, result in sorted(all_strate_results.items()):
        meta = result.get('metadata', {})
        cl = result.get('clustering', {})
        if meta.get('skipped'): continue
        lines.append(f"  {sid:20s} : {meta.get('n_runs',0):>5} runs, "
                     f"{cl.get('n_clusters',0)} clusters")
        for nc in sorted(result.get('named_clusters',[]), key=lambda x: x.get('cluster_id',999)):
            cid = nc.get('cluster_id', -1)
            if cid == -1: continue
            comp = nc.get('composition', {})
            lines.append(f"    C{cid} ({nc.get('n',0):>3}, homo={nc.get('cluster_homogeneity',0):.2f}) "
                         f"P1={comp.get('dominant_regime','?'):12s} {nc.get('name','?')}")
    lines.append("")

    # Pathologiques
    if total_patho > 0:
        lines.append("--- PATHOLOGIQUES ---")
        for status in ('EXPLOSION', 'COLLAPSED'):
            stats = pathological_stats.get(status, {})
            n = stats.get('n', 0)
            if n == 0: continue
            lines.append(f"  {status} : {n} runs")
            for g in stats.get('top_gammas', [])[:5]:
                lines.append(f"    {g['gamma_id']} : {g['count']}/{g['total']} ({g['fraction']:.0%})")
        lines.append("")

    # Questions scientifiques
    lines.append(f"{'='*70}")
    lines.append("QUESTIONS COMPUTATIONNELLES (charter §1.3)")
    lines.append(f"{'='*70}")
    lines.append("")

    all_named = []
    for r in all_strate_results.values():
        for nc in r.get('named_clusters', []):
            if nc.get('cluster_id', -1) >= 0:
                all_named.append(nc)

    # Q1
    lines.append("Q1 — D encode-t-il de l'information structurée ?")
    for nc in all_named:
        slots = {s.get('slot'): s for s in nc.get('slots', [])}
        ent, rnk = slots.get('ENT'), slots.get('RNK')
        if ent or rnk:
            parts = [s['term'] for s in [ent, rnk] if s]
            lines.append(f"  C{nc['cluster_id']} ({nc['n']} runs) {nc['name']} : {', '.join(parts)}")
    lines.append("")

    # Q2
    lines.append("Q2 — Γ préserve / amplifie / dissipe l'information ?")
    for nc in all_named:
        slots = {s.get('slot'): s for s in nc.get('slots', [])}
        hits = [slots.get(k) for k in ('AMP', 'LYA', 'DMD') if slots.get(k)]
        if hits:
            parts = [s['term'] for s in hits]
            lines.append(f"  C{nc['cluster_id']} ({nc['n']} runs) {nc['name']} : {', '.join(parts)}")
    lines.append("")

    # Q3
    lines.append("Q3 — Structures émergentes stables ?")
    for nc in all_named:
        comp = nc.get('composition', {})
        dom = comp.get('dominant_regime', '')
        homo = nc.get('cluster_homogeneity', 0)
        label_q3 = {'FLAT': 'attracteur candidat', 'OSCILLATING': 'cycle limite',
                     'TRANSITIONAL': 'transitoire', 'EXPLOSIVE': 'divergent',
                     'MIXED': 'mixte'}.get(dom, dom)
        lines.append(f"  C{nc['cluster_id']} ({nc['n']} runs) {nc['name']} — {label_q3} (homo={homo:.2f})")
    lines.append("")

    # Q4
    lines.append("Q4 — Propriétés pré-émergentes ?")
    holo = [nc for nc in all_named
            if any('ENT-' in s.get('term','') for s in nc.get('slots',[]))
            and any('DMD<1' in s.get('term','') for s in nc.get('slots',[]))]
    if holo:
        lines.append("  Attracteurs holographiques candidats (ENT- + DMD stable) :")
        for nc in holo:
            lines.append(f"    C{nc['cluster_id']} ({nc['n']} runs) {nc['name']}")
    else:
        lines.append("  Aucun attracteur holographique identifié.")

    for r in all_strate_results.values():
        for key, title in [('universal_gammas', 'Gammas universels'),
                           ('convergent_encodings', 'Encodings convergents')]:
            items = r.get(key, [])
            if items:
                lines.append(f"  {title} :")
                for u in items[:5]:
                    eid = u.get('gamma_id', u.get('encoding_id', '?'))
                    lines.append(f"    {eid} → {u['cluster_name']} ({u['concentration']:.0%})")
    lines.append("")

    # A3
    lines.append("A3 — D résiste-t-il sous Γ ?")
    a3 = [nc for nc in all_named if any('AMP~0' in s.get('term','') for s in nc.get('slots',[]))]
    if a3:
        for nc in a3:
            lines.append(f"  C{nc['cluster_id']} ({nc['n']} runs) {nc['name']} — norme conservée → A3 compatible")
    else:
        lines.append("  Pas de signal clair.")
    lines.append("")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  ✓ TXT : {output_path}")


# =========================================================================
# VISUALIZER
# =========================================================================

DARK_BG, PANEL_BG, GRID_COLOR = '#0d1117', '#161b22', '#21262d'
TEXT_COLOR, DIM_COLOR = '#c9d1d9', '#444455'
LEVEL_COLORS = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']

def _style(ax, title=''):
    ax.set_facecolor(PANEL_BG)
    if title: ax.set_title(title, color=TEXT_COLOR, fontsize=10, fontweight='bold', pad=7)
    ax.tick_params(colors=DIM_COLOR, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.5)

def _save(fig, path, dpi=140):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f"  → {path}")

def _cluster_colormap(n):
    return matplotlib.colormaps['tab20'].resampled(max(n, 1))


class ClusterVisualizer:
    def __init__(self, M_2d, named_clusters, peeling_result, coherence=None):
        self.M_2d = M_2d
        self.named_clusters = named_clusters
        self.peeling_result = peeling_result
        self.labels = peeling_result.get('labels', np.full(len(M_2d), -1))
        self.extracted = peeling_result.get('extracted', [])
        self.n_clusters = peeling_result.get('n_clusters', 0)
        self.cmap = _cluster_colormap(self.n_clusters)
        self.nc_by_id = {nc.get('cluster_id'): nc for nc in named_clusters}

    def plot_peeling_summary(self, output_dir, label):
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        fig.patch.set_facecolor(DARK_BG)
        ax = axes[0]; _style(ax, f'Peeling — {self.n_clusters} clusters')
        noise = self.labels == -1
        valid = np.all(np.isfinite(self.M_2d), axis=1)
        ax.scatter(self.M_2d[noise&valid,0], self.M_2d[noise&valid,1],
                   c='#2d3142', s=5, alpha=0.4, label=f'Résidu ({noise.sum()})', zorder=1)
        for cid in range(self.n_clusters):
            m = (self.labels == cid) & valid
            if not m.any(): continue
            nc = self.nc_by_id.get(cid, {})
            ax.scatter(self.M_2d[m,0], self.M_2d[m,1], c=[self.cmap(cid)], s=15, alpha=0.85,
                       label=f'C{cid} n={m.sum()}', zorder=3)
            cx, cy = float(np.median(self.M_2d[m,0])), float(np.median(self.M_2d[m,1]))
            ax.annotate(nc.get('name',f'C{cid}')[:22], (cx,cy), fontsize=5.5, color=TEXT_COLOR,
                        ha='center', bbox=dict(boxstyle='round,pad=0.2', facecolor=DARK_BG, alpha=0.5, edgecolor='none'))
        ax.legend(framealpha=0.4, facecolor=DARK_BG, labelcolor=TEXT_COLOR, fontsize=6, loc='upper right')
        ax2 = axes[1]; _style(ax2, "Niveaux d'extraction")
        ax2.scatter(self.M_2d[noise&valid,0], self.M_2d[noise&valid,1], c='#2d3142', s=5, alpha=0.3, zorder=1)
        by_level = {}
        for ce in self.extracted: by_level.setdefault(ce['level'], []).append(ce)
        for lv, ces in sorted(by_level.items()):
            gidx = np.concatenate([np.array(ce['global_indices']) for ce in ces])
            v = valid[gidx]; gv = gidx[v]
            ax2.scatter(self.M_2d[gv,0], self.M_2d[gv,1], c=LEVEL_COLORS[lv%len(LEVEL_COLORS)],
                        s=14, alpha=0.8, label=f'Niveau {lv} ({len(gidx)} pts)', zorder=3)
        ax2.legend(framealpha=0.4, facecolor=DARK_BG, labelcolor=TEXT_COLOR, fontsize=8)
        n_u = int((self.labels==-1).sum())
        fig.suptitle(f'{label} — {self.n_clusters} clusters | résidu {n_u} ({100*n_u/max(len(self.labels),1):.0f}%)',
                     color=TEXT_COLOR, fontsize=11, fontweight='bold')
        _save(fig, str(out / f'{label}_peeling_summary.png'))

    def plot_signature_heatmap(self, output_dir, label):
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        ncs = [nc for nc in self.named_clusters if nc.get('cluster_id',-1)>=0]
        if not ncs: return
        slot_order = ncs[0].get('slot_order', [])
        if not slot_order: return
        data = np.zeros((len(ncs), len(slot_order)))
        masks = np.zeros_like(data, dtype=bool)
        for i, nc in enumerate(ncs):
            for j, val in enumerate(nc.get('signature_vector',[])[:len(slot_order)]):
                if val is None: masks[i,j]=True
                else: data[i,j]=float(val)
        fig, ax = plt.subplots(figsize=(max(8,len(slot_order)*1.3), max(5,len(ncs)*0.6+1.5)))
        fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(PANEL_BG)
        ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
        for i in range(len(ncs)):
            for j in range(len(slot_order)):
                txt = '?' if masks[i,j] else f'{data[i,j]:.2f}'
                col = DIM_COLOR if masks[i,j] else ('black' if data[i,j]>0.45 else 'white')
                ax.text(j,i,txt, ha='center', va='center', fontsize=7.5, color=col, fontweight='bold')
        ax.set_xticks(range(len(slot_order))); ax.set_xticklabels(slot_order, color=TEXT_COLOR, fontsize=9, rotation=20)
        ax.set_yticks(range(len(ncs)))
        ax.set_yticklabels([f'C{nc.get("cluster_id","?")} ({nc.get("n",0)}) {nc.get("name","?")[:18]}' for nc in ncs],
                           color=TEXT_COLOR, fontsize=7.5)
        ax.set_title(f'Signature — {label}', color=TEXT_COLOR, fontsize=11, fontweight='bold', pad=10)
        _save(fig, str(out / f'{label}_signature_heatmap.png'), dpi=130)

    def plot_all(self, output_dir, label):
        self.plot_peeling_summary(output_dir, label)
        self.plot_signature_heatmap(output_dir, label)
