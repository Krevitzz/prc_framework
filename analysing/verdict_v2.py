"""
analysing/verdict_v2.py

Responsabilité : Génération sorties verdict uniquement.

write_verdict_report()     → JSON
write_verdict_report_txt() → TXT lisible humain

Zéro logique métier — reçoit un dict résultat, écrit des fichiers.
Tout calcul se fait en amont dans pipeline.py.
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np


# =============================================================================
# JSON
# =============================================================================

def write_verdict_report(verdict_results: Dict, output_path: Path) -> None:
    """
    Rapport JSON — sérialise les types numpy et ndarray.

    Args:
        verdict_results : dict depuis run_analysing_pipeline()
        output_path     : Path vers fichier .json de sortie
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _serial(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating):return float(obj)
        if isinstance(obj, np.bool_):   return bool(obj)
        raise TypeError(f"Non-sérialisable : {type(obj)}")

    # M_2d est un ndarray → exclure du JSON (trop volumineux, pas utile)
    result_serializable = {
        k: v for k, v in verdict_results.items() if k != 'M_2d'
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_serializable, f, indent=2,
                  ensure_ascii=False, default=_serial)

    print(f"✓ JSON : {output_path}")


# =============================================================================
# TXT
# =============================================================================

def write_verdict_report_txt(verdict_results: Dict, output_path: Path) -> None:
    """
    Rapport TXT lisible humain.

    Sections :
      - Header (observations, statuts)
      - Outliers (récurrence atomics)
      - Clusters nommés (slots, signature)
      - Insights

    Args:
        verdict_results : dict depuis run_analysing_pipeline()
        output_path     : Path vers fichier .txt de sortie
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines    = []
    metadata = verdict_results.get('metadata', {})
    phase    = metadata.get('label', output_path.stem).upper()

    # ── Header ───────────────────────────────────────────────────────────────
    lines.append(f"=== VERDICT {phase} ===")
    n_obs      = metadata.get('n_observations', 0)
    n_gammas   = metadata.get('n_gammas', 0)
    n_encs     = metadata.get('n_encodings', 0)
    n_clusters = metadata.get('n_clusters', 0)
    n_unres    = metadata.get('n_unresolved', 0)
    n_feat_o   = metadata.get('n_features_ortho', 0)
    n_feat_t   = metadata.get('n_features_total', 0)

    lines.append(f"Observations : {n_obs}")
    lines.append(f"Gammas : {n_gammas}  |  Encodings : {n_encs}")
    lines.append(f"Features : {n_feat_o}/{n_feat_t} orthogonales")
    lines.append("")

    # ── Outliers ─────────────────────────────────────────────────────────────
    outliers     = verdict_results.get('outliers', {})
    n_out        = outliers.get('n_outliers', 0)
    out_frac     = outliers.get('outlier_fraction', 0.0)
    lines.append(f"OUTLIERS ({n_out} runs, {out_frac*100:.1f}%)")

    if n_out > 0:
        gamma_rec = outliers.get('recurrence', {}).get('gamma', {})
        if gamma_rec:
            lines.append("  Récurrence gammas :")
            for gid, rec in list(gamma_rec.items())[:5]:
                lines.append(f"    {gid} : {rec['count']}/{rec['total_subset']} "
                             f"({rec['fraction']*100:.0f}%)")
    else:
        lines.append("  (aucun outlier détecté)")
    lines.append("")

    # ── Clusters nommés ──────────────────────────────────────────────────────
    named = verdict_results.get('named_clusters', [])
    lines.append(f"CLUSTERS ({n_clusters} extraits, {n_unres} résidu)")
    lines.append("")

    for nc in sorted(named, key=lambda x: x.get('cluster_id', 999)):
        cid  = nc.get('cluster_id', '?')
        name = nc.get('name', '?')
        n    = nc.get('n', 0)
        homo = nc.get('cluster_homogeneity', 0.0)
        het  = ' ⚠ hétérogène' if nc.get('heterogeneous') else ''
        lv   = nc.get('level', '?')

        if cid == -1:
            lines.append(f"  RÉSIDU ({n} runs — non résolu)")
            lines.append("")
            continue

        lines.append(f"  Cluster {cid} ({n} runs, niveau {lv}) "
                     f"homo={homo:.2f}{het}")
        lines.append(f"    Nom : {name}")

        name_full = nc.get('name_full', '')
        if name_full and name_full != name:
            lines.append(f"    Complet : {name_full}")

        for s in nc.get('slots', []):
            conf_s = f'{s["conf"]:.2f}' if s.get('conf') is not None else '—'
            slot   = s.get('slot', '')
            term   = s.get('term', '?')
            lines.append(f"    [{slot:18s}] {term:28s} conf={conf_s}")

        sec = nc.get('slots_secondary', [])
        if sec:
            sec_str = ', '.join(f'({s["term"]})' for s in sec)
            lines.append(f"    Secondaires : {sec_str}")

        uncalib = [s['slot'] for s in nc.get('slots_uncalibrated', [])]
        if uncalib:
            lines.append(f"    Non calibrés : {', '.join(uncalib)}")

        # Signature vector
        sig      = nc.get('signature_vector', [])
        slot_ord = nc.get('slot_order', [])
        if sig and slot_ord:
            sig_parts = [
                f"{o}={f'{v:.2f}' if v is not None else '?'}"
                for o, v in zip(slot_ord, sig)
            ]
            lines.append(f"    Signature : {'  '.join(sig_parts)}")

        lines.append("")

    # ── Insights ─────────────────────────────────────────────────────────────
    lines.append("INSIGHTS")
    pure = [
        nc for nc in named
        if nc.get('cluster_id', -1) >= 0 and not nc.get('heterogeneous')
    ]
    if pure:
        best = max(pure, key=lambda x: x.get('cluster_homogeneity', 0))
        lines.append(f"  - Cluster le plus homogène : C{best['cluster_id']} "
                     f"'{best['name']}' "
                     f"(homo={best['cluster_homogeneity']:.2f}, n={best['n']})")

    if n_unres > 0 and n_obs > 0:
        pct = 100 * n_unres / n_obs
        lines.append(f"  - Résidu : {n_unres} runs ({pct:.0f}%) "
                     f"→ cible phases suivantes")

    if not any(l.startswith('  -') for l in lines[-5:]):
        lines.append("  (analyse préliminaire)")
    lines.append("")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"✓ TXT : {output_path}")
