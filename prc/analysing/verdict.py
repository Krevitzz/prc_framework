"""
prc.analysing.verdict

Responsabilité : Analyses post-batch → rapport

Pipeline :
    profiling (aggregation cross-runs)
    + analysing (peeling → namer → visualizer)
    + outliers (IsolationForest)
    + régimes (DEPRECATED — en transition)
    → rapport JSON + TXT

Changements v7.1 :
    - _cross_cluster_regimes() supprimé — remplacé par named_clusters
    - Sortie verdict gagne 'named_clusters' via analysing_results
    - Régimes conservés en lecture seule pendant transition
    - Rapport TXT affiche noms composés + signature slots
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.data_loading_lite import load_yaml
from analysing.profiling_lite import run_profiling
from analysing.hub_analysing import run_analysing
from analysing.concordance_lite import run_concordance_cross_phases
from analysing.outliers_lite import analyze_outliers
from analysing.regimes_lite import (          # DEPRECATED — transition
    load_regime_thresholds,
    classify_regimes_batch,
    refine_conserves_norm_with_cv,
)


# =============================================================================
# POOL REQUIREMENTS
# =============================================================================

def load_pool_requirements() -> Dict:
    req_path = Path('configs/constraints/pool_requirements.yaml')
    if not req_path.exists():
        return {
            'n_dof'        : {'min': None, 'max': None},
            'max_iterations': {'min': None, 'max': None},
            'deprecated'   : {'gammas': [], 'encodings': [], 'modifiers': []},
        }
    return load_yaml(req_path)


def filter_rows_by_pool(
    rows        : List[Dict],
    requirements: Dict,
    verbose     : bool = True,
) -> Tuple[List[Dict], int, Dict[str, int]]:
    filtered     = []
    skip_reasons = {}

    for row in rows:
        comp   = row['composition']
        skip   = False
        reason = None

        if requirements['n_dof']['min'] is not None:
            if comp['n_dof'] < requirements['n_dof']['min']:
                skip, reason = True, 'n_dof_too_low'

        if not skip and requirements['n_dof']['max'] is not None:
            if comp['n_dof'] > requirements['n_dof']['max']:
                skip, reason = True, 'n_dof_too_high'

        if not skip and comp['gamma_id'] in requirements['deprecated']['gammas']:
            skip, reason = True, 'gamma_deprecated'

        if skip:
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        else:
            filtered.append(row)

    n_skipped = len(rows) - len(filtered)
    if verbose and n_skipped > 0:
        print(f"\n⚠️  Filtrage pool : {n_skipped}/{len(rows)} observations skippées")
        for reason, count in skip_reasons.items():
            print(f"  {reason}: {count}")

    return filtered, n_skipped, skip_reasons


# =============================================================================
# MODE 1 : VERDICT INTRA (RAM)
# =============================================================================

def run_verdict_intra(
    rows              : List[Dict],
    filter_pool       : bool = True,
    regime_profile    : str  = 'default',
    peeling_cfg_path  : Optional[str] = None,
    namer_cfg_path    : Optional[str] = None,
    tsne_cache_path   : Optional[str] = None,
    output_dir        : Optional[str] = None,
    label             : str  = 'verdict',
    plot              : bool = True,
    n_skipped_batch   : int  = 0,   # runs skippés par le runner (RunnerError/featuring)
) -> Dict:
    """
    Verdict complet sur rows en RAM.

    Args:
        rows             : Liste {composition, features, layers}
        filter_pool      : Appliquer filtrage pool requirements
        regime_profile   : Profil seuils régimes (DEPRECATED — transition)
        peeling_cfg_path : Config clustering peeling
        namer_cfg_path   : Config cluster namer
        tsne_cache_path  : Cache t-SNE .npy
        output_dir       : Répertoire sorties PNG + trace
        label            : Préfixe fichiers
        plot             : Générer PNG

    Returns:
        {
            'profiling'  : Dict,
            'analysing'  : Dict,   # inclut named_clusters
            'outliers'   : Dict,
            'regimes'    : Dict,   # DEPRECATED — transition
            'metadata'   : Dict,
        }
    """
    print(f"\n=== Verdict intra (RAM) ===")
    print(f"Observations initiales : {len(rows)}")

    n_skipped_pool = 0
    if filter_pool:
        requirements   = load_pool_requirements()
        rows, n_skipped_pool, _ = filter_rows_by_pool(rows, requirements)
        print(f"Observations après filtrage pool : {len(rows)}")

    if not rows:
        print("\n⚠️  Aucune observation valide après filtrage pool")
        return _empty_verdict(n_skipped_pool)

    # ── Profiling ─────────────────────────────────────────────────────────
    profiling_results = run_profiling(rows)

    # ── Outliers ──────────────────────────────────────────────────────────
    print("\n=== Détection outliers ===")
    outliers_results = analyze_outliers(rows, contamination=0.1)
    print(f"Outliers : {outliers_results['n_outliers']} "
          f"({outliers_results['outlier_fraction']*100:.1f}%)")

    # ── Régimes (DEPRECATED — transition) ─────────────────────────────────
    print("\n=== Classification régimes (transition) ===")
    thresholds      = load_regime_thresholds(regime_profile)
    regimes_results = classify_regimes_batch(
        rows, outliers_results['stable_indices'], thresholds
    )
    regimes_results = refine_conserves_norm_with_cv(rows, regimes_results, thresholds)
    print(f"Régimes : {len(regimes_results['regimes'])}")

    # Vecteur régimes aligné sur rows → passé au namer
    run_regimes = _build_run_regimes(rows, regimes_results)

    # ── Analysing — peeling + namer + visualizer ───────────────────────────
    print("\n=== Analysing ===")
    analysing_results = run_analysing(
        rows             = rows,
        stratified       = True,
        run_regimes      = run_regimes,
        peeling_cfg_path = peeling_cfg_path,
        namer_cfg_path   = namer_cfg_path,
        tsne_cache_path  = tsne_cache_path,
        output_dir       = output_dir,
        label            = label,
        plot             = plot,
    )

    # ── Metadata ──────────────────────────────────────────────────────────
    ana_meta = analysing_results.get('metadata', {})
    metadata = {
        'n_observations_initial' : len(rows) + n_skipped_pool + n_skipped_batch,
        'n_observations_filtered': len(rows),
        'n_skipped_pool'         : n_skipped_pool,
        'n_skipped_batch'        : n_skipped_batch,
        'n_gammas'               : len(profiling_results.get('gamma', {})),
        'n_encodings'            : len(profiling_results.get('encoding', {})),
        'n_modifiers'            : len(profiling_results.get('modifier', {})),
        'regime_profile'         : regime_profile,
        'n_clusters'             : ana_meta.get('n_clusters', 0),
        'n_unresolved'           : ana_meta.get('n_unresolved', 0),
        'n_named_clusters'       : len([
            nc for nc in analysing_results.get('named_clusters', [])
            if nc.get('cluster_id', -1) >= 0
        ]),
        'label': label,
    }

    print(f"\n✓ Verdict complet")
    print(f"  Profiling : {metadata['n_gammas']} gammas")
    print(f"  Clusters  : {metadata['n_clusters']} extraits, "
          f"{metadata['n_unresolved']} résidu")

    return {
        'profiling' : profiling_results,
        'analysing' : analysing_results,
        'outliers'  : outliers_results,
        'regimes'   : regimes_results,   # DEPRECATED
        'metadata'  : metadata,
    }


# =============================================================================
# MODE 2 : VERDICT SINGLE-PHASE (PARQUET)
# =============================================================================

def run_verdict_from_parquet(
    parquet_path     : Path,
    regime_profile   : str  = 'default',
    peeling_cfg_path : Optional[str] = None,
    namer_cfg_path   : Optional[str] = None,
    tsne_cache_path  : Optional[str] = None,
    output_dir       : Optional[str] = None,
    label            : Optional[str] = None,
    plot             : bool = True,
) -> Dict:
    """Verdict depuis Parquet."""
    print(f"\n=== Verdict depuis Parquet ===")
    print(f"Loading : {parquet_path}\n")

    df = pd.read_parquet(parquet_path)
    print(f"Observations : {len(df)} | Colonnes : {len(df.columns)}\n")

    axes_cols     = ['gamma_id', 'encoding_id', 'modifier_id', 'n_dof', 'max_iterations']
    features_cols = [c for c in df.columns if c not in axes_cols + ['phase']]

    rows = []
    for _, row_df in df.iterrows():
        comp     = {col: row_df[col] for col in axes_cols}
        features = {col: row_df[col] for col in features_cols}
        rows.append({'composition': comp, 'features': features, 'layers': ['timeline']})

    _label = label or Path(parquet_path).stem

    return run_verdict_intra(
        rows             = rows,
        regime_profile   = regime_profile,
        peeling_cfg_path = peeling_cfg_path,
        namer_cfg_path   = namer_cfg_path,
        tsne_cache_path  = tsne_cache_path,
        output_dir       = output_dir,
        label            = _label,
        plot             = plot,
    )


# =============================================================================
# MODE 3 : VERDICT CROSS-PHASES
# =============================================================================

def scan_major_phases(results_dir: Path = None) -> List[Path]:
    if results_dir is None:
        results_dir = Path('data/results')
    if not results_dir.exists():
        return []
    return sorted([
        p for p in results_dir.glob('*.parquet')
        if p.stem.startswith('r') and len(p.stem) == 2 and p.stem[1].isdigit()
    ])


def run_verdict_cross_phases(
    results_dir    : Path = None,
    regime_profile : str  = 'default',
) -> Dict:
    """Verdict inter-phases (concordance)."""
    print(f"\n=== Verdict cross-phases ===")

    parquet_paths = scan_major_phases(results_dir)
    if not parquet_paths:
        print("⚠️  Aucune phase principale trouvée")
        return {'phases': {}, 'concordance': {}, 'metadata': {'n_phases': 0}}

    print(f"Phases : {len(parquet_paths)}")
    phases_results = {}
    phases_data    = {}

    for parquet_path in parquet_paths:
        phase_name = parquet_path.stem
        print(f"\n--- {phase_name} ---")

        verdict = run_verdict_from_parquet(parquet_path, regime_profile,
                                           plot=False)
        phases_results[phase_name] = verdict

        df = pd.read_parquet(parquet_path)
        axes_cols     = ['gamma_id', 'encoding_id', 'modifier_id', 'n_dof', 'max_iterations']
        features_cols = [c for c in df.columns if c not in axes_cols + ['phase']]
        rows = []
        for _, row_df in df.iterrows():
            comp     = {col: row_df[col] for col in axes_cols}
            features = {col: row_df[col] for col in features_cols}
            rows.append({'composition': comp, 'features': features,
                         'layers': ['timeline']})
        phases_data[phase_name] = rows

    print(f"\n--- Concordance cross-phases ---")
    concordance_results = run_concordance_cross_phases(phases_data)
    print(f"✓ Kappa pairs : {len(concordance_results.get('kappa', {}))}")

    return {
        'phases'     : phases_results,
        'concordance': concordance_results,
        'metadata'   : {
            'n_phases'        : len(parquet_paths),
            'phases_analyzed' : [p.stem for p in parquet_paths],
            'regime_profile'  : regime_profile,
        },
    }


# =============================================================================
# HELPERS INTERNES
# =============================================================================

def _empty_verdict(n_skipped: int = 0) -> Dict:
    return {
        'profiling' : {},
        'analysing' : {},
        'outliers'  : {},
        'regimes'   : {},
        'metadata'  : {'n_filtered': 0, 'n_skipped_pool': n_skipped},
    }


def _build_run_regimes(rows: List[Dict], regimes_results: Dict) -> List[str]:
    """Vecteur régimes aligné sur rows — 'UNKNOWN' si non assigné."""
    run_regimes = ['UNKNOWN'] * len(rows)
    for regime, data in regimes_results.get('regimes', {}).items():
        for idx in data.get('run_indices', []):
            if idx < len(run_regimes):
                run_regimes[idx] = regime
    return run_regimes


# =============================================================================
# EXPORT RAPPORTS
# =============================================================================

def write_verdict_report(verdict_results: Dict, output_path: Path):
    """Rapport JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _serial(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return obj

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(verdict_results, f, indent=2,
                  ensure_ascii=False, default=_serial)
    print(f"\n✓ JSON : {output_path}")


def write_verdict_report_txt(verdict_results: Dict, output_path: Path):
    """Rapport TXT lisible humain."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines    = []
    metadata = verdict_results.get('metadata', {})
    phase    = output_path.stem.replace('verdict_', '').upper()

    lines.append(f"=== VERDICT {phase} ===")
    n_success  = metadata.get('n_observations_filtered', 0)
    n_skip_pool  = metadata.get('n_skipped_pool', 0)
    n_skip_batch = metadata.get('n_skipped_batch', 0)
    n_skip_total = n_skip_pool + n_skip_batch
    skip_detail = ''
    if n_skip_pool > 0 and n_skip_batch > 0:
        skip_detail = f' (pool={n_skip_pool}, runner={n_skip_batch})'
    elif n_skip_pool > 0:
        skip_detail = f' (pool)'
    elif n_skip_batch > 0:
        skip_detail = f' (runner)'
    lines.append(f"Runs : {n_success} success, {n_skip_total} skipped{skip_detail}")
    lines.append("")

    # Outliers
    outliers     = verdict_results.get('outliers', {})
    n_outliers   = outliers.get('n_outliers', 0)
    outlier_frac = outliers.get('outlier_fraction', 0.0)
    lines.append(f"OUTLIERS ({n_outliers} runs, {outlier_frac*100:.1f}%)")
    if n_outliers > 0:
        gamma_rec = outliers.get('recurrence', {}).get('gamma', {})
        if gamma_rec:
            lines.append("  Récurrence atomics :")
            for gid, data in list(gamma_rec.items())[:5]:
                lines.append(f"    {gid} : {data['count']}/{data['total_subset']} "
                             f"({data['fraction']*100:.0f}%)")
    else:
        lines.append("  (aucun outlier détecté)")
    lines.append("")

    # Clusters nommés
    analysing    = verdict_results.get('analysing', {})
    named        = analysing.get('named_clusters', [])
    ana_meta     = analysing.get('metadata', {})
    n_cl         = ana_meta.get('n_clusters', 0)
    n_unresolved = ana_meta.get('n_unresolved', 0)
    n_feat       = ana_meta.get('n_features', 0)
    n_feat_total = ana_meta.get('n_features_total', 0)

    lines.append(f"CLUSTERS ({n_cl} extraits, {n_unresolved} résidu)")
    lines.append(f"  Features orthogonales : {n_feat}/{n_feat_total}")
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
        if nc.get('name_full') != name:
            lines.append(f"    Complet : {nc['name_full']}")

        for s in nc.get('slots', []):
            conf_s = f'{s["conf"]:.2f}' if s.get('conf') is not None else '—'
            lines.append(f"    [{s['slot']:15s}] {s.get('term','?'):25s} "
                         f"conf={conf_s}")

        sec = nc.get('slots_secondary', [])
        if sec:
            sec_str = ', '.join(f'({s["term"]})' for s in sec)
            lines.append(f"    Secondaires : {sec_str}")

        uncalib = [s['slot'] for s in nc.get('slots_uncalibrated', [])]
        if uncalib:
            lines.append(f"    Non calibrés : {', '.join(uncalib)}")

        lines.append("")

    # Régimes (DEPRECATED — transition)
    regimes_data = verdict_results.get('regimes', {}).get('regimes', {})
    if regimes_data:
        lines.append("RÉGIMES (transition)")
        for regime, data in list(
            {k: v for k, v in regimes_data.items() if v['count'] > 0}.items()
        )[:5]:
            lines.append(f"  {regime} : {data['count']} runs "
                         f"({data['fraction']*100:.0f}%)")
        lines.append("")

    # Insights
    lines.append("INSIGHTS")
    pure = [nc for nc in named
            if nc.get('cluster_id', -1) >= 0 and not nc.get('heterogeneous')]
    if pure:
        best = max(pure, key=lambda x: x.get('cluster_homogeneity', 0))
        lines.append(f"  - Cluster le plus homogène : C{best['cluster_id']} "
                     f"'{best['name']}' "
                     f"(homo={best['cluster_homogeneity']:.2f}, n={best['n']})")
    if n_unresolved > 0:
        n_obs = metadata.get('n_observations_filtered', 1)
        pct   = 100 * n_unresolved / max(n_obs, 1)
        lines.append(f"  - Résidu : {n_unresolved} runs ({pct:.0f}%) "
                     f"→ cible phases suivantes")
    if not any(l.startswith('  -') for l in lines):
        lines.append("  (analyse POC)")
    lines.append("")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"✓ TXT : {output_path}")
