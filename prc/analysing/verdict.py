"""
prc.analysing.verdict

Responsabilité : Analyses post-batch : profiling + analysing + outliers + régimes → rapport

FIX : Format récurrence + rapport TXT protection données
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from utils.data_loading_lite import load_yaml
from profiling.hub_profiling import run_profiling
from analysing.hub_analysing import run_analysing
from analysing.concordance_lite import run_concordance_cross_phases
from analysing.outliers_lite import analyze_outliers
from analysing.regimes_lite import (
    load_regime_thresholds,
    classify_regimes_batch,
    refine_conserves_norm_with_cv
)


# =============================================================================
# POOL REQUIREMENTS
# =============================================================================

def load_pool_requirements() -> Dict:
    """Charge contraintes pool actif."""
    req_path = Path('configs/constraints/pool_requirements.yaml')
    
    if not req_path.exists():
        return {
            'n_dof': {'min': None, 'max': None},
            'max_iterations': {'min': None, 'max': None},
            'deprecated': {'gammas': [], 'encodings': [], 'modifiers': []},
        }
    
    return load_yaml(req_path)


def filter_rows_by_pool(
    rows: List[Dict],
    requirements: Dict,
    verbose: bool = True
) -> Tuple[List[Dict], int, Dict[str, int]]:
    """Filtre rows selon pool requirements."""
    filtered = []
    skip_reasons = {}
    
    for row in rows:
        comp = row['composition']
        skip = False
        reason = None
        
        if requirements['n_dof']['min'] is not None:
            if comp['n_dof'] < requirements['n_dof']['min']:
                skip = True
                reason = 'n_dof_too_low'
        
        if not skip and requirements['n_dof']['max'] is not None:
            if comp['n_dof'] > requirements['n_dof']['max']:
                skip = True
                reason = 'n_dof_too_high'
        
        if not skip and comp['gamma_id'] in requirements['deprecated']['gammas']:
            skip = True
            reason = 'gamma_deprecated'
        
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
    rows: List[Dict],
    filter_pool: bool = True,
    regime_profile: str = 'default'
) -> Dict:
    """Exécute verdict sur rows (RAM)."""
    print(f"\n=== Verdict intra (RAM) ===")
    print(f"Observations initiales: {len(rows)}")
    
    n_skipped_pool = 0
    
    if filter_pool:
        requirements = load_pool_requirements()
        rows, n_skipped_pool, skip_reasons = filter_rows_by_pool(rows, requirements)
        print(f"Observations après filtrage pool: {len(rows)}")
    
    if len(rows) == 0:
        print("\n⚠️  Aucune observation valide après filtrage pool")
        return {
            'profiling': {},
            'analysing': {},
            'outliers': {},
            'regimes': {},
            'metadata': {'n_filtered': 0, 'n_skipped_pool': n_skipped_pool},
        }
    
    profiling_results = run_profiling(rows)
    analysing_results = run_analysing(rows, n_clusters=3)
    
    print("\n=== Détection outliers ===")
    outliers_results = analyze_outliers(rows, contamination=0.1)
    print(f"Outliers: {outliers_results['n_outliers']} ({outliers_results['outlier_fraction']*100:.1f}%)")
    print(f"Stables: {outliers_results['n_stables']}")
    
    print("\n=== Classification régimes ===")
    thresholds = load_regime_thresholds(regime_profile)
    
    regimes_results = classify_regimes_batch(
        rows,
        outliers_results['stable_indices'],
        thresholds
    )
    
    regimes_results = refine_conserves_norm_with_cv(rows, regimes_results, thresholds)
    
    print(f"Régimes détectés: {len(regimes_results['regimes'])}")
    for regime, data in list(regimes_results['regimes'].items())[:3]:
        print(f"  {regime}: {data['count']} runs ({data['fraction']*100:.1f}%)")
    
    metadata = {
        'n_observations_initial': len(rows) + n_skipped_pool,
        'n_observations_filtered': len(rows),
        'n_skipped_pool': n_skipped_pool,
        'n_gammas': len(profiling_results.get('gamma', {})),
        'n_encodings': len(profiling_results.get('encoding', {})),
        'n_modifiers': len(profiling_results.get('modifier', {})),
        'regime_profile': regime_profile,
    }
    
    print(f"\n✓ Verdict complete")
    print(f"  Profiling: {metadata['n_gammas']} gammas, {metadata['n_encodings']} encodings")
    
    # Clustering stratified (mode défaut) ou unified
    clustering = (
        analysing_results.get('clustering_stratified', {}).get('universal')
        or analysing_results.get('clustering')
        or {}
    )
    print(f"  Analysing: {clustering.get('n_clusters', 'N/A')} clusters")
    
    cross_results = _cross_cluster_regimes(analysing_results, regimes_results, rows, outliers_results)

    return {
        'profiling': profiling_results,
        'analysing': analysing_results,
        'outliers': outliers_results,
        'regimes': regimes_results,
        'cross': cross_results,
        'metadata': metadata,
    }


# =============================================================================
# MODE 2 : VERDICT SINGLE-PHASE (PARQUET)
# =============================================================================

def run_verdict_from_parquet(parquet_path: Path, regime_profile: str = 'default') -> Dict:
    """Exécute verdict depuis Parquet."""
    print(f"\n=== Verdict depuis Parquet ===")
    print(f"Loading: {parquet_path}\n")
    
    df = pd.read_parquet(parquet_path)
    print(f"Observations: {len(df)}")
    print(f"Colonnes: {len(df.columns)}\n")
    
    rows = []
    axes_cols = ['gamma_id', 'encoding_id', 'modifier_id', 'n_dof', 'max_iterations']
    features_cols = [c for c in df.columns if c not in axes_cols + ['phase']]
    
    for _, row_df in df.iterrows():
        comp = {col: row_df[col] for col in axes_cols}
        features = {col: row_df[col] for col in features_cols}
        rows.append({'composition': comp, 'features': features, 'layers': ['universal']})
    
    return run_verdict_intra(rows, regime_profile=regime_profile)


# =============================================================================
# MODE 3 : VERDICT CROSS-PHASES
# =============================================================================

def scan_major_phases(results_dir: Path = None) -> List[Path]:
    """Scan phases principales (r0, r1, r2, ...)."""
    if results_dir is None:
        results_dir = Path('data/results')
    
    if not results_dir.exists():
        return []
    
    all_parquets = list(results_dir.glob('*.parquet'))
    major_phases = []
    
    for p in all_parquets:
        stem = p.stem
        if stem.startswith('r') and len(stem) == 2 and stem[1].isdigit():
            major_phases.append(p)
    
    return sorted(major_phases)


def run_verdict_cross_phases(results_dir: Path = None, regime_profile: str = 'default') -> Dict:
    """Exécute verdict inter-phases (concordance)."""
    print(f"\n=== Verdict cross-phases ===")
    
    parquet_paths = scan_major_phases(results_dir)
    
    if len(parquet_paths) == 0:
        print("⚠️  Aucune phase principale trouvée")
        return {
            'phases': {},
            'concordance': {},
            'metadata': {'n_phases': 0, 'phases_analyzed': []},
        }
    
    print(f"Phases détectées: {len(parquet_paths)}")
    for p in parquet_paths:
        print(f"  {p.stem}")
    print()
    
    phases_results = {}
    phases_data = {}
    
    for parquet_path in parquet_paths:
        phase_name = parquet_path.stem
        print(f"--- Analysing {phase_name} ---")
        
        verdict = run_verdict_from_parquet(parquet_path, regime_profile)
        phases_results[phase_name] = verdict
        
        df = pd.read_parquet(parquet_path)
        axes_cols = ['gamma_id', 'encoding_id', 'modifier_id', 'n_dof', 'max_iterations']
        features_cols = [c for c in df.columns if c not in axes_cols + ['phase']]
        
        rows = []
        for _, row_df in df.iterrows():
            comp = {col: row_df[col] for col in axes_cols}
            features = {col: row_df[col] for col in features_cols}
            rows.append({'composition': comp, 'features': features, 'layers': ['universal']})
        
        phases_data[phase_name] = rows
    
    print(f"\n--- Concordance cross-phases ---")
    concordance_results = run_concordance_cross_phases(phases_data)
    
    print(f"✓ Concordance complete")
    print(f"  Kappa pairs: {len(concordance_results.get('kappa', {}))}")
    
    metadata = {
        'n_phases': len(parquet_paths),
        'phases_analyzed': [p.stem for p in parquet_paths],
        'regime_profile': regime_profile,
    }
    
    return {
        'phases': phases_results,
        'concordance': concordance_results,
        'metadata': metadata,
    }



# =============================================================================
# CROISEMENT CLUSTERS × RÉGIMES
# =============================================================================

def _cross_cluster_regimes(
    analysing_results: Dict,
    regimes_results: Dict,
    rows: List[Dict],
    outliers_results: Dict = None
) -> Dict:
    """
    Croise clusters HDBSCAN × régimes handcoded.

    Pour chaque cluster :
        - Distribution des régimes → régime dominant + pureté
        - Pureté < PURITY_THRESHOLD → divergence flagguée ⚠

    Pour chaque régime :
        - Distribution sur les clusters → concentré ou fragmenté

    Args:
        analysing_results : Retour de run_analysing()
        regimes_results   : Retour de classify_regimes_batch()
        rows              : Liste complète rows (pour mapping index → régime)
        outliers_results  : Retour de analyze_outliers() — outliers non assignés → 'OUTLIER'

    Returns:
        {
            'clusters': {
                cluster_id: {
                    'n_runs'          : int,
                    'dominant_regime' : str,
                    'purity'          : float,
                    'regime_dist'     : {regime: fraction},
                    'divergence'      : bool
                }
            },
            'regimes': {
                regime_name: {
                    'cluster_dist'    : {cluster_id: fraction},
                    'fragmented'      : bool   # True si > 2 clusters significatifs
                }
            },
            'divergences': List[str],   # messages lisibles
            'n_clusters'  : int,
            'n_noise'     : int,
        }
        {} si clustering indisponible
    """
    PURITY_THRESHOLD = 0.70

    # Récupérer résultat clustering universal
    clustering = (
        analysing_results.get('clustering_stratified', {}).get('universal')
        or analysing_results.get('clustering')
        or None
    )

    if clustering is None or not clustering.get('labels'):
        return {}

    labels       = clustering['labels']       # List[int], -1 = bruit
    valid_indices = clustering['valid_indices'] # indices dans rows

    # Construire mapping index_row → régime
    # Les runs outliers (non classifiés par classify_regimes_batch) → 'OUTLIER'
    outlier_indices = set(
        outliers_results.get('outlier_indices', [])
        if outliers_results else []
    )
    row_to_regime = {}
    for regime, data in regimes_results.get('regimes', {}).items():
        for idx in data.get('run_indices', []):
            row_to_regime[idx] = regime
    # Marquer les outliers non assignés à un régime
    for idx in outlier_indices:
        if idx not in row_to_regime:
            row_to_regime[idx] = 'OUTLIER'

    # Construire mapping index_row → cluster_label
    row_to_cluster = {}
    for label, row_idx in zip(labels, valid_indices):
        row_to_cluster[row_idx] = label

    # ── Par cluster ──────────────────────────────────────────────────────────
    cluster_ids = sorted(set(labels) - {-1})
    clusters_out = {}

    for cid in cluster_ids:
        cluster_rows = [
            row_idx for row_idx, lbl in row_to_cluster.items() if lbl == cid
        ]

        regime_counts = {}
        for row_idx in cluster_rows:
            regime = row_to_regime.get(row_idx, 'UNKNOWN')
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        n = len(cluster_rows)
        regime_dist = {r: c / n for r, c in sorted(
            regime_counts.items(), key=lambda x: x[1], reverse=True
        )}

        dominant = max(regime_counts, key=regime_counts.get) if regime_counts else 'UNKNOWN'
        purity   = regime_dist.get(dominant, 0.0)

        clusters_out[cid] = {
            'n_runs'          : n,
            'dominant_regime' : dominant,
            'purity'          : round(purity, 3),
            'regime_dist'     : {r: round(v, 3) for r, v in regime_dist.items()},
            'divergence'      : purity < PURITY_THRESHOLD,
        }

    # ── Par régime ───────────────────────────────────────────────────────────
    all_regimes = set(regimes_results.get('regimes', {}).keys())
    regimes_out = {}

    for regime in all_regimes:
        regime_rows = regimes_results['regimes'][regime].get('run_indices', [])
        cluster_counts = {}

        for row_idx in regime_rows:
            lbl = row_to_cluster.get(row_idx, -1)
            cluster_counts[lbl] = cluster_counts.get(lbl, 0) + 1

        n = len(regime_rows)
        cluster_dist = {c: round(cnt / n, 3) for c, cnt in sorted(
            cluster_counts.items(), key=lambda x: x[1], reverse=True
        )}

        # Fragmenté si > 2 clusters captent chacun > 15%
        significant = sum(1 for f in cluster_dist.values() if f > 0.15)
        regimes_out[regime] = {
            'cluster_dist': cluster_dist,
            'fragmented'  : significant > 2,
        }

    # ── Divergences lisibles ─────────────────────────────────────────────────
    divergences = []
    for cid, data in clusters_out.items():
        if data['divergence']:
            top2 = list(data['regime_dist'].items())[:2]
            parts = " / ".join(f"{r} {v*100:.0f}%" for r, v in top2)
            divergences.append(
                f"Cluster {cid} ({data['n_runs']} runs) : {parts} "
                f"→ sous-structure non capturée par régimes"
            )

    for regime, data in regimes_out.items():
        if data['fragmented']:
            divergences.append(
                f"Régime {regime} fragmenté sur {len(data['cluster_dist'])} clusters "
                f"→ hétérogénéité interne"
            )

    return {
        'clusters'    : clusters_out,
        'regimes'     : regimes_out,
        'divergences' : divergences,
        'n_clusters'  : clustering.get('n_clusters', 0),
        'n_noise'     : clustering.get('n_noise', 0),
        'n_features'  : clustering.get('n_features', 0),
        'n_features_total': clustering.get('n_features_total', 0),
    }


# =============================================================================
# EXPORT RAPPORTS
# =============================================================================

def write_verdict_report(verdict_results: Dict, output_path: Path):
    """Écrit rapport verdict JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(verdict_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Rapport JSON écrit: {output_path}")


def write_verdict_report_txt(verdict_results: Dict, output_path: Path):
    """
    Écrit rapport verdict TXT lisible humain.
    
    FIX : Protection accès dictionnaires vides + format récurrence uniforme
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    
    # Header
    metadata = verdict_results.get('metadata', {})
    phase = output_path.stem.replace('verdict_', '').replace('.txt', '').upper()
    
    lines.append(f"=== VERDICT {phase} ===")
    lines.append(f"Runs : {metadata.get('n_observations_filtered', 0)} success, "
                 f"{metadata.get('n_skipped_pool', 0)} skipped")
    lines.append(f"Régime profile : {metadata.get('regime_profile', 'default')}")
    lines.append("")
    
    # Outliers
    outliers = verdict_results.get('outliers', {})
    n_outliers = outliers.get('n_outliers', 0)
    outlier_frac = outliers.get('outlier_fraction', 0.0)
    
    lines.append(f"OUTLIERS ({n_outliers} runs, {outlier_frac*100:.1f}%)")
    
    if n_outliers > 0:
        lines.append("  Récurrence atomics :")
        
        gamma_rec = outliers.get('recurrence', {}).get('gamma', {})
        if gamma_rec:
            for gamma_id, data in list(gamma_rec.items())[:5]:
                lines.append(f"    {gamma_id} : {data['count']}/{data['total_subset']} "
                            f"({data['fraction']*100:.0f}%)")
        else:
            lines.append("    (aucune récurrence détectée)")
        
        encoding_rec = outliers.get('recurrence', {}).get('encoding', {})
        if encoding_rec:
            lines.append("  Encodings récurrents :")
            for enc_id, data in list(encoding_rec.items())[:3]:
                lines.append(f"    {enc_id} : {data['fraction']*100:.0f}%")
    else:
        lines.append("  (aucun outlier détecté)")
    
    lines.append("")
    
    # Stables + Régimes
    n_stables = outliers.get('n_stables', 0)
    stable_frac = 1.0 - outlier_frac if n_outliers > 0 else 1.0
    
    lines.append(f"STABLES ({n_stables} runs, {stable_frac*100:.1f}%)")
    
    regimes_data = verdict_results.get('regimes', {}).get('regimes', {})
    
    if regimes_data:
        lines.append("  Régimes dominants :")
        
        # Filtrer régimes actifs (count > 0)
        active_regimes = {k: v for k, v in regimes_data.items() if v['count'] > 0}
        
        if active_regimes:
            for regime, data in list(active_regimes.items())[:5]:
                count = data['count']
                frac = data['fraction']
                
                lines.append(f"    {regime} : {count} runs ({frac*100:.0f}%)")
                
                gamma_rec = data.get('recurrence_gamma', {})
                if gamma_rec:
                    atomics_str = ", ".join([
                        f"{k} ({v['fraction']*100:.0f}%)" 
                        for k, v in list(gamma_rec.items())[:3]
                    ])
                    lines.append(f"      Atomics : {atomics_str}")
                
                lines.append("")
        else:
            lines.append("  (aucun régime actif)")
    else:
        lines.append("  (aucun régime détecté)")
    
    lines.append("")
    
    # Insights
    lines.append("INSIGHTS")
    
    # Insight 1 : Gamma corrélé outliers
    if n_outliers > 0:
        gamma_rec = outliers.get('recurrence', {}).get('gamma', {})
        if gamma_rec:
            top_gamma = list(gamma_rec.items())[0]
            gamma_id = top_gamma[0]
            gamma_data = top_gamma[1]
            
            if gamma_data['fraction'] > 0.5:
                lines.append(f"  - {gamma_id} fortement corrélé outliers "
                            f"({gamma_data['fraction']*100:.0f}%) → calibration nécessaire")
    
    # Insight 2 : Combinaisons stables
    conserves_data = regimes_data.get('CONSERVES_NORM', {})
    if conserves_data.get('count', 0) > 0:
        gamma_stable = conserves_data.get('recurrence_gamma', {})
        enc_stable = conserves_data.get('recurrence_encoding', {})
        
        if gamma_stable and enc_stable:
            top_g = list(gamma_stable.items())[0]
            top_e = list(enc_stable.items())[0]
            lines.append(f"  - {top_g[0]} + {top_e[0]} → combinaison stable "
                        f"({conserves_data['fraction']*100:.0f}% CONSERVES_NORM)")
    
    # Insight 3 : Régime dominant
    if regimes_data:
        active_regimes = {k: v for k, v in regimes_data.items() if v['count'] > 0}
        if active_regimes:
            dominant = max(active_regimes.items(), key=lambda x: x[1]['count'])
            regime_name = dominant[0]
            regime_data = dominant[1]
            
            if regime_data['fraction'] > 0.6:
                lines.append(f"  - Régime dominant : {regime_name} "
                            f"({regime_data['fraction']*100:.0f}% des runs stables)")
    
    # Si aucun insight
    if len([l for l in lines if l.startswith('  -')]) == 0:
        lines.append("  (analyse POC — dataset trop petit pour insights robustes)")
    
    # Clustering × Régimes
    cross = verdict_results.get('cross', {})
    if cross and cross.get('n_clusters', 0) > 0:
        n_cl  = cross['n_clusters']
        n_noise_cl = cross.get('n_noise', 0)
        n_feat = cross.get('n_features', 0)
        n_feat_total = cross.get('n_features_total', 0)

        lines.append(f"CLUSTERING ({n_cl} clusters, {n_noise_cl} bruit)")
        lines.append(f"  Features orthogonales utilisées : {n_feat}/{n_feat_total}")
        lines.append("")

        for cid, cdata in sorted(cross.get('clusters', {}).items()):
            dom   = cdata['dominant_regime']
            pur   = cdata['purity']
            n_r   = cdata['n_runs']
            flag  = "⚠ divergence" if cdata['divergence'] else "✓ concordant régimes"

            # Top 2 régimes si divergence
            top2 = list(cdata['regime_dist'].items())[:2]
            if cdata['divergence']:
                dist_str = " / ".join(f"{r} {v*100:.0f}%" for r, v in top2)
                lines.append(f"  Cluster {cid} ({n_r} runs) → {dist_str}  {flag}")
            else:
                lines.append(f"  Cluster {cid} ({n_r} runs) → {dom} {pur*100:.0f}%  {flag}")

        lines.append("")

        divergences = cross.get('divergences', [])
        if divergences:
            lines.append("  Divergences :")
            for d in divergences:
                lines.append(f"    - {d}")
            lines.append("")
    elif cross is not None:
        lines.append("CLUSTERING")
        lines.append("  (aucun cluster dense trouvé — dataset trop petit ou trop hétérogène)")
        lines.append("")

    # Write
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ Rapport TXT écrit: {output_path}")