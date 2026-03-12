"""
prc.utils.parquet_to_json

Responsabilité : Convertit un Parquet en JSON lisible pour debug/analyse LLM

Usage :
    python -m utils.parquet_to_json poc2
    python -m utils.parquet_to_json poc2 --sample 20
    python -m utils.parquet_to_json poc2 --stats

    # Audit HDBSCAN — 4 fichiers JSON compacts
    python -m utils.parquet_to_json poc2 --audit

Outputs existants :
    reports/debug_{phase}.json          → dump complet (ou sample)
    reports/debug_{phase}_stats.json    → stats descriptives par feature

Outputs audit HDBSCAN :
    reports/summary_gamma_{phase}.json    → stats agrégées par gamma_id
    reports/summary_encoding_{phase}.json → stats agrégées par encoding_id
    reports/anomalies_{phase}.json        → rows avec NaN ou Inf (composition + comptes)
    reports/manifest_{phase}.json         → table gamma × encoding → n_runs + run_status
"""

import json
import argparse
import numpy as np
import pandas as pd
from itertools import product as iproduct
from pathlib import Path


# =============================================================================
# AXES_COLS — source de vérité partagée
# =============================================================================

AXES_COLS = [
    'run_status',
    'phase', 'gamma_id', 'encoding_id', 'modifier_id',
    'n_dof', 'rank_eff', 'max_it',
    'gamma_params', 'encoding_params', 'modifier_params',
    'seed_CI', 'seed_run',
    # v6 compat
    'max_iterations',
]


# =============================================================================
# HELPERS EXISTANTS
# =============================================================================

def _safe(val):
    """Convertit valeur numpy/float en type JSON-serializable."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        if np.isnan(val):
            return None  # JSON ne supporte pas NaN
        if np.isinf(val):
            return str(val)  # "inf" ou "-inf"
        return float(val)
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    return val


def df_to_rows(df: pd.DataFrame) -> list:
    """Convertit DataFrame → liste de rows JSON-sérialisables."""
    feature_cols = [c for c in df.columns if c not in AXES_COLS]

    rows = []
    for _, row in df.iterrows():
        comp     = {c: _safe(row[c]) for c in AXES_COLS if c in df.columns}
        features = {c: _safe(row[c]) for c in feature_cols}
        rows.append({'composition': comp, 'features': features})

    return rows


def compute_stats(df: pd.DataFrame) -> dict:
    """Stats descriptives par feature numérique."""
    bool_cols    = [c for c in df.columns if df[c].dtype == bool
                    or c.startswith('has_') or c.startswith('is_')]
    numeric_cols = [c for c in df.columns
                    if c not in AXES_COLS + bool_cols]

    stats = {}

    if 'run_status' in df.columns:
        counts = df['run_status'].value_counts().to_dict()
        stats['run_status'] = {
            'type'  : 'categorical',
            'counts': {str(k): int(v) for k, v in counts.items()},
        }

    for col in numeric_cols:
        vals  = pd.to_numeric(df[col], errors='coerce')
        valid = vals.dropna()

        if len(valid) == 0:
            stats[col] = {'n_valid': 0, 'n_nan': len(vals)}
            continue

        finite = valid[np.isfinite(valid)]

        stats[col] = {
            'n_valid': int(len(valid)),
            'n_nan':   int(vals.isna().sum()),
            'n_inf':   int(np.isinf(valid).sum()),
            'mean':    _safe(finite.mean())           if len(finite) > 0 else None,
            'median':  _safe(finite.median())         if len(finite) > 0 else None,
            'std':     _safe(finite.std())            if len(finite) > 0 else None,
            'min':     _safe(finite.min())            if len(finite) > 0 else None,
            'max':     _safe(finite.max())            if len(finite) > 0 else None,
            'q1':      _safe(finite.quantile(0.25))   if len(finite) > 0 else None,
            'q3':      _safe(finite.quantile(0.75))   if len(finite) > 0 else None,
        }

    for col in bool_cols:
        if col in df.columns:
            true_count = int(df[col].sum()) if df[col].dtype != object else 0
            stats[col] = {
                'type'          : 'bool',
                'n_true'        : true_count,
                'n_false'       : int(len(df) - true_count),
                'fraction_true' : round(true_count / len(df), 4) if len(df) > 0 else 0,
            }

    return stats


# =============================================================================
# HELPERS AUDIT
# =============================================================================

def _family(encoding_id: str) -> str:
    """
    Extrait la famille depuis l'encoding_id.
    'SYM-001' → 'SYM', 'ASY-002' → 'ASY', 'R3-003' → 'R3', 'RN-001' → 'RN'
    Fallback : 'UNKNOWN'
    """
    if not isinstance(encoding_id, str):
        return 'UNKNOWN'
    prefix = encoding_id.split('-')[0].upper()
    return prefix if prefix in ('SYM', 'ASY', 'R3', 'RN') else 'UNKNOWN'


def _nan_inf_counts(row: pd.Series, feature_cols: list) -> dict:
    """
    Compte NaN et Inf dans les feature_cols d'une row.
    I: row Series, feature_cols list[str]
    O: {'n_nan': int, 'n_inf': int}
    """
    n_nan = 0
    n_inf = 0
    for col in feature_cols:
        v = row[col]
        try:
            fv = float(v)
            if np.isnan(fv):
                n_nan += 1
            elif np.isinf(fv):
                n_inf += 1
        except (TypeError, ValueError):
            pass
    return {'n_nan': n_nan, 'n_inf': n_inf}


def _feature_stats_for_group(sub_df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Stats agrégées (mean, std, nan_rate, inf_rate) par feature sur un sous-df.
    I: sub_df DataFrame, feature_cols list[str]
    O: dict {feature_name: {mean, std, nan_rate, inf_rate}}
    """
    n   = len(sub_df)
    out = {}
    for col in feature_cols:
        vals   = pd.to_numeric(sub_df[col], errors='coerce')
        finite = vals[np.isfinite(vals)]
        out[col] = {
            'mean'    : _safe(finite.mean()) if len(finite) > 0 else None,
            'std'     : _safe(finite.std())  if len(finite) > 0 else None,
            'nan_rate': round(float(vals.isna().sum()) / n, 4)          if n > 0 else None,
            'inf_rate': round(float(np.isinf(vals.dropna()).sum()) / n, 4) if n > 0 else None,
        }
    return out


def _run_status_dist(sub_df: pd.DataFrame) -> dict:
    """
    Distribution run_status sur un sous-df.
    I: sub_df DataFrame
    O: dict {status_str: count_int}
    """
    if 'run_status' not in sub_df.columns:
        return {}
    counts = sub_df['run_status'].value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def _write_json(path: Path, obj: dict) -> None:
    """Écrit obj en JSON indenté."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# =============================================================================
# EXPORTEURS AUDIT
# =============================================================================

def export_summary_gamma(
    df          : pd.DataFrame,
    feature_cols: list,
    phase       : str,
    reports_dir : Path,
) -> Path:
    """
    Stats agrégées par gamma_id.

    Structure :
    {
      "phase": ...,
      "n_gammas": ...,
      "gammas": {
        "GAM-001": {
          "n_runs": int,
          "run_status_dist": {...},
          "feature_stats": {feature: {mean, std, nan_rate, inf_rate}}
        }, ...
      }
    }
    """
    result = {'phase': phase, 'n_gammas': 0, 'gammas': {}}

    for gamma_id, sub in df.groupby('gamma_id'):
        result['gammas'][str(gamma_id)] = {
            'n_runs'          : int(len(sub)),
            'run_status_dist' : _run_status_dist(sub),
            'feature_stats'   : _feature_stats_for_group(sub, feature_cols),
        }

    result['n_gammas'] = len(result['gammas'])
    path = reports_dir / f'summary_gamma_{phase}.json'
    _write_json(path, result)
    print(f"✓ summary_gamma      : {path}  ({result['n_gammas']} gammas)")
    return path


def export_summary_encoding(
    df          : pd.DataFrame,
    feature_cols: list,
    phase       : str,
    reports_dir : Path,
) -> Path:
    """
    Stats agrégées par encoding_id.

    Structure :
    {
      "phase": ...,
      "n_encodings": ...,
      "encodings": {
        "SYM-001": {
          "family": "SYM",
          "rank_eff_values": [2],
          "n_runs": int,
          "run_status_dist": {...},
          "feature_stats": {feature: {mean, std, nan_rate, inf_rate}}
        }, ...
      }
    }
    """
    result = {'phase': phase, 'n_encodings': 0, 'encodings': {}}

    for enc_id, sub in df.groupby('encoding_id'):
        rank_vals = sorted(sub['rank_eff'].dropna().unique().tolist()) \
                    if 'rank_eff' in sub.columns else []
        result['encodings'][str(enc_id)] = {
            'family'          : _family(str(enc_id)),
            'rank_eff_values' : [int(r) for r in rank_vals],
            'n_runs'          : int(len(sub)),
            'run_status_dist' : _run_status_dist(sub),
            'feature_stats'   : _feature_stats_for_group(sub, feature_cols),
        }

    result['n_encodings'] = len(result['encodings'])
    path = reports_dir / f'summary_encoding_{phase}.json'
    _write_json(path, result)
    print(f"✓ summary_encoding   : {path}  ({result['n_encodings']} encodings)")
    return path


def export_anomalies(
    df          : pd.DataFrame,
    feature_cols: list,
    phase       : str,
    reports_dir : Path,
) -> Path:
    """
    Rows ayant au moins un NaN ou Inf dans les features.
    Triées par gamma_id puis encoding_id.
    Contient composition + comptes seulement — pas les 68 valeurs.

    Structure :
    {
      "phase": ...,
      "n_anomalies": int,
      "n_total": int,
      "anomaly_rate": float,
      "rows": [
        {
          "gamma_id": ..., "encoding_id": ..., "modifier_id": ...,
          "n_dof": ..., "rank_eff": ..., "max_it": ...,
          "run_status": ..., "seed_CI": ..., "seed_run": ...,
          "n_nan": int, "n_inf": int
        }, ...
      ]
    }
    """
    comp_cols_present = [c for c in AXES_COLS if c in df.columns]

    rows_out = []
    for _, row in df.iterrows():
        counts = _nan_inf_counts(row, feature_cols)
        if counts['n_nan'] > 0 or counts['n_inf'] > 0:
            entry = {c: _safe(row[c]) for c in comp_cols_present}
            entry.update(counts)
            rows_out.append(entry)

    # Tri gamma_id puis encoding_id
    rows_out.sort(key=lambda r: (str(r.get('gamma_id', '')),
                                  str(r.get('encoding_id', ''))))

    n_total = len(df)
    n_anom  = len(rows_out)
    result  = {
        'phase'        : phase,
        'n_anomalies'  : n_anom,
        'n_total'      : n_total,
        'anomaly_rate' : round(n_anom / n_total, 4) if n_total > 0 else 0.0,
        'rows'         : rows_out,
    }

    path = reports_dir / f'anomalies_{phase}.json'
    _write_json(path, result)
    print(f"✓ anomalies          : {path}  ({n_anom}/{n_total} rows)")
    return path


def export_manifest(
    df          : pd.DataFrame,
    phase       : str,
    reports_dir : Path,
) -> Path:
    """
    Table gamma × encoding → n_runs + run_status_dist.
    Cases vides (combinaison absente) marquées null.

    Structure :
    {
      "phase": ...,
      "all_gammas": [...],
      "all_encodings": [...],
      "n_combinations_observed": int,
      "n_combinations_possible": int,
      "n_combinations_missing": int,
      "missing_combinations": [["GAM-X", "ENC-Y"], ...],
      "table": {
        "GAM-001": {
          "SYM-001": {"n_runs": int, "run_status_dist": {...}},
          "SYM-002": null,
          ...
        }, ...
      }
    }
    """
    all_gammas    = sorted(df['gamma_id'].dropna().unique().tolist())
    all_encodings = sorted(df['encoding_id'].dropna().unique().tolist())

    # Index des paires observées
    observed: dict = {}
    for (gid, eid), sub in df.groupby(['gamma_id', 'encoding_id']):
        observed[(str(gid), str(eid))] = {
            'n_runs'         : int(len(sub)),
            'run_status_dist': _run_status_dist(sub),
        }

    # Construction table complète
    table    = {}
    missing  = []
    for gid in all_gammas:
        table[gid] = {}
        for eid in all_encodings:
            key = (str(gid), str(eid))
            if key in observed:
                table[gid][eid] = observed[key]
            else:
                table[gid][eid] = None
                missing.append([gid, eid])

    n_possible = len(all_gammas) * len(all_encodings)
    n_observed = len(observed)

    result = {
        'phase'                   : phase,
        'all_gammas'              : [str(g) for g in all_gammas],
        'all_encodings'           : [str(e) for e in all_encodings],
        'n_combinations_observed' : n_observed,
        'n_combinations_possible' : n_possible,
        'n_combinations_missing'  : len(missing),
        'missing_combinations'    : missing,
        'table'                   : table,
    }

    path = reports_dir / f'manifest_{phase}.json'
    _write_json(path, result)
    print(f"✓ manifest           : {path}  "
          f"({n_observed}/{n_possible} combinaisons, {len(missing)} manquantes)")
    return path


# =============================================================================
# POINT D'ENTRÉE AUDIT
# =============================================================================

def export_audit(
    phase      : str,
    results_dir: Path = None,
    reports_dir: Path = None,
) -> None:
    """
    Charge le Parquet une fois, produit les 4 fichiers d'audit HDBSCAN.

    Args:
        phase       : Nom phase (ex: 'poc2')
        results_dir : Dossier Parquet (défaut: data/results/)
        reports_dir : Dossier output (défaut: reports/)
    """
    if results_dir is None:
        results_dir = Path('data/results')
    if reports_dir is None:
        reports_dir = Path('reports')

    parquet_path = results_dir / f'{phase}.parquet'
    if not parquet_path.exists():
        print(f"❌ Parquet introuvable : {parquet_path}")
        return

    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"  {len(df)} observations, {len(df.columns)} colonnes")

    reports_dir.mkdir(parents=True, exist_ok=True)

    # feature_cols isolé une seule fois — partagé entre les 4 exporteurs
    feature_cols = [c for c in df.columns if c not in AXES_COLS]

    export_summary_gamma(df, feature_cols, phase, reports_dir)
    export_summary_encoding(df, feature_cols, phase, reports_dir)
    export_anomalies(df, feature_cols, phase, reports_dir)
    export_manifest(df, phase, reports_dir)


# =============================================================================
# EXPORT PARQUET EXISTANT (inchangé)
# =============================================================================

def export_parquet(phase: str, sample: int = None, stats_only: bool = False,
                   results_dir: Path = None, reports_dir: Path = None):
    """
    Exporte Parquet → JSON debug.

    Args:
        phase       : Nom phase (ex: 'poc2')
        sample      : Nombre de rows à exporter (None = tout)
        stats_only  : N'exporte que les stats, pas les rows
        results_dir : Dossier Parquet (défaut: data/results/)
        reports_dir : Dossier output (défaut: reports/)
    """
    if results_dir is None:
        results_dir = Path('data/results')
    if reports_dir is None:
        reports_dir = Path('reports')

    parquet_path = results_dir / f'{phase}.parquet'

    if not parquet_path.exists():
        print(f"❌ Parquet introuvable : {parquet_path}")
        return

    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"  {len(df)} observations, {len(df.columns)} colonnes")

    reports_dir.mkdir(parents=True, exist_ok=True)

    stats      = compute_stats(df)
    stats_path = reports_dir / f'debug_{phase}_stats.json'

    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            'phase'         : phase,
            'n_observations': len(df),
            'n_features'    : len(df.columns),
            'feature_stats' : stats,
        }, f, indent=2, ensure_ascii=False)

    print(f"✓ Stats écrites : {stats_path}")

    if stats_only:
        return

    if sample is not None:
        df_export = df.sample(min(sample, len(df)), random_state=42)
        print(f"  Sample : {len(df_export)} rows")
    else:
        df_export = df

    rows      = df_to_rows(df_export)
    dump_path = reports_dir / f'debug_{phase}.json'

    with open(dump_path, 'w', encoding='utf-8') as f:
        json.dump({
            'phase'     : phase,
            'n_total'   : len(df),
            'n_exported': len(rows),
            'columns'   : list(df.columns),
            'rows'      : rows,
        }, f, indent=2, ensure_ascii=False)

    print(f"✓ Dump écrit : {dump_path}  ({len(rows)} rows)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Export Parquet → JSON debug / audit HDBSCAN',
        epilog="""
Exemples :
  python -m utils.parquet_to_json poc2
  python -m utils.parquet_to_json poc2 --sample 20
  python -m utils.parquet_to_json poc2 --stats
  python -m utils.parquet_to_json poc2 --audit
        """
    )
    parser.add_argument('phase', help='Nom phase (ex: poc2)')
    parser.add_argument('--sample', type=int, default=None,
                        help='Nombre de rows à exporter (défaut: tout)')
    parser.add_argument('--stats',  action='store_true',
                        help='Stats uniquement (pas les rows)')
    parser.add_argument('--audit',  action='store_true',
                        help='Produit les 4 fichiers audit HDBSCAN')

    args = parser.parse_args()

    if args.audit:
        export_audit(args.phase)
    else:
        export_parquet(args.phase, sample=args.sample, stats_only=args.stats)


if __name__ == '__main__':
    main()
