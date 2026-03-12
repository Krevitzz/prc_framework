"""
prc.utils.parquet_to_json

Responsabilité : Convertit un Parquet en JSON lisible pour debug/analyse LLM

Usage :
    python -m utils.parquet_to_json poc2
    python -m utils.parquet_to_json poc2 --sample 20
    python -m utils.parquet_to_json poc2 --stats

Outputs :
    reports/debug_{phase}.json     → dump complet (ou sample)
    reports/debug_{phase}_stats.json → stats descriptives par feature
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


# =============================================================================
# HELPERS
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
    axes_cols = ['gamma_id', 'encoding_id', 'modifier_id', 'n_dof',
                 'max_iterations', 'phase']
    feature_cols = [c for c in df.columns if c not in axes_cols]

    rows = []
    for _, row in df.iterrows():
        comp = {c: _safe(row[c]) for c in axes_cols if c in df.columns}
        features = {c: _safe(row[c]) for c in feature_cols}
        rows.append({'composition': comp, 'features': features})

    return rows


def compute_stats(df: pd.DataFrame) -> dict:
    """Stats descriptives par feature numérique."""
    axes_cols = ['gamma_id', 'encoding_id', 'modifier_id', 'n_dof',
                 'max_iterations', 'phase']
    bool_cols = [c for c in df.columns if df[c].dtype == bool
                 or c.startswith('has_') or c.startswith('is_')]
    numeric_cols = [c for c in df.columns
                    if c not in axes_cols + bool_cols]

    stats = {}

    for col in numeric_cols:
        vals = pd.to_numeric(df[col], errors='coerce')
        valid = vals.dropna()
        inf_count = np.isinf(vals.replace([np.inf, -np.inf], np.nan).isna()).sum()

        if len(valid) == 0:
            stats[col] = {'n_valid': 0, 'n_nan': len(vals)}
            continue

        finite = valid[np.isfinite(valid)]

        stats[col] = {
            'n_valid': int(len(valid)),
            'n_nan':   int(vals.isna().sum()),
            'n_inf':   int(np.isinf(valid).sum()),
            'mean':    _safe(finite.mean()) if len(finite) > 0 else None,
            'median':  _safe(finite.median()) if len(finite) > 0 else None,
            'std':     _safe(finite.std()) if len(finite) > 0 else None,
            'min':     _safe(finite.min()) if len(finite) > 0 else None,
            'max':     _safe(finite.max()) if len(finite) > 0 else None,
            'q1':      _safe(finite.quantile(0.25)) if len(finite) > 0 else None,
            'q3':      _safe(finite.quantile(0.75)) if len(finite) > 0 else None,
        }

    # Flags booléens
    for col in bool_cols:
        if col in df.columns:
            true_count = int(df[col].sum()) if df[col].dtype != object else 0
            stats[col] = {
                'type': 'bool',
                'n_true': true_count,
                'n_false': int(len(df) - true_count),
                'fraction_true': round(true_count / len(df), 4) if len(df) > 0 else 0
            }

    return stats


# =============================================================================
# MAIN
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

    # Stats descriptives
    stats = compute_stats(df)
    stats_path = reports_dir / f'debug_{phase}_stats.json'

    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            'phase': phase,
            'n_observations': len(df),
            'n_features': len(df.columns),
            'feature_stats': stats
        }, f, indent=2, ensure_ascii=False)

    print(f"✓ Stats écrites : {stats_path}")

    if stats_only:
        return

    # Rows complets (ou sample)
    if sample is not None:
        df_export = df.sample(min(sample, len(df)), random_state=42)
        print(f"  Sample : {len(df_export)} rows")
    else:
        df_export = df

    rows = df_to_rows(df_export)
    dump_path = reports_dir / f'debug_{phase}.json'

    with open(dump_path, 'w', encoding='utf-8') as f:
        json.dump({
            'phase': phase,
            'n_total': len(df),
            'n_exported': len(rows),
            'columns': list(df.columns),
            'rows': rows
        }, f, indent=2, ensure_ascii=False)

    print(f"✓ Dump écrit : {dump_path}  ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description='Export Parquet → JSON debug',
        epilog="""
Exemples :
  python -m utils.parquet_to_json poc2
  python -m utils.parquet_to_json poc2 --sample 20
  python -m utils.parquet_to_json poc2 --stats
        """
    )
    parser.add_argument('phase', help='Nom phase (ex: poc2)')
    parser.add_argument('--sample', type=int, default=None,
                        help='Nombre de rows à exporter (défaut: tout)')
    parser.add_argument('--stats', action='store_true',
                        help='Stats uniquement (pas les rows)')

    args = parser.parse_args()
    export_parquet(args.phase, sample=args.sample, stats_only=args.stats)


if __name__ == '__main__':
    main()
