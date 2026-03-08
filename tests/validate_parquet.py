"""
validate_parquet.py

Validation données parquet bourrin_v7.
Vérifications :
  V1 — Schéma : colonnes, types, count
  V2 — NaN structurels : F3 mode1, F4 Hutchinson
  V3 — Inf : aucun Inf dans les features (NaN OK, Inf non)
  V4 — Distributions : stats par famille F1-F7
  V5 — Cohérence status : health_ flags vs OK/EXPLOSION/INVALID
  V6 — Échantillon aléatoire : 5 runs complets

Usage :
    python validate_parquet.py [path_to_parquet]
    python validate_parquet.py  # cherche data/results/bourrin_v7.parquet
"""

import sys
import math
import warnings
import random
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore', message='.*SLASCLS.*')
warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# Config
# =============================================================================

PARQUET_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/results/bourrin_v7.parquet')

FEATURE_FAMILIES = {
    'F1_spectral'     : [c for c in [] ],  # rempli après chargement
    'F2_informational': [],
    'F3_entanglement' : [],
    'F4_dynamics'     : [],
    'F5_transport'    : [],
    'F6_transfer'     : [],
    'F7_dmd'          : [],
    'PS_postprocess'  : [],
    'health'          : [],
}

META_COLS = [
    'run_status', 'phase', 'gamma_id', 'encoding_id', 'modifier_id',
    'n_dof', 'rank_eff', 'max_it',
    'gamma_params', 'encoding_params', 'modifier_params',
    'seed_CI', 'seed_run',
]

RESULTS  = []
WARNINGS = []

def check(name, ok, detail=""):
    sym = "✓" if ok else "✗"
    msg = f"  [{sym}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    RESULTS.append((name, ok))
    return ok

def warn(msg):
    WARNINGS.append(msg)
    print(f"  [⚠] {msg}")

# =============================================================================
# Chargement
# =============================================================================

print(f"\n=== Chargement ===")
print(f"  {PARQUET_PATH}")

if not PARQUET_PATH.exists():
    print(f"  ERREUR : fichier introuvable")
    sys.exit(1)

df = pd.read_parquet(PARQUET_PATH)
print(f"  {len(df):,} rows × {len(df.columns)} colonnes")
print(f"  Taille mémoire : {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Identifier familles
all_cols = list(df.columns)
feat_cols = [c for c in all_cols if c not in META_COLS and not c.startswith('seed')]

for col in feat_cols:
    if col.startswith('f1_'):   FEATURE_FAMILIES['F1_spectral'].append(col)
    elif col.startswith('f2_'): FEATURE_FAMILIES['F2_informational'].append(col)
    elif col.startswith('f3_'): FEATURE_FAMILIES['F3_entanglement'].append(col)
    elif col.startswith('f4_'): FEATURE_FAMILIES['F4_dynamics'].append(col)
    elif col.startswith('f5_'): FEATURE_FAMILIES['F5_transport'].append(col)
    elif col.startswith('f6_'): FEATURE_FAMILIES['F6_transfer'].append(col)
    elif col.startswith('f7_'): FEATURE_FAMILIES['F7_dmd'].append(col)
    elif col.startswith('ps_'): FEATURE_FAMILIES['PS_postprocess'].append(col)
    elif col.startswith('health_'): FEATURE_FAMILIES['health'].append(col)

# =============================================================================
# V1 — Schéma
# =============================================================================

print(f"\n=== V1 — Schéma ===")

check("V1.1 — 68 features présentes",
      len(feat_cols) == 68, f"got {len(feat_cols)}")

for meta in ['gamma_id', 'encoding_id', 'n_dof', 'max_it']:
    check(f"V1.2 — colonne '{meta}' présente", meta in df.columns)

check("V1.3 — run_status présent", 'run_status' in df.columns)

for fam, cols in FEATURE_FAMILIES.items():
    if fam == 'health':
        continue
    check(f"V1.4 — famille {fam} non vide", len(cols) > 0, f"{len(cols)} features")

# Types
numeric_ok = all(pd.api.types.is_numeric_dtype(df[c]) for c in feat_cols)
non_numeric = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(df[c])]
check("V1.5 — toutes features numériques", numeric_ok,
      f"non-numériques: {non_numeric}" if non_numeric else "clean")

print(f"\n  Familles détectées :")
for fam, cols in FEATURE_FAMILIES.items():
    print(f"    {fam:<20} : {len(cols):2d} features")

# =============================================================================
# V2 — NaN structurels
# =============================================================================

print(f"\n=== V2 — NaN structurels ===")

# F3 mode1 : NaN sur rank=2 (SYM/ASY encodings), non-NaN sur R3
f3_mode1_cols = [c for c in feat_cols if 'mode1' in c]
if f3_mode1_cols:
    # rank_eff == 2 → tenseur rang 2 → mode1 NaN attendu
    r2_mask = df['rank_eff'] == 2
    r3_mask = df['rank_eff'] >= 3

    for col in f3_mode1_cols:
        nan_r2 = df.loc[r2_mask, col].isna().mean() if r2_mask.any() else None
        nan_r3 = df.loc[r3_mask, col].isna().mean() if r3_mask.any() else None

        if nan_r2 is not None:
            check(f"V2.1 — {col} : NaN sur rank=2 (>95%)",
                  nan_r2 > 0.95, f"{nan_r2*100:.1f}% NaN (n={r2_mask.sum():,})")
        if nan_r3 is not None:
            check(f"V2.2 — {col} : non-NaN sur rank≥3 (<20%)",
                  nan_r3 < 0.20, f"{nan_r3*100:.1f}% NaN (n={r3_mask.sum():,})")

# F4 Hutchinson : NaN sur gammas non-diff
f4_hutch_cols = [c for c in feat_cols if any(k in c for k in
                 ['trace_J', 'jvp_norm', 'jacobian_asymmetry', 'local_lyapunov'])
                 and not 'empirical' in c]

# Vérifier qu'il y a des NaN (gammas non-diff présents)
if f4_hutch_cols:
    nan_rate = df[f4_hutch_cols[0]].isna().mean()
    check("V2.3 — F4 Hutchinson : NaN présents (gammas non-diff)",
          nan_rate > 0, f"{nan_rate*100:.1f}% NaN sur {f4_hutch_cols[0]}")

    # Lyapunov empirical jamais NaN sur OK
    emp_cols = [c for c in feat_cols if 'lyapunov_empirical' in c and 'mean' in c]
    if emp_cols:
        ok_mask = df['run_status'] == 'OK'
        nan_emp = df.loc[ok_mask, emp_cols[0]].isna().mean()
        check("V2.4 — lyapunov_empirical : non-NaN sur OK (<5%)",
              nan_emp < 0.05, f"{nan_emp*100:.1f}% NaN")

# =============================================================================
# V3 — Pas d'Inf
# =============================================================================

print(f"\n=== V3 — Absence Inf ===")

inf_counts = {}
for col in feat_cols:
    n_inf = np.isinf(df[col].values).sum()
    if n_inf > 0:
        inf_counts[col] = n_inf

check("V3.1 — zéro Inf dans toutes les features",
      len(inf_counts) == 0,
      f"Inf dans : {inf_counts}" if inf_counts else "clean")

# =============================================================================
# V4 — Distributions par famille
# =============================================================================

print(f"\n=== V4 — Distributions ===")

ok_df = df[df['run_status'] == 'OK']

for fam, cols in FEATURE_FAMILIES.items():
    if not cols or fam == 'health':
        continue
    sub = ok_df[cols]
    nan_pct   = sub.isna().mean().mean() * 100
    finite_df = sub.apply(lambda x: x.dropna())
    print(f"\n  {fam} ({len(cols)} features, runs OK) :")
    print(f"    NaN moyen     : {nan_pct:.1f}%")
    for col in cols[:3]:  # 3 exemples par famille
        s = ok_df[col].dropna()
        if len(s) > 0:
            print(f"    {col:<45} : "
                  f"mean={s.mean():.3f}  std={s.std():.3f}  "
                  f"[{s.min():.3f}, {s.max():.3f}]")

# =============================================================================
# V5 — Cohérence status / health flags
# =============================================================================

print(f"\n=== V5 — Cohérence status ===")

print(f"\n  Distribution run_status :")
status_counts = df['run_status'].value_counts()
for status, count in status_counts.items():
    pct = count / len(df) * 100
    print(f"    {status:<12} : {count:>7,}  ({pct:.1f}%)")

if 'health_has_nan_inf' in df.columns:
    exp_mask   = df['run_status'] == 'EXPLOSION'
    ok_mask    = df['run_status'] == 'OK'
    nan_on_exp = df.loc[exp_mask, 'health_has_nan_inf'].mean() if exp_mask.any() else None
    nan_on_ok  = df.loc[ok_mask,  'health_has_nan_inf'].mean() if ok_mask.any()  else None

    if nan_on_exp is not None:
        check("V5.1 — health_has_nan_inf élevé sur EXPLOSION (>80%)",
              nan_on_exp > 0.80, f"{nan_on_exp*100:.1f}%")
    if nan_on_ok is not None:
        check("V5.2 — health_has_nan_inf bas sur OK (<10%)",
              nan_on_ok < 0.10, f"{nan_on_ok*100:.1f}%")

print(f"\n  OK rate par gamma_id :")
for gid, grp in df.groupby('gamma_id'):
    ok_rate  = (grp['run_status'] == 'OK').mean() * 100
    exp_rate = (grp['run_status'] == 'EXPLOSION').mean() * 100
    print(f"    {gid:<10} : OK={ok_rate:5.1f}%  EXP={exp_rate:4.1f}%  n={len(grp):,}")

# =============================================================================
# V6 — Échantillon aléatoire
# =============================================================================

print(f"\n=== V6 — Échantillon aléatoire (5 runs OK) ===")

ok_sample = ok_df.sample(min(5, len(ok_df)), random_state=42)

for idx, row in ok_sample.iterrows():
    print(f"\n  Run #{idx} — {row.get('gamma_id','?')} × {row.get('encoding_id','?')} "
          f"n_dof={row.get('n_dof','?')} max_it={row.get('max_it','?')}")

    # F1 key features
    for col in ['f1_effective_rank_mean', 'f1_spectral_gap_mean',
                'f2_von_neumann_entropy_mean', 'f3_mode_asymmetry_mean',
                'f4_lyapunov_empirical_mean', 'f5_delta_D_mean',
                'f7_dmd_spectral_radius', 'ps_autocorr_lag1']:
        if col in row:
            val = row[col]
            display = f"{val:.4f}" if not (isinstance(val, float) and math.isnan(val)) else "NaN"
            print(f"    {col:<45} : {display}")

# =============================================================================
# Résumé
# =============================================================================

print(f"\n{'='*58}")
n_pass = sum(1 for _, ok in RESULTS if ok)
n_fail = sum(1 for _, ok in RESULTS if not ok)
print(f"  VALIDATION : {n_pass}/{len(RESULTS)} checks OK")
if WARNINGS:
    print(f"  WARNINGS   : {len(WARNINGS)}")
if n_fail:
    print(f"  ÉCHECS :")
    for name, ok in RESULTS:
        if not ok:
            print(f"    ✗ {name}")
print(f"{'='*58}")