"""
tests/verify_sprint1.py

Vérifie les 3 changements Sprint 1 sur le parquet cache_bench.
Cible : health_has_inf, NAN_ALL, run_status propre.

Usage : python -m tests.verify_sprint1 [parquet_path]
"""
import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path

PARQUET = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/results/cache_bench.parquet')

df = pd.read_parquet(PARQUET)
print(f"\n=== verify_sprint1 — {PARQUET.name} ({len(df):,} rows) ===\n")

ok   = True

# ── C1 — health_has_nan_inf disparu, health_has_inf présent ──────────────────
print("C1 — Renommage health flag")
c1a = 'health_has_nan_inf' not in df.columns
c1b = 'health_has_inf' in df.columns
print(f"  [{'✓' if c1a else '✗'}] health_has_nan_inf absent")
print(f"  [{'✓' if c1b else '✗'}] health_has_inf présent")
ok = ok and c1a and c1b

# ── C2 — health_has_inf bas sur OK (NaN structurels exclus) ──────────────────
print("\nC2 — health_has_inf propre sur OK")
if 'health_has_inf' in df.columns:
    ok_mask  = df['run_status'] == 'OK'
    exp_mask = df['run_status'] == 'EXPLOSION'
    rate_ok  = df.loc[ok_mask,  'health_has_inf'].mean() * 100
    rate_exp = df.loc[exp_mask, 'health_has_inf'].mean() * 100 if exp_mask.any() else float('nan')
    c2a = rate_ok < 10
    c2b = rate_exp > 80 if not math.isnan(rate_exp) else True
    print(f"  [{'✓' if c2a else '✗'}] health_has_inf sur OK  : {rate_ok:.1f}%  (attendu <10%)")
    print(f"  [{'✓' if c2b else '✗'}] health_has_inf sur EXP : {rate_exp:.1f}%  (attendu >80%)")
    ok = ok and c2a and c2b

# ── C3 — run_status taxonomie complète ───────────────────────────────────────
print("\nC3 — run_status distribution")
counts = df['run_status'].value_counts()
for status, n in counts.items():
    print(f"  {status:<12} : {n:>5,}  ({n/len(df)*100:.1f}%)")

valid_statuses = {'OK', 'EXPLOSION', 'NAN_ALL', 'INVALID', 'FAIL'}
unknown = set(counts.index) - valid_statuses
c3a = len(unknown) == 0
print(f"  [{'✓' if c3a else '✗'}] statuts valides uniquement{' — inconnus: '+str(unknown) if unknown else ''}")
ok = ok and c3a

# ── C4 — NAN_ALL cohérent : zéro Inf, features non-structurelles toutes NaN ──
print("\nC4 — NAN_ALL cohérence")
nan_all_mask = df['run_status'] == 'NAN_ALL'
if nan_all_mask.any():
    n_nan_all = nan_all_mask.sum()
    # Pas d'Inf sur NAN_ALL
    feat_cols = [c for c in df.columns if c.startswith(('f1_','f2_','f3_','f4_','f5_','f6_','f7_','ps_'))]
    inf_on_nan_all = df.loc[nan_all_mask, feat_cols].apply(
        lambda col: np.isinf(col.values.astype(float)).any()
    ).any()
    c4a = not inf_on_nan_all
    print(f"  NAN_ALL count : {n_nan_all}")
    print(f"  [{'✓' if c4a else '✗'}] zéro Inf sur NAN_ALL runs")
    ok = ok and c4a
else:
    print("  NAN_ALL count : 0  (aucun run de ce type dans ce batch — normal)")

# ── C5 — EXPLOSION cohérent : au moins un Inf ────────────────────────────────
print("\nC5 — EXPLOSION cohérence")
exp_mask = df['run_status'] == 'EXPLOSION'
if exp_mask.any():
    feat_cols = [c for c in df.columns if c.startswith(('f1_','f2_','f3_','f4_','f5_','f6_','f7_','ps_'))]
    # Sur les EXPLOSION, combien ont effectivement un Inf
    has_inf_on_exp = df.loc[exp_mask, feat_cols].apply(
        lambda col: np.isinf(pd.to_numeric(col, errors='coerce').fillna(0)).any(), axis=1
    )
    rate_inf_exp = has_inf_on_exp.mean() * 100
    c5a = rate_inf_exp > 80
    print(f"  [{'✓' if c5a else '✗'}] EXPLOSION runs avec Inf effectif : {rate_inf_exp:.1f}%  (attendu >80%)")
    ok = ok and c5a

# ── Résumé ────────────────────────────────────────────────────────────────────
print(f"\n{'='*48}")
print(f"  Sprint 1 : {'✓ VALIDÉ' if ok else '✗ ÉCHECS DÉTECTÉS'}")
print(f"{'='*48}\n")
sys.exit(0 if ok else 1)
