"""
tests/validate_parquet.py

Outil d'inspection et comparaison de parquets PRC v7.

Modes :
  Inspection  : python -m tests.validate_parquet parquet_a
  Comparaison : python -m tests.validate_parquet parquet_a parquet_b

Philosophie (charter S6) :
  - Checks durs UNIQUEMENT sur structure (colonnes, types) — pass/fail
  - Tout le reste : observations numériques brutes, pas de seuils arbitraires
  - NaN structurels lisibles depuis rank_eff + gamma_id, pas re-encodés ici
"""

import sys, math, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore', message='.*SLASCLS.*')
warnings.filterwarnings('ignore', category=FutureWarning)

META_COLS = {
    'run_status', 'phase', 'gamma_id', 'encoding_id', 'modifier_id',
    'n_dof', 'rank_eff', 'max_it',
    'gamma_params', 'encoding_params', 'modifier_params',
    'seed_CI', 'seed_run',
}
EXPECTED_N_FEATURES = 68
STRUCTURAL_NAN_COLS = {
    'f3_entanglement_entropy_mode1_mean',
    'f3_entanglement_entropy_mode1_final',
    'f4_trace_J_mean', 'f4_trace_J_std', 'f4_trace_J_final',
    'f4_jvp_norm_mean', 'f4_jvp_norm_final',
    'f4_jacobian_asymmetry_mean', 'f4_jacobian_asymmetry_final',
    'f4_local_lyapunov_mean', 'f4_local_lyapunov_std',
}
FAMILY_PREFIXES = [
    ('F1','f1_'),('F2','f2_'),('F3','f3_'),('F4','f4_'),
    ('F5','f5_'),('F6','f6_'),('F7','f7_'),('PS','ps_'),('health','health_'),
]
VALID_STATUSES = {'OK','EXPLOSION','NAN_ALL','INVALID','FAIL'}
KEY_FEATURES = [
    'f1_effective_rank_mean','f1_condition_number_mean',
    'f2_von_neumann_entropy_mean','f2_entropy_production_rate',
    'f3_mode_asymmetry_mean','f3_entanglement_entropy_mode1_mean',
    'f4_lyapunov_empirical_mean','f4_trace_J_mean',
    'f5_delta_D_mean','f5_bregman_cost_mean',
    'f6_causal_asymmetry_index',
    'f7_dmd_spectral_radius','f7_dmd_spectral_entropy',
    'ps_norm_ratio','ps_pnn40_von_neumann',
    'health_has_inf','health_is_collapsed',
]

def _feat_cols(df):
    return [c for c in df.columns if c not in META_COLS]

def _families(cols):
    fam = {n:[] for n,_ in FAMILY_PREFIXES}
    for c in cols:
        for n,p in FAMILY_PREFIXES:
            if c.startswith(p): fam[n].append(c); break
    return fam

def _nan_pct(s):   return s.isna().mean()*100
def _inf_count(s):
    try: return int(np.isinf(pd.to_numeric(s,errors='coerce')).sum())
    except: return 0

def _stats(s):
    s = pd.to_numeric(s,errors='coerce').dropna()
    s = s[np.isfinite(s)]
    if not len(s): return dict(n=0,mean=float('nan'),std=float('nan'),min=float('nan'),max=float('nan'))
    return dict(n=len(s),mean=float(s.mean()),std=float(s.std()),min=float(s.min()),max=float(s.max()))

def _f(v,w=10):
    if isinstance(v,float) and math.isnan(v): return 'NaN'.rjust(w)
    if isinstance(v,float): return f'{v:.4g}'.rjust(w)
    return str(v).rjust(w)

def _chk(label,ok,detail=''):
    print(f"  [{'✓' if ok else '✗'}] {label}" + (f' — {detail}' if detail else ''))
    return ok

# =============================================================================
# INSPECTION
# =============================================================================

def inspect(path: Path) -> dict:
    if not path.exists():
        print(f'  ERREUR : {path} introuvable'); sys.exit(1)

    df   = pd.read_parquet(path)
    feat = _feat_cols(df)
    fams = _families(feat)
    ok_df = df[df['run_status']=='OK'] if 'run_status' in df.columns else df

    print(f'\n{"="*62}')
    print(f'  {path.name}')
    print(f'  {len(df):,} rows × {len(df.columns)} colonnes  |  RAM {df.memory_usage(deep=True).sum()/1024**2:.1f} MB')
    print(f'{"="*62}')

    # ── Checks durs ───────────────────────────────────────────────
    print('\n── Schéma (checks durs) ──────────────────────────────────')
    hard_ok = True
    hard_ok &= _chk(f'{EXPECTED_N_FEATURES} features', len(feat)==EXPECTED_N_FEATURES, f'got {len(feat)}')
    for col in ['run_status','gamma_id','encoding_id','n_dof','rank_eff','max_it']:
        hard_ok &= _chk(f"colonne '{col}'", col in df.columns)
    non_num = [c for c in feat if not pd.api.types.is_numeric_dtype(df[c])]
    hard_ok &= _chk('features numériques', not non_num, f'non-num: {non_num}' if non_num else 'clean')
    unk = set(df['run_status'].unique())-VALID_STATUSES if 'run_status' in df.columns else set()
    hard_ok &= _chk('statuts valides', not unk, f'inconnus: {unk}' if unk else 'clean')

    print(f'\n  {"Famille":8}  {"features":>8}')
    for n,cols in fams.items(): print(f'  {n:<8}  {len(cols):>8}')

    # ── run_status ────────────────────────────────────────────────
    print('\n── run_status ────────────────────────────────────────────')
    if 'run_status' in df.columns:
        for s,n in df['run_status'].value_counts().items():
            print(f'  {s:<14} {n:>7,}  ({n/len(df)*100:5.1f}%)')

    # ── NaN / Inf par famille sur OK ──────────────────────────────
    print('\n── NaN & Inf par famille (runs OK) ───────────────────────')
    print(f'  {"Famille":8}  {"cols":>4}  {"NaN%":>7}  {"Inf":>8}')
    for n,cols in fams.items():
        if not cols or n=='health': continue
        sub = ok_df[cols] if len(ok_df) else pd.DataFrame(columns=cols)
        print(f'  {n:<8}  {len(cols):>4}  {sub.isna().mean().mean()*100:>6.1f}%'
              f'  {sum(_inf_count(sub[c]) for c in cols):>8,}')

    # ── NaN structurels par rank_eff ──────────────────────────────
    print('\n── F3 mode1 NaN par rank_eff ─────────────────────────────')
    m1 = [c for c in feat if 'mode1' in c]
    if m1 and 'rank_eff' in df.columns:
        print(f'  {"rank_eff":>8}  {"n":>7}', end='')
        for c in m1: print(f'  {c[-10:]:>12}', end='')
        print()
        for rv in sorted(df['rank_eff'].unique()):
            mask = df['rank_eff']==rv
            print(f'  {rv:>8}  {mask.sum():>7}', end='')
            for c in m1: print(f'  {_nan_pct(df.loc[mask,c]):>11.1f}%', end='')
            print()

    # ── F4 Hutchinson NaN par gamma_id ────────────────────────────
    print('\n── F4 Hutchinson NaN par gamma_id ────────────────────────')
    hutch = next((c for c in feat if 'trace_J_mean' in c), None)
    lyap  = next((c for c in feat if 'lyapunov_empirical_mean' in c), None)
    if hutch and 'gamma_id' in df.columns:
        print(f'  {"gamma_id":12}  {"Hutchinson NaN%":>16}  {"Lyapunov_emp NaN%":>18}')
        for gid,grp in df.groupby('gamma_id'):
            hn = _nan_pct(grp[hutch])
            ln = _nan_pct(grp[lyap]) if lyap else float('nan')
            print(f'  {gid:<12}  {hn:>15.1f}%  {ln:>17.1f}%')

    # ── Distributions features clés (runs OK) ─────────────────────
    print('\n── Distributions (runs OK, valeurs finies) ───────────────')
    print(f'  {"Feature":<46} {"mean":>10} {"std":>10} {"min":>10} {"max":>10}')
    for col in KEY_FEATURES:
        if col not in ok_df.columns: continue
        st = _stats(ok_df[col])
        print(f'  {col:<46}{_f(st["mean"])}{_f(st["std"])}{_f(st["min"])}{_f(st["max"])}')

    # ── OK rate par gamma ──────────────────────────────────────────
    print('\n── OK rate par gamma_id ──────────────────────────────────')
    if 'gamma_id' in df.columns and 'run_status' in df.columns:
        print(f'  {"gamma_id":12} {"OK%":>7} {"EXP%":>7} {"NAN_ALL%":>9} {"INV%":>7} {"n":>7}')
        for gid,grp in df.groupby('gamma_id'):
            vc = grp['run_status'].value_counts()
            p = lambda s: vc.get(s,0)/len(grp)*100
            print(f'  {gid:<12} {p("OK"):>6.1f}% {p("EXPLOSION"):>6.1f}%'
                  f' {p("NAN_ALL"):>8.1f}% {p("INVALID"):>6.1f}% {len(grp):>7,}')

    # ── Échantillon ───────────────────────────────────────────────
    print('\n── Échantillon (3 runs OK) ───────────────────────────────')
    sample = ok_df.sample(min(3,len(ok_df)),random_state=42) if len(ok_df) else pd.DataFrame()
    for idx,row in sample.iterrows():
        print(f'\n  #{idx}  {row.get("gamma_id","?")} × {row.get("encoding_id","?")}'
              f'  n_dof={row.get("n_dof","?")}  max_it={row.get("max_it","?")}'
              f'  rank_eff={row.get("rank_eff","?")}')
        for col in ['f1_effective_rank_mean','f2_von_neumann_entropy_mean',
                    'f3_mode_asymmetry_mean','f4_lyapunov_empirical_mean',
                    'f5_delta_D_mean','f7_dmd_spectral_radius','health_has_inf']:
            if col in row.index:
                v = row[col]
                print(f'    {col:<44} {"NaN" if (isinstance(v,float) and math.isnan(v)) else f"{v:.4g}"}')
    print()
    return {'hard_ok': hard_ok, 'df': df, 'feat': feat, 'fams': fams}


# =============================================================================
# COMPARAISON
# =============================================================================

def compare(path_a: Path, path_b: Path):
    df_a, df_b = pd.read_parquet(path_a), pd.read_parquet(path_b)
    feat_a, feat_b = _feat_cols(df_a), _feat_cols(df_b)
    common = [c for c in feat_a if c in feat_b]
    ok_a = df_a[df_a['run_status']=='OK'] if 'run_status' in df_a.columns else df_a
    ok_b = df_b[df_b['run_status']=='OK'] if 'run_status' in df_b.columns else df_b

    print(f'\n{"="*70}')
    print(f'  COMPARAISON')
    print(f'  A : {path_a.name}  ({len(df_a):,} rows)')
    print(f'  B : {path_b.name}  ({len(df_b):,} rows)')
    print(f'{"="*70}')

    # Schéma
    print('\n── Schéma ────────────────────────────────────────────────')
    only_a = set(feat_a)-set(feat_b)
    only_b = set(feat_b)-set(feat_a)
    print(f'  Features communes : {len(common)}')
    if only_a: print(f'  Seulement dans A  : {sorted(only_a)}')
    if only_b: print(f'  Seulement dans B  : {sorted(only_b)}')

    # run_status
    print('\n── run_status ────────────────────────────────────────────')
    print(f'  {"Status":14} {"A%":>9} {"B%":>9} {"delta":>9}')
    all_s = (set(df_a['run_status'].unique())|set(df_b['run_status'].unique())
             if 'run_status' in df_a.columns and 'run_status' in df_b.columns else set())
    for s in sorted(all_s):
        pa = (df_a['run_status']==s).mean()*100 if 'run_status' in df_a.columns else 0
        pb = (df_b['run_status']==s).mean()*100 if 'run_status' in df_b.columns else 0
        print(f'  {s:<14} {pa:>8.1f}% {pb:>8.1f}% {pb-pa:>+8.1f}%')

    # Delta distributions
    print('\n── Delta distributions (runs OK) ─────────────────────────')
    print(f'  {"Feature":<44} {"mean_A":>10} {"mean_B":>10} {"delta%":>10}')
    for col in [c for c in KEY_FEATURES if c in common]:
        ma = _stats(ok_a[col])['mean'] if col in ok_a.columns else float('nan')
        mb = _stats(ok_b[col])['mean'] if col in ok_b.columns else float('nan')
        if not math.isnan(ma) and not math.isnan(mb) and ma != 0:
            ds = f'{(mb-ma)/abs(ma)*100:>+9.1f}%'
        else:
            ds = '       n/a'
        print(f'  {col:<44}{_f(ma)}{_f(mb)}{ds}')

    # Delta NaN structurels
    print('\n── Delta NaN structurels (runs OK) ───────────────────────')
    print(f'  {"Feature":<44} {"NaN%_A":>8} {"NaN%_B":>8} {"delta":>8}')
    for col in sorted(STRUCTURAL_NAN_COLS):
        if col not in ok_a.columns or col not in ok_b.columns: continue
        na, nb = _nan_pct(ok_a[col]), _nan_pct(ok_b[col])
        print(f'  {col:<44} {na:>7.1f}% {nb:>7.1f}% {nb-na:>+7.1f}%')
    print()


# =============================================================================
# MAIN
# =============================================================================

args = [Path(a) for a in sys.argv[1:] if not a.startswith('-')]

if len(args) == 0:
    r = inspect(Path('data/results/bourrin_v7.parquet'))
    sys.exit(0 if r['hard_ok'] else 1)
elif len(args) == 1:
    r = inspect(args[0])
    sys.exit(0 if r['hard_ok'] else 1)
elif len(args) == 2:
    inspect(args[0]); inspect(args[1]); compare(args[0], args[1])
else:
    print('Usage: validate_parquet [parquet_a] [parquet_b]'); sys.exit(1)