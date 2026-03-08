"""
tests/test_featuring.py

Suite de tests featuring PRC v7 — F1→F7 complets.

Couvre :
  T1 — measure_state : structure fixe, 22 signaux
  T2 — post_scan     : 68 features, zéro clé manquante/en trop
  T3 — run_one_jax   : rank=2, gamma différentiable
  T4 — run_one_jax   : rank=3, entanglement_entropy_mode1 non-NaN
  T5 — run_one_jax   : gamma non-différentiable → NaN Hutchinson, Lyapunov non-NaN
  T6 — run_one_jax   : gamma différentiable → Hutchinson non-NaN
  T7 — cohérence     : FEATURE_NAMES aligné avec post_scan output

Usage :
    python -m tests.test_featuring
    pytest tests/test_featuring.py -v
"""

import math
import sys
import jax
import jax.numpy as jnp
from jax import lax

# =============================================================================
# Helpers
# =============================================================================

PASS = "✓"
FAIL = "✗"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    results.append((name, condition))
    return condition


def is_nan(v):
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def is_finite(v):
    try:
        f = float(v)
        return math.isfinite(f)
    except (TypeError, ValueError):
        return False


# =============================================================================
# Imports pipeline
# =============================================================================

print("\n=== Imports ===")
try:
    from featuring.jax_features import measure_state, post_scan, FEATURE_NAMES
    from running.run_one_jax import run_one_jax
    check("Imports OK", True)
except Exception as e:
    check("Imports OK", False, str(e))
    sys.exit(1)

# =============================================================================
# Fixtures
# =============================================================================

key        = jax.random.PRNGKey(42)
state_r2   = jax.random.normal(key, (10, 10))
state_r3   = jax.random.normal(key, (8, 8, 8))
state_next = jax.random.normal(jax.random.PRNGKey(1), (10, 10))
state_prev = jnp.zeros_like(state_r2)
A_k        = jnp.zeros((10, 10))
P_k        = jnp.eye(10) * 1e4

gamma_diff    = lambda s, p, k: s * p.get('beta', jnp.array(0.99))
gamma_nondiff = lambda s, p, k: jnp.zeros_like(s).at[jnp.argmax(jnp.abs(s))].set(1.0)
mod_fn        = lambda s, p, k: s

# =============================================================================
# T1 — measure_state : structure
# =============================================================================

print("\n=== T1 — measure_state ===")
try:
    measures, A_k2, P_k2 = measure_state(
        state_r2, state_next, state_prev,
        gamma_diff, {'beta': jnp.array(0.99)},
        key, A_k, P_k,
        is_differentiable=True,
    )
    check("T1.1 — retourne 3 éléments (measures, A_k, P_k)", True)
    check("T1.2 — 22 signaux dans-scan", len(measures) == 22, f"got {len(measures)}")
    check("T1.3 — A_k shape (10,10)", A_k2.shape == (10, 10), f"got {A_k2.shape}")
    check("T1.4 — scalaires shape ()", all(v.shape == () for v in measures.values()),
          f"shapes: {set(v.shape for v in measures.values())}")
    check("T1.5 — frob_norm fini", is_finite(measures['f1_frob_norm']))
    check("T1.6 — von_neumann_entropy fini", is_finite(measures['f2_von_neumann_entropy']))
    check("T1.7 — mode_asymmetry fini", is_finite(measures['f3_mode_asymmetry']))
except Exception as e:
    check("T1 — measure_state", False, str(e))

# =============================================================================
# T2 — post_scan : structure vs FEATURE_NAMES
# =============================================================================

print("\n=== T2 — post_scan structure ===")
try:
    T = 50
    fake_signals = {
        'f1_effective_rank'            : jnp.ones(T) * 3.0,
        'f1_spectral_gap'              : jnp.linspace(0.1, 0.5, T),
        'f1_nuclear_frobenius_ratio'   : jnp.ones(T) * 0.8,
        'f1_sv_decay_rate'             : jnp.ones(T) * 0.2,
        'f1_rank1_residual'            : jnp.ones(T) * 0.3,
        'f1_condition_number'          : jnp.linspace(2.0, 5.0, T),
        'f1_frob_norm'                 : jnp.linspace(1.0, 0.5, T),
        'f2_von_neumann_entropy'       : jnp.linspace(1.0, 2.0, T),
        'f2_renyi2_entropy'            : jnp.ones(T) * 1.5,
        'f2_shannon_entropy'           : jnp.ones(T) * 2.0,
        'f3_entanglement_entropy_mode0': jnp.ones(T) * 1.2,
        'f3_entanglement_entropy_mode1': jnp.ones(T) * float('nan'),
        'f3_mode_asymmetry'            : jnp.linspace(0.0, 0.3, T),
        'f3_inter_mode_sv_var'         : jnp.ones(T) * 0.1,
        'f4_trace_J'                   : jnp.ones(T) * 0.5,
        'f4_jvp_norm'                  : jnp.ones(T) * 1.0,
        'f4_jacobian_asymmetry'        : jnp.ones(T) * 0.2,
        'f4_local_lyapunov'            : jnp.ones(T) * -0.1,
        'f4_lyapunov_empirical'        : jnp.linspace(-0.5, 0.1, T),
        'f5_delta_D'                   : jnp.ones(T) * 0.05,
        'f5_frob_gradient'             : jnp.ones(T) * 0.1,
        'f5_bregman_cost'              : jnp.ones(T) * 0.02,
    }

    features = post_scan(fake_signals, state_r2, A_k, P_k)

    missing = set(FEATURE_NAMES) - set(features.keys())
    extra   = set(features.keys()) - set(FEATURE_NAMES)

    check("T2.1 — 68 features retournées", len(features) == 68, f"got {len(features)}")
    check("T2.2 — zéro clé manquante", len(missing) == 0, f"manquantes: {missing}")
    check("T2.3 — zéro clé en trop",   len(extra)   == 0, f"en trop: {extra}")
    check("T2.4 — FEATURE_NAMES count", len(FEATURE_NAMES) == 68, f"got {len(FEATURE_NAMES)}")
    check("T2.5 — toutes valeurs float", all(isinstance(v, float) for v in features.values()),
          f"types: {set(type(v).__name__ for v in features.values())}")
except Exception as e:
    check("T2 — post_scan", False, str(e))

# =============================================================================
# T3 — run_one_jax rank=2, différentiable
# =============================================================================

print("\n=== T3 — run_one_jax rank=2 différentiable ===")
try:
    f = run_one_jax(
        gamma_diff, {'beta': jnp.array(0.99)},
        mod_fn, {},
        state_r2, key, 50,
        is_differentiable=True,
    )
    check("T3.1 — 68 features", len(f) == 68, f"got {len(f)}")
    check("T3.2 — f1_effective_rank_mean fini", is_finite(f['f1_effective_rank_mean']))
    check("T3.3 — f2_von_neumann_entropy_mean fini", is_finite(f['f2_von_neumann_entropy_mean']))
    check("T3.4 — f3_mode_asymmetry_mean fini", is_finite(f['f3_mode_asymmetry_mean']))
    check("T3.5 — f4_trace_J_mean fini", is_finite(f['f4_trace_J_mean']))
    check("T3.6 — f5_delta_D_mean fini", is_finite(f['f5_delta_D_mean']))
    check("T3.7 — f7_dmd_spectral_radius fini ou NaN (pas crash)",
          is_finite(f['f7_dmd_spectral_radius']) or is_nan(f['f7_dmd_spectral_radius']))
    # rank=2 → mode1 NaN attendu
    check("T3.8 — f3_mode1 NaN pour rank=2",
          is_nan(f['f3_entanglement_entropy_mode1_mean']),
          f"got {f['f3_entanglement_entropy_mode1_mean']}")
except Exception as e:
    check("T3 — run_one_jax rank=2", False, str(e))

# =============================================================================
# T4 — run_one_jax rank=3
# =============================================================================

print("\n=== T4 — run_one_jax rank=3 ===")
try:
    gamma_r3 = lambda s, p, k: s * p.get('beta', jnp.array(0.99))
    f_r3 = run_one_jax(
        gamma_r3, {'beta': jnp.array(0.99)},
        mod_fn, {},
        state_r3, key, 30,
        is_differentiable=True,
    )
    check("T4.1 — 68 features", len(f_r3) == 68, f"got {len(f_r3)}")
    check("T4.2 — f3_mode1 non-NaN pour rank=3",
          is_finite(f_r3['f3_entanglement_entropy_mode1_mean']),
          f"got {f_r3['f3_entanglement_entropy_mode1_mean']:.4f}")
    check("T4.3 — f3_mode_asymmetry fini", is_finite(f_r3['f3_mode_asymmetry_mean']))
    check("T4.4 — f1_effective_rank fini", is_finite(f_r3['f1_effective_rank_mean']))
except Exception as e:
    check("T4 — run_one_jax rank=3", False, str(e))

# =============================================================================
# T5 — gamma non-différentiable : NaN Hutchinson, Lyapunov non-NaN
# =============================================================================

print("\n=== T5 — gamma non-différentiable ===")
try:
    f_nd = run_one_jax(
        gamma_nondiff, {},
        mod_fn, {},
        state_r2, key, 30,
        is_differentiable=False,
    )
    hutch_keys = [
        'f4_trace_J_mean', 'f4_trace_J_std', 'f4_trace_J_final',
        'f4_jvp_norm_mean', 'f4_jvp_norm_final',
        'f4_jacobian_asymmetry_mean', 'f4_jacobian_asymmetry_final',
        'f4_local_lyapunov_mean', 'f4_local_lyapunov_std',
    ]
    all_nan = all(is_nan(f_nd[k]) for k in hutch_keys)
    check("T5.1 — Hutchinson features NaN", all_nan,
          f"non-NaN: {[k for k in hutch_keys if not is_nan(f_nd[k])]}")
    check("T5.2 — lyapunov_empirical_mean non-NaN",
          is_finite(f_nd['f4_lyapunov_empirical_mean']),
          f"got {f_nd['f4_lyapunov_empirical_mean']:.4f}")
    check("T5.3 — lyapunov_empirical_std non-NaN",
          is_finite(f_nd['f4_lyapunov_empirical_std']))
    check("T5.4 — F1 features finies (non touchées)",
          is_finite(f_nd['f1_effective_rank_mean']))
except Exception as e:
    check("T5 — gamma non-différentiable", False, str(e))

# =============================================================================
# T6 — gamma différentiable : Hutchinson non-NaN
# =============================================================================

print("\n=== T6 — gamma différentiable, Hutchinson non-NaN ===")
try:
    f_d = run_one_jax(
        gamma_diff, {'beta': jnp.array(0.99)},
        mod_fn, {},
        state_r2, key, 30,
        is_differentiable=True,
    )
    check("T6.1 — trace_J_mean non-NaN", is_finite(f_d['f4_trace_J_mean']),
          f"got {f_d['f4_trace_J_mean']:.4f}")
    check("T6.2 — jvp_norm_mean non-NaN", is_finite(f_d['f4_jvp_norm_mean']))
    check("T6.3 — jacobian_asymmetry_mean non-NaN", is_finite(f_d['f4_jacobian_asymmetry_mean']))
    check("T6.4 — local_lyapunov_mean non-NaN", is_finite(f_d['f4_local_lyapunov_mean']))
except Exception as e:
    check("T6 — gamma différentiable Hutchinson", False, str(e))

# =============================================================================
# T7 — cohérence FEATURE_NAMES exhaustif
# =============================================================================

print("\n=== T7 — cohérence FEATURE_NAMES ===")
families = {
    'F1': [k for k in FEATURE_NAMES if k.startswith('f1_')],
    'F2': [k for k in FEATURE_NAMES if k.startswith('f2_')],
    'F3': [k for k in FEATURE_NAMES if k.startswith('f3_')],
    'F4': [k for k in FEATURE_NAMES if k.startswith('f4_')],
    'F5': [k for k in FEATURE_NAMES if k.startswith('f5_')],
    'F6': [k for k in FEATURE_NAMES if k.startswith('f6_')],
    'F7': [k for k in FEATURE_NAMES if k.startswith('f7_')],
    'PS': [k for k in FEATURE_NAMES if k.startswith('ps_')],
    'HE': [k for k in FEATURE_NAMES if k.startswith('health_')],
}
for fam, keys in families.items():
    check(f"T7 — {fam} présente ({len(keys)} clés)", len(keys) > 0)

# =============================================================================
# Résumé
# =============================================================================

print("\n" + "=" * 56)
n_pass = sum(1 for _, ok in results if ok)
n_fail = sum(1 for _, ok in results if not ok)
print(f"  TOTAL : {n_pass}/{len(results)} tests passés")
if n_fail:
    print(f"  ÉCHECS :")
    for name, ok in results:
        if not ok:
            print(f"    ✗ {name}")
print("=" * 56)

sys.exit(0 if n_fail == 0 else 1)