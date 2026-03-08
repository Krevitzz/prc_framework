"""
tests/test_run_one_jax.py

Validation run_one_jax : bloc compilé central du pipeline JAX v7.
Exécution : python -m tests.test_run_one_jax  (depuis prc_framework/)

Six validations :
  T1 — retourne exactement FEATURE_NAMES comme clés
  T2 — toutes valeurs sont des floats Python purs
  T3 — JIT : 2ème appel → cache hit (t_exec << t_compile)
  T4 — vmap sur 3 keys → 3 résultats distincts
  T5 — contraction cohérente : GAM-011 scale=0.5 → frob décroissante
  T6 — NaN passthrough : gamma NaN → pas de crash, NaN dans features
"""

import sys
import time
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helper affichage
# ---------------------------------------------------------------------------

def _fmt(label, ok, details):
    status = "✓" if ok else "✗"
    return f"  {status}  {label:<35} {details}"


# ---------------------------------------------------------------------------
# Construction composition de base
# ---------------------------------------------------------------------------

def _build_composition(n_dof=10, scale=1.0, max_it=50, seed=0):
    """
    Retourne (gamma_fn, gamma_params, modifier_fn, modifier_params,
              D_initial, key, max_it) prêt pour run_one_jax.
    """
    import jax
    from atomics.operators.gam_011_linear_tensordot import apply as gamma_fn
    from atomics.operators.gam_011_linear_tensordot import generate_W
    from atomics.D_encodings.sym_001_identity import create as enc_create
    from atomics.modifiers.m0_baseline import apply as modifier_fn

    key    = jax.random.PRNGKey(seed)
    key_W, key_run = jax.random.split(key)

    W      = generate_W(n_dof=n_dof, scale=scale, key=key_W)
    D      = enc_create(n_dof=n_dof, params={}, key=key_run)

    return (
        gamma_fn,
        {'W': W},
        modifier_fn,
        {},
        D,
        key_run,
        max_it,
    )


# ---------------------------------------------------------------------------
# T1 — retourne exactement FEATURE_NAMES
# ---------------------------------------------------------------------------

def t1_feature_names():
    try:
        from running.run_one_jax import run_one_jax
        from featuring.hub_featuring import FEATURE_NAMES

        args   = _build_composition()
        result = run_one_jax(*args)

        assert set(result.keys()) == set(FEATURE_NAMES), (
            f"clés reçues: {set(result.keys())} "
            f"attendues: {set(FEATURE_NAMES)}"
        )
        return True, f"clés: {sorted(result.keys())}"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T2 — toutes valeurs floats Python purs
# ---------------------------------------------------------------------------

def t2_floats_python():
    try:
        from running.run_one_jax import run_one_jax

        args   = _build_composition()
        result = run_one_jax(*args)

        non_float = {
            k: type(v).__name__
            for k, v in result.items()
            if not isinstance(v, float)
        }
        assert not non_float, f"non-floats: {non_float}"

        vals = list(result.values())
        return True, f"{len(vals)} floats Python ✓ — ex: {vals[0]:.4f}"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T3 — JIT cache hit
# ---------------------------------------------------------------------------

def t3_jit_cache_hit():
    try:
        from running.run_one_jax import run_one_jax

        args = _build_composition(n_dof=20, max_it=100)

        # Premier appel — compile
        t0 = time.perf_counter()
        run_one_jax(*args)
        t_compile = time.perf_counter() - t0

        # Deuxième appel — cache hit
        t0 = time.perf_counter()
        run_one_jax(*args)
        t_exec = time.perf_counter() - t0

        # Cache hit = t_exec significativement plus rapide
        # On vérifie juste que le 2ème appel ne recompile pas
        # (pas de ratio fixe — dépend de la machine)
        assert t_exec < t_compile, (
            f"2ème appel plus lent que 1er : {t_exec:.3f}s > {t_compile:.3f}s"
        )

        ratio = t_compile / t_exec if t_exec > 0 else float('inf')
        return True, (
            f"compile: {t_compile:.3f}s | "
            f"exec: {t_exec:.4f}s | "
            f"gain: {ratio:.1f}x"
        )
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T4 — vmap sur gamma_params distincts (3 scales)
# ---------------------------------------------------------------------------

def t4_vmap_3_params():
    try:
        import jax
        import jax.numpy as jnp
        from running.run_one_jax import _run_jit
        from featuring.hub_featuring import post_scan

        n_dof   = 10
        max_it  = 50
        n_batch = 3
        scales  = [0.3, 1.0, 2.5]   # scales distincts → W distincts → résultats distincts

        from atomics.operators.gam_011_linear_tensordot import apply as gamma_fn
        from atomics.operators.gam_011_linear_tensordot import generate_W
        from atomics.D_encodings.sym_001_identity import create as enc_create
        from atomics.modifiers.m0_baseline import apply as modifier_fn

        master_key = jax.random.PRNGKey(0)
        key        = jax.random.PRNGKey(42)
        D          = enc_create(n_dof=n_dof, params={}, key=master_key)

        # Générer 3 W distincts (un par scale)
        W_batch = jnp.stack([
            generate_W(n_dof=n_dof, scale=s, key=jax.random.PRNGKey(i))
            for i, s in enumerate(scales)
        ])  # (3, n_dof, n_dof)

        # vmap sur gamma_params['W'] — in_axes sur W_batch(1)
        # gamma_params est un dict → in_axes dict avec même structure
        run_batch = jax.vmap(
            _run_jit,
            in_axes=(None, {'W': 0}, None, None, None, None, None)
        )

        signals_batch, last_states = run_batch(
            gamma_fn,
            {'W': W_batch},
            modifier_fn,
            {},
            D,
            key,
            max_it,
        )

        # post_scan sur chaque résultat du batch
        results = [
            post_scan(
                {k: signals_batch[k][i] for k in signals_batch},
                last_states[i]
            )
            for i in range(n_batch)
        ]

        # Vérifier que les 3 résultats sont distincts
        finals = [r['frob_norm_final'] for r in results]
        all_distinct = len(set(f"{v:.6f}" for v in finals)) == n_batch
        assert all_distinct, f"résultats non distincts: {finals}"

        return True, (
            f"scales={scales} | "
            f"finals: {[f'{v:.4f}' for v in finals]}"
        )
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T5 — contraction cohérente
# ---------------------------------------------------------------------------

def t5_contraction_coherente():
    try:
        from running.run_one_jax import run_one_jax

        # scale=0.5 → norme décroissante → frob_norm_final < frob_norm_mean
        args   = _build_composition(n_dof=15, scale=0.5, max_it=100)
        result = run_one_jax(*args)

        final = result['frob_norm_final']
        mean  = result['frob_norm_mean']

        assert final < mean, (
            f"scale=0.5 attendu final < mean, "
            f"reçu final={final:.4f} mean={mean:.4f}"
        )

        return True, (
            f"scale=0.5 | "
            f"mean={mean:.4f} | "
            f"final={final:.4f} ↓"
        )
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T6 — NaN passthrough
# ---------------------------------------------------------------------------

def t6_nan_passthrough():
    try:
        import jax
        import jax.numpy as jnp
        import math
        from running.run_one_jax import run_one_jax
        from atomics.D_encodings.sym_001_identity import create as enc_create
        from atomics.modifiers.m0_baseline import apply as modifier_fn

        # Gamma qui injecte NaN dès le premier pas
        def gamma_nan(state, params, key):
            return jnp.full_like(state, jnp.nan)

        n_dof = 8
        key   = jax.random.PRNGKey(0)
        D     = enc_create(n_dof=n_dof, params={}, key=key)

        result = run_one_jax(
            gamma_nan, {},
            modifier_fn, {},
            D, key, 20,
        )

        # Pas de crash — NaN dans les features
        assert math.isnan(result['frob_norm_final']), (
            f"attendu NaN, reçu {result['frob_norm_final']}"
        )
        assert math.isnan(result['frob_norm_mean']), (
            f"attendu NaN mean, reçu {result['frob_norm_mean']}"
        )

        return True, "gamma NaN → pas de crash | features=NaN ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 70)
    print("  PRC v7 — test_run_one_jax : bloc compilé central")
    print("=" * 70)
    print()

    validations = [
        ("T1  feature names            ", t1_feature_names),
        ("T2  floats Python purs       ", t2_floats_python),
        ("T3  JIT cache hit            ", t3_jit_cache_hit),
        ("T4  vmap 3 params            ", t4_vmap_3_params),
        ("T5  contraction cohérente    ", t5_contraction_coherente),
        ("T6  NaN passthrough          ", t6_nan_passthrough),
    ]

    results = []
    for label, fn in validations:
        ok, details = fn()
        print(_fmt(label, ok, details))
        results.append(ok)

    print()
    print("=" * 70)
    n_ok   = sum(results)
    n_fail = len(results) - n_ok
    if n_fail == 0:
        print(f"  Résultat : {n_ok}/{len(results)} ✓ — run_one_jax opérationnel")
    else:
        print(f"  Résultat : {n_ok}/{len(results)} ✓  {n_fail} ✗ — corriger avant de continuer")
    print("=" * 70)
    print()

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == '__main__':
    main()