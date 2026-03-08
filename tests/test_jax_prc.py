"""
test_jax_prc.py — Validation JAX machine pour PRC v7
Emplacement : prc_framework/tests/test_jax_prc.py
Exécution   : python -m tests.test_jax_prc  (depuis prc_framework/)

Six validations des primitives JAX utilisées dans le pipeline v7.
Aucune dépendance PRC — teste uniquement l'outillage.
"""

import time
import sys
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(label: str, ok: bool, details: str) -> str:
    status = "✓" if ok else "✗"
    return f"  {status}  {label:<30} {details}"


# ---------------------------------------------------------------------------
# V1 — Import et device
# ---------------------------------------------------------------------------

def v1_import_device():
    try:
        import jax
        devices = jax.devices()
        backend  = devices[0].platform
        name     = str(devices[0])
        details  = f"backend: {backend} | device: {name} | jax {jax.__version__}"
        return True, details
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# V2 — lax.scan + jit
# ---------------------------------------------------------------------------

def v2_lax_scan_jit():
    try:
        import jax
        import jax.numpy as jnp
        from jax import lax

        key   = jax.random.PRNGKey(0)
        state = jax.random.normal(key, (100, 100))

        def step_fn(carry, _t):
            # Multiplication matrice + amortissement pour éviter l'explosion
            next_carry = jnp.dot(carry, carry) * 0.01 + carry * 0.999
            return next_carry, jnp.linalg.norm(next_carry)

        @jax.jit
        def run_scan(s):
            final, norms = lax.scan(step_fn, s, None, length=1000)
            return final, norms

        # Premier appel : compilation JIT
        t0 = time.perf_counter()
        out1, norms1 = run_scan(state)
        out1.block_until_ready()
        t_compile = time.perf_counter() - t0

        # Deuxième appel : exécution compilée
        t0 = time.perf_counter()
        out2, norms2 = run_scan(state)
        out2.block_until_ready()
        t_exec = time.perf_counter() - t0

        gain = t_compile / t_exec if t_exec > 0 else float("inf")

        # Sanity : norms non vides, pas tous NaN
        assert norms1.shape == (1000,), f"shape inattendue: {norms1.shape}"
        import jax.numpy as jnp
        assert not jnp.all(jnp.isnan(norms1)), "toutes les normes sont NaN"

        details = (
            f"compile: {t_compile:.2f}s | "
            f"exec: {t_exec:.4f}s | "
            f"gain: {gain:.0f}x"
        )
        return True, details
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# V3 — jnp.linalg.svd mode-0 unfolding
# ---------------------------------------------------------------------------

def v3_svd_mode0():
    try:
        import jax
        import jax.numpy as jnp

        shapes  = [(10, 10), (50, 50), (100, 100)]
        timings = []

        for shape in shapes:
            key   = jax.random.PRNGKey(1)
            state = jax.random.normal(key, shape)

            # Convention universelle roadmap section 3
            M      = state.reshape(state.shape[0], -1)
            t0     = time.perf_counter()
            sigmas = jnp.linalg.svd(M, compute_uv=False)
            sigmas.block_until_ready()
            elapsed = (time.perf_counter() - t0) * 1000  # ms

            # Vérifications
            assert sigmas.shape == (min(shape),), \
                f"shape sigmas inattendue: {sigmas.shape}"
            diffs = jnp.diff(sigmas)
            assert jnp.all(diffs <= 1e-5), \
                "valeurs singulières non décroissantes"
            assert jnp.all(sigmas >= -1e-6), \
                "valeurs singulières négatives"

            timings.append(f"{shape[0]}x{shape[1]}: {elapsed:.2f}ms")

        details = " | ".join(timings)
        return True, details
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# V4 — jax.jvp
# ---------------------------------------------------------------------------

def v4_jvp():
    try:
        import jax
        import jax.numpy as jnp

        key = jax.random.PRNGKey(2)
        W   = jax.random.normal(key, (10, 10))
        x   = jax.random.normal(jax.random.PRNGKey(3), (10,))
        v   = jax.random.normal(jax.random.PRNGKey(4), (10,))  # tangent

        def f(x_):
            return jnp.dot(W, x_)

        t0 = time.perf_counter()
        primals, tangents = jax.jvp(f, (x,), (v,))
        jax.block_until_ready((primals, tangents))
        elapsed = (time.perf_counter() - t0) * 1000

        assert primals.shape  == (10,), f"primals shape: {primals.shape}"
        assert tangents.shape == (10,), f"tangents shape: {tangents.shape}"

        # Vérification analytique : J*v = W @ v
        expected = jnp.dot(W, v)
        err = jnp.max(jnp.abs(tangents - expected)).item()
        assert err < 1e-5, f"erreur JVP: {err:.2e}"

        details = f"elapsed: {elapsed:.2f}ms | erreur vs analytique: {err:.2e}"
        return True, details
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# V5 — PRNG : PRNGKey + split dans scan
# ---------------------------------------------------------------------------

def v5_prng_scan():
    try:
        import jax
        import jax.numpy as jnp
        from jax import lax

        def run_with_seed(seed: int):
            key = jax.random.PRNGKey(seed)

            def step_fn(carry, _t):
                key = carry
                key, subkey = jax.random.split(key)
                sample = jax.random.normal(subkey, ())
                return key, sample

            _, samples = lax.scan(step_fn, key, None, length=10)
            return samples

        run_jit = jax.jit(run_with_seed)

        samples_a = run_jit(42)
        samples_a.block_until_ready()

        # Vérifier que toutes les valeurs sont distinctes
        n_unique = jnp.unique(samples_a).shape[0]
        assert n_unique == 10, f"valeurs non distinctes: {n_unique}/10 uniques"

        # Vérifier reproductibilité
        samples_b = run_jit(42)
        reproductible = jnp.allclose(samples_a, samples_b).item()
        assert reproductible, "résultats non reproductibles avec même seed"

        # Vérifier que seed différent → résultat différent
        samples_c  = run_jit(99)
        different  = not jnp.allclose(samples_a, samples_c).item()
        assert different, "seeds différents produisent le même résultat"

        details = (
            f"10 valeurs distinctes ✓ | "
            f"reproductible: {reproductible} | "
            f"seeds distincts: {different}"
        )
        return True, details
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# V6 — vmap sur 5 seeds simultanés
# ---------------------------------------------------------------------------

def v6_vmap_seeds():
    try:
        import jax
        import jax.numpy as jnp
        from jax import lax

        N_SEEDS = 5
        T       = 100
        SHAPE   = (20, 20)

        def run_single(key, sigma):
            state = jax.random.normal(key, SHAPE) * sigma

            def step_fn(carry, _t):
                key, s = carry
                key, subkey = jax.random.split(key)
                noise  = jax.random.normal(subkey, SHAPE) * 0.01
                s_next = s * 0.999 + noise
                return (key, s_next), jnp.linalg.norm(s_next)

            _, norms = lax.scan(step_fn, (key, state), None, length=T)
            return norms

        keys   = jax.random.split(jax.random.PRNGKey(0), N_SEEDS)
        sigmas = jnp.array([0.5, 0.7, 1.0, 1.2, 1.5])

        run_single_jit  = jax.jit(run_single)
        run_batch       = jax.jit(jax.vmap(run_single, in_axes=(0, 0)))

        # Séquentiel
        t0 = time.perf_counter()
        seq_results = []
        for i in range(N_SEEDS):
            r = run_single_jit(keys[i], sigmas[i])
            r.block_until_ready()
            seq_results.append(r)
        t_seq = time.perf_counter() - t0

        # vmap (premier appel = compile)
        _ = run_batch(keys, sigmas)  # warmup compile
        t0 = time.perf_counter()
        vmap_results = run_batch(keys, sigmas)
        vmap_results.block_until_ready()
        t_vmap = time.perf_counter() - t0

        gain = t_seq / t_vmap if t_vmap > 0 else float("inf")

        # Vérifier que les 5 outputs sont distincts entre seeds
        all_distinct = True
        for i in range(N_SEEDS - 1):
            if jnp.allclose(vmap_results[i], vmap_results[i + 1]):
                all_distinct = False
                break

        assert all_distinct, "vmap: seeds différents produisent des outputs identiques"

        details = (
            f"seq: {t_seq:.3f}s | "
            f"vmap: {t_vmap:.4f}s | "
            f"gain: {gain:.1f}x | "
            f"outputs distincts: {all_distinct}"
        )
        return True, details
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 65)
    print("  PRC v7 — Validation JAX machine")
    print("=" * 65)
    print()

    validations = [
        ("V1  import + device    ", v1_import_device),
        ("V2  lax.scan + jit     ", v2_lax_scan_jit),
        ("V3  svd mode-0         ", v3_svd_mode0),
        ("V4  jvp                ", v4_jvp),
        ("V5  prng split scan    ", v5_prng_scan),
        ("V6  vmap 5 seeds       ", v6_vmap_seeds),
    ]

    results = []
    for label, fn in validations:
        ok, details = fn()
        line = _fmt(label, ok, details)
        print(line)
        results.append(ok)

    print()
    print("=" * 65)
    n_ok   = sum(results)
    n_fail = len(results) - n_ok
    if n_fail == 0:
        print(f"  Résultat : {n_ok}/{len(results)} validations ✓ — JAX opérationnel pour PRC v7")
    else:
        print(f"  Résultat : {n_ok}/{len(results)} ✓  {n_fail} ✗ — corriger avant de continuer")
    print("=" * 65)
    print()

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()