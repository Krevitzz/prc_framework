"""
tests/test_atomics_p1.py

Validation GAM-011 : transformation linéaire tensordot JAX.
Exécution : python -m tests.test_atomics_p1  (depuis prc_framework/)

Neuf validations couvrant :
  - METADATA structure
  - generate_W : shape, rayon spectral, reproductibilité
  - apply : rang 2, rang 3, rang 4
  - différentiabilité jvp
  - comportement scale contraction/expansion
"""

import sys
import time
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helper affichage
# ---------------------------------------------------------------------------

def _fmt(label: str, ok: bool, details: str) -> str:
    status = "✓" if ok else "✗"
    return f"  {status}  {label:<35} {details}"


# ---------------------------------------------------------------------------
# Import atomic — chemin relatif prc_framework/
# ---------------------------------------------------------------------------

def _import_gam011():
    try:
        from atomics.operators import gam_011_linear_tensordot as gam011
        return gam011, None
    except ImportError as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# T1 — METADATA structure
# ---------------------------------------------------------------------------

def t1_metadata(gam011):
    try:
        required = {
            'id'              : str,
            'family'          : str,
            'rank_constraint' : (type(None),),
            'differentiable'  : bool,
            'stochastic'      : bool,
            'non_markovian'   : bool,
        }
        md = gam011.METADATA
        errors = []

        for key, expected_type in required.items():
            if key not in md:
                errors.append(f"clé manquante: {key}")
                continue
            if isinstance(expected_type, tuple):
                if not isinstance(md[key], expected_type):
                    errors.append(f"{key}: type inattendu {type(md[key])}")
            else:
                if not isinstance(md[key], expected_type):
                    errors.append(f"{key}: attendu {expected_type}, reçu {type(md[key])}")

        assert md['id'] == 'GAM-011', f"id incorrect: {md['id']}"
        assert md['family'] == 'markovian', f"family incorrect: {md['family']}"
        assert md['differentiable'] is True
        assert md['stochastic'] is False
        assert md['non_markovian'] is False
        assert md['rank_constraint'] is None

        if errors:
            return False, " | ".join(errors)
        return True, f"id={md['id']} family={md['family']} diff={md['differentiable']}"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T2 — generate_W shape
# ---------------------------------------------------------------------------

def t2_generate_W_shape(gam011):
    try:
        import jax
        for n_dof in [5, 10, 50]:
            key = jax.random.PRNGKey(0)
            W = gam011.generate_W(n_dof=n_dof, scale=1.0, key=key)
            assert W.shape == (n_dof, n_dof), \
                f"shape attendue ({n_dof},{n_dof}), reçue {W.shape}"
        return True, "shapes (5,5) (10,10) (50,50) ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T3 — generate_W rayon spectral ≈ scale
# ---------------------------------------------------------------------------

def t3_generate_W_spectral_radius(gam011):
    try:
        import jax
        import jax.numpy as jnp

        results = []
        for scale in [0.5, 1.0, 2.0]:
            key = jax.random.PRNGKey(7)
            W = gam011.generate_W(n_dof=20, scale=scale, key=key)
            eigvals = jnp.linalg.eigvals(W)
            rho = jnp.max(jnp.abs(eigvals)).item()
            err = abs(rho - scale) / scale
            assert err < 0.01, \
                f"scale={scale} → ρ={rho:.4f}, erreur relative {err:.2%} > 1%"
            results.append(f"scale={scale}→ρ={rho:.4f}")

        return True, " | ".join(results)
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T4 — apply rang 2 : tensordot == matmul
# ---------------------------------------------------------------------------

def t4_apply_rank2(gam011):
    try:
        import jax
        import jax.numpy as jnp

        key  = jax.random.PRNGKey(1)
        n    = 10
        W    = gam011.generate_W(n_dof=n, scale=1.0, key=key)
        state = jax.random.normal(jax.random.PRNGKey(2), (n, n))

        params  = {'W': W}
        out_gam = gam011.apply(state, params, key)
        out_ref = W @ state  # équivalence rang 2

        assert out_gam.shape == (n, n), f"shape: {out_gam.shape}"
        err = jnp.max(jnp.abs(out_gam - out_ref)).item()
        assert err < 1e-5, f"écart vs W@T: {err:.2e}"

        return True, f"shape ({n},{n}) ✓ | écart vs W@T: {err:.2e}"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T5 — apply rang 3 : vérif slice analytique
# ---------------------------------------------------------------------------

def t5_apply_rank3(gam011):
    try:
        import jax
        import jax.numpy as jnp

        key   = jax.random.PRNGKey(3)
        n     = 8
        W     = gam011.generate_W(n_dof=n, scale=1.0, key=key)
        state = jax.random.normal(jax.random.PRNGKey(4), (n, n, n))

        params = {'W': W}
        out    = gam011.apply(state, params, key)

        assert out.shape == (n, n, n), f"shape: {out.shape}"

        # Vérification analytique sur un slice :
        # out[i,j,k] = Σ_l W[i,l] * state[l,j,k]
        i, j, k = 2, 3, 1
        expected = jnp.sum(W[i, :] * state[:, j, k]).item()
        got      = out[i, j, k].item()
        err      = abs(got - expected)
        assert err < 1e-5, f"slice [{i},{j},{k}]: attendu {expected:.6f}, reçu {got:.6f}"

        return True, f"shape ({n},{n},{n}) ✓ | slice analytique erreur: {err:.2e}"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T6 — apply rang 4 : shape correcte
# ---------------------------------------------------------------------------

def t6_apply_rank4(gam011):
    try:
        import jax
        import jax.numpy as jnp

        key   = jax.random.PRNGKey(5)
        n     = 5
        W     = gam011.generate_W(n_dof=n, scale=1.0, key=key)
        state = jax.random.normal(jax.random.PRNGKey(6), (n, n, n, n))

        params = {'W': W}
        out    = gam011.apply(state, params, key)

        assert out.shape == (n, n, n, n), f"shape: {out.shape}"

        # Vérif slice rang 4
        i, j, k, l = 1, 2, 3, 0
        expected = jnp.sum(W[i, :] * state[:, j, k, l]).item()
        got      = out[i, j, k, l].item()
        err      = abs(got - expected)
        assert err < 1e-5, f"slice erreur: {err:.2e}"

        return True, f"shape ({n},{n},{n},{n}) ✓ | slice analytique erreur: {err:.2e}"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T7 — différentiabilité jvp
# ---------------------------------------------------------------------------

def t7_differentiable(gam011):
    try:
        import jax
        import jax.numpy as jnp

        key     = jax.random.PRNGKey(8)
        n       = 10
        W       = gam011.generate_W(n_dof=n, scale=1.0, key=key)
        state   = jax.random.normal(jax.random.PRNGKey(9), (n, n))
        tangent = jax.random.normal(jax.random.PRNGKey(10), (n, n))
        params  = {'W': W}

        def apply_fn(s):
            return gam011.apply(s, params, key)

        primals, tangents_out = jax.jvp(apply_fn, (state,), (tangent,))

        # Analytique : J·v = W @ v  (tensordot linéaire)
        expected_tangents = W @ tangent
        err = jnp.max(jnp.abs(tangents_out - expected_tangents)).item()
        assert err < 1e-5, f"erreur JVP vs analytique: {err:.2e}"

        # trace_J analytique = trace(W)
        trace_W   = jnp.trace(W).item()
        # Hutchinson approximation avec v=tangent (single sample, juste un check)
        trace_est = jnp.sum(tangent * tangents_out).item() / jnp.sum(tangent**2).item() * n**2
        # Note : Hutchinson est un estimateur, on vérifie juste que le JVP fonctionne
        assert primals.shape == (n, n)
        assert tangents_out.shape == (n, n)

        return True, f"JVP erreur vs analytique: {err:.2e} | trace(W)={trace_W:.4f}"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T8 — comportement scale : contraction / expansion
# ---------------------------------------------------------------------------

def t8_scale_behaviour(gam011):
    try:
        import jax
        import jax.numpy as jnp

        n     = 20
        state = jax.random.normal(jax.random.PRNGKey(11), (n, n))
        norm0 = jnp.linalg.norm(state).item()

        results = []
        for scale, expected in [(0.5, 'contraction'), (2.0, 'expansion')]:
            key = jax.random.PRNGKey(12)
            W   = gam011.generate_W(n_dof=n, scale=scale, key=key)
            out = gam011.apply(state, {'W': W}, key)
            norm1 = jnp.linalg.norm(out).item()

            if expected == 'contraction':
                assert norm1 < norm0, \
                    f"scale={scale}: attendu norm↓, reçu {norm0:.3f}→{norm1:.3f}"
            else:
                assert norm1 > norm0, \
                    f"scale={scale}: attendu norm↑, reçu {norm0:.3f}→{norm1:.3f}"

            results.append(f"scale={scale}: {norm0:.3f}→{norm1:.3f} ({expected} ✓)")

        return True, " | ".join(results)
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T9 — reproductibilité generate_W
# ---------------------------------------------------------------------------

def t9_reproductibilite(gam011):
    try:
        import jax
        import jax.numpy as jnp

        n = 15

        # Même key → même W
        key = jax.random.PRNGKey(42)
        W_a = gam011.generate_W(n_dof=n, scale=1.0, key=key)
        W_b = gam011.generate_W(n_dof=n, scale=1.0, key=key)
        same = jnp.allclose(W_a, W_b).item()
        assert same, "même key → W différents"

        # Key différente → W différent
        key2 = jax.random.PRNGKey(99)
        W_c  = gam011.generate_W(n_dof=n, scale=1.0, key=key2)
        diff = not jnp.allclose(W_a, W_c).item()
        assert diff, "keys différentes → W identiques"

        return True, f"reproductible: {same} | seeds distincts: {diff}"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# Import atomic SYM-001
# ---------------------------------------------------------------------------

def _import_sym001():
    try:
        from atomics.D_encodings import sym_001_identity as sym001
        return sym001, None
    except ImportError as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# T10 — SYM-001 METADATA structure
# ---------------------------------------------------------------------------

def t10_sym001_metadata(sym001):
    try:
        required = {
            'id'         : str,
            'rank'       : int,
            'stochastic' : bool,
        }
        md = sym001.METADATA
        for key, expected_type in required.items():
            if key not in md:
                raise AssertionError(f"clé manquante: {key}")
            if not isinstance(md[key], expected_type):
                raise AssertionError(
                    f"{key}: attendu {expected_type}, reçu {type(md[key])}"
                )

        assert md['id'] == 'SYM-001', f"id incorrect: {md['id']}"
        assert md['rank'] == 2,       f"rank incorrect: {md['rank']}"
        assert md['stochastic'] is False

        return True, f"id={md['id']} rank={md['rank']} stochastic={md['stochastic']}"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T11 — SYM-001 create shape
# ---------------------------------------------------------------------------

def t11_sym001_create_shape(sym001):
    try:
        import jax

        for n_dof in [5, 10, 50]:
            key = jax.random.PRNGKey(0)
            out = sym001.create(n_dof=n_dof, params={}, key=key)
            assert out.shape == (n_dof, n_dof), \
                f"shape attendue ({n_dof},{n_dof}), reçue {out.shape}"

        return True, "shapes (5,5) (10,10) (50,50) ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T12 — SYM-001 est bien la matrice identité
# ---------------------------------------------------------------------------

def t12_sym001_is_identity(sym001):
    try:
        import jax
        import jax.numpy as jnp

        for n_dof in [10, 30]:
            key = jax.random.PRNGKey(0)
            out = sym001.create(n_dof=n_dof, params={}, key=key)
            assert jnp.allclose(out, jnp.eye(n_dof)).item(), \
                f"n_dof={n_dof}: pas une matrice identité"

        return True, "I[i,j] = δ_ij ✓ pour n_dof=10 et 30"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T13 — SYM-001 propriétés spectrales
# ---------------------------------------------------------------------------

def t13_sym001_spectral(sym001):
    try:
        import jax
        import jax.numpy as jnp

        n_dof = 20
        key   = jax.random.PRNGKey(0)
        out   = sym001.create(n_dof=n_dof, params={}, key=key)

        # Toutes valeurs singulières == 1
        sigmas = jnp.linalg.svd(out, compute_uv=False)
        err_sigmas = jnp.max(jnp.abs(sigmas - 1.0)).item()
        assert err_sigmas < 1e-5, f"sigmas ≠ 1 : erreur max {err_sigmas:.2e}"

        # S_VN = log(n_dof) — entropie maximale uniforme
        eps   = 1e-8
        p_sq  = sigmas**2 / (jnp.sum(sigmas**2) + eps)
        s_vn  = -jnp.sum(p_sq * jnp.log(p_sq + eps)).item()
        s_ref = float(jnp.log(jnp.array(n_dof)).item())
        err_svn = abs(s_vn - s_ref)
        assert err_svn < 1e-4, f"S_VN={s_vn:.4f} ≠ log({n_dof})={s_ref:.4f}"

        return True, (
            f"sigmas=1 (err:{err_sigmas:.2e}) | "
            f"S_VN={s_vn:.4f} ≈ log({n_dof})={s_ref:.4f}"
        )
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# T14 — doublet GAM-011 × SYM-001
# ---------------------------------------------------------------------------

def t14_doublet_gam011_sym001(gam011, sym001):
    try:
        import jax
        import jax.numpy as jnp

        n_dof = 10
        key   = jax.random.PRNGKey(42)

        # D initial = identité
        D = sym001.create(n_dof=n_dof, params={}, key=key)

        # W contractant (scale=0.5)
        W      = gam011.generate_W(n_dof=n_dof, scale=0.5, key=key)
        params = {'W': W}

        # Pas 1 : W @ I = W
        T1  = gam011.apply(D, params, key)
        err_T1 = jnp.max(jnp.abs(T1 - W)).item()
        assert err_T1 < 1e-5, f"W @ I ≠ W : erreur {err_T1:.2e}"

        # Pas 2 : W @ W = W²
        T2      = gam011.apply(T1, params, key)
        W2      = W @ W
        err_T2  = jnp.max(jnp.abs(T2 - W2)).item()
        assert err_T2 < 1e-5, f"W @ W ≠ W² : erreur {err_T2:.2e}"

        # Contraction : ||T2|| < ||T1|| < ||D||
        norm_D  = jnp.linalg.norm(D).item()
        norm_T1 = jnp.linalg.norm(T1).item()
        norm_T2 = jnp.linalg.norm(T2).item()
        assert norm_T1 < norm_D,  f"||T1||={norm_T1:.3f} ≥ ||D||={norm_D:.3f}"
        assert norm_T2 < norm_T1, f"||T2||={norm_T2:.3f} ≥ ||T1||={norm_T1:.3f}"

        return True, (
            f"W@I=W (err:{err_T1:.2e}) | "
            f"W²=W@W (err:{err_T2:.2e}) | "
            f"norms {norm_D:.3f}→{norm_T1:.3f}→{norm_T2:.3f} ↓"
        )
    except Exception as e:
        return False, f"ERREUR: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 70)
    print("  PRC v7 — test_atomics_p1 : GAM-011 + SYM-001 + doublet")
    print("=" * 70)
    print()

    # Imports
    gam011, err = _import_gam011()
    if gam011 is None:
        print(f"  ✗  Import GAM-011 échoué : {err}")
        sys.exit(1)

    sym001, err = _import_sym001()
    if sym001 is None:
        print(f"  ✗  Import SYM-001 échoué : {err}")
        sys.exit(1)

    validations = [
        # GAM-011
        ("T1  GAM-011 METADATA         ", lambda: t1_metadata(gam011)),
        ("T2  GAM-011 generate_W shape ", lambda: t2_generate_W_shape(gam011)),
        ("T3  GAM-011 rayon spectral   ", lambda: t3_generate_W_spectral_radius(gam011)),
        ("T4  GAM-011 apply rang 2     ", lambda: t4_apply_rank2(gam011)),
        ("T5  GAM-011 apply rang 3     ", lambda: t5_apply_rank3(gam011)),
        ("T6  GAM-011 apply rang 4     ", lambda: t6_apply_rank4(gam011)),
        ("T7  GAM-011 jvp              ", lambda: t7_differentiable(gam011)),
        ("T8  GAM-011 scale contraction", lambda: t8_scale_behaviour(gam011)),
        ("T9  GAM-011 reproductibilité ", lambda: t9_reproductibilite(gam011)),
        # SYM-001
        ("T10 SYM-001 METADATA         ", lambda: t10_sym001_metadata(sym001)),
        ("T11 SYM-001 create shape     ", lambda: t11_sym001_create_shape(sym001)),
        ("T12 SYM-001 est identité     ", lambda: t12_sym001_is_identity(sym001)),
        ("T13 SYM-001 spectral         ", lambda: t13_sym001_spectral(sym001)),
        # Doublet
        ("T14 doublet GAM-011×SYM-001  ", lambda: t14_doublet_gam011_sym001(gam011, sym001)),
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
        print(f"  Résultat : {n_ok}/{len(results)} ✓ — paire 1 opérationnelle")
    else:
        print(f"  Résultat : {n_ok}/{len(results)} ✓  {n_fail} ✗ — corriger avant de continuer")
    print("=" * 70)
    print()

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == '__main__':
    main()