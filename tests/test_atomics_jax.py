"""
tests/test_atomics_jax.py

Validation atomics JAX v7 complets — inventaire 38 atomics.
Exécution : python -m tests.test_atomics_jax  (depuis prc_framework/)

BLOC A — SYM-*        T1  - T8
BLOC B — ASY-*        T9  - T14
BLOC C — RN-* + R3-*  T15 - T20
BLOC D — Modifiers    T21 - T23
BLOC E — Gammas       T24 - T38
BLOC F — Compositions T39 - T41
BLOC G — Pipeline     T42
"""

import sys
import warnings
import math

warnings.filterwarnings("ignore")


def _fmt(label, ok, details):
    status = "✓" if ok else "✗"
    return f"  {status}  {label:<44} {details}"


# ============================================================
# BLOC A — SYM-*
# ============================================================

def t1_sym001():
    try:
        import jax, jax.numpy as jnp
        from atomics.D_encodings.sym_001_identity import create, METADATA
        key = jax.random.PRNGKey(0)
        D   = create(8, {}, key)
        assert jnp.allclose(D, jnp.eye(8)).item()
        return True, "identité ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t2_sym002():
    try:
        import jax, jax.numpy as jnp
        from atomics.D_encodings.sym_002_random_uniform import create, METADATA
        assert METADATA['stochastic'] is True
        key = jax.random.PRNGKey(0)
        D   = create(10, {}, key)
        assert jnp.allclose(D, D.T).item()
        assert float(jnp.max(jnp.abs(D))) <= 1.0
        assert not jnp.allclose(D, create(10, {}, jax.random.PRNGKey(1))).item()
        return True, "symétrie ✓ | bornes [-1,1] ✓ | stochastique ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t3_sym003():
    try:
        import jax, jax.numpy as jnp
        from atomics.D_encodings.sym_003_random_gaussian import create
        key = jax.random.PRNGKey(0)
        D1  = create(10, {}, key)
        D2  = create(10, {'sigma': 1.0}, key)
        assert jnp.allclose(D1, D1.T).item()
        assert float(jnp.std(D2)) > float(jnp.std(D1))
        return True, f"symétrie ✓ | sigma scaling ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t4_sym004():
    try:
        import jax, jax.numpy as jnp
        from atomics.D_encodings.sym_004_correlation_matrix import create
        key = jax.random.PRNGKey(0)
        D   = create(10, {}, key)
        assert jnp.allclose(D, D.T, atol=1e-5).item()
        assert jnp.allclose(jnp.diag(D), jnp.ones(10), atol=1e-5).item()
        return True, "symétrie ✓ | diag=1 ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t5_sym005():
    try:
        import jax, jax.numpy as jnp
        from atomics.D_encodings.sym_005_banded import create, METADATA
        assert METADATA['stochastic'] is False
        key = jax.random.PRNGKey(0)
        D   = create(10, {'bandwidth': 2, 'amplitude': 0.5}, key)
        assert jnp.allclose(D, D.T).item()
        assert float(D[0, 5]) == 0.0   # hors bande
        assert float(D[0, 1]) != 0.0   # dans bande
        return True, "symétrie ✓ | bande ✓ | hors-bande=0 ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t6_sym006():
    try:
        import jax, jax.numpy as jnp
        from atomics.D_encodings.sym_006_block_hierarchical import create
        key   = jax.random.PRNGKey(0)
        D     = create(20, {'n_blocks': 4, 'intra': 0.7, 'inter': 0.1}, key)
        assert jnp.allclose(D, D.T, atol=1e-5).item()
        intra = float(jnp.mean(D[:5, :5]))
        inter = float(jnp.mean(D[:5, 15:]))
        assert intra > inter
        return True, f"blocs ✓ | intra={intra:.3f} > inter={inter:.3f} ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t7_sym007():
    try:
        import jax, jax.numpy as jnp
        from atomics.D_encodings.sym_007_uniform_correlation import create, METADATA
        assert METADATA['stochastic'] is False
        key  = jax.random.PRNGKey(0)
        rho  = 0.5
        D    = create(8, {'correlation': rho}, key)
        assert jnp.allclose(jnp.diag(D), jnp.ones(8)).item()
        mask = ~jnp.eye(8, dtype=bool)
        off  = float(jnp.mean(D[mask]))
        assert abs(off - rho) < 1e-5
        return True, f"diag=1 ✓ | off-diag={off:.4f}=ρ ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t8_sym008():
    try:
        import jax, jax.numpy as jnp
        from atomics.D_encodings.sym_008_random_clipped import create, METADATA
        assert METADATA['stochastic'] is True
        key = jax.random.PRNGKey(0)
        D   = create(10, {}, key)
        assert jnp.allclose(D, D.T, atol=1e-5).item()
        assert float(jnp.min(D)) >= -1.0 - 1e-6
        assert float(jnp.max(D)) <=  1.0 + 1e-6
        return True, "symétrie ✓ | clipping [-1,1] ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ============================================================
# BLOC B — ASY-*
# ============================================================

def t9_asy001():
    try:
        import jax, jax.numpy as jnp
        from atomics.D_encodings.asy_001_random_asymmetric import create
        key = jax.random.PRNGKey(0)
        D   = create(10, {}, key)
        assert not jnp.allclose(D, D.T).item()
        assert float(jnp.max(jnp.abs(D))) <= 1.0 + 1e-6
        return True, "asymétrie ✓ | bornes [-1,1] ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t10_asy002():
    try:
        import jax, jax.numpy as jnp
        from atomics.D_encodings.asy_002_lower_triangular import create
        key   = jax.random.PRNGKey(0)
        D     = create(8, {}, key)
        upper = jnp.triu(D, k=1)
        assert jnp.allclose(upper, jnp.zeros_like(upper)).item()
        return True, "triangulaire inf ✓ | upper=0 ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t11_asy003():
    try:
        import jax, jax.numpy as jnp
        from atomics.D_encodings.asy_003_antisymmetric import create
        key = jax.random.PRNGKey(0)
        D   = create(8, {}, key)
        assert jnp.allclose(D + D.T, jnp.zeros_like(D), atol=1e-6).item()
        assert jnp.allclose(jnp.diag(D), jnp.zeros(8), atol=1e-6).item()
        return True, "A+Aᵀ=0 ✓ | diag=0 ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t12_asy004():
    try:
        import jax, jax.numpy as jnp
        from atomics.D_encodings.asy_004_directional_gradient import create
        key = jax.random.PRNGKey(0)
        D   = create(10, {'gradient': 0.5, 'noise': 0.0}, key)
        assert float(D[0, 9]) < 0
        assert float(D[9, 0]) > 0
        return True, "gradient directionnel ✓ | D[0,9]<0, D[9,0]>0 ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t13_asy005():
    try:
        import jax, jax.numpy as jnp
        from atomics.D_encodings.asy_005_circulant import create, METADATA
        assert METADATA['stochastic'] is False
        key = jax.random.PRNGKey(0)
        D   = create(8, {}, key)
        assert not jnp.allclose(D, D.T).item()
        assert jnp.allclose(D[1], jnp.roll(D[0], 1)).item()
        return True, "circulante ✓ | asymétrie ✓ | roll ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t14_asy006():
    try:
        import jax, jax.numpy as jnp
        from atomics.D_encodings.asy_006_sparse import create
        key     = jax.random.PRNGKey(0)
        density = 0.3
        D       = create(20, {'density': density}, key)
        nnz     = float(jnp.sum(D != 0.0)) / (20 * 20)
        assert abs(nnz - density) < 0.15
        return True, f"sparse ✓ | densité≈{nnz:.3f} ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ============================================================
# BLOC C — RN-* + R3-*
# ============================================================

def t15_rn001():
    try:
        import jax, jax.numpy as jnp
        from atomics.D_encodings.rn_001_random_uniform import create
        key = jax.random.PRNGKey(0)
        for rank in [2, 3, 4]:
            D = create(5, {'rank': rank}, key)
            assert D.shape == (5,) * rank
            assert float(jnp.max(jnp.abs(D))) <= 1.0 + 1e-6
        return True, "rang 2,3,4 ✓ | shapes ✓ | bornes ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t16_rn002():
    try:
        import jax, jax.numpy as jnp
        from atomics.D_encodings.rn_002_partial_symmetric import create
        key = jax.random.PRNGKey(0)
        for rank in [2, 3]:
            D    = create(6, {'rank': rank}, key)
            perm = list(range(rank)); perm[0], perm[1] = 1, 0
            assert jnp.allclose(D, jnp.transpose(D, perm), atol=1e-6).item()
        return True, "rang 2,3 ✓ | symétrie axes 0,1 ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t17_rn003():
    try:
        import jax, jax.numpy as jnp
        from atomics.D_encodings.rn_003_diagonal import create, METADATA
        assert METADATA['stochastic'] is False
        key = jax.random.PRNGKey(0)
        for rank in [2, 3]:
            D = create(5, {'rank': rank}, key)
            for i in range(5):
                assert float(D[(i,) * rank]) == 1.0
            assert float(D[(0, 1) + (0,) * (rank - 2)]) == 0.0
        return True, "rang 2,3 ✓ | diagonale=1 ✓ | off-diag=0 ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t18_rn004():
    try:
        import jax, jax.numpy as jnp
        from atomics.D_encodings.rn_004_separable import create
        key = jax.random.PRNGKey(0)
        for rank in [2, 3]:
            D = create(5, {'rank': rank}, key)
            assert D.shape == (5,) * rank
        D2       = create(6, {'rank': 2}, key)
        rank_mat = int(jnp.linalg.matrix_rank(D2, tol=1e-4))
        assert rank_mat == 1
        return True, f"rang 2,3 ✓ | séparable → rang mat=1 ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t19_r3003():
    try:
        import jax, jax.numpy as jnp
        from atomics.D_encodings.r3_003_local_coupling import create, METADATA
        assert METADATA['rank'] == 3
        key = jax.random.PRNGKey(0)
        D   = create(10, {'radius': 2}, key)
        assert D.shape == (10, 10, 10)
        assert float(D[0, 9, 0]) == 0.0
        assert float(D[0, 1, 2]) != 0.0
        return True, "shape (10,10,10) ✓ | local coupling ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t20_r3007():
    try:
        import jax, jax.numpy as jnp
        from atomics.D_encodings.r3_007_block_structure import create, METADATA
        assert METADATA['rank'] == 3
        key   = jax.random.PRNGKey(0)
        D     = create(12, {'n_blocks': 3, 'intra': 0.8, 'inter': 0.05}, key)
        assert D.shape == (12, 12, 12)
        intra = float(jnp.mean(D[:4, :4, :4]))
        inter = float(jnp.mean(D[:4, 8:, :4]))
        assert intra > inter
        return True, f"shape (12,12,12) ✓ | intra={intra:.3f} > inter={inter:.3f} ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ============================================================
# BLOC D — Modifiers
# ============================================================

def t21_m0():
    try:
        import jax, jax.numpy as jnp
        from atomics.modifiers.m0_baseline import apply, METADATA
        assert METADATA['stochastic'] is False
        key   = jax.random.PRNGKey(0)
        state = jax.random.normal(key, (8, 8))
        assert jnp.allclose(apply(state, {}, key), state).item()
        return True, "identité ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t22_m1():
    try:
        import jax, jax.numpy as jnp
        from atomics.modifiers.m1_gaussian_noise import apply, METADATA
        assert METADATA['stochastic'] is True
        state  = jnp.zeros((8, 8))
        o1     = apply(state, {'sigma': 0.1}, jax.random.PRNGKey(0))
        o2     = apply(state, {'sigma': 0.1}, jax.random.PRNGKey(1))
        assert not jnp.allclose(o1, o2).item()
        o_zero = apply(state, {'sigma': 0.0}, jax.random.PRNGKey(0))
        assert jnp.allclose(o_zero, state).item()
        return True, f"stochastique ✓ | sigma=0 identité ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t23_m2():
    try:
        import jax, jax.numpy as jnp
        from atomics.modifiers.m2_uniform_noise import apply, METADATA
        assert METADATA['stochastic'] is True
        state = jnp.zeros((8, 8))
        amp   = 0.5
        out   = apply(state, {'amplitude': amp}, jax.random.PRNGKey(0))
        assert float(jnp.min(out)) >= -amp - 1e-6
        assert float(jnp.max(out)) <=  amp + 1e-6
        return True, f"bornes [-{amp},{amp}] ✓ | stochastique ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ============================================================
# BLOC E — Gammas (complets)
# ============================================================

def t24_gam001():
    try:
        import jax, jax.numpy as jnp
        from atomics.operators.gam_001_tanh import apply
        key = jax.random.PRNGKey(0)
        out = apply(jnp.ones((8, 8)) * 2.0, {'beta': 2.0}, key)
        assert float(jnp.max(jnp.abs(out))) < 1.0 + 1e-6
        out3 = apply(jnp.ones((6, 6, 6)), {'beta': 1.0}, key)
        assert out3.shape == (6, 6, 6)
        return True, "bornes tanh ✓ | rang-agnostique ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t25_gam002():
    try:
        import jax, jax.numpy as jnp
        from atomics.operators.gam_002_diffusion import apply, METADATA
        assert METADATA['rank_constraint'] == 2
        key   = jax.random.PRNGKey(0)
        state = jax.random.normal(key, (10, 10))
        out   = apply(state, {'alpha': 0.1}, key)
        assert float(jnp.std(out)) < float(jnp.std(state))
        raised = False
        try: apply(jnp.ones((4, 4, 4)), {}, key)
        except ValueError: raised = True
        assert raised
        return True, "diffusion ✓ | std↓ ✓ | ValueError rang3 ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t26_gam003():
    try:
        import jax, jax.numpy as jnp
        from atomics.operators.gam_003_exp_growth import apply
        key   = jax.random.PRNGKey(0)
        state = jnp.ones((5, 5))
        out   = state
        for _ in range(10):
            out = apply(out, {'gamma': 1.0}, key)
        ratio = float(jnp.linalg.norm(out)) / float(jnp.linalg.norm(state))
        assert abs(ratio - float(jnp.exp(10.0))) < 1.0
        return True, f"explosion ✓ | ratio={ratio:.0f} ≈ e^10 ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t27_gam004():
    try:
        import jax, jax.numpy as jnp
        from atomics.operators.gam_004_exp_decay import apply
        key   = jax.random.PRNGKey(0)
        state = jax.random.normal(key, (8, 8))
        out   = apply(state, {'gamma': 0.1}, key)
        assert jnp.allclose(out, state * jnp.exp(-0.1), atol=1e-6).item()
        return True, "décroissance analytique ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t28_gam005():
    try:
        import jax, jax.numpy as jnp
        from atomics.operators.gam_005_harmonic import apply, METADATA
        assert METADATA['non_markovian'] is True
        omega = math.pi / 4
        out   = apply(jnp.ones((5, 5)), {'omega': omega}, jax.random.PRNGKey(0))
        assert jnp.allclose(out, jnp.cos(omega) * jnp.ones((5, 5)), atol=1e-6).item()
        return True, "non_markovian=True ✓ | fallback cos(ω)·T ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t29_gam006():
    try:
        import jax, jax.numpy as jnp
        from atomics.operators.gam_006_memory_tanh import apply, METADATA
        assert METADATA['non_markovian'] is True
        key   = jax.random.PRNGKey(0)
        state = jax.random.normal(key, (6, 6))
        out   = apply(state, {'beta': 1.5}, key)
        assert jnp.allclose(out, jnp.tanh(1.5 * state), atol=1e-6).item()
        return True, "non_markovian=True ✓ | fallback tanh(β·T) ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t30_gam007():
    try:
        import jax, jax.numpy as jnp
        from atomics.operators.gam_007_sliding_avg import apply, METADATA
        assert METADATA['rank_constraint'] == 2
        key   = jax.random.PRNGKey(0)
        state = jax.random.normal(key, (10, 10))
        out   = apply(state, {'epsilon': 0.5}, key)
        assert float(jnp.std(out)) < float(jnp.std(state))
        raised = False
        try: apply(jnp.ones((4, 4, 4)), {}, key)
        except ValueError: raised = True
        assert raised
        return True, "lissage 8-voisins ✓ | ValueError rang3 ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t31_gam008():
    try:
        import jax, jax.numpy as jnp
        from atomics.operators.gam_008_diff_memory import apply, METADATA
        assert METADATA['non_markovian'] is True
        key   = jax.random.PRNGKey(0)
        state = jax.random.normal(key, (6, 6))
        out   = apply(state, {'beta': 1.0}, key)
        assert jnp.allclose(out, jnp.tanh(2.0 * state), atol=1e-6).item()
        return True, "non_markovian=True ✓ | fallback tanh((1+β)·T) ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t32_gam009():
    try:
        import jax, jax.numpy as jnp
        from atomics.operators.gam_009_stochastic_tanh import apply, METADATA
        assert METADATA['stochastic'] is True
        state = jnp.ones((8, 8)) * 0.5
        o1 = apply(state, {'beta': 1.0, 'sigma': 0.1}, jax.random.PRNGKey(0))
        o2 = apply(state, {'beta': 1.0, 'sigma': 0.1}, jax.random.PRNGKey(1))
        assert not jnp.allclose(o1, o2).item()
        return True, "stochastique ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t33_gam010():
    try:
        import jax, jax.numpy as jnp
        from atomics.operators.gam_010_mult_noise import apply, METADATA
        assert METADATA['stochastic'] is True
        state = jax.random.normal(jax.random.PRNGKey(0), (8, 8))
        o1    = apply(state, {'sigma': 0.1}, jax.random.PRNGKey(0))
        o2    = apply(state, {'sigma': 0.1}, jax.random.PRNGKey(1))
        assert not jnp.allclose(o1, o2).item()
        assert float(jnp.max(jnp.abs(o1))) < 1.0 + 1e-6
        return True, "stochastique ✓ | bornes tanh ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t34_gam011():
    try:
        import jax, jax.numpy as jnp
        from atomics.operators.gam_011_linear_tensordot import apply, prepare_params
        key   = jax.random.PRNGKey(0)
        n_dof = 8
        p     = prepare_params({'scale': 0.5}, n_dof, key)
        assert 'W' in p
        state  = jax.random.normal(key, (n_dof, n_dof))
        out    = apply(state, p, key)
        assert out.shape == (n_dof, n_dof)
        out3   = apply(jax.random.normal(key, (n_dof, n_dof, n_dof)), p, key)
        assert out3.shape == (n_dof, n_dof, n_dof)
        return True, "rang2 ✓ | rang3 ✓ | prepare_params ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t35_gam012():
    try:
        import jax, jax.numpy as jnp
        from atomics.operators.gam_012_forced_sym import apply, METADATA
        assert METADATA['rank_constraint'] == 2
        key = jax.random.PRNGKey(0)
        out = apply(jax.random.normal(key, (8, 8)), {'beta': 2.0}, key)
        assert jnp.allclose(out, out.T, atol=1e-6).item()
        assert float(jnp.max(jnp.abs(out))) < 1.0 + 1e-6
        return True, "symétrie garantie ✓ | bornes tanh ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t36_gam013():
    try:
        import jax, jax.numpy as jnp
        from atomics.operators.gam_013_hebbian import apply, METADATA
        assert METADATA['rank_constraint'] == 'square'
        key   = jax.random.PRNGKey(0)
        state = jax.random.normal(key, (6, 6)) * 0.1
        out   = apply(state, {'eta': 0.01}, key)
        assert jnp.allclose(out, state + 0.01 * (state @ state), atol=1e-6).item()
        raised = False
        try: apply(jnp.ones((4, 6)), {}, key)
        except ValueError: raised = True
        assert raised
        return True, "hebbien ✓ | ValueError non-carré ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t37_gam014():
    try:
        import jax, jax.numpy as jnp
        from atomics.operators.gam_014_orthogonal import apply, prepare_params, METADATA
        assert METADATA['rank_constraint'] == 2
        key   = jax.random.PRNGKey(0)
        n_dof = 8
        p     = prepare_params({}, n_dof, key)
        state = jax.random.normal(key, (n_dof, n_dof))
        out   = apply(state, p, key)
        norm_in  = float(jnp.linalg.norm(state))
        norm_out = float(jnp.linalg.norm(out))
        assert abs(norm_in - norm_out) < 1e-4
        raised = False
        try: apply(jnp.ones((4, 4, 4)), p, key)
        except ValueError: raised = True
        assert raised
        return True, f"isométrie ||F||={norm_in:.4f}≈{norm_out:.4f} ✓ | ValueError rang3 ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t38_gam015():
    try:
        import jax, jax.numpy as jnp
        from atomics.operators.gam_015_svd_truncation import apply, METADATA
        assert METADATA['differentiable'] is False
        key   = jax.random.PRNGKey(0)
        n_dof = 8
        k     = 3
        state = jax.random.normal(key, (n_dof, n_dof))
        out   = apply(state, {'k': k}, key)
        assert out.shape == (n_dof, n_dof)
        _, s_out, _ = jnp.linalg.svd(out, full_matrices=False)
        rank_out    = int(jnp.sum(s_out > 1e-4))
        assert rank_out <= k
        assert abs(float(s_out[0]) - 1.0) < 0.1
        # Rang 3
        out3 = apply(jax.random.normal(key, (n_dof, n_dof, n_dof)), {'k': k}, key)
        assert out3.shape == (n_dof, n_dof, n_dof)
        return True, f"rang2 ✓ | rang3 ✓ | rang_out={rank_out}≤k={k} ✓ | σ₁≈1 ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ============================================================
# BLOC F — Compositions
# ============================================================

def t39_composed_sequential():
    try:
        import jax, jax.numpy as jnp
        from atomics.operators.gam_001_tanh import apply as gam001
        from atomics.operators.gam_004_exp_decay import apply as gam004
        from compositions.compositions_jax import make_composed_gamma
        key   = jax.random.PRNGKey(0)
        state = jax.random.normal(key, (8, 8))
        comp  = make_composed_gamma([gam001, gam004], weights=None)
        out   = comp(state, [{'beta': 2.0}, {'gamma': 0.1}], key)
        exp   = gam004(gam001(state, {'beta': 2.0}, key), {'gamma': 0.1}, key)
        assert jnp.allclose(out, exp, atol=1e-6).item()
        return True, "séquentiel correct ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t40_composed_weighted():
    try:
        import jax, jax.numpy as jnp
        from atomics.operators.gam_001_tanh import apply as gam001
        from atomics.operators.gam_004_exp_decay import apply as gam004
        from compositions.compositions_jax import make_composed_gamma
        key   = jax.random.PRNGKey(0)
        state = jax.random.normal(key, (8, 8))
        comp  = make_composed_gamma([gam001, gam004], weights=[0.5, 0.5])
        out   = comp(state, [{'beta': 1.0}, {'gamma': 0.05}], key)
        exp   = 0.5 * gam001(state, {'beta': 1.0}, key) + \
                0.5 * gam004(state, {'gamma': 0.05}, key)
        assert jnp.allclose(out, exp, atol=1e-5).item()
        return True, "pondéré correct ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


def t41_vmap_composition():
    try:
        import jax, jax.numpy as jnp
        from atomics.operators.gam_001_tanh import apply as gam001
        from atomics.operators.gam_004_exp_decay import apply as gam004
        from compositions.compositions_jax import make_composed_gamma
        from running.run_one_jax import _run_jit
        from featuring.hub_featuring import post_scan
        from atomics.D_encodings.sym_001_identity import create as enc_create
        from atomics.modifiers.m0_baseline import apply as mod_apply

        n_dof, max_it, n_batch = 8, 20, 3
        comp = make_composed_gamma([gam001, gam004], weights=[0.5, 0.5])
        key  = jax.random.PRNGKey(0)
        D    = enc_create(n_dof, {}, key)
        betas  = jnp.array([0.5, 1.0, 2.0])
        gammas = jnp.array([0.05, 0.05, 0.05])
        keys   = jax.random.split(key, n_batch)

        def run_c(beta, gamma, k):
            return _run_jit(comp, [{'beta': beta}, {'gamma': gamma}],
                            mod_apply, {}, D, k, max_it)

        signals_batch, last_states = jax.vmap(run_c)(betas, gammas, keys)
        results = [
            post_scan({k: signals_batch[k][i] for k in signals_batch}, last_states[i])
            for i in range(n_batch)
        ]
        finals = [r['frob_norm_final'] for r in results]
        assert all(not math.isnan(f) for f in finals)
        assert len(set(round(f, 4) for f in finals)) > 1
        return True, f"vmap ✓ | finals distincts={[f'{v:.4f}' for v in finals]} ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ============================================================
# BLOC G — Pipeline end-to-end
# ============================================================

def t42_pipeline_end_to_end():
    try:
        import jax
        from running.run_one_jax import run_one_jax
        from atomics.operators.gam_009_stochastic_tanh import apply as gam009
        from atomics.D_encodings.sym_002_random_uniform import create as sym002
        from atomics.modifiers.m1_gaussian_noise import apply as m1
        from featuring.hub_featuring import FEATURE_NAMES

        key        = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)
        D          = sym002(10, {}, k1)
        D_mod      = m1(D, {'sigma': 0.05}, k2)
        result     = run_one_jax(gam009, {'beta': 1.0, 'sigma': 0.05},
                                 lambda s, p, k: s, {}, D_mod, k3, 80)
        assert set(result.keys()) == set(FEATURE_NAMES)
        assert all(isinstance(v, float) for v in result.values())
        return True, f"GAM-009 × SYM-002 + M1 ✓ | features={sorted(result.keys())} ✓"
    except Exception as e:
        return False, f"ERREUR: {e}"


# ============================================================
# Main
# ============================================================

def main():
    print()
    print("=" * 78)
    print("  PRC v7 — test_atomics_jax : inventaire complet")
    print("=" * 78)
    print()

    sections = [
        ("BLOC A — SYM-* encodings", [
            ("T1   SYM-001 identity              ", t1_sym001),
            ("T2   SYM-002 random_uniform         ", t2_sym002),
            ("T3   SYM-003 random_gaussian        ", t3_sym003),
            ("T4   SYM-004 correlation_matrix     ", t4_sym004),
            ("T5   SYM-005 banded                 ", t5_sym005),
            ("T6   SYM-006 block_hierarchical     ", t6_sym006),
            ("T7   SYM-007 uniform_correlation    ", t7_sym007),
            ("T8   SYM-008 random_clipped         ", t8_sym008),
        ]),
        ("BLOC B — ASY-* encodings", [
            ("T9   ASY-001 random_asymmetric      ", t9_asy001),
            ("T10  ASY-002 lower_triangular       ", t10_asy002),
            ("T11  ASY-003 antisymmetric          ", t11_asy003),
            ("T12  ASY-004 directional_gradient   ", t12_asy004),
            ("T13  ASY-005 circulant              ", t13_asy005),
            ("T14  ASY-006 sparse                 ", t14_asy006),
        ]),
        ("BLOC C — RN-* + R3-* encodings", [
            ("T15  RN-001 random_uniform          ", t15_rn001),
            ("T16  RN-002 partial_symmetric       ", t16_rn002),
            ("T17  RN-003 diagonal                ", t17_rn003),
            ("T18  RN-004 separable               ", t18_rn004),
            ("T19  R3-003 local_coupling          ", t19_r3003),
            ("T20  R3-007 block_structure         ", t20_r3007),
        ]),
        ("BLOC D — Modifiers", [
            ("T21  M0 baseline                    ", t21_m0),
            ("T22  M1 gaussian_noise              ", t22_m1),
            ("T23  M2 uniform_noise               ", t23_m2),
        ]),
        ("BLOC E — Gammas", [
            ("T24  GAM-001 tanh                   ", t24_gam001),
            ("T25  GAM-002 diffusion rank=2       ", t25_gam002),
            ("T26  GAM-003 explosion              ", t26_gam003),
            ("T27  GAM-004 exp_decay              ", t27_gam004),
            ("T28  GAM-005 fallback cos(ω)        ", t28_gam005),
            ("T29  GAM-006 fallback tanh(β)       ", t29_gam006),
            ("T30  GAM-007 sliding avg rank=2     ", t30_gam007),
            ("T31  GAM-008 fallback tanh(1+β)     ", t31_gam008),
            ("T32  GAM-009 stochastic             ", t32_gam009),
            ("T33  GAM-010 mult_noise             ", t33_gam010),
            ("T34  GAM-011 linear_tensordot       ", t34_gam011),
            ("T35  GAM-012 forced_sym rank=2      ", t35_gam012),
            ("T36  GAM-013 hebbian square         ", t36_gam013),
            ("T37  GAM-014 orthogonal             ", t37_gam014),
            ("T38  GAM-015 svd_truncation         ", t38_gam015),
        ]),
        ("BLOC F — Compositions", [
            ("T39  Composition séquentielle       ", t39_composed_sequential),
            ("T40  Composition pondérée           ", t40_composed_weighted),
            ("T41  vmap composition               ", t41_vmap_composition),
        ]),
        ("BLOC G — Pipeline end-to-end", [
            ("T42  GAM-009 × SYM-002 + M1         ", t42_pipeline_end_to_end),
        ]),
    ]

    all_results = []
    for section_label, tests in sections:
        print(f"  — {section_label} —")
        for label, fn in tests:
            ok, details = fn()
            print(_fmt(label, ok, details))
            all_results.append(ok)
        print()

    print("=" * 78)
    n_ok   = sum(all_results)
    n_fail = len(all_results) - n_ok
    if n_fail == 0:
        print(f"  Résultat : {n_ok}/{len(all_results)} ✓ — inventaire atomics JAX complet")
    else:
        print(f"  Résultat : {n_ok}/{len(all_results)} ✓  {n_fail} ✗")
    print("=" * 78)
    print()

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == '__main__':
    main()
