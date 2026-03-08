"""
atomics/operators/gam_011_linear_tensordot.py

GAM-011 : Transformation linéaire tensordot (mode-0)
Forme   : T_{n+1} = W ⊗₀ T  =  tensordot(W, T, axes=([1],[0]))
Famille : markovian

W est une matrice (n_dof, n_dof) générée une fois via generate_W()
et injectée dans params['W'] avant le scan — jamais recalculée dans apply().

Rang-agnostique : W agit sur l'axe 0, tous les autres axes sont propagés.
  rang 2 : équivalent à W @ T
  rang 3 : T'[i,j,k] = Σ_l W[i,l] · T[l,j,k]
  rang N : identique, axe 0 uniquement

Rôle v7 :
  F4 — trace_J = jnp.trace(W)  (analytique, validation Hutchinson)
  F7 — spectre DMD = valeurs propres de W  (validation analytique DMD)
"""

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Métadonnées discovery
# ---------------------------------------------------------------------------

METADATA = {
    'id'              : 'GAM-011',
    'family'          : 'markovian',
    'rank_constraint' : None,       # rang-agnostique
    'differentiable'  : True,       # tensordot linéaire → jacfwd analytique
    'stochastic'      : False,      # key ignorée dans apply()
    'non_markovian'   : False,      # carry standard dans kernel
}

# ---------------------------------------------------------------------------
# Constante numérique
# ---------------------------------------------------------------------------

_EPS = 1e-8

# ---------------------------------------------------------------------------
# Génération de W — appelée en Python avant le scan
# ---------------------------------------------------------------------------

def generate_W(n_dof: int, scale: float = 1.0, key: jax.Array = None) -> jnp.ndarray:
    """
    Génère la matrice de transformation W normalisée par son rayon spectral.

    W_raw ~ N(0, 1/n_dof)
    W     = W_raw / (ρ(W_raw) + eps) × scale

    où ρ(W_raw) = max(|valeurs propres de W_raw|).

    Args:
        n_dof : Dimension de W — doit correspondre à state.shape[0]
        scale : Contrôle le rayon spectral de W
                  < 1.0 → W contractant  (T → 0)
                  = 1.0 → W isométrique spectralement
                  > 1.0 → W expansif     (T peut diverger)
        key   : PRNGKey JAX. Si None, utilise PRNGKey(0).

    Returns:
        W : jnp.ndarray shape (n_dof, n_dof)

    Note:
        Appelée UNE FOIS avant le scan, résultat injecté dans params['W'].
        Ne pas appeler dans apply() — W est un paramètre fixe du run.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    W_raw = jax.random.normal(key, (n_dof, n_dof)) / jnp.sqrt(n_dof)

    # Rayon spectral via valeurs propres
    eigenvalues = jnp.linalg.eigvals(W_raw)
    rho = jnp.max(jnp.abs(eigenvalues))

    return W_raw / (rho + _EPS) * scale


# ---------------------------------------------------------------------------
# prepare_params — convention hub_running (optionnelle)
# ---------------------------------------------------------------------------

def prepare_params(params: dict, n_dof: int, key: jax.Array) -> dict:
    """
    Pré-calcule W depuis scale + key — appelée par hub_running avant vmap.

    Convention : hub_running appelle prepare_params(raw_params, n_dof, key_CI)
    si la fonction est présente dans le module. Permet de garder hub_running
    aveugle aux internals de chaque atomic.

    Args:
        params : {'scale': float}  — params bruts depuis YAML
        n_dof  : Dimension du tenseur
        key    : PRNGKey JAX (key_CI du run)

    Returns:
        {'W': jnp.ndarray (n_dof, n_dof)}
    """
    scale = params.get('scale', 1.0)
    W     = generate_W(n_dof=n_dof, scale=float(scale), key=key)
    return {'W': W}


# ---------------------------------------------------------------------------
# Gamma apply — appelée à chaque pas dans lax.scan
# ---------------------------------------------------------------------------

def apply(
    state : jnp.ndarray,
    params: dict,
    key   : jax.Array,
) -> jnp.ndarray:
    """
    Applique T_{n+1} = tensordot(W, T, axes=([1],[0])).

    Args:
        state  : Tenseur courant, shape (n_dof, ...) — tout rang
        params : {'W': jnp.ndarray (n_dof, n_dof), 'scale': float}
                 W doit être généré via generate_W() avant le scan
        key    : Ignorée (stochastic: False)

    Returns:
        jnp.ndarray même shape que state
    """
    W = params['W']
    return jnp.tensordot(W, state, axes=([1], [0]))