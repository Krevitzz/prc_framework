"""
running/run_one_jax.py

Bloc compilé central du pipeline JAX v7.
Responsabilité unique : un (D_initial, gamma, modifier) → dict features scalaires.

Trois phases :
  1. Modifier appliqué une fois avant le scan (Python → XLA)
  2. lax.scan : gamma + measure_state sur max_it pas (XLA pur)
  3. post_scan : agrégation signals → floats Python (XLA → Python)

K1-K5 esprit conservé :
  - Aveugle au domaine — ne connaît pas le contenu de D ou Γ
  - Aucun branchement sur D ou Γ
  - Aucune validation sémantique

Carry étendu (v7 features complètes) :
  carry = (state, key, state_prev, A_k, P_k)
  - state_prev : état t-1 — requis F4 Lyapunov empirique, F5 transport
  - A_k, P_k   : carry DMD streaming dans espace spectral (n_dof × n_dof)

dmd_rank : argument statique → recompilation si changé.
           Configurable via YAML (clé : dmd_rank, défaut : 16).
           Wiper jax_cache/ obligatoire si modifié entre deux runs.
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

from featuring.hub_featuring import measure_state, post_scan, FEATURE_NAMES


# ---------------------------------------------------------------------------
# Step function — capturée dans lax.scan
# ---------------------------------------------------------------------------

def _step_fn(gamma_fn, gamma_params, is_differentiable, carry, _):
    """
    Un pas de la dynamique.

    Args:
        gamma_fn     : Fonction gamma (statique — capturée à la compilation)
        gamma_params : Dict params gamma (dynamique — tracé par JAX)
        carry        : (state, key, state_prev, A_k, P_k)
        _            : index pas (ignoré — None length scan)

    Returns:
        carry_next : (state_next, key_next, state, A_k_new, P_k_new)
        measures   : dict scalaires shape ()
    """
    state, key, state_prev, A_k, P_k = carry

    # Trois splits : gamma, features stochastiques (F4 Hutchinson), réservé
    key, subkey_gamma = jax.random.split(key)
    key, subkey_feat  = jax.random.split(key)

    state_next = gamma_fn(state, gamma_params, subkey_gamma)

    measures, A_k_new, P_k_new = measure_state(
        state,
        state_next,
        state_prev,
        gamma_fn,
        gamma_params,
        subkey_feat,
        A_k,
        P_k,
        is_differentiable,
    )

    # state devient state_prev au prochain pas
    carry_next = (state_next, key, state, A_k_new, P_k_new)
    return carry_next, measures


# ---------------------------------------------------------------------------
# Bloc JIT — modifier + scan
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(0, 2, 6, 7, 8))
def _run_jit(
    gamma_fn,        # statique (0) — clé cache XLA
    gamma_params,    # dynamique   — vmappable
    modifier_fn,     # statique (2) — clé cache XLA
    modifier_params, # dynamique   — vmappable
    D_initial,       # dynamique   — vmappable
    key,             # dynamique   — vmappable
    max_it,              # statique (6) — longueur scan fixe à la compilation
    dmd_rank,            # statique (7) — shape carry DMD fixe à la compilation
    is_differentiable,   # statique (8) — NaN sur F4 Hutchinson si False
):
    """
    Modifier + lax.scan compilés ensemble.

    Returns:
        signals    : dict {feature_name: jnp.array(max_it,)}
        last_state : jnp.ndarray (*dims)
        A_k_final  : jnp.ndarray (n_dof, n_dof) — matrice DMD finale
        P_k_final  : jnp.ndarray (n_dof, n_dof) — covariance DMD finale
    """
    # Phase 1 — modifier (une seule fois avant le scan)
    key, subkey_mod = jax.random.split(key)
    D_modified = modifier_fn(D_initial, modifier_params, subkey_mod)

    # Initialisation carry étendu
    n_dof      = D_modified.shape[0]
    state_prev = jnp.zeros_like(D_modified)
    A_k        = jnp.zeros((n_dof, n_dof))
    P_k        = jnp.eye(n_dof) * 1e4          # prior faiblement informatif

    carry_init = (D_modified, key, state_prev, A_k, P_k)

    # Phase 2 — scan
    step = partial(_step_fn, gamma_fn, gamma_params, is_differentiable)
    (last_state, _, _, A_k_final, P_k_final), signals = lax.scan(
        step,
        carry_init,
        None,
        length=max_it,
    )

    return signals, last_state, A_k_final, P_k_final


# ---------------------------------------------------------------------------
# Interface publique
# ---------------------------------------------------------------------------

def run_one_jax(
    gamma_fn,
    gamma_params,
    modifier_fn,
    modifier_params,
    D_initial,
    key,
    max_it,
    dmd_rank=16,
    is_differentiable=True,
):
    """
    Exécute un run complet et retourne les features scalaires.

    Args:
        gamma_fn        : Fonction apply() d'un atomic gamma JAX
        gamma_params    : Dict params gamma (ex: {'W': array, 'scale': 1.0})
        modifier_fn     : Fonction apply() d'un atomic modifier JAX
        modifier_params : Dict params modifier (ex: {} pour M0)
        D_initial       : Tenseur initial, shape (n_dof, ...) — tout rang
        key             : PRNGKey JAX — source de toute l'aléatoire du run
        max_it          : Nombre de pas (statique — détermine la compilation)
        dmd_rank        : Rang DMD streaming (statique — recompilation si changé)
                          Configurable via YAML clé 'dmd_rank', défaut=16.
                          ATTENTION : wiper jax_cache/ si modifié.

    Returns:
        dict {str: float} — clés = FEATURE_NAMES, valeurs = floats Python purs

    Notes:
        Premier appel : compilation JIT (~0.1s à 2s selon gamma/shape).
        Appels suivants mêmes gamma_fn/modifier_fn/shape/max_it/dmd_rank : cache hit.
        F4 Hutchinson : NaN automatique pour gammas non-différentiables
                        (exception capturée dans measure_state).
    """
    signals, last_state, A_k_final, P_k_final = _run_jit(
        gamma_fn,
        gamma_params,
        modifier_fn,
        modifier_params,
        D_initial,
        key,
        max_it,
        dmd_rank,
        is_differentiable,
    )

    # Force l'évaluation JAX avant post_scan
    last_state.block_until_ready()

    return post_scan(signals, last_state, A_k_final, P_k_final)