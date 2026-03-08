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

TODO: NON_MARKOVIAN — carry étendu (state, prev_state, key)
      à implémenter quand premier gamma non-markovien disponible
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

from featuring.hub_featuring import measure_state, post_scan, FEATURE_NAMES


# ---------------------------------------------------------------------------
# Step function — capturée dans lax.scan
# ---------------------------------------------------------------------------

def _step_fn(gamma_fn, gamma_params, carry, _):
    """
    Un pas de la dynamique.

    Args:
        gamma_fn     : Fonction gamma (statique — capturée à la compilation)
        gamma_params : Dict params gamma (dynamique — tracé par JAX)
        carry        : (state, key)
        _            : index pas (ignoré — None length scan)

    Returns:
        carry_next : (state_next, key)
        measures   : dict scalaires shape ()
    """
    state, key = carry

    # Deux splits : un pour gamma, un réservé pour features stochastiques (phase 2)
    key, subkey_gamma = jax.random.split(key)
    key, subkey_feat  = jax.random.split(key)   # noqa: F841 — réservé CLEANUP_PHASE2

    state_next = gamma_fn(state, gamma_params, subkey_gamma)
    measures   = measure_state(state, state_next)

    return (state_next, key), measures


# ---------------------------------------------------------------------------
# Bloc JIT — modifier + scan
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(0, 2, 6))
def _run_jit(
    gamma_fn,        # statique (0) — clé cache XLA
    gamma_params,    # dynamique   — vmappable
    modifier_fn,     # statique (2) — clé cache XLA
    modifier_params, # dynamique   — vmappable
    D_initial,       # dynamique   — vmappable
    key,             # dynamique   — vmappable
    max_it,          # statique (6) — longueur scan fixe à la compilation
):
    """
    Modifier + lax.scan compilés ensemble.

    Returns:
        signals    : {'frob_norm': jnp.array(max_it,)}
        last_state : jnp.ndarray (*dims)
    """
    # Phase 1 — modifier (une seule fois avant le scan)
    key, subkey_mod = jax.random.split(key)
    D_modified = modifier_fn(D_initial, modifier_params, subkey_mod)

    # Phase 2 — scan
    step = partial(_step_fn, gamma_fn, gamma_params)
    (last_state, _), signals = lax.scan(
        step,
        (D_modified, key),
        None,
        length=max_it,
    )

    return signals, last_state


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

    Returns:
        dict {str: float} — clés = FEATURE_NAMES, valeurs = floats Python purs

    Notes:
        Premier appel : compilation JIT (~0.1s à 2s selon gamma/shape).
        Appels suivants mêmes gamma_fn/modifier_fn/shape/max_it : cache hit.
        Non-markovien : TODO (carry étendu — voir module docstring)
    """
    signals, last_state = _run_jit(
        gamma_fn,
        gamma_params,
        modifier_fn,
        modifier_params,
        D_initial,
        key,
        max_it,
    )

    # Force l'évaluation JAX avant post_scan
    last_state.block_until_ready()

    return post_scan(signals, last_state)