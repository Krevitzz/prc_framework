"""
running/kernel_new.py

Bloc compilé central du pipeline JAX v7.
Responsabilité unique : un (D_initial, gamma, modifier) → dict features scalaires.

Trois phases :
  1. Modifier appliqué une fois avant le scan (Python → XLA)
  2. lax.scan : gamma + measure_state sur max_it pas (XLA pur)
  3. _post_scan_jax : agrégation signals → scalaires JAX (XLA pur, dans le kernel)

Différences vs run_one_jax.py :
  1. _run_fn séparée de _run_jit — permet jit(vmap(_run_fn)) dans batching_new.py
  2. _post_scan_jax intégrée dans _run_fn — signals (T,) jamais matérialisés hors XLA
     En mode batch vmap : (B,T) n'existe pas en RAM, consommé dans le kernel
  3. block_until_ready() supprimé — float() sur scalaire JAX est bloquant
  4. post_scan devient wrapper de _run_jit pour compatibilité run_one_jax

K1-K5 esprit conservé :
  - Aveugle au domaine — ne connaît pas le contenu de D ou Γ
  - Aucun branchement sur D ou Γ
  - Aucune validation sémantique

Carry étendu (v7 features complètes) :
  carry = (state, key, state_prev, sigmas_prev, A_k, P_k)
  - state_prev  : état t-1 — requis F4 Lyapunov empirique, F5 transport
  - sigmas_prev : valeurs singulières t-1 — O1, évite 1 SVD/pas (DMD update)
  - A_k, P_k    : carry DMD streaming dans espace spectral (n_dof × n_dof)

dmd_rank : argument statique → recompilation si changé.
           Configurable via YAML (clé : dmd_rank, défaut : 16).
           Wiper jax_cache/ obligatoire si modifié entre deux runs.
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

from featuring.jax_features_new import (
    measure_state,
    _post_scan_jax,
    post_scan,
    FEATURE_NAMES,
)


# ---------------------------------------------------------------------------
# Step function — capturée dans lax.scan
# Identique à run_one_jax.py — aucun changement.
# ---------------------------------------------------------------------------

def _step_fn(gamma_fn, gamma_params, is_differentiable, carry, _):
    """
    Un pas de la dynamique.

    Args:
        gamma_fn         : Fonction gamma (statique — capturée à la compilation)
        gamma_params     : Dict params gamma (dynamique — tracé par JAX)
        is_differentiable: Python bool statique
        carry            : (state, key, state_prev, sigmas_prev, A_k, P_k)
        _                : index pas (ignoré — None length scan)

    Returns:
        carry_next : (state_next, key_next, state, sigmas, A_k_new, P_k_new)
        measures   : dict 22 scalaires shape ()
    """
    state, key, state_prev, sigmas_prev, A_k, P_k = carry

    key, subkey_gamma = jax.random.split(key)
    key, subkey_feat  = jax.random.split(key)

    state_next = gamma_fn(state, gamma_params, subkey_gamma)

    measures, A_k_new, P_k_new, sigmas = measure_state(
        state,
        state_next,
        state_prev,
        gamma_fn,
        gamma_params,
        subkey_feat,
        A_k,
        P_k,
        is_differentiable,
        sigmas_prev,
    )

    carry_next = (state_next, key, state, sigmas, A_k_new, P_k_new)
    return carry_next, measures


# ---------------------------------------------------------------------------
# _run_fn — fonction nue, vmappable
#
# Différence centrale vs _run_jit de run_one_jax.py :
#   - Pas de décorateur @jit → jax.vmap(_run_fn) fusionne scan+vmap en 1 kernel XLA
#   - _post_scan_jax intégré → signals (max_it,) consommés dans XLA, jamais en RAM
#   - Retourne features dict shape () directement
# ---------------------------------------------------------------------------

def _run_fn(
    gamma_fn,            # statique via static_argnums dans _run_jit
    gamma_params,        # dynamique — vmappable
    modifier_fn,         # statique via static_argnums dans _run_jit
    modifier_params,     # dynamique — vmappable
    D_initial,           # dynamique — vmappable
    key,                 # dynamique — vmappable
    max_it,              # statique via static_argnums dans _run_jit
    dmd_rank,            # statique via static_argnums dans _run_jit
    is_differentiable,   # statique via static_argnums dans _run_jit
):
    """
    Modifier + lax.scan + _post_scan_jax en un seul graphe XLA.

    Vmappable : pas de décorateur @jit.
    Utilisé directement par batching_new.py via jit(vmap(_run_fn)).

    Returns:
        dict {feature_name: jnp.array shape ()} — 68 scalaires JAX
        Convertir en floats Python via {k: float(v) for k, v in out.items()}
        uniquement au moment de matérialiser (sync GPU→CPU).
    """
    # Phase 1 — modifier (une seule fois avant le scan)
    key, subkey_mod = jax.random.split(key)
    D_modified = modifier_fn(D_initial, modifier_params, subkey_mod)

    # Initialisation carry étendu
    n_dof       = D_modified.shape[0]
    state_prev  = jnp.zeros_like(D_modified)
    sigmas_prev = jnp.zeros(n_dof)
    A_k         = jnp.zeros((n_dof, n_dof))
    P_k         = jnp.eye(n_dof) * 1e4

    carry_init = (D_modified, key, state_prev, sigmas_prev, A_k, P_k)

    # Phase 2 — scan
    step = partial(_step_fn, gamma_fn, gamma_params, is_differentiable)
    (last_state, _, _, _, A_k_final, _P_k_final), signals = lax.scan(
        step,
        carry_init,
        None,
        length=max_it,
    )

    # Phase 3 — post_scan intégré : signals (max_it,) consommés ici
    # Jamais matérialisés hors du kernel — en vmap : (B, max_it) n'existe pas en RAM
    return _post_scan_jax(signals, last_state, A_k_final)


# ---------------------------------------------------------------------------
# _run_jit — version JIT pour usage scalaire (run_one_jax)
# ---------------------------------------------------------------------------

_run_jit = partial(
    jax.jit,
    static_argnums=(0, 2, 6, 7, 8),
)(_run_fn)


# ---------------------------------------------------------------------------
# Interface publique — run_one_jax
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
        is_differentiable : Si False, F4 Hutchinson = NaN (économie JVP)

    Returns:
        dict {str: float} — clés = FEATURE_NAMES, valeurs = floats Python purs

    Notes:
        Premier appel : compilation JIT (~0.1s à 2s selon gamma/shape).
        Appels suivants mêmes gamma_fn/modifier_fn/shape/max_it/dmd_rank : cache hit.
        float() sur le premier scalaire force block_until_ready implicitement.
    """
    features_jax = _run_jit(
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
    # float() force la matérialisation GPU→CPU — block_until_ready implicite
    # Pas besoin d'appel explicite à block_until_ready()
    return {k: float(v) for k, v in features_jax.items()}