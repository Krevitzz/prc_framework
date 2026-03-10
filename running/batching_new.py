"""
running/batching_new.py

Construction des inputs batché et exécution vmappée.

Responsabilité :
  batch (depuis split_into_batches) → features_b JAX async → rows parquet

Flux :
  _get_kernel_fn(batch, dmd_rank)       → callable jit(vmap(_run_fn)) [depuis cache]
  _build_batch_inputs(batch)            → (D_b, gp_b, mp_b, keys_b)
  execute_batch(batch, dmd_rank)        → features_b dict {str: jnp.array(B,)} [async]
  sync_features_b(features_b)           → {str: np.ndarray(B,)} [sync GPU→CPU]
  rows_from_synced(batch, synced, phase) → List[Dict] rows parquet

Différences vs hub_running.py :
  1. AOT layer supprimé (ThreadPoolExecutor, _compile_group_aot, _run_chunk_aot)
     — reimplémenter le cache JAX incorrectement, complexité sans gain
  2. _vmap_chunk à chaque chunk → recompile à chaque appel
     → remplacé par _get_kernel_fn + _vmap_cache : jit(vmap) créé une fois par kernel_group
  3. jax.vmap(_run_jit, in_axes=(None, ...)) avec callables Python
     → remplacé par closure _run_one(gp, mp, D, key) capturant les statiques
     → tracé correctement, zéro ambiguïté vmap sur callables
  4. block_until_ready() dans _process_chunk_results
     → remplacé par np.array() dans sync_features_b (sync implicite, vectorisé)
  5. gc.collect() supprimé (GC synchrone inutile entre kernels async)
  6. post_scan appelé row-par-row dans _rows_from_chunk
     → remplacé par _post_scan_jax intégré dans _run_fn (1 kernel XLA, vmappable)
     → features_b déjà agrégées, pas de signals (T,) à matérialiser

Architecture _vmap_cache :
  Clé = (id(gamma_fn), id(mod_fn), n_dof, max_it, dmd_rank,
          is_differentiable, gamma_param_keys, mod_param_keys)
  Une entrée par kernel_group — partagée entre tous les batches du groupe.
  Lifetime = process — pas de LRU (3700 entrées max, ~quelques KB).

Statuts run :
  OK        — features finies, comportement nominal
  EXPLOSION — au moins un Inf (P2 charter : Inf = information, pas erreur)
  NAN_ALL   — toutes features non-structurelles NaN, zéro Inf
              (rien de mesurable — distinct d'une explosion)
"""

from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from running.kernel_new import _run_fn
from featuring.jax_features_new import FEATURE_NAMES, FEATURES_STRUCTURAL_NAN


# =============================================================================
# _VMAP_CACHE
# =============================================================================

_vmap_cache: Dict[tuple, Any] = {}
"""
Cache global jit(vmap(_run_fn)).

Clé : (id(gamma_fn), id(mod_fn), n_dof, max_it, dmd_rank,
        is_differentiable, gamma_param_keys, mod_param_keys)

Valeur : callable (gp_b, mp_b, D_b, keys_b) → features_b dict

Lifetime process — pas de LRU. ~3700 entrées max, ~quelques KB de metadata.
"""


def _cache_key(batch: Dict, dmd_rank: int) -> tuple:
    """
    Construit la clé de cache depuis un batch.

    id(gamma_fn) et id(mod_fn) : identifient la fonction Python — stable
    dans le lifetime process si les fonctions sont créées une fois (atomics).

    gamma_param_keys et mod_param_keys : tuples triés depuis plan_new.
    Distinguent les in_axes du vmap (dict vide vs dict non vide).
    """
    return (
        id(batch['gamma_fn']),
        id(batch['mod_fn']),
        batch['n_dof'],
        batch['max_it'],
        dmd_rank,
        batch['is_differentiable'],
        batch['gamma_param_keys'],
        batch['mod_param_keys'],
    )


def _get_kernel_fn(batch: Dict, dmd_rank: int):
    """
    Retourne jit(vmap(_run_fn)) depuis _vmap_cache ou le crée.

    Une seule compilation XLA par kernel_group — partagée entre tous
    ses batches. La clé encode exactement les dimensions statiques XLA :
    shapes (n_dof, rank_eff via D), longueur scan (max_it), flags statiques.

    Closure _run_one :
        Capture gamma_fn, mod_fn, max_it, dmd_rank, is_differentiable.
        vmap porte uniquement sur (gp, mp, D, key) — les args dynamiques.
        Pas de in_axes=None sur callables Python (non supporté par vmap) —
        les statiques sont capturés proprement dans la closure.

    in_axes :
        gp  : {k: 0} si gamma a des params, {} si aucun
        mp  : {k: 0} si modifier a des params, {} si aucun (M0)
        D   : 0  (batch dim sur les encodings)
        key : 0  (batch dim sur les seeds)
    """
    key = _cache_key(batch, dmd_rank)
    if key in _vmap_cache:
        return _vmap_cache[key]

    # Capturer les statiques dans la closure
    gamma_fn    = batch['gamma_fn']
    mod_fn      = batch['mod_fn']
    max_it      = batch['max_it']
    is_diff     = batch['is_differentiable']
    gp_keys     = batch['gamma_param_keys']
    mp_keys     = batch['mod_param_keys']

    in_axes_gp = {k: 0 for k in gp_keys} if gp_keys else {}
    in_axes_mp = {k: 0 for k in mp_keys} if mp_keys else {}

    def _run_one(gp, mp, D, key):
        return _run_fn(
            gamma_fn, gp,
            mod_fn,   mp,
            D, key,
            max_it, dmd_rank, is_diff,
        )

    vmapped = jax.vmap(_run_one, in_axes=(in_axes_gp, in_axes_mp, 0, 0))
    kernel  = jax.jit(vmapped)

    _vmap_cache[key] = kernel
    return kernel


# =============================================================================
# CONSTRUCTION DES INPUTS BATCHÉ
# =============================================================================

def _build_enc_batch(
    samples       : List[Dict],
    enc_fn        : Any,
    n_dof         : int,
    enc_vmappable : bool,
) -> jnp.ndarray:
    """
    Construit D_batch (B, n_dof, ...) en appelant enc_fn pour chaque sample.

    Si enc_vmappable = True (flag METADATA['jax_vmappable']) :
        Stack enc_params et key_CI → vmap enc_fn → un seul kernel XLA.
    Si enc_vmappable = False :
        Loop Python → jnp.stack — encoding contient de la logique non-JAX.

    Dans les deux cas, le résultat est identique numériquement.
    enc_vmappable est un flag de performance, pas de correction.

    Args:
        samples       : Liste de dicts sample depuis split_into_batches
        enc_fn        : callable create(n_dof, params, key) → jnp.ndarray
        n_dof         : int (statique pour ce batch)
        enc_vmappable : bool depuis kernel_group['enc_vmappable']

    Returns:
        jnp.ndarray shape (B, n_dof, ...) — rank selon encoding
    """
    if enc_vmappable:
        # Stack les params et keys pour vmap
        # enc_params peut être {} (identity encoding) ou dict de scalaires
        sample0_enc_params = samples[0]['enc_params']

        if sample0_enc_params:
            enc_params_b = {
                k: jnp.stack([jnp.array(s['enc_params'][k]) for s in samples])
                for k in sample0_enc_params
            }
            keys_CI_b = jnp.stack([s['key_CI'] for s in samples])
            in_axes_ep = {k: 0 for k in enc_params_b}

            def _enc_one(ep, k):
                return enc_fn(n_dof, ep, k)

            return jax.vmap(_enc_one, in_axes=(in_axes_ep, 0))(enc_params_b, keys_CI_b)

        else:
            # Pas de params (ex: SYM-001 identity) — seule la key varie
            keys_CI_b = jnp.stack([s['key_CI'] for s in samples])
            return jax.vmap(lambda k: enc_fn(n_dof, {}, k))(keys_CI_b)

    else:
        # Loop Python — encoding non JAX-pur
        return jnp.stack([
            enc_fn(n_dof, s['enc_params'], s['key_CI'])
            for s in samples
        ])


def _build_gamma_params_batch(
    samples         : List[Dict],
    n_dof           : int,
    prepare_params_fn,
) -> Dict:
    """
    Construit gamma_params batché pour vmap.

    Si prepare_params disponible : appelé par sample pour pré-calculer
    les params dépendant de n_dof et key_CI (ex: GAM-011 génère W).
    Loop Python obligatoire — prepare_params peut retourner des shapes
    variables (W matrice pour différents n_dof) ou contenir de la logique
    Python non tracée.

    Args:
        samples          : Liste de dicts sample
        n_dof            : int (statique pour ce batch)
        prepare_params_fn: callable(raw_params, n_dof, key_CI) → dict
                           ou None si absent

    Returns:
        {param_name: jnp.ndarray(B, ...)} ou {} si aucun param
    """
    params_list = []
    for s in samples:
        raw = s['gamma_params']
        if prepare_params_fn is not None:
            p = prepare_params_fn(raw, n_dof, s['key_CI'])
        else:
            p = raw
        params_list.append(p)

    if not params_list or not params_list[0]:
        return {}

    keys = params_list[0].keys()
    return {
        k: jnp.stack([jnp.array(p[k]) for p in params_list])
        for k in keys
    }


def _build_mod_params_batch(samples: List[Dict]) -> Dict:
    """
    Construit mod_params batché pour vmap.

    M0 retourne {} directement — pas de vmap sur params vides.

    Returns:
        {param_name: jnp.ndarray(B, ...)} ou {}
    """
    if not samples[0]['mod_params']:
        return {}

    keys = samples[0]['mod_params'].keys()
    return {
        k: jnp.stack([jnp.array(s['mod_params'][k]) for s in samples])
        for k in keys
    }


def _build_keys_batch(samples: List[Dict]) -> jnp.ndarray:
    """
    Construit keys_batch (B, 2) depuis key_run de chaque sample.

    Returns:
        jnp.ndarray shape (B, 2) — PRNGKeys batché pour vmap
    """
    return jnp.stack([s['key_run'] for s in samples])


def _build_batch_inputs(batch: Dict) -> Tuple:
    """
    Construit les 4 inputs vmappables depuis un batch.

    Args:
        batch : Un batch depuis split_into_batches — contient 'samples',
                'enc_fn', 'n_dof', 'enc_vmappable', 'prepare_params', ...

    Returns:
        (D_b, gp_b, mp_b, keys_b)
        D_b    : jnp.ndarray (B, n_dof, ...)
        gp_b   : dict {k: (B, ...)} ou {}
        mp_b   : dict {k: (B, ...)} ou {}
        keys_b : jnp.ndarray (B, 2)
    """
    samples = batch['samples']
    n_dof   = batch['n_dof']

    D_b    = _build_enc_batch(samples, batch['enc_fn'], n_dof, batch['enc_vmappable'])
    gp_b   = _build_gamma_params_batch(samples, n_dof, batch['prepare_params'])
    mp_b   = _build_mod_params_batch(samples)
    keys_b = _build_keys_batch(samples)

    return D_b, gp_b, mp_b, keys_b


# =============================================================================
# EXECUTE BATCH
# =============================================================================

def execute_batch(
    batch    : Dict,
    dmd_rank : int,
) -> Dict:
    """
    Execute un batch via jit(vmap(_run_fn)) — résultat JAX async.

    Zéro synchronisation CPU/GPU — les arrays retournés sont en vol
    sur le device. La sync se fait dans sync_features_b() via np.array().
    Cela permet d'accumuler plusieurs batches avant de synchroniser
    (overlap GPU compute et écriture parquet).

    Args:
        batch    : Un batch depuis split_into_batches
        dmd_rank : Rang DMD (statique XLA — doit être constant dans une phase)

    Returns:
        features_b : {feature_name: jnp.ndarray shape (B,)} — JAX async
                     68 clés (FEATURE_NAMES)
    """
    kernel_fn           = _get_kernel_fn(batch, dmd_rank)
    D_b, gp_b, mp_b, keys_b = _build_batch_inputs(batch)
    return kernel_fn(gp_b, mp_b, D_b, keys_b)


# =============================================================================
# SYNC + ROWS
# =============================================================================

def sync_features_b(features_b: Dict) -> Dict:
    """
    Force la synchronisation GPU→CPU des features batch.

    np.array() sur un jnp.ndarray bloque jusqu'à ce que le device
    ait terminé le calcul — équivalent à block_until_ready() mais
    vectorisé sur toutes les features en un seul transfert.

    Args:
        features_b : {str: jnp.ndarray(B,)} — arrays JAX async

    Returns:
        {str: np.ndarray(B,)} — arrays numpy CPU
    """
    return {k: np.array(v) for k, v in features_b.items()}


def _run_status_from_features(features_row: Dict) -> str:
    """
    Détermine run_status depuis les features d'un run individuel.

    OK        — features finies, comportement nominal
    EXPLOSION — au moins un Inf (divergence réelle)
                P2 charter : Inf = information sur le comportement candidat
    NAN_ALL   — toutes features non-structurelles NaN, zéro Inf
                (rien de mesurable — distinct d'une explosion)

    Features structurellement NaN (FEATURES_STRUCTURAL_NAN) et health_*
    sont exclues du calcul NAN_ALL — elles ne signalent pas d'erreur.

    Args:
        features_row : {str: float} — un sample (pas batch)

    Returns:
        'OK' | 'EXPLOSION' | 'NAN_ALL'
    """
    has_inf              = False
    non_structural_vals  = []

    for k, v in features_row.items():
        fv = float(v)
        if np.isinf(fv):
            has_inf = True
        if k not in FEATURES_STRUCTURAL_NAN and not k.startswith('health_'):
            non_structural_vals.append(fv)

    if has_inf:
        return 'EXPLOSION'

    if non_structural_vals and all(np.isnan(v) for v in non_structural_vals):
        return 'NAN_ALL'

    return 'OK'


def rows_from_synced(
    batch          : Dict,
    synced_features: Dict,
    phase          : str,
) -> List[Dict]:
    """
    Construit la liste de rows parquet depuis les features sync CPU.

    Une row par sample. Chaque row contient :
      - composition : metadata du run (gamma_id, enc_id, params, seeds, ...)
      - features    : dict {str: float} — 68 scalaires Python
      - run_status  : 'OK' | 'EXPLOSION' | 'NAN_ALL'
      - phase       : str

    Args:
        batch           : Batch dont les samples fournissent les metadata
        synced_features : {str: np.ndarray(B,)} depuis sync_features_b()
        phase           : Nom de la phase (pour la colonne parquet)

    Returns:
        List[Dict] — B rows prêtes pour write_rows_to_parquet()
    """
    samples = batch['samples']
    B       = len(samples)
    rows    = []

    for i in range(B):
        sample = samples[i]

        # Extraire scalaires Python pour ce sample
        features_row = {k: float(synced_features[k][i]) for k in synced_features}

        status = _run_status_from_features(features_row)

        rows.append({
            'run_status'     : status,
            'phase'          : phase,
            'gamma_id'       : batch['gamma_id'],
            'encoding_id'    : batch['enc_id'],
            'modifier_id'    : batch['mod_id'],
            'n_dof'          : batch['n_dof'],
            'rank_eff'       : batch['rank_eff'],
            'max_it'         : batch['max_it'],
            'gamma_params'   : sample['gamma_params'],
            'encoding_params': sample['enc_params'],
            'modifier_params': sample['mod_params'],
            'seed_CI'        : sample['seed_CI'],
            'seed_run'       : sample['seed_run'],
            'features'       : features_row,
        })

    return rows


# =============================================================================
# SENTINELLE NaN (INVALID / FAIL)
# =============================================================================

def make_nan_rows(
    batch     : Dict,
    run_status: str,
    phase     : str,
) -> List[Dict]:
    """
    Génère des rows NaN pour un batch entier (INVALID ou FAIL).

    Utilisé pour les incompatibilités structurelles (rank_constraint)
    et les exceptions inattendues après fallback individuel.

    Args:
        batch      : Batch dont les samples fournissent les metadata
        run_status : 'INVALID' | 'FAIL'
        phase      : Nom de la phase

    Returns:
        List[Dict] — B rows avec features = NaN
    """
    nan_features = {f: float('nan') for f in FEATURE_NAMES}

    return [
        {
            'run_status'     : run_status,
            'phase'          : phase,
            'gamma_id'       : batch['gamma_id'],
            'encoding_id'    : batch['enc_id'],
            'modifier_id'    : batch['mod_id'],
            'n_dof'          : batch['n_dof'],
            'rank_eff'       : batch['rank_eff'],
            'max_it'         : batch['max_it'],
            'gamma_params'   : s['gamma_params'],
            'encoding_params': s['enc_params'],
            'modifier_params': s['mod_params'],
            'seed_CI'        : s['seed_CI'],
            'seed_run'       : s['seed_run'],
            'features'       : dict(nan_features),
        }
        for s in batch['samples']
    ]
