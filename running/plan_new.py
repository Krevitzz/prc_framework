"""
running/plan_new.py

Génération des kernel_groups vmappables depuis un YAML de run.

Responsabilité :
  YAML + registries → kernel_groups (gamma_fn, enc_fn, mod_fn, n_dof, rank_eff, max_it, ...)
  Chaque kernel_group contient la liste des samples vmappables (params + keys).

Différences vs compositions/compositions_jax.py :
  1. generate_groups_lazy    → generate_kernel_groups  (renommage)
  2. chunk_group             → split_into_batches       (renommage)
  3. generate_chunks supprimé — doublon de generate_kernel_groups + split_into_batches
  4. Bug #3 corrigé : _resolve_gamma_sequence stocke sequence_fns + sequence_ids
     dans le dict composé → bloc pondération peut les relire correctement
  5. kernel_group yield étendu :
       is_differentiable, enc_vmappable, gamma_param_keys, mod_param_keys, prepare_params
  6. Import depuis data_loading_new (plus data_loading_jax)

Terminologie :
  kernel_group : même clé de compilation XLA — (gamma_id, enc_id, mod_id, n_dof, rank_eff, max_it)
  sample       : une combinaison de params scalaires + seeds dans un kernel_group
  batch        : sous-liste de samples d'un kernel_group → un appel vmap
"""

import random as _random
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jax
import jax.numpy as jnp

from utils.data_loading_new import load_yaml


# =============================================================================
# HELPERS GÉNÉRAUX
# =============================================================================

def _normalize_to_list(x) -> list:
    """Scalaire → [scalaire], liste inchangée."""
    return x if isinstance(x, list) else [x]


def _generate_weights_grid(
    weights_config: Dict,
    n_gammas      : int,
) -> List[List[float]]:
    """
    Génère grid pondération depuis config.

    Args:
        weights_config : {'range': [min, max], 'step': float}
        n_gammas       : Nombre de gammas dans la séquence

    Returns:
        Liste de combinaisons de poids [[w1,w2,...], ...]
    """
    range_min, range_max = weights_config['range']
    step    = weights_config['step']
    n_steps = int((range_max - range_min) / step) + 1
    values  = [range_min + i * step for i in range(n_steps)]
    return list(product(values, repeat=n_gammas))


def _format_gamma_id(
    sequence_ids: List[str],
    weights     : Optional[List[float]] = None,
) -> str:
    """
    Formate ID gamma composé.

    Format : "GAM-NNN-GAM-MMM" sans weights
             "w_0.50-GAM-NNN-w_0.50-GAM-MMM" avec weights
    """
    if weights is None:
        return '-'.join(sequence_ids)
    parts = [f"w_{w:.2f}-{gid}" for w, gid in zip(weights, sequence_ids)]
    return '-'.join(parts)


# =============================================================================
# SEEDS → PRNGKEYS
# =============================================================================

def _seed_to_key(seed) -> jax.Array:
    """
    Convertit un seed entier ou None en PRNGKey JAX.

    Args:
        seed : int ou None

    Returns:
        jax.Array — PRNGKey shape (2,)
    """
    if seed is None:
        return jax.random.PRNGKey(0)
    return jax.random.PRNGKey(int(seed))


def _make_run_key(seed_run_key: jax.Array, gamma_id: str, enc_id: str) -> jax.Array:
    """
    Dérive une key unique par composition depuis seed_run.

    Garantit l'unicité par composition même avec même seed_run.
    fold_in est JAX-natif et déterministe.

    Args:
        seed_run_key : PRNGKey de base
        gamma_id     : ID gamma (ex: 'GAM-011')
        enc_id       : ID encoding (ex: 'SYM-001')

    Returns:
        jax.Array — PRNGKey dérivé
    """
    data = abs(hash(gamma_id + enc_id)) % (2 ** 32)
    return jax.random.fold_in(seed_run_key, data)


# =============================================================================
# GAMMAS COMPOSÉS
# =============================================================================

def make_composed_gamma(
    gamma_fns: List,
    weights  : Optional[List[float]] = None,
):
    """
    Retourne une fonction gamma pure JAX composant plusieurs gammas.

    Args:
        gamma_fns : Liste de fonctions apply(state, params, key)
        weights   : None → séquentiel | [w1, w2, ...] → pondéré

    Returns:
        callable apply(state, params_list, key) → jnp.ndarray

        params_list : liste de dicts, un par gamma composé
                      ex: [{'W': W1}, {'beta': 1.0}]

    Note :
        Retourne une nouvelle fonction Python à chaque appel →
        identité propre → clé de cache XLA distincte.
    """
    if weights is None:
        # Séquentiel : gamma_{n+1}(gamma_n(...(state)))
        def apply_sequential(state, params_list, key):
            for fn, p in zip(gamma_fns, params_list):
                key, subkey = jax.random.split(key)
                state = fn(state, p, subkey)
            return state
        return apply_sequential

    else:
        # Pondéré : Σ wᵢ · gammaᵢ(state)
        w_array = jnp.array(weights)

        def apply_weighted(state, params_list, key):
            outputs = jnp.stack([
                fn(state, p, key)
                for fn, p in zip(gamma_fns, params_list)
            ])
            return jnp.tensordot(w_array, outputs, axes=([0], [0]))

        return apply_weighted


# =============================================================================
# RÉSOLUTION AXES
# =============================================================================

def resolve_axis_atomic(
    axis_config: Union[str, Dict, List],
    registry   : Dict[str, Dict],
) -> List[Dict]:
    """
    Résout un axe atomique → liste d'entrées résolues.

    Args:
        axis_config : 'all' | {'random': N} | liste de configs
        registry    : {id → {'callable', 'metadata', 'prepare_params'}}
                      depuis discover_*_jax()

    Returns:
        [{'id', 'fn', 'params': dict, 'metadata': dict,
          'prepare_params': fn|None}, ...]
        Une entrée par combinaison de params scalaires.
    """
    # Résoudre liste d'IDs
    if axis_config == 'all':
        ids_config = [{'id': k} for k in registry.keys()]

    elif isinstance(axis_config, dict) and 'random' in axis_config:
        n   = axis_config['random']
        ids = _random.sample(list(registry.keys()), min(n, len(registry)))
        ids_config = [{'id': k} for k in ids]

    elif isinstance(axis_config, list):
        ids_config = axis_config

    else:
        raise ValueError(f"Format axe inconnu : {axis_config}")

    resolved = []

    for item in ids_config:
        if isinstance(item, str):
            item = {'id': item}

        entity_id = item['id']

        if entity_id not in registry:
            raise ValueError(
                f"ID '{entity_id}' non trouvé dans le registry\n"
                f"Disponibles : {list(registry.keys())}"
            )

        entry    = registry[entity_id]
        fn       = entry['callable']
        metadata = entry['metadata']
        prepare_params = entry.get('prepare_params', None)

        # Params inline depuis YAML (peuvent contenir des listes → produit cartésien)
        inline = item.get('params', {})

        list_params   = {k: v for k, v in inline.items() if isinstance(v, list)}
        scalar_params = {k: v for k, v in inline.items() if not isinstance(v, list)}

        if not list_params:
            resolved.append({
                'id'            : entity_id,
                'fn'            : fn,
                'params'        : scalar_params,
                'metadata'      : metadata,
                'prepare_params': prepare_params,
            })
        else:
            for combo in product(*[
                [(k, v) for v in vals]
                for k, vals in list_params.items()
            ]):
                params = {**scalar_params, **dict(combo)}
                resolved.append({
                    'id'            : entity_id,
                    'fn'            : fn,
                    'params'        : params,
                    'metadata'      : metadata,
                    'prepare_params': prepare_params,
                })

    return resolved


def _get_rank_eff(enc_entry: Dict) -> int:
    """
    Détermine le rang effectif d'un encoding.

    SYM-*/ASY-*/R3-* : rank figé dans METADATA['rank']
    RN-*              : rank paramétrique dans enc_entry['params']['rank']

    Returns:
        int — rang effectif (2, 3, 4, ...)
    """
    metadata = enc_entry.get('metadata', {})
    params   = enc_entry.get('params', {})

    if 'rank' in metadata and metadata['rank'] is not None:
        return int(metadata['rank'])

    return int(params.get('rank', 3))


def _resolve_gamma_axis(
    gamma_config  : Union[str, Dict, List],
    gamma_registry: Dict,
) -> List[Dict]:
    """
    Résout l'axe gamma — supporte séquences composées et pondération.

    Returns:
        [{'id', 'fn', 'params', 'metadata', 'is_composed': bool,
          'prepare_params': fn|None}, ...]
    """
    # Séquence multi-lignes avec pondération optionnelle
    if isinstance(gamma_config, dict) and 'sequence' in gamma_config:
        lines       = gamma_config['sequence']
        weights_cfg = gamma_config.get('weights')
        return _resolve_gamma_sequence(lines, gamma_registry, weights_cfg)

    # Liste de listes → séquences composées
    if isinstance(gamma_config, list) and any(
        isinstance(x, list) for x in gamma_config
    ):
        return _resolve_gamma_sequence(gamma_config, gamma_registry)

    # Cas standard
    entries = resolve_axis_atomic(gamma_config, gamma_registry)
    for e in entries:
        e['is_composed'] = False
    return entries


def _resolve_gamma_sequence(
    lines_config : List,
    registry     : Dict,
    weights_cfg  : Optional[Dict] = None,
) -> List[Dict]:
    """
    Résout séquences de gammas composés.

    Fix Bug #3 vs compositions_jax.py :
      - 'sequence_fns' et 'sequence_ids' stockés dans le dict composé
      - Le bloc pondération les relit directement — plus de seq.get('sequence', [seq])
      - n_gammas = len(seq['sequence_fns']) — correct pour tout type de séquence

    Args:
        lines_config : Liste de configs (une par ligne)
        registry     : Registre gammas
        weights_cfg  : Config pondération optionnelle

    Returns:
        [{'id', 'fn', 'params', 'metadata', 'is_composed': bool,
          'sequence_fns': list, 'sequence_ids': list,
          'prepare_params': None}, ...]
    """
    lines_resolved = [
        resolve_axis_atomic(line_cfg, registry)
        for line_cfg in lines_config
    ]

    sequences = []
    for combo in product(*lines_resolved):
        sequence_ids  = [e['id']     for e in combo]
        sequence_fns  = [e['fn']     for e in combo]
        shared_params = {}
        for e in combo:
            shared_params.update(e['params'])

        if len(combo) == 1:
            entry = combo[0]
            sequences.append({
                'id'           : entry['id'],
                'fn'           : entry['fn'],
                'params'       : entry['params'],
                'metadata'     : entry['metadata'],
                'is_composed'  : False,
                # Stockés pour le bloc pondération — uniforme avec is_composed=True
                'sequence_fns' : [entry['fn']],
                'sequence_ids' : [entry['id']],
                'prepare_params': entry.get('prepare_params', None),
            })
        else:
            composed_id = _format_gamma_id(sequence_ids)
            composed_fn = make_composed_gamma(sequence_fns)
            sequences.append({
                'id'           : composed_id,
                'fn'           : composed_fn,
                'params'       : shared_params,
                'metadata'     : {
                    'id'            : composed_id,
                    'non_markovian' : False,
                    'stochastic'    : False,
                    'differentiable': False,
                },
                'is_composed'  : True,
                # Fix Bug #3 : stockés ici pour que le bloc pondération
                # puisse relire les fns individuelles sans combo en scope
                'sequence_fns' : sequence_fns,
                'sequence_ids' : sequence_ids,
                'prepare_params': None,
            })

    # Pondération
    if weights_cfg:
        sequences_weighted = []
        for seq in sequences:
            # Fix Bug #3 : n_gammas depuis sequence_fns stocké — pas seq.get('sequence', [seq])
            n_gammas     = len(seq['sequence_fns'])
            weights_grid = _generate_weights_grid(weights_cfg, n_gammas)

            for weights in weights_grid:
                # Fix Bug #3 : fns depuis sequence_fns stocké — pas seq['fn'] (opaque)
                composed_fn = make_composed_gamma(
                    seq['sequence_fns'],
                    list(weights),
                )
                wid = _format_gamma_id(seq['sequence_ids'], list(weights))
                sequences_weighted.append({
                    'id'           : wid,
                    'fn'           : composed_fn,
                    'params'       : seq['params'],
                    'metadata'     : seq['metadata'],
                    'is_composed'  : True,
                    'sequence_fns' : seq['sequence_fns'],
                    'sequence_ids' : seq['sequence_ids'],
                    'weights'      : list(weights),
                    'prepare_params': None,
                })
        return sequences_weighted

    return sequences


# =============================================================================
# GÉNÉRATION KERNEL_GROUPS (streaming — 1 groupe en RAM à la fois)
# =============================================================================

def generate_kernel_groups(
    run_config : Dict[str, Any],
    registries : Dict[str, Dict],
):
    """
    Générateur de kernel_groups vmappables — version streaming.

    Renommé depuis generate_groups_lazy (terminologie charter v7).
    Streaming pur — yield un kernel_group à la fois.
    Jamais plus d'un kernel_group (+ ses samples) en RAM simultanément.
    Le split en batches est délégué à split_into_batches() dans la hot loop.

    Args:
        run_config : Dict YAML chargé via load_yaml()
        registries : {'gamma': {...}, 'encoding': {...}, 'modifier': {...}}
                     depuis discover_*_jax()

    Yields:
        kernel_group dict :
        {
            # Clé structurelle XLA
            'gamma_id'        : str,
            'gamma_fn'        : callable,
            'enc_id'          : str,
            'enc_fn'          : callable,
            'mod_id'          : str,
            'mod_fn'          : callable,
            'n_dof'           : int,
            'rank_eff'        : int,
            'max_it'          : int,
            # Flags statiques (static_argnums dans _run_jit)
            'is_differentiable': bool,
            'non_markovian'    : bool,
            # Vmappabilité encoding
            'enc_vmappable'    : bool,
            # Clés params pour _vmap_cache
            'gamma_param_keys' : tuple[str, ...],
            'mod_param_keys'   : tuple[str, ...],
            # prepare_params optionnel (GAM-011, GAM-014, ...)
            'prepare_params'   : callable | None,
            # Métadonnées
            'phase'            : str,
            # Samples vmappables
            'samples'          : list[dict],
        }

    samples[i] dict :
        {
            'gamma_params' : dict,
            'enc_params'   : dict,
            'mod_params'   : dict,
            'seed_CI'      : int | None,
            'seed_run'     : int | None,
            'key_CI'       : jax.Array,
            'key_run'      : jax.Array,
        }
    """
    axes    = run_config.get('axes', {})
    phase   = run_config.get('phase', 'unknown')
    n_dofs  = _normalize_to_list(run_config.get('n_dof', 10))
    max_its = _normalize_to_list(run_config.get('max_iterations', 200))

    gamma_config    = axes.get('gamma', 'all')
    encoding_config = axes.get('encoding', 'all')
    modifier_config = axes.get('modifier', [{'id': 'M0'}])

    gammas    = _resolve_gamma_axis(gamma_config, registries['gamma'])
    encodings = resolve_axis_atomic(encoding_config, registries['encoding'])
    modifiers = resolve_axis_atomic(modifier_config, registries['modifier'])

    seed_CIs  = _normalize_to_list(run_config.get('seed_CI',  [None]))
    seed_runs = _normalize_to_list(run_config.get('seed_run', [None]))

    # Indexer par clé structurelle pour outer loop
    gammas_by_id: Dict[str, List] = {}
    for g in gammas:
        gammas_by_id.setdefault(g['id'], []).append(g)

    encs_by_key: Dict[tuple, List] = {}
    for e in encodings:
        k = (e['id'], _get_rank_eff(e))
        encs_by_key.setdefault(k, []).append(e)

    mods_by_id: Dict[str, List] = {}
    for m in modifiers:
        mods_by_id.setdefault(m['id'], []).append(m)

    for (gamma_id, gamma_entries), (enc_key, enc_entries), (mod_id, mod_entries), n_dof, max_it \
        in product(
            gammas_by_id.items(),
            encs_by_key.items(),
            mods_by_id.items(),
            n_dofs,
            max_its,
        ):
        enc_id, rank_eff = enc_key

        # Représentants pour les champs fixes du kernel_group
        g_rep = gamma_entries[0]
        e_rep = enc_entries[0]
        m_rep = mod_entries[0]

        # Flags statiques depuis metadata
        is_differentiable = bool(g_rep['metadata'].get('differentiable', True))
        non_markovian     = bool(g_rep['metadata'].get('non_markovian',  False))
        enc_vmappable     = bool(e_rep['metadata'].get('jax_vmappable',  True))

        # Clés params pour _vmap_cache (tuple hashable)
        gamma_param_keys = tuple(sorted(g_rep['params'].keys()))
        mod_param_keys   = tuple(sorted(m_rep['params'].keys()))

        # prepare_params depuis le représentant gamma
        prepare_params_fn = g_rep.get('prepare_params', None)

        # Accumulation légère des samples
        samples: List[Dict] = []
        for gamma_e, enc_e, mod_e, seed_CI, seed_run in product(
            gamma_entries, enc_entries, mod_entries, seed_CIs, seed_runs
        ):
            key_CI  = _seed_to_key(seed_CI)
            key_run = _make_run_key(_seed_to_key(seed_run), gamma_id, enc_id)
            samples.append({
                'gamma_params': gamma_e['params'],
                'enc_params'  : enc_e['params'],
                'mod_params'  : mod_e['params'],
                'seed_CI'     : seed_CI,
                'seed_run'    : seed_run,
                'key_CI'      : key_CI,
                'key_run'     : key_run,
            })

        yield {
            'gamma_id'         : gamma_id,
            'gamma_fn'         : g_rep['fn'],
            'enc_id'           : enc_id,
            'enc_fn'           : e_rep['fn'],
            'mod_id'           : mod_id,
            'mod_fn'           : m_rep['fn'],
            'n_dof'            : n_dof,
            'rank_eff'         : rank_eff,
            'max_it'           : max_it,
            'is_differentiable': is_differentiable,
            'non_markovian'    : non_markovian,
            'enc_vmappable'    : enc_vmappable,
            'gamma_param_keys' : gamma_param_keys,
            'mod_param_keys'   : mod_param_keys,
            'prepare_params'   : prepare_params_fn,
            'phase'            : phase,
            'samples'          : samples,
        }

        del samples


# =============================================================================
# SPLIT EN BATCHES
# =============================================================================

def split_into_batches(
    kernel_group: Dict,
    batch_size  : int,
) -> List[Dict]:
    """
    Découpe les samples d'un kernel_group en batches vmappables.

    Renommé depuis chunk_group (terminologie charter v7).

    Args:
        kernel_group : Un kernel_group depuis generate_kernel_groups()
        batch_size   : Taille max d'un batch

    Returns:
        Liste de batches — même structure que kernel_group,
        avec 'samples' remplacé par une sous-liste de taille <= batch_size.
    """
    samples = kernel_group['samples']
    batches = []
    for i in range(0, len(samples), batch_size):
        batch = {**kernel_group, 'samples': samples[i:i + batch_size]}
        batches.append(batch)
    return batches


# =============================================================================
# STATS
# =============================================================================

def count_total_samples(
    run_config : Dict[str, Any],
    registries : Dict[str, Dict],
) -> Dict[str, int]:
    """
    Compte les samples totaux et kernel_groups sans les matérialiser.

    Utilisé pour le dry-run avant confirmation utilisateur.

    Args:
        run_config : Dict YAML
        registries : {'gamma', 'encoding', 'modifier'}

    Returns:
        {
            'n_kernel_groups' : int,
            'n_samples_total' : int,
            'n_compilations'  : int,  # = n_kernel_groups (1 compile/groupe)
        }
    """
    n_groups  = 0
    n_samples = 0

    for kg in generate_kernel_groups(run_config, registries):
        n_groups  += 1
        n_samples += len(kg['samples'])
        del kg

    return {
        'n_kernel_groups' : n_groups,
        'n_samples_total' : n_samples,
        'n_compilations'  : n_groups,
    }
