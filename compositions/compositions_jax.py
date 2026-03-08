"""
compositions/compositions_jax.py

Génération des groupes vmappables depuis un YAML de run.

Responsabilité :
  YAML → groupes (gamma_fn, enc_fn, mod_fn, n_dof, rank_eff, max_it)
  Chaque groupe contient une liste de runs vmappables (params + keys).

Convention :
  Un groupe = même clé de compilation XLA.
  Un run    = une combinaison de params scalaires + seeds.
  Un chunk  = sous-liste de runs d'un groupe → un appel vmap.

Gammas composés :
  Séquentiel  : gamma2(gamma1(state))
  Pondéré     : w1·gamma1(state) + w2·gamma2(state)
  make_composed_gamma() retourne une fonction pure JAX.
"""

import random as _random
from itertools import product
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jax
import jax.numpy as jnp

from utils.data_loading_jax import load_yaml


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
        registry    : {id → {'callable', 'metadata'}} depuis discovery

    Returns:
        [{'id', 'fn', 'params': dict, 'metadata': dict}, ...]
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

        # Params inline depuis YAML (peuvent contenir des listes → produit cartésien)
        inline = item.get('params', {})

        list_params   = {k: v for k, v in inline.items() if isinstance(v, list)}
        scalar_params = {k: v for k, v in inline.items() if not isinstance(v, list)}

        if not list_params:
            resolved.append({
                'id'      : entity_id,
                'fn'      : fn,
                'params'  : scalar_params,
                'metadata': metadata,
            })
        else:
            for combo in product(*[
                [(k, v) for v in vals]
                for k, vals in list_params.items()
            ]):
                params = {**scalar_params, **dict(combo)}
                resolved.append({
                    'id'      : entity_id,
                    'fn'      : fn,
                    'params'  : params,
                    'metadata': metadata,
                })

    return resolved


def _resolve_gamma_axis(
    gamma_config  : Union[str, Dict, List],
    gamma_registry: Dict,
) -> List[Dict]:
    """
    Résout l'axe gamma — supporte séquences composées et pondération.

    Returns:
        [{'id', 'fn', 'params', 'metadata', 'is_composed': bool}, ...]
    """
    # Séquence multi-lignes avec pondération optionnelle
    if isinstance(gamma_config, dict) and 'sequence' in gamma_config:
        lines        = gamma_config['sequence']
        weights_cfg  = gamma_config.get('weights')
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

    Args:
        lines_config : Liste de configs (une par ligne)
        registry     : Registre gammas
        weights_cfg  : Config pondération optionnelle

    Returns:
        [{'id', 'fn', 'params', 'metadata', 'is_composed': True}, ...]
    """
    lines_resolved = [
        resolve_axis_atomic(line_cfg, registry)
        for line_cfg in lines_config
    ]

    sequences = []
    for combo in product(*lines_resolved):
        sequence_ids  = [e['id'] for e in combo]
        sequence_fns  = [e['fn'] for e in combo]
        shared_params = {}
        for e in combo:
            shared_params.update(e['params'])

        if len(combo) == 1:
            entry = combo[0]
            sequences.append({
                'id'         : entry['id'],
                'fn'         : entry['fn'],
                'params'     : entry['params'],
                'metadata'   : entry['metadata'],
                'is_composed': False,
            })
        else:
            composed_id = _format_gamma_id(sequence_ids)
            composed_fn = make_composed_gamma(sequence_fns)
            sequences.append({
                'id'         : composed_id,
                'fn'         : composed_fn,
                'params'     : shared_params,
                'metadata'   : {'id': composed_id, 'non_markovian': False,
                                'stochastic': False, 'differentiable': False},
                'is_composed': True,
            })

    # Pondération
    if weights_cfg:
        sequences_weighted = []
        for seq in sequences:
            n_gammas     = len(seq.get('sequence', [seq]))
            weights_grid = _generate_weights_grid(weights_cfg, n_gammas)
            for weights in weights_grid:
                fns         = [seq['fn']] if not seq['is_composed'] else seq['fn']
                composed_fn = make_composed_gamma(
                    fns if isinstance(fns, list) else [fns],
                    list(weights)
                )
                wid = _format_gamma_id(
                    [seq['id']] if not seq['is_composed'] else [seq['id']],
                    list(weights)
                )
                sequences_weighted.append({
                    'id'         : wid,
                    'fn'         : composed_fn,
                    'params'     : seq['params'],
                    'metadata'   : seq['metadata'],
                    'is_composed': True,
                    'weights'    : list(weights),
                })
        return sequences_weighted

    return sequences


# =============================================================================
# RANK EFFECTIF
# =============================================================================

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

    # Rank fixe dans metadata (SYM, ASY, R3)
    if 'rank' in metadata and metadata['rank'] is not None:
        return int(metadata['rank'])

    # Rank paramétrique (RN-*)
    return int(params.get('rank', 3))


# =============================================================================
# GÉNÉRATION GROUPES
# =============================================================================

def generate_groups(
    run_config : Dict[str, Any],
    registries : Dict[str, Dict],
    chunk_size : int = 256,
) -> List[Dict]:
    """
    Génère les groupes vmappables depuis un YAML de run.

    Un groupe = même clé de compilation XLA :
      (gamma_fn, enc_fn, mod_fn, n_dof, rank_eff, max_it)

    Chaque groupe contient une liste de runs :
      (gamma_params, enc_params, mod_params, key_CI, key_run)

    Args:
        run_config : Dict chargé depuis YAML de run
        registries : {'gamma': registry, 'encoding': registry, 'modifier': registry}
        chunk_size : Taille max d'un chunk vmap (défaut 256)

    Returns:
        Liste de groupes, chacun avec 'runs' prêts pour vmap
    """
    axes    = run_config.get('axes', {})
    phase   = run_config.get('phase', 'unknown')
    n_dofs  = _normalize_to_list(run_config.get('n_dof', 10))
    max_its = _normalize_to_list(run_config.get('max_iterations', 200))

    # Résolution axes
    gamma_config    = axes.get('gamma', 'all')
    encoding_config = axes.get('encoding', 'all')
    modifier_config = axes.get('modifier', [{'id': 'M0'}])

    gammas    = _resolve_gamma_axis(gamma_config, registries['gamma'])
    encodings = resolve_axis_atomic(encoding_config, registries['encoding'])
    modifiers = resolve_axis_atomic(modifier_config, registries['modifier'])

    # Seeds globaux
    seed_CIs  = _normalize_to_list(run_config.get('seed_CI', [None]))
    seed_runs = _normalize_to_list(run_config.get('seed_run', [None]))

    # Construction groupes
    groups_map = {}

    for gamma_e, enc_e, mod_e, n_dof, max_it in product(
        gammas, encodings, modifiers, n_dofs, max_its
    ):
        rank_eff = _get_rank_eff(enc_e)

        group_key = (
            gamma_e['id'], enc_e['id'], mod_e['id'],
            n_dof, rank_eff, max_it,
        )

        if group_key not in groups_map:
            groups_map[group_key] = {
                'gamma_id'      : gamma_e['id'],
                'gamma_fn'      : gamma_e['fn'],
                'enc_id'        : enc_e['id'],
                'enc_fn'        : enc_e['fn'],
                'mod_id'        : mod_e['id'],
                'mod_fn'        : mod_e['fn'],
                'n_dof'         : n_dof,
                'rank_eff'      : rank_eff,
                'max_it'        : max_it,
                'non_markovian' : bool(
                    gamma_e['metadata'].get('non_markovian', False)
                ),
                'phase'         : phase,
                'runs'          : [],
            }

        grp = groups_map[group_key]

        # Runs : produit cartésien seeds × params déjà résolus dans les entries
        for seed_CI, seed_run in product(seed_CIs, seed_runs):
            key_CI  = _seed_to_key(seed_CI)
            key_run = _make_run_key(
                _seed_to_key(seed_run),
                gamma_e['id'],
                enc_e['id'],
            )

            grp['runs'].append({
                'gamma_params': gamma_e['params'],
                'enc_params'  : enc_e['params'],
                'mod_params'  : mod_e['params'],
                'seed_CI'     : seed_CI,
                'seed_run'    : seed_run,
                'key_CI'      : key_CI,
                'key_run'     : key_run,
            })

    return list(groups_map.values())


# =============================================================================
# CHUNKING
# =============================================================================

def chunk_group(group: Dict, chunk_size: int) -> List[Dict]:
    """
    Découpe les runs d'un groupe en chunks vmappables.

    Args:
        group      : Un groupe depuis generate_groups()
        chunk_size : Taille max d'un chunk

    Returns:
        Liste de chunks — même structure que group,
        avec 'runs' remplacé par une sous-liste.
    """
    runs   = group['runs']
    chunks = []
    for i in range(0, len(runs), chunk_size):
        chunk = {**group, 'runs': runs[i:i + chunk_size]}
        chunks.append(chunk)
    return chunks


# =============================================================================
# GÉNÉRATEUR DE CHUNKS (streaming — 1 groupe en RAM à la fois)
# =============================================================================

def generate_chunks(
    run_config : Dict[str, Any],
    registries : Dict[str, Dict],
    chunk_size : int = 256,
):
    """
    Générateur de chunks vmappables — version streaming.

    Contrairement à generate_groups(), ne matérialise jamais plus d'un groupe
    à la fois en RAM. Chaque groupe est yielded chunk par chunk puis libéré (GC).

    Stratégie :
      Outer loop = clés XLA structurelles
        (gamma_id, enc_id, rank_eff, mod_id, n_dof, max_it)
      Inner loop = variants scalaires (params déjà résolus) × seeds
        → même clé XLA, vmappés ensemble

    Garantie mémoire :
      - runs du groupe courant accumulés (dicts légers, pas de JAX arrays)
      - chunks yieldés dès que le groupe est complet
      - `del runs` libère le groupe avant le suivant
      - JAX arrays (D_batch, signals, last_states) créés uniquement dans hub_running

    Yields:
        chunk dict — même structure que chunk_group() :
        {
            'gamma_id', 'gamma_fn', 'enc_id', 'enc_fn',
            'mod_id', 'mod_fn', 'n_dof', 'rank_eff', 'max_it',
            'non_markovian', 'phase',
            'runs': [list de run dicts, taille <= chunk_size]
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

    seed_CIs  = _normalize_to_list(run_config.get('seed_CI', [None]))
    seed_runs = _normalize_to_list(run_config.get('seed_run', [None]))

    # Indexer par clé structurelle pour outer loop
    gammas_by_id: Dict[str, List] = {}
    for g in gammas:
        gammas_by_id.setdefault(g['id'], []).append(g)

    encs_by_key: Dict[tuple, List] = {}
    for e in encodings:
        key = (e['id'], _get_rank_eff(e))
        encs_by_key.setdefault(key, []).append(e)

    mods_by_id: Dict[str, List] = {}
    for m in modifiers:
        mods_by_id.setdefault(m['id'], []).append(m)

    # Outer : produit cartésien des clés structurelles × n_dof × max_it
    for (gamma_id, gamma_entries), (enc_key, enc_entries), (mod_id, mod_entries), n_dof, max_it \
        in product(
            gammas_by_id.items(),
            encs_by_key.items(),
            mods_by_id.items(),
            n_dofs,
            max_its,
        ):
        enc_id, rank_eff = enc_key

        # Représentants pour les champs fixes du groupe
        g_rep = gamma_entries[0]
        e_rep = enc_entries[0]
        m_rep = mod_entries[0]

        group_header = {
            'gamma_id'     : gamma_id,
            'gamma_fn'     : g_rep['fn'],
            'enc_id'       : enc_id,
            'enc_fn'       : e_rep['fn'],
            'mod_id'       : mod_id,
            'mod_fn'       : m_rep['fn'],
            'n_dof'        : n_dof,
            'rank_eff'     : rank_eff,
            'max_it'       : max_it,
            'non_markovian': bool(g_rep['metadata'].get('non_markovian', False)),
            'phase'        : phase,
        }

        # Inner : accumulation légère des runs
        runs: List[Dict] = []
        for gamma_e, enc_e, mod_e, seed_CI, seed_run in product(
            gamma_entries, enc_entries, mod_entries, seed_CIs, seed_runs
        ):
            key_CI  = _seed_to_key(seed_CI)
            key_run = _make_run_key(_seed_to_key(seed_run), gamma_id, enc_id)
            runs.append({
                'gamma_params': gamma_e['params'],
                'enc_params'  : enc_e['params'],
                'mod_params'  : mod_e['params'],
                'seed_CI'     : seed_CI,
                'seed_run'    : seed_run,
                'key_CI'      : key_CI,
                'key_run'     : key_run,
            })

        # Yield chunk par chunk, puis libérer
        for i in range(0, len(runs), chunk_size):
            yield {**group_header, 'runs': runs[i:i + chunk_size]}

        del runs


# =============================================================================
# STATS + DRY RUN
# =============================================================================

def count_stats(groups: List[Dict], chunk_size: int = 256) -> Dict:
    """
    Calcule les statistiques du plan d'exécution.

    Returns:
        {
            'n_groups'  : int  — compilations XLA max
            'n_runs'    : int  — lignes parquet totales
            'n_chunks'  : int  — appels vmap totaux
        }
    """
    n_runs   = sum(len(g['runs']) for g in groups)
    n_chunks = sum(ceil(len(g['runs']) / chunk_size) for g in groups)
    return {
        'n_groups' : len(groups),
        'n_runs'   : n_runs,
        'n_chunks' : n_chunks,
    }


def dry_run(
    run_config : Dict[str, Any],
    registries : Dict[str, Dict],
    chunk_size : int = 256,
) -> bool:
    """
    Affiche le plan d'exécution et demande confirmation.

    Returns:
        True si l'utilisateur confirme, False sinon.
    """
    groups = generate_groups(run_config, registries, chunk_size)
    stats  = count_stats(groups, chunk_size)
    phase  = run_config.get('phase', 'unknown')

    print()
    print("=" * 50)
    print(f"  Dry run — phase : {phase}")
    print("=" * 50)
    print(f"  Groupes   (compilations XLA max) : {stats['n_groups']}")
    print(f"  Runs      (lignes parquet)        : {stats['n_runs']}")
    print(f"  Chunks    (appels vmap)           : {stats['n_chunks']}")
    print(f"  Chunk size                        : {chunk_size}")
    print("=" * 50)
    print()

    answer = input("  Continuer ? (o/n) : ").strip().lower()
    return answer == 'o'