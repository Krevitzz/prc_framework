"""
running/hub_running.py

Hub running — orchestration batch JAX.
Point de jonction : compositions → vmap → parquet.

Flux :
  YAML → dry_run arithmétique → generate_chunks (streaming) → vmap → parquet

Convention hub :
  - Importe uniquement depuis running/ et hubs inférieurs
  - Zéro logique métier PRC — orchestration pure
  - Aveugle au contenu des atomics (dispatch via prepare_params optionnel)

Statuts run :
  OK        — run nominal
  EXPLOSION — run terminé, features contiennent inf (comportement système, P2 charter)
  INVALID   — incompatibilité structurelle (rank_constraint), détectée avant vmap
  FAIL      — exception inattendue après fallback individuel

Flush :
  ParquetWriter ouvert pendant toute la phase (zéro réécriture fichier).
  Flush tous les flush_every rows (défaut 1000).
  Flush final + fermeture propre dans finally.

Mémoire :
  dry_run  : arithmétique pur — zéro instanciation de groupes ou runs
  hot loop : generate_chunks() — générateur, 1 groupe en RAM à la fois
"""

import concurrent.futures
import gc
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from compositions.compositions_jax import (
    chunk_group,
    generate_groups_lazy,
    generate_chunks,
)
from running.run_one_jax import _run_jit
from featuring.hub_featuring import post_scan, FEATURE_NAMES
from featuring.jax_features import FEATURES_STRUCTURAL_NAN
from utils.data_loading_jax import (
    discover_gammas_jax,
    discover_encodings_jax,
    discover_modifiers_jax,
    load_yaml,
)



# =============================================================================
# DRY RUN ARITHMÉTIQUE (zéro instanciation)
# =============================================================================

def _normalize_list(val) -> list:
    """Normalise un scalaire ou liste → liste."""
    if val is None:
        return [None]
    return val if isinstance(val, list) else [val]


def _count_axis_entries(axis_list) -> tuple:
    """
    Compte entrées et IDs uniques d'un axe depuis le YAML brut.
    Pas de registry, pas de product() — arithmétique pure.

    Returns:
        (n_entries, n_unique_ids)
        n_entries = somme des produits cartésiens de params par item
    """
    if not axis_list or axis_list == 'all':
        return 0, 0

    total, ids = 0, set()
    for item in (axis_list if isinstance(axis_list, list) else [axis_list]):
        if isinstance(item, str):
            item = {'id': item}
        ids.add(item.get('id', '?'))
        params = item.get('params', {})
        combos = 1
        for v in params.values():
            if isinstance(v, list):
                combos *= len(v)
        total += max(combos, 1)

    return total, len(ids)


def dry_run_stats(run_config: Dict, chunk_size: int = 256) -> Dict:
    """
    Calcule les métriques du plan d'exécution depuis le YAML uniquement.

    Zéro instanciation de groupes ou de runs — arithmétique pure.
    Durée < 100ms quelle que soit la taille du plan.

    Returns:
        {
            'n_runs'          : int    — lignes parquet totales (estimation)
            'n_groups_est'    : int    — compilations XLA estimées
            'worst_state_mb'  : float  — RAM worst-case state unique (MB)
            'vmap_ram_mb'     : float  — RAM worst-case par call vmap (MB)
            'parquet_est_mb'  : float  — taille parquet estimée (MB)
            'worst_shape'     : str    — shape worst-case
        }
    """
    axes    = run_config.get('axes', {})
    n_dofs  = _normalize_list(run_config.get('n_dof', 10))
    max_its = _normalize_list(run_config.get('max_iterations', 200))
    seeds   = (
        len(_normalize_list(run_config.get('seed_CI',  [None]))) *
        len(_normalize_list(run_config.get('seed_run', [None])))
    )

    n_ge, n_gi = _count_axis_entries(axes.get('gamma',    []))
    n_ee, n_ei = _count_axis_entries(axes.get('encoding', []))
    n_me, n_mi = _count_axis_entries(axes.get('modifier', [{'id': 'M0'}]))

    n_dof_v  = len(n_dofs)
    n_max_v  = len(max_its)
    n_runs   = n_ge * n_ee * n_me * seeds * n_dof_v * n_max_v
    # Groupes ≈ IDs uniques × enc_entries (rank_eff varie) × mod_ids × axes structurels
    n_groups = n_gi * n_ee * n_mi * n_dof_v * n_max_v

    # Worst-case RAM : plus grand n_dof × plus grand rank trouvé dans les encodings
    worst_dof  = max((v for v in n_dofs if v is not None), default=10)
    worst_rank = 2
    for item in (axes.get('encoding', []) or []):
        if isinstance(item, dict):
            r = item.get('params', {}).get('rank', 2)
            worst_rank = max(worst_rank, max(r) if isinstance(r, list) else r)

    worst_elems      = worst_dof ** worst_rank
    state_mb         = worst_elems * 4 / 1024 / 1024   # float32
    worst_max_it     = max((v for v in max_its if v is not None), default=200)
    # RAM vmap : chunk_size states simultanés (lax.scan ne matérialise pas l'historique)
    # signals accumulés = chunk_size × max_it × n_features scalaires — négligeable
    vmap_ram_mb      = chunk_size * worst_elems * 4 / 1024 / 1024
    parquet_est_mb   = n_runs * 600 / 1024 / 1024       # ~600 bytes/row empirique

    return {
        'n_runs'        : n_runs,
        'n_groups_est'  : n_groups,
        'worst_shape'   : f"({'×'.join([str(worst_dof)]*worst_rank)})",
        'worst_state_mb': round(state_mb, 3),
        'vmap_ram_mb'   : round(vmap_ram_mb, 1),
        'parquet_est_mb': round(parquet_est_mb, 1),
    }


# =============================================================================
# STATUTS RUN + HELPERS
# =============================================================================

# Sentinel features pour runs INVALID / FAIL
_NAN_FEATURES = {f: float('nan') for f in FEATURE_NAMES}


def _check_invalid(group: Dict, gamma_registry: Dict) -> bool:
    """
    Détecte statiquement une incompatibilité structurelle.

    Vérifie rank_constraint du gamma contre rank_eff du groupe.
    Pas d'appel JAX — O(1).

    Returns:
        True si le groupe entier est INVALID (à écrire sans vmap)
    """
    gamma_id   = group['gamma_id']
    rank_eff   = group['rank_eff']
    n_dof      = group['n_dof']

    entry      = gamma_registry.get(gamma_id, {})
    metadata   = entry.get('metadata', {})
    rc         = metadata.get('rank_constraint')

    if rc == 2 and rank_eff >= 3:
        return True
    if rc == 'square' and rank_eff >= 3:
        return True
        return True

    return False


def _run_status_from_features(features: Dict) -> str:
    """
    Détermine run_status depuis les features d'un run individuel.

    OK        — features finies, comportement nominal
    EXPLOSION — au moins un Inf (divergence réelle, P2 charter : Inf = information)
    NAN_ALL   — toutes features non-structurelles NaN, zéro Inf
                ("rien de mesurable" — distinct d'une explosion)
    """
    has_inf     = False
    non_structural_vals = []

    for k, v in features.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isinf(fv):
            has_inf = True
        # Exclure features structurellement conditionnelles + health flags
        if k not in FEATURES_STRUCTURAL_NAN and not k.startswith('health_'):
            non_structural_vals.append(fv)

    if has_inf:
        return 'EXPLOSION'

    if non_structural_vals and all(np.isnan(v) for v in non_structural_vals):
        return 'NAN_ALL'

    return 'OK'




def _build_gamma_params_batch(
    chunk     : Dict,
    group     : Dict,
    registries: Dict,
) -> dict:
    """
    Construit gamma_params batché pour vmap.

    Appelle prepare_params(raw_params, n_dof, key_CI) si disponible
    dans le module gamma — permet de pré-calculer W, etc. sans
    que hub_running connaisse les internals du gamma.

    Args:
        chunk      : Un chunk depuis chunk_group()
        group      : Le groupe parent
        registries : {'gamma': {id: {'callable', 'metadata', 'module'}}}

    Returns:
        dict {param_name: jnp.array(B, ...)} — prêt pour vmap in_axes=0
        ou {} si aucun param
    """
    gamma_id  = group['gamma_id']
    n_dof     = group['n_dof']
    runs      = chunk['runs']

    # Récupérer module pour prepare_params optionnel
    gamma_entry  = registries['gamma'].get(gamma_id, {})
    prepare_fn   = gamma_entry.get('prepare_params', None)

    params_list = []
    for run in runs:
        raw = run['gamma_params']
        if prepare_fn is not None:
            p = prepare_fn(raw, n_dof, run['key_CI'])
        else:
            p = raw
        params_list.append(p)

    if not params_list or not params_list[0]:
        return {}

    # Stack par clé
    keys = params_list[0].keys()
    return {
        k: jnp.stack([p[k] for p in params_list])
        for k in keys
    }


def _build_enc_batch(chunk: Dict, group: Dict) -> jnp.ndarray:
    """
    Construit D_batch en appelant enc_fn pour chaque run du chunk.

    Returns:
        jnp.ndarray (B, n_dof, ...) selon rank_eff
    """
    enc_fn = group['enc_fn']
    n_dof  = group['n_dof']

    D_list = [
        enc_fn(n_dof, run['enc_params'], run['key_CI'])
        for run in chunk['runs']
    ]
    return jnp.stack(D_list)


def _build_mod_params_batch(chunk: Dict) -> dict:
    """
    Construit mod_params batché pour vmap.

    M0 ({}) → retourne {} directement (pas de vmap sur params vides).

    Returns:
        dict {param_name: jnp.array(B, ...)} ou {}
    """
    runs = chunk['runs']
    if not runs[0]['mod_params']:
        return {}

    keys = runs[0]['mod_params'].keys()
    return {
        k: jnp.stack([run['mod_params'][k] for run in runs])
        for k in keys
    }


def _build_keys_batch(chunk: Dict) -> jnp.ndarray:
    """
    Construit keys_batch depuis key_run de chaque run.

    Returns:
        jnp.ndarray (B, 2)
    """
    return jnp.stack([run['key_run'] for run in chunk['runs']])


# =============================================================================
# in_axes DYNAMIQUE
# =============================================================================

def _in_axes_for_params(params_batch: dict):
    """
    Construit in_axes compatible vmap depuis un params_batch.

    dict non vide → {clé: 0, ...}  (batch sur dim 0)
    dict vide     → {}              (pas de vmap — M0)
    """
    if not params_batch:
        return {}
    return {k: 0 for k in params_batch}


# =============================================================================
# AOT PRE-COMPILATION
# =============================================================================

def _compile_group_aot(
    group      : Dict,
    registries : Dict,
    dmd_rank   : int,
    chunk_size : int,
) -> Optional[Dict]:
    """
    AOT-compile le kernel vmappé pour un groupe.

    Appelée dans un thread worker (ThreadPoolExecutor) — thread-safe.
    Compile un binaire par taille de chunk distincte du groupe
    (chunk_size pour les chunks pleins, reste pour le dernier si different).

    Utilise prepare_params si disponible dans le module gamma — critique pour
    les gammas dont les shapes preparees different des shapes brutes (ex: GAM-011 W).

    Returns:
        {n_batch: compiled_fn, ...}  — appel direct (gp_b, mp_b, D_b, keys_b)
        None si echec (la hot loop bascule sur JIT standard)
    """
    gamma_id  = group['gamma_id']
    n_dof     = group['n_dof']
    max_it    = group['max_it']
    dmd_rank  = dmd_rank
    is_diff   = registries['gamma'].get(
        gamma_id, {}
    ).get('metadata', {}).get('differentiable', True)

    gamma_entry = registries['gamma'].get(gamma_id, {})
    prepare_fn  = gamma_entry.get('prepare_params', None)

    # Fonction avec statiques fermes dans la closure
    gamma_fn = group['gamma_fn']
    mod_fn   = group['mod_fn']

    def _run_dynamic(gp, mp, D, key):
        return _run_jit(gamma_fn, gp, mod_fn, mp, D, key, max_it, dmd_rank, is_diff)

    # Tailles de chunks distinctes pour ce groupe
    chunks = chunk_group(group, chunk_size)
    sizes  = sorted({len(c['runs']) for c in chunks})

    compiled_by_n = {}

    for n_batch in sizes:
        # Construire un exemple batch de taille n_batch avec prepare_params
        runs_ex = (group['runs'] * n_batch)[:n_batch]

        # gamma_params avec prepare_params si disponible
        params_list = []
        for run in runs_ex:
            raw = run['gamma_params']
            p   = prepare_fn(raw, n_dof, run['key_CI']) if prepare_fn else raw
            params_list.append(p)

        if params_list and params_list[0]:
            gp_ex = {k: jnp.stack([p[k] for p in params_list]) for k in params_list[0]}
            in_g  = {k: 0 for k in gp_ex}
        else:
            gp_ex = {}
            in_g  = {}

        # mod_params
        mp_list = [run['mod_params'] for run in runs_ex]
        if mp_list and mp_list[0]:
            mp_ex = {k: jnp.stack([p[k] for p in mp_list]) for k in mp_list[0]}
            in_m  = {k: 0 for k in mp_ex}
        else:
            mp_ex = {}
            in_m  = {}

        # D_batch
        D_ex = jnp.stack([
            group['enc_fn'](n_dof, run['enc_params'], run['key_CI'])
            for run in runs_ex
        ])

        # keys_batch
        keys_ex = jnp.stack([run['key_run'] for run in runs_ex])

        try:
            vmapped  = jax.vmap(_run_dynamic, in_axes=(in_g, in_m, 0, 0))
            lowered  = jax.jit(vmapped).lower(gp_ex, mp_ex, D_ex, keys_ex)
            compiled_by_n[n_batch] = lowered.compile()
        except Exception as e:
            warnings.warn(
                f"[AOT compile FAIL] {gamma_id} n_dof={n_dof} max_it={max_it} "
                f"n_batch={n_batch} : {e} — fallback JIT"
            )
            return None

    return compiled_by_n if compiled_by_n else None


def _run_chunk_aot(
    chunk         : Dict,
    group         : Dict,
    compiled_by_n : Dict,
    registries    : Dict,
    dmd_rank      : int,
    is_diff       : bool,
) -> Tuple:
    """
    Execute un chunk via le binaire AOT pre-compile.

    Interface identique au bloc vmap de la hot loop JIT :
    retourne (signals_b, last_states, A_k_b, P_k_b).

    Construit les batches avec prepare_params si disponible.
    """
    gamma_id   = group['gamma_id']
    n_dof      = group['n_dof']
    runs       = chunk['runs']
    n_batch    = len(runs)

    gamma_entry = registries['gamma'].get(gamma_id, {})
    prepare_fn  = gamma_entry.get('prepare_params', None)

    # gamma_params avec prepare_params
    params_list = []
    for run in runs:
        raw = run['gamma_params']
        p   = prepare_fn(raw, n_dof, run['key_CI']) if prepare_fn else raw
        params_list.append(p)

    if params_list and params_list[0]:
        gp_b = {k: jnp.stack([p[k] for p in params_list]) for k in params_list[0]}
    else:
        gp_b = {}

    # mod_params
    mp_list = [run['mod_params'] for run in runs]
    if mp_list and mp_list[0]:
        mp_b = {k: jnp.stack([p[k] for p in mp_list]) for k in mp_list[0]}
    else:
        mp_b = {}

    D_b    = _build_enc_batch(chunk, group)
    keys_b = _build_keys_batch(chunk)

    fn = compiled_by_n[n_batch]
    signals_b, last_states, A_k_b, P_k_b = fn(gp_b, mp_b, D_b, keys_b)
    return signals_b, last_states, A_k_b, P_k_b


# =============================================================================
# POST-SCAN + ROWS
# =============================================================================

def _rows_from_chunk(
    chunk        : Dict,
    group        : Dict,
    signals_batch: Dict,
    last_states  : jnp.ndarray,
    A_k_batch    : jnp.ndarray,
    P_k_batch    : jnp.ndarray,
    run_status   : str = 'OK',
) -> List[Dict]:
    """
    Construit la liste de rows depuis les résultats vmap d'un chunk.

    run_status peut être forcé ('INVALID') ou dérivé par run depuis les features.

    Returns:
        [{'composition': {...}, 'features': {str: float}, 'run_status': str}, ...]
    """
    rows = []
    for i, run in enumerate(chunk['runs']):
        if run_status in ('INVALID', 'FAIL'):
            features = dict(_NAN_FEATURES)
            status   = run_status
        else:
            features = post_scan(
                {k: signals_batch[k][i] for k in signals_batch},
                last_states[i],
                A_k_batch[i],
                P_k_batch[i],
            )
            status = _run_status_from_features(features)

        rows.append({
            'composition': {
                'gamma_id'       : group['gamma_id'],
                'encoding_id'    : group['enc_id'],
                'modifier_id'    : group['mod_id'],
                'n_dof'          : group['n_dof'],
                'rank_eff'       : group['rank_eff'],
                'max_it'         : group['max_it'],
                'gamma_params'   : run['gamma_params'],
                'encoding_params': run['enc_params'],
                'modifier_params': run['mod_params'],
                'seed_CI'        : run['seed_CI'],
                'seed_run'       : run['seed_run'],
            },
            'features'  : features,
            'run_status': status,
        })
    return rows


def _rows_invalid(chunk: Dict, group: Dict) -> List[Dict]:
    """Génère des rows INVALID pour tout un chunk (incompatibilité structurelle)."""
    return [
        {
            'composition': {
                'gamma_id'       : group['gamma_id'],
                'encoding_id'    : group['enc_id'],
                'modifier_id'    : group['mod_id'],
                'n_dof'          : group['n_dof'],
                'rank_eff'       : group['rank_eff'],
                'max_it'         : group['max_it'],
                'gamma_params'   : run['gamma_params'],
                'encoding_params': run['enc_params'],
                'modifier_params': run['mod_params'],
                'seed_CI'        : run['seed_CI'],
                'seed_run'       : run['seed_run'],
            },
            'features'  : dict(_NAN_FEATURES),
            'run_status': 'INVALID',
        }
        for run in chunk['runs']
    ]


def _fallback_individual(
    chunk            : Dict,
    group            : Dict,
    registries       : Dict,
    dmd_rank         : int  = 16,
    is_differentiable: bool = True,
) -> List[Dict]:
    """
    Fallback individuel sur vmap failure.

    Tente chaque run séparément via _run_jit scalaire.
    Produit 'OK', 'EXPLOSION' ou 'FAIL' par run — zéro skip.
    """
    rows     = []
    gamma_id = group['gamma_id']
    n_dof    = group['n_dof']

    gamma_entry = registries['gamma'].get(gamma_id, {})
    prepare_fn  = gamma_entry.get('prepare_params', None)

    for run in chunk['runs']:
        try:
            raw_gp = run['gamma_params']
            gp     = prepare_fn(raw_gp, n_dof, run['key_CI']) if prepare_fn else raw_gp

            D = group['enc_fn'](n_dof, run['enc_params'], run['key_CI'])

            signals, last_state, A_k, P_k = _run_jit(
                group['gamma_fn'], gp,
                group['mod_fn'],  run['mod_params'],
                D, run['key_run'], group['max_it'], dmd_rank, is_differentiable,
            )

            features = post_scan(
                {k: signals[k] for k in signals},
                last_state,
                A_k,
                P_k,
            )
            status = _run_status_from_features(features)

        except Exception as exc:
            features          = dict(_NAN_FEATURES)
            features['_exc']  = str(exc)   # stocké temporairement, retiré avant parquet
            status            = 'FAIL'

        rows.append({
            'composition': {
                'gamma_id'       : gamma_id,
                'encoding_id'    : group['enc_id'],
                'modifier_id'    : group['mod_id'],
                'n_dof'          : n_dof,
                'rank_eff'       : group['rank_eff'],
                'max_it'         : group['max_it'],
                'gamma_params'   : run['gamma_params'],
                'encoding_params': run['enc_params'],
                'modifier_params': run['mod_params'],
                'seed_CI'        : run['seed_CI'],
                'seed_run'       : run['seed_run'],
            },
            'features'  : features,
            'run_status': status,
        })

    return rows


# =============================================================================
# PARQUET WRITER (append incrémental)
# =============================================================================

def _rows_to_table(rows: List[Dict], phase: str) -> pa.Table:
    """
    Convertit rows en pyarrow Table.

    Schéma :
        run_status + colonnes composition + colonnes features plates
    """
    import json

    records = []
    for row in rows:
        comp     = row['composition']
        features = row['features']
        record   = {
            'run_status'     : row.get('run_status', 'OK'),
            'phase'          : phase,
            'gamma_id'       : comp.get('gamma_id', ''),
            'encoding_id'    : comp.get('encoding_id', ''),
            'modifier_id'    : comp.get('modifier_id', ''),
            'n_dof'          : comp.get('n_dof', 0),
            'rank_eff'       : comp.get('rank_eff', 2),
            'max_it'         : comp.get('max_it', 0),
            'gamma_params'   : json.dumps(comp.get('gamma_params', {})),
            'encoding_params': json.dumps(comp.get('encoding_params', {})),
            'modifier_params': json.dumps(comp.get('modifier_params', {})),
            'seed_CI'        : comp.get('seed_CI'),
            'seed_run'       : comp.get('seed_run'),
            **features,
        }
        records.append(record)

    return pa.Table.from_pylist(records)


def _flush_buffer(
    buffer      : List[Dict],
    writer_state: Dict,
    phase       : str,
    output_dir  : Path,
) -> Dict:
    """
    Flush le buffer vers le ParquetWriter.

    Premier flush : crée le writer et définit le schéma.
    Flushes suivants : append sur le writer existant.

    Args:
        buffer       : Liste de rows à écrire
        writer_state : {'writer': PqWriter|None, 'path': Path|None, 'n_written': int}
        phase        : Nom phase
        output_dir   : Dossier output

    Returns:
        writer_state mis à jour
    """
    if not buffer:
        return writer_state

    table = _rows_to_table(buffer, phase)

    if writer_state['writer'] is None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f'{phase}.parquet'

        writer = pq.ParquetWriter(str(filepath), table.schema)
        writer_state['writer'] = writer
        writer_state['path']   = filepath

    writer_state['writer'].write_table(table)
    writer_state['n_written'] += len(buffer)

    return writer_state


# =============================================================================
# INTERFACE PUBLIQUE
# =============================================================================

def run_batch_jax(
    yaml_path   : Path,
    output_dir  : Path = Path('data/results'),
    auto_confirm: bool = False,
    chunk_size  : int  = 256,
    flush_every : int  = 1000,
    verbose     : bool = False,
) -> Dict[str, Any]:
    """
    Orchestration batch complète : YAML → parquet.

    Dry-run : arithmétique pur (< 100ms, zéro instanciation).
    Hot loop : generate_chunks() — générateur, 1 groupe en RAM à la fois.
    Statuts  : OK / EXPLOSION / INVALID / FAIL — zéro skip.
    Progress : stderr, une ligne par groupe XLA.

    Args:
        yaml_path    : Path vers le YAML de run
        output_dir   : Dossier output parquet
        auto_confirm : True → skip confirmation (tests, CI)
        chunk_size   : Taille max d'un chunk vmap (défaut 256)
        flush_every  : Flush parquet tous les N rows (défaut 1000)
        verbose      : Affiche détails FAIL et exceptions

    Returns:
        {
            'phase'      : str,
            'n_ok'       : int,
            'n_explosion': int,
            'n_invalid'  : int,
            'n_fail'     : int,
            'n_groups'   : int,
            'parquet'    : Path | None,
        }
    """
    run_config = load_yaml(yaml_path)
    phase      = run_config.get('phase', 'unknown')
    dmd_rank   = int(run_config.get('dmd_rank', 16))   # statique XLA — wiper cache si changé
    aot_window  = int(run_config.get('aot_window',  80))    # groupes compilés d'avance
    aot_workers = int(run_config.get('aot_workers', 16))   # threads compilateurs
    aot_ram_gb  = float(run_config.get('aot_ram_gb', 4.0)) # plafond RAM binaires (Go)

    # ------------------------------------------------------------------
    # Dry run arithmétique — zéro instanciation
    # ------------------------------------------------------------------
    stats = dry_run_stats(run_config, chunk_size)

    print()
    print("=" * 56)
    print(f"  Dry run — phase : {phase}")
    print("=" * 56)
    print(f"  Runs estimés      (lignes parquet)  : {stats['n_runs']:>10,}")
    print(f"  Groupes estimés   (compilations XLA) : {stats['n_groups_est']:>10,}")
    print(f"  Worst-case state  {stats['worst_shape']:<20}  : {stats['worst_state_mb']:>7.3f} MB")
    print(f"  RAM/vmap call     (chunk={chunk_size})         : {stats['vmap_ram_mb']:>7.1f} MB")
    print(f"  Parquet estimé                       : {stats['parquet_est_mb']:>7.1f} MB")
    print("=" * 56)
    print()

    if not auto_confirm:
        answer = input("  Continuer ? (o/n) : ").strip().lower()
        if answer != 'o':
            return {
                'phase': phase, 'n_ok': 0, 'n_explosion': 0,
                'n_invalid': 0, 'n_fail': 0, 'n_groups': 0, 'parquet': None,
            }

    # ------------------------------------------------------------------
    # Registries + prepare_params
    # ------------------------------------------------------------------
    registries = {
        'gamma'   : discover_gammas_jax(),
        'encoding': discover_encodings_jax(),
        'modifier': discover_modifiers_jax(),
    }

    import importlib
    for gid, entry in registries['gamma'].items():
        module_name = entry['callable'].__module__
        try:
            mod  = importlib.import_module(module_name)
            prep = getattr(mod, 'prepare_params', None)
            if prep is not None:
                entry['prepare_params'] = prep
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Hot loop — générateur streaming + statuts
    # ------------------------------------------------------------------
    writer_state     = {'writer': None, 'path': None, 'n_written': 0}
    rows_buffer      = []
    rows_since_flush = 0
    n_ok             = 0
    n_explosion   = 0
    n_invalid     = 0
    n_fail        = 0
    n_groups_seen = 0
    last_xla_key  = None
    t_start       = time.time()
    n_runs_est    = stats['n_runs']   # pour % progression
    backup_every  = max(flush_every * 10, 10_000)  # backup parquet tous les N rows

    # Helpers internes partagés JIT et AOT
    def _progress(n_grp):
        n_done  = n_ok + n_explosion + n_invalid + n_fail
        pct     = n_done / n_runs_est * 100 if n_runs_est > 0 else 0
        elapsed = time.time() - t_start
        eta_str = ''
        if pct > 1 and elapsed > 0:
            eta_s   = elapsed / pct * (100 - pct)
            eta_str = f"  ETA {int(eta_s//60)}m{int(eta_s%60):02d}s"
        print(
            f"\r  [{n_grp:>5}g | {pct:5.1f}% | "
            f"{int(elapsed//60)}m{int(elapsed%60):02d}s{eta_str}]  "
            f"OK={n_ok}  EXP={n_explosion}  INV={n_invalid}  FAIL={n_fail}",
            end='', file=sys.stderr
        )

    def _process_chunk_results(chunk, group, signals_b, last_states, A_k_b, P_k_b):
        nonlocal n_ok, n_explosion
        last_states.block_until_ready()
        rows = _rows_from_chunk(chunk, group, signals_b, last_states, A_k_b, P_k_b)
        del signals_b, last_states, A_k_b, P_k_b
        for r in rows:
            if r['run_status'] == 'EXPLOSION': n_explosion += 1
            else:                              n_ok        += 1
        return rows

    def _process_chunk_fallback(chunk, group, is_diff):
        nonlocal n_ok, n_explosion, n_fail
        rows = _fallback_individual(chunk, group, registries, dmd_rank, is_diff)
        for r in rows:
            s = r['run_status']
            if s == 'OK':          n_ok        += 1
            elif s == 'EXPLOSION': n_explosion += 1
            else:                  n_fail      += 1
        return rows

    def _vmap_chunk(chunk, group):
        is_diff_g      = registries['gamma'].get(
            group['gamma_id'], {}
        ).get('metadata', {}).get('differentiable', True)
        D_batch        = _build_enc_batch(chunk, group)
        gamma_params_b = _build_gamma_params_batch(chunk, group, registries)
        mod_params_b   = _build_mod_params_batch(chunk)
        keys_batch     = _build_keys_batch(chunk)
        in_axes_gamma  = _in_axes_for_params(gamma_params_b)
        in_axes_mod    = _in_axes_for_params(mod_params_b)
        run_batch_fn   = jax.vmap(
            _run_jit,
            in_axes=(None, in_axes_gamma, None, in_axes_mod, 0, 0, None, None, None)
        )
        s_b, ls, ak, pk = run_batch_fn(
            group['gamma_fn'], gamma_params_b,
            group['mod_fn'],   mod_params_b,
            D_batch, keys_batch, group['max_it'], dmd_rank, is_diff_g,
        )
        del D_batch
        return s_b, ls, ak, pk, is_diff_g

    try:
        if aot_window > 0:
            # ── HOT LOOP AOT ROLLING — streaming + N compilateurs parallèles ──
            #
            # Principe :
            #   - generate_groups_lazy() : streaming, 1 groupe en RAM à la fois
            #   - ThreadPoolExecutor(aot_workers) : N compilations XLA parallèles
            #   - deque pending (FIFO) : max aot_window groupes en vol
            #   - RAM binaires bornée par aot_ram_gb (plafond souple)
            #   - GPU démarre dès que la compile du groupe courant est prête
            #   - CPU ne s'arrête jamais de compiler tant que pending < window

            from collections import deque

            AVG_BINARY_MB   = 5.0   # estimation fixe (5MB/binaire mesuré sur bench)
            ram_binaries_gb = 0.0
            pending         = deque()   # (group, future)
            groups_gen      = generate_groups_lazy(run_config, registries)

            def _submit(group):
                nonlocal ram_binaries_gb
                f = pool.submit(_compile_group_aot, group, registries, dmd_rank, chunk_size)
                pending.append((group, f))
                ram_binaries_gb += AVG_BINARY_MB / 1024.0

            def _consume():
                nonlocal ram_binaries_gb, n_groups_seen
                group, future = pending.popleft()
                compiled_by_n = future.result()
                ram_binaries_gb = max(0.0, ram_binaries_gb - AVG_BINARY_MB / 1024.0)
                n_groups_seen += 1
                _progress(n_groups_seen)
                return group, compiled_by_n

            def _execute_group(group, compiled_by_n):
                nonlocal n_invalid, rows_buffer, rows_since_flush, writer_state
                is_diff = registries['gamma'].get(
                    group['gamma_id'], {}
                ).get('metadata', {}).get('differentiable', True)

                if _check_invalid(group, registries['gamma']):
                    for chunk in chunk_group(group, chunk_size):
                        rows = _rows_invalid(chunk, group)
                        n_invalid += len(rows)
                        for r in rows: r['features'].pop('_exc', None)
                        rows_buffer.extend(rows)
                        rows_since_flush += len(rows)
                    return

                for chunk in chunk_group(group, chunk_size):
                    try:
                        if compiled_by_n is not None:
                            s_b, ls, ak, pk = _run_chunk_aot(
                                chunk, group, compiled_by_n, registries, dmd_rank, is_diff
                            )
                        else:
                            s_b, ls, ak, pk, is_diff = _vmap_chunk(chunk, group)
                        rows = _process_chunk_results(chunk, group, s_b, ls, ak, pk)
                    except Exception as e:
                        if verbose:
                            warnings.warn(
                                f"[hub_running AOT] ({group['gamma_id']} × "
                                f"{group['enc_id']}) : {e} — fallback individuel"
                            )
                        rows = _process_chunk_fallback(chunk, group, is_diff)

                    for r in rows: r['features'].pop('_exc', None)
                    rows_buffer.extend(rows)
                    rows_since_flush += len(rows)
                    if rows_since_flush >= flush_every:
                        writer_state = _flush_buffer(
                            rows_buffer, writer_state, phase, output_dir
                        )
                        rows_buffer.clear()
                        rows_since_flush = 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=aot_workers) as pool:

                # Pré-remplir le pipeline
                for group in groups_gen:
                    _submit(group)
                    if len(pending) >= aot_window or ram_binaries_gb >= aot_ram_gb:
                        break

                # Boucle principale
                for next_group in groups_gen:
                    group_exec, compiled_by_n = _consume()

                    # Soumettre suivant immédiatement (CPU ne s'arrête pas)
                    _submit(next_group)

                    _execute_group(group_exec, compiled_by_n)
                    del group_exec, compiled_by_n
                    gc.collect()

                # Vider le pipeline
                while pending:
                    group_exec, compiled_by_n = _consume()
                    _execute_group(group_exec, compiled_by_n)
                    del group_exec, compiled_by_n
                    gc.collect()

        else:
            # ── HOT LOOP JIT BASELINE (inchangée) ───────────────────────────
            for chunk in generate_chunks(run_config, registries, chunk_size):
                xla_key = (
                    chunk['gamma_id'], chunk['enc_id'], chunk['mod_id'],
                    chunk['n_dof'], chunk['rank_eff'], chunk['max_it'],
                )
                if xla_key != last_xla_key:
                    n_groups_seen += 1
                    last_xla_key   = xla_key
                    _progress(n_groups_seen)

                is_diff = registries['gamma'].get(
                    chunk['gamma_id'], {}
                ).get('metadata', {}).get('differentiable', True)

                if _check_invalid(chunk, registries['gamma']):
                    rows = _rows_invalid(chunk, chunk)
                    n_invalid += len(rows)
                else:
                    try:
                        s_b, ls, ak, pk, is_diff = _vmap_chunk(chunk, chunk)
                        rows = _process_chunk_results(chunk, chunk, s_b, ls, ak, pk)
                    except Exception as e:
                        if verbose:
                            warnings.warn(
                                f"[hub_running] vmap failed "
                                f"({chunk['gamma_id']} × {chunk['enc_id']}) : {e}"
                                f" — fallback individuel"
                            )
                        rows = _process_chunk_fallback(chunk, chunk, is_diff)

                for r in rows: r['features'].pop('_exc', None)
                rows_buffer.extend(rows)
                rows_since_flush += len(rows)

                if rows_since_flush >= flush_every:
                    writer_state     = _flush_buffer(rows_buffer, writer_state, phase, output_dir)
                    rows_buffer.clear()
                    rows_since_flush = 0
                elif rows_since_flush >= backup_every:
                    writer_state     = _flush_buffer(rows_buffer, writer_state, phase, output_dir)
                    rows_buffer.clear()
                    rows_since_flush = 0
                    n_done_total     = n_ok + n_explosion + n_invalid + n_fail
                    print(
                        f"\n  [backup] {n_done_total} rows sur disque → "
                        f"{writer_state['path']}",
                        file=sys.stderr
                    )

            # Purger _exc (défense)
            for r in rows:
                r['features'].pop('_exc', None)

            # Un seul extend — toujours ici
            rows_buffer.extend(rows)
            rows_since_flush += len(rows)

            # Flush régulier
            if rows_since_flush >= flush_every:
                writer_state     = _flush_buffer(
                    rows_buffer, writer_state, phase, output_dir
                )
                rows_buffer.clear()
                rows_since_flush = 0

            # Backup périodique — compteur dédié
            elif rows_since_flush >= backup_every:
                writer_state     = _flush_buffer(
                    rows_buffer, writer_state, phase, output_dir
                )
                rows_buffer.clear()
                rows_since_flush = 0
                n_done_total     = n_ok + n_explosion + n_invalid + n_fail
                print(
                    f"\n  [backup] {n_done_total} rows sur disque → "
                    f"{writer_state['path']}",
                    file=sys.stderr
                )

    finally:
        elapsed_total = time.time() - t_start
        print(file=sys.stderr)  # newline après progress
        if rows_buffer:
            writer_state = _flush_buffer(
                rows_buffer, writer_state, phase, output_dir
            )
        if writer_state['writer'] is not None:
            writer_state['writer'].close()

    n_total  = n_ok + n_explosion + n_invalid + n_fail
    rate     = n_total / elapsed_total if elapsed_total > 0 else 0
    elapsed_str = f"{int(elapsed_total//60)}m{int(elapsed_total%60):02d}s"

    print(f"\n✓ Phase '{phase}' terminée  ({n_total} rows — {elapsed_str} — {rate:.0f} rows/s)")
    print(f"  OK        : {n_ok}")
    print(f"  EXPLOSION : {n_explosion}")
    print(f"  INVALID   : {n_invalid}")
    print(f"  FAIL      : {n_fail}")
    print(f"  Groupes   : {n_groups_seen}  (compilations XLA réelles)")
    if writer_state['path']:
        print(f"  Parquet   : {writer_state['path']}")

    return {
        'phase'      : phase,
        'n_ok'       : n_ok,
        'n_explosion': n_explosion,
        'n_invalid'  : n_invalid,
        'n_fail'     : n_fail,
        'n_groups'   : n_groups_seen,
        'parquet'    : writer_state['path'],
    }