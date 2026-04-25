"""
Plan d'exécution v16 – génération de jobs prêts à exécuter.

Zéro JAX. Produit une liste de jobs contenant :
- group_meta (dict)
- sub_batch (list de SampleDesc, déjà résolus avec IDs et params)
- découpage en sous-batches uniquement si nécessaire
"""

import hashlib
import random as _random
import time
from itertools import product
from typing import List, Dict, Any, Generator

from configs.pipeline_constants import (
    BATCH_SIZE_MAP, DEFAULT_P1_BATCH, DEFAULT_P2_BATCH,
    #SEMAPHORE_MAP, DEFAULT_SEMAPHORE,
    MAX_N_DOF_RANK3,
)


def _resolve_seed(seed):
    if seed is None:
        return int(time.time_ns() % (2**63)) ^ _random.randint(0, 2**62)
    return int(seed)


def _make_run_seed(seed_run, gamma_id, encoding_id):
    # Hachage déterministe cross-invocations (remplace hash() Python, non-déterministe).
    # Seuls gamma_id et encoding_id participent au mixing — les autres axes (modifier,
    # seed_CI, n_dof) sont des dimensions indépendantes du produit cartésien.
    id_bytes = (gamma_id + ':' + encoding_id).encode('utf-8')
    id_hash = int.from_bytes(hashlib.md5(id_bytes).digest()[:4], 'little')
    mixed = ((seed_run & 0xFFFFFFFF) * 2654435761 + id_hash) & 0xFFFFFFFF
    return ((seed_run >> 32) << 32) | mixed


def _get_rank_eff(enc_entry):
    metadata = enc_entry.get('metadata', {})
    params = enc_entry.get('params', {})
    if 'rank' in metadata and metadata['rank'] is not None:
        return int(metadata['rank'])
    return int(params.get('rank', 3))


def _is_compatible(gamma_metadata, rank_eff):
    rc = gamma_metadata.get('rank_constraint')
    if rc is None:
        return True
    if rc == 2 and rank_eff == 2:
        return True
    if rc == 'square' and rank_eff == 2:
        return True
    return False


def _resolve_axis_atomic(axis_config, registry):
    """Résout un axe YAML en liste de dicts {id, params, metadata} (pas de callable)."""
    if axis_config == 'all':
        ids_config = [{'id': k} for k in registry.keys()]
    elif isinstance(axis_config, dict) and 'random' in axis_config:
        n = axis_config['random']
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
            raise ValueError(f"ID '{entity_id}' non trouvé")
        entry = registry[entity_id]
        inline = item.get('params', {})
        list_params = {k: v for k, v in inline.items() if isinstance(v, list)}
        scalar_params = {k: v for k, v in inline.items() if not isinstance(v, list)}
        if not list_params:
            resolved.append({
                'id': entity_id,
                'params': scalar_params,
                'metadata': entry['metadata'],
            })
        else:
            for combo in product(*[[(k, v) for v in vals] for k, vals in list_params.items()]):
                params = {**scalar_params, **dict(combo)}
                resolved.append({
                    'id': entity_id,
                    'params': params,
                    'metadata': entry['metadata'],
                })
    return resolved


def _resolve_gamma_axis(gamma_config, gamma_registry):
    """Résout l'axe gamma (supporte séquences). Retourne liste de dicts légers."""
    if isinstance(gamma_config, dict) and 'sequence' in gamma_config:
        lines = gamma_config['sequence']
        weights_cfg = gamma_config.get('weights')
        return _resolve_gamma_sequence(lines, gamma_registry, weights_cfg)
    entries = _resolve_axis_atomic(gamma_config, gamma_registry)
    return [{
        'id': e['id'],
        'is_composed': False,
        'component_ids': None,
        'weights': None,
        'params': e['params'],
        'metadata': e['metadata'],
    } for e in entries]


def _resolve_gamma_sequence(lines_config, registry, weights_cfg=None):
    lines_resolved = [_resolve_axis_atomic(line_cfg, registry) for line_cfg in lines_config]
    sequences = []
    for combo in product(*lines_resolved):
        component_ids = [e['id'] for e in combo]
        merged_params = {}
        for e in combo:
            merged_params.update(e['params'])
        all_diff = all(e['metadata'].get('differentiable', True) for e in combo)
        if len(combo) == 1:
            e = combo[0]
            sequences.append({
                'id': e['id'],
                'is_composed': False,
                'component_ids': None,
                'weights': None,
                'params': e['params'],
                'metadata': e['metadata'],
            })
        else:
            composed_id = '-'.join(component_ids)
            sequences.append({
                'id': composed_id,
                'is_composed': True,
                'component_ids': component_ids,
                'weights': None,
                'params': merged_params,
                'metadata': {
                    'id': composed_id,
                    'differentiable': all_diff,
                    'rank_constraint': None,
                    'non_markovian': False,
                    'stochastic': False,
                },
            })
    if not weights_cfg:
        return sequences
    # Pondération
    rng_w = weights_cfg['range']
    step = weights_cfg['step']
    vals = [rng_w[0] + i * step for i in range(int((rng_w[1] - rng_w[0]) / step) + 1)]
    weighted = []
    for seq in sequences:
        cids = seq['component_ids'] or [seq['id']]
        for w_combo in product(vals, repeat=len(cids)):
            w_list = list(w_combo)
            wid = '-'.join(f"w_{w:.2f}-{gid}" for w, gid in zip(w_list, cids))
            weighted.append({
                'id': wid,
                'is_composed': True,
                'component_ids': cids,
                'weights': w_list,
                'params': seq['params'],
                'metadata': {**seq['metadata'], 'id': wid},
            })
    return weighted


def calibrate_group(rank_eff, n_dof):
    """Retourne la taille de batch P1 pour un groupe (entier)."""
    p1_batch, _ = BATCH_SIZE_MAP.get((rank_eff, n_dof), (DEFAULT_P1_BATCH, DEFAULT_P2_BATCH))
    return p1_batch


def build_jobs(run_config, registries, batch_size_override=None):
    """
    Construit la liste des jobs à exécuter.
    """
    # Validation max_it
    max_it_cfg = run_config.get('max_it', 200)
    if isinstance(max_it_cfg, list):
        if len(max_it_cfg) != 1:
            raise ValueError(f"Un seul max_it autorisé, reçu {len(max_it_cfg)}")
        max_it = int(max_it_cfg[0])
    else:
        max_it = int(max_it_cfg)

    # Résolution des axes
    gammas = _resolve_gamma_axis(run_config.get('gamma', 'all'), registries['gamma'])
    encodings = _resolve_axis_atomic(run_config.get('encoding', 'all'), registries['encoding'])
    modifiers = _resolve_axis_atomic(run_config.get('modifier', [{'id': 'M0'}]), registries['modifier'])

    n_dofs = run_config.get('n_dof', [10])
    if not isinstance(n_dofs, list):
        n_dofs = [n_dofs]
    n_dofs = [int(n) for n in n_dofs]

    seed_CIs_raw = run_config.get('seed_CI', [None])
    if not isinstance(seed_CIs_raw, list):
        seed_CIs_raw = [seed_CIs_raw]
    seed_CIs = [_resolve_seed(s) for s in seed_CIs_raw]

    seed_runs_raw = run_config.get('seed_run', [None])
    if not isinstance(seed_runs_raw, list):
        seed_runs_raw = [seed_runs_raw]
    seed_runs = [_resolve_seed(s) for s in seed_runs_raw]

    phase = run_config.get('phase', 'unknown')

    # Collecte des samples par groupe (gamma_id, rank_eff, n_dof)
    group_samples = {}
    for gamma_desc in gammas:
        gamma_id = gamma_desc['id']
        is_diff = gamma_desc['metadata'].get('differentiable', True)
        for enc_desc in encodings:
            rank_eff = _get_rank_eff(enc_desc)
            if not _is_compatible(gamma_desc['metadata'], rank_eff):
                continue
            for n_dof in n_dofs:
                if rank_eff >= 3 and n_dof >= MAX_N_DOF_RANK3:
                    continue
                key = (gamma_id, rank_eff, n_dof)
                if key not in group_samples:
                    group_samples[key] = {
                        'samples': [],
                        'is_diff': is_diff,
                        'phase': phase,
                        'max_it': max_it,
                    }
                # Générer tous les samples pour cette combinaison
                for mod_desc in modifiers:
                    for seed_CI in seed_CIs:
                        for seed_run_base in seed_runs:
                            run_seed = _make_run_seed(seed_run_base, gamma_id, enc_desc['id'])
                            sample = {
                                'gamma_id': gamma_id,
                                'encoding_id': enc_desc['id'],
                                'modifier_id': mod_desc['id'],
                                'gamma_component_ids': gamma_desc.get('component_ids'),
                                'gamma_weights': gamma_desc.get('weights'),
                                'gamma_params': gamma_desc['params'],
                                'encoding_params': enc_desc['params'],
                                'modifier_params': mod_desc['params'],
                                'seed_CI': seed_CI,
                                'seed_run': run_seed,
                                'rank_eff': rank_eff,
                                'n_dof': n_dof,
                                'max_it': max_it,
                                'is_differentiable': is_diff,
                                'phase': phase,
                            }
                            group_samples[key]['samples'].append(sample)

    # Construction des jobs avec découpage
    jobs = []
    for (gamma_id, rank_eff, n_dof), info in group_samples.items():
        samples = info['samples']
        group_meta = {
            'gamma_id': gamma_id,
            'rank_eff': rank_eff,
            'n_dof': n_dof,
            'max_it': max_it,
            'is_differentiable': info['is_diff'],
            'phase': info['phase'],
        }
        batch_size = batch_size_override or calibrate_group(rank_eff, n_dof)
        if len(samples) <= batch_size:
            jobs.append({'group_meta': group_meta, 'sub_batch': samples})
        else:
            for i in range(0, len(samples), batch_size):
                jobs.append({'group_meta': group_meta, 'sub_batch': samples[i:i+batch_size]})
    return jobs


def dry_run_stats(jobs):
    n_samples = sum(len(job['sub_batch']) for job in jobs)
    n_groups = len({(job['group_meta']['gamma_id'], job['group_meta']['rank_eff'], job['group_meta']['n_dof']) for job in jobs})
    max_it = jobs[0]['group_meta']['max_it'] if jobs else 0
    return {'n_samples': n_samples, 'n_groups': n_groups, 'max_it': max_it}
