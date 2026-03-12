"""
running/hub_running_new.py

Hub running — orchestration batch JAX v7.
Point de jonction : YAML → kernel_groups → vmap → parquet.

Responsabilité : routing pur. Zéro logique métier, zéro calcul.

Flux :
  YAML → dry_run_stats → confirmation → registries
       → generate_kernel_groups (streaming)
       → split_into_batches
       → _check_invalid → make_nan_rows (INVALID)
       → execute_batch → sync_features_b → rows_from_synced
       → write_rows_to_parquet
       → close_parquet_writer (finally)

Statuts run :
  OK        — features finies, comportement nominal
  EXPLOSION — au moins un Inf (P2 charter)
  NAN_ALL   — toutes features non-structurelles NaN, zéro Inf
  INVALID   — incompatibilité rank_constraint, détectée avant vmap
  FAIL      — exception inattendue (batch entier → make_nan_rows)
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

from utils.data_loading_new import (
    discover_gammas_jax,
    discover_encodings_jax,
    discover_modifiers_jax,
    load_yaml,
    open_parquet_writer,
    write_rows_to_parquet,
    close_parquet_writer,
)
from running.plan_new import (
    generate_kernel_groups,
    split_into_batches,
    count_total_samples,
)
from running.batching_new import (
    execute_batch,
    sync_features_b,
    rows_from_synced,
    make_nan_rows,
    rolling_compile_window,
)


# =============================================================================
# DRY RUN ARITHMÉTIQUE (zéro instanciation — repris de hub_running.py)
# =============================================================================

def _normalize_list(val) -> list:
    if val is None:
        return [None]
    return val if isinstance(val, list) else [val]


def _count_axis_entries(axis_list) -> tuple:
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


def dry_run_stats(run_config: Dict, batch_size: int = 256) -> Dict:
    """Métriques du plan depuis le YAML uniquement. < 100ms, zéro instanciation."""
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
    n_groups = n_gi * n_ee * n_mi * n_dof_v * n_max_v

    worst_dof  = max((v for v in n_dofs if v is not None), default=10)
    worst_rank = 2
    for item in (axes.get('encoding', []) or []):
        if isinstance(item, dict):
            r = item.get('params', {}).get('rank', 2)
            worst_rank = max(worst_rank, max(r) if isinstance(r, list) else r)

    worst_elems    = worst_dof ** worst_rank
    state_mb       = worst_elems * 4 / 1024 / 1024
    vmap_ram_mb    = batch_size * worst_elems * 4 / 1024 / 1024
    parquet_est_mb = n_runs * 600 / 1024 / 1024

    return {
        'n_runs'        : n_runs,
        'n_groups_est'  : n_groups,
        'worst_shape'   : f"({'×'.join([str(worst_dof)] * worst_rank)})",
        'worst_state_mb': round(state_mb, 3),
        'vmap_ram_mb'   : round(vmap_ram_mb, 1),
        'parquet_est_mb': round(parquet_est_mb, 1),
    }


# =============================================================================
# INVALID CHECK (statique, O(1), zéro JAX)
# =============================================================================

def _check_invalid(kernel_group: Dict, gamma_registry: Dict) -> bool:
    """Détecte incompatibilité rank_constraint × rank_eff avant vmap."""
    rc = gamma_registry.get(
        kernel_group['gamma_id'], {}
    ).get('metadata', {}).get('rank_constraint')
    rank_eff = kernel_group['rank_eff']
    if rc == 2       and rank_eff >= 3: return True
    if rc == 'square' and rank_eff >= 3: return True
    return False


# =============================================================================
# INTERFACE PUBLIQUE
# =============================================================================

def run_phase(
    yaml_path   : Path,
    output_dir  : Path = Path('data/results'),
    auto_confirm: bool = False,
    batch_size  : int  = 256,
    flush_every : int  = 1000,
    aot_ram_gb  : float = 5.0,
    verbose     : bool = False,
) -> Dict[str, Any]:
    """
    Orchestration batch complète : YAML → parquet.

    Args:
        yaml_path    : Path vers le YAML de run
        output_dir   : Dossier output parquet
        auto_confirm : True → skip confirmation (tests, CI)
        batch_size   : Taille max d'un batch vmap
        flush_every  : Flush parquet tous les N rows
        aot_ram_gb   : Budget RAM fenêtre pré-compilation (défaut 5.0 GB)
                       Configurable via YAML clé 'aot_ram_gb'.
        verbose      : Affiche détails FAIL

    Returns:
        {'phase', 'n_ok', 'n_explosion', 'n_invalid', 'n_fail',
         'n_groups', 'parquet'}
    """
    run_config = load_yaml(yaml_path)
    phase      = run_config.get('phase', 'unknown')
    dmd_rank   = int(run_config.get('dmd_rank', 16))
    aot_ram_gb = float(run_config.get('aot_ram_gb', aot_ram_gb))

    # Dry run
    stats = dry_run_stats(run_config, batch_size)
    print()
    print("=" * 56)
    print(f"  Dry run — phase : {phase}")
    print("=" * 56)
    print(f"  Runs estimés       (lignes parquet)   : {stats['n_runs']:>10,}")
    print(f"  Groupes estimés    (compilations XLA)  : {stats['n_groups_est']:>10,}")
    print(f"  Worst-case state   {stats['worst_shape']:<20} : {stats['worst_state_mb']:>7.3f} MB")
    print(f"  RAM/vmap call      (batch={batch_size})        : {stats['vmap_ram_mb']:>7.1f} MB")
    print(f"  Parquet estimé                         : {stats['parquet_est_mb']:>7.1f} MB")
    print("=" * 56)
    print()

    if not auto_confirm:
        if input("  Continuer ? (o/n) : ").strip().lower() != 'o':
            return {'phase': phase, 'n_ok': 0, 'n_explosion': 0,
                    'n_invalid': 0, 'n_fail': 0, 'n_groups': 0, 'parquet': None}

    registries = {
        'gamma'   : discover_gammas_jax(),
        'encoding': discover_encodings_jax(),
        'modifier': discover_modifiers_jax(),
    }

    # Compteurs
    n_ok = n_explosion = n_invalid = n_fail = n_groups = 0
    rows_buffer      = []
    rows_since_flush = 0
    writer           = None
    parquet_path     = None
    n_runs_est       = stats['n_runs']
    t_start          = time.time()

    def _progress():
        n_done  = n_ok + n_explosion + n_invalid + n_fail
        pct     = n_done / n_runs_est * 100 if n_runs_est > 0 else 0
        elapsed = time.time() - t_start
        eta_str = ''
        if pct > 1:
            eta_s   = elapsed / pct * (100 - pct)
            eta_str = f"  ETA {int(eta_s//60)}m{int(eta_s%60):02d}s"
        print(
            f"\r  [{n_groups:>5}g | {pct:5.1f}% | "
            f"{int(elapsed//60)}m{int(elapsed%60):02d}s{eta_str}]  "
            f"OK={n_ok}  EXP={n_explosion}  INV={n_invalid}  FAIL={n_fail}",
            end='', file=sys.stderr,
        )

    def _flush(rows):
        nonlocal writer, parquet_path
        if not rows:
            return
        if writer is None:
            writer       = open_parquet_writer(phase, output_dir, rows[0])
            parquet_path = Path(output_dir) / f'{phase}.parquet'
        write_rows_to_parquet(writer, rows, phase)

    def _sync_and_count(pending_batch, pending_features):
        """Sync GPU→CPU + compte statuts. Appelé pendant que le GPU compute le suivant."""
        nonlocal n_ok, n_explosion, n_fail
        try:
            synced = sync_features_b(pending_features)
            rows   = rows_from_synced(pending_batch, synced, phase)
            for r in rows:
                if   r['run_status'] == 'EXPLOSION': n_explosion += 1
                elif r['run_status'] == 'NAN_ALL':   n_fail      += 1
                else:                                n_ok        += 1
            return rows
        except Exception as exc:
            if verbose:
                warnings.warn(f"[hub_running_new] FAIL sync : {exc}")
            rows    = make_nan_rows(pending_batch, 'FAIL', phase)
            n_fail += len(rows)
            return rows

    def _accumulate(rows):
        nonlocal rows_since_flush
        rows_buffer.extend(rows)
        rows_since_flush += len(rows)
        if rows_since_flush >= flush_every:
            _flush(rows_buffer)
            rows_buffer.clear()
            rows_since_flush = 0

    try:
        pending = None   # {'batch': ..., 'features': jnp async}

        n_workers  = max(1, (os.cpu_count() or 4) - 2)
        kg_stream  = rolling_compile_window(
            generate_kernel_groups(run_config, registries),
            dmd_rank, batch_size, n_workers, aot_ram_gb,
        )

        for kg in kg_stream:
            n_groups += 1
            _progress()

            for batch in split_into_batches(kg, batch_size):

                if _check_invalid(kg, registries['gamma']):
                    # Vider le pending avant d'écrire les INVALID
                    if pending is not None:
                        _accumulate(_sync_and_count(pending['batch'], pending['features']))
                        pending = None
                    _accumulate(make_nan_rows(batch, 'INVALID', phase))
                    n_invalid += len(batch['samples'])

                else:
                    try:
                        # Dispatch GPU — retour immédiat (async)
                        features_async = execute_batch(batch, dmd_rank)

                        # Pendant que le GPU compute le batch courant,
                        # sync + traite le batch précédent sur CPU
                        if pending is not None:
                            _accumulate(_sync_and_count(pending['batch'], pending['features']))

                        pending = {'batch': batch, 'features': features_async}

                    except Exception as exc:
                        if verbose:
                            warnings.warn(
                                f"[hub_running_new] FAIL dispatch "
                                f"({kg['gamma_id']} × {kg['enc_id']}) : {exc}"
                            )
                        if pending is not None:
                            _accumulate(_sync_and_count(pending['batch'], pending['features']))
                            pending = None
                        _accumulate(make_nan_rows(batch, 'FAIL', phase))
                        n_fail += len(batch['samples'])

            del kg

        # Vider le dernier pending
        if pending is not None:
            _accumulate(_sync_and_count(pending['batch'], pending['features']))
            pending = None

    finally:
        print(file=sys.stderr)
        _flush(rows_buffer)
        if writer is not None:
            close_parquet_writer(writer)

    elapsed     = time.time() - t_start
    n_total     = n_ok + n_explosion + n_invalid + n_fail
    rate        = n_total / elapsed if elapsed > 0 else 0
    elapsed_str = f"{int(elapsed//60)}m{int(elapsed%60):02d}s"

    print(f"\n✓ Phase '{phase}' terminée  ({n_total} rows — {elapsed_str} — {rate:.0f} rows/s)")
    print(f"  OK        : {n_ok}")
    print(f"  EXPLOSION : {n_explosion}")
    print(f"  INVALID   : {n_invalid}")
    print(f"  FAIL      : {n_fail}")
    print(f"  Groupes   : {n_groups}  (compilations XLA réelles)")
    if parquet_path:
        print(f"  Parquet   : {parquet_path}")

    return {
        'phase'      : phase,
        'n_ok'       : n_ok,
        'n_explosion': n_explosion,
        'n_invalid'  : n_invalid,
        'n_fail'     : n_fail,
        'n_groups'   : n_groups,
        'parquet'    : parquet_path,
    }