"""
tests/bench_aot.py

Comparaison stratégies compilation JAX — bench autonome.

Stratégies :
    S0 — JIT baseline   : comportement actuel, compilation à la demande
    S1 — AOT séquentiel : pré-compilation de tous les groupes upfront
    S2 — AOT rolling    : pré-compilation W groupes à l'avance + purge immédiate

Mesures par stratégie :
    - wall clock total
    - temps compilation cumulé
    - temps compute cumulé
    - RAM peak binaires (psutil)
    - taille binaire par groupe
    - GPU utilization % (poll nvidia-smi toutes les 200ms)

Usage :
    cd ~/Bureau/prc_framework
    cp <ce fichier> tests/bench_aot.py
    cp <aot_bench.yaml> configs/phases/aot_bench.yaml

    python -m tests.bench_aot --config configs/phases/aot_bench.yaml
    python -m tests.bench_aot --config configs/phases/aot_bench.yaml --strategies 0 2
    python -m tests.bench_aot --config configs/phases/aot_bench.yaml --strategies 2 --window 4
"""

import argparse
import gc
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from functools import partial
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from compositions.compositions_jax import generate_groups, chunk_group, count_stats
from running.run_one_jax import _run_jit
from utils.data_loading_jax import (
    discover_gammas_jax,
    discover_encodings_jax,
    discover_modifiers_jax,
    load_yaml,
)


# =============================================================================
# STRUCTURES
# =============================================================================

@dataclass
class GroupMeasure:
    group_key : str
    gamma_id  : str
    enc_id    : str
    n_dof     : int
    rank_eff  : int
    max_it    : int
    n_runs    : int
    compile_s : float = 0.0
    compute_s : float = 0.0
    binary_mb : float = 0.0
    stall_s   : float = 0.0


@dataclass
class StrategyResult:
    name          : str
    wall_s        : float = 0.0
    compile_s     : float = 0.0
    compute_s     : float = 0.0
    peak_ram_mb   : float = 0.0
    binary_avg_mb : float = 0.0
    binary_max_mb : float = 0.0
    gpu_mean_pct  : float = 0.0
    gpu_max_pct   : float = 0.0
    groups        : List[GroupMeasure] = field(default_factory=list)
    n_groups      : int = 0
    n_runs        : int = 0


# =============================================================================
# GPU MONITOR
# =============================================================================

class GPUMonitor:
    def __init__(self, interval_s: float = 0.2):
        self.interval_s = interval_s
        self.samples: List[float] = []
        self._stop   = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> Tuple[float, float]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if not self.samples:
            return 0.0, 0.0
        return float(np.mean(self.samples)), float(np.max(self.samples))

    def _run(self):
        while not self._stop.wait(self.interval_s):
            try:
                out = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=utilization.gpu',
                     '--format=csv,noheader,nounits'],
                    stderr=subprocess.DEVNULL, timeout=0.5
                )
                self.samples.append(float(out.decode().strip().split('\n')[0]))
            except Exception:
                pass


# =============================================================================
# HELPERS
# =============================================================================

def _ram_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def _group_key(g: Dict) -> str:
    return (f"{g['gamma_id']}|{g['enc_id']}|{g['mod_id']}"
            f"|n{g['n_dof']}|r{g['rank_eff']}|it{g['max_it']}")


def _is_diff(g: Dict, registries: Dict) -> bool:
    return registries['gamma'].get(
        g['gamma_id'], {}
    ).get('metadata', {}).get('differentiable', True)


def _build_batch(g: Dict, chunk: Dict, registries: Dict) -> Tuple:
    """Construit les arrays vmappables pour un chunk."""
    runs     = chunk['runs']
    n_dof    = g['n_dof']
    rank_eff = g['rank_eff']
    shape    = (n_dof,) * rank_eff

    # D_initial via enc_fn
    D_list = []
    for r in runs:
        try:
            D = g['enc_fn'](n_dof, r['enc_params'], r['key_CI'])
        except Exception:
            D = jnp.ones(shape, dtype=jnp.float32) * 0.01
        D_list.append(D.astype(jnp.float32))
    D_b = jnp.stack(D_list)

    keys_b = jnp.stack([r['key_run'] for r in runs])

    def _stack_params(plist):
        if not plist or not plist[0]:
            return {}, {}
        out, axes = {}, {}
        for k in plist[0]:
            vals = [p[k] for p in plist]
            try:
                out[k] = jnp.stack([
                    v if hasattr(v, 'shape') else jnp.array(v, dtype=jnp.float32)
                    for v in vals
                ])
                axes[k] = 0
            except Exception:
                out[k]  = plist[0][k]
                axes[k] = None
        return out, axes

    gp_b, in_g = _stack_params([r['gamma_params'] for r in runs])
    mp_b, in_m = _stack_params([r['mod_params']   for r in runs])

    return gp_b, mp_b, D_b, keys_b, in_g, in_m


def _exec_chunk(g, chunk, registries, chunk_size, dmd_rank, jit_fn,
                aot: bool = False) -> float:
    """
    Exécute un chunk, retourne temps compute.

    aot=False : jit_fn est un jax.jit standard — appele avec les 9 args.
    aot=True  : jit_fn est un binaire AOT vmapped — appele avec (gp_b, mp_b, D_b, keys_b).
                Le vmap et les statiques sont baked in a la compilation.
    """
    gp_b, mp_b, D_b, keys_b, in_g, in_m = _build_batch(g, chunk, registries)
    max_it  = g['max_it']
    is_diff = _is_diff(g, registries)

    t0 = time.perf_counter()

    if aot:
        # jit_fn est un dict {n_batch -> compiled_fn}.
        # Selectionner le binaire compile pour la taille de ce chunk.
        actual_n = D_b.shape[0]
        fn_for_n = jit_fn.get(actual_n)
        if fn_for_n is None:
            raise RuntimeError(f"Pas de binaire AOT pour batch_size={actual_n}. "
                               f"Disponibles: {list(jit_fn.keys())}")
        result = fn_for_n(gp_b, mp_b, D_b, keys_b)
    else:
        run_vmap = jax.vmap(
            jit_fn,
            in_axes=(None, in_g, None, in_m, 0, 0, None, None, None)
        )
        result = run_vmap(
            g['gamma_fn'], gp_b,
            g['mod_fn'],   mp_b,
            D_b, keys_b,
            max_it, dmd_rank, is_diff,
        )

    jax.block_until_ready(result)
    return time.perf_counter() - t0


def _run_group_with(g, registries, chunk_size, dmd_rank, jit_fn,
                    aot: bool = False) -> float:
    """Exécute tous les chunks d'un groupe, retourne temps compute total."""
    t = 0.0
    for chunk in chunk_group(g, chunk_size):
        t += _exec_chunk(g, chunk, registries, chunk_size, dmd_rank, jit_fn, aot=aot)
    return t


def _compile_group(g, registries, dmd_rank, chunk_size) -> Tuple[object, float, float]:
    """
    AOT-compile la fonction vmappée pour ce groupe avec batch_size=chunk_size fixe.

    La batch size est baked in dans le binaire XLA — le lowering doit utiliser
    la meme shape que les vrais appels. On compile avec chunk_size, et on paddera
    les derniers chunks plus petits dans _exec_chunk.

    Retourne (compiled_fn, compile_s, binary_mb).
    compiled_fn : appel direct avec (gp_b, mp_b, D_b, keys_b) de taille chunk_size.
    """
    n_dof    = g['n_dof']
    rank_eff = g['rank_eff']
    shape    = (n_dof,) * rank_eff
    max_it   = g['max_it']
    is_diff  = _is_diff(g, registries)

    # Fonction avec statiques fermes — ne prend que les 4 args dynamiques
    def _run_dynamic(gp, mp, D, key):
        return _run_jit(g['gamma_fn'], gp, g['mod_fn'], mp, D, key,
                        max_it, dmd_rank, is_diff)

    # Compiler avec la vraie batch size du groupe.
    # Chaque groupe a un nombre de runs fixe connu upfront — c'est la shape a baker in.
    # Pour les groupes > chunk_size, on compile un binaire par taille de chunk
    # (chunk_size pour les chunks pleins, reste pour le dernier chunk si different).
    # Dans notre bench tous les groupes ont <= chunk_size runs donc un seul binaire suffit.
    n_runs_group = len(g['runs'])
    # Chunks : tous chunk_size sauf eventuellement le dernier
    chunk_sizes_needed = set()
    for c in chunk_group(g, chunk_size):
        chunk_sizes_needed.add(len(c['runs']))

    # Compiler un binaire par taille de chunk distincte
    compiled_by_n = {}
    ram_pre = _ram_mb()
    t0 = time.perf_counter()
    compile_ok = True
    for n_batch in sorted(chunk_sizes_needed):
        runs_ex = (g['runs'] * n_batch)[:n_batch]
        gp_ex, mp_ex, D_b_ex, keys_b_ex, in_g_ex, in_m_ex = _build_batch(
            g, {'runs': runs_ex}, registries
        )
        try:
            vmapped  = jax.vmap(_run_dynamic, in_axes=(in_g_ex, in_m_ex, 0, 0))
            lowered  = jax.jit(vmapped).lower(gp_ex, mp_ex, D_b_ex, keys_b_ex)
            compiled_by_n[n_batch] = lowered.compile()
        except Exception as e:
            print(f"  [compile FAIL n={n_batch}] {_group_key(g)[:45]} — {e}", flush=True)
            compile_ok = False
            break
    compile_s = time.perf_counter() - t0
    compiled = compiled_by_n if compile_ok and compiled_by_n else None

    binary_mb = max(0.0, _ram_mb() - ram_pre)
    return compiled, compile_s, binary_mb


def _verify_coherence(g, registries, chunk_size, dmd_rank) -> bool:
    """
    Verifie que S0 (JIT) et AOT produisent les memes resultats numeriques.
    Teste sur le premier chunk du groupe.
    Retourne True si coherent, False sinon.
    """
    chunk = chunk_group(g, chunk_size)[0]

    # S0 — JIT reference
    jit_fn = jax.jit(_run_jit, static_argnums=(0, 2, 6, 7, 8))
    gp_b, mp_b, D_b, keys_b, in_g, in_m = _build_batch(g, chunk, registries)
    run_vmap = jax.vmap(jit_fn, in_axes=(None, in_g, None, in_m, 0, 0, None, None, None))
    ref = run_vmap(
        g['gamma_fn'], gp_b, g['mod_fn'], mp_b,
        D_b, keys_b, g['max_it'], dmd_rank, _is_diff(g, registries)
    )
    jax.block_until_ready(ref)

    # AOT — compile + call
    compiled, _, _ = _compile_group(g, registries, dmd_rank, chunk_size)
    if compiled is None:
        print(f"  [COHERENCE] SKIP {_group_key(g)[:40]} — compile failed")
        return False

    actual_n = D_b.shape[0]
    actual_n = D_b.shape[0]
    fn_for_n = compiled.get(actual_n)
    if fn_for_n is None:
        print(f"  [COHERENCE] SKIP — pas de binaire pour n={actual_n}")
        return False
    aot_out = fn_for_n(gp_b, mp_b, D_b, keys_b)
    jax.block_until_ready(aot_out)

    # Comparaison leaf par leaf
    ok = True
    ref_leaves  = jax.tree_util.tree_leaves(ref)
    aot_leaves  = jax.tree_util.tree_leaves(aot_out)
    if len(ref_leaves) != len(aot_leaves):
        print(f"  [COHERENCE] FAIL {_group_key(g)[:40]} — nb outputs differ: {len(ref_leaves)} vs {len(aot_leaves)}")
        return False

    for i, (r_leaf, a_leaf) in enumerate(zip(ref_leaves, aot_leaves)):
        try:
            max_diff = float(jnp.max(jnp.abs(r_leaf - a_leaf)))
            mean_ref = float(jnp.mean(jnp.abs(r_leaf)))
            rel = max_diff / (mean_ref + 1e-8)
            if rel > 1e-3:
                print(f"  [COHERENCE] FAIL leaf[{i}] {_group_key(g)[:35]} — max_diff={max_diff:.2e}  rel={rel:.2e}")
                ok = False
        except Exception as e:
            print(f"  [COHERENCE] WARN leaf[{i}] — {e}")

    if ok:
        print(f"  [COHERENCE] OK   {_group_key(g)[:50]}")
    del compiled
    gc.collect()
    return ok


# =============================================================================
# S0 — JIT BASELINE
# =============================================================================

def run_s0(groups, registries, chunk_size, dmd_rank, monitor) -> StrategyResult:
    result = StrategyResult(name='S0_JIT_baseline')
    monitor.start()
    ram_start = _ram_mb()
    ram_peak  = ram_start
    t_wall    = time.perf_counter()

    for g in groups:
        gkey   = _group_key(g)
        n_runs = len(g['runs'])
        # Nouveau jit_fn par groupe — simule le comportement actuel
        jit_fn = jax.jit(_run_jit, static_argnums=(0, 2, 6, 7, 8))

        t0 = time.perf_counter()
        t_compute = _run_group_with(g, registries, chunk_size, dmd_rank, jit_fn)
        wall_g = time.perf_counter() - t0
        ram_peak = max(ram_peak, _ram_mb())

        result.groups.append(GroupMeasure(
            group_key=gkey, gamma_id=g['gamma_id'], enc_id=g['enc_id'],
            n_dof=g['n_dof'], rank_eff=g['rank_eff'], max_it=g['max_it'],
            n_runs=n_runs, compute_s=t_compute,
        ))
        result.compute_s += t_compute
        result.n_runs    += n_runs
        print(f"  [S0] {gkey[:52]:52s}  wall={wall_g:.2f}s", flush=True)

    gm, gx       = monitor.stop()
    result.wall_s      = time.perf_counter() - t_wall
    result.peak_ram_mb = ram_peak - ram_start
    result.gpu_mean_pct, result.gpu_max_pct = gm, gx
    result.n_groups    = len(groups)
    return result


# =============================================================================
# S1 — AOT COMPLET UPFRONT
# =============================================================================

def run_s1(groups, registries, chunk_size, dmd_rank, monitor) -> StrategyResult:
    result    = StrategyResult(name='S1_AOT_full_upfront')
    ram_start = _ram_mb()
    t_wall    = time.perf_counter()
    binary_mbs = []

    print(f"  [S1] Compilation {len(groups)} groupes...", flush=True)
    compiled_map = {}
    for g in groups:
        gkey = _group_key(g)
        compiled, cs, bm = _compile_group(g, registries, dmd_rank, chunk_size)
        compiled_map[gkey] = compiled
        binary_mbs.append(bm)
        result.compile_s += cs
        print(f"  [S1] compiled {gkey[:46]:46s}  {cs:.2f}s  +{bm:.1f}MB", flush=True)

    ram_compiled = _ram_mb()
    print(f"  [S1] RAM binaires : {ram_compiled-ram_start:.0f}MB — compute...\n", flush=True)

    monitor.start()
    for i, g in enumerate(groups):
        gkey   = _group_key(g)
        n_runs = len(g['runs'])
        _compiled = compiled_map.get(gkey)
        fn         = _compiled if _compiled is not None else jax.jit(_run_jit, static_argnums=(0, 2, 6, 7, 8))
        _is_aot    = _compiled is not None

        t_compute = _run_group_with(g, registries, chunk_size, dmd_rank, fn, aot=_is_aot)
        result.groups.append(GroupMeasure(
            group_key=gkey, gamma_id=g['gamma_id'], enc_id=g['enc_id'],
            n_dof=g['n_dof'], rank_eff=g['rank_eff'], max_it=g['max_it'],
            n_runs=n_runs, compile_s=0.0, compute_s=t_compute, binary_mb=binary_mbs[i],
        ))
        result.compute_s += t_compute
        result.n_runs    += n_runs
        print(f"  [S1] {gkey[:52]:52s}  compute={t_compute:.2f}s", flush=True)

    gm, gx = monitor.stop()

    del compiled_map
    gc.collect()
    ram_purge = _ram_mb()
    print(f"  [S1] Purge : libéré {ram_compiled-ram_purge:.0f}MB", flush=True)

    result.wall_s        = time.perf_counter() - t_wall
    result.peak_ram_mb   = ram_compiled - ram_start
    result.binary_avg_mb = float(np.mean(binary_mbs)) if binary_mbs else 0.0
    result.binary_max_mb = float(np.max(binary_mbs)) if binary_mbs else 0.0
    result.gpu_mean_pct, result.gpu_max_pct = gm, gx
    result.n_groups      = len(groups)
    return result


# =============================================================================
# S2 — AOT ROLLING WINDOW
# =============================================================================

def run_s2(groups, registries, chunk_size, dmd_rank, monitor, window=2) -> StrategyResult:
    import concurrent.futures

    result     = StrategyResult(name=f'S2_AOT_rolling_W{window}')
    ram_start  = _ram_mb()
    ram_peak   = ram_start
    n          = len(groups)
    binary_mbs = []

    monitor.start()
    t_wall = time.perf_counter()

    def _worker(g):
        return _compile_group(g, registries, dmd_rank, chunk_size)

    with concurrent.futures.ThreadPoolExecutor(max_workers=window) as pool:
        futures = {j: pool.submit(_worker, groups[j]) for j in range(min(window, n))}

        for i, g in enumerate(groups):
            gkey   = _group_key(g)
            n_runs = len(g['runs'])

            if i + window < n:
                futures[i + window] = pool.submit(_worker, groups[i + window])

            t0_wait = time.perf_counter()
            compiled, cs, bm = futures.pop(i).result()
            stall_s = time.perf_counter() - t0_wait

            result.compile_s += cs
            binary_mbs.append(bm)
            ram_peak = max(ram_peak, _ram_mb())

            _is_aot = compiled is not None
            fn = compiled if _is_aot else jax.jit(_run_jit, static_argnums=(0, 2, 6, 7, 8))
            t_compute = _run_group_with(g, registries, chunk_size, dmd_rank, fn, aot=_is_aot)

            del compiled, fn
            gc.collect()

            flag = ' ⚠STALL' if stall_s > 0.05 else ''
            print(
                f"  [S2] {gkey[:40]:40s}  "
                f"compile={cs:.2f}s  stall={stall_s:.2f}s{flag}  "
                f"compute={t_compute:.2f}s  {bm:.1f}MB",
                flush=True
            )

            result.groups.append(GroupMeasure(
                group_key=gkey, gamma_id=g['gamma_id'], enc_id=g['enc_id'],
                n_dof=g['n_dof'], rank_eff=g['rank_eff'], max_it=g['max_it'],
                n_runs=n_runs, compile_s=cs, compute_s=t_compute,
                binary_mb=bm, stall_s=stall_s,
            ))
            result.compute_s += t_compute
            result.n_runs    += n_runs

    gm, gx = monitor.stop()
    result.wall_s        = time.perf_counter() - t_wall
    result.peak_ram_mb   = ram_peak - ram_start
    result.binary_avg_mb = float(np.mean(binary_mbs)) if binary_mbs else 0.0
    result.binary_max_mb = float(np.max(binary_mbs)) if binary_mbs else 0.0
    result.gpu_mean_pct, result.gpu_max_pct = gm, gx
    result.n_groups      = len(groups)
    return result


# =============================================================================
# RAPPORT
# =============================================================================

def _pct(n, d):
    return f'{100*n/d:.0f}%' if d > 0 else 'N/A'


def write_report(results, path):
    L = ['='*72, 'BENCH AOT — RAPPORT COMPARATIF', '='*72, '']

    hdr = (f"{'Stratégie':<28} {'Wall':>6} {'Compile':>12} {'Compute':>12} "
           f"{'GPU moy':>8} {'GPU max':>8} {'RAM bin':>9}")
    L += [hdr, '-'*len(hdr)]
    for r in results:
        L.append(
            f"{r.name:<28} {r.wall_s:>5.1f}s "
            f"{r.compile_s:>7.1f}s({_pct(r.compile_s,r.wall_s):>4}) "
            f"{r.compute_s:>7.1f}s({_pct(r.compute_s,r.wall_s):>4}) "
            f"{r.gpu_mean_pct:>7.1f}% {r.gpu_max_pct:>7.1f}% "
            f"{r.peak_ram_mb:>7.0f}MB"
        )

    L += ['', 'Binaires compilés :']
    for r in results:
        if r.binary_avg_mb > 0:
            L.append(f"  {r.name} : avg={r.binary_avg_mb:.1f}MB  max={r.binary_max_mb:.1f}MB  "
                     f"extrap 5k groupes={r.binary_avg_mb*5000/1024:.1f}GB")

    s0 = next((r for r in results if 'S0' in r.name), None)
    if s0:
        L += ['', 'Gains wall vs S0 :']
        for r in results:
            if r is s0: continue
            d = s0.wall_s - r.wall_s
            L.append(f"  {r.name} : {d:+.1f}s ({100*d/s0.wall_s:+.1f}%)")

    for r in results:
        if 'S2' not in r.name: continue
        stalls = [g for g in r.groups if g.stall_s > 0.05]
        L += ['', f'Stalls {r.name} : {len(stalls)}/{len(r.groups)} groupes']
        if stalls:
            L.append('  → Augmenter --window')

    for r in results:
        L += ['', f'── Détail : {r.name}',
              f"  {'Groupe':<48} {'n':>5} {'compile':>9} {'compute':>9} {'stall':>7} {'bin':>6}"]
        for g in r.groups:
            L.append(
                f"  {g.group_key[:48]:<48} {g.n_runs:>5} "
                f"{g.compile_s:>8.2f}s {g.compute_s:>8.2f}s "
                f"{g.stall_s:>6.2f}s {g.binary_mb:>4.1f}MB"
            )

    L += [
        '', '='*72, 'INTERPRÉTATION', '='*72, '',
        'GPU% :',
        '  < 30%  → PCIe bottleneck — gains AOT limités (transfert domine)',
        '  30-70% → compile/idle dominant — AOT rolling aide significativement',
        '  > 70%  → compute dominant — bon régime',
        '', 'Stalls S2 :',
        '  stall > 0.05s → GPU a attendu CPU → augmenter --window',
        '', 'RAM binaires :',
        '  extrap > 8GB → S1 non viable sur bourrin, S2 obligatoire',
    ]

    txt = '\n'.join(L)
    path.write_text(txt, encoding='utf-8')
    print('\n' + txt)
    print(f'\n→ Rapport : {path}')


def write_csv(results, path):
    import csv
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['strategy','group_key','gamma_id','enc_id','n_dof',
                    'rank_eff','max_it','n_runs','compile_s','compute_s','stall_s','binary_mb'])
        for r in results:
            for g in r.groups:
                w.writerow([r.name, g.group_key, g.gamma_id, g.enc_id, g.n_dof,
                            g.rank_eff, g.max_it, g.n_runs,
                            f'{g.compile_s:.4f}', f'{g.compute_s:.4f}',
                            f'{g.stall_s:.4f}', f'{g.binary_mb:.2f}'])
    print(f'→ CSV : {path}')


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',     required=True)
    p.add_argument('--strategies', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--window',     type=int, default=2)
    p.add_argument('--chunk-size', type=int, default=64)
    p.add_argument('--dmd-rank',   type=int, default=16)
    p.add_argument('--out',        default='reports/bench_aot')
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n=== BENCH AOT ===')
    print(f'Config     : {args.config}')
    print(f'Stratégies : {args.strategies}  Window : {args.window}')
    print(f'JAX device : {jax.default_backend()} — {jax.devices()}\n')

    registries = {
        'gamma'   : discover_gammas_jax(),
        'encoding': discover_encodings_jax(),
        'modifier': discover_modifiers_jax(),
    }
    print(f"Atomics : {len(registries['gamma'])}g  "
          f"{len(registries['encoding'])}e  {len(registries['modifier'])}m")

    cfg    = load_yaml(Path(args.config))
    groups = generate_groups(cfg, registries, args.chunk_size)
    stats  = count_stats(groups, args.chunk_size)
    print(f"Groupes : {stats['n_groups']}  runs : {stats['n_runs']}  chunks : {stats['n_chunks']}")
    for g in groups:
        print(f"  {_group_key(g)}  ({len(g['runs'])} runs)")
    print()

    # Warm-up
    print('Warm-up...', flush=True)
    jax.block_until_ready(jax.jit(lambda x: x + 1)(jnp.array(1.0)))
    print('OK\n', flush=True)

    results = []
    ts = time.strftime('%Y%m%d_%H%M%S')
    cs, dr = args.chunk_size, args.dmd_rank

    # Vérification cohérence S0 vs AOT sur un sous-ensemble de groupes
    print('─'*60)
    print('VÉRIFICATION COHÉRENCE JIT vs AOT (3 premiers groupes)')
    print('─'*60)
    for g in groups[:3]:
        _verify_coherence(g, registries, cs, dr)
    print()

    if 0 in args.strategies:
        print('─'*60, '\nS0 — JIT baseline\n' + '─'*60)
        r = run_s0(groups, registries, cs, dr, GPUMonitor())
        results.append(r)
        print(f'\nS0 done : wall={r.wall_s:.1f}s  GPU={r.gpu_mean_pct:.0f}%\n')

    if 1 in args.strategies:
        print('─'*60, '\nS1 — AOT upfront\n' + '─'*60)
        r = run_s1(groups, registries, cs, dr, GPUMonitor())
        results.append(r)
        print(f'\nS1 done : wall={r.wall_s:.1f}s  compile={r.compile_s:.1f}s  GPU={r.gpu_mean_pct:.0f}%\n')

    if 2 in args.strategies:
        print('─'*60, f'\nS2 — AOT rolling W={args.window}\n' + '─'*60)
        r = run_s2(groups, registries, cs, dr, GPUMonitor(), window=args.window)
        results.append(r)
        print(f'\nS2 done : wall={r.wall_s:.1f}s  compile={r.compile_s:.1f}s  GPU={r.gpu_mean_pct:.0f}%\n')

    write_report(results, out_dir / f'bench_aot_{ts}.txt')
    write_csv(results,    out_dir / f'bench_aot_{ts}.csv')


if __name__ == '__main__':
    main()