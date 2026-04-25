#!/usr/bin/env python3
"""
Orchestrateur principal — budget VRAM + sémaphore compute GPU.

Architecture de concurrence (v17) :

    Le hub lance des processus librement (jusqu'à MAX_CONCURRENT_GROUPS).
    Chaque processus prépare son travail CPU (discovery, matérialisation,
    compilation) PENDANT que d'autres utilisent le GPU.

    Deux mécanismes de contrôle indépendants :

    1. Budget VRAM (VramBudget) — contrôle la MÉMOIRE GPU.
       Plusieurs processus peuvent réserver leur VRAM simultanément
       tant que le budget total n'est pas dépassé.

    2. Sémaphore compute (GPU_COMPUTE_SLOTS) — contrôle l'EXÉCUTION GPU.
       Limite le nombre de processus qui exécutent P1+P2 en même temps.
       Élimine la contention GPU qui causait 3-12× d'overhead P2.

    Le GPU n'attend jamais : quand un processus libère le sémaphore
    compute, le suivant (déjà préparé, VRAM réservée) prend le relais.

    Jobs trop gros : pré-découpés au lancement (split_job).
    Timeout acquire : filet de sécurité, le processus meurt proprement.

@ROLE    Orchestrateur pipeline — YAML → processus → parquet
@LAYER   running

@LIFECYCLE
  CREATES  VramBudget       partagé entre hub et tous les SubBatchProcess
  CREATES  gpu_semaphore    multiprocessing.Semaphore(GPU_COMPUTE_SLOTS)
  CREATES  ParquetWriter    ouvert avant le premier run
  DELETES  ParquetWriter    close() dans finally — toujours fermé

@CONFORMITY
  OK    Pipeline GPU — la préparation CPU chevauche l'exécution GPU
  OK    Budget VRAM dynamique — concurrence adaptée à la taille des jobs
  OK    Sémaphore compute — élimine la contention GPU (v17)
  OK    Pré-split — aucun deadlock possible sur les jobs trop gros
  OK    Parquet écrit et fermé avant toute analyse (P6)
"""

import multiprocessing
import logging
import signal
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter
import heapq
import psutil

from configs.pipeline_constants import (
    MAX_CONCURRENT_GROUPS,
    FLUSH_BATCH_SIZE,
    MIN_AVAILABLE_RAM_GB,
    VRAM_USAGE_THRESHOLD,
    DEFAULT_TOTAL_VRAM_GB,
    XLA_FIXED_OVERHEAD_GB,
    XLA_SAFETY_MARGIN,
    VRAM_ACQUIRE_TIMEOUT_S,
    RANK3_EXTRA_MARGIN,
    VERBOSE,
    USE_BFLOAT16,
    EXPECTED_MASK_DENSITY,
    JVP_INTERMEDIATES_FACTOR,
    GPU_COMPUTE_SLOTS,
)
from running.plan import build_jobs, dry_run_stats
from running.subbatch_process import SubBatchProcess
from utils.io_v8 import (
    build_schema_v15,
    load_yaml,
    open_parquet_writer,
    write_col_data_to_parquet,
    close_parquet_writer,
    discover_gammas_metadata,
    discover_encodings_metadata,
    discover_modifiers_metadata,
)
from running.features_registry import (
    METADATA_COLUMNS,
    FEATURE_NAMES,
    TIMELINE_COLUMNS_A,
    TIMELINE_COLUMNS_B,
    MASK_INDICES_COLUMN,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("hub")


# =========================================================================
# ESTIMATION VRAM — modèle analytique
# =========================================================================

def estimate_gpu_peak(rank, n_dof, max_it, B, is_diff=False):
    """
    Estimation du pic VRAM d'un processus pendant la phase GPU (Go).

    Modèle SC8 (architecture chantier 1) :
        P1 scan : 3 copies état + 4 timelines screening (bf16 ou f32)
        P2 scan : 3 copies état + 11 tables couche A + 15 tables couche B
                  + carry auxiliaires (prev_delta, count_active)
        classify : mask + buffers intermédiaires (entre P1 et P2)
        JVP (si is_diff) : intermédiaires tangent/cotangent/résultat,
                           amortis par EXPECTED_MASK_DENSITY

    P1 et P2 ne sont pas simultanés → on prend le max.
    Le coût fixe (compilation XLA + runtime) est ajouté séparément.

    Args:
        rank    : rang effectif du tenseur (2 ou 3)
        n_dof   : dimension
        max_it  : nombre d'itérations
        B       : taille du batch
        is_diff : True si le gamma est différentiable (JVP couche B)

    Returns:
        float — estimation du pic VRAM en Go.
    """
    state_bytes = (n_dof ** rank) * 2  # un tenseur D en float32

    # Bytes par élément de timeline P1 (bf16 si USE_BFLOAT16, sinon f32)
    tl_p1_bpe = 2 if USE_BFLOAT16 else 4

    # P1 : propagation + screening (SC2 — 4 timelines, 0 accumulateur)
    p1 = B * (3 * state_bytes                 # state, prev, state_next
              + 4 * max_it * tl_p1_bpe)       # 4 timelines screening

    # P2 : re-propagation + couche A (11) + couche B (15) — SC5
    # Tous les buffers P2 sont float32 (SD-STRUCT-1 SC5)
    p2_bytes_per_element = 2 if USE_BFLOAT16 else 4

    p2 = B * (3 * state_bytes                 # state, prev_state, state_next
              + 11 * max_it * p2_bytes_per_element                # timeline_tables_A (11 obs × T)
              + 15 * max_it * p2_bytes_per_element                # feature_tables_B (15 feat × T)
              + 4                              # prev_delta_b (B,) float32
              + 4)                             # count_active (B,) int32

    # Classify — entre P1 et P2, persist en mémoire
    classify = B * max_it * 2 * 2              # mask + buffers classification

    # JVP intermédiaires couche B (si différentiable)
    # Le JVP alloue ~JVP_INTERMEDIATES_FACTOR copies d'état par sample masqué.
    # Au pic, tous les B samples peuvent être masqués au même step.
    # Amorti par EXPECTED_MASK_DENSITY car le lax.cond ne paie que si masqué.
    if is_diff:
        jvp_overhead = (B * state_bytes
                        * JVP_INTERMEDIATES_FACTOR
                        * EXPECTED_MASK_DENSITY)
    else:
        jvp_overhead = 0

    # P1 et P2 séquentiels — pic = max des deux + classify + JVP overhead
    scan_peak = max(p1, p2) + classify + jvp_overhead

    # Coût fixe par processus (compilation XLA + runtime)
    fixed = XLA_FIXED_OVERHEAD_GB * (1024 ** 3)

    # Marge intermédiaires XLA
    total = (scan_peak + fixed) * XLA_SAFETY_MARGIN

    if rank >= 3:
        total *= RANK3_EXTRA_MARGIN

    return total / (1024 ** 3)


# =========================================================================
# BUDGET VRAM PARTAGÉ
# =========================================================================

class VramBudget:
    """
    Budget VRAM partagé entre le hub et les SubBatchProcess.

    Remplace le sémaphore GPU. Permet la concurrence dynamique :
    plusieurs petits jobs en parallèle, un seul gros job à la fois.

    Thread-safe et process-safe via multiprocessing.Value + Lock.
    """

    def __init__(self, total_gb, threshold):
        self._budget = total_gb * threshold
        self._used = multiprocessing.Value('d', 0.0, lock=True)

    def acquire(self, amount_gb, timeout=120.0):
        """
        Réserve amount_gb dans le budget. Bloque jusqu'à disponibilité.

        Retourne True si acquis, False si timeout (deadlock safety).
        """
        deadline = time.time() + timeout
        while True:
            with self._used.get_lock():
                if self._used.value + amount_gb <= self._budget:
                    self._used.value += amount_gb
                    return True
            if time.time() > deadline:
                return False
            time.sleep(0.1)

    def release(self, amount_gb):
        """Libère amount_gb du budget."""
        with self._used.get_lock():
            self._used.value = max(0.0, self._used.value - amount_gb)

    @property
    def available(self):
        return self._budget - self._used.value

    @property
    def used(self):
        return self._used.value

    @property
    def total(self):
        return self._budget


# =========================================================================
# SPLIT DYNAMIQUE
# =========================================================================

def split_job(job, budget_gb):
    """
    Pré-découpe un job trop gros pour le budget VRAM.

    Divise le sub_batch par 2 jusqu'à ce que l'estimation rentre.
    Si B=1 ne rentre toujours pas, on lance quand même (warning).

    Retourne une liste de jobs (1 si pas de split nécessaire).
    """
    gm = job['group_meta']
    sub = job['sub_batch']
    B = len(sub)
    is_diff = gm.get('is_differentiable', False)

    est = estimate_gpu_peak(gm['rank_eff'], gm['n_dof'], gm['max_it'], B, is_diff)
    if est <= budget_gb:
        return [job]

    # Trouver le B max qui rentre
    new_B = B
    while new_B > 1:
        new_B = max(1, new_B // 2)
        est = estimate_gpu_peak(gm['rank_eff'], gm['n_dof'], gm['max_it'], new_B, is_diff)
        if est <= budget_gb:
            break

    if new_B >= B:
        return [job]

    est_final = estimate_gpu_peak(gm['rank_eff'], gm['n_dof'], gm['max_it'], new_B, is_diff)
    if est_final > budget_gb:
        logger.warning(
            f"Job {gm['gamma_id']} rank={gm['rank_eff']} n_dof={gm['n_dof']} "
            f"B=1 estimé {est_final:.2f} Go > budget {budget_gb:.1f} Go — "
            f"tentative quand même (risque OOM)")

    chunks = []
    for i in range(0, len(sub), new_B):
        chunks.append({'group_meta': gm, 'sub_batch': sub[i:i + new_B]})

    logger.info(
        f"Split : {gm['gamma_id']} rank={gm['rank_eff']} n_dof={gm['n_dof']} "
        f"B={B} → {len(chunks)}×{new_B} (estimé {est_final:.2f} Go/chunk)")

    return chunks


# =========================================================================
# File d'attente FIFO
# =========================================================================

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0

    def push(self, job):
        heapq.heappush(self._heap, (self._counter, job))
        self._counter += 1

    def __len__(self):
        return len(self._heap)

    def pop(self):
        if not self._heap:
            return None
        _, job = heapq.heappop(self._heap)
        return job


# =========================================================================
# Orchestration principale
# =========================================================================

def run_phase(yaml_path: Path, output_dir: Path, auto_confirm=False) -> None:
    logger.info(f"Lecture du plan : {yaml_path}")
    config = load_yaml(yaml_path)

    registries = {
        'gamma': discover_gammas_metadata(),
        'encoding': discover_encodings_metadata(),
        'modifier': discover_modifiers_metadata(),
    }

    jobs = build_jobs(config, registries)
    if not jobs:
        logger.warning("Aucun job généré.")
        return

    stats = dry_run_stats(jobs)
    phase_name = jobs[0]['group_meta']['phase']
    budget_gb = DEFAULT_TOTAL_VRAM_GB * VRAM_USAGE_THRESHOLD

    # ── Pré-split des jobs trop gros ──
    split_jobs_list = []
    n_splits = 0
    for job in jobs:
        chunks = split_job(job, budget_gb)
        if len(chunks) > 1:
            n_splits += 1
        split_jobs_list.extend(chunks)

    print(f"\n{'=' * 56}")
    print(f"  Dry run — phase : {phase_name}")
    print(f"{'=' * 56}")
    print(f"  Runs valides     : {stats['n_samples']:>10,}")
    print(f"  Groupes          : {stats['n_groups']:>10,}")
    print(f"  Jobs (post-split): {len(split_jobs_list):>10,}")
    if n_splits:
        print(f"  Jobs re-découpés : {n_splits:>10}")
    print(f"  max_it           : {stats['max_it']:>10,}")
    print(f"  Budget VRAM      :    {budget_gb:.1f} Go")
    print(f"{'=' * 56}")

    if not auto_confirm:
        response = input("  Lancer l'exécution ? (o/n) : ").strip().lower()
        if response != 'o':
            print("  Annulé.")
            return

    schema = build_schema_v15(
        METADATA_COLUMNS,
        FEATURE_NAMES,
        TIMELINE_COLUMNS_A + TIMELINE_COLUMNS_B,
        timeline_int_columns=[MASK_INDICES_COLUMN],
    )
    writer = open_parquet_writer(phase_name, output_dir, schema)
    #logger.info(f"Fichier Parquet ouvert : {output_dir / f'{phase_name}.parquet'}")

    # Budget VRAM partagé — contrôle combien de processus peuvent préparer
    vram_budget = VramBudget(DEFAULT_TOTAL_VRAM_GB, VRAM_USAGE_THRESHOLD)
    # Sémaphore compute — contrôle combien de processus exécutent sur le GPU
    # Séparé du budget VRAM : la préparation (matérialisation, compilation)
    # se fait hors sémaphore, seuls P1+P2+transfert le requièrent.
    gpu_semaphore = multiprocessing.Semaphore(GPU_COMPUTE_SLOTS)
    col_queue = multiprocessing.Queue(maxsize=5000)

    pending = JobQueue()
    for job in split_jobs_list:
        pending.push(job)

    active_processes: List[Tuple[multiprocessing.Process, int]] = []
    col_data_buffer = []
    buffered_rows = 0
    total_rows_written = 0
    status_counter = Counter()
    runs_processed = 0
    total_runs = stats['n_samples']
    groups_processed = 0
    total_groups = len(split_jobs_list)

    start_time = time.time()
    last_progress_log = start_time
    last_flush_time = start_time
    progress_interval = 1.0
    flush_time_interval = 60.0

    def get_available_ram_gb():
        return psutil.virtual_memory().available / (1024 ** 3)

    try:
        while pending or active_processes:
            # 1. Nettoyer les processus terminés
            still_active = []
            for p, sz in active_processes:
                if p.is_alive():
                    still_active.append((p, sz))
                else:
                    p.join()
                    groups_processed += 1
            active_processes = still_active

            # 2. Lancer de nouveaux jobs — le GPU ne doit jamais attendre
            #    Pas de check VRAM ici — le processus gère via budget.acquire()
            #    On lance tant qu'on a de la RAM et des slots de processus
            while len(active_processes) < MAX_CONCURRENT_GROUPS and pending:
                if get_available_ram_gb() < MIN_AVAILABLE_RAM_GB:
                    break

                job = pending.pop()
                B = len(job['sub_batch'])
                gm = job['group_meta']
                is_diff = gm.get('is_differentiable', False)
                est = estimate_gpu_peak(
                    gm['rank_eff'], gm['n_dof'], gm['max_it'], B, is_diff)

                if not vram_budget.acquire(est, timeout=0.1):
                    # Pas assez de VRAM : on remet le job en tête de file
                    pending.push(job)  # ou utiliser une file différente
                    break   # on arrête de lancer de nouveaux jobs pour l'instant

                p = SubBatchProcess(
                    group_meta=gm,
                    sub_batch=job['sub_batch'],
                    vram_budget=vram_budget,
                    gpu_semaphore=gpu_semaphore,
                    est_vram_gb=est,
                    col_queue=col_queue,
                )
                p.start()
                active_processes.append((p, B))
                if VERBOSE:
                    print(
                        f"Job lancé (PID={p.pid}) : {B} runs, "
                        f"{gm['gamma_id']} r{gm['rank_eff']} d{gm['n_dof']} "
                        f"(est {est:.2f} Go, budget dispo {vram_budget.available:.1f} Go)")

            # 3. Récupérer les résultats
            try:
                while True:
                    col_data = col_queue.get_nowait()
                    run_status = col_data.get('run_status')
                    n_rows_in_batch = (len(run_status)
                                       if run_status is not None else 0)
                    if run_status is not None:
                        status_counter.update(run_status)
                        runs_processed += n_rows_in_batch
                    col_data_buffer.append(col_data)
                    buffered_rows += n_rows_in_batch
            except multiprocessing.queues.Empty:
                pass

            # 3b. Flush
            now = time.time()
            time_flush = (now - last_flush_time) >= flush_time_interval
            rows_flush = buffered_rows >= FLUSH_BATCH_SIZE
            if col_data_buffer and (rows_flush or time_flush):
                for cd in col_data_buffer:
                    total_rows_written += write_col_data_to_parquet(writer, cd)
                n_flushed = buffered_rows
                col_data_buffer.clear()
                buffered_rows = 0
                last_flush_time = now
                if VERBOSE:
                    print(
                        f"Flush : {n_flushed} rows → parquet "
                        f"(total {total_rows_written})")

            # 4. Progression
            if now - last_progress_log >= progress_interval:
                elapsed = now - start_time
                ok_cnt = (status_counter.get('OK', 0)
                          + status_counter.get('OK_TRUNCATED', 0))
                exp_cnt = status_counter.get('EXPLOSION', 0)
                col_cnt = status_counter.get('COLLAPSED', 0)
                pct = ((runs_processed / total_runs) * 100
                       if total_runs else 0)
                vram_used = vram_budget.used
                msg = (
                    f"{runs_processed}/{total_runs} ({pct:.1f}%) | "
                    f"G {groups_processed}/{total_groups} | "
                    f"VRAM {vram_used:.1f}/{budget_gb:.1f}Go | "
                    f"{elapsed:.0f}s | {runs_processed/elapsed:.0f} run/s | "
                    f"OK={ok_cnt} EXP={exp_cnt} COL={col_cnt}"
                )
                logger.info(msg)#, end='\r', flush=True)
                last_progress_log = now

            # 5. Pause
            time.sleep(0.02)

    finally:
        print()
        logger.info("Flush final du buffer parquet...")
        try:
            for cd in col_data_buffer:
                total_rows_written += write_col_data_to_parquet(writer, cd)
            col_data_buffer.clear()
            buffered_rows = 0
        except Exception as e:
            logger.error(f"Erreur pendant le flush final : {e}")
        try:
            close_parquet_writer(writer)
            logger.info(f"Parquet fermé. Total rows écrits : {total_rows_written}")
        except Exception as e:
            logger.error(f"Erreur fermeture parquet : {e}")

    elapsed_total = time.time() - start_time
    print(f"\n{'=' * 56}")
    print(f"  Résumé final — phase {phase_name}")
    print(f"{'=' * 56}")
    print(f"  Runs traités : {total_rows_written}")
    print(f"  Groupes traités : {groups_processed}")
    print(f"  Temps total : {elapsed_total:.1f} secondes")
    print(f"  Répartition des statuts :")
    for status, count in sorted(status_counter.items()):
        print(f"    {status:15} : {count:>8,}")
    print(f"{'=' * 56}")
    logger.info(f"Phase terminée. Total rows : {total_rows_written}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: hub.py <config.yaml> <output_dir>")
        sys.exit(1)

    yaml_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    def signal_handler(sig, frame):
        logger.info("Interruption utilisateur, arrêt propre...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    run_phase(yaml_path, output_dir)
