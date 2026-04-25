"""
SubBatchProcess — orchestrateur de calcul pour un groupe de runs.

Un sub-batch = un groupe homogène (même gamma, même rank_eff, même n_dof).
Ce module est un processus isolé (charter P9): tout ce qui est alloué ici
est purgé à la terminaison du processus.

Flux d'exécution (10 étapes):
    1. Discovery JAX (import atomics)
    2. Extraction paramètres du groupe
    3. Matérialisation (encoding + modifier → tenseurs JAX)
    4. Compilation des briques (propagate, screening, layer_a, layer_b)
    5. P1 scan (lax.scan: propagation + screening)
    6. Classification + masque (JAX GPU, zéro transfert CPU)
    7. P2 scan (lax.scan: re-propagation + couche A + couche B conditionnelle)
    8. Transfert unique GPU → CPU + libération GPU
    9. Post-traitement numpy (agrégations, DMD, F6, temporelles, phasiques)
    10. Assemblage col_data → queue

Architecture double passe (charter §1.5):
    P1 calcule les 4 signaux screening à pleine résolution temporelle.
    Le masque adaptatif identifie les instants de transition significative.
    P2 re-propage la trajectoire identique, calcule les 11 observables
    couche A O(n²) à chaque step, et les 15 features couche B (SVD + JVP)
    uniquement aux instants masqués via lax.map + lax.cond (décisions D1-D5).

@ROLE    Processus isolé de calcul — YAML→features→parquet
@LAYER   running

@EXPORTS
  SubBatchProcess(group_meta, sub_batch, vram_budget, gpu_semaphore, est_vram_gb, col_queue)
  materialize_batch(sample_descs, group_meta, gamma_reg, encoding_reg, modifier_reg)

@LIFECYCLE
  CREATES  D_b, gp_b, keys_b (JAX tensors) — libérés à la mort du processus
  CREATES  timeline_tables_A, feature_tables_B (JAX → numpy) — libérés après assemblage
  PASSES   col_data (dict numpy) → col_queue → hub.py

@CONFORMITY
  OK    Processus isolé, purge mémoire garantie (P9)
  OK    Kernel aveugle au domaine — gamma_fn opaque (P3)
  OK    Persistance avant analyse — col_data envoyé au hub qui écrit le parquet (P6)
"""

import multiprocessing
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from typing import Dict, List, Any, Tuple

from running.jit_builders import (
    split_keys_batch,
    build_propagate_vmap,
    build_screening_vmap,
    build_layer_a_vmap,
    build_layer_b_fn,
)
from running.classify_jax import (
    classify_and_mask,
    STATUS_NAMES,
    REGIME_NAMES,
)
from running.postprocess import (
    transfer_to_cpu,
    aggregate_by_agg_map,
    compute_dmd_on_observables,
    compute_f6_on_observables,
    compute_stationarity,
    compute_entropy_production,
    compute_autocorrelations,
    compute_pnn_features,
    compute_temporal_features,
    compute_phasic_features,
    build_col_data,
)
from running.features_registry import (
    LAYER_A_KEYS,
    LAYER_B_KEYS,
    TURBULENCE_CONSTANT,
    EPS,
)
from utils.io_v8 import (
    discover_gammas_jax,
    discover_encodings_jax,
    discover_modifiers_jax,
)
from configs.pipeline_constants import VERBOSE, VRAM_ACQUIRE_TIMEOUT_S, USE_BFLOAT16


# =========================================================================
# BLOC 3 — MATÉRIALISATION
#
# Résout les encodings, modifiers et gamma_params pour chaque sample.
# Empile en tenseurs JAX prêts pour le vmap.
# =========================================================================

def materialize_batch(sample_descs, group_meta, gamma_reg, encoding_reg, modifier_reg):
    """
    Matérialise un sub-batch en tenseurs JAX.

    Args:
        sample_descs:  liste de dicts décrivant chaque sample
        group_meta:    dict du groupe (gamma_id, rank_eff, n_dof, ...)
        gamma_reg:     registre JAX des gammas
        encoding_reg:  registre JAX des encodings
        modifier_reg:  registre JAX des modifiers

    Returns:
        D_b:          (B, *state_shape) — états initiaux empilés
        gp_b:         {param_name: (B, ...)} — gamma params empilés
        keys_b:       (B, 2) — clés PRNG
        gamma_fn:     callable — fonction gamma brute (pour les closures)
        in_axes_gp:   {param_name: 0} — axes vmap pour gp
    """
    B = len(sample_descs)
    n_dof = group_meta['n_dof']
    dtype = jnp.bfloat16 if USE_BFLOAT16 else jnp.float32

    gamma_id = sample_descs[0]['gamma_id']
    gamma_entry = gamma_reg[gamma_id]
    gamma_fn = gamma_entry['callable']
    gamma_prepare = gamma_entry.get('prepare_params')

    D_list = []
    keys_list = []
    gp_list = []

    for desc in sample_descs:
        key_CI = jax.random.PRNGKey(desc['seed_CI'])

        # Encoding
        enc_entry = encoding_reg[desc['encoding_id']]
        enc_params = dict(desc['encoding_params'])
        if enc_entry.get('prepare_params') is not None:
            enc_params = enc_entry['prepare_params'](enc_params, n_dof, key_CI)
        D = enc_entry['callable'](n_dof, enc_params, key_CI)

        # Modifier
        mod_entry = modifier_reg[desc['modifier_id']]
        mod_params = dict(desc['modifier_params'])
        if mod_entry.get('prepare_params') is not None:
            mod_params = mod_entry['prepare_params'](mod_params, n_dof, key_CI)
        D = mod_entry['callable'](D, mod_params, key_CI)

        # Gamma params
        gamma_params = dict(desc['gamma_params'])
        if gamma_prepare is not None:
            gamma_params = gamma_prepare(gamma_params, n_dof, key_CI)

        D_list.append(D)
        keys_list.append(jax.random.PRNGKey(desc['seed_run']))
        gp_list.append(gamma_params)

    D_b = jnp.stack([jnp.asarray(d) for d in D_list])
    keys_b = jnp.stack(keys_list)

    if gp_list and gp_list[0]:
        gp_b = {k: jnp.stack([jnp.asarray(gp[k]) for gp in gp_list])
                for k in gp_list[0]}
        in_axes_gp = {k: 0 for k in gp_b}
    else:
        gp_b = {}
        in_axes_gp = {}

    return D_b, gp_b, keys_b, gamma_fn, in_axes_gp


# =========================================================================
# BLOC 5 — P1 SCAN (minimal, SC2)
#
# Propagation + screening uniquement, sur max_it steps. Tout en JAX via lax.scan.
# Aucun transfert CPU. Aucune différentiation. Aucun accumulateur dynamique.
#
# Rôle scientifique : produire à grain temporel uniforme les 4 signaux O(n²)
# nécessaires à classify_and_mask pour identifier les virages et les statuts
# pathologiques. P1 ne calcule aucune feature scientifique finale (charter §1.6
# distinction mesures dynamiques / mesures spectrales — couches A et B en P2).
#
# Produit:
# - 4 timelines (B, max_it) alignées sur PASS1_KEYS du registre :
#     delta_D, frob, is_finite, cos_dissim
# - last_states (B, *state_shape) — état final pour la détection COLLAPSED
#
# Convention de masquage (M13, conservée) : pour les runs déjà inactifs
# (explosion détectée à un pas antérieur), on écrit 0.0 dans delta_D/frob/cos
# et on conserve la valeur antérieure dans is_finite. Cette convention garantit
# l'invariance comportementale de classify_and_mask par rapport à la version
# pré-SC2 (charter D4 — masque inchangé).
#
# @CONFORMITY
#   OK   P1 minimal — propagate + screening uniquement (charter D2)
#   OK   Aucun JVP en P1 (charter D3, JVP migré en couche B SC4)
#   OK   Kernel aveugle au domaine — gamma_fn opaque via propagate_fn (P3/K4)
# =========================================================================

def run_pass1(D_b, gp_b, keys_b,
              propagate_fn, screening_fn,
              max_it, B):
    """
    Passe 1 minimale : propagation + screening sur max_it steps.

    propagate_fn et screening_fn sont des fonctions vmappées prêtes à composer
    dans le lax.scan. Elles ne sont pas jit-ées individuellement — le scan
    compile l'ensemble en un programme XLA unique.

    Args:
        D_b:           (B, *state_shape) — états initiaux empilés
        gp_b:          {param: (B, ...)} — gamma params empilés
        keys_b:        (B, 2) — clés PRNG par run
        propagate_fn:  callable vmappé (gp, state, prev, subk) → state_next
        screening_fn:  callable vmappé (state, prev) → {4 scalaires}
        max_it:        int — nombre d'itérations (static)
        B:             int — taille du batch (static)

    Returns:
        dict avec 5 clés (toutes encore sur GPU) :
          'delta_D'    : (B, max_it) timeline_dtype
          'frob'       : (B, max_it) timeline_dtype
          'is_finite'  : (B, max_it) timeline_dtype  (valeurs 0/1)
          'cos_dissim' : (B, max_it) timeline_dtype
          'last_states': (B, *state_shape) state_dtype
    """
    state = D_b
    prev = D_b
    keys = keys_b
    active = jnp.ones(B, dtype=jnp.bool_)

    timeline_dtype = jnp.bfloat16 if USE_BFLOAT16 else jnp.float32

    # Initialisation alignée sur PASS1_KEYS du registre :
    # delta_D, frob, cos_dissim → 0.0 ; is_finite → 1.0 (valeur neutre).
    delta_tl = jnp.zeros((B, max_it), dtype=timeline_dtype)
    frob_tl  = jnp.zeros((B, max_it), dtype=timeline_dtype)
    fin_tl   = jnp.ones((B, max_it),  dtype=timeline_dtype)
    cos_tl   = jnp.zeros((B, max_it), dtype=timeline_dtype)

    def step(carry, t):
        (state, prev, keys, active,
         delta_tl, frob_tl, fin_tl, cos_tl) = carry

        # 1. Scission PRNG
        keys, subk = split_keys_batch(keys)

        # 2. Propagation
        state_next = propagate_fn(gp_b, state, prev, subk)

        # 3. Screening (4 scalaires, O(n²))
        scr = screening_fn(state, prev)

        # 4. Écriture des 4 timelines avec masquage par `active` (M13)
        delta_tl = delta_tl.at[:, t].set(jnp.where(active, scr['delta_D'],    0.0))
        frob_tl  = frob_tl .at[:, t].set(jnp.where(active, scr['frob'],       0.0))
        fin_tl   = fin_tl  .at[:, t].set(jnp.where(active, scr['is_finite'],  fin_tl[:, t]))
        cos_tl   = cos_tl  .at[:, t].set(jnp.where(active, scr['cos_dissim'], 0.0))

        # 5. Absorption explosion (active reste False une fois passée à False)
        active = active & (scr['is_finite'] > 0.5)

        # 6. Avancement d'état
        prev, state = state, state_next

        return (state, prev, keys, active,
                delta_tl, frob_tl, fin_tl, cos_tl), None

    init_carry = (state, prev, keys, active,
                  delta_tl, frob_tl, fin_tl, cos_tl)
    final_carry, _ = lax.scan(step, init_carry, jnp.arange(max_it))

    (state_final, _, _, _,
     delta_tl, frob_tl, fin_tl, cos_tl) = final_carry

    return {
        'delta_D':     delta_tl,
        'frob':        frob_tl,
        'is_finite':   fin_tl,
        'cos_dissim':  cos_tl,
        'last_states': state_final,
    }


# =========================================================================
# BLOC 6 — P2 SCAN (refonte SC5)
#
# Re-propagation identique à P1 + couche A continue + couche B aux virages.
#
# Architecture deux couches (charter D1) :
#   Couche A — 11 observables O(n²) calculés à CHAQUE step, inconditionnellement
#              (grammaire signal, timelines uniformes consommées par DMD/F6/autocorr).
#   Couche B — 15 features (SVD + JVP) calculées UNIQUEMENT aux instants masqués
#              via lax.map + lax.cond par sample (grammaire virages).
#
# Trajectoire identique à P1 (SD18 — charter §2.3 vases communicants) :
#   Mêmes D_b, gp_b, keys_b, même propagate_fn.
#   Le carry `keys` évolue exactement comme en P1 (un seul split_keys_batch
#   produit `subk` pour propagate). La key additionnelle pour la couche B
#   (Hutchinson tangent F4, SC4-D7) est dérivée de `subk` — opération sans
#   effet sur le carry, donc la séquence des states P2 est bit-à-bit
#   identique à P1.
#
# Scatter conditionnel couche A (SD-SC5-3) :
#   Pour t >= t_effectives[b], les valeurs couche A sont NaN en buffer
#   (D13 — couche A préservée jusqu'à t_effective, au-delà corrompue par
#   état pollué — on écrit NaN directement plutôt que tronquer en post-process).
#
# Scatter couche B :
#   Inconditionnel (le lax.cond interne à layer_b_fn produit déjà NaN si
#   mask_t[b] == False). Buffer NaN-éparse par construction.
#
# Ordre du step (charter correction SC5) :
#   mesure AVANT propagate — couche A/B observent state au temps t,
#   prev_state au temps t-1, puis propagate vers t+1. Corrige un décalage
#   d'indice silencieux de l'ancien P2 (SVD observait state post-propagate).
#
# Init au step 0 (SD-SC5-2) :
#   prev_state   = NaN_like(D)   → delta_D/cos_dissim/bregman = NaN au pas 0
#   prev_delta_b = NaN(B,)       → lyap_empirical = NaN aux pas 0 et 1
#   Conséquence pour SC6 : agrégats mean/std sur couche A doivent ignorer
#   les NaN runtime (déjà cadré par charter §1.8 — NaN ≠ erreur).
#
# Dtype : float32 strict pour les buffers A et B (SD-STRUCT-1), indépendant
# de USE_BFLOAT16 qui reste appliqué aux timelines screening P1.
#
# OPT-SCATTER (post-SC7) :
#   Les 26 buffers dict individuels (11 A + 15 B) sont remplacés par
#   2 arrays empilés (B, max_it, 11) et (B, max_it, 15). Le scatter
#   par step passe de 26 opérations individuelles à 2. Le carry du scan
#   a 2 feuilles au lieu de 26, réduisant la taille du programme HLO
#   et permettant à XLA de mieux optimiser. Gain mesuré 4-7× sur le
#   scatter seul (exp_scatter_fusion). Après le scan, les arrays sont
#   dépilés en dicts nommés pour préserver l'interface vers transfer_to_cpu.
#
# @CONFORMITY
#   OK   Trajectoire P1=P2 déterministe (même séquence de carry keys)
#   OK   Couche A inconditionnelle, couche B conditionnelle (charter D1)
#   OK   Kernel aveugle — propagate_fn/layer_a_fn/layer_b_fn opaques (P3/K4)
#   OK   Clés de sortie = LAYER_A_KEYS / LAYER_B_KEYS exact (P13)
# =========================================================================

def run_pass2(D_b, gp_b, keys_b, mask, t_effectives,
              propagate_fn, layer_a_fn, layer_b_fn,
              max_it, B):
    """
    Passe 2 : re-propagation + couche A continue + couche B aux virages.

    Args:
        D_b:          (B, *state_shape) — états initiaux
        gp_b:         {param: (B, ...)} — gamma params empilés,
                      toutes feuilles leading B (materialize_batch L163-169)
        keys_b:       (B, 2) uint32 — clés PRNG, carry identique à P1
        mask:         (B, max_it) bool — produit par classify, True aux virages
        t_effectives: (B,) int32 — instant de fin effective par sample (D13)
        propagate_fn: callable vmappé (gp, state, prev, subk) → state_next
        layer_a_fn:   callable vmappé (state, prev, prev_delta)
                      → (dict_A : {str: (B,)}, current_delta : (B,))
        layer_b_fn:   callable unitaire ((state, prev, gp, key, active))
                      → (n_B,) array — à consommer via lax.map
        max_it:       int static — nombre d'itérations
        B:            int static — taille du batch

    Returns:
        dict avec 4 clés (encore sur GPU) :
          'timeline_tables_A': {k: (B, max_it) float32} pour k in LAYER_A_KEYS
                               NaN pour t >= t_effectives[b]
          'feature_tables_B':  {k: (B, max_it) float32} pour k in LAYER_B_KEYS
                               NaN sauf aux (b, t) avec mask[b, t] == True
          'last_states':       (B, *state_shape) — état après max_it propagations
          'count_active':      (B,) int32 — nombre d'instants masqués par sample
    """
    # ── Initialisation (step 0) ──
    state = D_b
    prev_state = jnp.full_like(D_b, jnp.nan)                   # SD-SC5-2(B)
    keys = keys_b
    prev_delta_b = jnp.full((B,), jnp.nan, dtype=jnp.bfloat16)  # SD-SC5-2(A)

    # Buffers empilés NaN-éparses, float32 strict (SD-STRUCT-1)
    # OPT-SCATTER : 2 arrays empilés au lieu de 26 buffers dict individuels.
    # Réduit le carry de 26 feuilles pytree à 2 feuilles, et passe de
    # 26 scatter par step à 2. Gain mesuré 4-7× sur exp_scatter_fusion.
    n_A = len(LAYER_A_KEYS)   # 11
    n_B = len(LAYER_B_KEYS)   # 15
    buf_A = jnp.full((B, max_it, n_A), jnp.nan, dtype=jnp.bfloat16)
    buf_B = jnp.full((B, max_it, n_B), jnp.nan, dtype=jnp.bfloat16)
    count_active = jnp.zeros(B, dtype=jnp.int32)

    def step(carry, t):
        (state, prev_state, keys, prev_delta_b,
         buf_A, buf_B, count_active) = carry

        # 1. Split PRNG — IDENTIQUE à P1 (déterminisme trajectoire)
        new_keys, subk = split_keys_batch(keys)

        # 2. Key auxiliaire couche B — dérivée de subk, ne touche pas new_keys
        #    (split sur subk réutilisable comme PRNGKey ; carry reste = P1)
        _, subk_b = split_keys_batch(subk)

        # 3. Couche A inconditionnelle — 11 observables sur (state, prev_state)
        dict_A, current_delta = layer_a_fn(state, prev_state, prev_delta_b)
        current_delta = current_delta.astype(jnp.bfloat16)   # <-- Ajoutez cette ligne
        # 4. Couche B conditionnelle — skip total si aucun sample masqué
        #    OPT-SKIP-B : au lieu d'itérer lax.map sur B samples pour
        #    retourner 15 NaN chacun quand mask_t est tout-False, on
        #    conditionne l'appel entier au step. Gain proportionnel à
        #    (1 - densité_masque) : 100% pour les gammas COLLAPSED/FLAT,
        #    ~78% pour GAM-001 à 22% actif, ~40% pour GAM-009 à 60%.
        mask_t = mask[:, t]  # (B,) bool
        any_masked = jnp.any(mask_t)

        def _compute_b(operand):
            s, ps, gp, sk, mt = operand
            return lax.map(layer_b_fn, (s, ps, gp, sk, mt))

        def _skip_b(operand):
            return jnp.full((B, n_B), jnp.nan, dtype=jnp.bfloat16)

        vals_B = lax.cond(any_masked, _compute_b, _skip_b,
                          operand=(state, prev_state, gp_b, subk_b, mask_t))

        # 5. Empilement couche A dans l'ordre du registre (R1 — piège P13)
        #    vals_B est déjà empilé par le lax.cond ci-dessus.
        vals_A = jnp.stack([dict_A[k] for k in LAYER_A_KEYS], axis=1)  # (B, 11)

        # 6. Scatter couche A — conditionnel t < t_effectives (SD-SC5-3)
        #    Un seul where broadcast (B,) → (B,1) pour couvrir les 11 colonnes.
        valid_a = t < t_effectives  # (B,) bool
        buf_A = buf_A.at[:, t, :].set(
            jnp.where(valid_a[:, None], vals_A, jnp.nan))

        # 7. Scatter couche B — inconditionnel (NaN déjà produit par lax.cond)
        buf_B = buf_B.at[:, t, :].set(vals_B)
        count_active = count_active + mask_t.astype(jnp.int32)

        # 8. Propagation + mise à jour du carry
        state_next = propagate_fn(gp_b, state, prev_state, subk)
        new_prev_state = state
        new_state = state_next
        new_prev_delta_b = current_delta

        new_carry = (new_state, new_prev_state, new_keys, new_prev_delta_b,
                     buf_A, buf_B, count_active)
        return new_carry, None

    init_carry = (state, prev_state, keys, prev_delta_b,
                  buf_A, buf_B, count_active)
    final_carry, _ = lax.scan(step, init_carry, jnp.arange(max_it))

    (state_final, _, _, _,
     final_A_stacked, final_B_stacked, final_count) = final_carry

    # ── Dépilage en dicts nommés — itération sur LAYER_*_KEYS (R1) ──
    # Coût : O(1) par clé, slicing sans copie.
    timeline_tables_A = {k: final_A_stacked[:, :, i]
                         for i, k in enumerate(LAYER_A_KEYS)}
    feature_tables_B = {k: final_B_stacked[:, :, i]
                        for i, k in enumerate(LAYER_B_KEYS)}

    return {
        'timeline_tables_A': timeline_tables_A,
        'feature_tables_B':  feature_tables_B,
        'last_states':       state_final,
        'count_active':      final_count,
    }


# =========================================================================
# BLOC 9 — SUBBATCHPROCESS
# =========================================================================

class SubBatchProcess(multiprocessing.Process):
    """
    Processus isolé pour le traitement d'un sub-batch.

    Pipeline GPU (v17) :
        Discovery + matérialisation + compilation se font HORS budget,
        pendant que d'autres processus utilisent le GPU.
        Le budget VRAM est acquis juste avant les scans — contrôle la mémoire.
        Le sémaphore compute est acquis après le VRAM — contrôle l'exclusivité
        d'exécution GPU (évite la contention multi-processus sur les kernels).
        → Le GPU n'attend jamais : quand un processus libère le sémaphore,
          le suivant (déjà prêt et VRAM réservée) prend le relais instantanément.
    """

    def __init__(self, group_meta, sub_batch, vram_budget, gpu_semaphore,
                 est_vram_gb, col_queue):
        super().__init__()
        self.group_meta = group_meta
        self.sub_batch = sub_batch
        self.vram_budget = vram_budget
        self.gpu_semaphore = gpu_semaphore
        self.est_vram_gb = est_vram_gb
        self.col_queue = col_queue

    def run(self):
        # ── Étape 1: Discovery JAX (CPU — imports) ──
        gamma_reg = discover_gammas_jax()
        encoding_reg = discover_encodings_jax()
        modifier_reg = discover_modifiers_jax()

        # ── Étape 2: Paramètres du groupe ──
        B = len(self.sub_batch)
        rank_eff = self.group_meta['rank_eff']
        n_dof = self.group_meta['n_dof']
        max_it = self.group_meta['max_it']
        is_diff = self.group_meta['is_differentiable']

        # ── Étape 3: Matérialisation (HORS budget — prêt pendant que le GPU travaille) ──
        t0 = time.perf_counter()
        D_b, gp_b, keys_b, gamma_fn, in_axes_gp = materialize_batch(
            self.sub_batch, self.group_meta,
            gamma_reg, encoding_reg, modifier_reg)
        t_mat = time.perf_counter() - t0

        # ── Étape 4: Compilation des briques (HORS budget — idem) ──
        t0 = time.perf_counter()
        propagate_fn = build_propagate_vmap(gamma_fn, in_axes_gp)
        screening_fn = build_screening_vmap()
        # Builders couche A (11 observables O(n²)) et couche B (15 features
        # coûteuses sous lax.cond) — compilés ici hors budget VRAM.
        # Ordre des arguments figé par SC4-D8 (gamma_fn passé en closure JVP).
        layer_a_fn = build_layer_a_vmap(rank_eff)
        layer_b_fn = build_layer_b_fn(rank_eff, is_diff, gamma_fn)
        t_comp = time.perf_counter() - t0


        
        # Pré-transfert des données complètes sur GPU
        t0 = time.perf_counter()
        device = jax.devices()[0]
        D_b_gpu = jax.device_put(D_b, device)
        gp_b_gpu = {k: jax.device_put(v, device) for k, v in gp_b.items()}
        keys_b_gpu = jax.device_put(keys_b, device)
        t_transfer_init = time.perf_counter() - t0


        try:
            # ── Acquire sémaphore compute — exclusivité GPU ──
            # Le processus a déjà préparé son travail (matérialisation,
            # compilation) et réservé sa VRAM. Il attend maintenant que
            # le GPU soit libre pour exécuter P1+P2+transfert.
            # Quand le sémaphore est relâché, le processus suivant
            # (déjà prêt) prend le relais instantanément.
            t0 = time.perf_counter()
            self.gpu_semaphore.acquire()
            t_acq_sem = time.perf_counter() - t0
            try:
                # ── Étape 5: P1 scan ──
                t0 = time.perf_counter()
                p1_out = run_pass1(
                    D_b, gp_b, keys_b,
                    propagate_fn, screening_fn,
                    max_it, B)
                #jax.block_until_ready(p1_out['frob'])
                t_p1 = time.perf_counter() - t0

                # ── Étape 6: Classification + masque ──
                t0 = time.perf_counter()
                (statuses, t_effectives, regimes, periods, mask,
                 p1_features, mask_features) = classify_and_mask(
                    p1_out['frob'], p1_out['is_finite'], p1_out['last_states'],
                    p1_out['cos_dissim'], p1_out['delta_D'], max_it)
                #jax.block_until_ready(mask)
                t_classify = time.perf_counter() - t0

                # ── Étape 7: P2 scan (refondu SC5 — couches A et B) ──
                t0 = time.perf_counter()
                p2_out = run_pass2(
                    D_b, gp_b, keys_b, mask, t_effectives,
                    propagate_fn, layer_a_fn, layer_b_fn,
                    max_it, B)
                #jax.block_until_ready(p2_out['count_active'])
                t_p2 = time.perf_counter() - t0

                # ── Étape 8: Transfert GPU → CPU ──
                t0 = time.perf_counter()
                classify_out = {
                    'statuses': statuses,
                    't_effectives': t_effectives,
                    'regimes': regimes,
                    'periods': periods,
                    'mask': mask,
                    'p1_features': p1_features,
                    'mask_features': mask_features,
                }
                cpu_data = transfer_to_cpu(p1_out, p2_out, classify_out)
                t_transfer = time.perf_counter() - t0

            finally:
                # ── Release sémaphore compute — le GPU est libre ──
                self.gpu_semaphore.release()


        finally:
            # ── Release budget VRAM — garanti même en cas de crash ──
            self.vram_budget.release(self.est_vram_gb)

        # ── Étape 9: Post-traitement numpy (HORS budget — CPU seul) ──
        # Refondu SC6 : séparation stricte C1 / C2 / C3 (DOC1 §5).
        # - C1 : agrégats unifiés sur timeline_tables_A + feature_tables_B
        # - C2 : post-process grammaire signal sur timeline_tables_A
        # - C3 : post-process grammaire virages (phasic)
        # Les 2 annotations SC2 et la surface élargie SC5 sont
        # intégralement refermées par cette refonte (cf. HANDOFF P22).
        t0 = time.perf_counter()

        t_eff = cpu_data['t_effectives']
        tables_A = cpu_data['timeline_tables_A']
        tables_B = cpu_data['feature_tables_B']
        mask = cpu_data['mask']

        # ── C1 — agrégats scalaires (69 sorties, SC1-D2 unifiée) ──
        c1_agg = aggregate_by_agg_map({**tables_A, **tables_B})

        # ── C2 — post-process grammaire signal sur couche A (34 sorties) ──
        dmd_features      = compute_dmd_on_observables(tables_A, t_eff)
        f6_features       = compute_f6_on_observables(tables_A, t_eff)
        stationarity      = compute_stationarity(tables_A, t_eff)
        entropy_prod      = compute_entropy_production(tables_A, t_eff)
        autocorr_features = compute_autocorrelations(tables_A)
        pnn_features      = compute_pnn_features(tables_A)
        temporal_features = compute_temporal_features(tables_A, t_eff)

        c2_signal = {
            **dmd_features,
            **f6_features,
            **stationarity,
            **autocorr_features,
            **pnn_features,
            **temporal_features,
            'f2_entropy_production_rate': entropy_prod,
        }

        # ── C3 — post-process grammaire virages (9 sorties) ──
        # Sources : svn + rank natifs couche B ; mode_asym_o2 projeté
        # depuis couche A au masque (§1.7 Option 2 licite, SC1-D6).
        c3_virages = compute_phasic_features(tables_B, tables_A, mask)

        # ── Metadata parquet : p1_features + mask_features + meta (15) ──
        # meta_turbulence corrigé SC6-D5 : utiliser frob[b, t_eff-1] au lieu
        # de frob[b, -1] (ancien code silencieusement cassé avec NaN SC5-D3).
        frob_tl = tables_A['frob']
        B_local = t_eff.shape[0]
        frob_final = np.array(
            [frob_tl[i, int(t_eff[i]) - 1] if int(t_eff[i]) >= 1 else np.nan
             for i in range(B_local)],
            dtype=np.float32,
        )
        frob_initial = frob_tl[:, 0].astype(np.float32)
        meta_turbulence = ((frob_final - frob_initial)
                           / (float(max_it) + EPS)
                           * TURBULENCE_CONSTANT).astype(np.float32)

        metadata = {
            **cpu_data['p1_features'],
            **cpu_data['mask_features'],
            'meta_n_svd':       cpu_data['count_active'].astype(np.float32),
            'meta_turbulence':  meta_turbulence,
            'meta_t_effective': t_eff.astype(np.float32),
        }

        t_post = time.perf_counter() - t0

        # ── Étape 10: Assemblage + envoi ──
        col_data = build_col_data(
            buffers=cpu_data,
            c1_agg=c1_agg,
            c2_signal=c2_signal,
            c3_virages=c3_virages,
            metadata=metadata,
            group_meta=self.group_meta,
            sub_batch=self.sub_batch,
            is_diff=is_diff,
            rank_eff=rank_eff,
            B=B,
            max_it=max_it,
        )

        if VERBOSE:
            n_active = int(np.sum(cpu_data['count_active']))
            n_active_possible = B * max_it
            print(f"[PID {self.pid}] B={B} max_it={max_it} DoF={n_dof} Rang={rank_eff} Diff={is_diff}   "
                  f"active={n_active}/{n_active_possible} "
                  f"({100*n_active/n_active_possible:.1f}%) "
                  f"VRAM est={self.est_vram_gb:.2f}Go")
            # Afficher tous les temps détaillés
            print(f"  Matérialisation: {t_mat:.3f}s | Compilation: {t_comp:.3f}s")
            print(f"  P1: {t_p1:.3f}s | Classification: {t_classify:.3f}s | P2: {t_p2:.3f}s | Transfert: {t_transfer:.3f}s | Post-traitement: {t_post:.3f}s")

        self.col_queue.put(col_data)
