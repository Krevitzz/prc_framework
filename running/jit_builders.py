"""
Briques JIT compilables du pipeline PRC — propagation, screening, couche A, couche B.

Chaque builder capture les paramètres de compilation (gamma_fn, rank_eff, is_diff)
dans une closure et retourne une fonction vmappée/mappée prête à l'emploi.

Principe (charter §1.6):
    Les mesures se répartissent en deux classes de coût:
    - Dynamiques O(n²) : couplées à chaque itération de propagation
      (screening 4 timelines + couche A 11 observables)
    - Spectrales O(n³) + JVP : couche B, uniquement aux instants masqués

Architecture (décisions D1-D5 chantier 1, livrées SC3+SC4):
    - Propagation, screening, couche A : jax.vmap — parallèle inconditionnel sur B
    - Couche B (15 features) : lax.map + lax.cond unique englobant SVD + JVP
      (benchmark D2: 15× plus rapide que vmap à 10% de densité masque)

Les briques ne sont PAS jit-ées individuellement. Elles sont composées
dans les passes (P1/P2) qui forment le programme XLA compilé.

@ROLE    Briques de calcul compilables — fonctions pures vmappées/mappées
@LAYER   running

@EXPORTS
  split_keys_batch(keys)                          → (Keys_B, Keys_B)
  build_propagate_vmap(gamma_fn, in_axes)         → fn(gp, state, prev, keys) → state_next
  build_screening_vmap()                          → fn(state, prev) → {4 scalaires}
  build_layer_a_vmap(rank_eff)                    → fn(state, prev, prev_delta) → ({11 scalaires}, current_delta)
  build_layer_b_fn(rank_eff, is_diff, gamma_fn)   → fn((state, prev, gp, key, active)) → (n_B,) array

@CONFORMITY
  OK    Fonctions pures, zéro état global (P5)
  OK    Constantes depuis features_registry (P4)
  OK    Kernel aveugle au domaine — gamma_fn est un callable opaque (P3/K4)
  OK    Couche A en grammaire signal, couche B en grammaire virages (P10)
  OK    Branchements rank_eff / is_diff au niveau Python (compilation), pas runtime (P3/K3)
  OK    SVD mode 0 unique partagée F1+F2+F3.mode0 ; SVD mode 1 unique conditionnelle (D11)
  OK    JVP F4 sous lax.cond unique avec SVD — payé seulement aux virages (D3, D5)

@RESOLVED
  D-PURGE-DYN  build_dynamic_vmap, DynamicAccumulator, ACCUMULATOR_KEYS,
               init_dynamic_acc, update_dynamic_acc retirés (SC2, charter D2/D3).
  D-SC3-LAYER-A  build_layer_a_vmap livrée — 11 observables O(n²) calculés
                 à chaque step P2 sans condition. Inclut frob, delta_D,
                 cos_dissim, is_finite (recouvrement P1 assumé, piège P6),
                 shannon_comp, ipr, mode_asymmetry_o2 (migré depuis couche B,
                 piège P5), bregman_cost, lyap_empirical, frob_gradient
                 (rank≥3, NaN structurel sinon), volume_proxy.
  D-SC4-LAYER-B  build_layer_b_fn livrée — 15 features sous lax.cond unique
                 englobant SVD + JVP. Anciens helpers _f1_spectral,
                 _f2_informational, _f3_rank2, _f3_rank3 et build_spectral_fn
                 supprimés. SVD factorisée (mode 0 toujours, mode 1 si rank≥3).
                 JVP via 1 split key + 2 calls (jvp + vjp) sur tangent gaussien.
                 Sigmas plus exposés en sortie (D9 — DMD migré sur observables
                 couche A).
"""

import jax
import jax.numpy as jnp
from jax import lax, vmap

from running.features_registry import (
    EPS,
    LAYER_A_KEYS,
    LAYER_B_KEYS,
)


# =========================================================================
# HELPERS PRNG
# =========================================================================

_split_keys_one = jax.vmap(lambda k: jax.random.split(k))

def split_keys_batch(keys_b):
    """Split un batch de clés PRNG. Retourne (keys_next, subkeys)."""
    both = _split_keys_one(keys_b)
    return both[:, 0, :], both[:, 1, :]


# =========================================================================
# 1a — PROPAGATION (vmap)
#
# Applique Γ à l'état D pour produire D_next.
# Ancrage axiomatique: A2 — c'est l'action de Γ sur D.
# Le kernel ne sait pas ce que Γ représente (P3/K4).
# =========================================================================

def build_propagate_vmap(gamma_fn, in_axes_gp):
    """
    Construit la brique de propagation vmappée.

    gamma_fn:     callable(state, prev, gp, key) → state_next
    in_axes_gp:   Dict[str, int] — axes vmap pour les gamma_params

    Retourne: vmap(gamma_one) — non jit-ée, composée dans les passes.
    """
    def _gamma_one(gp, state, prev_state, key):
        return gamma_fn(state, prev_state, gp, key)
    return vmap(_gamma_one, in_axes=(in_axes_gp, 0, 0, 0))


# =========================================================================
# 1b — SCREENING (vmap)
#
# 4 scalaires par itération, O(n²).
# Servent à la fois au suivi temps réel (timelines P1)
# et à la classification/masque (bloc 4).
#
# - delta_D:     variation relative de D — A1 (D change-t-il?)
# - frob:        norme de D — A1 (D existe-t-il?)
# - is_finite:   santé numérique du state
# - cos_dissim:  dissimilarité directionnelle — sensible aux changements
#                de direction indépendamment de l'amplitude
#
# Note: ces 4 mesures sont recalculées à l'identique en couche A (P2).
# Recouvrement assumé (charter §3.4 piège P6, handoff §8 P6) au profit
# de la simplicité — P1 reste minimal et la couche A reste autonome.
# =========================================================================

def build_screening_vmap():
    """
    Construit la brique de screening vmappée.
    Aucune dépendance à gamma, rank, n_dof — opère sur des tenseurs bruts.
    """
    def _screen_one(state, prev_state):
        frob_s = jnp.linalg.norm(state)
        frob_p = jnp.linalg.norm(prev_state)
        delta_D = jnp.linalg.norm(state - prev_state) / (frob_s + EPS)
        is_fin = jnp.all(jnp.isfinite(state)).astype(jnp.bfloat16)
        dot = jnp.sum(state * prev_state)
        cos_sim = dot / (frob_s * frob_p + EPS)
        cos_dissim = 1.0 - jnp.clip(cos_sim, -1.0, 1.0)
        return {
            'delta_D': delta_D,
            'frob': frob_s,
            'is_finite': is_fin,
            'cos_dissim': cos_dissim,
        }
    return vmap(_screen_one)


# =========================================================================
# 1c — COUCHE A (vmap) — 11 observables O(n²)
#
# Calculée à chaque step de P2 pour TOUS les samples du batch (pas de
# condition runtime). Forme les 11 timelines complètes uniformes qui
# nourrissent la grammaire signal du post-process (DMD sur observables,
# F6 sur paires d'observables, autocorr, PNN, temporal, stationnarité).
#
# Aucune SVD, aucune différentiation. Coût O(n²) dominé par les normes
# et le repli matriciel pour mode_asymmetry_o2 et frob_gradient.
#
# Branchement Python sur rank_eff :
#   - rank_eff == 2 : frob_gradient = NaN structurel (un seul mode non trivial)
#   - rank_eff >= 3 : frob_gradient calculé sur ratio std(row_norms) M0/M1
# Le branchement est à la compilation (deux variantes JIT distinctes),
# pas à l'exécution (charter P3/K3).
#
# lyap_empirical nécessite prev_delta (variation au step précédent).
# P2 (SC5) maintient prev_delta_b dans son carry et le passe à chaque step.
# Convention : prev_delta = NaN au step 0 → lyap_empirical = NaN au step 1
# (premier step où le calcul est possible structurellement).
#
# La fonction retourne :
#   - features : dict 11 entrées dans l'ordre EXACT de LAYER_A_KEYS
#                (piège P13 — clés et ordre figés par le registre SC1)
#   - current_delta : (B,) — ||state - prev_state|| par sample,
#                     à passer comme prev_delta_b au step suivant
# =========================================================================

def build_layer_a_vmap(rank_eff: int):
    """
    Construit la brique couche A vmappée.

    Args:
        rank_eff: rang effectif (≥2). Détermine si frob_gradient est calculé
                  (rank≥3) ou retourné NaN constant (rank==2). Branchement
                  Python à la compilation, pas runtime.

    Returns:
        vmap(_layer_a_one) — fonction vmappée non jit-ée, composée dans P2.

        Signature de la fonction retournée :
            fn(state_b, prev_state_b, prev_delta_b)
              → (features_dict, current_delta_b)

            state_b       : (B, n_dof, ...) — tenseur D au step courant
            prev_state_b  : (B, n_dof, ...) — tenseur D au step précédent
            prev_delta_b  : (B,)           — ||D_{t-1} - D_{t-2}|| par sample
                                              (NaN attendu aux steps 0 et 1)

            features_dict : Dict[str, jax.Array(B,)]
                            clés = LAYER_A_KEYS exact, ordre identique
                            11 entrées, chaque valeur de shape (B,)
            current_delta_b : (B,) — ||state - prev_state|| par sample
    """
    has_frob_gradient = (rank_eff >= 3)

    def _layer_a_one(state, prev_state, prev_delta):
        # ── Quantités intermédiaires partagées ──
        flat = state.reshape(-1)
        abs_flat = jnp.abs(flat)
        frob_s = jnp.linalg.norm(state)
        frob_p = jnp.linalg.norm(prev_state)
        diff = state - prev_state
        current_delta = jnp.linalg.norm(diff)

        # ── A.1 — frob (||D||_F, A1) ──
        a1_frob = frob_s

        # ── A.2 — delta_D (||D_t - D_{t-1}|| / ||D_t||, A1+A2) ──
        a2_delta_D = current_delta / (frob_s + EPS)

        # ── A.3 — cos_dissim (1 - cos(D_t, D_{t-1}), A2) ──
        dot_sp = jnp.sum(state * prev_state)
        cos_sim = dot_sp / (frob_s * frob_p + EPS)
        a3_cos_dissim = 1.0 - jnp.clip(cos_sim, -1.0, 1.0)

        # ── A.4 — is_finite (santé numérique) ──
        a4_is_finite = jnp.all(jnp.isfinite(state)).astype(jnp.bfloat16)

        # ── A.5 — shannon_comp (entropie sur composantes |x_i|, A1) ──
        sum_abs = jnp.sum(abs_flat)
        p_comp = abs_flat / (sum_abs + EPS)
        a5_shannon = -jnp.sum(p_comp * jnp.log(p_comp + EPS))

        # ── A.6 — ipr ((Σ|x|²)² / Σ|x|⁴ / n, A1) ──
        s2 = jnp.sum(flat * flat)
        s4 = jnp.sum(flat ** 4)
        n_components = flat.shape[0]
        a6_ipr = (s2 * s2 / (s4 + EPS)) / n_components

        # ── A.7 — mode_asymmetry_o2 (||M-M^T|| / ||M||, A1, migré depuis couche B) ──
        M = state.reshape(state.shape[0], -1)
        k = min(M.shape[0], M.shape[1])
        M_sq = M[:k, :k]
        a7_mode_asym = jnp.linalg.norm(M_sq - M_sq.T) / (jnp.linalg.norm(M) + EPS)

        # ── A.8 — bregman_cost (||D_t-D_{t-1}||² / ||D_t||², A2) ──
        a8_bregman = (current_delta * current_delta) / (frob_s * frob_s + EPS)

        # ── A.9 — lyap_empirical (log(current_delta / prev_delta), A2) ──
        # NaN structurel aux steps 0-1 via prev_delta=NaN injecté par P2.
        a9_lyap = jnp.log((current_delta + EPS) / (prev_delta + EPS))

        # ── A.10 — frob_gradient (std(row_norms_M0)/std(row_norms_M1), A2, rank≥3) ──
        if has_frob_gradient:
            row_norms_M0 = jnp.linalg.norm(M, axis=1)
            M1 = jnp.moveaxis(state, 1, 0).reshape(state.shape[1], -1)
            row_norms_M1 = jnp.linalg.norm(M1, axis=1)
            a10_frob_grad = jnp.std(row_norms_M0) / (jnp.std(row_norms_M1) + EPS)
        else:
            # NaN structurel rank_eff == 2 (déjà déclaré dans
            # FEATURES_STRUCTURAL_NAN du registre).
            a10_frob_grad = jnp.array(jnp.nan, dtype=jnp.bfloat16)

        # ── A.11 — volume_proxy (||D||_F · ||D||_∞ / ||D||_1, A2) ──
        norm_inf = jnp.max(abs_flat)
        norm_1 = sum_abs  # déjà calculé pour shannon_comp
        a11_volume = (frob_s * norm_inf) / (norm_1 + EPS)

        # ── Assemblage dans l'ordre EXACT de LAYER_A_KEYS (piège P13) ──
        features = {
            'frob':              a1_frob,
            'delta_D':           a2_delta_D,
            'cos_dissim':        a3_cos_dissim,
            'is_finite':         a4_is_finite,
            'shannon_comp':      a5_shannon,
            'ipr':               a6_ipr,
            'mode_asymmetry_o2': a7_mode_asym,
            'bregman_cost':      a8_bregman,
            'lyap_empirical':    a9_lyap,
            'frob_gradient':     a10_frob_grad,
            'volume_proxy':      a11_volume,
        }
        return features, current_delta

    return vmap(_layer_a_one, in_axes=(0, 0, 0))


# =========================================================================
# 1d — COUCHE B (lax.cond unique) — 15 features coûteuses aux virages
#
# Conçue pour être consommée par lax.map à l'intérieur du scan P2 (SC5).
# Le lax.cond englobe TOUT le calcul lourd : SVD mode 0, SVD mode 1
# (si rank≥3), JVP forward + vjp (si différentiable). Si active=False,
# rien n'est exécuté pour ce sample — c'est le vrai conditionnel XLA qui
# réalise l'économie compute (DOC2 SC4 point critique).
#
# Trois familles couvertes :
#
#   F1 — Spectrale (A1) — 6 features depuis sigmas mode 0
#     f1_effective_rank, f1_spectral_gap, f1_nuclear_frobenius_ratio,
#     f1_sv_decay_rate, f1_rank1_residual, f1_condition_number
#
#   F2 — Entropie spectrale (A1, holographie Ryu-Takayanagi) — 2 features
#     f2_von_neumann_entropy (test direct A3), f2_renyi2_entropy
#
#   F3 — Enchevêtrement multi-modal (A1, Rovelli) — 3 features
#     f3_entanglement_entropy_mode0 (= VN numériquement, SD-SC4-6)
#     f3_entanglement_entropy_mode1 (NaN structurel si rank==2)
#     f3_inter_mode_sv_var          (NaN structurel si rank==2)
#
#   F4 — Géométrie locale différentielle (A2, lien A3) — 4 features
#     f4_trace_J, f4_jvp_norm, f4_jacobian_asymmetry, f4_local_lyapunov
#     (NaN structurels si is_diff==False)
#
# Quatre variantes JIT distinctes selon (rank_eff, is_diff) :
#   (2, False) : F1+F2+F3.mode0 calculés ; F3.mode1 + F4 = NaN structurels
#   (2, True)  : F1+F2+F3.mode0 + F4 calculés ; F3.mode1 = NaN structurels
#   (≥3,False) : F1+F2+F3.mode0 + F3.mode1 calculés ; F4 = NaN structurels
#   (≥3,True)  : tout calculé
#
# Note : `mode_asymmetry` n'est PAS en couche B — migré en couche A
# sous le nom mode_asymmetry_o2 (handoff piège P5). Ne pas le réintroduire.
# `shannon_comp` également retiré de la couche B (D11) — calculé uniquement
# en couche A.
# =========================================================================

# Listes statiques utilisées par build_layer_b_fn pour les NaN structurels.
_F4_KEYS = ['f4_trace_J', 'f4_jvp_norm',
            'f4_jacobian_asymmetry', 'f4_local_lyapunov']
_F3_MODE1_KEYS = ['f3_entanglement_entropy_mode1', 'f3_inter_mode_sv_var']


def _compute_f1(sigmas_0, frob_state):
    """6 features F1 depuis les valeurs singulières du mode 0."""
    # f1_effective_rank = exp(Shannon spectrale normalisée)
    p_s = sigmas_0 / (jnp.sum(sigmas_0) + EPS)
    eff_rank = jnp.exp(-jnp.sum(p_s * jnp.log(p_s + EPS)))

    # f1_spectral_gap = (σ_0 - σ_1) / σ_0
    gap = (sigmas_0[0] - sigmas_0[1]) / (sigmas_0[0] + EPS)

    # f1_nuclear_frobenius_ratio = Σσ_i / sqrt(Σσ_i²)  (SD-SC4-5 : sans sqrt(n))
    nuclear = jnp.sum(sigmas_0)
    frob_sigmas = jnp.linalg.norm(sigmas_0)
    nf_ratio = nuclear / (frob_sigmas + EPS)

    # f1_sv_decay_rate = pente de régression linéaire log(σ_i) vs i
    log_s = jnp.log(sigmas_0 + EPS)
    i_vals = jnp.arange(sigmas_0.shape[0], dtype=jnp.bfloat16)
    i_mean = jnp.mean(i_vals)
    s_mean = jnp.mean(log_s)
    cov_is = jnp.mean((i_vals - i_mean) * (log_s - s_mean))
    var_i = jnp.mean((i_vals - i_mean) ** 2) + EPS
    decay = -cov_is / var_i

    # f1_rank1_residual = sqrt(Σ_{i≥1} σ_i²) / ||D||_F
    rank1_res = jnp.sqrt(jnp.sum(sigmas_0[1:] ** 2)) / (frob_state + EPS)

    # f1_condition_number = σ_0 / σ_{-1}
    cond_num = sigmas_0[0] / (sigmas_0[-1] + EPS)

    return {
        'f1_effective_rank':           eff_rank,
        'f1_spectral_gap':             gap,
        'f1_nuclear_frobenius_ratio':  nf_ratio,
        'f1_sv_decay_rate':            decay,
        'f1_rank1_residual':           rank1_res,
        'f1_condition_number':         cond_num,
    }


def _compute_f2_and_f3_mode0(sigmas_0):
    """3 features depuis les sigmas mode 0 — F2 (VN, Rényi-2) + F3 mode 0.

    Note (SD-SC4-6 option a) : f3_entanglement_entropy_mode0 est
    mathématiquement identique à f2_von_neumann_entropy (entropie spectrale
    de M0 = repli mode 0). Les deux clés sont conservées pour cohérence
    sémantique avec le registre, mais pointent vers la même valeur.
    """
    p_sq = sigmas_0 ** 2 / (jnp.sum(sigmas_0 ** 2) + EPS)
    log_p_sq = jnp.log(p_sq + EPS)

    von_neumann = -jnp.sum(p_sq * log_p_sq)
    renyi2 = -jnp.log(jnp.sum(p_sq ** 2) + EPS)
    entanglement_mode0 = von_neumann  # SD-SC4-6 — valeur partagée

    return {
        'f2_von_neumann_entropy':         von_neumann,
        'f2_renyi2_entropy':              renyi2,
        'f3_entanglement_entropy_mode0':  entanglement_mode0,
    }


def _compute_f3_mode1(state, sigma_0_max):
    """2 features F3 depuis SVD mode 1 (rank ≥ 3 uniquement).

    Args:
        state: tenseur D
        sigma_0_max: σ_0 du mode 0 déjà calculé (réutilisé pour inter_mode_sv_var)
    """
    M1 = jnp.moveaxis(state, 1, 0).reshape(state.shape[1], -1)
    sigmas_1 = jnp.linalg.svd(M1, compute_uv=False)

    # f3_entanglement_entropy_mode1 = VN sur sigmas_1
    p1_sq = sigmas_1 ** 2 / (jnp.sum(sigmas_1 ** 2) + EPS)
    entanglement_mode1 = -jnp.sum(p1_sq * jnp.log(p1_sq + EPS))

    # f3_inter_mode_sv_var = variance des premiers SV des deux modes
    inter_var = jnp.var(jnp.array([sigma_0_max, sigmas_1[0]]))

    return {
        'f3_entanglement_entropy_mode1':  entanglement_mode1,
        'f3_inter_mode_sv_var':           inter_var,
    }


def _compute_f4(state, prev_state, gp, key, gamma_fn):
    """4 features F4 depuis 2 JVPs (différentiable uniquement).

    SD-SC4-1 : JVP par rapport à `state` seul, autres args fixés par closure.
    SD-SC4-2 : 1 tangent gaussien par virage (Hutchinson un-shot).
    SD-SC4-7 : split propre key → (key_v, key_gamma) pour indépendance
               statistique entre tangent et stochasticité éventuelle de gamma.
    """
    key_v, key_gamma = jax.random.split(key)
    v = jax.random.normal(key_v, state.shape, dtype=state.dtype)

    # Closure : gamma vu comme fonction du seul state
    def gamma_state(s):
        return gamma_fn(s, prev_state, gp, key_gamma)

    # JVP forward : J·v
    _, Jv = jax.jvp(gamma_state, (state,), (v,))
    # VJP : J^T·v (adjoint sur le même point)
    _, vjp_fn = jax.vjp(gamma_state, state)
    (Jtv,) = vjp_fn(v)

    norm_Jv = jnp.linalg.norm(Jv)

    return {
        'f4_trace_J':            jnp.sum(v * Jv),
        'f4_jvp_norm':           norm_Jv,
        'f4_jacobian_asymmetry': jnp.linalg.norm(Jv - Jtv) / (norm_Jv + EPS),
        'f4_local_lyapunov':     jnp.log(norm_Jv + EPS),
    }


def build_layer_b_fn(rank_eff: int, is_diff: bool, gamma_fn):
    """
    Construit la fonction unitaire couche B — 15 features pour UN sample.

    Le corps du compute est assemblé en Python : les branchements rank_eff
    et is_diff sont résolus AVANT tracing. XLA ne voit qu'un seul chemin
    de calcul par job — pas de branchement SVD/SVD+JVP dans le programme
    HLO. Seul le lax.cond(active) reste dans le XLA : compute ou NaN.

    Retourne un array (n_B,) au lieu d'un dict de 15 scalaires — réduit
    le nombre de feuilles pytree dans le lax.map de 15 à 1, éliminant
    ~90% des opérations HLO de plumbing du while_loop.

    Args:
        rank_eff:  rang effectif (≥2). Branchement Python : si rank_eff==2,
                   F3 mode 1 = NaN constant dans le compute.
        is_diff:   True si gamma est différentiable. Branchement Python :
                   si False, F4 = NaN constant dans le compute.
        gamma_fn:  callable opaque (P3/K4).

    Returns:
        layer_b_one: fonction unitaire pour lax.map.
            Signature : ((state, prev_state, gp, key, active)) → (n_B,)
            n_B = len(LAYER_B_KEYS) = 15
            Tout NaN si active=False.
    """
    n_B = len(LAYER_B_KEYS)
    has_mode1 = (rank_eff >= 3)
    nan_result = jnp.full((n_B,), jnp.nan, dtype=jnp.bfloat16)

    # ── Compute assemblé en Python — XLA ne voit qu'UN chemin ──
    # Les if Python (has_mode1, is_diff) sont résolus au moment du build,
    # avant toute compilation XLA. Le programme HLO résultant ne contient
    # que les opérations pertinentes au job.
    def _compute(state, prev_state, gp, key):
        # SVD mode 0 unique
        M0 = state.reshape(state.shape[0], -1)
        sigmas_0 = jnp.linalg.svd(M0, compute_uv=False)
        frob_state = jnp.linalg.norm(state)

        # F1 + F2 + F3 mode 0 (toujours)
        f1 = _compute_f1(sigmas_0, frob_state)
        f23m0 = _compute_f2_and_f3_mode0(sigmas_0)

        # F3 mode 1 — Python branch, résolu avant JIT
        if has_mode1:
            f3m1 = _compute_f3_mode1(state, sigmas_0[0])
        else:
            f3m1 = {k: jnp.array(jnp.nan, dtype=jnp.bfloat16)
                    for k in _F3_MODE1_KEYS}

        # F4 JVP — Python branch, résolu avant JIT
        if is_diff:
            f4 = _compute_f4(state, prev_state, gp, key, gamma_fn)
        else:
            f4 = {k: jnp.array(jnp.nan, dtype=jnp.bfloat16)
                  for k in _F4_KEYS}

        # Assemblage ordonné → (n_B,) — piège P13
        merged = {**f1, **f23m0, **f3m1, **f4}
        result = jnp.stack([merged[k] for k in LAYER_B_KEYS])
        return result.astype(jnp.bfloat16)   # <-- conversion explicite

    def layer_b_one(args):
        state, prev_state, gp, key, active = args
        def true_fun():
            return _compute(state, prev_state, gp, key)
        def false_fun():
            return nan_result
        return lax.cond(active, true_fun, false_fun)

    return layer_b_one
