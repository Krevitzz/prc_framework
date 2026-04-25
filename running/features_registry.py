"""
Source de vérité unique pour les features, clés, constantes et schéma PRC.

Registre pur : exporte des constantes, listes et dicts.
Aucune fonction de calcul, aucun import lourd.
Tout module du pipeline qui a besoin d'un nom de feature,
d'une constante scientifique ou d'une définition de colonne
importe depuis ce fichier.

@ROLE    Registre central features + constantes scientifiques + schéma
@LAYER   running

@EXPORTS
  EPS, MIN_ABS_VARIATION                → float      | epsilon numériques globaux
  DMD_*, PNN_*, IQR_*, PLATEAU_*        → float|int  | constantes traitement signal
  TURBULENCE_CONSTANT                   → float      | constante meta
  EXPLOSION_*, COLLAPSE_*               → float|int  | constantes classification
  ZONE_MARGIN_*, MIN_SVD_*              → float|int  | constantes masque adaptatif
  TUKEY_FENCE_MULT, MONOTONE_TOL        → float|int  | constantes détection virages
  FLAT_CV_THRESHOLD                     → float      | constante classification FLAT
  AUTOCORR_MAX_LAG_DIVISOR              → int        | constante autocorrélation P1
  STATIONARITY_TAIL_FRAC                → float      | constante stationnarité C2
  PASS1_KEYS                            → List[str]  | clés screening P1 (4)
  LAYER_A_KEYS                          → List[str]  | clés observables couche A (11)
  LAYER_B_KEYS                          → List[str]  | clés features couche B (15)
  AGG_MAP                               → Dict       | mapping agrégation C1 (26 entrées)
  AGG_FEATURE_NAMES                     → List[str]  | noms scalaires C1 dérivés (69)
  DMD_FEATURE_NAMES                     → List[str]  | C2 DMD (4)
  F6_FEATURE_NAMES                      → List[str]  | C2 transfer entropy (5)
  AUTOCORR_FEATURE_NAMES                → List[str]  | C2 autocorrélation (3)
  PNN_FEATURE_NAMES                     → List[str]  | C2 PNN40 (2)
  TEMPORAL_FEATURE_NAMES                → List[str]  | C2 temporal (8)
  STATIONARITY_FEATURE_NAMES            → List[str]  | C2 stationnarité (11)
  ENTROPY_PRODUCTION_FEATURE_NAMES      → List[str]  | C2 entropy production (1)
  PHASIC_FEATURE_NAMES                  → List[str]  | C3 phasic virages (9)
  P1_FEATURE_NAMES                      → List[str]  | metadata P1 (5)
  MASK_FEATURE_NAMES                    → List[str]  | metadata mask (7)
  META_FEATURE_NAMES                    → List[str]  | metadata meta (3)
  FEATURE_NAMES                         → List[str]  | tous les scalaires float32 (127)
  METADATA_COLUMNS                      → Dict       | colonnes non-float32 du schéma
  TIMELINE_COLUMNS_A                    → List[str]  | timelines uniformes couche A (11)
  TIMELINE_COLUMNS_B                    → List[str]  | timelines tronquées couche B (15)
  MASK_INDICES_COLUMN                   → str        | colonne indices du masque
  FEATURES_STRUCTURAL_NAN               → Dict       | NaN connus à la compilation
  FEATURES_RUNTIME_NAN                  → Dict       | NaN dépendants des données du run

@LIFECYCLE
  Aucun objet coûteux. Constantes et listes immuables.

@CONFORMITY
  OK   Zéro paramètre hardcodé dans le pipeline — tout est ici (P4)
  OK   Registre pur, pas de fonction de calcul (P5)
  OK   NaN structurels vs runtime séparés — clustering stratifié possible (P7)
  OK   Cohérence grammaticale déclarée par feature (P10)
  OK   Doublons éliminés — frob_norm/shannon_entropy une seule fois (D11)
  OK   health_has_inf/health_is_collapsed retirés — redondants avec run_status (D11)

@RESOLVED
  D-PURGE-01  Purge complète des anciens modes d'enregistrement (v14)
  D-PURGE-02  Retrait health_has_inf, health_is_collapsed (v15/SC1)
              Redondants avec run_status ∈ {EXPLOSION, COLLAPSED} et
              timeline_is_finite. Migration downstream : lire run_status
              ou le dernier élément de timeline_is_finite.
  D-REFACTOR-01  Suppression DYNAMIC_KEYS, SPECTRAL_KEYS, DYNAMIC_AGG_MAP,
                 SPECTRAL_AGG_MAP, DYNAMIC_KEY_TO_PREFIX, SPECTRAL_SKIP_KEYS.
                 Remplacés par LAYER_A_KEYS, LAYER_B_KEYS, AGG_MAP unifié.
"""

from typing import Dict, List


# =========================================================================
# S1 — CONSTANTES SCIENTIFIQUES
# Chaque constante : rôle, provenance, conséquence d'un changement.
# =========================================================================

# --- Numériques globales ---

# Epsilon numérique global — protège divisions et log.
# Provenance : convention numérique standard.
# Changer : < 1e-10 risque underflow, > 1e-6 masque des différences réelles.
EPS: float = 1e-8
MIN_ABS_VARIATION: float = 1e-12

# --- DMD (F7) ---

# Seuil pour compter les paires complexes DMD.
# Provenance : calibration EXP-3.
# Changer : plus petit → plus de faux positifs, plus grand → manque des oscillations faibles.
DMD_COMPLEX_THRESHOLD: float = 1e-4

# Facteur d'oubli DMD RLS streaming.
# Provenance : 1.0 = pas d'oubli (toutes les observations ont le même poids).
# Changer : < 1.0 donne plus de poids aux observations récentes.
DMD_FORGET_FACTOR: float = 1.0

# Échelle initiale matrice de covariance DMD (P_k = I * scale).
# Provenance : convention RLS — a priori diffus.
# Changer : plus grand → apprentissage initial plus rapide, potentiellement instable.
DMD_PK_INIT_SCALE: float = 1e4

# --- Temporelles / statistiques ---

# Seuil PNN (fraction de l'écart-type).
# Provenance : convention cardiologie adaptée (pNN40).
# Changer : modifie la sensibilité au changement point-à-point.
PNN_THRESHOLD: float = 0.4

# Quantiles IQR.
# Provenance : convention statistique standard.
IQR_Q_LOW: float = 0.25
IQR_Q_HIGH: float = 0.75

# Seuil plateau (fraction de la médiane des deltas).
# Provenance : calibration EXP-3.
# Changer : plus petit → plus de points classés plateau, plus grand → moins.
PLATEAU_THRESHOLD_MULT: float = 0.1

# Plancher seuil plateau (pour signaux quasi-constants).
# Provenance : calibration EXP-3.
# Changer : plus grand → signaux faibles échappent à la détection plateau.
PLATEAU_THRESHOLD_FLOOR: float = 1e-10

# Constante de turbulence : multiplicateur pour le ratio k/max_it.
# Provenance : réglable. 1.0 = ratio brut.
TURBULENCE_CONSTANT: float = 1.0

# --- Classification des runs ---

# Facteur d'explosion : frob > frob_mean × ce facteur = divergence candidate.
# Provenance : heuristique. 1e9 = le state a grandi d'un facteur milliard.
# Changer : plus petit → cutoff plus précoce, plus grand → tolère plus de croissance.
EXPLOSION_FACTOR: float = 1e9

# Fenêtre de monotonie pour le cutoff explosion.
# Le frob doit être strictement croissant sur les M derniers steps avant le seuil
# pour être classé en divergence. Si ça oscille, c'est pas une explosion.
# Provenance : heuristique. 10 steps = suffisant pour distinguer oscillation de divergence.
# Changer : plus grand → plus conservateur (moins de faux cutoffs).
EXPLOSION_MONOTONE_WINDOW: int = 10

# Seuil collapse relatif : std(state) / (mean(abs(state)) + eps).
# En-dessous, le state est numériquement constant.
# Provenance : convention — relatif aux données du run.
# Changer : plus petit → tolère plus de quasi-constance, plus grand → plus de collapses.
COLLAPSE_RELATIVE_THRESHOLD: float = 1e-4

# --- Mask adaptatif ---

# Marge minimale autour des virages détectés (itérations).
# Provenance : heuristique — capturer le contexte autour de la transition.
# Changer : plus grand → plus de SVD, meilleure résolution. Plus petit → moins de SVD.
ZONE_MARGIN_MIN: int = 1

# Marge proportionnelle à max_it.
# margin = max(ZONE_MARGIN_MIN, ceil(max_it × ZONE_MARGIN_FRACTION))
# Provenance : corrige MASK-W3 (marge fixe indépendante de max_it).
# Changer : plus grand → marge croît plus vite avec max_it.
ZONE_MARGIN_FRACTION: float = 0.01

# Plancher absolu de points SVD.
# Provenance : corrige MASK-W2 (runs monotones à 2 points = DMD impossible).
# Changer : plus grand → DMD plus fiable, plus de SVD. Plus petit → économie GPU.
MIN_SVD_POINTS: int = 4

# Plancher proportionnel de points SVD.
# K_min = max(MIN_SVD_POINTS, ceil(max_it × MIN_SVD_FRACTION))
# Provenance : corrige MASK-W2.
# Changer : plus grand → plancher croît avec max_it.
MIN_SVD_FRACTION: float = 0.02

# Multiplicateur fence Tukey pour détection outliers dans |diff(signal)|.
# Provenance : convention statistique (Tukey, 1977). 1.5 = standard.
# Changer : plus grand → moins d'outliers détectés, plus petit → plus sensible.
TUKEY_FENCE_MULT: float = 2

# Tolérance monotonie : nombre max de changements de signe du diff(frob)
# pour classifier un run comme monotone.
# Provenance : heuristique. 3 = tolère du bruit numérique.
# Changer : plus petit → plus de runs classés monotones. Plus grand → moins strict.
MONOTONE_TOL: int = 2

# Seuil CV pour classification FLAT.
# Si CV(cos_dissim) < ce seuil, le run est classé FLAT.
# Provenance : heuristique. 0.3 = variabilité faible.
# Changer : plus grand → plus de runs classés FLAT (moins de SVD).
FLAT_CV_THRESHOLD: float = 0.3

# --- Caractérisation P1 ---

# Diviseur pour tronquer l'autocorrélation de cos_dissim.
# max_lag = t_effective // AUTOCORR_MAX_LAG_DIVISOR
# Provenance : convention — chercher des périodes jusqu'à 1/4 de la durée.
# Changer : plus grand → recherche de périodes plus courtes seulement.
AUTOCORR_MAX_LAG_DIVISOR: int = 4

# --- Stationnarité (NOUVEAU SC1) ---

# Fraction de la timeline pour les agrégats first/last de stationnarité.
# stat_delta = |mean(last FRAC) - mean(first FRAC)| / (std(all) + eps)
# Provenance : DOC1 §5.3 C2.6. 0.2 = 20% du signal de chaque côté.
# Changer : plus petit → plus sensible aux bords, plus grand → plus lissé.
STATIONARITY_TAIL_FRAC: float = 0.2


# =========================================================================
# S2 — CLÉS SCAN PAR ÉTAPE
# =========================================================================
# Ces listes sont importées par les builders SC2/SC3/SC4 pour savoir
# quelles clés émettre dans leurs dicts de sortie à chaque step pertinent.
# Chaque feature est documentée par un bloc de commentaire standardisé.

# ── Screening P1 — 4 scalaires par itération, O(n²), inchangé ──
PASS1_KEYS: List[str] = [
    'delta_D',      # ||state - prev|| / (||state|| + eps)
    'frob',         # ||state||_F
    'is_finite',    # all(isfinite(state)) → 1.0 ou 0.0
    'cos_dissim',   # 1 - cos_sim(state, prev)
]

# ── Couche A — 11 observables, calculés à chaque step P2 pour tous les runs ──
# Grammaire : signal (timelines uniformes à pas 1, longueur t_effective).
# Calculés en O(n²) ou moins. Aucune SVD, aucune différentiation.
# Nourrit le post-process C2 (DMD, F6, autocorr, PNN, temporal, stationnarité)
# et les agrégats C1.
LAYER_A_KEYS: List[str] = [
    # ─────────────────────────────────────────────────────────────
    # A.1 — frob
    # Couche     : A (timeline complète, uniforme)
    # Grammaire  : signal
    # Question   : Q1 (D encode-t-il ?)
    # Ancrage    : A1 — D a une magnitude mesurable
    # Mesure     : ||D||_F (norme de Frobenius)
    # Rôle       : observable de volume. Nourrit temporal, autocorr, DMD, F6.
    # Agrégats   : mean, final, delta
    # NaN        : aucun (universel)
    # ─────────────────────────────────────────────────────────────
    'frob',

    # ─────────────────────────────────────────────────────────────
    # A.2 — delta_D
    # Couche     : A
    # Grammaire  : signal
    # Question   : Q1 + Q2 (D existe et change sous Γ)
    # Ancrage    : A1 + A2 — variation relative de D
    # Mesure     : ||D_t - D_{t-1}|| / (||D_t|| + eps)
    # Rôle       : détecte virages locaux. Nourrit classify, temporal.
    # Agrégats   : mean, std, total
    # NaN        : aucun
    # ─────────────────────────────────────────────────────────────
    'delta_D',

    # ─────────────────────────────────────────────────────────────
    # A.3 — cos_dissim
    # Couche     : A
    # Grammaire  : signal
    # Question   : Q2 (Γ agit-il ?)
    # Ancrage    : A2 — rotation de D dans son espace sous Γ
    # Mesure     : 1 - cos(D_t, D_{t-1})
    # Rôle       : changements de direction indépendants de l'amplitude.
    #              Nourrit classify, autocorr, F6.
    # Agrégats   : mean, std, final
    # NaN        : aucun
    # ─────────────────────────────────────────────────────────────
    'cos_dissim',

    # ─────────────────────────────────────────────────────────────
    # A.4 — is_finite
    # Couche     : A
    # Grammaire  : signal
    # Question   : (santé numérique, pas d'ancrage axiomatique direct)
    # Ancrage    : —
    # Mesure     : 1.0 si tous les éléments de D sont finis, 0.0 sinon
    # Rôle       : détection explosion. Passage 1→0 = t_effective.
    # Agrégats   : final
    # NaN        : aucun
    # ─────────────────────────────────────────────────────────────
    'is_finite',

    # ─────────────────────────────────────────────────────────────
    # A.5 — shannon_comp
    # Couche     : A
    # Grammaire  : signal
    # Question   : Q1 (D encode-t-il ?)
    # Ancrage    : A1 — distribution de l'information dans les composantes
    # Mesure     : entropie de Shannon sur |x_i| / Σ|x_j|
    # Rôle       : proxy O(n²) de la dispersion informationnelle.
    #              Nourrit F6, autocorr, PNN, temporal, stationnarité.
    #              Renommé depuis shannon_entropy pour marquer qu'il opère
    #              sur les composantes, pas sur le spectre (VN est en couche B).
    # Agrégats   : mean, std, final, delta
    # NaN        : aucun
    # ─────────────────────────────────────────────────────────────
    'shannon_comp',

    # ─────────────────────────────────────────────────────────────
    # A.6 — ipr
    # Couche     : A
    # Grammaire  : signal
    # Question   : Q1
    # Ancrage    : A1 — localisation de l'information dans D
    # Mesure     : (Σ|x_i|²)² / Σ|x_i|⁴, normalisé par n
    # Rôle       : complémentaire à shannon_comp. Distingue "diffus
    #              uniformément" de "concentré sur quelques composantes".
    #              Plus sensible aux queues que shannon_comp.
    # Agrégats   : mean, std, final
    # NaN        : aucun
    # NOUVEAU SC1
    # ─────────────────────────────────────────────────────────────
    'ipr',

    # ─────────────────────────────────────────────────────────────
    # A.7 — mode_asymmetry_o2
    # Couche     : A
    # Grammaire  : signal
    # Question   : Q1 + Q4
    # Ancrage    : A1 — structure relationnelle non triviale + Rovelli
    # Mesure     : ||M - M^T||_F / (||M||_F + eps), M = repli carré de D
    # Rôle       : asymétrie structurelle O(n²) sans SVD.
    #              Migré depuis couche B (ancien f3_mode_asymmetry) parce
    #              que le calcul est O(n²), pas O(n³).
    #              Nourrit autocorr, PNN, F6, phasic (sous-échantillonné).
    # Agrégats   : mean, std, final, delta
    # NaN        : aucun
    # MIGRÉ SC1 (ancien f3_mode_asymmetry en couche B)
    # ─────────────────────────────────────────────────────────────
    'mode_asymmetry_o2',

    # ─────────────────────────────────────────────────────────────
    # A.8 — bregman_cost
    # Couche     : A
    # Grammaire  : signal
    # Question   : Q2 + Q4
    # Ancrage    : A2 — coût informationnel du déplacement de D sous Γ
    # Mesure     : ||D_t - D_{t-1}||² / (||D_t||² + eps)
    # Rôle       : distance informationnelle entre états consécutifs,
    #              différente de delta_D (norme brute vs coût quadratique).
    #              Nourrit F6, temporal.
    # Agrégats   : mean, total
    # NaN        : aucun
    # ─────────────────────────────────────────────────────────────
    'bregman_cost',

    # ─────────────────────────────────────────────────────────────
    # A.9 — lyap_empirical
    # Couche     : A
    # Grammaire  : signal
    # Question   : Q2
    # Ancrage    : A2 — expansion/contraction exponentielle sous Γ
    # Mesure     : log(||D_{t+1} - D_t|| / (||D_t - D_{t-1}|| + eps))
    # Rôle       : seul proxy Lyapunov pour gammas non différentiables.
    #              Les gammas diff ont aussi f4_local_lyapunov (couche B).
    #              Nourrit F6.
    # Agrégats   : mean, std, final
    # NaN        : aucun
    # ─────────────────────────────────────────────────────────────
    'lyap_empirical',

    # ─────────────────────────────────────────────────────────────
    # A.10 — frob_gradient
    # Couche     : A
    # Grammaire  : signal
    # Question   : Q2 + Q4
    # Ancrage    : A2 — gradient inter-modes, précurseur Verlinde
    # Mesure     : std(row_norms_M0) / (std(row_norms_M1) + eps)
    # Rôle       : asymétrie entre modes tensoriels. Applicable seulement
    #              à rank ≥ 3 (rank 2 n'a qu'un seul mode non trivial).
    # Agrégats   : mean, final
    # NaN        : structurel si rank_eff == 2
    # ─────────────────────────────────────────────────────────────
    'frob_gradient',

    # ─────────────────────────────────────────────────────────────
    # A.11 — volume_proxy
    # Couche     : A
    # Grammaire  : signal
    # Question   : Q2
    # Ancrage    : A2 — contraction/expansion sans différentiation
    # Mesure     : ||D||_F · ||D||_∞ / (||D||_1 + eps)
    # Rôle       : proxy géométrique de volume informationnel.
    #              Donne aux gammas non-diff une mesure de compression
    #              absente avant SC1. Complémentaire à lyap_empirical
    #              (qui mesure la variation, pas l'occupation).
    # Agrégats   : mean, std, final, delta
    # NaN        : aucun
    # Caveat     : pas un vrai volume mathématique, proxy grossier mais
    #              canonique (invariant aux permutations de composantes).
    # NOUVEAU SC1
    # ─────────────────────────────────────────────────────────────
    'volume_proxy',
]

# ── Couche B — 15 features, calculées sous lax.cond aux points du masque ──
# Grammaire : virages (timelines tronquées, longueur K_i variable par run).
# Calculés en O(n³) (SVD) ou coût JVP significatif.
# Le lax.cond garantit que les runs à masque parcimonieux paient peu et que
# les runs pathologiques (masque vide) ne paient rien.
# Ordre des features reflète la factorisation SVD de SC4 :
#   F1 × 6 → une SVD principale (mode 0)
#   F2 × 2 → partage la SVD de F1
#   F3 × 3 → SVD mode 0 partagée + SVD mode 1 si rank ≥ 3
#   F4 × 4 → JVP, indépendant de la SVD
LAYER_B_KEYS: List[str] = [
    # ─────────────────────────────────────────────────────────────
    # B.1 — f1_effective_rank
    # Couche     : B (timeline tronquée, virages)
    # Grammaire  : virages
    # Question   : Q1
    # Ancrage    : A1 — combien de dimensions spectrales actives
    # Mesure     : exp(Shannon(p_i)) avec p_i = σ_i² / Σσ_j²
    # Rôle       : rang effectif entropique. Nourrit phasic_rank.
    # Agrégats   : mean, final, delta
    # NaN        : aucun (universel, SVD toujours possible)
    # ─────────────────────────────────────────────────────────────
    'f1_effective_rank',

    # ─────────────────────────────────────────────────────────────
    # B.2 — f1_spectral_gap
    # Couche     : B
    # Grammaire  : virages
    # Question   : Q1
    # Ancrage    : A1 — séparation du mode dominant
    # Mesure     : (σ_0 - σ_1) / (σ_0 + eps)
    # Rôle       : isolement du premier mode. Grand → attracteur bas rang.
    # Agrégats   : mean, final
    # NaN        : aucun
    # ─────────────────────────────────────────────────────────────
    'f1_spectral_gap',

    # ─────────────────────────────────────────────────────────────
    # B.3 — f1_nuclear_frobenius_ratio
    # Couche     : B
    # Grammaire  : virages
    # Question   : Q1
    # Ancrage    : A1 — compression spectrale
    # Mesure     : Σσ_i / sqrt(Σσ_i²)
    # Rôle       : concentration spectrale complémentaire à l'entropie.
    #              Décroît de sqrt(n) (spectre plat) vers 1 (un seul mode).
    # Agrégats   : mean, final
    # NaN        : aucun
    # ─────────────────────────────────────────────────────────────
    'f1_nuclear_frobenius_ratio',

    # ─────────────────────────────────────────────────────────────
    # B.4 — f1_sv_decay_rate
    # Couche     : B
    # Grammaire  : virages
    # Question   : Q1
    # Ancrage    : A1 — loi de décroissance des modes
    # Mesure     : pente de régression linéaire de log(σ_i) vs i
    # Rôle       : signe du régime spectral. Pente forte → peu de modes.
    # Agrégats   : mean, final
    # NaN        : aucun
    # ─────────────────────────────────────────────────────────────
    'f1_sv_decay_rate',

    # ─────────────────────────────────────────────────────────────
    # B.5 — f1_rank1_residual
    # Couche     : B
    # Grammaire  : virages
    # Question   : Q1
    # Ancrage    : A1 — dominance du premier mode
    # Mesure     : sqrt(Σ_{i≥1} σ_i²) / (||D||_F + eps)
    # Rôle       : énergie résiduelle hors mode 1. Proche de 0 → quasi rank-1.
    # Agrégats   : mean, final
    # NaN        : aucun
    # ─────────────────────────────────────────────────────────────
    'f1_rank1_residual',

    # ─────────────────────────────────────────────────────────────
    # B.6 — f1_condition_number
    # Couche     : B
    # Grammaire  : virages
    # Question   : Q1
    # Ancrage    : A1 — dispersion spectrale
    # Mesure     : σ_0 / (σ_{-1} + eps)
    # Rôle       : étirement du spectre. Grand → modes très dispersés.
    # Agrégats   : mean, delta
    # NaN        : aucun
    # ─────────────────────────────────────────────────────────────
    'f1_condition_number',

    # ─────────────────────────────────────────────────────────────
    # B.7 — f2_von_neumann_entropy
    # Couche     : B
    # Grammaire  : virages
    # Question   : Q1 + Q4 (holographie Ryu-Takayanagi)
    # Ancrage    : A1 + A3 — proxy direct aire holographique
    # Mesure     : -Σ p_i log(p_i + eps) avec p_i = σ_i² / Σσ_j²
    # Rôle       : feature la plus fondamentale du pipeline pour A3.
    #              Son delta (final - initial) = production nette d'entropie
    #              holographique. Nourrit phasic_svn.
    # Agrégats   : mean, std, final, delta
    # NaN        : aucun
    # ─────────────────────────────────────────────────────────────
    'f2_von_neumann_entropy',

    # ─────────────────────────────────────────────────────────────
    # B.8 — f2_renyi2_entropy
    # Couche     : B
    # Grammaire  : virages
    # Question   : Q1
    # Ancrage    : A1 — entropie sensible aux modes dominants
    # Mesure     : -log(Σ p_i² + eps)
    # Rôle       : complément de VN. Plus sensible aux modes dominants.
    #              Différence (VN - Rényi-2) caractérise la queue du spectre.
    # Agrégats   : mean, final, delta
    # NaN        : aucun
    # ─────────────────────────────────────────────────────────────
    'f2_renyi2_entropy',

    # ─────────────────────────────────────────────────────────────
    # B.9 — f3_entanglement_entropy_mode0
    # Couche     : B
    # Grammaire  : virages
    # Question   : Q1 + Q4
    # Ancrage    : A1 + géométrie relationnelle Rovelli
    # Mesure     : VN de D replié en matrice selon mode 0
    # Rôle       : couplage du mode 0 avec les autres modes.
    # Agrégats   : mean, final
    # NaN        : aucun (la SVD mode 0 est toujours calculable)
    # ─────────────────────────────────────────────────────────────
    'f3_entanglement_entropy_mode0',

    # ─────────────────────────────────────────────────────────────
    # B.10 — f3_entanglement_entropy_mode1
    # Couche     : B
    # Grammaire  : virages
    # Question   : Q1
    # Ancrage    : A1
    # Mesure     : VN de D replié selon mode 1
    # Rôle       : symétrique de mode0 sur l'autre mode. Comparaison
    #              mode0 vs mode1 → asymétrie inter-modes.
    # Agrégats   : mean, final
    # NaN        : structurel si rank_eff == 2 (un seul mode non trivial)
    # ─────────────────────────────────────────────────────────────
    'f3_entanglement_entropy_mode1',

    # ─────────────────────────────────────────────────────────────
    # B.11 — f3_inter_mode_sv_var
    # Couche     : B
    # Grammaire  : virages
    # Question   : Q1
    # Ancrage    : A1
    # Mesure     : var([σ_0^{mode0}, σ_0^{mode1}])
    # Rôle       : disparité entre modes. 0 → équilibre parfait.
    # Agrégats   : mean, final
    # NaN        : structurel si rank_eff == 2
    # ─────────────────────────────────────────────────────────────
    'f3_inter_mode_sv_var',

    # ─────────────────────────────────────────────────────────────
    # B.12 — f4_trace_J
    # Couche     : B
    # Grammaire  : virages
    # Question   : Q2
    # Ancrage    : A2 — divergence locale de Γ
    # Mesure     : estimation Hutchinson de trace(J) via v^T · (J·v)
    # Rôle       : > 0 expansion, < 0 contraction, ≈ 0 hamiltonien.
    # Agrégats   : mean, std, final
    # NaN        : structurel si is_differentiable == False
    # ─────────────────────────────────────────────────────────────
    'f4_trace_J',

    # ─────────────────────────────────────────────────────────────
    # B.13 — f4_jvp_norm
    # Couche     : B
    # Grammaire  : virages
    # Question   : Q2
    # Ancrage    : A2 — amplitude de l'action locale de Γ
    # Mesure     : ||J·v||
    # Rôle       : échelle locale de Γ. Combiné avec ||v|| → ratio
    #              de contraction/dilatation.
    # Agrégats   : mean, final
    # NaN        : structurel si is_differentiable == False
    # ─────────────────────────────────────────────────────────────
    'f4_jvp_norm',

    # ─────────────────────────────────────────────────────────────
    # B.14 — f4_jacobian_asymmetry
    # Couche     : B
    # Grammaire  : virages
    # Question   : Q2 + Q4
    # Ancrage    : A2 + lien A3 — réversibilité locale / flèche du temps
    # Mesure     : ||J·v - J^T·v|| / (||J·v|| + eps)
    # Rôle       : 0 → J symétrique → Γ localement réversible.
    #              Grand → production d'entropie, flèche du temps émergente.
    #              Asymétrie systématique > 0 → candidat A3 observé.
    # Agrégats   : mean, final, delta
    # NaN        : structurel si is_differentiable == False
    # ─────────────────────────────────────────────────────────────
    'f4_jacobian_asymmetry',

    # ─────────────────────────────────────────────────────────────
    # B.15 — f4_local_lyapunov
    # Couche     : B
    # Grammaire  : virages
    # Question   : Q2
    # Ancrage    : A2 — Lyapunov local linéarisé
    # Mesure     : log(||J·v|| + eps)
    # Rôle       : comparable à lyap_empirical (couche A). Écart
    #              systématique entre les deux → non-linéarité forte.
    # Agrégats   : mean, std, final
    # NaN        : structurel si is_differentiable == False
    # ─────────────────────────────────────────────────────────────
    'f4_local_lyapunov',
]


# =========================================================================
# S3 — MAPPING D'AGRÉGATION C1
# =========================================================================
# Pour chaque feature couche A et couche B, liste des agrégats à calculer.
# Couche A : agrégats sur timeline uniforme [0, t_effective).
# Couche B : agrégats sur les K_i points du masque (sans interpolation).
#
# Les noms parquet sont dérivés mécaniquement : "{clé}_{agrégat}"
# par SC6 lors de la construction de col_data.
#
# Vocabulaire clos : mean, std, final, delta, total.
#   mean  — valeur moyenne (niveau typique)
#   std   — écart-type (variabilité, régime turbulent vs plateau)
#   final — dernière valeur valide (convergence, attracteur)
#   delta — final - initial (changement net, production d'entropie)
#   total — somme (uniquement pour les quantités cumulables)
#
# La justification feature-par-feature est dans l'Algo SC1.

AGG_MAP: Dict[str, List[str]] = {
    # ── Couche A (11 observables) ──
    'frob':              ['mean', 'final', 'delta'],
    'delta_D':           ['mean', 'std', 'total'],
    'cos_dissim':        ['mean', 'std', 'final'],
    'is_finite':         ['final'],
    'shannon_comp':      ['mean', 'std', 'final', 'delta'],
    'ipr':               ['mean', 'std', 'final'],
    'mode_asymmetry_o2': ['mean', 'std', 'final', 'delta'],
    'bregman_cost':      ['mean', 'total'],
    'lyap_empirical':    ['mean', 'std', 'final'],
    'frob_gradient':     ['mean', 'final'],
    'volume_proxy':      ['mean', 'std', 'final', 'delta'],

    # ── Couche B (15 features) ──
    'f1_effective_rank':             ['mean', 'final', 'delta'],
    'f1_spectral_gap':               ['mean', 'final'],
    'f1_nuclear_frobenius_ratio':    ['mean', 'final'],
    'f1_sv_decay_rate':              ['mean', 'final'],
    'f1_rank1_residual':             ['mean', 'final'],
    'f1_condition_number':           ['mean', 'delta'],
    'f2_von_neumann_entropy':        ['mean', 'std', 'final', 'delta'],
    'f2_renyi2_entropy':             ['mean', 'final', 'delta'],
    'f3_entanglement_entropy_mode0': ['mean', 'final'],
    'f3_entanglement_entropy_mode1': ['mean', 'final'],
    'f3_inter_mode_sv_var':          ['mean', 'final'],
    'f4_trace_J':                    ['mean', 'std', 'final'],
    'f4_jvp_norm':                   ['mean', 'final'],
    'f4_jacobian_asymmetry':         ['mean', 'final', 'delta'],
    'f4_local_lyapunov':             ['mean', 'std', 'final'],
}

# Noms scalaires C1 dérivés mécaniquement depuis AGG_MAP.
# La boucle garantit la cohérence : si AGG_MAP change, cette liste suit.
# Ordre : LAYER_A_KEYS d'abord, LAYER_B_KEYS ensuite, chacun dans l'ordre
# de ses agrégats.
AGG_FEATURE_NAMES: List[str] = [
    f'{key}_{agg}'
    for key in LAYER_A_KEYS + LAYER_B_KEYS
    for agg in AGG_MAP[key]
]
# Total attendu : 32 (couche A) + 37 (couche B) = 69 scalaires.


# =========================================================================
# S4 — FEATURES POST-SCAN C2 (GRAMMAIRE SIGNAL)
# =========================================================================
# Calculées dans postprocess CPU sur les timelines couche A (uniformes).
# Jamais appliquées aux timelines couche B (virages) — règle de cohérence
# grammaticale (charter §1.5, P10).

# ── DMD sur observables couche A (4) ──
# Entrée : vecteur des 11 observables à chaque step → trajectoire (T, 11).
# Méthode : DMD streaming RLS.
# Ancrage : A2 — universalité de Γ (charter §1.4).
# Sémantique SC1 : le DMD opère sur les observables canoniques O(n²),
# PAS sur les sigmas_buf comme dans le pipeline antérieur.
DMD_FEATURE_NAMES: List[str] = [
    'f7_dmd_spectral_radius',       # rayon spectral dominant
    'f7_dmd_n_complex_pairs',       # paires conjuguées complexes (oscillation)
    'f7_dmd_spectral_entropy',      # entropie de Shannon du spectre DMD
    'f7_dmd_decay_rate',            # pente de décroissance des modes
]

# ── F6 Transfer Entropy sur 4 paires d'observables (5) ──
# Entrée : paires de timelines couche A, uniformes.
# Ancrage : A2 — causalité Wolfram (charter §1.4).
# Noms explicites {source}_to_{cible} pour auto-documentation.
# Paires validées DOC1 §5.3 C2.2 :
#   frob → shannon_comp           le volume cause-t-il la dispersion ?
#   lyap_empirical → frob         l'expansion cause-t-elle le volume ?
#   cos_dissim → mode_asym_o2     la rotation cause-t-elle l'asymétrie ?
#   delta_D → bregman_cost        la variation cause-t-elle le transport ?
F6_FEATURE_NAMES: List[str] = [
    'f6_te_frob_to_shannon_comp',
    'f6_te_lyap_empirical_to_frob',
    'f6_te_cos_dissim_to_mode_asym_o2',
    'f6_te_delta_D_to_bregman_cost',
    'f6_causal_asymmetry_index',    # agrégat global des 4 précédentes
]

# ── Autocorrélations premier minimum (3) ──
# Entrée : timelines couche A (frob, shannon_comp, mode_asymmetry_o2).
# Ancrage : Q3 — détection de périodicité et régimes stables.
AUTOCORR_FEATURE_NAMES: List[str] = [
    'ps_first_min_ac_frob',
    'ps_first_min_ac_shannon_comp',     # renommé depuis ps_first_min_ac_von_neumann
    'ps_first_min_ac_mode_asym_o2',     # renommé depuis ps_first_min_ac_mode_asymmetry
]

# ── PNN40 (2) ──
# Entrée : timelines couche A (shannon_comp, mode_asymmetry_o2).
# Ancrage : Q3 — variabilité point-à-point.
PNN_FEATURE_NAMES: List[str] = [
    'ps_pnn40_shannon_comp',            # renommé depuis ps_pnn40_von_neumann
    'ps_pnn40_mode_asym_o2',            # renommé depuis ps_pnn40_mode_asymmetry
]

# ── Temporal features (8) ──
# Entrée : timelines couche A (frob, shannon_comp).
# Méthode : IQR, plateau_frac, cusum_delta, changepoint_t_norm.
# Ancrage : Q3 — stationnarité et changements de phase.
TEMPORAL_FEATURE_NAMES: List[str] = [
    'temporal_frob_iqr',
    'temporal_frob_plateau_frac',
    'temporal_frob_cusum_delta',
    'temporal_frob_changepoint_t_norm',
    'temporal_shannon_comp_iqr',                # renommé depuis temporal_svn_iqr
    'temporal_shannon_comp_plateau_frac',        # renommé depuis temporal_svn_plateau_frac
    'temporal_shannon_comp_cusum_delta',         # renommé depuis temporal_svn_cusum_delta
    'temporal_shannon_comp_changepoint_t_norm',  # renommé depuis temporal_svn_changepoint_t_norm
]

# ── Stationnarité (11) — une par observable couche A ──
# Entrée : chaque timeline couche A.
# Mesure : |mean(last 20%) - mean(first 20%)| / (std(all) + eps)
# Ancrage : Q3 — le run atteint-il un régime stationnaire ?
# NOUVEAU SC1 (DOC1 §5.3 C2.6, Q-sci-5)
STATIONARITY_FEATURE_NAMES: List[str] = [
    'stat_delta_frob',
    'stat_delta_delta_D',
    'stat_delta_cos_dissim',
    'stat_delta_is_finite',
    'stat_delta_shannon_comp',
    'stat_delta_ipr',
    'stat_delta_mode_asymmetry_o2',
    'stat_delta_bregman_cost',
    'stat_delta_lyap_empirical',
    'stat_delta_frob_gradient',         # NaN structurel rank_eff == 2
    'stat_delta_volume_proxy',
]

# ── Entropy production rate (1) ──
# Entrée : timeline couche A shannon_comp.
# Mesure : pente de régression linéaire (taux moyen de croissance).
# Ancrage : A3 — production d'entropie irréductible.
# ATTENTION : nom conservé pour continuité mais la sémantique change.
# L'ancien f2_entropy_production_rate opérait sur f2_von_neumann_entropy
# (spectral). Le nouveau opère sur shannon_comp (proxy O(n²) en grammaire
# signal). Ce n'est PAS la même quantité physique — c'est un proxy.
ENTROPY_PRODUCTION_FEATURE_NAMES: List[str] = [
    'f2_entropy_production_rate',
]


# =========================================================================
# S5 — FEATURES POST-SCAN C3 (GRAMMAIRE VIRAGES)
# =========================================================================
# Méthodes ordinales calculées sur les K_i points masqués.
# Pas de supposée d'uniformité temporelle — charter §1.5 et §1.7.

# ── Phasic features (9) ──
# Méthode : n_reversals, max_monotone_frac, range_ratio sur séquence
# ordonnée des K valeurs aux points du masque.
# Ancrage : Q3 — forme du parcours aux virages.
#
# Sources :
#   phasic_svn_*           → timeline_f2_von_neumann_entropy (couche B)
#   phasic_rank_*          → timeline_f1_effective_rank (couche B)
#   phasic_mode_asym_o2_*  → timeline_mode_asymmetry_o2 (couche A)
#                            SOUS-ÉCHANTILLONNÉ aux points du masque.
#                            Option 2 DOC1 §5.4 — projection grammaire
#                            signal → grammaire virages, licite (charter §1.7).
#
# NaN runtime : K_i < 2 rend les mesures ordinales non définies.
PHASIC_FEATURE_NAMES: List[str] = [
    'phasic_svn_n_reversals',
    'phasic_svn_max_monotone_frac',
    'phasic_svn_range_ratio',
    'phasic_rank_n_reversals',
    'phasic_rank_max_monotone_frac',
    'phasic_rank_range_ratio',
    'phasic_mode_asym_o2_n_reversals',
    'phasic_mode_asym_o2_max_monotone_frac',
    'phasic_mode_asym_o2_range_ratio',
]


# =========================================================================
# S6 — METADATA CLASSIFY / MASK / META
# =========================================================================
# Features produites par classify_and_mask entre P1 et P2, plus metadata
# du run. Inchangées en structure par rapport au registre antérieur.

# ── P1 caractérisation (5) — activité directionnelle du run ──
P1_FEATURE_NAMES: List[str] = [
    'p1_cos_dissim_mean',       # activité directionnelle moyenne
    'p1_cos_dissim_std',        # variabilité de l'activité
    'p1_cos_dissim_cv',         # coefficient de variation (std/mean)
    'p1_estimated_period',      # période par autocorrélation (NaN si apériodique)
    'p1_n_zero_crossings',      # zero-crossings de diff(cos_dissim)
]

# ── Mask géométrie (7) — forme et couverture du masque ──
MASK_FEATURE_NAMES: List[str] = [
    'mask_n_transitions',       # nombre de segments contigus dans le mask
    'mask_t_first_norm',        # position temporelle normalisée du 1er virage
    'mask_t_last_norm',         # position du dernier virage
    'mask_mean_amplitude',      # cos_dissim moyen aux itérations actives du mask
    'mask_max_amplitude',       # cos_dissim max aux itérations actives du mask
    'mask_mean_spacing_norm',   # espacement moyen entre virages / max_it
    'mask_coverage_frac',       # fraction de la timeline dans les zones actives
]

# ── Meta (3) — métadonnées numériques du run ──
META_FEATURE_NAMES: List[str] = [
    'meta_n_svd',               # nombre effectif de points couche B (= K_i)
    'meta_turbulence',          # ratio k/max_it × TURBULENCE_CONSTANT
    'meta_t_effective',         # durée effective (max_it si OK, t_cutoff sinon)
]

# ─────────────────────────────────────────────────────────────────────────
# RETRAIT v15/SC1 — health_has_inf, health_is_collapsed
# ─────────────────────────────────────────────────────────────────────────
# Retirées car redondantes avec :
#   • run_status ∈ {OK, OK_TRUNCATED, EXPLOSION, COLLAPSED}
#     (METADATA_COLUMNS, produit par classify)
#   • is_finite en timeline couche A (passage 1→0 = explosion)
#   • t_effective (durée effective du run)
# Héritage de versions antérieures aux statuts implémentés proprement.
# Migration downstream : lire run_status ou timeline_is_finite[-1].
# ─────────────────────────────────────────────────────────────────────────


# =========================================================================
# S7 — FEATURE_NAMES — SOURCE DE VÉRITÉ PARQUET (SCALAIRES FLOAT32)
# =========================================================================
# Toute colonne float32 du parquet est listée ici et nulle part ailleurs.
# Construite par concaténation des listes de sections précédentes.

FEATURE_NAMES: List[str] = [
    *AGG_FEATURE_NAMES,                     #  69 — C1 agrégats (A + B)
    *DMD_FEATURE_NAMES,                     #   4 — C2 DMD
    *F6_FEATURE_NAMES,                      #   5 — C2 F6
    *AUTOCORR_FEATURE_NAMES,                #   3 — C2 autocorr
    *PNN_FEATURE_NAMES,                     #   2 — C2 PNN
    *TEMPORAL_FEATURE_NAMES,                #   8 — C2 temporal
    *STATIONARITY_FEATURE_NAMES,            #  11 — C2 stationnarité
    *ENTROPY_PRODUCTION_FEATURE_NAMES,      #   1 — C2 entropy production
    *PHASIC_FEATURE_NAMES,                  #   9 — C3 phasic
    *P1_FEATURE_NAMES,                      #   5 — metadata P1
    *MASK_FEATURE_NAMES,                    #   7 — metadata mask
    *META_FEATURE_NAMES,                    #   3 — metadata meta
]
# Total attendu : 69 + 34 + 9 + 15 = 127 scalaires float32.


# =========================================================================
# S8 — COLONNES NON-SCALAIRES (metadata, classifications)
# =========================================================================
# Colonnes parquet qui ne sont pas des float32.
# Source de vérité pour build_schema.

METADATA_COLUMNS: Dict[str, str] = {
    'phase':            'string',
    'gamma_id':         'string',
    'encoding_id':      'string',
    'modifier_id':      'string',
    'gamma_params':     'string',    # JSON sérialisé — params hétérogènes entre atomics
    'encoding_params':  'string',    # JSON sérialisé
    'modifier_params':  'string',    # JSON sérialisé
    'n_dof':            'int32',
    'rank_eff':         'int32',
    'max_it':           'int32',
    'seed_CI':          'int64',
    'seed_run':         'int64',
    'run_status':       'string',       # {OK, OK_TRUNCATED, EXPLOSION, COLLAPSED}
    'p1_regime_class':  'string',       # {FLAT, OSCILLATING, TRANSITIONAL, EXPLOSIVE, MIXED}
}


# =========================================================================
# S9 — TIMELINES PARQUET (LIST<FLOAT32>)
# =========================================================================
# Trois familles de listes variables par run, sémantiques distinctes.
# SC6 lit ces listes pour savoir quel format d'écriture appliquer.

# Longueur : t_effective (uniforme, 1 valeur par step P2).
# Grammaire : signal — nourrit le post-process C2.
TIMELINE_COLUMNS_A: List[str] = [
    f'timeline_{k}' for k in LAYER_A_KEYS
]  # 11 entrées

# Longueur : K_i (variable, points du masque).
# Grammaire : virages — nourrit les agrégats C1 couche B et les phasic C3.
TIMELINE_COLUMNS_B: List[str] = [
    f'timeline_{k}' for k in LAYER_B_KEYS
]  # 15 entrées

# Longueur : K_i. Partagé par toutes les colonnes B d'un run.
# Contient les indices temporels (entiers) des points du masque.
MASK_INDICES_COLUMN: str = 'mask_t_indices'


# =========================================================================
# S10 — NaN STRUCTURELS ET RUNTIME
# =========================================================================
# Deux dicts séparés (Charter P7 — clustering stratifié) :
#
# STRUCTURAL : connus à la compilation depuis les métadonnées (rank_eff,
#              is_differentiable). Le clustering les masque a priori —
#              ce ne sont PAS des observations.
#
# RUNTIME    : dépendants des données du run (K_i < 2, n_transitions == 0).
#              Ce SONT des observations — un NaN runtime porte de
#              l'information (le run n'a pas produit de transition =
#              signal, pas artefact).

FEATURES_STRUCTURAL_NAN: Dict[str, str] = {
    # ── rank_eff == 2 — features dépendantes de modes multiples ──
    # Agrégats couche A
    'frob_gradient_mean':                       'rank_eff == 2',
    'frob_gradient_final':                      'rank_eff == 2',
    # Agrégats couche B
    'f3_entanglement_entropy_mode1_mean':        'rank_eff == 2',
    'f3_entanglement_entropy_mode1_final':       'rank_eff == 2',
    'f3_inter_mode_sv_var_mean':                 'rank_eff == 2',
    'f3_inter_mode_sv_var_final':                'rank_eff == 2',
    # Post-scan C2
    'stat_delta_frob_gradient':                  'rank_eff == 2',
    # Timelines (entrée entière NaN si rank_eff == 2)
    'timeline_frob_gradient':                    'rank_eff == 2',
    'timeline_f3_entanglement_entropy_mode1':    'rank_eff == 2',
    'timeline_f3_inter_mode_sv_var':             'rank_eff == 2',

    # ── is_differentiable == False — JVP impossible ──
    # Agrégats couche B
    'f4_trace_J_mean':                           'is_differentiable == False',
    'f4_trace_J_std':                            'is_differentiable == False',
    'f4_trace_J_final':                          'is_differentiable == False',
    'f4_jvp_norm_mean':                          'is_differentiable == False',
    'f4_jvp_norm_final':                         'is_differentiable == False',
    'f4_jacobian_asymmetry_mean':                'is_differentiable == False',
    'f4_jacobian_asymmetry_final':               'is_differentiable == False',
    'f4_jacobian_asymmetry_delta':               'is_differentiable == False',
    'f4_local_lyapunov_mean':                    'is_differentiable == False',
    'f4_local_lyapunov_std':                     'is_differentiable == False',
    'f4_local_lyapunov_final':                   'is_differentiable == False',
    # Timelines (entrée entière NaN si non-différentiable)
    'timeline_f4_trace_J':                       'is_differentiable == False',
    'timeline_f4_jvp_norm':                      'is_differentiable == False',
    'timeline_f4_jacobian_asymmetry':            'is_differentiable == False',
    'timeline_f4_local_lyapunov':                'is_differentiable == False',
}

FEATURES_RUNTIME_NAN: Dict[str, str] = {
    # P1 — autocorrélation sans pic détecté
    'p1_estimated_period':                       'non_periodic',
    # Mask — aucune transition détectée
    'mask_t_first_norm':                         'n_transitions == 0',
    'mask_t_last_norm':                          'n_transitions == 0',
    'mask_mean_spacing_norm':                    'n_transitions < 2',
    # Phasic — K_i < 2 rend les mesures ordinales non définies
    'phasic_svn_n_reversals':                    'K_i < 2',
    'phasic_svn_max_monotone_frac':              'K_i < 2',
    'phasic_svn_range_ratio':                    'K_i < 2',
    'phasic_rank_n_reversals':                   'K_i < 2',
    'phasic_rank_max_monotone_frac':             'K_i < 2',
    'phasic_rank_range_ratio':                   'K_i < 2',
    'phasic_mode_asym_o2_n_reversals':           'K_i < 2',
    'phasic_mode_asym_o2_max_monotone_frac':     'K_i < 2',
    'phasic_mode_asym_o2_range_ratio':           'K_i < 2',
}
