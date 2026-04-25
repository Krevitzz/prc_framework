"""
Constantes machine du pipeline PRC v15.

Constantes qui affectent les performances mais JAMAIS les résultats scientifiques.
Deux machines avec des constantes machine différentes produisent les mêmes données.
Les constantes scientifiques sont dans features_registry.py.

@ROLE    Constantes machine — workers, queues, batch sizes, VRAM
@LAYER   configs

@EXPORTS
  VERBOSE                → bool   | mode verbeux (env var)
  RECORD_TIMING          → bool   | enregistrer les temps d'exécution dans Parquet
  BATCH_SIZE_MAP         → Dict   | (rank, n_dof) → (p1_batch, p2_batch)
  MAX_N_DOF_RANK3        → int    | seuil n_dof pour rank=3
  MAX_CONCURRENT_GROUPS  → int    | plafond de processus simultanés (RAM)
  FLUSH_BATCH_SIZE       → int    | lot d'écriture parquet
  DEFAULT_P1_BATCH, DEFAULT_P2_BATCH → int | fallback si shape absente du map
  MIN_AVAILABLE_RAM_GB   → float  | RAM minimale avant nouveau processus
  VRAM_USAGE_THRESHOLD   → float  | seuil VRAM (0.8 = 80%)
  DEFAULT_TOTAL_VRAM_GB  → float  | VRAM totale de la carte
  XLA_FIXED_OVERHEAD_GB  → float  | coût fixe compilation + runtime par process
  XLA_SAFETY_MARGIN      → float  | multiplicateur intermédiaires XLA
  VRAM_ACQUIRE_TIMEOUT_S → float  | timeout acquire budget (détection deadlock)
  EXPECTED_MASK_DENSITY  → float  | densité masque estimée pour coût JVP
  JVP_INTERMEDIATES_FACTOR → int  | copies d'état simultanées pendant JVP
"""

import os


# =========================================================================
# VERBOSE
# =========================================================================

VERBOSE: bool = os.environ.get('PIPELINE_VERBOSE', '1') == '1'


# =========================================================================
# RECORD TIMING
# =========================================================================

# Si True, ajoute les colonnes timing_p1_s, timing_p2_s, timing_post_s dans Parquet.
# Peut être activé via variable d'environnement pour ne pas surcharger les résultats.
RECORD_TIMING: bool = os.environ.get('PRC_RECORD_TIMING', '0') == '1'


# =========================================================================
# PRÉCISION DES CALCULS (bfloat16 pour économiser VRAM)
# =========================================================================
# Activer bfloat16 (True) ou float32 (False)
# bfloat16 conserve la même plage que float32 mais avec une précision réduite.
# À tester sur un petit échantillon avant de généraliser.
USE_BFLOAT16: bool = True  # passer à False pour float32


# =========================================================================
# BATCH SIZE CALIBRATION
# =========================================================================

# Calibration post-chantier 1 : le P1 est minimal (screening seul),
# le découpage VRAM-aware est géré par split_job dans hub.py.
# Ces valeurs sont des plafonds permissifs — le split_job découpe
# en dessous si l'estimation VRAM l'exige.
# p2_batch conservé pour référence mais non utilisé par le pipeline.
BATCH_SIZE_MAP: dict = {
    (2, 10):   (1024, 1024),
    (2, 50):   (1024, 512),
    (2, 100):  (512, 256),
    (2, 200):  (256, 128),
    (2, 500):  (128, 64),
    (2, 1000): (64, 32),
    (3, 10):   (512, 256),
    (3, 50):   (256, 128),
    (3, 100):  (16, 8),    # réduit de (128,64) → (64,32)
    (3, 200):  (8, 4),     # réduit de (64,32) → (16,8) → B final ~8
}

DEFAULT_P1_BATCH: int = 512
DEFAULT_P2_BATCH: int = 256


# =========================================================================
# SEMAPHORE GPU CALIBRATION
# =========================================================================

# SEMAPHORE_MAP et DEFAULT_SEMAPHORE supprimés v16.
# Remplacés par VramBudget dans hub.py — le budget VRAM partagé
# gère la concurrence GPU dynamiquement selon la taille des jobs.


# =========================================================================
# GESTION DES RESSOURCES MÉMOIRE
# =========================================================================

MIN_AVAILABLE_RAM_GB: float = 2.0

# VRAM totale de la carte GPU.
# GTX 1080 Ti = 11 Go. À ajuster selon la machine.
DEFAULT_TOTAL_VRAM_GB: float = 10.0

# Seuil d'utilisation VRAM — budget = TOTAL × THRESHOLD.
# 0.80 → 8.8 Go de budget sur une 1080 Ti. Les 2.2 Go restants
# sont réservés au système, au display, et aux allocations XLA imprévues.
VRAM_USAGE_THRESHOLD: float = 0.90

# Coût fixe par processus : compilation XLA des 4 briques + runtime JAX.
# Mesuré empiriquement : ~500 Mo compilation + ~300 Mo runtime.
# Provenance : observation bourrin_scaled sur GTX 1080 Ti.
# Changer : plus grand → moins de concurrence, plus conservateur.
XLA_FIXED_OVERHEAD_GB: float = 0.9

# Multiplicateur de sécurité pour les intermédiaires XLA non modélisés.
# Les buffers de gradient (JVP), les copies temporaires du scan, les
# allocations de fusion XLA ne sont pas dans le modèle analytique.
# 1.5 = 50% de marge au-dessus du modèle.
# Provenance : ratio observé pic_réel / modèle_analytique.
# Changer : plus grand → plus conservateur, moins de concurrence.
XLA_SAFETY_MARGIN: float = 0.5

# Surcharge spécifique pour rank=3 (plus gourmand)
RANK3_EXTRA_MARGIN = 1.4 # multiplicatif en plus

# Densité masque attendue — fraction des steps où couche B est calculée.
# Constante machine : n'affecte pas les résultats scientifiques, seulement
# la précision de l'estimation VRAM et le découpage des jobs.
# Provenance : heuristique 25% (75e percentile estimé K_i / t_effective
# sur test_v9_baseline). À calibrer par EXP-C1-5.
# Changer : plus grand → estimation plus conservatrice (plus de splits).
EXPECTED_MASK_DENSITY: float = 0.25

# Facteur d'intermédiaires JVP — nombre de copies d'état allouées
# simultanément pendant le calcul du Jacobien couche B (tangent vector,
# cotangent vector, résultat JVP).
# Constante machine : calibrable empiriquement.
# Provenance : estimation théorique (3 tenseurs ~ state_size au pic).
# Changer : plus grand → estimation JVP plus conservatrice.
JVP_INTERMEDIATES_FACTOR: int = 3

# Timeout d'acquisition du budget VRAM (secondes).
# Si un processus attend plus longtemps → il meurt proprement et
# le hub logge l'échec. Filet de sécurité contre les deadlocks.
# Ne devrait jamais se déclencher en fonctionnement normal (le hub
# pré-découpe les jobs trop gros avant lancement).
VRAM_ACQUIRE_TIMEOUT_S: float = 3600.0


# =========================================================================
# SHAPE LIMITS
# =========================================================================

MAX_N_DOF_RANK3: int = 250

# =========================================================================
# BLOCK PROCESSING P2
# =========================================================================

# Taille des blocs pour le traitement par blocs de la passe 2.
# Permet de réduire le nombre d'appels JIT et les transferts CPU/GPU.
# À ajuster selon la VRAM disponible. Une valeur de 20 est un bon compromis.
BLOCK_SIZE: int = 500

# =========================================================================
# BACK-PRESSURE ET ORCHESTRATION
# =========================================================================

P2_QUEUE_PAUSE_THRESHOLD: int = 6
MAX_CONCURRENT_GROUPS: int = 9

# Nombre de processus autorisés à exécuter simultanément sur le GPU (P1+P2).
# Séparé du budget VRAM : la VRAM contrôle combien de processus peuvent
# PRÉPARER leur travail, le sémaphore compute contrôle combien peuvent
# EXÉCUTER sur le GPU en même temps.
# 1 = exclusivité GPU totale (zéro contention, performances P2 optimales).
# 2-3 = parallélisme limité (utile si le GPU n'est pas saturé par un seul job).
# Constante machine : n'affecte pas les résultats scientifiques.
# Provenance : benchmark baseline — contention GPU cause 3-12× d'overhead P2.
GPU_COMPUTE_SLOTS: int = 1


# =========================================================================
# FLUSH PARQUET
# =========================================================================

FLUSH_BATCH_SIZE: int = 10000
