"""
tests/TM-GAM-001.py

Toy Model pour tester HYP-GAM-001 (saturation pure).

STRUCTURE:
1. Préparation D (composition via prepare_state)
2. Définition Γ
3. Exécution (kernel aveugle)
4. Analyse immédiate
5. Génération LOG
"""

import numpy as np
import sys
from pathlib import Path

# Ajoute projet au path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ============================================================================
# IMPORTS
# ============================================================================

# Core (2 fonctions aveugles)
from core.state_preparation import prepare_state
from core.kernel import run_kernel

# Encodage D^(base)
from D_encodings.rank2_symmetric import create_identity

# Modificateurs D^(bruit)
from modifiers.noise import add_gaussian_noise

# Opérateur Γ
from operators.gamma_hyp_001 import PureSaturationGamma


# ============================================================================
# CONFIGURATION
# ============================================================================

N_DOF = 50
EPSILON_NOISE = 0.05
BETA_GAMMA = 2.0
MAX_ITERATIONS = 2000
SEED = 42


# ============================================================================
# 1. PRÉPARATION DE D
# ============================================================================

print("Préparation de D...")

# D^(base) - État minimal pour ce test
D_base = create_identity(N_DOF)

# Modificateurs à appliquer
modifiers = [
    add_gaussian_noise(sigma=EPSILON_NOISE, seed=SEED)  # D^(bruit)
]

# Composition aveugle dans le core
D_initial = prepare_state(D_base, modifiers)

print(f"  D préparé: shape={D_initial.shape}, rank={D_initial.ndim}")


# ============================================================================
# 2. DÉFINITION DE Γ
# ============================================================================

print(f"Définition de Γ: PureSaturationGamma(β={BETA_GAMMA})...")

gamma = PureSaturationGamma(beta=BETA_GAMMA)


# ============================================================================
# 3. EXÉCUTION (kernel aveugle)
# ============================================================================

print("Exécution du kernel...")

# Condition de convergence (optionnelle)
def check_convergence(state_n, state_next):
    """Arrête si variation < seuil."""
    diff = np.linalg.norm(state_next - state_n)
    return diff < 1e-6

# Fonction de détection d'explosion
def detect_explosion(state):
    """Détecte si valeurs explosent."""
    return np.max(np.abs(state)) > 100

# Stockage manuel de l'historique (échantillonné)
history_sampled = []
sample_interval = 10

# Générateur kernel
final_iteration = 0
final_state = None

for iteration, state in run_kernel(
    D_initial,
    gamma,
    max_iterations=MAX_ITERATIONS,
    convergence_check=check_convergence,
    record_history=False  # Économie mémoire, on stocke manuellement
):
    # Échantillonne historique
    if iteration % sample_interval == 0:
        history_sampled.append(state.copy())
    
    # Check explosion (dans le TM)
    if detect_explosion(state):
        print(f"  ⚠️  Explosion détectée à iteration {iteration}")
        final_iteration = iteration
        final_state = state
        break
    
    # Progress
    if iteration % 200 == 0:
        print(f"  Iteration {iteration}/{MAX_ITERATIONS}")
    
    final_iteration = iteration
    final_state = state

print(f"  Kernel terminé après {final_iteration} itérations")


# ============================================================================
# 4. ANALYSE
# ============================================================================

print("Analyse des résultats...")

# Test symétrie
def test_symmetry(state):
    """Teste si state est symétrique."""
    asymmetry = np.linalg.norm(state - state.T)
    return asymmetry < 1e-6

# Test diversité
def test_diversity(state):
    """Mesure diversité structurelle."""
    n = state.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off_diag = state[mask]
    return float(np.std(off_diag))

symmetry_preserved = test_symmetry(final_state)
diversity_final = test_diversity(final_state)

print(f"  Symétrie préservée: {symmetry_preserved}")
print(f"  Diversité finale: {diversity_final:.4f}")


# ============================================================================
# 5. GÉNÉRATION LOG
# ============================================================================

print("Génération LOG...")

log_data = {
    "hypothesis": "HYP-GAM-001",
    "toy_model": "TM-GAM-001",
    "configuration": {
        "n_dof": N_DOF,
        "epsilon": EPSILON_NOISE,
        "beta": BETA_GAMMA,
        "seed": SEED
    },
    "execution": {
        "max_iterations": MAX_ITERATIONS,
        "completed_iterations": final_iteration,
        "converged": final_iteration < MAX_ITERATIONS
    },
    "results": {
        "symmetry_preserved": symmetry_preserved,
        "diversity_final": diversity_final
    }
}

# Sauvegarder ou afficher
print(f"\nLOG:")
import json
print(json.dumps(log_data, indent=2))

print("\n✅ TM-GAM-001 terminé")