"""
test_gamma_protocol.py

Tests de validation du protocole normatif pour les opérateurs Γ.

Implémente STRICTEMENT le protocole défini dans test_gamma.txt:
1. Symétrie (BLOQUANT)
2. Bornes (diagnostic)
3. Positivité (alerte)
4. Convergence (ambigu)
5. Divergence (informative)
6. Cycles (précieux)
7. Points fixes (cœur)

Usage:
    python test_gamma_protocol.py
"""

import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

# Ajoute le projet au path
current_file = Path(__file__).resolve()
tests_dir = current_file.parent  # prc_framework/tests/
prc_framework_dir = tests_dir.parent  # prc_framework/
project_root = prc_framework_dir.parent  # au-dessus de prc_framework/
core_dir = prc_framework_dir / 'core'  # prc_framework/core/

# Ajoute le chemin correct au sys.path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from prc_framework.core import InformationSpace, PRCKernel, EvolutionOperator
    from prc_framework.core import IdentityOperator, ScalingOperator, CompositeOperator
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print(f"   Chemin recherché: {project_root}")
    print(f"   Dossier core existe: {core_dir.exists()}")
    sys.exit(1)


# ============================================================================
# CONSTANTES DU PROTOCOLE
# ============================================================================

# Conditions initiales standard (OBLIGATOIRES)
N_DOF_STANDARD = 16  # Petit mais > 10
EPSILON_INIT = 0.05  # ε << 1 pour perturbation initiale
N_ITERATIONS_TEST = 100  # Nombre d'itérations standard
SEED_STANDARD = 42  # Pour reproductibilité

# Tolérances
TOL_SYMMETRY = 1e-6
TOL_BOUNDS = 1e-2
TOL_EIGENVALUE = 1e-8
TOL_CONVERGENCE = 1e-6
TOL_CYCLE_DETECTION = 1e-4


# ============================================================================
# STATUTS DE VALIDATION
# ============================================================================

class ValidationStatus(Enum):
    """Statuts possibles après validation."""
    PASS = "✅ PASS"
    FAIL_BLOCKING = "❌ REJET IMMÉDIAT"
    FAIL_DIAGNOSTIC = "⚠️ DIAGNOSTIC REQUIS"
    ALERT = "⚠️ ALERTE"
    INTERESTING = "🔬 INTÉRESSANT"
    EXCELLENT = "⭐ EXCELLENT"


@dataclass
class TestResult:
    """Résultat d'un test individuel."""
    property_name: str
    status: ValidationStatus
    value: float
    threshold: float
    message: str
    blocking: bool = False


@dataclass
class GammaValidation:
    """Validation complète d'un opérateur Γ."""
    gamma_name: str
    tests: List[TestResult]
    final_verdict: ValidationStatus
    recommendation: str
    
    def is_rejected(self) -> bool:
        """Vérifie si Γ est rejeté."""
        return any(t.status == ValidationStatus.FAIL_BLOCKING for t in self.tests)
    
    def has_alerts(self) -> bool:
        """Vérifie si Γ a des alertes."""
        return any(t.status == ValidationStatus.ALERT for t in self.tests)
    
    def is_interesting(self) -> bool:
        """Vérifie si Γ est intéressant."""
        return any(t.status in [ValidationStatus.INTERESTING, ValidationStatus.EXCELLENT] 
                  for t in self.tests)


# ============================================================================
# OPÉRATEURS DE TEST DE BASE
# ============================================================================

# Création d'opérateurs de test basiques pour la démonstration

class TestScalingOperator(EvolutionOperator):
    """Opérateur de mise à l'échelle simple pour test."""
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        
    def apply(self, C: np.ndarray) -> np.ndarray:
        """Applique C' = α·C + (1-α)·I."""
        n = C.shape[0]
        return self.alpha * C + (1 - self.alpha) * np.eye(n)
    
    def get_parameters(self) -> Dict:
        return {"type": "TestScaling", "alpha": self.alpha}


class TestRandomWalkOperator(EvolutionOperator):
    """Opérateur de marche aléatoire pour test."""
    def __init__(self, step_size: float = 0.01, seed: int = 42):
        self.step_size = step_size
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def apply(self, C: np.ndarray) -> np.ndarray:
        """Ajoute du bruit aux éléments hors diagonale."""
        n = C.shape[0]
        noise = self.rng.randn(n, n) * self.step_size
        noise = (noise + noise.T) / 2  # Rend symétrique
        np.fill_diagonal(noise, 0)      # Annule diagonale
        C_new = C + noise
        # Projette sur [-1, 1]
        C_new = np.clip(C_new, -1, 1)
        np.fill_diagonal(C_new, 1)      # Rétablit diagonale à 1
        return C_new
    
    def get_parameters(self) -> Dict:
        return {"type": "TestRandomWalk", "step_size": self.step_size, "seed": self.seed}


class TestDecayOperator(EvolutionOperator):
    """Opérateur de décay exponentiel pour test."""
    def __init__(self, decay_rate: float = 0.1):
        self.decay_rate = decay_rate
        
    def apply(self, C: np.ndarray) -> np.ndarray:
        """Décay exponentiel vers l'identité: C' = I + (C-I)*exp(-decay_rate)."""
        n = C.shape[0]
        I = np.eye(n)
        return I + (C - I) * np.exp(-self.decay_rate)
    
    def get_parameters(self) -> Dict:
        return {"type": "TestDecay", "decay_rate": self.decay_rate}


# ============================================================================
# GÉNÉRATEUR DE CONDITIONS INITIALES STANDARD
# ============================================================================

def create_standard_initial_state(n_dof: int = N_DOF_STANDARD,
                                 epsilon: float = EPSILON_INIT,
                                 seed: int = SEED_STANDARD) -> InformationSpace:
    """
    Crée l'état initial standard selon le protocole.
    
    C_state(0) = I + ε·R
    où R est symétrique, diag(R) = 0
    
    Args:
        n_dof: Nombre de DOF (standard: 16)
        epsilon: Amplitude de perturbation (standard: 0.05)
        seed: Graine pour reproductibilité
        
    Returns:
        InformationSpace avec conditions initiales standard
    """
    np.random.seed(seed)
    
    # Génère matrice aléatoire R
    R = np.random.randn(n_dof, n_dof)
    
    # Rend symétrique
    R = (R + R.T) / 2
    
    # Annule diagonale
    np.fill_diagonal(R, 0)
    
    # Normalise pour que ||R||_max = 1
    R_max = np.max(np.abs(R))
    if R_max > 0:
        R = R / R_max
    
    # Construit C_state(0) = I + ε·R
    C_init = np.eye(n_dof) + epsilon * R
    
    # Assure les contraintes (au cas où)
    C_init = np.clip(C_init, -1.0, 1.0)
    np.fill_diagonal(C_init, 1.0)
    
    metadata = {
        "protocol": "standard_initial_state",
        "epsilon": epsilon,
        "seed": seed,
        "n_dof": n_dof
    }
    
    return InformationSpace(C_init, metadata)


# ============================================================================
# TESTS INDIVIDUELS
# ============================================================================

def test_symmetry(C_history: List[np.ndarray]) -> TestResult:
    """
    TEST 1: Symétrie (C = C^T)
    
    STATUT: BLOQUANT
    Si échec → REJET IMMÉDIAT
    """
    max_asymmetry = 0.0
    
    for C in C_history:
        asymmetry = np.linalg.norm(C - C.T, ord='fro')
        max_asymmetry = max(max_asymmetry, asymmetry)
    
    if max_asymmetry < TOL_SYMMETRY:
        return TestResult(
            property_name="Symétrie",
            status=ValidationStatus.PASS,
            value=max_asymmetry,
            threshold=TOL_SYMMETRY,
            message=f"Symétrie préservée (erreur max: {max_asymmetry:.2e})",
            blocking=False
        )
    else:
        return TestResult(
            property_name="Symétrie",
            status=ValidationStatus.FAIL_BLOCKING,
            value=max_asymmetry,
            threshold=TOL_SYMMETRY,
            message=f"VIOLATION DE SYMÉTRIE (erreur: {max_asymmetry:.2e})",
            blocking=True
        )


def test_bounds(C_history: List[np.ndarray]) -> TestResult:
    """
    TEST 2: Bornes (|C_ij| ≤ 1)
    
    STATUT: Diagnostic requis si échec
    """
    max_violation = 0.0
    explosive = False
    
    for i, C in enumerate(C_history):
        # Ignore diagonale
        n = C.shape[0]
        mask = ~np.eye(n, dtype=bool)
        off_diag = C[mask]
        
        violation = max(0, np.max(np.abs(off_diag)) - 1.0)
        max_violation = max(max_violation, violation)
        
        # Détecte croissance explosive
        if violation > 0.5 and i < len(C_history) // 2:
            explosive = True
    
    if max_violation < TOL_BOUNDS:
        return TestResult(
            property_name="Bornes",
            status=ValidationStatus.PASS,
            value=max_violation,
            threshold=1.0,
            message=f"Bornes respectées (violation max: {max_violation:.3f})",
            blocking=False
        )
    elif explosive:
        return TestResult(
            property_name="Bornes",
            status=ValidationStatus.FAIL_BLOCKING,
            value=max_violation,
            threshold=1.0,
            message=f"CROISSANCE EXPLOSIVE (violation: {max_violation:.3f})",
            blocking=True
        )
    else:
        return TestResult(
            property_name="Bornes",
            status=ValidationStatus.FAIL_DIAGNOSTIC,
            value=max_violation,
            threshold=1.0,
            message=f"Dépassement marginal (violation: {max_violation:.3f}) - diagnostic requis",
            blocking=False
        )


def test_positivity(C_history: List[np.ndarray]) -> TestResult:
    """
    TEST 3: Positivité (semi-définie positive)
    
    STATUT: Alerte majeure si échec fort
    """
    min_eigenvalue = float('inf')
    persistent_negative = False
    
    negative_count = 0
    for C in C_history:
        eigenvalues = np.linalg.eigvalsh(C)
        min_eig = np.min(eigenvalues)
        min_eigenvalue = min(min_eigenvalue, min_eig)
        
        if min_eig < -TOL_EIGENVALUE:
            negative_count += 1
    
    # Vérifie si violations persistantes
    if negative_count > len(C_history) * 0.5:
        persistent_negative = True
    
    if min_eigenvalue >= -TOL_EIGENVALUE:
        return TestResult(
            property_name="Positivité",
            status=ValidationStatus.PASS,
            value=min_eigenvalue,
            threshold=0.0,
            message=f"Semi-définie positive (λ_min: {min_eigenvalue:.3e})",
            blocking=False
        )
    elif min_eigenvalue < -0.1 or persistent_negative:
        return TestResult(
            property_name="Positivité",
            status=ValidationStatus.FAIL_BLOCKING,
            value=min_eigenvalue,
            threshold=0.0,
            message=f"VIOLATION FORTE (λ_min: {min_eigenvalue:.3e}, {negative_count}/{len(C_history)} négatif)",
            blocking=True
        )
    else:
        return TestResult(
            property_name="Positivité",
            status=ValidationStatus.ALERT,
            value=min_eigenvalue,
            threshold=0.0,
            message=f"Violations faibles/transitoires (λ_min: {min_eigenvalue:.3e})",
            blocking=False
        )


def test_convergence(C_history: List[np.ndarray]) -> TestResult:
    """
    TEST 4: Convergence
    
    STATUT: Ambigu - dépend vers quoi on converge
    """
    if len(C_history) < 10:
        return TestResult(
            property_name="Convergence",
            status=ValidationStatus.PASS,
            value=0.0,
            threshold=0.0,
            message="Historique trop court pour analyse",
            blocking=False
        )
    
    # Calcule différences successives sur derniers 20%
    window = max(10, len(C_history) // 5)
    recent = C_history[-window:]
    
    diffs = []
    for i in range(len(recent) - 1):
        diff = np.linalg.norm(recent[i+1] - recent[i], ord='fro')
        diffs.append(diff)
    
    mean_diff = np.mean(diffs)
    converged = mean_diff < TOL_CONVERGENCE
    
    if not converged:
        return TestResult(
            property_name="Convergence",
            status=ValidationStatus.PASS,
            value=mean_diff,
            threshold=TOL_CONVERGENCE,
            message=f"Pas de convergence (diff moy: {mean_diff:.3e})",
            blocking=False
        )
    
    # Converge - vers quoi?
    C_final = C_history[-1]
    n = C_final.shape[0]
    
    # Teste si trivial (identité ou zéro)
    dist_to_identity = np.linalg.norm(C_final - np.eye(n), ord='fro')
    mean_corr = np.mean(np.abs(C_final[~np.eye(n, dtype=bool)]))
    
    if dist_to_identity < 0.1 or mean_corr < 0.05:
        return TestResult(
            property_name="Convergence",
            status=ValidationStatus.FAIL_BLOCKING,
            value=mean_diff,
            threshold=TOL_CONVERGENCE,
            message=f"CONVERGENCE VERS TRIVIAL (dist I: {dist_to_identity:.3f})",
            blocking=True
        )
    else:
        return TestResult(
            property_name="Convergence",
            status=ValidationStatus.EXCELLENT,
            value=mean_diff,
            threshold=TOL_CONVERGENCE,
            message=f"Convergence vers structure non triviale (corr moy: {mean_corr:.3f})",
            blocking=False
        )


def test_divergence(C_history: List[np.ndarray]) -> TestResult:
    """
    TEST 5: Divergence
    
    STATUT: Pas éliminatoire par défaut
    """
    norms = [np.linalg.norm(C, ord='fro') for C in C_history]
    
    # Teste croissance exponentielle
    if len(norms) > 20:
        early = np.mean(norms[:10])
        late = np.mean(norms[-10:])
        growth_factor = late / (early + 1e-10)
        
        if growth_factor > 10:
            return TestResult(
                property_name="Divergence",
                status=ValidationStatus.FAIL_BLOCKING,
                value=growth_factor,
                threshold=2.0,
                message=f"DIVERGENCE GÉNÉRIQUE (facteur: {growth_factor:.2f})",
                blocking=True
            )
        elif growth_factor > 2:
            return TestResult(
                property_name="Divergence",
                status=ValidationStatus.INTERESTING,
                value=growth_factor,
                threshold=2.0,
                message=f"Croissance conditionnelle (facteur: {growth_factor:.2f}) - à étudier",
                blocking=False
            )
    
    return TestResult(
        property_name="Divergence",
        status=ValidationStatus.PASS,
        value=norms[-1] if norms else 0,
        threshold=100.0,
        message="Pas de divergence détectée",
        blocking=False
    )


def test_cycles(C_history: List[np.ndarray]) -> TestResult:
    """
    TEST 6: Cycles / Oscillations
    
    STATUT: Très précieux si détecté
    """
    if len(C_history) < 30:
        return TestResult(
            property_name="Cycles",
            status=ValidationStatus.PASS,
            value=0.0,
            threshold=0.0,
            message="Historique trop court pour détecter cycles",
            blocking=False
        )
    
    # Cherche périodicité sur dernière moitié
    recent = C_history[len(C_history)//2:]
    n_recent = len(recent)
    
    # Teste périodes candidates
    best_period = 0
    best_score = float('inf')
    
    for period in range(2, min(20, n_recent // 3)):
        scores = []
        for i in range(n_recent - period):
            diff = np.linalg.norm(recent[i] - recent[i + period], ord='fro')
            scores.append(diff)
        
        mean_score = np.mean(scores)
        if mean_score < best_score:
            best_score = mean_score
            best_period = period
    
    if best_score < TOL_CYCLE_DETECTION:
        return TestResult(
            property_name="Cycles",
            status=ValidationStatus.EXCELLENT,
            value=best_period,
            threshold=TOL_CYCLE_DETECTION,
            message=f"CYCLE DÉTECTÉ (période: {best_period}, erreur: {best_score:.3e})",
            blocking=False
        )
    else:
        return TestResult(
            property_name="Cycles",
            status=ValidationStatus.PASS,
            value=best_score,
            threshold=TOL_CYCLE_DETECTION,
            message="Pas de cycle détecté",
            blocking=False
        )


def test_fixed_points(C_history: List[np.ndarray]) -> TestResult:
    """
    TEST 7: Points fixes non triviaux
    
    STATUT: Cœur du programme
    """
    if len(C_history) < 50:
        return TestResult(
            property_name="Points fixes",
            status=ValidationStatus.PASS,
            value=0.0,
            threshold=0.0,
            message="Historique insuffisant",
            blocking=False
        )
    
    # Vérifie si point fixe atteint (derniers pas quasi identiques)
    last_10 = C_history[-10:]
    diffs = []
    for i in range(len(last_10) - 1):
        diff = np.linalg.norm(last_10[i+1] - last_10[i], ord='fro')
        diffs.append(diff)
    
    is_fixed = np.max(diffs) < TOL_CONVERGENCE
    
    if not is_fixed:
        return TestResult(
            property_name="Points fixes",
            status=ValidationStatus.PASS,
            value=np.max(diffs),
            threshold=TOL_CONVERGENCE,
            message="Pas encore de point fixe",
            blocking=False
        )
    
    # Point fixe atteint - est-il trivial?
    C_final = C_history[-1]
    n = C_final.shape[0]
    
    dist_to_identity = np.linalg.norm(C_final - np.eye(n), ord='fro')
    mean_corr = np.mean(np.abs(C_final[~np.eye(n, dtype=bool)]))
    
    if dist_to_identity < 0.1 or mean_corr < 0.05:
        return TestResult(
            property_name="Points fixes",
            status=ValidationStatus.FAIL_BLOCKING,
            value=mean_corr,
            threshold=0.1,
            message=f"Point fixe TRIVIAL (corr moy: {mean_corr:.3f})",
            blocking=True
        )
    else:
        # Analyse structure
        eigenvalues = np.linalg.eigvalsh(C_final)
        effective_rank = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)
        
        return TestResult(
            property_name="Points fixes",
            status=ValidationStatus.EXCELLENT,
            value=mean_corr,
            threshold=0.1,
            message=f"Point fixe NON TRIVIAL (corr: {mean_corr:.3f}, rang eff: {effective_rank:.1f})",
            blocking=False
        )


# ============================================================================
# VALIDATION COMPLÈTE D'UN Γ
# ============================================================================

def validate_gamma(gamma: EvolutionOperator,
                  n_iterations: int = N_ITERATIONS_TEST,
                  verbose: bool = False) -> GammaValidation:
    """
    Valide un opérateur Γ selon le protocole complet.
    
    Args:
        gamma: Opérateur à tester
        n_iterations: Nombre d'itérations
        verbose: Affiche progression
        
    Returns:
        GammaValidation avec tous les résultats
    """
    gamma_name = gamma.__class__.__name__
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"VALIDATION: {gamma_name}")
        print(f"{'='*70}")
    
    # 1. Crée état initial standard
    D_init = create_standard_initial_state()
    
    # 2. Simule avec enregistrement historique
    kernel = PRCKernel(D_init, gamma)
    kernel.start_recording()
    
    try:
        kernel.step(n_iterations)
    except Exception as e:
        # Si crash pendant simulation
        return GammaValidation(
            gamma_name=gamma_name,
            tests=[],
            final_verdict=ValidationStatus.FAIL_BLOCKING,
            recommendation=f"CRASH pendant simulation: {e}"
        )
    
    history = kernel.get_history()
    C_history = [state.C for state in history]
    
    # 3. Exécute tous les tests
    tests = [
        test_symmetry(C_history),
        test_bounds(C_history),
        test_positivity(C_history),
        test_convergence(C_history),
        test_divergence(C_history),
        test_cycles(C_history),
        test_fixed_points(C_history),
    ]
    
    # 4. Détermine verdict final
    if any(t.blocking for t in tests):
        final_verdict = ValidationStatus.FAIL_BLOCKING
        recommendation = "REJET IMMÉDIAT - violations bloquantes"
    elif any(t.status == ValidationStatus.EXCELLENT for t in tests):
        final_verdict = ValidationStatus.EXCELLENT
        recommendation = "CANDIDAT FORT - passer à analyse de stabilité"
    elif any(t.status == ValidationStatus.INTERESTING for t in tests):
        final_verdict = ValidationStatus.INTERESTING
        recommendation = "INTÉRESSANT - à conserver pour recherche"
    elif any(t.status == ValidationStatus.ALERT for t in tests):
        final_verdict = ValidationStatus.ALERT
        recommendation = "ACCEPTABLE en phase exploratoire - noter explicitement"
    else:
        final_verdict = ValidationStatus.PASS
        recommendation = "VALIDE - continuer tests"
    
    return GammaValidation(
        gamma_name=gamma_name,
        tests=tests,
        final_verdict=final_verdict,
        recommendation=recommendation
    )


# ============================================================================
# RAPPORT ET AFFICHAGE
# ============================================================================

def print_validation_report(validation: GammaValidation, detailed: bool = True):
    """Affiche le rapport de validation."""
    print(f"\n{'='*70}")
    print(f"Γ: {validation.gamma_name}")
    print(f"{'='*70}")
    
    if detailed:
        for test in validation.tests:
            status_str = test.status.value
            print(f"\n{status_str:12} | {test.property_name}")
            print(f"             | {test.message}")
            if test.blocking:
                print(f"             | ⚠️  BLOQUANT")
    
    print(f"\n{'─'*70}")
    print(f"VERDICT FINAL: {validation.final_verdict.value}")
    print(f"RECOMMANDATION: {validation.recommendation}")
    print(f"{'='*70}")


def generate_summary_table(validations: List[GammaValidation]):
    """Génère un tableau récapitulatif."""
    print(f"\n{'='*70}")
    print("TABLEAU RÉCAPITULATIF")
    print(f"{'='*70}\n")
    
    # Regroupe par verdict
    by_verdict = {}
    for v in validations:
        verdict = v.final_verdict
        if verdict not in by_verdict:
            by_verdict[verdict] = []
        by_verdict[verdict].append(v.gamma_name)
    
    # Affiche par catégorie
    order = [
        ValidationStatus.EXCELLENT,
        ValidationStatus.INTERESTING,
        ValidationStatus.PASS,
        ValidationStatus.ALERT,
        ValidationStatus.FAIL_DIAGNOSTIC,
        ValidationStatus.FAIL_BLOCKING,
    ]
    
    for verdict in order:
        if verdict in by_verdict:
            print(f"{verdict.value}")
            for name in by_verdict[verdict]:
                print(f"  • {name}")
            print()
    
    # Stats
    total = len(validations)
    rejected = sum(1 for v in validations if v.is_rejected())
    excellent = sum(1 for v in validations if v.final_verdict == ValidationStatus.EXCELLENT)
    interesting = sum(1 for v in validations if v.final_verdict == ValidationStatus.INTERESTING)
    
    print(f"{'─'*70}")
    print(f"STATISTIQUES:")
    print(f"  Total testé: {total}")
    print(f"  Rejetés: {rejected} ({100*rejected/total:.1f}%)")
    print(f"  Excellents: {excellent} ({100*excellent/total:.1f}%)")
    print(f"  Intéressants: {interesting} ({100*interesting/total:.1f}%)")
    print(f"{'='*70}\n")


# ============================================================================
# SUITE DE TESTS PRINCIPALE
# ============================================================================

def run_full_protocol(operators_subset: Optional[List[str]] = None,
                     verbose: bool = False):
    """
    Exécute le protocole complet sur tous les opérateurs.
    
    Args:
        operators_subset: Liste de noms d'opérateurs à tester (None = tous)
        verbose: Affiche détails pour chaque opérateur
    """
    print("\n" + "="*70)
    print("PROTOCOLE NORMATIF — VALIDATION DE Γ")
    print("="*70)
    print(f"\nConditions standard:")
    print(f"  • N_DOF = {N_DOF_STANDARD}")
    print(f"  • ε = {EPSILON_INIT}")
    print(f"  • Itérations = {N_ITERATIONS_TEST}")
    print(f"  • Seed = {SEED_STANDARD}")
    
    # Liste des opérateurs de test de base
    all_operators = {
        "Identity": IdentityOperator(),
        "Scaling_0.8": ScalingOperator(alpha=0.8),
        "Scaling_0.5": ScalingOperator(alpha=0.5),
        "Scaling_0.2": ScalingOperator(alpha=0.2),
        "Composite": CompositeOperator([
            ScalingOperator(alpha=0.9),
            ScalingOperator(alpha=0.95)
        ]),
        "TestScaling": TestScalingOperator(alpha=0.5),
        "TestRandomWalk": TestRandomWalkOperator(step_size=0.01, seed=SEED_STANDARD),
        "TestDecay": TestDecayOperator(decay_rate=0.1),
    }
    
    # Filtre si subset spécifié
    if operators_subset:
        operators = {k: v for k, v in all_operators.items() if k in operators_subset}
    else:
        operators = all_operators
    
    # Valide chaque opérateur
    validations = []
    
    for name, gamma in operators.items():
        print(f"\nTest {len(validations)+1}/{len(operators)}: {name}...", end=" ")
        
        try:
            validation = validate_gamma(gamma, verbose=verbose)
            validations.append(validation)
            
            # Indicateur rapide
            if validation.is_rejected():
                print("❌ REJETÉ")
            elif validation.final_verdict == ValidationStatus.EXCELLENT:
                print("⭐ EXCELLENT")
            elif validation.final_verdict == ValidationStatus.INTERESTING:
                print("🔬 INTÉRESSANT")
            else:
                print("✅ PASS")
            
            if verbose:
                print_validation_report(validation, detailed=True)
        
        except Exception as e:
            print(f"⚠️  ERREUR: {e}")
            import traceback
            traceback.print_exc()
    
    # Génère rapport final
    print("\n" + "="*70)
    print("VALIDATION TERMINÉE")
    print("="*70)
    
    generate_summary_table(validations)
    
    # Affiche détails des excellents et intéressants
    print("DÉTAILS DES CANDIDATS FORTS:\n")
    for v in validations:
        if v.final_verdict in [ValidationStatus.EXCELLENT, ValidationStatus.INTERESTING]:
            print_validation_report(v, detailed=True)
    
    return validations


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validation protocole Γ")
    parser.add_argument("--operators", nargs="+", help="Liste d'opérateurs à tester")
    parser.add_argument("--verbose", action="store_true", help="Mode verbeux")
    parser.add_argument("--iterations", type=int, default=N_ITERATIONS_TEST, 
                       help="Nombre d'itérations")
    
    args = parser.parse_args()
    
    if args.iterations != N_ITERATIONS_TEST:
        N_ITERATIONS_TEST = args.iterations
    
    validations = run_full_protocol(
        operators_subset=args.operators,
        verbose=args.verbose
    )
    
    # Code de sortie: 0 si au moins un excellent, 1 sinon
    has_excellent = any(v.final_verdict == ValidationStatus.EXCELLENT 
                       for v in validations)
    sys.exit(0 if has_excellent else 1)