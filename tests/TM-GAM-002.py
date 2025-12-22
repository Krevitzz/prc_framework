"""
tests/TM-GAM-002.py

Toy Model pour tester HYP-GAM-002 : Diffusion pure comme mécanisme Γ.

APPROCHE HYBRIDE :
1. Réutilise les tests de base de test_gamma_protocol.py
2. Ajoute des tests spécifiques à la diffusion
3. Protocole complet documenté pour génération LOG/RES

QUESTION CENTRALE :
    "La propagation pure peut-elle maintenir structure ?"

MÉTADONNÉES :
    ID : TM-GAM-002
    LEVEL : L3
    ASSOCIATED_HYPOTHESIS : HYP-GAM-002
    STATUS : WIP
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# Import du projet
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core import InformationSpace, PRCKernel
from operators.gamma_hyp_002 import PureDiffusionGamma, get_operator_metadata

# Import des tests de base (réutilisation)
from tests.test_gamma_protocol import (
    test_symmetry,
    test_bounds,
    test_positivity,
    test_convergence,
    TestResult,
    ValidationStatus,
    create_standard_initial_state,
)


# ============================================================================
# MÉTADONNÉES TM-GAM-002
# ============================================================================

TOY_MODEL_METADATA = {
    "id": "TM-GAM-002",
    "level": "L3",
    "associated_hypothesis": "HYP-GAM-002",
    "status": "WIP",
    "operator": "PureDiffusionGamma",
    "question": "La propagation pure peut-elle maintenir structure ?",
}


# ============================================================================
# CONFIGURATION SPÉCIFIQUE
# ============================================================================

# Configuration standard (héritée de test_gamma_protocol)
N_DOF_STANDARD = 50
EPSILON_INIT = 0.05
SEED_STANDARD = 42
N_ITERATIONS_BASE = 1000

# Configuration spécifique HYP-GAM-002
ALPHA_VALUES_TO_TEST = [0.005, 0.01, 0.02, 0.05, 0.1]  # Balayage α
N_ITERATIONS_EXTENDED = 2000  # Pour observer convergence complète

# Seuils numériques explicites
TOL_HOMOGENEITY = 0.05  # Std des corrélations < 5% → homogène
TOL_DIVERSITY_LOSS = 0.1  # Diversité < 10% de l'initiale → échec
TOL_PLATEAU = 1e-4  # ||C_{n+1} - C_n|| < tol → plateau atteint
WINDOW_PLATEAU = 50  # Fenêtre d'observation pour plateau


# ============================================================================
# TESTS SPÉCIFIQUES HYP-GAM-002 (ajouts au protocole de base)
# ============================================================================

def test_homogenization(C_history: List[np.ndarray]) -> TestResult:
    """
    Test L2.2 : Homogénéisation des corrélations.
    
    Vérifie si la diffusion conduit à une uniformisation progressive
    des corrélations (toutes → même valeur).
    
    Métrique : écart-type des corrélations hors-diagonale
    
    Args:
        C_history: Historique complet [C_0, ..., C_n]
        
    Returns:
        TestResult avec verdict sur homogénéisation
    """
    def compute_correlation_std(C: np.ndarray) -> float:
        """Écart-type des corrélations hors-diagonale."""
        n = C.shape[0]
        mask = ~np.eye(n, dtype=bool)
        off_diag = C[mask]
        return np.std(off_diag)
    
    # Calcule std initiale et finale
    std_init = compute_correlation_std(C_history[0])
    std_final = compute_correlation_std(C_history[-1])
    
    # Calcule évolution temporelle
    stds = [compute_correlation_std(C) for C in C_history[::10]]  # Échantillonne
    
    # Vérifie monotonie décroissante
    is_monotone_decreasing = all(
        stds[i] >= stds[i+1] - 0.01  # Tolérance numérique
        for i in range(len(stds)-1)
    )
    
    # Calcule perte relative
    homogenization_rate = 1 - (std_final / (std_init + 1e-10))
    
    # Verdict
    if std_final < TOL_HOMOGENEITY:
        status = ValidationStatus.FAIL_BLOCKING
        message = f"Homogénéisation complète (std final = {std_final:.3f})"
    elif is_monotone_decreasing and homogenization_rate > 0.7:
        status = ValidationStatus.FAIL_BLOCKING
        message = f"Homogénéisation monotone forte ({homogenization_rate*100:.1f}%)"
    elif homogenization_rate < 0.3:
        status = ValidationStatus.EXCELLENT
        message = f"Diversité maintenue ({homogenization_rate*100:.1f}% homogénéisation)"
    else:
        status = ValidationStatus.ALERT
        message = f"Homogénéisation partielle ({homogenization_rate*100:.1f}%)"
    
    return TestResult(
        property_name="Homogenization",
        status=status,
        value=std_final,
        threshold=TOL_HOMOGENEITY,
        message=message,
        blocking=(status == ValidationStatus.FAIL_BLOCKING)
    )


def test_structural_diversity_preservation(C_history: List[np.ndarray]) -> TestResult:
    """
    Test L2.3 : Préservation de la diversité structurelle.
    
    Mesure l'évolution de la diversité informationnelle au fil du temps.
    
    Métriques :
    - Entropie hors-diagonale (Shannon sur distribution des valeurs)
    - Variance des corrélations
    - Rang effectif (participation ratio)
    
    Args:
        C_history: Historique complet [C_0, ..., C_n]
        
    Returns:
        TestResult avec verdict sur préservation diversité
    """
    def compute_diversity(C: np.ndarray) -> float:
        """Diversité = moyenne de 3 métriques normalisées."""
        n = C.shape[0]
        mask = ~np.eye(n, dtype=bool)
        off_diag = C[mask]
        
        # 1. Entropie (distribution discrétisée)
        hist, _ = np.histogram(off_diag, bins=20, range=(-1, 1))
        prob = hist / (hist.sum() + 1e-10)
        entropy = -np.sum(prob * np.log(prob + 1e-10))
        entropy_norm = entropy / np.log(20)  # Normalise [0, 1]
        
        # 2. Variance
        variance = np.var(off_diag)
        variance_norm = np.clip(variance, 0, 1)  # Déjà dans [0, 1]
        
        # 3. Rang effectif
        eig = np.linalg.eigvalsh(C)
        eig_pos = eig - eig.min() + 1e-10
        eig_norm = eig_pos / eig_pos.sum()
        participation = 1 / np.sum(eig_norm**2)
        rank_norm = participation / n  # Normalise par taille
        
        # Moyenne des 3
        return (entropy_norm + variance_norm + rank_norm) / 3
    
    # Calcule diversité initiale et finale
    diversity_init = compute_diversity(C_history[0])
    diversity_final = compute_diversity(C_history[-1])
    
    # Calcule perte relative
    diversity_loss = 1 - (diversity_final / (diversity_init + 1e-10))
    
    # Verdict
    if diversity_loss > (1 - TOL_DIVERSITY_LOSS):
        status = ValidationStatus.FAIL_BLOCKING
        message = f"Annihilation informationnelle ({diversity_loss*100:.1f}% perdue)"
    elif diversity_loss < 0.2:
        status = ValidationStatus.EXCELLENT
        message = f"Diversité préservée ({diversity_loss*100:.1f}% perdue)"
    else:
        status = ValidationStatus.PASS
        message = f"Diversité partiellement préservée ({diversity_loss*100:.1f}% perdue)"
    
    return TestResult(
        property_name="Structural diversity",
        status=status,
        value=diversity_final,
        threshold=TOL_DIVERSITY_LOSS,
        message=message,
        blocking=(status == ValidationStatus.FAIL_BLOCKING)
    )


def test_alpha_sensitivity(C_histories: Dict[float, List[np.ndarray]]) -> TestResult:
    """
    Test L2.4 : Sensibilité à α (bifurcation ?).
    
    Vérifie si différentes valeurs de α produisent des attracteurs
    qualitativement différents ou juste une convergence plus/moins rapide.
    
    Args:
        C_histories: {alpha: [C_0, C_1, ..., C_n]} pour chaque α testé
        
    Returns:
        TestResult avec verdict sur la richesse dynamique
    """
    # Compare états finaux pour chaque α
    final_states = {alpha: history[-1] for alpha, history in C_histories.items()}
    
    # Calcule distances entre tous les états finaux
    alphas = list(final_states.keys())
    distances = []
    
    for i, alpha_i in enumerate(alphas):
        for alpha_j in alphas[i+1:]:
            dist = np.linalg.norm(final_states[alpha_i] - final_states[alpha_j])
            distances.append((alpha_i, alpha_j, dist))
    
    mean_distance = np.mean([d[2] for d in distances])
    max_distance = np.max([d[2] for d in distances])
    
    # Verdict
    if max_distance < 0.1:
        status = ValidationStatus.FAIL_BLOCKING
        message = f"Tous les α convergent vers le même attracteur (max_dist={max_distance:.3f})"
    elif mean_distance < 0.3:
        status = ValidationStatus.ALERT
        message = f"Faible différenciation entre α (mean_dist={mean_distance:.3f})"
    else:
        status = ValidationStatus.EXCELLENT
        message = f"Bifurcations détectées selon α (mean_dist={mean_distance:.3f})"
    
    return TestResult(
        property_name="Alpha-sensitivity",
        status=status,
        value=mean_distance,
        threshold=0.3,
        message=message,
        blocking=(status == ValidationStatus.FAIL_BLOCKING)
    )


def test_transitive_coupling(C_init: np.ndarray, 
                              C_final: np.ndarray) -> TestResult:
    """
    Test L1.3 : Effet du couplage transitif.
    
    Vérifie que la diffusion introduit bien des couplages via transitivité
    (contrairement à opérateur pointwise).
    
    Métrique : Si C_init[i,j] faible mais (C_init[i,k] * C_init[k,j]) fort,
               alors C_final[i,j] doit avoir augmenté.
    
    Args:
        C_init: Matrice initiale
        C_final: Matrice finale
        
    Returns:
        TestResult confirmant couplage transitif
    """
    n = C_init.shape[0]
    
    # Identifie paires (i,j) avec corrélation initiale faible
    # mais corrélation indirecte forte (via k)
    mask_diag = np.eye(n, dtype=bool)
    
    transitive_influence = C_init @ C_init
    
    # Cherche cas où :
    # - C_init[i,j] faible (< 0.2)
    # - Transitive_influence[i,j] fort (> 0.5)
    weak_direct = np.abs(C_init) < 0.2
    strong_indirect = transitive_influence > 0.5
    
    candidates = weak_direct & strong_indirect & ~mask_diag
    
    if not np.any(candidates):
        return TestResult(
            property_name="Transitive coupling",
            status=ValidationStatus.PASS,
            value=0.0,
            threshold=0.0,
            message="Aucun cas de couplage transitif détectable dans C_init",
            blocking=False
        )
    
    # Pour ces candidats, vérifie si C_final a augmenté
    changes = C_final[candidates] - C_init[candidates]
    mean_increase = np.mean(changes)
    fraction_increased = np.mean(changes > 0.01)
    
    # Verdict
    if fraction_increased > 0.7:
        status = ValidationStatus.PASS
        message = f"Couplage transitif détecté ({fraction_increased*100:.0f}% paires renforcées)"
    elif fraction_increased > 0.3:
        status = ValidationStatus.ALERT
        message = f"Couplage transitif faible ({fraction_increased*100:.0f}% paires renforcées)"
    else:
        status = ValidationStatus.FAIL_BLOCKING
        message = f"Pas de couplage transitif ({fraction_increased*100:.0f}% paires renforcées)"
    
    return TestResult(
        property_name="Transitive coupling",
        status=status,
        value=fraction_increased,
        threshold=0.5,
        message=message,
        blocking=(status == ValidationStatus.FAIL_BLOCKING)
    )


# ============================================================================
# ANALYSE 3 : DÉPENDANCE À C₀
# ============================================================================

def test_initial_dependence():
    """
    Teste si la dynamique dépend réellement de C₀.
    
    Simule avec plusieurs C₀ différents et compare les états finaux.
    """
    print("\n" + "="*70)
    print("ANALYSE 3: DÉPENDANCE À C₀")
    print("="*70)
    
    gamma = PureDiffusionGamma(alpha=0.02)
    
    # Teste 5 conditions initiales différentes
    initial_states = []
    final_states = []
    
    for seed in [42, 123, 456, 789, 1011]:
        # Crée C₀ différent
        np.random.seed(seed)
        R = np.random.randn(N_DOF_STANDARD, N_DOF_STANDARD)
        R = (R + R.T) / 2
        np.fill_diagonal(R, 0)
        R = R / np.max(np.abs(R))
        C0 = np.eye(N_DOF_STANDARD) + EPSILON_INIT * R
        C0 = np.clip(C0, -1, 1)
        np.fill_diagonal(C0, 1)
        
        D0 = InformationSpace(C0, {"seed": seed})
        initial_states.append(C0)
        
        # Simule
        kernel = PRCKernel(D0, gamma)
        kernel.step(N_ITERATIONS_EXTENDED)
        final_states.append(kernel.C_current)
    
    # Compare états finaux
    print(f"\n📊 Variabilité des états finaux:")
    
    diversities = []
    for i in range(len(final_states)):
        for j in range(i+1, len(final_states)):
            diff = np.linalg.norm(final_states[i] - final_states[j])
            diversities.append(diff)
    
    mean_diversity = np.mean(diversities)
    
    print(f"  Distance moyenne entre états finaux: {mean_diversity:.6f}")
    
    if mean_diversity < 1e-3:
        print(f"  ⚠️  États finaux quasi-identiques (attracteur unique)")
        dependence = "weak"
    elif mean_diversity < 0.1:
        print(f"  ✅ Variabilité modérée (dépendance à C₀)")
        dependence = "moderate"
    else:
        print(f"  ✅ Forte variabilité (forte dépendance à C₀)")
        dependence = "strong"
    
    return dependence


# ============================================================================
# PROTOCOLE COMPLET TM-GAM-002
# ============================================================================

@dataclass
class TM_GAM_002_Results:
    """Résultats complets du toy model."""
    toy_model_id: str
    hypothesis_id: str
    
    # Tests de base (hérités)
    base_tests: Dict[float, List[TestResult]]  # {alpha: [tests]}
    
    # Tests spécifiques
    homogenization_tests: Dict[float, TestResult]  # {alpha: test}
    diversity_tests: Dict[float, TestResult]  # {alpha: test}
    transitive_tests: Dict[float, TestResult]  # {alpha: test}
    alpha_sensitivity: TestResult
    
    # Données brutes
    C_histories: Dict[float, List[np.ndarray]]
    
    # Verdict global
    global_status: ValidationStatus
    summary: str


def run_tm_gam_002(verbose: bool = True) -> TM_GAM_002_Results:
    """
    Exécute le protocole complet pour TM-GAM-002.
    
    PROTOCOLE HYBRIDE :
    
    Niveau L1 (Invariants formels) :
    ├─ test_symmetry (base)
    ├─ test_bounds (base)
    ├─ test_positivity (base)
    └─ test_transitive_coupling (spécifique) ✨
    
    Niveau L2 (Dynamique itérée) :
    ├─ test_convergence (base)
    ├─ test_homogenization (spécifique) ✨
    ├─ test_structural_diversity_preservation (spécifique) ✨
    └─ test_alpha_sensitivity (spécifique) ✨
    
    Args:
        verbose: Affiche progression si True
        
    Returns:
        TM_GAM_002_Results avec tous les résultats
    """
    if verbose:
        print("\n" + "="*70)
        print("TM-GAM-002 : Test de HYP-GAM-002 (Diffusion pure)")
        print("="*70)
        print(f"\nQuestion centrale : {TOY_MODEL_METADATA['question']}")
        print(f"\nConfiguration :")
        print(f"  N_DOF = {N_DOF_STANDARD}")
        print(f"  ε = {EPSILON_INIT}")
        print(f"  α values = {ALPHA_VALUES_TO_TEST}")
        print(f"  Iterations = {N_ITERATIONS_EXTENDED}")
    
    # Stockage résultats
    base_tests_all = {}
    homogenization_tests = {}
    diversity_tests = {}
    transitive_tests = {}
    C_histories = {}
    
    # Pour chaque valeur de α
    for alpha in ALPHA_VALUES_TO_TEST:
        if verbose:
            print(f"\n{'─'*70}")
            print(f"Testing α = {alpha}")
            print(f"{'─'*70}")
        
        # 1. Crée état initial standard
        D_init = create_standard_initial_state(
            n_dof=N_DOF_STANDARD,
            epsilon=EPSILON_INIT,
            seed=SEED_STANDARD
        )
        
        # 2. Crée opérateur
        gamma = PureDiffusionGamma(alpha=alpha)
        
        # 3. Simule
        kernel = PRCKernel(D_init, gamma)
        kernel.start_recording()
        kernel.step(N_ITERATIONS_EXTENDED)
        
        # 4. Extrait historique
        history = kernel.get_history()
        C_history = [state.C for state in history]
        C_histories[alpha] = C_history
        
        # 5. TESTS DE BASE (réutilisés)
        if verbose:
            print("\n  Tests L1 (invariants) :")
        
        base_tests = [
            test_symmetry(C_history),
            test_bounds(C_history),
            test_positivity(C_history),
        ]
        
        # 6. TEST SPÉCIFIQUE L1 : Transitive coupling
        transitive_test = test_transitive_coupling(C_history[0], C_history[-1])
        transitive_tests[alpha] = transitive_test
        base_tests.append(transitive_test)
        
        if verbose:
            for test in base_tests:
                print(f"    {test.status.value:12} | {test.property_name}")
        
        # 7. TESTS L2
        if verbose:
            print("\n  Tests L2 (dynamique) :")
        
        convergence_test = test_convergence(C_history)
        base_tests.append(convergence_test)
        
        # 8. TESTS SPÉCIFIQUES L2
        homogenization_test = test_homogenization(C_history)
        homogenization_tests[alpha] = homogenization_test
        
        diversity_test = test_structural_diversity_preservation(C_history)
        diversity_tests[alpha] = diversity_test
        
        if verbose:
            print(f"    {convergence_test.status.value:12} | Convergence")
            print(f"    {homogenization_test.status.value:12} | Homogenization")
            print(f"    {diversity_test.status.value:12} | Structural diversity")
        
        base_tests_all[alpha] = base_tests
    
    # 9. TEST GLOBAL : Alpha sensitivity
    if verbose:
        print(f"\n{'─'*70}")
        print("Test global : α-sensitivity")
        print(f"{'─'*70}")
    
    alpha_sensitivity_test = test_alpha_sensitivity(C_histories)
    
    if verbose:
        print(f"  {alpha_sensitivity_test.status.value:12} | {alpha_sensitivity_test.message}")
    
    # 10. VERDICT GLOBAL
    all_tests = []
    for tests_list in base_tests_all.values():
        all_tests.extend(tests_list)
    all_tests.append(alpha_sensitivity_test)
    all_tests.extend(diversity_tests.values())
    all_tests.extend(homogenization_tests.values())
    
    # Compte rejets bloquants
    blocking_failures = sum(1 for t in all_tests if t.blocking)
    
    if blocking_failures > 0:
        global_status = ValidationStatus.FAIL_BLOCKING
        summary = f"REJET : {blocking_failures} échec(s) bloquant(s)"
    elif alpha_sensitivity_test.status == ValidationStatus.FAIL_BLOCKING:
        global_status = ValidationStatus.FAIL_BLOCKING
        summary = "REJET : Aucune richesse dynamique (α insensible)"
    elif any(t.status == ValidationStatus.EXCELLENT for t in all_tests):
        global_status = ValidationStatus.INTERESTING
        summary = "PARTIEL : Certaines propriétés remarquables"
    else:
        global_status = ValidationStatus.PASS
        summary = "SURVIT aux tests de base"
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"VERDICT GLOBAL : {global_status.value}")
        print(f"{summary}")
        print(f"{'='*70}\n")
    
    return TM_GAM_002_Results(
        toy_model_id="TM-GAM-002",
        hypothesis_id="HYP-GAM-002",
        base_tests=base_tests_all,
        homogenization_tests=homogenization_tests,
        diversity_tests=diversity_tests,
        transitive_tests=transitive_tests,
        alpha_sensitivity=alpha_sensitivity_test,
        C_histories=C_histories,
        global_status=global_status,
        summary=summary
    )


# ============================================================================
# GÉNÉRATION AUTOMATIQUE LOG-GAM-002
# ============================================================================

def generate_log_entry(results: TM_GAM_002_Results) -> str:
    """
    Génère une entrée LOG-GAM-002 au format charte 4.0.
    
    Args:
        results: Résultats complets TM-GAM-002
        
    Returns:
        String formatée pour log.db
    """
    from datetime import datetime
    
    log_entry = f"""
\\LOG{{
  ID = LOG-GAM-002
  DATE = {datetime.now().strftime("%Y-%m-%d")}
  TYPE = test_output
  TARGET = TM-GAM-002
  
  RAW_OUTPUT = {{
    alpha_values = {ALPHA_VALUES_TO_TEST}
    iterations = {N_ITERATIONS_EXTENDED}
    
    # Tests par α
"""
    
    for alpha in ALPHA_VALUES_TO_TEST:
        tests = results.base_tests[alpha]
        diversity = results.diversity_tests[alpha]
        homog = results.homogenization_tests[alpha]
        
        log_entry += f"""
    alpha_{alpha} = {{
      symmetry_error = {tests[0].value:.3e}
      bounds_violation = {tests[1].value:.3f}
      min_eigenvalue = {tests[2].value:.3e}
      transitive_coupling = {tests[3].value:.4f}
      diversity_final = {diversity.value:.4f}
      diversity_loss = {1 - diversity.value:.2%}
      homogeneity_std = {homog.value:.4f}
    }}
"""
    
    log_entry += f"""
    # Test global
    alpha_sensitivity = {{
      mean_distance = {results.alpha_sensitivity.value:.4f}
      status = "{results.alpha_sensitivity.status.value}"
    }}
    
    # Verdict
    global_status = "{results.global_status.value}"
    summary = "{results.summary}"
  }}
}}
"""
    
    return log_entry


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TM-GAM-002 : Test HYP-GAM-002")
    parser.add_argument("--verbose", action="store_true", help="Affiche détails")
    parser.add_argument("--save-log", action="store_true", help="Sauvegarde LOG-GAM-002")
    
    args = parser.parse_args()
    
    # Exécute protocole
    results = run_tm_gam_002(verbose=args.verbose)
    
    # Sauvegarde log si demandé
    if args.save_log:
        log_entry = generate_log_entry(results)
        
        log_path = project_root / "prc_documentation" / "logs" / "log.db"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)
        
        print(f"\n✅ LOG-GAM-002 sauvegardé dans {log_path}")
    
    # Code de sortie
    sys.exit(0 if results.global_status != ValidationStatus.FAIL_BLOCKING else 1)