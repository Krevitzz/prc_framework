"""
tests/TM-GAM-001.py

Toy Model pour tester HYP-GAM-001 : Saturation pure comme mécanisme Γ.

APPROCHE HYBRIDE :
1. Réutilise les tests de base de test_gamma_protocol.py
2. Ajoute des tests spécifiques à la saturation pure
3. Protocole complet documenté pour génération LOG/RES

QUESTION CENTRALE (Section 0 du document GPT) :
    "Que reste-t-il quand Γ ne fait que borner ?"

MÉTADONNÉES :
    ID : TM-GAM-001
    LEVEL : L3
    ASSOCIATED_HYPOTHESIS : HYP-GAM-001
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
from operators.gamma_hyp_001 import PureSaturationGamma, get_operator_metadata

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
# MÉTADONNÉES TM-GAM-001
# ============================================================================

TOY_MODEL_METADATA = {
    "id": "TM-GAM-001",
    "level": "L3",
    "associated_hypothesis": "HYP-GAM-001",
    "status": "WIP",
    "operator": "PureSaturationGamma",
    "question": "Que reste-t-il quand Γ ne fait que borner ?",
}


# ============================================================================
# CONFIGURATION SPÉCIFIQUE
# ============================================================================

# Configuration standard (héritée de test_gamma_protocol)
N_DOF_STANDARD = 50
EPSILON_INIT = 0.05
SEED_STANDARD = 42
N_ITERATIONS_BASE = 1000

# Configuration spécifique HYP-GAM-001
BETA_VALUES_TO_TEST = [0.5, 1.0, 2.0, 5.0, 10.0]  # Balayage β
N_ITERATIONS_EXTENDED = 2000  # Pour observer saturation complète

# Seuils numériques explicites
TOL_DIVERSITY_LOSS = 0.1  # Diversité < 10% de l'initiale → échec
TOL_PLATEAU = 1e-4  # ||C_{n+1} - C_n|| < tol → plateau atteint
WINDOW_PLATEAU = 50  # Fenêtre d'observation pour plateau


# ============================================================================
# TESTS SPÉCIFIQUES HYP-GAM-001 (ajouts au protocole de base)
# ============================================================================

def test_beta_sensitivity(C_histories: Dict[float, List[np.ndarray]]) -> TestResult:
    """
    Test L2.3 : Sensibilité à β (bifurcation ?).
    
    Vérifie si différentes valeurs de β produisent des attracteurs
    qualitativement différents ou juste une convergence plus/moins rapide
    vers le même état.
    
    Args:
        C_histories: {beta: [C_0, C_1, ..., C_n]} pour chaque β testé
        
    Returns:
        TestResult avec verdict sur la richesse dynamique
    """
    # Compare états finaux pour chaque β
    final_states = {beta: history[-1] for beta, history in C_histories.items()}
    
    # Calcule distances entre tous les états finaux
    betas = list(final_states.keys())
    distances = []
    
    for i, beta_i in enumerate(betas):
        for beta_j in betas[i+1:]:
            dist = np.linalg.norm(final_states[beta_i] - final_states[beta_j])
            distances.append((beta_i, beta_j, dist))
    
    mean_distance = np.mean([d[2] for d in distances])
    max_distance = np.max([d[2] for d in distances])
    
    # Verdict
    if max_distance < 0.1:
        status = ValidationStatus.FAIL_BLOCKING
        message = f"Tous les β convergent vers le même attracteur (max_dist={max_distance:.3f})"
    elif mean_distance < 0.3:
        status = ValidationStatus.ALERT
        message = f"Faible différenciation entre β (mean_dist={mean_distance:.3f})"
    else:
        status = ValidationStatus.EXCELLENT
        message = f"Bifurcations détectées selon β (mean_dist={mean_distance:.3f})"
    
    return TestResult(
        property_name="Beta-sensitivity",
        status=status,
        value=mean_distance,
        threshold=0.3,
        message=message,
        blocking=(status == ValidationStatus.FAIL_BLOCKING)
    )


def test_structural_diversity_preservation(C_history: List[np.ndarray]) -> TestResult:
    """
    Test L2.2 : Préservation de la diversité structurelle.
    
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
    
    # Calcule évolution temporelle
    diversities = [compute_diversity(C) for C in C_history[::10]]  # Échantillonne
    
    # Vérifie monotonie décroissante
    is_monotone_decreasing = all(
        diversities[i] >= diversities[i+1] - 0.01  # Tolérance numérique
        for i in range(len(diversities)-1)
    )
    
    # Calcule perte relative
    diversity_loss = 1 - (diversity_final / (diversity_init + 1e-10))
    
    # Verdict
    if diversity_loss > (1 - TOL_DIVERSITY_LOSS):
        status = ValidationStatus.FAIL_BLOCKING
        message = f"Annihilation informationnelle ({diversity_loss*100:.1f}% perdue)"
    elif is_monotone_decreasing and diversity_loss > 0.5:
        status = ValidationStatus.ALERT
        message = f"Décroissance monotone forte ({diversity_loss*100:.1f}% perdue)"
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


def test_pointwise_independence(C_init: np.ndarray, 
                                C_final: np.ndarray) -> TestResult:
    """
    Test L1.3 : Absence de dépendance au contexte.
    
    Vérifie que C'[i,j] dépend UNIQUEMENT de C[i,j],
    pas du voisinage, spectre, ou structure globale.
    
    Pour saturation pure : tanh(β·C[i,j]) est fonction de C[i,j] seul.
    Test par reconstruction : peut-on prédire C_final élément par élément ?
    
    Args:
        C_init: Matrice initiale
        C_final: Matrice finale (après n itérations)
        
    Returns:
        TestResult confirmant indépendance pointwise
    """
    # Pour saturation itérée : C_n ≈ tanh(β·tanh(β·...tanh(β·C_0)))
    # À convergence : C_∞[i,j] = fonction monotone(C_0[i,j])
    
    n = C_init.shape[0]
    mask = ~np.eye(n, dtype=bool)
    
    # Extrait paires (C_init[i,j], C_final[i,j])
    init_vals = C_init[mask]
    final_vals = C_final[mask]
    
    # Vérifie monotonie : ordre préservé
    # Si C_init[i,j] < C_init[k,l], alors C_final[i,j] < C_final[k,l]
    violations = 0
    n_pairs = len(init_vals)
    
    # Échantillonne 1000 paires pour test
    n_samples = min(1000, n_pairs * (n_pairs - 1) // 2)
    
    for _ in range(n_samples):
        idx1, idx2 = np.random.choice(n_pairs, 2, replace=False)
        
        if init_vals[idx1] < init_vals[idx2]:
            if final_vals[idx1] >= final_vals[idx2]:
                violations += 1
    
    violation_rate = violations / n_samples
    
    # Verdict
    if violation_rate < 0.01:
        status = ValidationStatus.PASS
        message = f"Indépendance pointwise confirmée (violations: {violation_rate*100:.2f}%)"
    else:
        status = ValidationStatus.FAIL_BLOCKING
        message = f"Dépendance contextuelle détectée (violations: {violation_rate*100:.2f}%)"
    
    return TestResult(
        property_name="Pointwise independence",
        status=status,
        value=violation_rate,
        threshold=0.01,
        message=message,
        blocking=(status == ValidationStatus.FAIL_BLOCKING)
    )


# ============================================================================
# PROTOCOLE COMPLET TM-GAM-001
# ============================================================================

@dataclass
class TM_GAM_001_Results:
    """Résultats complets du toy model."""
    toy_model_id: str
    hypothesis_id: str
    
    # Tests de base (hérités)
    base_tests: Dict[float, List[TestResult]]  # {beta: [tests]}
    
    # Tests spécifiques
    beta_sensitivity: TestResult
    diversity_tests: Dict[float, TestResult]  # {beta: test}
    pointwise_tests: Dict[float, TestResult]  # {beta: test}
    
    # Données brutes
    C_histories: Dict[float, List[np.ndarray]]
    
    # Verdict global
    global_status: ValidationStatus
    summary: str


def run_tm_gam_001(verbose: bool = True) -> TM_GAM_001_Results:
    """
    Exécute le protocole complet pour TM-GAM-001.
    
    PROTOCOLE HYBRIDE :
    
    Niveau L1 (Invariants formels) :
    ├─ test_symmetry (base)
    ├─ test_bounds (base)
    ├─ test_positivity (base)
    └─ test_pointwise_independence (spécifique) ✨
    
    Niveau L2 (Dynamique itérée) :
    ├─ test_convergence (base)
    ├─ test_structural_diversity_preservation (spécifique) ✨
    └─ test_beta_sensitivity (spécifique) ✨
    
    Args:
        verbose: Affiche progression si True
        
    Returns:
        TM_GAM_001_Results avec tous les résultats
    """
    if verbose:
        print("\n" + "="*70)
        print("TM-GAM-001 : Test de HYP-GAM-001 (Saturation pure)")
        print("="*70)
        print(f"\nQuestion centrale : {TOY_MODEL_METADATA['question']}")
        print(f"\nConfiguration :")
        print(f"  N_DOF = {N_DOF_STANDARD}")
        print(f"  ε = {EPSILON_INIT}")
        print(f"  β values = {BETA_VALUES_TO_TEST}")
        print(f"  Iterations = {N_ITERATIONS_EXTENDED}")
    
    # Stockage résultats
    base_tests_all = {}
    diversity_tests = {}
    pointwise_tests = {}
    C_histories = {}
    
    # Pour chaque valeur de β
    for beta in BETA_VALUES_TO_TEST:
        if verbose:
            print(f"\n{'─'*70}")
            print(f"Testing β = {beta}")
            print(f"{'─'*70}")
        
        # 1. Crée état initial standard
        D_init = create_standard_initial_state(
            n_dof=N_DOF_STANDARD,
            epsilon=EPSILON_INIT,
            seed=SEED_STANDARD
        )
        
        # 2. Crée opérateur
        gamma = PureSaturationGamma(beta=beta)
        
        # 3. Simule
        kernel = PRCKernel(D_init, gamma)
        kernel.start_recording()
        kernel.step(N_ITERATIONS_EXTENDED)
        
        # 4. Extrait historique
        history = kernel.get_history()
        C_history = [state.C for state in history]
        C_histories[beta] = C_history
        
        # 5. TESTS DE BASE (réutilisés)
        if verbose:
            print("\n  Tests L1 (invariants) :")
        
        base_tests = [
            test_symmetry(C_history),
            test_bounds(C_history),
            test_positivity(C_history),
        ]
        
        # 6. TEST SPÉCIFIQUE L1 : Pointwise independence
        pointwise_test = test_pointwise_independence(C_history[0], C_history[-1])
        pointwise_tests[beta] = pointwise_test
        base_tests.append(pointwise_test)
        
        if verbose:
            for test in base_tests:
                print(f"    {test.status.value:12} | {test.property_name}")
        
        # 7. TESTS L2
        if verbose:
            print("\n  Tests L2 (dynamique) :")
        
        convergence_test = test_convergence(C_history)
        base_tests.append(convergence_test)
        
        # 8. TEST SPÉCIFIQUE L2 : Structural diversity
        diversity_test = test_structural_diversity_preservation(C_history)
        diversity_tests[beta] = diversity_test
        
        if verbose:
            print(f"    {convergence_test.status.value:12} | Convergence")
            print(f"    {diversity_test.status.value:12} | Structural diversity")
        
        base_tests_all[beta] = base_tests
    
    # 9. TEST GLOBAL : Beta sensitivity
    if verbose:
        print(f"\n{'─'*70}")
        print("Test global : β-sensitivity")
        print(f"{'─'*70}")
    
    beta_sensitivity_test = test_beta_sensitivity(C_histories)
    
    if verbose:
        print(f"  {beta_sensitivity_test.status.value:12} | {beta_sensitivity_test.message}")
    
    # 10. VERDICT GLOBAL
    all_tests = []
    for tests_list in base_tests_all.values():
        all_tests.extend(tests_list)
    all_tests.append(beta_sensitivity_test)
    all_tests.extend(diversity_tests.values())
    
    # Compte rejets bloquants
    blocking_failures = sum(1 for t in all_tests if t.blocking)
    
    if blocking_failures > 0:
        global_status = ValidationStatus.FAIL_BLOCKING
        summary = f"REJET : {blocking_failures} échec(s) bloquant(s)"
    elif beta_sensitivity_test.status == ValidationStatus.FAIL_BLOCKING:
        global_status = ValidationStatus.FAIL_BLOCKING
        summary = "REJET : Aucune richesse dynamique (β insensible)"
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
    
    return TM_GAM_001_Results(
        toy_model_id="TM-GAM-001",
        hypothesis_id="HYP-GAM-001",
        base_tests=base_tests_all,
        beta_sensitivity=beta_sensitivity_test,
        diversity_tests=diversity_tests,
        pointwise_tests=pointwise_tests,
        C_histories=C_histories,
        global_status=global_status,
        summary=summary
    )


# ============================================================================
# GÉNÉRATION AUTOMATIQUE LOG-GAM-001
# ============================================================================

def generate_log_entry(results: TM_GAM_001_Results) -> str:
    """
    Génère une entrée LOG-GAM-001 au format charte 4.0.
    
    Args:
        results: Résultats complets TM-GAM-001
        
    Returns:
        String formatée pour log.db
    """
    from datetime import datetime
    
    log_entry = f"""
\\LOG{{
  ID = LOG-GAM-001
  DATE = {datetime.now().strftime("%Y-%m-%d")}
  TYPE = test_output
  TARGET = TM-GAM-001
  
  RAW_OUTPUT = {{
    beta_values = {BETA_VALUES_TO_TEST}
    iterations = {N_ITERATIONS_EXTENDED}
    
    # Tests par β
"""
    
    for beta in BETA_VALUES_TO_TEST:
        tests = results.base_tests[beta]
        diversity = results.diversity_tests[beta]
        
        log_entry += f"""
    beta_{beta} = {{
      symmetry_error = {tests[0].value:.3e}
      bounds_violation = {tests[1].value:.3f}
      min_eigenvalue = {tests[2].value:.3e}
      diversity_final = {diversity.value:.4f}
      diversity_loss = {1 - diversity.value:.2%}
    }}
"""
    
    log_entry += f"""
    # Test global
    beta_sensitivity = {{
      mean_distance = {results.beta_sensitivity.value:.4f}
      status = "{results.beta_sensitivity.status.value}"
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
    
    parser = argparse.ArgumentParser(description="TM-GAM-001 : Test HYP-GAM-001")
    parser.add_argument("--verbose", action="store_true", help="Affiche détails")
    parser.add_argument("--save-log", action="store_true", help="Sauvegarde LOG-GAM-001")
    
    args = parser.parse_args()
    
    # Exécute protocole
    results = run_tm_gam_001(verbose=args.verbose)
    
    # Sauvegarde log si demandé
    if args.save_log:
        log_entry = generate_log_entry(results)
        
        log_path = project_root / "prc_documentation" / "logs" / "log.db"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, "a") as f:
            f.write(log_entry)
        
        print(f"\n✅ LOG-GAM-001 sauvegardé dans {log_path}")
    
    # Code de sortie
    sys.exit(0 if results.global_status != ValidationStatus.FAIL_BLOCKING else 1)
```

**Caractéristiques de l'approche hybride** :

✅ **Réutilisation propre** :
- Import direct de `test_gamma_protocol.py`
- Pas de duplication de code
- Tests de base gardés tels quels

✅ **Ajouts cohérents** :
- 3 nouveaux tests alignés sur le document GPT
- Métriques quantitatives explicites (seuils, formules)
- Format conforme charte 4.0

✅ **Architecture claire** :
```
Tests L1 (invariants)
├─ Hérités : symmetry, bounds, positivity
└─ Nouveau : pointwise_independence ✨

Tests L2 (dynamique)
├─ Hérité : convergence
└─ Nouveaux : diversity, beta_sensitivity ✨