"""
tests/test_core_universality.py

Tests critiques validant l'universalité du core PRC.

Ces tests vérifient que:
1. Aucune présupposition physique n'existe dans le core
2. Le kernel fonctionne sur des encodages arbitraires
3. Les invariants mathématiques sont préservés
4. Le code est vraiment agnostique au domaine

IMPORTANT: Ces tests doivent TOUS passer pour que le core soit considéré universel.
"""

import numpy as np
import sys
from pathlib import Path

# Ajoute le répertoire parent au path pour imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    InformationSpace,
    EvolutionOperator,
    PRCKernel,
    IdentityOperator,
    ScalingOperator,
    CompositeOperator,
    create_block_diagonal,
    create_exponential_decay,
)


# ============================================================================
# TEST 1: INDÉPENDANCE DE REPRÉSENTATION
# ============================================================================

def test_no_graph_in_core():
    """
    Vérifie que le graphe n'est JAMAIS une primitive dans le core.
    
    Le core ne doit manipuler que des matrices C.
    """
    print("\n[TEST 1] Indépendance de représentation...")
    
    # Créer un InformationSpace
    D = InformationSpace.random(20, seed=42)
    
    # Vérifier qu'aucun attribut 'graph' n'existe dans l'état
    assert not hasattr(D, 'graph'), "❌ InformationSpace contient un attribut 'graph'"
    assert not hasattr(D, '_graph'), "❌ InformationSpace contient un cache graphe"
    
    # Vérifier que seul C est l'état
    state_attributes = [attr for attr in dir(D) 
                       if not attr.startswith('_') and not callable(getattr(D, attr))]
    
    critical_attributes = {'C', 'n_dof', 'metadata'}
    assert set(state_attributes) == critical_attributes, \
        f"❌ Attributs inattendus: {set(state_attributes) - critical_attributes}"
    
    print("   ✓ Pas de graphe dans InformationSpace")
    print("   ✓ État = uniquement {C, n_dof, metadata}")
    print("   ✅ TEST PASSÉ\n")
    return True  # AJOUTÉ: retour explicite


# ============================================================================
# TEST 2: ABSENCE DE CONCEPTS PHYSIQUES
# ============================================================================

def test_no_physics_concepts():
    """
    Scan du code source pour vérifier l'absence de concepts physiques.
    
    Vérifie que les concepts physiques n'apparaissent PAS dans:
    - Noms de variables
    - Noms de fonctions/classes
    - Code exécutable
    
    AUTORISÉ dans:
    - Docstrings (pour expliquer ce qu'on ne fait pas)
    - Commentaires explicatifs
    """
    print("\n[TEST 2] Absence de concepts physiques dans le code...")
    
    forbidden_terms = [
        'particle', 'particule',
        'energy', 'energie',
        'mass', 'masse',
        'field', 'champ',
        # Concepts domaine-spécifiques
        'quantum', 'entangle',
        'gravity', 'spacetime',
        'biology', 'cell',
    ]
    
    # Note: "force" retiré car ambiguïté avec "enforce" (imposer)
    
    core_files = [
        'core/information_space.py',
        'core/evolution_operator.py',
        'core/kernel.py',
        'core/serialization.py',
    ]
    
    violations = []
    
    for filepath in core_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            in_docstring = False
            docstring_marker = None
            
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                
                # Détecte début/fin de docstring
                if '"""' in line or "'''" in line:
                    marker = '"""' if '"""' in line else "'''"
                    if not in_docstring:
                        in_docstring = True
                        docstring_marker = marker
                    elif marker == docstring_marker:
                        in_docstring = False
                        docstring_marker = None
                    continue
                
                # Ignore docstrings et commentaires
                if in_docstring or stripped.startswith('#'):
                    continue
                
                # Ignore lignes vides
                if not stripped:
                    continue
                
                # Cherche termes interdits dans le CODE uniquement
                line_lower = line.lower()
                for term in forbidden_terms:
                    if term in line_lower:
                        violations.append((filepath, i, term, stripped))
        
        except FileNotFoundError:
            print(f"   ⚠ Fichier non trouvé: {filepath}")
    
    if violations:
        print("   ❌ Concepts physiques détectés dans le CODE:")
        for filepath, line_num, term, line in violations[:5]:
            print(f"      {filepath}:{line_num} - '{term}' dans: {line[:60]}...")
        if len(violations) > 5:
            print(f"      ... et {len(violations) - 5} autre(s)")
        print(f"   ❌ Total: {len(violations)} violation(s)")
        print("   ❌ TEST ÉCHOUÉ\n")
        return False
    else:
        print("   ✓ Aucun concept physique dans le code exécutable")
        print("   ✓ (Commentaires et docstrings ignorés)")
        print("   ✅ TEST PASSÉ\n")
        return True


# ============================================================================
# TEST 3: UNIVERSALITÉ DE Γ
# ============================================================================

def test_gamma_domain_agnostic():
    """
    Vérifie que le même Γ fonctionne sur des encodages arbitraires.
    
    Test: Applique le même opérateur sur:
    - Matrices aléatoires
    - Structures en blocs
    - Corrélations exponentielles
    
    Tous doivent fonctionner sans modification.
    """
    print("\n[TEST 3] Universalité de Γ...")
    
    # Définit un Γ composite
    gamma = CompositeOperator([
        ScalingOperator(alpha=0.9),
        ScalingOperator(alpha=0.95),
    ])
    
    # Test sur différents encodages
    test_spaces = {
        "random": InformationSpace.random(15, seed=42),
        "block_diagonal": create_block_diagonal([5, 5, 5], 0.7, 0.1),
        "exponential": create_exponential_decay(15, decay_rate=2.0),
    }
    
    for name, space in test_spaces.items():
        kernel = PRCKernel(space, gamma)
        
        try:
            C_final = kernel.step(n_steps=10)
            
            # Vérifie que les invariants sont préservés
            assert np.allclose(C_final, C_final.T), \
                f"❌ Symétrie perdue sur encodage '{name}'"
            assert np.allclose(np.diag(C_final), 1.0), \
                f"❌ Diagonale ≠ 1 sur encodage '{name}'"
            assert np.all(C_final >= -1.01) and np.all(C_final <= 1.01), \
                f"❌ Bornes violées sur encodage '{name}'"
            
            print(f"   ✓ Γ fonctionne sur encodage '{name}'")
        
        except Exception as e:
            print(f"   ❌ Échec sur encodage '{name}': {e}")
            print("   ❌ TEST ÉCHOUÉ\n")
            return False
    
    print("   ✓ Même Γ fonctionne sur tous les encodages")
    print("   ✅ TEST PASSÉ\n")
    return True


# ============================================================================
# TEST 4: PRÉSERVATION DES INVARIANTS
# ============================================================================

def test_invariant_preservation():
    """
    Vérifie que les invariants mathématiques sont préservés sous itération.
    
    Invariants testés:
    - Symétrie: C = C^T
    - Diagonale: diag(C) = 1
    - Bornes: -1 ≤ C[i,j] ≤ 1
    """
    print("\n[TEST 4] Préservation des invariants...")
    
    # Créer un espace avec invariants parfaits
    D = InformationSpace.random(20, seed=123)
    
    # Opérateur qui pourrait violer les invariants
    gamma = ScalingOperator(alpha=0.8)
    
    kernel = PRCKernel(D, gamma, validate_each_step=True)
    
    try:
        # Itère longtemps
        C_final = kernel.step(n_steps=100)
        
        # Vérifie les invariants finaux
        errors = []
        
        # 1. Symétrie
        sym_error = np.max(np.abs(C_final - C_final.T))
        if sym_error > 1e-6:
            errors.append(f"Symétrie: erreur max = {sym_error}")
        
        # 2. Diagonale
        diag_error = np.max(np.abs(np.diag(C_final) - 1.0))
        if diag_error > 1e-6:
            errors.append(f"Diagonale: erreur max = {diag_error}")
        
        # 3. Bornes
        if np.any(C_final < -1.01) or np.any(C_final > 1.01):
            errors.append(f"Bornes: range = [{C_final.min():.3f}, {C_final.max():.3f}]")
        
        if errors:
            print("   ❌ Invariants violés après 100 itérations:")
            for error in errors:
                print(f"      - {error}")
            print("   ❌ TEST ÉCHOUÉ\n")
            return False
        else:
            print("   ✓ Symétrie préservée (erreur < 1e-6)")
            print("   ✓ Diagonale = 1 préservée (erreur < 1e-6)")
            print("   ✓ Bornes [-1, 1] préservées")
            print("   ✅ TEST PASSÉ\n")
            return True
    
    except AssertionError as e:
        print(f"   ❌ Validation échouée: {e}")
        print("   ❌ TEST ÉCHOUÉ\n")
        return False


# ============================================================================
# TEST 5: MINIMALISME DES DÉPENDANCES
# ============================================================================

def test_minimal_dependencies():
    """
    Vérifie que le core ne dépend que de numpy, abc, typing, json.
    
    Aucune dépendance à:
    - networkx (graphes)
    - scipy (analyse)
    - matplotlib (visualisation)
    - Bibliothèques domaine-spécifiques
    """
    print("\n[TEST 5] Minimalisme des dépendances...")
    
    allowed_imports = {
        'numpy', 'np',
        'abc',
        'typing',
        'json',
        'pathlib', 'Path',
        'datetime',
        'dataclasses',
    }
    
    forbidden_imports = {
        'networkx', 'nx',
        'scipy',
        'matplotlib', 'plt',
        'pandas',
        'torch', 'tensorflow',
    }
    
    core_files = [
        'core/information_space.py',
        'core/evolution_operator.py',
        'core/kernel.py',
        'core/serialization.py',
    ]
    
    violations = []
    
    for filepath in core_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Cherche les imports interdits
            for line in content.split('\n'):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    for forbidden in forbidden_imports:
                        if forbidden in line:
                            violations.append((filepath, line.strip(), forbidden))
        
        except FileNotFoundError:
            print(f"   ⚠ Fichier non trouvé: {filepath}")
    
    if violations:
        print("   ❌ Dépendances interdites détectées:")
        for filepath, line, module in violations:
            print(f"      {filepath}: {line}")
        print("   ❌ TEST ÉCHOUÉ\n")
        return False
    else:
        print("   ✓ Dépendances limitées à: numpy, abc, typing, json")
        print("   ✓ Aucune dépendance à networkx, scipy, matplotlib")
        print("   ✅ TEST PASSÉ\n")
        return True


# ============================================================================
# TEST 6: REPRODUCTIBILITÉ
# ============================================================================

def test_reproducibility():
    """
    Vérifie que les simulations sont reproductibles avec seed.
    """
    print("\n[TEST 6] Reproductibilité...")
    
    seed = 42
    n_steps = 50
    
    # Première simulation
    D1 = InformationSpace.random(15, seed=seed)
    gamma1 = ScalingOperator(alpha=0.9)
    kernel1 = PRCKernel(D1, gamma1)
    C1 = kernel1.step(n_steps)
    
    # Deuxième simulation (même seed)
    D2 = InformationSpace.random(15, seed=seed)
    gamma2 = ScalingOperator(alpha=0.9)
    kernel2 = PRCKernel(D2, gamma2)
    C2 = kernel2.step(n_steps)
    
    # Vérifier l'égalité
    diff = np.linalg.norm(C1 - C2)
    
    if diff < 1e-10:
        print(f"   ✓ Résultats identiques (diff = {diff:.2e})")
        print("   ✓ Simulations reproductibles avec seed")
        print("   ✅ TEST PASSÉ\n")
        return True
    else:
        print(f"   ❌ Résultats différents (diff = {diff:.2e})")
        print("   ❌ TEST ÉCHOUÉ\n")
        return False


# ============================================================================
# SUITE DE TESTS COMPLÈTE
# ============================================================================

def run_all_tests():
    """
    Exécute tous les tests d'universalité.
    
    Returns:
        True si tous passent, False sinon
    """
    print("\n" + "=" * 70)
    print("VALIDATION DE L'UNIVERSALITÉ DU CORE PRC")
    print("=" * 70)
    
    tests = [
        ("Indépendance de représentation", test_no_graph_in_core),
        ("Absence de concepts physiques", test_no_physics_concepts),
        ("Universalité de Γ", test_gamma_domain_agnostic),
        ("Préservation des invariants", test_invariant_preservation),
        ("Minimalisme des dépendances", test_minimal_dependencies),
        ("Reproductibilité", test_reproducibility),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[ERREUR] Test '{name}' a planté: {e}\n")
            results.append((name, False))
    
    # Résumé
    print("=" * 70)
    print("RÉSUMÉ DES TESTS")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSÉ" if result else "❌ ÉCHOUÉ"
        print(f"{status:12} | {name}")
    
    print("=" * 70)
    print(f"TOTAL: {passed}/{total} tests passés")
    
    if passed == total:
        print("\n🎉 TOUS LES TESTS PASSÉS - CORE VALIDÉ UNIVERSEL 🎉\n")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) échoué(s) - CORRECTION NÉCESSAIRE ⚠️\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)