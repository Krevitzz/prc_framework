"""
Exemple d'utilisation des générateurs D^(base) et tests de vérification.

Ce script montre comment:
1. Générer toutes les bases du catalogue
2. Vérifier leurs propriétés
3. Produire un rapport complet
"""

# Imports
from D_encodings.rank2_symmetric import (
    create_identity,
    create_random_uniform,
    create_random_gaussian,
    create_correlation_matrix,
    create_banded,
    create_block_hierarchical
)

from D_encodings.rank2_asymmetric import (
    create_random_asymmetric,
    create_lower_triangular,
    create_antisymmetric,
    create_directional_gradient
)

from D_encodings.rank3_correlations import (
    create_random_rank3,
    create_partial_symmetric_rank3,
    create_local_coupling_rank3
)

from D_encodings.verification_tests import (
    run_verification_suite,
    print_verification_report
)


# ============================================================================
# CATALOGUE COMPLET DES BASES
# ============================================================================

def test_all_symmetric_bases(n_dof: int = 50):
    """Teste toutes les bases symétriques."""
    
    print("\n" + "="*70)
    print("BASES SYMÉTRIQUES RANG 2")
    print("="*70)
    
    # D-SYM-001: Identité
    results = run_verification_suite(
        create_identity,
        {'n_dof': n_dof},
        ['shape_2d', 'symmetry', 'diagonal_ones', 'bounds', 
         'positive_definite', 'reproducibility'],
        name="SYM-001"
    )
    print_verification_report(results, "D-SYM-001: Identité")
    
    # D-SYM-002: Aléatoire uniforme
    results = run_verification_suite(
        create_random_uniform,
        {'n_dof': n_dof, 'seed': 42},
        ['shape_2d', 'symmetry', 'bounds', 'non_constant', 'reproducibility'],
        name="SYM-002"
    )
    print_verification_report(results, "D-SYM-002: Aléatoire uniforme")
    
    # D-SYM-003: Aléatoire gaussienne
    results = run_verification_suite(
        create_random_gaussian,
        {'n_dof': n_dof, 'sigma': 0.3, 'seed': 42},
        ['shape_2d', 'symmetry', 'non_constant', 'reproducibility'],
        name="SYM-003"
    )
    print_verification_report(results, "D-SYM-003: Aléatoire gaussienne")
    
    # D-SYM-004: Matrice de corrélation
    results = run_verification_suite(
        create_correlation_matrix,
        {'n_dof': n_dof, 'seed': 42},
        ['shape_2d', 'symmetry', 'diagonal_ones', 'positive_definite', 
         'reproducibility'],
        name="SYM-004"
    )
    print_verification_report(results, "D-SYM-004: Corrélation (SPD)")
    
    # D-SYM-005: Bande
    results = run_verification_suite(
        create_banded,
        {'n_dof': n_dof, 'bandwidth': 3, 'seed': 42},
        ['shape_2d', 'symmetry', 'diagonal_ones', 'banded_3', 'reproducibility'],
        name="SYM-005"
    )
    print_verification_report(results, "D-SYM-005: Bande")
    
    # D-SYM-006: Hiérarchique par blocs
    results = run_verification_suite(
        create_block_hierarchical,
        {'n_dof': n_dof, 'n_blocks': 10, 'seed': 42},
        ['shape_2d', 'symmetry', 'diagonal_ones', 'bounds', 'reproducibility'],
        name="SYM-006"
    )
    print_verification_report(results, "D-SYM-006: Hiérarchique")


def test_all_asymmetric_bases(n_dof: int = 50):
    """Teste toutes les bases asymétriques."""
    
    print("\n" + "="*70)
    print("BASES ASYMÉTRIQUES RANG 2")
    print("="*70)
    
    # D-ASY-001: Aléatoire uniforme
    results = run_verification_suite(
        create_random_asymmetric,
        {'n_dof': n_dof, 'seed': 42},
        ['shape_2d', 'bounds', 'non_constant', 'reproducibility'],
        name="ASY-001"
    )
    print_verification_report(results, "D-ASY-001: Asymétrique uniforme")
    
    # D-ASY-002: Triangulaire inférieure
    results = run_verification_suite(
        create_lower_triangular,
        {'n_dof': n_dof, 'seed': 42},
        ['shape_2d', 'bounds', 'triangular_lower', 'reproducibility'],
        name="ASY-002"
    )
    print_verification_report(results, "D-ASY-002: Triangulaire inférieure")
    
    # D-ASY-003: Antisymétrique
    results = run_verification_suite(
        create_antisymmetric,
        {'n_dof': n_dof, 'seed': 42},
        ['shape_2d', 'antisymmetry', 'bounds', 'reproducibility'],
        name="ASY-003"
    )
    print_verification_report(results, "D-ASY-003: Antisymétrique")
    
    # D-ASY-004: Gradient directionnel
    results = run_verification_suite(
        create_directional_gradient,
        {'n_dof': n_dof, 'gradient': 0.1, 'seed': 42},
        ['shape_2d', 'non_constant', 'reproducibility'],
        name="ASY-004"
    )
    print_verification_report(results, "D-ASY-004: Gradient directionnel")


def test_all_rank3_bases(n_dof: int = 20):
    """Teste toutes les bases rang 3."""
    
    print("\n" + "="*70)
    print("BASES RANG 3")
    print("="*70)
    
    # D-R3-001: Aléatoire uniforme
    results = run_verification_suite(
        create_random_rank3,
        {'n_dof': n_dof, 'seed': 42},
        ['shape_3d', 'bounds', 'non_constant', 'reproducibility'],
        name="R3-001"
    )
    print_verification_report(results, "D-R3-001: Rang 3 aléatoire")
    
    # D-R3-002: Symétrie partielle
    results = run_verification_suite(
        create_partial_symmetric_rank3,
        {'n_dof': n_dof, 'seed': 42},
        ['shape_3d', 'bounds', 'partial_symmetry_23', 'reproducibility'],
        name="R3-002"
    )
    print_verification_report(results, "D-R3-002: Symétrie partielle")
    
    # D-R3-003: Couplages locaux
    results = run_verification_suite(
        create_local_coupling_rank3,
        {'n_dof': n_dof, 'radius': 2, 'seed': 42},
        ['shape_3d', 'bounds', 'reproducibility'],
        name="R3-003"
    )
    print_verification_report(results, "D-R3-003: Couplages locaux")


# ============================================================================
# GÉNÉRATION RAPPORT COMPLET
# ============================================================================

def generate_full_catalog_report():
    """Génère un rapport complet de tous les générateurs."""
    
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + " "*15 + "RAPPORT DE VÉRIFICATION COMPLET" + " "*22 + "#")
    print("#" + " "*20 + "Catalogue D^(base)" + " "*28 + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    # Tester toutes les familles
    test_all_symmetric_bases(n_dof=50)
    test_all_asymmetric_bases(n_dof=50)
    test_all_rank3_bases(n_dof=20)  # Plus petit pour rang 3
    
    print("\n" + "#"*70)
    print("#" + " "*25 + "FIN DU RAPPORT" + " "*29 + "#")
    print("#"*70 + "\n")


# ============================================================================
# EXEMPLE D'UTILISATION DANS UN TM
# ============================================================================

def example_tm_usage():
    """
    Exemple d'utilisation dans un Toy Model.
    
    Montre comment:
    1. Charger une base du catalogue
    2. Vérifier ses propriétés
    3. L'utiliser avec prepare_state
    """
    from core.state_preparation import prepare_state
    from modifiers.noise import add_gaussian_noise
    
    print("\n" + "="*70)
    print("EXEMPLE: Utilisation dans un TM")
    print("="*70)
    
    # 1. Choisir une base (par exemple D-SYM-002)
    print("\nÉtape 1: Génération D^(base)")
    D_base = create_random_uniform(n_dof=50, seed=42)
    print(f"✓ D^(base) généré: shape={D_base.shape}")
    
    # 2. Vérification rapide
    print("\nÉtape 2: Vérification propriétés")
    results = run_verification_suite(
        create_random_uniform,
        {'n_dof': 50, 'seed': 42},
        ['shape_2d', 'symmetry', 'bounds'],
        name="example"
    )
    for result in results.values():
        print(f"  {result}")
    
    # 3. Application modifiers (optionnel)
    print("\nÉtape 3: Application modifiers")
    D_final = prepare_state(D_base, [
        add_gaussian_noise(sigma=0.05, seed=123)
    ])
    print(f"✓ D^(final) préparé avec bruit")
    
    # 4. Prêt pour exécution avec Γ
    print("\nÉtape 4: Prêt pour kernel")
    print(f"✓ D^(final) peut être passé à run_kernel(D_final, gamma, ...)")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test et vérification des générateurs D^(base)"
    )
    parser.add_argument(
        '--mode',
        choices=['full', 'symmetric', 'asymmetric', 'rank3', 'example'],
        default='full',
        help='Mode de test'
    )
    parser.add_argument(
        '--n_dof',
        type=int,
        default=50,
        help='Dimension des tenseurs'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        generate_full_catalog_report()
    
    elif args.mode == 'symmetric':
        test_all_symmetric_bases(args.n_dof)
    
    elif args.mode == 'asymmetric':
        test_all_asymmetric_bases(args.n_dof)
    
    elif args.mode == 'rank3':
        test_all_rank3_bases(min(args.n_dof, 30))  # Limite pour rang 3
    
    elif args.mode == 'example':
        example_tm_usage()
    
    print("\n✓ Tests terminés\n")