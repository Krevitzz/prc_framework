"""
PRC Framework - Core Module

Ce module contient le kernel universel du framework PRC.
AUCUNE présupposition physique n'est faite à ce niveau.

Composants:
- InformationSpace: Représentation pure de D (matrice de corrélation)
- EvolutionOperator: Interface abstraite pour Γ
- PRCKernel: Moteur de simulation (itération pure de Γ sur C)
- Serialization: Import/export JSON standardisé

Usage minimal:
    >>> from prc.core import InformationSpace, PRCKernel, IdentityOperator
    >>> 
    >>> # 1. Créer un espace d'information
    >>> D = InformationSpace.identity(n_dof=20)
    >>> 
    >>> # 2. Définir un opérateur (ici: identité pour test)
    >>> gamma = IdentityOperator()
    >>> 
    >>> # 3. Simuler
    >>> kernel = PRCKernel(D, gamma)
    >>> C_final = kernel.step(n_steps=100)

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "PRC Framework Team"

# ============================================================================
# IMPORTS PRINCIPAUX
# ============================================================================

# InformationSpace
from .information_space import (
    InformationSpace,
    create_block_diagonal,
    create_exponential_decay,
    create_power_law,
)

# EvolutionOperator
from .evolution_operator import (
    EvolutionOperator,
    IdentityOperator,
    ScalingOperator,
    CompositeOperator,
    ConstrainedOperator,
    # Utilitaires de contraintes
    enforce_symmetry,
    enforce_diagonal,
    enforce_bounds,
    enforce_correlation_invariants,
)

# PRCKernel
from .kernel import (
    PRCKernel,
    SimulationState,
    # Utilitaires de simulation
    run_until_convergence,
    run_parallel_simulations,
    compare_operators,
)

# Serialization
from .serialization import (
    save_information_space,
    load_information_space,
    save_simulation_state,
    load_simulation_state,
    save_simulation_history,
    load_simulation_history,
    load_experiment_spec,
    save_experiment_spec,
    validate_format,
    get_format_info,
)


# ============================================================================
# API PUBLIQUE (ce qui est exposé via `from prc.core import *`)
# ============================================================================

__all__ = [
    # Version
    "__version__",
    
    # InformationSpace
    "InformationSpace",
    "create_block_diagonal",
    "create_exponential_decay",
    "create_power_law",
    
    # EvolutionOperator
    "EvolutionOperator",
    "IdentityOperator",
    "ScalingOperator",
    "CompositeOperator",
    "ConstrainedOperator",
    "enforce_symmetry",
    "enforce_diagonal",
    "enforce_bounds",
    "enforce_correlation_invariants",
    
    # PRCKernel
    "PRCKernel",
    "SimulationState",
    "run_until_convergence",
    "run_parallel_simulations",
    "compare_operators",
    
    # Serialization
    "save_information_space",
    "load_information_space",
    "save_simulation_state",
    "load_simulation_state",
    "save_simulation_history",
    "load_simulation_history",
    "load_experiment_spec",
    "save_experiment_spec",
    "validate_format",
    "get_format_info",
]


# ============================================================================
# VALIDATION AU CHARGEMENT
# ============================================================================

def _validate_core_imports():
    """
    Vérifie que tous les imports critiques ont réussi.
    
    Appelé automatiquement à l'import du module.
    """
    critical_classes = [
        InformationSpace,
        EvolutionOperator,
        PRCKernel,
    ]
    
    for cls in critical_classes:
        assert cls is not None, f"Import critique échoué: {cls.__name__}"


# Exécute la validation
_validate_core_imports()


# ============================================================================
# HELPERS POUR DEBUGGING
# ============================================================================

def get_core_info() -> dict:
    """
    Retourne des informations sur le core module.
    
    Utile pour diagnostique et support.
    
    Returns:
        Dictionnaire avec métadonnées du module
    """
    import sys
    import numpy as np
    
    return {
        "version": __version__,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "module_path": __file__,
        "exported_symbols": len(__all__),
        "classes": [
            "InformationSpace",
            "EvolutionOperator",
            "PRCKernel",
        ],
        "utilities": [
            "create_block_diagonal",
            "create_exponential_decay",
            "create_power_law",
            "run_until_convergence",
            "run_parallel_simulations",
            "compare_operators",
        ],
    }


def print_core_info():
    """Affiche les informations du core de manière lisible."""
    info = get_core_info()
    
    print("=" * 60)
    print("PRC FRAMEWORK - CORE MODULE")
    print("=" * 60)
    print(f"Version:        {info['version']}")
    print(f"Python:         {info['python_version']}")
    print(f"NumPy:          {info['numpy_version']}")
    print(f"Module path:    {info['module_path']}")
    print()
    print("Classes principales:")
    for cls in info['classes']:
        print(f"  - {cls}")
    print()
    print("Utilitaires disponibles:")
    for util in info['utilities']:
        print(f"  - {util}")
    print("=" * 60)


# ============================================================================
# QUICK START EXAMPLE
# ============================================================================

def quick_start_example():
    """
    Exemple minimal d'utilisation du core.
    
    Démonstration de la chaîne complète:
    1. Créer un InformationSpace
    2. Définir un EvolutionOperator
    3. Simuler avec PRCKernel
    4. Analyser les résultats
    """
    print("\n" + "=" * 60)
    print("PRC CORE - QUICK START EXAMPLE")
    print("=" * 60 + "\n")
    
    # 1. Créer un espace d'information simple
    print("1. Création d'un InformationSpace...")
    D = InformationSpace.random(n_dof=10, mean=0.3, std=0.2, seed=42)
    print(f"   → {D}")
    print(f"   → Corrélation moyenne: {D.C[~np.eye(10, dtype=bool)].mean():.3f}")
    
    # 2. Définir un opérateur simple
    print("\n2. Définition d'un opérateur (Scaling)...")
    gamma = ScalingOperator(alpha=0.95)
    print(f"   → {gamma}")
    
    # 3. Simuler
    print("\n3. Simulation...")
    kernel = PRCKernel(D, gamma)
    C_final = kernel.step(n_steps=50)
    print(f"   → {kernel.iteration} itérations complétées")
    
    # 4. Résultats
    print("\n4. Résultats:")
    summary = kernel.get_trajectory_summary()
    print(f"   → Corrélation finale: {summary['mean_correlation']:.3f}")
    print(f"   → Écart-type: {summary['std_correlation']:.3f}")
    print(f"   → Rang: {summary['rank']}")
    
    print("\n" + "=" * 60)
    print("Simulation terminée avec succès!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Si le module est exécuté directement, affiche l'exemple
    quick_start_example()