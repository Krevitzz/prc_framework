"""
PRC Framework - Operators Module

Bibliothèque d'opérateurs Γ standards pour le framework PRC.

PRINCIPE:
- Tous les opérateurs agissent DIRECTEMENT sur la matrice C
- AUCUNE présupposition physique
- Transformations mathématiques pures

CATÉGORIES:
1. Diffusion: Propagation de corrélations
2. Hebbian: Renforcement transitif
3. Stochastic: Perturbations aléatoires
4. Nonlinear: Transformations non-linéaires

Usage:
    >>> from prc.operators import PureDiffusion, Hebbian, CompositeOperator
    >>> 
    >>> gamma = CompositeOperator([
    ...     PureDiffusion(alpha=0.02),
    ...     Hebbian(beta=0.01)
    ... ])

Version: 1.0.0
"""

__version__ = "1.0.0"

# ============================================================================
# IMPORTS PAR CATÉGORIE
# ============================================================================

# Diffusion operators
from .diffusion import (
    PureDiffusionOperator,
    AdaptiveDiffusionOperator,
    MultiScaleDiffusionOperator,
    AnisotropicDiffusionOperator,
)

# Hebbian (transitivity reinforcement) operators
from .hebbian import (
    HebbianOperator,
    AntiHebbianOperator,
    NonlinearHebbianOperator,
    CompetitiveHebbianOperator,
    OjaRuleOperator,
    BCMRuleOperator,
)

# Stochastic operators
from .stochastic import (
    GaussianNoiseOperator,
    UniformNoiseOperator,
    LangevinOperator,
    CorrelatedNoiseOperator,
    AnisotropicNoiseOperator,
    OrnsteinUhlenbeckOperator,
)

# Nonlinear operators
from .nonlinear import (
    ThresholdOperator,
    SaturationOperator,
    PolynomialOperator,
    LogisticOperator,
    PowerLawOperator,
    RectifiedOperator,
    ContrastEnhancementOperator,
    NormalizationOperator,
    WinnerTakeAllOperator,
)


# ============================================================================
# ALIASES COURTS (pour usage fréquent)
# ============================================================================

# Diffusion
PureDiffusion = PureDiffusionOperator
AdaptiveDiffusion = AdaptiveDiffusionOperator
MultiScaleDiffusion = MultiScaleDiffusionOperator
AnisotropicDiffusion = AnisotropicDiffusionOperator

# Hebbian
Hebbian = HebbianOperator
AntiHebbian = AntiHebbianOperator
NonlinearHebbian = NonlinearHebbianOperator
CompetitiveHebbian = CompetitiveHebbianOperator
OjaRule = OjaRuleOperator
BCMRule = BCMRuleOperator

# Stochastic
GaussianNoise = GaussianNoiseOperator
UniformNoise = UniformNoiseOperator
Langevin = LangevinOperator
CorrelatedNoise = CorrelatedNoiseOperator
AnisotropicNoise = AnisotropicNoiseOperator
OrnsteinUhlenbeck = OrnsteinUhlenbeckOperator

# Nonlinear
Threshold = ThresholdOperator
Saturation = SaturationOperator
Polynomial = PolynomialOperator
Logistic = LogisticOperator
PowerLaw = PowerLawOperator
Rectified = RectifiedOperator
ContrastEnhancement = ContrastEnhancementOperator
Normalization = NormalizationOperator
WinnerTakeAll = WinnerTakeAllOperator


# ============================================================================
# API PUBLIQUE
# ============================================================================

__all__ = [
    # Diffusion
    "PureDiffusionOperator", "PureDiffusion",
    "AdaptiveDiffusionOperator", "AdaptiveDiffusion",
    "MultiScaleDiffusionOperator", "MultiScaleDiffusion",
    "AnisotropicDiffusionOperator", "AnisotropicDiffusion",
    
    # Hebbian
    "HebbianOperator", "Hebbian",
    "AntiHebbianOperator", "AntiHebbian",
    "NonlinearHebbianOperator", "NonlinearHebbian",
    "CompetitiveHebbianOperator", "CompetitiveHebbian",
    "OjaRuleOperator", "OjaRule",
    "BCMRuleOperator", "BCMRule",
    
    # Stochastic
    "GaussianNoiseOperator", "GaussianNoise",
    "UniformNoiseOperator", "UniformNoise",
    "LangevinOperator", "Langevin",
    "CorrelatedNoiseOperator", "CorrelatedNoise",
    "AnisotropicNoiseOperator", "AnisotropicNoise",
    "OrnsteinUhlenbeckOperator", "OrnsteinUhlenbeck",
    
    # Nonlinear
    "ThresholdOperator", "Threshold",
    "SaturationOperator", "Saturation",
    "PolynomialOperator", "Polynomial",
    "LogisticOperator", "Logistic",
    "PowerLawOperator", "PowerLaw",
    "RectifiedOperator", "Rectified",
    "ContrastEnhancementOperator", "ContrastEnhancement",
    "NormalizationOperator", "Normalization",
    "WinnerTakeAllOperator", "WinnerTakeAll",
]


# ============================================================================
# PRESETS COURANTS (combinaisons testées)
# ============================================================================

from core import CompositeOperator

def get_preset(name: str, **kwargs):
    """
    Retourne un opérateur preset.
    
    Presets disponibles:
    - "exploration": Diffusion + bruit (exploration large)
    - "consolidation": Hebbian + saturation (renforce structures)
    - "sparse": Threshold + WTA (sparsifie)
    - "balanced": Diffusion + Hebbian (équilibré)
    - "competitive": Competitive Hebbian + threshold (groupes distincts)
    
    Args:
        name: Nom du preset
        **kwargs: Paramètres à override
        
    Returns:
        EvolutionOperator configuré
        
    Example:
        >>> gamma = get_preset("balanced", alpha=0.02, beta=0.01)
    """
    if name == "exploration":
        alpha = kwargs.get("alpha", 0.02)
        sigma = kwargs.get("sigma", 0.01)
        return CompositeOperator([
            PureDiffusion(alpha=alpha),
            GaussianNoise(sigma=sigma),
        ])
    
    elif name == "consolidation":
        beta = kwargs.get("beta", 0.01)
        saturation_beta = kwargs.get("saturation_beta", 2.0)
        return CompositeOperator([
            Hebbian(beta=beta),
            Saturation(beta=saturation_beta),
        ])
    
    elif name == "sparse":
        threshold = kwargs.get("threshold", 0.2)
        k = kwargs.get("k", 3)
        return CompositeOperator([
            Threshold(threshold=threshold),
            WinnerTakeAll(k=k),
        ])
    
    elif name == "balanced":
        alpha = kwargs.get("alpha", 0.01)
        beta = kwargs.get("beta", 0.01)
        return CompositeOperator([
            PureDiffusion(alpha=alpha),
            Hebbian(beta=beta),
        ])
    
    elif name == "competitive":
        beta_win = kwargs.get("beta_win", 0.02)
        beta_lose = kwargs.get("beta_lose", 0.01)
        k_winners = kwargs.get("k_winners", 5)
        threshold = kwargs.get("threshold", 0.1)
        return CompositeOperator([
            CompetitiveHebbian(beta_win, beta_lose, k_winners),
            Threshold(threshold=threshold),
        ])
    
    else:
        raise ValueError(f"Preset inconnu: {name}")


# ============================================================================
# HELPERS POUR EXPLORATION
# ============================================================================

def list_operators() -> dict:
    """
    Liste tous les opérateurs disponibles par catégorie.
    
    Returns:
        Dictionnaire {catégorie: [opérateur1, opérateur2, ...]}
    """
    return {
        "Diffusion": [
            "PureDiffusion", "AdaptiveDiffusion",
            "MultiScaleDiffusion", "AnisotropicDiffusion"
        ],
        "Hebbian": [
            "Hebbian", "AntiHebbian", "NonlinearHebbian",
            "CompetitiveHebbian", "OjaRule", "BCMRule"
        ],
        "Stochastic": [
            "GaussianNoise", "UniformNoise", "Langevin",
            "CorrelatedNoise", "AnisotropicNoise", "OrnsteinUhlenbeck"
        ],
        "Nonlinear": [
            "Threshold", "Saturation", "Polynomial", "Logistic",
            "PowerLaw", "Rectified", "ContrastEnhancement",
            "Normalization", "WinnerTakeAll"
        ]
    }


def print_operators_info():
    """Affiche la liste des opérateurs disponibles."""
    print("\n" + "=" * 70)
    print("PRC OPERATORS - BIBLIOTHÈQUE DISPONIBLE")
    print("=" * 70)
    
    operators = list_operators()
    
    for category, ops in operators.items():
        print(f"\n{category}:")
        for op in ops:
            print(f"  - {op}")
    
    print("\n" + "=" * 70)
    print("Presets disponibles: exploration, consolidation, sparse,")
    print("                     balanced, competitive")
    print("=" * 70 + "\n")


# ============================================================================
# VALIDATION AU CHARGEMENT
# ============================================================================

def _validate_operators_imports():
    """Vérifie que tous les imports critiques ont réussi."""
    critical_operators = [
        PureDiffusionOperator,
        HebbianOperator,
        GaussianNoiseOperator,
        ThresholdOperator,
    ]
    
    for op_class in critical_operators:
        assert op_class is not None, f"Import échoué: {op_class.__name__}"


# Exécute la validation
_validate_operators_imports()


# ============================================================================
# QUICK START EXAMPLE
# ============================================================================

def quick_start_example():
    """
    Exemple minimal d'utilisation des opérateurs.
    """
    print("\n" + "=" * 70)
    print("PRC OPERATORS - QUICK START")
    print("=" * 70 + "\n")
    
    import numpy as np
    from core import InformationSpace, PRCKernel
    
    # 1. Créer un espace
    print("1. Création d'un InformationSpace...")
    D = InformationSpace.random(n_dof=20, mean=0.3, std=0.2, seed=42)
    print(f"   → Corrélation initiale: {np.mean(D.C[~np.eye(20, dtype=bool)]):.3f}")
    
    # 2. Tester différents opérateurs
    print("\n2. Test de différents opérateurs...\n")
    
    test_operators = [
        ("PureDiffusion", PureDiffusion(alpha=0.02)),
        ("Hebbian", Hebbian(beta=0.01)),
        ("Balanced (preset)", get_preset("balanced", alpha=0.02, beta=0.01)),
    ]
    
    for name, gamma in test_operators:
        kernel = PRCKernel(D.copy(), gamma)
        kernel.step(n_steps=50)
        summary = kernel.get_trajectory_summary()
        
        print(f"   {name:20} → corr finale: {summary['mean_correlation']:.3f}")
    
    print("\n" + "=" * 70)
    print("Pour plus d'infos: print_operators_info()")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Si le module est exécuté directement
    quick_start_example()