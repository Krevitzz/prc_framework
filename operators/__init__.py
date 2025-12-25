# operators/__init__.py

"""
Package operators - Définitions Γ pour toutes les hypothèses R0

Ce package contient les 14 opérateurs Γ pour l'exploration R0 exhaustive.

PRINCIPE FONDAMENTAL:
Chaque Γ est AVEUGLE - ne connaît ni la dimension, ni la structure,
ni l'interprétation de l'état qu'il manipule.

CATALOGUE R0 (14 hypothèses):
Famille Markovienne:
  - GAM-001: Saturation pure pointwise ✅
  - GAM-002: Diffusion pure ✅
  - GAM-003: Croissance exponentielle ✅
  - GAM-004: Décroissance exponentielle (✅
  - GAM-005: Oscillateur harmonique ✅

Famille Non-markovienne:
  - GAM-006: Saturation + mémoire ordre-1 ✅
  - GAM-007: Régulation moyenne glissante ✅
  - GAM-008: Mémoire différentielle ✅

Famille Stochastique:
  - GAM-009: Saturation + bruit additif ✅
  - GAM-010: Bruit multiplicatif ✅
  - GAM-011: Branchement tensoriel (TODO)

Famille Structurelle:
  - GAM-012: Préservation symétrie forcée ✅
  - GAM-013: Renforcement hebbien ✅
  - GAM-014: Projection sous-espace (TODO)


STATUT: 10/14 opérateurs implémentés
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Famille Markovienne
from .gamma_hyp_001 import (
    PureSaturationGamma,
    create_gamma_hyp_001,
    PARAM_GRID_PHASE1 as PARAM_GRID_PHASE1_001,
    PARAM_GRID_PHASE2 as PARAM_GRID_PHASE2_001,
    METADATA as METADATA_GAM_001,
)

from .gamma_hyp_002 import (
    PureDiffusionGamma,
    create_gamma_hyp_002,
    PARAM_GRID_PHASE1 as PARAM_GRID_PHASE1_002,
    PARAM_GRID_PHASE2 as PARAM_GRID_PHASE2_002,
    METADATA as METADATA_GAM_002,
)

from .gamma_hyp_003 import (
    ExponentialGrowthGamma,
    create_gamma_hyp_003,
    PARAM_GRID_PHASE1 as PARAM_GRID_PHASE1_003,
    PARAM_GRID_PHASE2 as PARAM_GRID_PHASE2_003,
    METADATA as METADATA_GAM_003,
)

from .gamma_hyp_004 import (
    ExponentialDecayGamma,
    create_gamma_hyp_004,
    PARAM_GRID_PHASE1 as PARAM_GRID_PHASE1_004,
    PARAM_GRID_PHASE2 as PARAM_GRID_PHASE2_004,
    METADATA as METADATA_GAM_004,
)

from .gamma_hyp_005 import (
    HarmonicOscillatorGamma,
    create_gamma_hyp_005,
    PARAM_GRID_PHASE1 as PARAM_GRID_PHASE1_005,
    PARAM_GRID_PHASE2 as PARAM_GRID_PHASE2_005,
    METADATA as METADATA_GAM_005,
)

# Famille Non-markovienne
from .gamma_hyp_006 import (
    MemorySaturationGamma,
    create_gamma_hyp_006,
    PARAM_GRID_PHASE1 as PARAM_GRID_PHASE1_006,
    PARAM_GRID_PHASE2 as PARAM_GRID_PHASE2_006,
    METADATA as METADATA_GAM_006,
)

from .gamma_hyp_007 import (
    SlidingAverageGamma,
    create_gamma_hyp_007,
    PARAM_GRID_PHASE1 as PARAM_GRID_PHASE1_007,
    PARAM_GRID_PHASE2 as PARAM_GRID_PHASE2_007,
    METADATA as METADATA_GAM_007,
)

from .gamma_hyp_008 import (
    DifferentialMemoryGamma,
    create_gamma_hyp_008,
    PARAM_GRID_PHASE1 as PARAM_GRID_PHASE1_008,
    PARAM_GRID_PHASE2 as PARAM_GRID_PHASE2_008,
    METADATA as METADATA_GAM_008,
)

# Famille Stochastique
from .gamma_hyp_009 import (
    StochasticSaturationGamma,
    create_gamma_hyp_009,
    PARAM_GRID_PHASE1 as PARAM_GRID_PHASE1_009,
    PARAM_GRID_PHASE2 as PARAM_GRID_PHASE2_009,
    METADATA as METADATA_GAM_009,
)

from .gamma_hyp_010 import (
    MultiplicativeNoiseGamma,
    create_gamma_hyp_010,
    PARAM_GRID_PHASE1 as PARAM_GRID_PHASE1_010,
    PARAM_GRID_PHASE2 as PARAM_GRID_PHASE2_010,
    METADATA as METADATA_GAM_010,
)

# Famille Structurelle
from .gamma_hyp_012 import (
    ForcedSymmetryGamma,
    create_gamma_hyp_012,
    PARAM_GRID_PHASE1 as PARAM_GRID_PHASE1_012,
    PARAM_GRID_PHASE2 as PARAM_GRID_PHASE2_012,
    METADATA as METADATA_GAM_012,
)

from .gamma_hyp_013 import (
    HebbianReinforcementGamma,
    create_gamma_hyp_013,
    PARAM_GRID_PHASE1 as PARAM_GRID_PHASE1_013,
    PARAM_GRID_PHASE2 as PARAM_GRID_PHASE2_013,
    METADATA as METADATA_GAM_013,
)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Markoviens
    'PureSaturationGamma',
    'create_gamma_hyp_001',
    'PureDiffusionGamma',
    'create_gamma_hyp_002',
    'ExponentialGrowthGamma',
    'create_gamma_hyp_003',
    'ExponentialDecayGamma',
    'create_gamma_hyp_004',
    'HarmonicOscillatorGamma',
    'create_gamma_hyp_005',
    
    # Non-markoviens
    'MemorySaturationGamma',
    'create_gamma_hyp_006',
    'SlidingAverageGamma',
    'create_gamma_hyp_007',
    'DifferentialMemoryGamma',
    'create_gamma_hyp_008',
    
    # Stochastiques
    'StochasticSaturationGamma',
    'create_gamma_hyp_009',
    'MultiplicativeNoiseGamma',
    'create_gamma_hyp_010',
    
    # Structurels
    'ForcedSymmetryGamma',
    'create_gamma_hyp_012',
    'HebbianReinforcementGamma',
    'create_gamma_hyp_013',
]

__version__ = '0.7.0'  # 10/14 implémentés
__author__ = 'PRC Operators Team'
__description__ = 'Opérateurs Γ pour exploration R0'


# ============================================================================
# REGISTRE DES OPÉRATEURS
# ============================================================================

OPERATOR_REGISTRY = {
    # Famille Markovienne (5 opérateurs)
    'GAM-001': {
        'name': 'Saturation pure pointwise',
        'class': PureSaturationGamma,
        'factory': create_gamma_hyp_001,
        'family': 'markovian',
        'implemented': True,
        'metadata': METADATA_GAM_001,
    },
    'GAM-002': {
        'name': 'Diffusion pure',
        'class': PureDiffusionGamma,
        'factory': create_gamma_hyp_002,
        'family': 'markovian',
        'implemented': True,
        'metadata': METADATA_GAM_002,
    },
    'GAM-003': {
        'name': 'Croissance exponentielle',
        'class': ExponentialGrowthGamma,
        'factory': create_gamma_hyp_003,
        'family': 'markovian',
        'implemented': True,
        'metadata': METADATA_GAM_003,
    },
    'GAM-004': {
        'name': 'Décroissance exponentielle',
        'class': ExponentialDecayGamma,
        'factory': create_gamma_hyp_004,
        'family': 'markovian',
        'implemented': True,
        'metadata': METADATA_GAM_004,
    },
    'GAM-005': {
        'name': 'Oscillateur harmonique',
        'class': HarmonicOscillatorGamma,
        'factory': create_gamma_hyp_005,
        'family': 'markovian',
        'implemented': True,
        'metadata': METADATA_GAM_005,
    },
    
    # Famille Non-markovienne (3 opérateurs)
    'GAM-006': {
        'name': 'Saturation + mémoire ordre-1',
        'class': MemorySaturationGamma,
        'factory': create_gamma_hyp_006,
        'family': 'non_markovian',
        'implemented': True,
        'metadata': METADATA_GAM_006,
    },
    'GAM-007': {
        'name': 'Régulation moyenne glissante',
        'class': SlidingAverageGamma,
        'factory': create_gamma_hyp_007,
        'family': 'non_markovian',
        'implemented': True,
        'metadata': METADATA_GAM_007,
    },
    'GAM-008': {
        'name': 'Mémoire différentielle',
        'class': DifferentialMemoryGamma,
        'factory': create_gamma_hyp_008,
        'family': 'non_markovian',
        'implemented': True,
        'metadata': METADATA_GAM_008,
    },
    
    # Famille Stochastique (3 opérateurs, 1 manquant)
    'GAM-009': {
        'name': 'Saturation + bruit additif',
        'class': StochasticSaturationGamma,
        'factory': create_gamma_hyp_009,
        'family': 'stochastic',
        'implemented': True,
        'metadata': METADATA_GAM_009,
    },
    'GAM-010': {
        'name': 'Bruit multiplicatif',
        'class': MultiplicativeNoiseGamma,
        'factory': create_gamma_hyp_010,
        'family': 'stochastic',
        'implemented': True,
        'metadata': METADATA_GAM_010,
    },
    'GAM-011': {
        'name': 'Branchement tensoriel',
        'family': 'stochastic',
        'implemented': False,
        'note': 'Complexité élevée, R3 uniquement'
    },
    
    # Famille Structurelle (3 opérateurs, 1 manquant)
    'GAM-012': {
        'name': 'Préservation symétrie forcée',
        'class': ForcedSymmetryGamma,
        'factory': create_gamma_hyp_012,
        'family': 'structural',
        'implemented': True,
        'metadata': METADATA_GAM_012,
    },
    'GAM-013': {
        'name': 'Renforcement hebbien',
        'class': HebbianReinforcementGamma,
        'factory': create_gamma_hyp_013,
        'family': 'structural',
        'implemented': True,
        'metadata': METADATA_GAM_013,
    },
    'GAM-014': {
        'name': 'Projection sous-espace',
        'family': 'structural',
        'implemented': False,
        'note': 'Nécessite décomposition spectrale'
    },
}


# ============================================================================
# HELPERS
# ============================================================================

def list_operators():
    """Liste tous les opérateurs avec leur statut d'implémentation."""
    print("\n" + "="*70)
    print("CATALOGUE DES OPÉRATEURS Γ (R0)")
    print("="*70)
    
    families = {}
    for op_id, info in OPERATOR_REGISTRY.items():
        family = info['family']
        if family not in families:
            families[family] = []
        families[family].append((op_id, info))
    
    for family, operators in sorted(families.items()):
        print(f"\n{family.upper().replace('_', ' ')}")
        for op_id, info in operators:
            status = "✅" if info['implemented'] else "⏳"
            print(f"  {status} {op_id}: {info['name']}")
    
    # Statistiques
    n_implemented = sum(1 for info in OPERATOR_REGISTRY.values() if info['implemented'])
    n_total = len(OPERATOR_REGISTRY)
    
    print(f"\n{'─'*70}")
    print(f"Progrès: {n_implemented}/{n_total} opérateurs implémentés ({100*n_implemented/n_total:.0f}%)")
    print("="*70 + "\n")


def get_operator_by_id(operator_id: str, **params):
    """
    Retourne une instance de l'opérateur correspondant à un ID.
    
    Args:
        operator_id: ID du catalogue (ex: "GAM-001")
        **params: Paramètres pour l'opérateur
    
    Returns:
        Instance de l'opérateur (callable)
    
    Raises:
        ValueError: Si ID inconnu ou non implémenté
    
    Exemple:
        gamma = get_operator_by_id("GAM-001", beta=2.0)
        state_next = gamma(state)
    """
    if operator_id not in OPERATOR_REGISTRY:
        available = ', '.join(OPERATOR_REGISTRY.keys())
        raise ValueError(f"Unknown operator_id '{operator_id}'. Available: {available}")
    
    info = OPERATOR_REGISTRY[operator_id]
    
    if not info['implemented']:
        raise ValueError(f"Operator '{operator_id}' not yet implemented")
    
    # Utiliser factory si disponible
    if 'factory' in info:
        return info['factory'](**params)
    # Sinon utiliser classe directement
    elif 'class' in info:
        return info['class'](**params)
    else:
        raise ValueError(f"Operator '{operator_id}' has no factory or class")


def get_implementation_status() -> dict:
    """
    Retourne le statut d'implémentation par famille.
    
    Returns:
        dict {family: (n_implemented, n_total)}
    """
    status = {}
    
    for info in OPERATOR_REGISTRY.values():
        family = info['family']
        if family not in status:
            status[family] = {'implemented': 0, 'total': 0}
        
        status[family]['total'] += 1
        if info['implemented']:
            status[family]['implemented'] += 1
    
    return status


def print_implementation_status():
    """Affiche le statut d'implémentation."""
    status = get_implementation_status()
    
    print("\n" + "="*70)
    print("STATUT D'IMPLÉMENTATION")
    print("="*70)
    
    for family, counts in sorted(status.items()):
        impl = counts['implemented']
        total = counts['total']
        pct = 100 * impl / total if total > 0 else 0
        
        bar = "█" * impl + "░" * (total - impl)
        print(f"\n{family.upper().replace('_', ' '):<20} [{bar}] {impl}/{total} ({pct:.0f}%)")
    
    # Total
    total_impl = sum(c['implemented'] for c in status.values())
    total_all = sum(c['total'] for c in status.values())
    total_pct = 100 * total_impl / total_all
    
    print(f"\n{'─'*70}")
    print(f"TOTAL: {total_impl}/{total_all} ({total_pct:.0f}%)")
    print("="*70 + "\n")


# ============================================================================
# MÉTADONNÉES
# ============================================================================

__catalog__ = {
    'total_operators': len(OPERATOR_REGISTRY),
    'implemented': sum(1 for info in OPERATOR_REGISTRY.values() if info['implemented']),
    'families': {
        'markovian': 5,
        'non_markovian': 3,
        'stochastic': 3,
        'structural': 3,
    }
}

# ============================================================================
# VALIDATION
# ============================================================================

def validate_operator(operator_id: str, test_state: 'np.ndarray' = None):
    """
    Valide qu'un opérateur est correctement implémenté.
    
    Vérifie:
    - Callable
    - Retourne np.ndarray
    - Préserve shape
    - Ne produit pas NaN/Inf
    
    Args:
        operator_id: ID de l'opérateur
        test_state: État de test (si None, utilise identité 10×10)
    
    Returns:
        bool: True si validation réussie
    """
    import numpy as np
    
    if test_state is None:
        test_state = np.eye(10)
    
    try:
        # Créer opérateur avec paramètres par défaut
        info = OPERATOR_REGISTRY[operator_id]
        
        if not info['implemented']:
            print(f"⏳ {operator_id}: Non implémenté")
            return False
        
        if 'factory' in info:
            # Utiliser grille Phase 1 pour paramètres nominaux
            grid = get_param_grid(operator_id, phase=1)
            if grid:
                params = list(grid.values())[0]  # Premier set de params
            else:
                params = {}
            
            gamma = info['factory'](**params)
        else:
            gamma = info['class']()
        
        # Tester appel
        result = gamma(test_state)
        
        # Vérifications
        assert isinstance(result, np.ndarray), "Result must be np.ndarray"
        assert result.shape == test_state.shape, "Shape must be preserved"
        assert not np.any(np.isnan(result)), "Result contains NaN"
        assert not np.any(np.isinf(result)), "Result contains Inf"
        
        print(f"✅ {operator_id}: Validation réussie")
        return True
    
    except Exception as e:
        print(f"❌ {operator_id}: Validation échouée - {str(e)}")
        return False


def validate_all_operators():
    """Valide tous les opérateurs implémentés."""
    print("\n" + "="*70)
    print("VALIDATION DES OPÉRATEURS")
    print("="*70 + "\n")
    
    results = {}
    for op_id in OPERATOR_REGISTRY.keys():
        results[op_id] = validate_operator(op_id)
    
    n_valid = sum(1 for v in results.values() if v)
    n_total = len([op for op in OPERATOR_REGISTRY.values() if op['implemented']])
    
    print(f"\n{'─'*70}")
    print(f"Résultat: {n_valid}/{n_total} opérateurs valides")
    print("="*70 + "\n")
    
    return results
	
__status_summary__ = """
✅ IMPLÉMENTÉS (10/14):
  Markoviens: GAM-001, 002, 003, 004, 005
  Non-markoviens: GAM-006, 007, 008
  Stochastiques: GAM-009, 010
  Structurels: GAM-012, 013

⏳ MANQUANTS (4/14):
  Stochastiques: GAM-011 (branchement tensoriel - complexe)
  Structurels: GAM-014 (projection sous-espace)
  
Note: Les 10 opérateurs implémentés couvrent toutes les familles
      et permettent de lancer Phase 1 complète.
"""