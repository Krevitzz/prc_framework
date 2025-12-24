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
  - GAM-002: Diffusion pure (TODO)
  - GAM-003: Croissance exponentielle (TODO)
  - GAM-004: Décroissance exponentielle (TODO)
  - GAM-005: Oscillateur harmonique (TODO)

Famille Non-markovienne:
  - GAM-006: Saturation + mémoire ordre-1 (TODO)
  - GAM-007: Régulation moyenne glissante (TODO)
  - GAM-008: Mémoire différentielle (TODO)

Famille Stochastique:
  - GAM-009: Saturation + bruit additif (TODO)
  - GAM-010: Bruit multiplicatif (TODO)
  - GAM-011: Branchement tensoriel (TODO)

Famille Structurelle:
  - GAM-012: Préservation symétrie forcée (TODO)
  - GAM-013: Renforcement hebbien (TODO)
  - GAM-014: Projection sous-espace (TODO)
"""

# ============================================================================
# IMPORTS
# ============================================================================

from .gamma_hyp_001 import (
    PureSaturationGamma,
    create_gamma_hyp_001,
    PARAM_GRID_PHASE1,
    PARAM_GRID_PHASE2,
    METADATA as METADATA_GAM_001,
)

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'PureSaturationGamma',
    'create_gamma_hyp_001',
    'get_operator_by_id',
    'validate_operator',
    'list_operators',
]

__version__ = '0.1.0'  # Version partielle (1/14 implémenté)
__author__ = 'PRC Operators Team'
__description__ = 'Opérateurs Γ pour exploration R0'

# ============================================================================
# REGISTRE DES OPÉRATEURS
# ============================================================================

OPERATOR_REGISTRY = {
    'GAM-001': {
        'name': 'Saturation pure pointwise',
        'class': PureSaturationGamma,
        'factory': create_gamma_hyp_001,
        'family': 'markovian',
        'implemented': True,
        'metadata': METADATA_GAM_001,
    },
    # GAM-002 à GAM-014: TODO
    'GAM-002': {
        'name': 'Diffusion pure',
        'family': 'markovian',
        'implemented': False,
    },
    'GAM-003': {
        'name': 'Croissance exponentielle',
        'family': 'markovian',
        'implemented': False,
    },
    'GAM-004': {
        'name': 'Décroissance exponentielle',
        'family': 'markovian',
        'implemented': False,
    },
    'GAM-005': {
        'name': 'Oscillateur harmonique',
        'family': 'markovian',
        'implemented': False,
    },
    'GAM-006': {
        'name': 'Saturation + mémoire ordre-1',
        'family': 'non_markovian',
        'implemented': False,
    },
    'GAM-007': {
        'name': 'Régulation moyenne glissante',
        'family': 'non_markovian',
        'implemented': False,
    },
    'GAM-008': {
        'name': 'Mémoire différentielle',
        'family': 'non_markovian',
        'implemented': False,
    },
    'GAM-009': {
        'name': 'Saturation + bruit additif',
        'family': 'stochastic',
        'implemented': False,
    },
    'GAM-010': {
        'name': 'Bruit multiplicatif',
        'family': 'stochastic',
        'implemented': False,
    },
    'GAM-011': {
        'name': 'Branchement tensoriel',
        'family': 'stochastic',
        'implemented': False,
    },
    'GAM-012': {
        'name': 'Préservation symétrie forcée',
        'family': 'structural',
        'implemented': False,
    },
    'GAM-013': {
        'name': 'Renforcement hebbien',
        'family': 'structural',
        'implemented': False,
    },
    'GAM-014': {
        'name': 'Projection sous-espace',
        'family': 'structural',
        'implemented': False,
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


def validate_operator(operator_id: str, test_state=None):
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
            # Paramètres nominaux
            params = {'beta': 2.0} if operator_id == 'GAM-001' else {}
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