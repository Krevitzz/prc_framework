# operators/__init__.py

"""
Package operators - Définitions Γ pour toutes les hypothèses R0
"""

from .gamma_hyp_001 import (
    PureSaturationGamma,
    create_gamma_hyp_001,
    PARAM_GRID_PHASE1,
    PARAM_GRID_PHASE2,
    METADATA as METADATA_GAM_001,
)

__all__ = [
    'PureSaturationGamma',
    'create_gamma_hyp_001',
]

OPERATOR_REGISTRY = {
    'GAM-001': {
        'name': 'Saturation pure pointwise',
        'class': PureSaturationGamma,
        'factory': create_gamma_hyp_001,
        'family': 'markovian',
        'implemented': True,
        'metadata': METADATA_GAM_001,
    },
}

def get_operator_by_id(operator_id: str, **params):
    """Retourne une instance de l'opérateur."""
    if operator_id not in OPERATOR_REGISTRY:
        raise ValueError(f"Unknown operator_id '{operator_id}'")
    
    info = OPERATOR_REGISTRY[operator_id]
    if not info['implemented']:
        raise ValueError(f"Operator '{operator_id}' not yet implemented")
    
    if 'factory' in info:
        return info['factory'](**params)
    elif 'class' in info:
        return info['class'](**params)
