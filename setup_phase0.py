#!/usr/bin/env python3
"""
setup_phase0.py

Script de création automatique de tous les fichiers Phase 0.

Usage:
    python setup_phase0.py
    
Crée:
- operators/__init__.py
- operators/gamma_hyp_001.py
- modifiers/__init__.py
- tests/utilities/__init__.py
- tests/utilities/test_*.py
- tests/TM-GAM-001.py
"""

import os
from pathlib import Path

# Dictionnaire : chemin relatif → contenu
FILES_TO_CREATE = {
    
    # ========================================================================
    # modifiers/__init__.py
    # ========================================================================
    "modifiers/__init__.py": '''# modifiers/__init__.py

"""
Package modifiers - Modificateurs pour états D
"""

from .noise import (
    add_gaussian_noise,
    add_uniform_noise,
)

__all__ = [
    'add_gaussian_noise',
    'add_uniform_noise',
]

__version__ = '1.0.0'
''',

    # ========================================================================
    # operators/__init__.py
    # ========================================================================
    "operators/__init__.py": '''# operators/__init__.py

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
''',

    # ========================================================================
    # operators/gamma_hyp_001.py
    # ========================================================================
    "operators/gamma_hyp_001.py": '''"""
operators/gamma_hyp_001.py

HYP-GAM-001: Saturation pure pointwise (tanh)
"""

import numpy as np
from typing import Callable

class PureSaturationGamma:
    """Γ de saturation pure pointwise."""
    
    def __init__(self, beta: float = 2.0):
        assert beta > 0, "beta doit être strictement positif"
        self.beta = beta
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        return np.tanh(self.beta * state)
    
    def __repr__(self):
        return f"PureSaturationGamma(beta={self.beta})"

def create_gamma_hyp_001(beta: float = 2.0) -> Callable[[np.ndarray], np.ndarray]:
    return PureSaturationGamma(beta=beta)

PARAM_GRID_PHASE1 = {'nominal': {'beta': 2.0}}
PARAM_GRID_PHASE2 = {
    'beta_low': {'beta': 0.5},
    'beta_nominal': {'beta': 1.0},
    'beta_high': {'beta': 2.0},
    'beta_very_high': {'beta': 5.0},
}

METADATA = {
    'id': 'GAM-001',
    'name': 'Saturation pure pointwise',
    'family': 'markovian',
    'form': 'T_{n+1}[i,j] = tanh(β · T_n[i,j])',
    'parameters': {'beta': {'type': 'float', 'range': '(0, +∞)', 'nominal': 2.0}},
    'd_applicability': ['SYM', 'ASY', 'R3'],
}
''',

}


def create_file(filepath: str, content: str):
    """Crée un fichier avec son contenu."""
    path = Path(filepath)
    
    # Créer répertoires parents si nécessaire
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Écrire fichier
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Créé: {filepath}")


def main():
    print("\n" + "="*70)
    print("SETUP PHASE 0 - Création des fichiers")
    print("="*70 + "\n")
    
    # Vérifier qu'on est dans le bon répertoire
    if not Path('core').exists():
        print("❌ ERREUR: Lancez ce script depuis le répertoire prc_framework/")
        return 1
    
    # Créer tous les fichiers
    for filepath, content in FILES_TO_CREATE.items():
        try:
            create_file(filepath, content)
        except Exception as e:
            print(f"❌ Erreur lors de la création de {filepath}: {e}")
            return 1
    
    print("\n" + "="*70)
    print(f"✓ {len(FILES_TO_CREATE)} fichiers créés avec succès")
    print("="*70)
    
    print("\nProchaines étapes:")
    print("  1. python validate_phase0.py          # Valider installation")
    print("  2. cd tests && python TM-GAM-001.py --single  # Test rapide")
    print("  3. python TM-GAM-001.py --phase 0     # Phase 0 complète")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())