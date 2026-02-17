"""
operators/gamma_hyp_001.py

HYP-GAM-001: Saturation pure pointwise (tanh)
"""

import numpy as np
from typing import Callable

class PureSaturationGamma:
    """Γ de saturation pure pointwise."""
    
    def __init__(self, beta: float = 2.0, seed: int = None):
        assert beta > 0, "beta doit être strictement positif"
        self.beta = beta
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        return np.tanh(self.beta * state)
    
    def __repr__(self):
        return f"PureSaturationGamma(beta={self.beta})"

def create_gamma_hyp_001(beta: float = 2.0, seed: int = None) -> Callable[[np.ndarray], np.ndarray]:
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
    'PHASE' : "R0",
    'name': 'Saturation pure pointwise',
    'family': 'markovian',
    'form': 'T_{n+1}[i,j] = tanh(β · T_n[i,j])',
    'parameters': {'beta': {'type': 'float', 'range': '(0, +∞)', 'nominal': 2.0}},
    'd_applicability': ['SYM', 'ASY', 'R3'],
}
