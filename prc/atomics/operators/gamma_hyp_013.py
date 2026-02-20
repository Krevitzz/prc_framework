"""
operators/gamma_hyp_013.py

GAM-013 : Renforcement hebbien local
Forme   : T_{n+1}[i,j] = T_n[i,j] + η·Σ_k T_n[i,k]·T_n[k,j]
Famille : structural
Applicabilité : SYM, ASY (rang 2 carré uniquement)

Comportement attendu :
- Émergence possible de clusters (renforcement mutuel)
- Instable sans régulation additionnelle (risque explosion)
- Structures hiérarchiques possibles
- Coût O(N³) — surveiller les normes
"""

import numpy as np
from typing import Callable


class HebbianReinforcementGamma:
    """Γ de renforcement hebbien."""

    def __init__(self, eta: float = 0.01):
        assert 0 < eta <= 0.1, "eta doit être dans (0, 0.1] pour stabilité"
        self.eta = eta

    def __call__(self, state: np.ndarray) -> np.ndarray:
        if state.ndim != 2:
            raise ValueError(
                f"GAM-013 applicable rang 2 uniquement, reçu ndim={state.ndim}"
            )
        if state.shape[0] != state.shape[1]:
            raise ValueError(
                f"GAM-013 nécessite matrice carrée, reçu shape={state.shape}"
            )
        return state + self.eta * (state @ state)

    def __repr__(self):
        return f"HebbianReinforcementGamma(eta={self.eta})"


def create(eta: float = 0.01) -> Callable:
    """Factory GAM-013."""
    return HebbianReinforcementGamma(eta=eta)


METADATA = {
    'id': 'GAM-013',
    'd_applicability': ['SYM', 'ASY'],
}
