"""
core/kernel.py

Moteur d'exécution PRC universel.

RESPONSABILITÉ UNIQUE:
- Appliquer itérativement state_{n+1} = gamma(state_n)
- Générer les états successifs
- AUCUNE connaissance de gamma, du state, ou de leur interprétation

USAGE:
    for iteration, state in run_kernel(D_initial, gamma, max_iterations=1000):
        if detect_explosion(state):
            break
"""

import numpy as np
from typing import Callable, Optional, Tuple, Generator, Union, List


def run_kernel(
    initial_state: np.ndarray,
    gamma: Callable[[np.ndarray], np.ndarray],
    max_iterations: int = 10000,
    convergence_check: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None,
    record_history: bool = False
) -> Generator[Union[Tuple[int, np.ndarray], Tuple[int, np.ndarray, List[np.ndarray]]], None, None]:
    """
    Générateur d'états successifs state_{n+1} = gamma(state_n).
    
    FONCTION AVEUGLE:
    - Ne connaît ni gamma (juste l'applique)
    - Ne connaît ni le state (juste le propage)
    - Ne connaît ni leur dimension, structure, ou interprétation
    
    Le kernel ne décide JAMAIS d'arrêter par lui-même.
    L'arrêt vient soit de :
    - max_iterations atteint
    - convergence_check retourne True
    - Le TM break la boucle
    
    Args:
        initial_state: Tenseur initial (np.ndarray de shape quelconque)
        gamma: Fonction (np.ndarray → np.ndarray)
        max_iterations: Limite de sécurité
        convergence_check: Fonction (state_n, state_{n+1}) → bool
                           Si retourne True, arrête l'itération
                           Si None, ne vérifie jamais
        record_history: Si True, stocke et yield l'historique complet
                        Si False, ne yield que (iteration, state)
    
    Yields:
        Si record_history=False : (iteration, state)
        Si record_history=True  : (iteration, state, history)
        
        où:
        - iteration : int (0, 1, 2, ...)
        - state : np.ndarray (état à l'iteration n)
        - history : List[np.ndarray] (tous les états jusqu'à n)
    
    Exemples:
        # Sans historique (économie mémoire)
        for i, state in run_kernel(D, gamma, max_iterations=1000):
            if i % 100 == 0:
                print(f"Iteration {i}")
            if detect_explosion(state):
                break
        
        # Avec historique complet
        for i, state, history in run_kernel(D, gamma, max_iterations=1000, 
                                            record_history=True):
            pass
        
        # Avec convergence automatique
        def check_conv(s_n, s_next):
            return np.linalg.norm(s_next - s_n) < 1e-6
        
        for i, state in run_kernel(D, gamma, convergence_check=check_conv):
            pass
    
    Notes:
        - Le TM contrôle entièrement quand arrêter (via break)
        - convergence_check est optionnel (défini dans le TM)
        - Si record_history=True, mémoire O(n × size(state))
    """
    state = initial_state.copy()
    history = [] if record_history else None
    
    for iteration in range(max_iterations):
        # Enregistre état actuel (si demandé)
        if record_history:
            history.append(state.copy())
        
        # Yield état actuel
        if record_history:
            yield iteration, state, history
        else:
            yield iteration, state
        
        # Calcule état suivant
        state_next = gamma(state)
        
        # Vérifie convergence (si check fourni)
        if convergence_check is not None:
            if convergence_check(state, state_next):
                # Yield état final avant de sortir
                if record_history:
                    history.append(state_next.copy())
                    yield iteration + 1, state_next, history
                else:
                    yield iteration + 1, state_next
                return
        
        # Met à jour état
        state = state_next
    
    # Si max_iterations atteint, yield dernier état
    if record_history:
        yield max_iterations, state, history
    else:
        yield max_iterations, state