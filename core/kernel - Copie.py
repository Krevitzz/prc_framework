"""
core/kernel.py

Moteur de simulation PRC universel.

RESPONSABILITÉ UNIQUE (Thèse 3.3, Ch. 2.5):
- Appliquer itérativement Γ sur C
- Enregistrer l'historique (optionnel)
- RIEN d'autre

CE QUE LE KERNEL NE FAIT PAS:
- Aucune analyse de patterns
- Aucune détection de structures
- Aucune interprétation physique
- Aucune visualisation
- Aucune optimisation

PRINCIPE: Le kernel est l'implémentation pure de la dynamique D_{n+1} = Γ(D_n)
Toute autre fonctionnalité appartient à d'autres modules.
"""

import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from .information_space import InformationSpace
from .evolution_operator import EvolutionOperator


@dataclass
class SimulationState:
    """
    État du système à un instant donné.
    
    Snapshot minimal pour l'historique ou l'export.
    """
    iteration: int
    C: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        """Sérialise l'état."""
        return {
            "iteration": self.iteration,
            "C": self.C.tolist()
        }


class PRCKernel:
    """
    Moteur de simulation PRC universel.
    
    Le kernel implémente la boucle fondamentale:
        for n in range(iterations):
            C_{n+1} = Γ(C_n)
    
    C'est tout. Toute la complexité du framework PRC repose sur:
    1. Le choix de l'encodage initial (D^(X))
    2. Le choix de l'opérateur Γ
    3. L'analyse post-hoc des patterns émergents
    
    Le kernel ne fait que l'itération aveugle.
    """
    
    def __init__(self, 
                 initial_space: InformationSpace,
                 gamma: EvolutionOperator,
                 validate_each_step: bool = False):
        """
        Initialise le kernel avec état initial et opérateur d'évolution.
        
        Args:
            initial_space: État initial D_0 (InformationSpace)
            gamma: Opérateur d'évolution Γ
            validate_each_step: Si True, valide les invariants après chaque Γ
                               (utile pour debug, coûteux en production)
        """
        # ÉTAT INITIAL (immutable une fois défini)
        self.D_0 = initial_space.copy()
        
        # OPÉRATEUR D'ÉVOLUTION (immutable)
        self.gamma = gamma
        
        # ÉTAT ACTUEL (mutable)
        self.C_current = initial_space.C.copy()
        self.iteration = 0
        
        # CONFIGURATION
        self.validate_each_step = validate_each_step
        
        # HISTORIQUE (optionnel)
        self._history: List[SimulationState] = []
        self._recording = False
    
    def step(self, n_steps: int = 1) -> np.ndarray:
        """
        Exécute n itérations de Γ.
        
        Implémentation pure:
            for _ in range(n_steps):
                C_{n+1} = Γ(C_n)
                n += 1
        
        Args:
            n_steps: Nombre d'itérations à exécuter
        
        Returns:
            C après les n itérations
        """
        for _ in range(n_steps):
            # ENREGISTRE AVANT TRANSFORMATION (si recording actif)
            if self._recording:
                self._history.append(SimulationState(
                    iteration=self.iteration,
                    C=self.C_current.copy()
                ))
            
            # CŒUR DU KERNEL: C_{n+1} = Γ(C_n)
            self.C_current = self.gamma.apply(self.C_current)
            
            # VALIDATION (si mode debug)
            if self.validate_each_step:
                self.gamma.validate_output(self.C_current)
            
            # INCRÉMENTE LE COMPTEUR
            self.iteration += 1
        
        return self.C_current
    
    def run(self, 
            max_iterations: int,
            convergence_threshold: Optional[float] = None,
            record_history: bool = False,
            record_interval: int = 1) -> Dict[str, Any]:
        """
        Exécute la simulation jusqu'à convergence ou limite d'itérations.
        
        Args:
            max_iterations: Nombre maximum d'itérations
            convergence_threshold: Si défini, arrête si ||C_{n+1} - C_n|| < seuil
            record_history: Si True, enregistre l'historique complet
            record_interval: Enregistre tous les N pas (si record_history=True)
        
        Returns:
            Dictionnaire avec résultats de la simulation:
            {
                "completed_iterations": int,
                "converged": bool,
                "final_state": SimulationState,
                "convergence_info": {...}  # si convergence détectée
            }
        """
        # Active l'enregistrement si demandé
        if record_history:
            self.start_recording()
        
        converged = False
        convergence_iteration = None
        final_diff = None
        
        for i in range(max_iterations):
            # Sauvegarde l'état précédent pour convergence
            if convergence_threshold is not None:
                C_prev = self.C_current.copy()
            
            # Exécute une étape
            self.step(1)
            
            # Vérifie la convergence
            if convergence_threshold is not None:
                diff = np.linalg.norm(self.C_current - C_prev)
                final_diff = diff
                
                if diff < convergence_threshold:
                    converged = True
                    convergence_iteration = self.iteration
                    break
        
        # Prépare les résultats
        results = {
            "completed_iterations": self.iteration,
            "converged": converged,
            "final_state": SimulationState(
                iteration=self.iteration,
                C=self.C_current.copy()
            )
        }
        
        if converged:
            results["convergence_info"] = {
                "iteration": convergence_iteration,
                "final_diff": final_diff,
                "threshold": convergence_threshold
            }
        
        # Désactive l'enregistrement
        if record_history:
            self.stop_recording()
            results["history"] = self.get_history()
        
        return results
    
    def reset(self) -> None:
        """
        Réinitialise le kernel à l'état initial.
        
        Remet C à C_0, compteur à 0, efface l'historique.
        """
        self.C_current = self.D_0.C.copy()
        self.iteration = 0
        self._history = []
        self._recording = False
    
    def start_recording(self) -> None:
        """
        Active l'enregistrement de l'historique.
        
        Les états seront stockés à chaque appel à step().
        Attention: coûteux en mémoire pour longues simulations.
        """
        self._recording = True
    
    def stop_recording(self) -> None:
        """Désactive l'enregistrement."""
        self._recording = False
    
    def get_history(self) -> List[SimulationState]:
        """
        Retourne l'historique enregistré.
        
        Returns:
            Liste des états [SimulationState_0, ..., SimulationState_n]
        """
        return self._history.copy()
    
    def clear_history(self) -> None:
        """Efface l'historique (libère la mémoire)."""
        self._history = []
    
    def get_current_state(self) -> InformationSpace:
        """
        Retourne l'état actuel comme InformationSpace.
        
        Returns:
            InformationSpace avec C actuel et métadonnées mises à jour
        """
        metadata = {
            **self.D_0.metadata,
            "iteration": self.iteration,
            "gamma_type": self.gamma.get_parameters()["type"]
        }
        
        return InformationSpace(self.C_current, metadata)
    
    def get_trajectory_summary(self) -> Dict[str, Any]:
        """
        Calcule un résumé statistique de la trajectoire.
        
        Utile pour analyse rapide sans stocker tout l'historique.
        
        Returns:
            Statistiques sur l'évolution:
            {
                "mean_correlation": float,
                "std_correlation": float,
                "frobenius_norm": float,
                "trace": float,
                "rank": int
            }
        """
        C = self.C_current
        n = C.shape[0]
        
        # Corrélations hors diagonale
        mask = ~np.eye(n, dtype=bool)
        off_diag = C[mask]
        
        return {
            "mean_correlation": float(np.mean(off_diag)),
            "std_correlation": float(np.std(off_diag)),
            "min_correlation": float(np.min(off_diag)),
            "max_correlation": float(np.max(off_diag)),
            "frobenius_norm": float(np.linalg.norm(C, 'fro')),
            "trace": float(np.trace(C)),
            "rank": int(np.linalg.matrix_rank(C))
        }
    
    def export_state(self, filepath: str) -> None:
        """
        Export l'état actuel en JSON.
        
        Args:
            filepath: Chemin du fichier de sortie
        """
        from .serialization import export_state
        export_state(self, filepath)
    
    def export_history(self, filepath: str) -> None:
        """
        Export l'historique complet en JSON.
        
        Args:
            filepath: Chemin du fichier de sortie
        """
        from .serialization import export_history
        export_history(self, filepath)
    
    def __repr__(self) -> str:
        """Représentation textuelle pour debug."""
        return (f"PRCKernel(iteration={self.iteration}, "
                f"n_dof={self.D_0.n_dof}, "
                f"gamma={self.gamma})")


# ============================================================================
# UTILITAIRES DE SIMULATION (patterns d'usage courants)
# ============================================================================

def run_until_convergence(initial_space: InformationSpace,
                         gamma: EvolutionOperator,
                         threshold: float = 1e-6,
                         max_iterations: int = 10000,
                         check_interval: int = 10) -> Dict[str, Any]:
    """
    Exécute jusqu'à convergence avec vérification périodique.
    
    Plus efficace que vérifier à chaque pas pour grandes simulations.
    
    Args:
        initial_space: État initial
        gamma: Opérateur d'évolution
        threshold: Seuil de convergence sur ||C_{n+k} - C_n||
        max_iterations: Limite de sécurité
        check_interval: Vérifie la convergence tous les N pas
    
    Returns:
        Résultats de simulation + info de convergence
    """
    kernel = PRCKernel(initial_space, gamma)
    
    converged = False
    for i in range(0, max_iterations, check_interval):
        C_before = kernel.C_current.copy()
        
        # Exécute check_interval pas
        kernel.step(check_interval)
        
        # Vérifie convergence
        diff = np.linalg.norm(kernel.C_current - C_before)
        if diff < threshold:
            converged = True
            break
    
    return {
        "kernel": kernel,
        "converged": converged,
        "iterations": kernel.iteration,
        "final_diff": diff if 'diff' in locals() else None
    }


def run_parallel_simulations(initial_spaces: List[InformationSpace],
                            gamma: EvolutionOperator,
                            n_steps: int) -> List[InformationSpace]:
    """
    Exécute plusieurs simulations indépendantes en parallèle.
    
    Utile pour tester robustesse ou explorer variations d'encodage.
    
    Args:
        initial_spaces: Liste d'états initiaux différents
        gamma: Même opérateur pour tous
        n_steps: Nombre d'itérations
    
    Returns:
        Liste des états finaux [InformationSpace_1, ..., InformationSpace_n]
    
    Note:
        Cette version est séquentielle. Pour vrai parallélisme,
        utiliser multiprocessing ou joblib dans un script séparé.
    """
    results = []
    
    for space in initial_spaces:
        kernel = PRCKernel(space, gamma)
        kernel.step(n_steps)
        results.append(kernel.get_current_state())
    
    return results


def compare_operators(initial_space: InformationSpace,
                     operators: List[EvolutionOperator],
                     n_steps: int) -> Dict[str, Any]:
    """
    Compare plusieurs opérateurs Γ sur le même état initial.
    
    Utile pour choisir le meilleur Γ ou étudier leurs différences.
    
    Args:
        initial_space: État initial commun
        operators: Liste d'opérateurs à comparer
        n_steps: Nombre d'itérations
    
    Returns:
        Résultats comparatifs:
        {
            "operators": [nom1, nom2, ...],
            "final_states": [InformationSpace_1, ...],
            "summaries": [stats_1, stats_2, ...],
            "divergences": [[d_01, d_02, ...], ...]  # distances finales
        }
    """
    results = {
        "operators": [op.get_parameters()["type"] for op in operators],
        "final_states": [],
        "summaries": [],
        "final_C": []
    }
    
    # Exécute chaque opérateur
    for op in operators:
        kernel = PRCKernel(initial_space, op)
        kernel.step(n_steps)
        
        final_state = kernel.get_current_state()
        summary = kernel.get_trajectory_summary()
        
        results["final_states"].append(final_state)
        results["summaries"].append(summary)
        results["final_C"].append(kernel.C_current)
    
    # Calcule les divergences entre opérateurs
    n_ops = len(operators)
    divergences = np.zeros((n_ops, n_ops))
    
    for i in range(n_ops):
        for j in range(i+1, n_ops):
            div = np.linalg.norm(results["final_C"][i] - results["final_C"][j])
            divergences[i, j] = div
            divergences[j, i] = div
    
    results["divergences"] = divergences.tolist()
    
    return results