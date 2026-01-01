# tests/utilities/registries/graph_registry.py
"""
Registre des métriques graphes (interprétation matrice comme adjacence).

Fonctions disponibles :
- density : Densité graphe
- degree_variance : Variance degrés
- clustering_local : Clustering local moyen
- average_path_length : Longueur chemin moyenne (approximation)
- centrality_concentration : Concentration centralité
- small_world_coefficient : Coefficient petit-monde
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from .base_registry import BaseRegistry, register_function


class GraphRegistry(BaseRegistry):
    """Registre des métriques graphes."""
    
    REGISTRY_KEY = "graph"
    
    @register_function("density")
    def compute_density(
        self,
        state: np.ndarray,
        threshold: float = 0.0,
        epsilon: float = 1e-10
    ) -> float:
        """
        Densité du graphe.
        
        Args:
            state: Matrice adjacence 2D
            threshold: Seuil arêtes (|w| > threshold)
            epsilon: Protection division
        
        Returns:
            float: Densité = E / E_max dans [0, 1]
        
        Raises:
            ValueError: Si non 2D ou non carrée
        
        Notes:
            - E = nombre arêtes
            - E_max = n(n-1)/2 (graphe complet)
            - 1.0 → graphe complet
            - 0.0 → pas d'arêtes
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        n = state.shape[0]
        
        # Compter arêtes
        adjacency = np.abs(state) > threshold
        np.fill_diagonal(adjacency, False)  # Pas de boucles
        
        num_edges = np.sum(adjacency) / 2  # Non-orienté
        
        # Max arêtes (graphe complet)
        max_edges = n * (n - 1) / 2
        
        if max_edges < epsilon:
            return 0.0
        
        density = num_edges / max_edges
        
        return float(np.clip(density, 0.0, 1.0))
    
    @register_function("degree_variance")
    def compute_degree_variance(
        self,
        state: np.ndarray,
        threshold: float = 0.0,
        normalize: bool = False,
        epsilon: float = 1e-10
    ) -> float:
        """
        Variance des degrés.
        
        Args:
            state: Matrice adjacence 2D
            threshold: Seuil arêtes
            normalize: Normaliser par degré moyen² ?
            epsilon: Protection division
        
        Returns:
            float: Variance degrés ou CV²
        
        Raises:
            ValueError: Si non 2D ou non carrée
        
        Notes:
            - Mesure hétérogénéité degrés
            - Élevé → hubs et nœuds isolés
            - Faible → degrés homogènes
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        # Matrice adjacence
        adjacency = np.abs(state) > threshold
        np.fill_diagonal(adjacency, False)
        
        # Degrés
        degrees = np.sum(adjacency, axis=1)
        
        # Variance
        var = float(np.var(degrees))
        
        if normalize:
            mean_degree = np.mean(degrees) + epsilon
            var = var / (mean_degree ** 2)
        
        return var
    
    @register_function("clustering_local")
    def compute_clustering_local(
        self,
        state: np.ndarray,
        threshold: float = 0.0,
        epsilon: float = 1e-10
    ) -> float:
        """
        Coefficient clustering local moyen.
        
        Args:
            state: Matrice adjacence 2D
            threshold: Seuil arêtes
            epsilon: Protection division
        
        Returns:
            float: Clustering moyen dans [0, 1]
        
        Raises:
            ValueError: Si non 2D ou non carrée
        
        Notes:
            - Mesure transitivité locale
            - 1.0 → tous triangles fermés
            - 0.0 → pas de triangles
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        # Matrice adjacence binaire
        adjacency = (np.abs(state) > threshold).astype(int)
        np.fill_diagonal(adjacency, 0)
        
        n = adjacency.shape[0]
        clustering_coeffs = []
        
        for i in range(n):
            # Voisins de i
            neighbors = np.where(adjacency[i] > 0)[0]
            k = len(neighbors)
            
            if k < 2:
                continue
            
            # Arêtes entre voisins
            subgraph = adjacency[np.ix_(neighbors, neighbors)]
            num_edges = np.sum(subgraph) / 2
            
            # Clustering local
            max_edges = k * (k - 1) / 2
            if max_edges > epsilon:
                clustering = num_edges / max_edges
                clustering_coeffs.append(clustering)
        
        if not clustering_coeffs:
            return 0.0
        
        return float(np.mean(clustering_coeffs))
    
    @register_function("average_path_length")
    def compute_average_path_length(
        self,
        state: np.ndarray,
        threshold: float = 0.0,
        sample_fraction: float = 0.1,
        epsilon: float = 1e-10
    ) -> float:
        """
        Longueur chemin moyenne (approximation échantillonnée).
        
        Args:
            state: Matrice adjacence 2D
            threshold: Seuil arêtes
            sample_fraction: Fraction nœuds échantillonnés
            epsilon: Protection calculs
        
        Returns:
            float: Longueur chemin moyenne
        
        Raises:
            ValueError: Si non 2D ou non carrée
        
        Notes:
            - Mesure compacité graphe
            - Petit → graphe compact
            - Grand → graphe étendu
            - Échantillonné pour performance O(n²) → O(n)
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        # Adjacence
        adjacency = (np.abs(state) > threshold).astype(float)
        np.fill_diagonal(adjacency, 0)
        
        # Convertir en distances (1 / poids)
        distances = adjacency.copy()
        distances[distances == 0] = np.inf
        
        # Échantillonner nœuds
        n = adjacency.shape[0]
        n_sample = max(1, int(n * sample_fraction))
        sample_nodes = np.random.choice(n, size=n_sample, replace=False)
        
        # Chemins plus courts (échantillonnés)
        try:
            dist_matrix = shortest_path(
                csr_matrix(distances),
                directed=False,
                indices=sample_nodes
            )
            
            # Moyenne chemins finis
            finite_dists = dist_matrix[np.isfinite(dist_matrix)]
            
            if len(finite_dists) == 0:
                return float('inf')
            
            avg_path = float(np.mean(finite_dists))
        
        except:
            return float('inf')
        
        return avg_path
    
    @register_function("centrality_concentration")
    def compute_centrality_concentration(
        self,
        state: np.ndarray,
        threshold: float = 0.0,
        epsilon: float = 1e-10
    ) -> float:
        """
        Concentration de centralité (degré).
        
        Args:
            state: Matrice adjacence 2D
            threshold: Seuil arêtes
            epsilon: Protection division
        
        Returns:
            float: Concentration dans [0, 1]
        
        Raises:
            ValueError: Si non 2D ou non carrée
        
        Notes:
            - 1.0 → centralité concentrée (graphe étoile)
            - 0.0 → centralité distribuée
            - Basé sur variance degrés normalisée
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        # Adjacence
        adjacency = (np.abs(state) > threshold).astype(int)
        np.fill_diagonal(adjacency, 0)
        
        # Degrés
        degrees = np.sum(adjacency, axis=1)
        
        if np.sum(degrees) == 0:
            return 0.0
        
        n = len(degrees)
        max_degree = np.max(degrees)
        
        # Concentration (Freeman)
        numerator = np.sum(max_degree - degrees)
        denominator = (n - 1) * (n - 2) + epsilon
        
        concentration = numerator / denominator
        
        return float(np.clip(concentration, 0.0, 1.0))
    
    @register_function("small_world_coefficient")
    def compute_small_world(
        self,
        state: np.ndarray,
        threshold: float = 0.0,
        epsilon: float = 1e-10
    ) -> float:
        """
        Coefficient petit-monde (σ = (C/C_rand) / (L/L_rand)).
        
        Args:
            state: Matrice adjacence 2D
            threshold: Seuil arêtes
            epsilon: Protection division
        
        Returns:
            float: Coefficient petit-monde
        
        Raises:
            ValueError: Si non 2D ou non carrée
        
        Notes:
            - σ >> 1 : Propriété petit-monde
            - σ ≈ 1 : Graphe aléatoire
            - σ < 1 : Réseau régulier
            - Approximation via comparaison clustering/path
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        if state.shape[0] != state.shape[1]:
            raise ValueError(f"Matrice doit être carrée, shape={state.shape}")
        
        # Clustering observé
        C = self.compute_clustering_local(state, threshold)
        
        # Path length observé (approximation)
        L = self.compute_average_path_length(state, threshold, sample_fraction=0.1)
        
        if not np.isfinite(L):
            return 1.0
        
        # Densité
        density = self.compute_density(state, threshold)
        
        # Clustering random (approximation)
        C_rand = density + epsilon
        
        # Path random (approximation log(n)/log(k))
        n = state.shape[0]
        k = density * (n - 1)
        if k > 1:
            L_rand = np.log(n) / np.log(k)
        else:
            L_rand = n
        
        # Coefficient
        if L_rand < epsilon or C_rand < epsilon:
            return 1.0
        
        sigma = (C / C_rand) / (L / L_rand + epsilon)
        
        return float(sigma)