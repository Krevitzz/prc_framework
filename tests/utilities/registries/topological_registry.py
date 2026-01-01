# tests/utilities/registries/topological_registry.py
"""
Registre des analyses topologiques simplifiées.

Fonctions disponibles :
- connected_components : Nombre composantes connexes
- euler_characteristic : Caractéristique Euler (2D)
- perimeter_area_ratio : Ratio périmètre/aire
- compactness : Compacité formes
- holes_count : Estimation nombre trous
- fractal_dimension : Dimension fractale (box-counting)
"""

import numpy as np
from scipy import ndimage
from .base_registry import BaseRegistry, register_function


class TopologicalRegistry(BaseRegistry):
    """Registre des analyses topologiques simplifiées."""
    
    REGISTRY_KEY = "topological"
    
    @register_function("connected_components")
    def compute_connected_components(
        self,
        state: np.ndarray,
        threshold: float = 0.0,
        connectivity: int = 1
    ) -> float:
        """
        Nombre de composantes connexes.
        
        Args:
            state: Matrice 2D
            threshold: Seuil binarisation
            connectivity: 1 (4-connexe) ou 2 (8-connexe)
        
        Returns:
            float: Nombre composantes connexes
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - Binarise state > threshold
            - Compte régions connexes
            - Élevé → structure fragmentée
            - Faible → structure cohérente
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        # Binariser
        binary = state > threshold
        
        # Labelliser composantes
        labeled, num_components = ndimage.label(binary, structure=np.ones((3, 3)) if connectivity == 2 else None)
        
        return float(num_components)
    
    @register_function("euler_characteristic")
    def compute_euler_characteristic(
        self,
        state: np.ndarray,
        threshold: float = 0.0
    ) -> float:
        """
        Caractéristique Euler (χ = V - E + F).
        
        Args:
            state: Matrice 2D
            threshold: Seuil binarisation
        
        Returns:
            float: χ ≈ composantes - trous
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - Invariant topologique fondamental
            - χ > 0 : Plus de composantes que de trous
            - χ = 0 : Composantes = trous (ex: tore)
            - χ < 0 : Plus de trous
            - Approximation via composantes et contours
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        # Binariser
        binary = state > threshold
        
        # Composantes (objets)
        labeled, num_objects = ndimage.label(binary)
        
        # Trous = composantes dans inverse
        labeled_holes, num_holes = ndimage.label(~binary)
        
        # χ ≈ objets - trous (simplifié)
        # Note: Vrai Euler nécessite analyse plus fine
        euler = num_objects - num_holes
        
        return float(euler)
    
    @register_function("perimeter_area_ratio")
    def compute_perimeter_area_ratio(
        self,
        state: np.ndarray,
        threshold: float = 0.0,
        epsilon: float = 1e-10
    ) -> float:
        """
        Ratio périmètre/aire moyen.
        
        Args:
            state: Matrice 2D
            threshold: Seuil binarisation
            epsilon: Protection division
        
        Returns:
            float: Moyenne P/A des composantes
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - Mesure complexité frontière
            - Élevé → frontières irrégulières, complexes
            - Faible → formes compactes
            - Cercle: P/A minimal pour aire donnée
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        # Binariser
        binary = state > threshold
        
        # Labelliser
        labeled, num_objects = ndimage.label(binary)
        
        if num_objects == 0:
            return 0.0
        
        ratios = []
        for i in range(1, num_objects + 1):
            # Masque objet
            mask = labeled == i
            
            # Aire
            area = np.sum(mask)
            
            if area < 4:  # Trop petit
                continue
            
            # Périmètre (approximation via gradient)
            boundary = mask ^ ndimage.binary_erosion(mask)
            perimeter = np.sum(boundary)
            
            # Ratio
            if area > epsilon:
                ratios.append(perimeter / area)
        
        if not ratios:
            return 0.0
        
        return float(np.mean(ratios))
    
    @register_function("compactness")
    def compute_compactness(
        self,
        state: np.ndarray,
        threshold: float = 0.0,
        epsilon: float = 1e-10
    ) -> float:
        """
        Compacité (isoperimetric ratio).
        
        Args:
            state: Matrice 2D
            threshold: Seuil binarisation
            epsilon: Protection division
        
        Returns:
            float: Moyenne 4πA/P² dans [0, 1]
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - 1.0 → cercle parfait (compacité max)
            - < 1.0 → forme moins compacte
            - Mesure "circularité"
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        # Binariser
        binary = state > threshold
        
        # Labelliser
        labeled, num_objects = ndimage.label(binary)
        
        if num_objects == 0:
            return 0.0
        
        compactness_values = []
        for i in range(1, num_objects + 1):
            mask = labeled == i
            
            area = np.sum(mask)
            if area < 4:
                continue
            
            boundary = mask ^ ndimage.binary_erosion(mask)
            perimeter = np.sum(boundary)
            
            if perimeter > epsilon:
                compactness = (4 * np.pi * area) / (perimeter ** 2)
                compactness = np.clip(compactness, 0.0, 1.0)
                compactness_values.append(compactness)
        
        if not compactness_values:
            return 0.0
        
        return float(np.mean(compactness_values))
    
    @register_function("holes_count")
    def compute_holes_count(
        self,
        state: np.ndarray,
        threshold: float = 0.0,
        min_hole_size: int = 4
    ) -> float:
        """
        Estimation nombre trous (régions fermées vides).
        
        Args:
            state: Matrice 2D
            threshold: Seuil binarisation
            min_hole_size: Taille minimale trou
        
        Returns:
            float: Nombre trous détectés
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - Trou = composante connexe de background encerclée
            - Approximation simple via fill_holes
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        # Binariser
        binary = state > threshold
        
        # Remplir trous
        filled = ndimage.binary_fill_holes(binary)
        
        # Différence = trous
        holes = filled & ~binary
        
        # Labelliser trous
        labeled_holes, num_holes = ndimage.label(holes)
        
        # Filtrer petits trous
        valid_holes = 0
        for i in range(1, num_holes + 1):
            if np.sum(labeled_holes == i) >= min_hole_size:
                valid_holes += 1
        
        return float(valid_holes)
    
    @register_function("fractal_dimension")
    def compute_fractal_dimension(
        self,
        state: np.ndarray,
        threshold: float = 0.0,
        max_box_size: int = None,
        num_scales: int = 10
    ) -> float:
        """
        Dimension fractale (box-counting).
        
        Args:
            state: Matrice 2D
            threshold: Seuil binarisation
            max_box_size: Taille box max (None = min(shape)/4)
            num_scales: Nombre échelles testées
        
        Returns:
            float: Dimension fractale (1 ≤ D ≤ 2 pour 2D)
        
        Raises:
            ValueError: Si non 2D
        
        Notes:
            - Mesure complexité auto-similaire
            - D ≈ 1 : Structure linéaire
            - D ≈ 2 : Remplissage plan
            - D intermédiaire : Structure fractale
        """
        if state.ndim != 2:
            raise ValueError(f"Attendu 2D, reçu {state.ndim}D")
        
        # Binariser
        binary = state > threshold
        
        if max_box_size is None:
            max_box_size = min(binary.shape) // 4
        
        # Échelles box
        box_sizes = np.logspace(
            0, np.log2(max_box_size), num=num_scales, base=2
        ).astype(int)
        box_sizes = np.unique(box_sizes)
        box_sizes = box_sizes[box_sizes > 0]
        
        if len(box_sizes) < 2:
            return 1.0
        
        # Compter boxes à chaque échelle
        counts = []
        for box_size in box_sizes:
            # Sous-échantillonner
            if box_size > 1:
                downsampled = binary[::box_size, ::box_size]
            else:
                downsampled = binary
            
            # Compter boxes occupées
            count = np.sum(downsampled)
            counts.append(count)
        
        counts = np.array(counts)
        counts = counts[counts > 0]  # Filtrer zéros
        
        if len(counts) < 2:
            return 1.0
        
        # Régression log-log
        box_sizes = box_sizes[:len(counts)]
        
        coeffs = np.polyfit(
            np.log(box_sizes),
            np.log(counts),
            1
        )
        
        # Dimension = pente
        dimension = -coeffs[0]
        
        # Clipper dans [1, 2]
        dimension = np.clip(dimension, 1.0, 2.0)
        
        return float(dimension)