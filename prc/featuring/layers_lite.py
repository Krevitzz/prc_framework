"""
featuring.layers_lite.py

Helpers analyses génériques
"""

from typing import Dict, List
import numpy as np

def group_rows_by_layers(rows: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Groupe rows par layers (découverte automatique).
    
    Args:
        rows : Liste {composition, features, layers}
    
    Returns:
        {
            'universal': [...],    # Tous runs
            'matrix_2d': [...],    # Runs avec matrix_2d
            'tensor_3d': [...],    # Runs avec tensor_3d
            'matrix_square': [...] # Runs avec matrix_square
        }
    
    Notes:
        - Un run peut apparaître dans plusieurs layers
        - Extensible : Nouveaux layers auto-détectés
    
    Examples:
        >>> groups = group_rows_by_layers(rows)
        >>> groups.keys()
        dict_keys(['universal', 'matrix_2d', 'tensor_3d'])
    """
    layers_groups = {}
    
    for row in rows:
        for layer_name in row['layers']:
            if layer_name not in layers_groups:
                layers_groups[layer_name] = []
            
            layers_groups[layer_name].append(row)
    
    return layers_groups
    
# =============================================================================
# INSPECTION HISTORY
# =============================================================================

def inspect_history(history: np.ndarray) -> Dict:
    """
    Inspecte history → métadonnées pour applicabilité.
    
    Returns:
        {
            'rank': 2,
            'shape': (201, 10, 10),
            'is_square': True,
            'n_dof': 10,
        }
    """
    rank = history.ndim - 1
    shape = history.shape
    
    is_square = False
    if rank == 2 and len(shape) >= 3:
        if shape[-2] == shape[-1]:
            is_square = True
    
    n_dof = shape[-1] if len(shape) > 1 else 0
    
    return {
        'rank': rank,
        'shape': shape,
        'is_square': is_square,
        'n_dof': n_dof,
    }

    # =============================================================================
# APPLICABILITÉ LAYERS
# =============================================================================

def check_applicability(history_info: Dict, layer_config: Dict) -> bool:
    """
    Vérifie applicabilité layer selon conditions YAML.
    
    Args:
        history_info : {'rank': 2, 'is_square': True, ...}
        layer_config : Config YAML layer complète
    
    Returns:
        True si toutes conditions satisfaites
    
    Notes:
        - Toutes conditions doivent être satisfaites (AND logique)
        - Extensible : Nouveaux critères = enrichir history_info
    """
    applicability = layer_config.get('applicability', {})
    
    # Pas de contraintes = toujours applicable
    if not applicability:
        return True
    
    # Vérifier chaque condition
    for key, required_value in applicability.items():
        if key not in history_info:
            return False
        
        actual_value = history_info[key]
        
        # Comparaison flexible
        if isinstance(required_value, bool):
            if actual_value != required_value:
                return False
        elif isinstance(required_value, (int, float)):
            if actual_value != required_value:
                return False
        elif isinstance(required_value, list):
            if actual_value not in required_value:
                return False
        else:
            if actual_value != required_value:
                return False
    
    return True