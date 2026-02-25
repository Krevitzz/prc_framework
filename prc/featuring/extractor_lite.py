"""
prc.featuring.extractor_lite

Responsabilité : Extraction features scalaires depuis history np.ndarray

Architecture simplifiée : Pas de fonctions intermédiaires par layer
"""

import numpy as np
from typing import Dict
from featuring.layers_lite import inspect_history, check_applicability


# =============================================================================
# PROJECTIONS TEMPORELLES
# =============================================================================

def compute_projection(history: np.ndarray, projection: str) -> np.ndarray:
    """
    Calcule projection temporelle sur history.
    
    Args:
        history    : Séquence états (T, *dims)
        projection : 'initial' | 'final' | 'mean' | 'max' | 'min'
    
    Returns:
        État projeté (*dims)
    """
    if projection == 'initial':
        return history[0]
    elif projection == 'final':
        return history[-1]
    elif projection == 'mean':
        return np.mean(history, axis=0)
    elif projection == 'max':
        return np.max(history, axis=0)
    elif projection == 'min':
        return np.min(history, axis=0)
    else:
        raise ValueError(f"Projection inconnue: {projection}")


# =============================================================================
# EXTRACTION FEATURES (GÉNÉRIQUE)
# =============================================================================

def extract_for_layer(
    history: np.ndarray,
    layer_name: str,
    layer_config: Dict
) -> Dict[str, float]:
    """
    Extraction features vraiment générique (appelle directement registres).
    
    Workflow:
        1. Lit layer_config['functions']
        2. Pour chaque fonction:
           - Import registre dynamique
           - Appelle fonction
           - Applique projections
    
    Args:
        history : Séquence états (T, *dims)
        layer_name : 'universal', 'matrix_2d', 'tensor_3d'
        layer_config : Config YAML layer
    
    Returns:
        Dict features extraites
    
    Notes:
        - Pas de fonction intermédiaire extract_{layer}_features()
        - Appelle directement registres
        - Extensible : Nouveau layer = YAML + registre
    """
    features = {}
    functions_config = layer_config.get('functions', [])
    
    # Import registre dynamique
    try:
        registre_module = __import__(
            f'featuring.registries.{layer_name}_lite',
            fromlist=['']
        )
    except ImportError:
        print(f"[WARNING] Registre {layer_name}_lite introuvable")
        return features
    
    # Pour chaque fonction du YAML
    for func_config in functions_config:
        # Extraire config fonction
        if isinstance(func_config, dict):
            func_name = func_config['name']
            projections = func_config.get('projections', ['final'])
            params = func_config.get('params', {})
        else:
            # Format simple (str)
            func_name = func_config
            projections = ['final']
            params = {}
        
        # Récupérer fonction depuis registre
        try:
            func = getattr(registre_module, func_name)
        except AttributeError:
            print(f"[WARNING] Fonction {func_name} absente dans {layer_name}_lite")
            continue
        
        # Appliquer projections
        for projection in projections:
            state = compute_projection(history, projection)
            
            try:
                value = func(state, **params)
                feature_name = f'{func_name}_{projection}'
                features[feature_name] = float(value)
            except Exception as e:
                print(f"[WARNING] Erreur {func_name}_{projection}: {e}")
    
    return features