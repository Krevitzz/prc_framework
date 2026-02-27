"""
featuring.layers_lite

Responsabilité : Inspection history + vérification applicabilité layers

Architecture :
    inspect_history  → calcule TOUTES propriétés dérivables de l'array (auto-suffisant)
    check_applicability → évalue liste de conditions YAML contre history_info (générique)

Extensibilité :
    Nouvelle propriété → ajouter une ligne dans inspect_history
    Nouveau layer      → créer YAML + registre, zéro touche ici
"""

from typing import Dict, List
import numpy as np


# =============================================================================
# GROUPEMENT ROWS PAR LAYER
# =============================================================================

def group_rows_by_layers(rows: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Groupe rows par layers (découverte automatique depuis tag 'layers').

    Args:
        rows : Liste {composition, features, layers}

    Returns:
        {
            'universal':     [...],  # Tous runs ayant ce layer
            'matrix_2d':     [...],
            'matrix_square': [...],
            'tensor_3d':     [...],
            # Nouveaux layers auto-détectés
        }

    Notes:
        - Un run peut apparaître dans plusieurs layers
        - Extensible : nouveaux layers auto-détectés depuis les tags
    """
    layers_groups = {}

    for row in rows:
        for layer_name in row.get('layers', []):
            if layer_name not in layers_groups:
                layers_groups[layer_name] = []
            layers_groups[layer_name].append(row)

    return layers_groups


# =============================================================================
# INSPECTION HISTORY
# =============================================================================

def inspect_history(history: np.ndarray) -> Dict:
    """
    Calcule toutes propriétés dérivables de l'array numpy.

    Auto-suffisant : aucun param externe, l'array se décrit lui-même.

    Args:
        history : np.ndarray (T, *dims) — séquence temporelle états

    Returns:
        {
            'rank'        : int,   # history.ndim - 1
            'shape'       : tuple, # history.shape complète
            'n_dof'       : int,   # shape[-1] (dernière dimension)
            'n_elements'  : int,   # produit dimensions état (shape[1:])
            'is_square'   : bool,  # rank==2 et shape[-2]==shape[-1]
            'is_cubic'    : bool,  # rank==3 et toutes dims état égales
            'is_symmetric': bool,  # rank==2, is_square, et état final ≈ sa transposée
            'dtype'       : str,   # history.dtype.name (ex: 'float64')
        }

    Notes:
        - is_symmetric : vérifié sur état final (history[-1]), tolérance numérique
        - Ajouter une propriété : une ligne ici, référençable immédiatement en YAML
    """
    rank = history.ndim - 1
    shape = history.shape

    # Dimensions de l'état (sans axe temporel)
    state_shape = shape[1:]
    n_dof = shape[-1] if len(shape) > 1 else 0
    n_elements = int(np.prod(state_shape)) if len(state_shape) > 0 else 0

    # Propriétés géométriques
    is_square = (rank == 2 and len(state_shape) == 2 and state_shape[0] == state_shape[1])

    is_cubic = (
        rank == 3
        and len(state_shape) == 3
        and state_shape[0] == state_shape[1] == state_shape[2]
    )

    # is_symmetric : propriété valeur (pas seulement forme)
    # Vérifiée sur état final — tolérance numérique standard
    is_symmetric = False
    if is_square:
        final_state = history[-1]
        try:
            is_symmetric = bool(np.allclose(final_state, final_state.T, rtol=1e-5, atol=1e-8))
        except Exception:
            is_symmetric = False

    return {
        'rank'        : rank,
        'shape'       : shape,
        'n_dof'       : n_dof,
        'n_elements'  : n_elements,
        'is_square'   : is_square,
        'is_cubic'    : is_cubic,
        'is_symmetric': is_symmetric,
        'dtype'       : history.dtype.name,
    }


# =============================================================================
# APPLICABILITÉ LAYERS
# =============================================================================

_OPS = {
    '=' : lambda a, b: a == b,
    '==': lambda a, b: a == b,
    '>=': lambda a, b: a >= b,
    '<=': lambda a, b: a <= b,
    '>' : lambda a, b: a > b,
    '<' : lambda a, b: a < b,
}


def check_applicability(history_info: Dict, layer_config: Dict) -> bool:
    """
    Évalue liste de conditions YAML contre history_info.

    Args:
        history_info : Dict depuis inspect_history()
        layer_config : Config YAML layer complète

    Returns:
        True si toutes conditions satisfaites (AND logique)
        True si liste vide (layer universel)

    Format YAML attendu :
        applicability: []                                    # toujours applicable
        applicability:
          - {field: rank, op: "=",  value: 2}               # égalité
          - {field: rank, op: ">=", value: 3}               # comparaison
          - {field: is_square, op: "=", value: true}        # booléen

    Notes:
        - Opérateurs supportés : '=', '==', '>=', '<=', '>', '<'
        - Si field absent de history_info → False (condition non évaluable)
        - Extensible : nouvelle propriété → ajouter dans inspect_history uniquement
    """
    conditions = layer_config.get('applicability', [])

    # Liste vide ou absente = toujours applicable
    if not conditions:
        return True

    for condition in conditions:
        field = condition.get('field')
        op    = condition.get('op', '=')
        value = condition.get('value')

        # Field absent de history_info → non évaluable → False
        if field not in history_info:
            return False

        actual = history_info[field]

        # Résoudre opérateur
        op_fn = _OPS.get(op)
        if op_fn is None:
            raise ValueError(
                f"Opérateur inconnu '{op}' dans applicabilité layer. "
                f"Supportés : {list(_OPS.keys())}"
            )

        if not op_fn(actual, value):
            return False

    return True
