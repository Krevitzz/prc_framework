"""
prc.running.compositions

Responsabilité : Génération compositions (produit cartésien axes YAML)
Support : Multi-lignes = séquences, pondération optionnelle sur gammas

Usage : Toujours lancer depuis prc/ (python -m ...)
"""

import random as _random
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from utils.data_loading_lite import (
    discover_encodings,
    discover_gammas,
    discover_modifiers,
    load_yaml,
)


# =============================================================================
# COMPOSEDGAMMA (inline)
# =============================================================================

class ComposedGamma:
    """
    Composition séquentielle de gammas avec pondération optionnelle.
    
    Modes :
    - weights=None : séquentiel pur (gamma1 → gamma2 → ...)
    - weights=[w1, w2, ...] : combinaison linéaire (w1·gamma1(D) + w2·gamma2(D) + ...)
    """
    
    def __init__(
        self,
        sequence_gammas: List[Callable],
        weights: Optional[List[float]] = None,
    ):
        self.sequence = sequence_gammas
        self.weights = weights
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Application séquentielle ou pondérée."""
        if self.weights is None:
            # Séquentiel pur
            for gamma in self.sequence:
                state = gamma(state)
            return state
        else:
            # Combinaison linéaire pondérée
            outputs = [gamma(state) for gamma in self.sequence]
            return sum(w * out for w, out in zip(self.weights, outputs))
    
    def reset(self):
        """Reset mémoire gammas non-markoviens."""
        for gamma in self.sequence:
            if hasattr(gamma, 'reset'):
                gamma.reset()
    
    def __repr__(self):
        if self.weights:
            return f"ComposedGamma({len(self.sequence)} gammas, weighted)"
        return f"ComposedGamma({len(self.sequence)} gammas)"


# =============================================================================
# DISCOVER FUNCTIONS MAPPING
# =============================================================================

_DISCOVER_FNS = {
    'gamma'   : discover_gammas,
    'encoding': discover_encodings,
    'modifier': discover_modifiers,
}


# =============================================================================
# HELPERS
# =============================================================================

def _normalize_to_list(x):
    """Scalaire → [scalaire], liste inchangée."""
    return x if isinstance(x, list) else [x]


def _generate_weights_grid(weights_config: Dict, n_gammas: int) -> List[List[float]]:
    """
    Génère grid pondération depuis config.
    
    Args:
        weights_config : {'range': [min, max], 'step': float}
        n_gammas       : Nombre de gammas dans la séquence
    
    Returns:
        Liste de combinaisons de poids [[w1,w2,...], ...]
    """
    range_min, range_max = weights_config['range']
    step = weights_config['step']
    
    # Valeurs possibles par poids
    n_steps = int((range_max - range_min) / step) + 1
    values = [range_min + i * step for i in range(n_steps)]
    
    # Produit cartésien n_gammas fois
    return list(product(values, repeat=n_gammas))


def _format_gamma_id(sequence_ids: List[str], weights: Optional[List[float]] = None) -> str:
    """
    Formate ID gamma composé.
    
    Format : "w_x-GAM-NNN-w_y-GAM-MMM" si weights
             "GAM-NNN-GAM-MMM" sinon
    """
    if weights is None:
        return '-'.join(sequence_ids)
    
    parts = []
    for w, gid in zip(weights, sequence_ids):
        parts.append(f"w_{w:.2f}-{gid}")
    return '-'.join(parts)


# =============================================================================
# CHARGEMENT CONFIG
# =============================================================================

def load_run_config(yaml_path: Path) -> Dict[str, Any]:
    """Charge YAML de run → dict."""
    return load_yaml(yaml_path)


# =============================================================================
# RÉSOLUTION AXES
# =============================================================================

def resolve_axis_atomic(
    axis_config: Union[str, Dict, List],
    axis_type: str,
    defaults: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Résout un axe atomique → liste de {'id', 'params', 'callable'}.
    
    Gère :
    - 'all'          : tous les ids via discovery
    - {'random': N}  : N ids aléatoires
    - liste          : ids listés
    
    Pour chaque id : defaults[id] mergé avec params inline (inline prime).
    
    Returns:
        [{'id': str, 'params': dict, 'callable': Callable}, ...]
    """
    discover_fn = _DISCOVER_FNS[axis_type]
    
    # Résoudre liste d'ids selon mode
    if axis_config == 'all':
        entities = discover_fn()
        ids_to_resolve = [e['id'] for e in entities]
        entities_map = {e['id']: e for e in entities}
    
    elif isinstance(axis_config, dict) and 'random' in axis_config:
        n = axis_config['random']
        entities = discover_fn()
        all_ids = [e['id'] for e in entities]
        ids_to_resolve = _random.sample(all_ids, min(n, len(all_ids)))
        entities_map = {e['id']: e for e in entities}
    
    elif isinstance(axis_config, list):
        # Collecter tous ids mentionnés
        ids_mentioned = []
        for item in axis_config:
            if isinstance(item, dict):
                ids_mentioned.append(item['id'])
            else:
                ids_mentioned.append(item)
        
        # Discovery pour obtenir callables
        entities = discover_fn()
        entities_map = {e['id']: e for e in entities}
        ids_to_resolve = ids_mentioned
    
    else:
        raise ValueError(f"Format axe inconnu pour '{axis_type}': {axis_config}")
    
    # Construire {id, params, callable} avec merge defaults + inline
    resolved = []
    inline_params_map = {}
    
    # Extraire params inline si liste de dicts
    if isinstance(axis_config, list):
        for item in axis_config:
            if isinstance(item, dict) and 'params' in item:
                inline_params_map[item['id']] = item['params']
    
    for entity_id in ids_to_resolve:
        entity = entities_map.get(entity_id)
        if entity is None:
            raise ValueError(f"{axis_type} '{entity_id}' non trouvé dans discovery")
        
        # Defaults pour cet id
        default_params = dict(defaults.get(entity_id) or {})
        
        # Inline prime sur defaults
        inline_params = inline_params_map.get(entity_id, {})
        merged_params = {**default_params, **inline_params}
        
        resolved.append({
            'id'      : entity_id,
            'params'  : merged_params,
            'callable': entity['callable'],
        })
    
    return resolved


def resolve_axis_sequence(
    lines_config: List,
    axis_type: str,
    defaults: Dict[str, Any],
    weights_config: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    """
    Résout axe multi-lignes → séquences composées.
    
    Args:
        lines_config   : Liste de configs (1 par ligne)
        axis_type      : 'gamma' | 'encoding' | 'modifier'
        defaults       : Defaults YAML pour cet axe
        weights_config : Config pondération optionnelle (gamma only)
    
    Returns:
        [{
            'id'      : str (composé),
            'params'  : dict (partagé),
            'callable': Callable (composé ou atomique),
            'sequence': List[dict] (pour traçabilité)
        }, ...]
    """
    # Résoudre chaque ligne → alternatives
    lines_resolved = []
    for line_config in lines_config:
        alternatives = resolve_axis_atomic(line_config, axis_type, defaults)
        lines_resolved.append(alternatives)
    
    # Produit cartésien vertical → séquences
    sequences = []
    for combo in product(*lines_resolved):
        # combo = tuple d'entités, 1 par ligne
        sequence_ids = [e['id'] for e in combo]
        sequence_callables = [e['callable'] for e in combo]
        
        # Params partagés (merge de tous)
        shared_params = {}
        for e in combo:
            shared_params.update(e['params'])
        
        # Séquence de 1 = atomique, >1 = composé
        if len(combo) == 1:
            # Gamma atomique — instancier si gamma, sinon juste la fonction
            if axis_type == 'gamma':
                gamma_instance = combo[0]['callable'](**combo[0]['params'])
                sequences.append({
                    'id'      : combo[0]['id'],
                    'params'  : combo[0]['params'],
                    'callable': gamma_instance,
                    'sequence': [combo[0]],
                })
            else:
                # Encoding/modifier restent fonctions (instanciés dans runner)
                sequences.append({
                    'id'      : combo[0]['id'],
                    'params'  : combo[0]['params'],
                    'callable': combo[0]['callable'],
                    'sequence': [combo[0]],
                })
        else:
            # Composition (sans weights pour l'instant)
            composed_id = '-'.join(sequence_ids)
            
            if axis_type == 'gamma':
                # Instancier gammas avec params individuels (pas shared)
                gamma_instances = [
                    e['callable'](**e['params']) for e in combo
                ]
                composed_callable = ComposedGamma(gamma_instances, weights=None)
            else:
                # Pour encoding/modifier : pas de composition pour l'instant
                # → alternatives seulement (traité avant l'appel)
                raise NotImplementedError(
                    f"Composition multi-lignes pas supportée pour {axis_type}"
                )
            
            sequences.append({
                'id'      : composed_id,
                'params'  : shared_params,
                'callable': composed_callable,
                'sequence': list(combo),
            })
    
    # Appliquer pondération si demandé (gamma only)
    if weights_config and axis_type == 'gamma':
        sequences_weighted = []
        for seq in sequences:
            n_gammas = len(seq['sequence'])
            weights_grid = _generate_weights_grid(weights_config, n_gammas)
            
            for weights in weights_grid:
                # Instancier gammas avec params
                gamma_instances = [
                    e['callable'](**seq['params']) for e in seq['sequence']
                ]
                composed_callable = ComposedGamma(gamma_instances, weights=list(weights))
                
                # ID avec weights
                sequence_ids = [e['id'] for e in seq['sequence']]
                weighted_id = _format_gamma_id(sequence_ids, list(weights))
                
                sequences_weighted.append({
                    'id'      : weighted_id,
                    'params'  : seq['params'],
                    'callable': composed_callable,
                    'sequence': seq['sequence'],
                    'weights' : list(weights),
                })
        
        return sequences_weighted
    
    return sequences


# =============================================================================
# GÉNÉRATION COMPOSITIONS
# =============================================================================

def generate_compositions(run_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Produit cartésien de tous les axes résolus.
    
    Gère :
    - Multi-lignes gamma → séquences composées (avec weights optionnel)
    - Multi-lignes encoding/modifier → alternatives seulement
    - Scalaires (n_dof, max_iterations) traités comme listes
    
    Returns:
        [{
            'gamma_id'       : str,
            'gamma_params'   : dict,
            'gamma_callable' : Callable,
            'encoding_id'    : str,
            'encoding_params': dict,
            'encoding_callable': Callable,
            'modifier_id'    : str,
            'modifier_params': dict,
            'modifier_callable': Callable,
            'n_dof'          : int,
            'max_iterations' : int,
            'phase'          : str,
        }, ...]
    """
    axes   = run_config.get('axes', {})
    phase  = run_config.get('phase', 'unknown')
    max_it = run_config.get('max_iterations', 200)
    config_mode = run_config.get('config_mode', 'default')
    
    # Charger defaults YAML avec mode
    defaults = {
        'gamma'   : load_yaml('operators', mode=config_mode),
        'encoding': load_yaml('D_encodings', mode=config_mode),
        'modifier': load_yaml('modifiers', mode=config_mode),
    }
    
    # Résoudre axes atomics
    gamma_config = axes.get('gamma', 'all')
    encoding_config = axes.get('encoding', 'all')
    modifier_config = axes.get('modifier', [{'id': 'M0'}])
    
    # Déterminer si multi-lignes ou single
    # Multi-lignes = liste de listes/configs
    # Single = 'all' | dict | liste simple
    
    # Gamma : support multi-lignes + weights
    if isinstance(gamma_config, dict) and 'sequence' in gamma_config:
        # Format explicite multi-lignes avec weights
        lines = gamma_config['sequence']
        weights_cfg = gamma_config.get('weights')
        gammas = resolve_axis_sequence(lines, 'gamma', defaults['gamma'], weights_cfg)
    elif isinstance(gamma_config, list) and any(isinstance(x, list) for x in gamma_config):
        # Multi-lignes détecté (liste de listes)
        gammas = resolve_axis_sequence(gamma_config, 'gamma', defaults['gamma'])
    else:
        # Single ligne → resolve_axis_atomic retourne fonctions create()
        # Il faut les instancier ici
        gammas_raw = resolve_axis_atomic(gamma_config, 'gamma', defaults['gamma'])
        gammas = []
        for gamma_dict in gammas_raw:
            gamma_instance = gamma_dict['callable'](**gamma_dict['params'])
            gammas.append({
                'id': gamma_dict['id'],
                'params': gamma_dict['params'],
                'callable': gamma_instance,
            })
    
    # Encoding : alternatives seulement (pas de composition pour l'instant)
    encodings = resolve_axis_atomic(encoding_config, 'encoding', defaults['encoding'])
    
    # Modifier : alternatives seulement
    modifiers = resolve_axis_atomic(modifier_config, 'modifier', defaults['modifier'])
    
    # n_dof : scalaire ou liste → toujours liste
    n_dofs_raw = run_config.get('n_dof', 10)
    n_dofs = _normalize_to_list(n_dofs_raw)
    
    # Produit cartésien final
    compositions = []
    for gamma, encoding, modifier, n_dof in product(gammas, encodings, modifiers, n_dofs):
        compositions.append({
            'gamma_id'         : gamma['id'],
            'gamma_params'     : gamma['params'],
            'gamma_callable'   : gamma['callable'],
            'encoding_id'      : encoding['id'],
            'encoding_params'  : {**encoding['params'], 'n_dof': n_dof},
            'encoding_callable': encoding['callable'],
            'modifier_id'      : modifier['id'],
            'modifier_params'  : modifier['params'],
            'modifier_callable': modifier['callable'],
            'n_dof'            : n_dof,
            'max_iterations'   : max_it,
            'phase'            : phase,
        })
    
    return compositions