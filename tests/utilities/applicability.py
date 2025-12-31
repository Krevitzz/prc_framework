# prc_framework/utilities/applicability.py

from typing import Tuple, Dict, Any, Callable

# Registre extensible de validators
VALIDATORS: Dict[str, Callable] = {
    'requires_rank': lambda run_metadata, expected: 
        len(run_metadata['state_shape']) == expected,
    
    'requires_square': lambda run_metadata, _: 
        len(run_metadata['state_shape']) == 2 and 
        run_metadata['state_shape'][0] == run_metadata['state_shape'][1],
    
    'allowed_d_types': lambda run_metadata, allowed: 
        'ALL' in allowed or run_metadata['d_base_id'].split('-')[0] in allowed,
    
    'requires_even_dimension': lambda run_metadata, _: 
        all(dim % 2 == 0 for dim in run_metadata['state_shape']),
    
    'minimum_dimension': lambda run_metadata, min_dim:
        all(dim >= min_dim for dim in run_metadata['state_shape']),
}


def check(test_module, run_metadata: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Vérifie applicabilité sur métadonnées uniquement.
    
    Args:
        test_module: Module test importé
        run_metadata: {
            'gamma_id': str,
            'd_base_id': str,
            'modifier_id': str,
            'seed': int,
            'state_shape': tuple,  # Déduit du premier snapshot
        }
    
    Returns:
        (applicable: bool, reason: str)
    
    Examples:
        >>> check(test_sym_001, {'state_shape': (10, 10), 'd_base_id': 'SYM-001', ...})
        (True, "")
        
        >>> check(test_sym_001, {'state_shape': (10, 20), 'd_base_id': 'SYM-001', ...})
        (False, "requires_square = True non satisfait")
    """
    spec = test_module.APPLICABILITY_SPEC
    
    for constraint_name, constraint_value in spec.items():
        if constraint_name not in VALIDATORS:
            return False, f"Contrainte inconnue : {constraint_name}"
        
        validator = VALIDATORS[constraint_name]
        
        try:
            is_valid = validator(run_metadata, constraint_value)
            
            if not is_valid:
                return False, f"{constraint_name} = {constraint_value} non satisfait"
        
        except KeyError as e:
            return False, f"Info manquante : {e}"
        
        except Exception as e:
            return False, f"Erreur validation {constraint_name}: {e}"
    
    return True, ""


def add_validator(name: str, validator: Callable) -> None:
    """
    Ajoute un validator custom.
    
    Args:
        name: Nom unique du validator
        validator: Fonction (run_metadata, constraint_value) -> bool
    
    Raises:
        ValueError: Si nom déjà existant
    """
    if name in VALIDATORS:
        raise ValueError(f"Validator '{name}' déjà existant")
    
    VALIDATORS[name] = validator