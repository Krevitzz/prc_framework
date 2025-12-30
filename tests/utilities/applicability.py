# tests/utilities/applicability.py

def filter_applicable_tests(
    all_tests: Dict[str, object],
    context: dict
) -> Dict[str, object]:
    """
    Filtre tests applicables pour ce contexte.
    
    Args:
        all_tests: dict {test_id: module}
        context: {
            'd_base_id': str,
            'gamma_id': str,
            'modifier_id': str,
            'seed': int,
            'state_shape': tuple,
        }
    
    Returns:
        dict {test_id: module} filtrée
    """
    applicable_tests = {}
    
    for test_id, test_module in all_tests.items():
        applicable, reason = test_module.is_applicable(context)
        
        if applicable:
            applicable_tests[test_id] = test_module
        else:
            # Log pour traçabilité
            log.debug(f"Test {test_id} not applicable: {reason}")
    
    return applicable_tests