
# tests/utilities/validation.py

def validate_test_output(result: dict, test_name: str) -> bool:
    """
    Valide qu'un résultat de test est conforme.
    
    Returns:
        True si valide, False sinon (avec warnings)
    """
    required_keys = {
        'test_name', 'status', 'message',
        'initial_value', 'final_value', 'transition',
        'value', 'metadata'
    }
    
    # Vérifier présence clés
    missing = required_keys - set(result.keys())
    if missing:
        print(f"❌ {test_name}: Clés manquantes: {missing}")
        return False
    
    # Vérifier types
    if not isinstance(result['test_name'], str):
        print(f"❌ {test_name}: test_name doit être str")
        return False
    
    if result['status'] not in ['SUCCESS', 'ERROR', 'NOT_APPLICABLE']:
        print(f"❌ {test_name}: status invalide '{result['status']}'")
        return False
    
    # Vérifier interdictions
    forbidden_keys = {'blocking', 'verdict', 'passed', 'score', 'weight'}
    found_forbidden = forbidden_keys & set(result.keys())
    if found_forbidden:
        print(f"❌ {test_name}: Clés interdites: {found_forbidden}")
        return False
    
    # Vérifier sérialisabilité JSON
    try:
        json.dumps(result)
    except (TypeError, ValueError) as e:
        print(f"❌ {test_name}: Non JSON-sérialisable: {e}")
        return False
    
    return True