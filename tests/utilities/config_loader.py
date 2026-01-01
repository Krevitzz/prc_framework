 tests/utilities/config_loader.py

def load_test_parameters(
    test_id: str,
    global_config_id: str
) -> dict:
    """
    Charge params avec fusion global + spécifique.
    
    Args:
        test_id: Ex "UNIV-001"
        global_config_id: Ex "params_default_v1"
    
    Returns:
        dict params fusionnés
    """
    # 1. Charger global
    global_path = f"tests/config/global/{global_config_id}.yaml"
    global_params = load_yaml(global_path)
    
    # 2. Chercher override spécifique (auto-détection)
    specific_path = f"tests/config/tests/{test_id}/params_*.yaml"
    specific_files = glob.glob(specific_path)
    
    if not specific_files:
        # Pas d'override → retourner global
        return global_params
    
    # 3. Charger spécifique (prendre dernier si plusieurs)
    specific_params = load_yaml(sorted(specific_files)[-1])
    
    # 4. Merge: specific override global
    merged = {**global_params, **specific_params}
    
    return merged