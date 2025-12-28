# tests/utilities/discovery.py

def run_all_applicable_tests(history, D_base, d_base_id, gamma_id):
    results = {}
    
    for test_id, test_meta in tests.items():
        result = test_meta['run'](history, context)
        
        # VALIDATION AUTOMATIQUE
        if not validate_test_output(result, test_id):
            print(f"⚠️  {test_id} retourne format invalide, skip")
            continue
        
        results[test_id] = result
    
    return results