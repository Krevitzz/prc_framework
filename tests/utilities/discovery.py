# tests/utilities/discovery.py
"""
Discovery automatique tests actifs.

Architecture Charter 5.5 - Section 12.10
"""

import importlib
import inspect
from pathlib import Path
from typing import Dict
import warnings

# Attributs requis architecture 5.5
REQUIRED_ATTRIBUTES = [
    'TEST_ID',
    'TEST_CATEGORY',
    'TEST_VERSION',
    'APPLICABILITY_SPEC',
    'COMPUTATION_SPECS',
]

def discover_active_tests() -> Dict[str, object]:
    """
    Découvre tous les tests actifs (non _deprecated).
    
    Returns:
        dict {test_id: module}
    
    Examples:
        >>> tests = discover_active_tests()
        >>> print(list(tests.keys()))
        ['UNIV-001', 'SYM-001', ...]
    """
    tests_dir = Path(__file__).parent.parent
    test_files = tests_dir.glob('test_*.py')
    
    active_tests = {}
    
    for test_file in test_files:
        # Skip deprecated
        if '_deprecated' in test_file.stem:
            continue
        
        # Charger module
        module_name = f'tests.{test_file.stem}'
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            warnings.warn(f"Failed to import {module_name}: {e}")
            continue
        
        # Valider structure
        try:
            validate_test_structure(module)
            test_id = module.TEST_ID
            active_tests[test_id] = module
        except AssertionError as e:
            warnings.warn(f"Invalid test structure {module_name}: {e}")
            continue
    
    return active_tests


def validate_test_structure(module) -> None:
    """
    Valide qu'un module test a structure 5.4 conforme.
    
    Args:
        module: Module test importé
    
    Raises:
        AssertionError: Si structure invalide
    """
    # 1. Attributs requis
    for attr in REQUIRED_ATTRIBUTES:
        assert hasattr(module, attr), f"Missing required attribute: {attr}"
    
    # 2. Types corrects
    assert isinstance(module.TEST_ID, str), "TEST_ID must be str"
    assert isinstance(module.TEST_CATEGORY, str), "TEST_CATEGORY must be str"
    assert isinstance(module.TEST_VERSION, str), "TEST_VERSION must be str"
    assert isinstance(module.APPLICABILITY_SPEC, dict), "APPLICABILITY_SPEC must be dict"
    assert isinstance(module.COMPUTATION_SPECS, dict), "COMPUTATION_SPECS must be dict"
    
    # 3. Version 5.4 obligatoire
    assert module.TEST_VERSION == "5.5", f"TEST_VERSION must be '5.5', got '{module.TEST_VERSION}'"
    
    # 4. TEST_ID format CAT-NNN
    import re
    assert re.match(r'^[A-Z]{3,4}-\d{3}$', module.TEST_ID), \
        f"TEST_ID invalid format: {module.TEST_ID} (expected CAT-NNN)"
    
    # 5. COMPUTATION_SPECS non vide
    assert len(module.COMPUTATION_SPECS) > 0, "COMPUTATION_SPECS must not be empty"
    
    # 6. COMPUTATION_SPECS entre 1 et 5 métriques
    assert 1 <= len(module.COMPUTATION_SPECS) <= 5, \
        f"COMPUTATION_SPECS must have 1-5 metrics, got {len(module.COMPUTATION_SPECS)}"
    
    # 7. Chaque métrique a registry_key et default_params
    for metric_name, spec in module.COMPUTATION_SPECS.items():
        assert 'registry_key' in spec, \
            f"Metric '{metric_name}' missing 'registry_key'"
        assert 'default_params' in spec, \
            f"Metric '{metric_name}' missing 'default_params'"
        assert '.' in spec['registry_key'], \
            f"Metric '{metric_name}' registry_key must be 'registry.function' format"
    
    # 8. Pas de FORMULAS legacy
    assert not hasattr(module, 'FORMULAS'), \
        "FORMULAS attribute is obsolete in 5.4, use COMPUTATION_SPECS"
    
    # 9. Pas de is_applicable/compute_metric legacy
    assert not hasattr(module, 'is_applicable'), \
        "is_applicable() is obsolete in 5.5, use APPLICABILITY_SPEC"
    assert not hasattr(module, 'compute_metric'), \
        "compute_metric() is obsolete in 5.5, use COMPUTATION_SPECS with registries"