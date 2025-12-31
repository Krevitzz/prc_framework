# tests/utilities/discovery.py

import importlib
import inspect
from pathlib import Path
from typing import Dict, List
import warnings

REQUIRED_ATTRIBUTES = [
    'TEST_ID',
    'TEST_CATEGORY',
    'REQUIRES_RANK',
    'D_TYPES',
    'is_applicable',
    'compute_metric',
]

def discover_active_tests() -> Dict[str, object]:
    """
    Découvre tous les tests actifs (non _deprecated).
    
    Returns:
        dict {test_id: TestSpecification module}
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
    Valide qu'un module test a tous les attributs requis.
    
    Raises:
        AssertionError si structure invalide
    """
    for attr in REQUIRED_ATTRIBUTES:
        assert hasattr(module, attr), f"Missing required attribute: {attr}"
    
    # Valider signatures
    assert callable(module.is_applicable), "is_applicable must be callable"
    assert callable(module.compute_metric), "compute_metric must be callable"
    
    # Valider types
    assert isinstance(module.TEST_ID, str), "TEST_ID must be str"
    assert isinstance(module.PARAMETERS_SPEC, dict), "PARAMETERS_SPEC must be dict"
    assert isinstance(module.SCORING_SPEC, dict), "SCORING_SPEC must be dict"
    assert isinstance(module.THRESHOLDS_SPEC, dict), "THRESHOLDS_SPEC must be dict"