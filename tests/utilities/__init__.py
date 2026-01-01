# tests/utilities/__init__.py
"""
Utilitaires framework PRC.

Architecture Charter 5.4
"""

# Discovery & Applicability
from .discovery import discover_active_tests, validate_test_structure
from .applicability import check as check_applicability, add_validator

# Test Engine
from .test_engine import TestEngine

# Config Management
from .config_loader import ConfigLoader, get_loader

# Registres
from .registries import (
    BaseRegistry,
    register_function,
    RegistryManager,
    get_post_processor,
    add_post_processor,
)

__all__ = [
    # Discovery
    'discover_active_tests',
    'validate_test_structure',
    
    # Applicability
    'check_applicability',
    'add_validator',
    
    # Test Engine
    'TestEngine',
    
    # Config
    'ConfigLoader',
    'get_loader',
    
    # Registres
    'BaseRegistry',
    'register_function',
    'RegistryManager',
    'get_post_processor',
    'add_post_processor',
]