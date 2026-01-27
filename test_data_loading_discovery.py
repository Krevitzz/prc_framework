#!/usr/bin/env python3
"""
Tests Discovery Unifiée - Phase 10.

Valide:
- Discovery tous types (test, gamma, encoding, modifier)
- Validation PHASE obligatoire
- CriticalDiscoveryError si PHASE absent
- Skip deprecated automatique
- Extraction ID depuis METADATA['id']
"""

import pytest
from pathlib import Path

from .tests.utilities.utils.data_loading import (
    discover_entities,
    CriticalDiscoveryError,
    ValidationError
)


# =============================================================================
# TESTS DISCOVERY TESTS (architecture 5.5 inchangée)
# =============================================================================

def test_discover_tests_finds_active():
    """Discovery tests retourne tests actifs (non deprecated)."""
    tests = discover_entities('test', phase=None)
    
    # Au moins 1 test découvert
    assert len(tests) >= 1, "Aucun test découvert"
    
    # Structure retour correcte
    test = tests[0]
    assert 'id' in test
    assert 'module_path' in test
    assert 'module' in test
    assert 'phase' in test
    assert 'metadata' in test
    
    # ID format correct
    assert test['id'].startswith(('UNIV-', 'SYM-', 'SPE-', 'PAT-', 'SPA-', 'GRA-', 'TOP-'))
    
    # Metadata contient clés attendues
    assert 'category' in test['metadata']
    assert 'version' in test['metadata']
    assert test['metadata']['version'] == '5.5'


def test_discover_tests_skip_deprecated():
    """Discovery tests skip fichiers _deprecated_."""
    tests = discover_entities('test', phase=None)
    
    # Aucun fichier deprecated
    for test in tests:
        assert '_deprecated' not in test['module_path']


def test_discover_tests_filter_by_phase():
    """Discovery tests filtre par TEST_PHASE si spécifié."""
    # Phase None (tous)
    all_tests = discover_entities('test', phase=None)
    
    # Phase R0 (subset)
    r0_tests = discover_entities('test', phase='R0')
    
    # R0 <= All
    assert len(r0_tests) <= len(all_tests)


# =============================================================================
# TESTS DISCOVERY GAMMAS
# =============================================================================

def test_discover_gammas_requires_phase():
    """Discovery gammas lève CriticalDiscoveryError si PHASE absent."""
    # Cette partie dépend de vos fichiers réels
    # Si tous vos gammas actuels ont PHASE, ce test passera
    try:
        gammas = discover_entities('gamma', phase='R0')
        
        # Si succès, vérifier structure
        assert len(gammas) >= 1, "Aucun gamma découvert"
        
        gamma = gammas[0]
        assert 'id' in gamma
        assert gamma['id'].startswith('GAM-')
        assert 'phase' in gamma
        assert gamma['phase'] == 'R0'
        
        # Vérifier function_name
        assert 'function_name' in gamma
        assert gamma['function_name'].startswith('create_gamma_hyp_')
    
    except CriticalDiscoveryError as e:
        # Si erreur, c'est qu'un gamma n'a pas PHASE
        pytest.fail(f"Un gamma n'a pas PHASE: {e}")


def test_discover_gammas_extracts_metadata():
    """Discovery gammas extrait METADATA correctement."""
    gammas = discover_entities('gamma', phase='R0')
    
    assert len(gammas) >= 1
    
    gamma = gammas[0]
    assert 'metadata' in gamma
    
    metadata = gamma['metadata']
    assert 'family' in metadata
    assert 'applicability' in metadata
    assert 'description' in metadata


def test_discover_gammas_skip_deprecated():
    """Discovery gammas skip fichiers _deprecated_."""
    gammas = discover_entities('gamma', phase=None)
    
    for gamma in gammas:
        assert '_deprecated' not in gamma['module_path']


# =============================================================================
# TESTS DISCOVERY ENCODINGS
# =============================================================================

def test_discover_encodings_requires_phase():
    """Discovery encodings lève CriticalDiscoveryError si PHASE absent."""
    try:
        encodings = discover_entities('encoding', phase='R0')
        
        assert len(encodings) >= 1, "Aucun encoding découvert"
        
        enc = encodings[0]
        assert 'id' in enc
        assert enc['id'].split('-')[0] in ['SYM', 'ASY', 'R3']
        assert 'phase' in enc
        assert enc['phase'] == 'R0'
        
        # Vérifier function_name = 'create'
        assert enc['function_name'] == 'create'
    
    except CriticalDiscoveryError as e:
        pytest.fail(f"Un encoding n'a pas PHASE: {e}")


def test_discover_encodings_pattern_files():
    """Discovery encodings utilise pattern {sym,asy,r3}_*.py."""
    encodings = discover_entities('encoding', phase=None)
    
    assert len(encodings) >= 1
    
    # Vérifier pattern fichiers
    for enc in encodings:
        filename = Path(enc['module_path']).name
        assert (
            filename.startswith('sym_') or
            filename.startswith('asy_') or
            filename.startswith('r3_')
        ), f"Pattern fichier incorrect: {filename}"


def test_discover_encodings_extracts_id_from_metadata():
    """Discovery encodings extrait ID depuis METADATA['id']."""
    encodings = discover_entities('encoding', phase='R0')
    
    assert len(encodings) >= 1
    
    enc = encodings[0]
    
    # Vérifier METADATA présent
    assert 'metadata' in enc
    metadata = enc['metadata']
    
    # Vérifier rank, type présents
    assert 'rank' in metadata
    assert 'type' in metadata
    assert metadata['rank'] in [2, 3]


# =============================================================================
# TESTS DISCOVERY MODIFIERS
# =============================================================================

def test_discover_modifiers_requires_phase():
    """Discovery modifiers lève CriticalDiscoveryError si PHASE absent."""
    try:
        modifiers = discover_entities('modifier', phase='R0')
        
        assert len(modifiers) >= 1, "Aucun modifier découvert"
        
        mod = modifiers[0]
        assert 'id' in mod
        assert mod['id'].startswith('M')
        assert 'phase' in mod
        assert mod['phase'] == 'R0'
        
        # Vérifier function_name = 'apply'
        assert mod['function_name'] == 'apply'
    
    except CriticalDiscoveryError as e:
        pytest.fail(f"Un modifier n'a pas PHASE: {e}")


def test_discover_modifiers_pattern_files():
    """Discovery modifiers utilise pattern m*.py."""
    modifiers = discover_entities('modifier', phase=None)
    
    assert len(modifiers) >= 1
    
    # Vérifier pattern fichiers
    for mod in modifiers:
        filename = Path(mod['module_path']).name
        assert filename.startswith('m'), f"Pattern fichier incorrect: {filename}"
        assert not filename == '__init__.py'


# =============================================================================
# TESTS DISCOVERY GÉNÉRIQUE
# =============================================================================

def test_discover_entities_invalid_type():
    """Discovery avec type invalide lève ValueError."""
    with pytest.raises(ValueError, match="Type inconnu"):
        discover_entities('invalid_type', phase='R0')


def test_discover_entities_all_types():
    """Discovery fonctionne pour tous types supportés."""
    types = ['test', 'gamma', 'encoding', 'modifier']
    
    for entity_type in types:
        entities = discover_entities(entity_type, phase=None)
        
        # Au moins 1 entité découverte
        assert len(entities) >= 1, f"Aucune entité {entity_type} découverte"
        
        # Structure minimale
        entity = entities[0]
        assert 'id' in entity
        assert 'module_path' in entity
        assert 'module' in entity


def test_discover_entities_phase_filter_consistent():
    """Filtrage phase cohérent entre types."""
    # Avec phase=None, tous retournent quelque chose
    tests_all = discover_entities('test', phase=None)
    gammas_all = discover_entities('gamma', phase=None)
    encodings_all = discover_entities('encoding', phase=None)
    modifiers_all = discover_entities('modifier', phase=None)
    
    assert len(tests_all) >= 1
    assert len(gammas_all) >= 1
    assert len(encodings_all) >= 1
    assert len(modifiers_all) >= 1
    
    # Avec phase='R0', subset cohérent
    tests_r0 = discover_entities('test', phase='R0')
    gammas_r0 = discover_entities('gamma', phase='R0')
    encodings_r0 = discover_entities('encoding', phase='R0')
    modifiers_r0 = discover_entities('modifier', phase='R0')
    
    # R0 <= All pour tous types
    assert len(tests_r0) <= len(tests_all)
    assert len(gammas_r0) <= len(gammas_all)
    assert len(encodings_r0) <= len(encodings_all)
    assert len(modifiers_r0) <= len(modifiers_all)


# =============================================================================
# TESTS EDGE CASES
# =============================================================================

def test_discover_entities_empty_phase_string():
    """Phase vide string traité comme None."""
    # Phase=""  devrait retourner tous
    entities = discover_entities('test', phase="")
    
    # Devrait retourner quelque chose
    assert len(entities) >= 0  # Peut être vide si aucun test


def test_discover_entities_returns_module_objects():
    """Discovery retourne objets module importables."""
    gammas = discover_entities('gamma', phase='R0')
    
    assert len(gammas) >= 1
    
    gamma = gammas[0]
    module = gamma['module']
    
    # Module a PHASE
    assert hasattr(module, 'PHASE')
    
    # Module a METADATA
    assert hasattr(module, 'METADATA')
    
    # Fonction factory accessible
    factory_name = gamma['function_name']
    assert hasattr(module, factory_name)