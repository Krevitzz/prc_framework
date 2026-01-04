#!/usr/bin/env python3
"""
Test Pipeline Charter 5.4 - Local

Validation unitaire + intégration verdict_engine.py existant.
Pas de modification structure - Test le code tel quel.
"""

import sys
import json
import numpy as np
from pathlib import Path

# Setup imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.utilities.test_engine import TestEngine
from tests.utilities.verdict_engine import (
    analyze_metric_patterns,
    compute_metric_quality,
    decide_verdict_from_patterns,
    generate_actionable_insights
)
from tests import test_uni_001


################################################################################
# HELPERS
################################################################################

def create_fake_scores_data(scenario='systematic_failure'):
    """
    Crée fake data format attendu par verdict_engine.py (doc 54).
    
    Format : Liste avec structure exacte de load_scores_with_context()
    
    Args:
        scenario: 'systematic_failure' | 'contextual' | 'survives'
    
    Returns:
        List[dict] format identique à sortie DB jointure
    """
    scores_data = []
    
    # Scénarios
    if scenario == 'systematic_failure':
        # 3 métriques échouent sur 100% runs
        for i in range(10):
            scores_data.append({
                'exec_id': 100 + i,
                'run_id': f'GAM-001_beta2.0_SYM-001_M0_s{i}',
                'gamma_id': 'GAM-001',
                'd_base_id': 'SYM-001',
                'modifier_id': 'M0',
                'seed': i,
                'test_name': 'UNIV-001',
                'test_score': 0.92,
                'metric_scores': {
                    'metric_A': {
                        'score': 0.92,
                        'flag': True,
                        'pathology_type': 'S2_EXPLOSION',
                        'weight': 2.0
                    },
                    'metric_B': {
                        'score': 0.88,
                        'flag': True,
                        'pathology_type': 'S2_EXPLOSION',
                        'weight': 2.0
                    },
                    'metric_C': {
                        'score': 0.85,
                        'flag': True,
                        'pathology_type': 'S1_COLLAPSE',
                        'weight': 1.5
                    }
                },
                'pathology_flags': ['metric_A', 'metric_B', 'metric_C'],
                'critical_metrics': ['metric_A', 'metric_B', 'metric_C'],
                'aggregation_mode': 'max'
            })
    
    elif scenario == 'contextual':
        # Échoue sur SYM-001, OK sur ASY-001
        for d_base in ['SYM-001', 'ASY-001']:
            for i in range(5):
                is_sym = d_base == 'SYM-001'
                scores_data.append({
                    'exec_id': 200 + len(scores_data),
                    'run_id': f'GAM-002_beta2.0_{d_base}_M0_s{i}',
                    'gamma_id': 'GAM-002',
                    'd_base_id': d_base,
                    'modifier_id': 'M0',
                    'seed': i,
                    'test_name': 'UNIV-001',
                    'test_score': 0.85 if is_sym else 0.12,
                    'metric_scores': {
                        'metric_A': {
                            'score': 0.85 if is_sym else 0.12,
                            'flag': is_sym,
                            'pathology_type': 'S2_EXPLOSION',
                            'weight': 2.0
                        }
                    },
                    'pathology_flags': ['metric_A'] if is_sym else [],
                    'critical_metrics': ['metric_A'] if is_sym else [],
                    'aggregation_mode': 'max'
                })
    
    elif scenario == 'survives':
        # Aucune pathologie
        for i in range(10):
            scores_data.append({
                'exec_id': 300 + i,
                'run_id': f'GAM-003_beta2.0_SYM-001_M0_s{i}',
                'gamma_id': 'GAM-003',
                'd_base_id': 'SYM-001',
                'modifier_id': 'M0',
                'seed': i,
                'test_name': 'UNIV-001',
                'test_score': 0.15,
                'metric_scores': {
                    'metric_A': {
                        'score': 0.15,
                        'flag': False,
                        'pathology_type': 'S2_EXPLOSION',
                        'weight': 2.0
                    }
                },
                'pathology_flags': [],
                'critical_metrics': [],
                'aggregation_mode': 'max'
            })
    
    return scores_data


################################################################################
# TESTS
################################################################################

def test_testengine_isolation():
    """Test TestEngine sur données fake."""
    print("="*80)
    print("TEST 1 : TestEngine Isolation")
    print("="*80)
    
    # Fake history
    history = [np.random.randn(10, 10) for _ in range(10)]
    
    run_metadata = {
        'gamma_id': 'GAM-TEST',
        'd_base_id': 'SYM-001',
        'modifier_id': 'M0',
        'seed': 1,
        'state_shape': (10, 10)
    }
    
    engine = TestEngine()
    result = engine.execute_test(
        test_uni_001,
        run_metadata,
        history,
        'params_default_v1'
    )
    
    # Vérifications
    assert result['status'] == 'SUCCESS', f"Status: {result['status']}"
    assert 'frobenius_norm' in result['statistics'], "Métrique manquante"
    
    print("✓ TestEngine OK")
    print(f"  - Status: {result['status']}")
    print(f"  - Métriques: {list(result['statistics'].keys())}")
    print()


def test_verdict_systematic_failure():
    """Test détection systematic_failures."""
    print("="*80)
    print("TEST 2 : Verdict - Systematic Failure")
    print("="*80)
    
    # Créer data avec 3 métriques échouant systématiquement
    scores_data = create_fake_scores_data('systematic_failure')
    
    print(f"  Données: {len(scores_data)} runs")
    print(f"  Format: {list(scores_data[0].keys())}")
    
    # Analyser patterns
    patterns = analyze_metric_patterns(scores_data)
    
    print("\n  Patterns détectés:")
    for pattern_type, values in patterns.items():
        if values:
            count = len(values) if isinstance(values, (list, dict)) else 0
            print(f"    - {pattern_type}: {count}")
    
    # Vérifier systematic_failures détectés
    systematic = patterns['systematic_failures']
    assert len(systematic) == 3, \
        f"Attendu 3 systematic_failures, trouvé {len(systematic)}"
    
    # Vérifier verdict
    metric_quality = compute_metric_quality(scores_data)
    verdict, reason = decide_verdict_from_patterns(patterns, metric_quality)
    
    assert verdict == "REJECTED[R0]", \
        f"Attendu REJECTED[R0], reçu {verdict}"
    
    print(f"\n  ✓ Systematic failures: {len(systematic)}")
    print(f"  ✓ Verdict: {verdict}")
    print()


def test_verdict_contextual():
    """Test détection comportements contextuels."""
    print("="*80)
    print("TEST 3 : Verdict - Contextual Behavior")
    print("="*80)
    
    # Créer data avec corrélation D
    scores_data = create_fake_scores_data('contextual')
    
    print(f"  Données: {len(scores_data)} runs")
    
    # Analyser patterns
    patterns = analyze_metric_patterns(scores_data)
    
    print("\n  Patterns détectés:")
    for pattern_type, values in patterns.items():
        if values:
            count = len(values) if isinstance(values, (list, dict)) else 0
            print(f"    - {pattern_type}: {count}")
    
    # Vérifier D_CORRELATED détecté
    assert patterns['d_correlated'], "D_CORRELATED devrait être détecté"
    
    # Vérifier verdict
    metric_quality = compute_metric_quality(scores_data)
    verdict, reason = decide_verdict_from_patterns(patterns, metric_quality)
    
    assert verdict == "WIP[R0-open]", \
        f"Attendu WIP[R0-open], reçu {verdict}"
    
    print(f"\n  ✓ D_CORRELATED: {len(patterns['d_correlated'])}")
    print(f"  ✓ Verdict: {verdict}")
    print()


def test_verdict_survives():
    """Test verdict SURVIVES."""
    print("="*80)
    print("TEST 4 : Verdict - SURVIVES")
    print("="*80)
    
    # Créer data sans pathologies
    scores_data = create_fake_scores_data('survives')
    
    print(f"  Données: {len(scores_data)} runs")
    
    # Analyser patterns
    patterns = analyze_metric_patterns(scores_data)
    
    print("\n  Patterns détectés:")
    for pattern_type, values in patterns.items():
        if values:
            count = len(values) if isinstance(values, (list, dict)) else 0
            print(f"    - {pattern_type}: {count}")
    
    # Vérifier aucun pattern critique
    assert not patterns['systematic_failures'], \
        "Aucun systematic_failure attendu"
    
    # Vérifier verdict
    metric_quality = compute_metric_quality(scores_data)
    verdict, reason = decide_verdict_from_patterns(patterns, metric_quality)
    
    assert verdict == "SURVIVES[R0]", \
        f"Attendu SURVIVES[R0], reçu {verdict}"
    
    print(f"\n  ✓ Aucune pathologie systématique")
    print(f"  ✓ Verdict: {verdict}")
    print()


def test_insights_generation():
    """Test génération insights."""
    print("="*80)
    print("TEST 5 : Génération Insights")
    print("="*80)
    
    # Data avec patterns variés
    scores_data = create_fake_scores_data('contextual')
    
    patterns = analyze_metric_patterns(scores_data)
    metric_quality = compute_metric_quality(scores_data)
    
    insights = generate_actionable_insights(patterns, metric_quality)
    
    assert insights, "Insights devraient être générés"
    
    print(f"  ✓ {len(insights)} insights générés")
    for i, insight in enumerate(insights[:3], 1):
        print(f"    {i}. {insight[:70]}...")
    print()


def test_metric_quality():
    """Test évaluation qualité métriques."""
    print("="*80)
    print("TEST 6 : Qualité Métriques")
    print("="*80)
    
    scores_data = create_fake_scores_data('systematic_failure')
    
    quality = compute_metric_quality(scores_data)
    
    assert quality, "Qualité métriques devrait être calculée"
    
    print(f"  ✓ {len(quality)} métriques évaluées")
    for metric_key, q in list(quality.items())[:3]:
        print(f"    - {metric_key}:")
        print(f"      valid={q['valid']}, mean={q['mean_score']:.2f}")
    print()


################################################################################
# MAIN
################################################################################

def main():
    """Exécute tous les tests."""
    print("\n" + "#"*80)
    print("# TEST PIPELINE CHARTER 5.4")
    print("#"*80)
    print("\nValidation verdict_engine.py existant (doc 54)")
    print()
    
    try:
        test_testengine_isolation()
        test_verdict_systematic_failure()
        test_verdict_contextual()
        test_verdict_survives()
        test_insights_generation()
        test_metric_quality()
        
        print("#"*80)
        print("# ✓ TOUS LES TESTS PASSENT")
        print("#"*80)
        print("\nPipeline validée. Verdict engine fonctionne correctement.")
        print()
        
    except AssertionError as e:
        print("\n" + "#"*80)
        print("# ❌ ÉCHEC TEST")
        print("#"*80)
        print(str(e))
        raise


if __name__ == "__main__":
    main()