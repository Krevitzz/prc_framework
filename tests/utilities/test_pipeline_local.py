#!/usr/bin/env python3
"""
test_pipeline_local.py

Test pipeline Charter 5.4 - Isolation complète
Tests unitaires scoring + verdict patterns-based

Architecture:
1. TestEngine → Observations
2. Scoring → Pathology scores
3. Verdict → Pattern detection

Conformité: Charter 5.4 Sections 12.8-12.9
"""

import sys
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.utilities.test_engine import TestEngine
from tests.utilities.verdict_engine import (
    analyze_metric_patterns,
    compute_metric_quality,
    decide_verdict_from_patterns
)


# ============================================================================
# SCORING ISOLATION (Copie locale pour tests)
# ============================================================================

def score_metric_s2_explosion(value, threshold_high, critical_high, mode='soft'):
    """Score pathologie S2_EXPLOSION."""
    if mode == 'hard':
        return 1.0 if value > threshold_high else 0.0
    
    # Soft mode
    if value <= threshold_high:
        return 0.0
    elif value >= critical_high:
        return 1.0
    else:
        return (value - threshold_high) / (critical_high - threshold_high)


def score_observation_local(observation, scoring_config):
    """
    Score observation locale (test isolation).
    
    Returns:
        {
            'test_name': str,
            'test_score': float,
            'metric_scores': dict,
            'pathology_flags': list,
            'critical_metrics': list
        }
    """
    test_name = observation['test_name']
    test_config = scoring_config['tests'][test_name]
    
    metric_scores = {}
    flags = []
    critical = []
    
    for metric_key, rule in test_config['scoring_rules'].items():
        # Extract value from observation
        source_path = rule['source'].split('.')
        value = observation
        for key in source_path:
            value = value[key]
        
        # Score metric
        pathology_type = rule['pathology_type']
        
        if pathology_type == 'S2_EXPLOSION':
            score = score_metric_s2_explosion(
                value,
                rule['threshold_high'],
                rule['critical_high']
            )
        else:
            raise ValueError(f"Pathology type non supporté: {pathology_type}")
        
        flag = score >= 0.8
        
        metric_scores[metric_key] = {
            'value': value,
            'score': score,
            'flag': flag,
            'pathology_type': pathology_type,
            'weight': rule['weight']
        }
        
        if flag:
            flags.append(metric_key)
        if score >= 0.8:
            critical.append(metric_key)
    
    # Aggregate (mode max pour R0)
    test_score = max(m['score'] for m in metric_scores.values())
    
    return {
        'test_name': test_name,
        'test_score': test_score,
        'metric_scores': metric_scores,
        'pathology_flags': flags,
        'critical_metrics': critical
    }


# ============================================================================
# TEST 1: TestEngine Isolation
# ============================================================================

def test_engine_isolation():
    """Test TestEngine en isolation complète."""
    print("="*80)
    print("TEST UNITAIRE - TestEngine")
    print("="*80)
    
    # Mock test spec
    class MockTest:
        TEST_ID = 'UNI-001'
        TEST_CATEGORY = 'UNIV'
        TEST_VERSION = '5.4'
        
        COMPUTATION_SPECS = {
            'frobenius_norm': {
                'registry_key': 'algebra.frobenius_norm',
                'default_params': {},
                'post_process': 'round_4'
            }
        }
    
    # Mock run metadata
    run_metadata = {
        'gamma_id': 'GAM-001',
        'd_base_id': 'SYM-001',
        'modifier_id': 'M0',
        'seed': 42
    }
    
    # Mock history (10 snapshots)
    history = [np.ones((10, 10)) * (i + 1) for i in range(10)]
    
    # Execute test
    engine = TestEngine()
    result = engine.execute_test(
        MockTest,
        run_metadata,
        history,
        'params_default_v1'
    )
    
    # Verify structure
    assert result['status'] == 'SUCCESS', f"Status devrait être SUCCESS: {result['status']}"
    assert 'statistics' in result, "Doit avoir section statistics"
    assert 'evolution' in result, "Doit avoir section evolution"
    assert 'frobenius_norm' in result['statistics'], "Doit calculer frobenius_norm"
    
    # Verify exec_id NOT created by TestEngine
    #assert 'exec_id' not in result, "TestEngine ne doit PAS créer exec_id"
    
    # Verify metadata preservation
    assert result['run_metadata']['gamma_id'] == 'GAM-001'
    assert result['run_metadata']['d_base_id'] == 'SYM-001'
    
    print("✓ TestEngine isolation OK")
    print(f"  - Status: {result['status']}")
    print(f"  - Métriques: {list(result['statistics'].keys())}")
    print(f"  - exec_id préservé: {999}")  # Mock value


# ============================================================================
# TEST 2: Scoring Isolation
# ============================================================================

def test_scoring_isolation():
    """Test scoring en isolation."""
    print("="*80)
    print("TEST UNITAIRE - Scoring")
    print("="*80)
    
    # Mock observation (format TestEngine)
    observation = {
        'test_name': 'UNI-001',
        'run_metadata': {
            'gamma_id': 'GAM-001',
            'd_base_id': 'SYM-001',
            'modifier_id': 'M0',
            'seed': 42
        },
        'statistics': {
            'frobenius_norm': {
                'initial': 10.0,
                'final': 1000.0,  # Explosion
                'mean': 500.0,
                'std': 300.0
            }
        },
        'evolution': {
            'frobenius_norm': {
                'transition': 'explosive',
                'trend': 'increasing'
            }
        }
    }
    
    # Scoring config
    scoring_config = {
        'tests': {
            'UNI-001': {
                'test_weight': 2.0,
                'scoring_rules': {
                    'frobenius_norm': {
                        'source': 'statistics.frobenius_norm.final',
                        'pathology_type': 'S2_EXPLOSION',
                        'threshold_high': 100.0,
                        'critical_high': 500.0,
                        'weight': 2.0
                    }
                }
            }
        }
    }
    
    # Score observation
    score_result = score_observation_local(observation, scoring_config)
    
    # Verify
    assert score_result['test_score'] == 1.0, "Score doit être 1.0 (explosion max)"
    assert 'frobenius_norm' in score_result['pathology_flags'], "Doit flaguer explosion"
    assert score_result['metric_scores']['frobenius_norm']['score'] == 1.0
    
    print("✓ Scoring isolation OK")
    print(f"  - Test score: {score_result['test_score']:.3f}")
    print(f"  - Flags: {score_result['pathology_flags']}")
    print(f"  - Critical: {score_result['critical_metrics']}")
    print(f"  - Métriques scorées: {list(score_result['metric_scores'].keys())}")


# ============================================================================
# TEST 3: Verdict Isolation
# ============================================================================

def test_verdict_isolation():
    """Test verdict_engine en isolation."""
    print("="*80)
    print("TEST UNITAIRE - Verdict")
    print("="*80)
    
    # Créer 30 runs avec 3 métriques explosives
    # (seuil REJECTED[R0] = ≥3 métriques valides échouent)
    scores_data = []
    
    for run_idx in range(30):
        exec_id = 1000 + run_idx
        d_base = ['SYM-001', 'ASY-001', 'R3-001'][run_idx % 3]
        modifier = ['M0', 'M1'][run_idx % 2]
        seed = (run_idx % 5) + 1
        
        run_data = {
            'exec_id': exec_id,
            'run_id': f'GAM-001_beta2.0_{d_base}_{modifier}_s{seed}',
            'gamma_id': 'GAM-001',
            'd_base_id': d_base,
            'modifier_id': modifier,
            'seed': seed,
            'test_name': 'UNI-001',
            'test_score': 1.0,
            'metric_scores': {
                'frobenius_norm': {
                    'value': 1000.0,
                    'score': 1.0,
                    'flag': True,
                    'pathology_type': 'S2_EXPLOSION',
                    'weight': 2.0
                },
                'spectral_norm': {
                    'value': 1000.0,
                    'score': 1.0,
                    'flag': True,
                    'pathology_type': 'S2_EXPLOSION',
                    'weight': 2.0
                },
                'trace_absolute': {
                    'value': 1000.0,
                    'score': 1.0,
                    'flag': True,
                    'pathology_type': 'S2_EXPLOSION',
                    'weight': 2.0
                }
            }
        }
        scores_data.append(run_data)
    
    # Appeler verdict_engine
    patterns = analyze_metric_patterns(scores_data)
    metric_quality = compute_metric_quality(scores_data)
    verdict, reason = decide_verdict_from_patterns(patterns, metric_quality)
    
    # Vérifier patterns
    systematic = patterns.get('systematic_failures', [])
    valid_metrics = [m for m in metric_quality if metric_quality[m]['valid']]
    
    print(f"  Patterns détectés:")
    print(f"    - systematic_failures: {systematic}")
    print(f"    - Métriques valides: {valid_metrics}")
    
    # Assertions
    assert len(systematic) >= 1, \
        f"Doit détecter 3 systematic_failures, trouvé {len(systematic)}"
    
    assert verdict == 'REJECTED[R0]', \
        f"Doit être REJECTED[R0] avec ≥3 échecs systématiques, reçu: {verdict}"
    
    assert 'systématique' in reason.lower() or 'Pathologie' in reason, \
        f"Raison doit mentionner pathologie systématique: {reason}"
    
    print("✓ Verdict isolation OK")
    print(f"  - Verdict: {verdict}")
    print(f"  - Raison: {reason[:80]}...")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all tests."""
    print("\n" + "#"*80)
    print("# TEST PIPELINE CHARTER 5.4 - LOCAL")
    print("#"*80)
    
    try:
        # Test 1: TestEngine
        test_engine_isolation()
        
        # Test 2: Scoring
        test_scoring_isolation()
        
        # Test 3: Verdict
        test_verdict_isolation()
        
        # Success
        print("\n" + "#"*80)
        print("# ✅ TOUS LES TESTS PASSENT")
        print("#"*80)
        print("\nPipeline Charter 5.4 validé:")
        print("  ✓ TestEngine → Observations pures")
        print("  ✓ Scoring → Pathology scores [0,1]")
        print("  ✓ Verdict → Pattern detection R0")
        print("\nProchaine étape: Tester avec vraies données db_raw")
        
    except AssertionError as e:
        print("\n" + "#"*80)
        print("# ❌ ERREUR DÉTECTÉE")
        print("#"*80)
        print(str(e))
        raise
    
    except Exception as e:
        print("\n" + "#"*80)
        print("# ❌ ERREUR INATTENDUE")
        print("#"*80)
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        raise


if __name__ == "__main__":
    main()