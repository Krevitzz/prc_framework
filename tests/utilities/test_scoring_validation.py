#!/usr/bin/env python3
"""
tests/test_scoring_validation.py

Tests validation scoring engine et verdicts R0.
Sans dépendance à runs réels (golden set synthétique).

Usage:
    python tests/test_scoring_validation.py
    python tests/test_scoring_validation.py --verbose
"""

import sys
import argparse
from tests.utilities.scoring_engine import get_scoring_engine


def test_s1_collapse():
    """Test TYPE S1: COLLAPSE."""
    print("\n" + "="*70)
    print("TEST S1_COLLAPSE")
    print("="*70)
    
    engine = get_scoring_engine()
    
    rule = {
        'pathology_type': 'S1_COLLAPSE',
        'threshold_low': 0.1,
        'critical_low': 0.01,
        'weight': 2.0,
        'mode': 'soft'
    }
    
    test_cases = [
        (0.0, 1.0, True, "Valeur 0 → score max, flag"),
        (0.005, 1.0, True, "Valeur < critical → score max, flag"),
        (0.05, 0.5, False, "Valeur entre critical et threshold"),
        (0.1, 0.0, False, "Valeur = threshold → score 0"),
        (0.5, 0.0, False, "Valeur > threshold → score 0"),
    ]
    
    passed = 0
    for value, expected_score, expected_flag, description in test_cases:
        result = engine.score_metric('test_metric', value, rule)
        
        score = result['score']
        flag = result['flag']
        
        score_ok = abs(score - expected_score) < 0.1
        flag_ok = flag == expected_flag
        
        status = "✓" if (score_ok and flag_ok) else "✗"
        
        print(f"  {status} {description}")
        print(f"      value={value}, score={score:.3f} (expected ~{expected_score}), flag={flag} (expected {expected_flag})")
        
        if score_ok and flag_ok:
            passed += 1
    
    print(f"\nRésultat: {passed}/{len(test_cases)} tests passés")
    return passed == len(test_cases)


def test_s2_explosion():
    """Test TYPE S2: EXPLOSION."""
    print("\n" + "="*70)
    print("TEST S2_EXPLOSION")
    print("="*70)
    
    engine = get_scoring_engine()
    
    rule = {
        'pathology_type': 'S2_EXPLOSION',
        'threshold_high': 1000.0,
        'critical_high': 10000.0,
        'weight': 2.0,
        'mode': 'soft'
    }
    
    test_cases = [
        (100.0, 0.0, False, "Valeur << threshold → score 0"),
        (1000.0, 0.0, False, "Valeur = threshold → score 0"),
        (5500.0, 0.5, False, "Valeur entre threshold et critical"),
        (10000.0, 1.0, True, "Valeur = critical → score max, flag"),
        (20000.0, 1.0, True, "Valeur > critical → score max, flag"),
    ]
    
    passed = 0
    for value, expected_score, expected_flag, description in test_cases:
        result = engine.score_metric('test_metric', value, rule)
        
        score = result['score']
        flag = result['flag']
        
        score_ok = abs(score - expected_score) < 0.1
        flag_ok = flag == expected_flag
        
        status = "✓" if (score_ok and flag_ok) else "✗"
        
        print(f"  {status} {description}")
        print(f"      value={value}, score={score:.3f} (expected ~{expected_score}), flag={flag} (expected {expected_flag})")
        
        if score_ok and flag_ok:
            passed += 1
    
    print(f"\nRésultat: {passed}/{len(test_cases)} tests passés")
    return passed == len(test_cases)


def test_s3_plateau():
    """Test TYPE S3: PLATEAU."""
    print("\n" + "="*70)
    print("TEST S3_PLATEAU")
    print("="*70)
    
    engine = get_scoring_engine()
    
    rule = {
        'pathology_type': 'S3_PLATEAU',
        'interval_toxic': [0.9, 1.0],
        'weight': 1.5,
        'mode': 'soft'
    }
    
    test_cases = [
        (0.5, 0.0, False, "Valeur loin intervalle → score faible"),
        (0.85, 0.5, False, "Valeur proche intervalle"),
        (0.95, 1.0, True, "Valeur dans intervalle → score max, flag"),
        (1.0, 1.0, True, "Valeur limite haute intervalle"),
        (1.1, 0.0, False, "Valeur au-delà intervalle"),
    ]
    
    passed = 0
    for value, expected_score, expected_flag, description in test_cases:
        result = engine.score_metric('test_metric', value, rule)
        
        score = result['score']
        flag = result['flag']
        
        score_ok = abs(score - expected_score) < 0.2  # Tolérance plus large
        flag_ok = flag == expected_flag
        
        status = "✓" if (score_ok and flag_ok) else "✗"
        
        print(f"  {status} {description}")
        print(f"      value={value}, score={score:.3f} (expected ~{expected_score}), flag={flag} (expected {expected_flag})")
        
        if score_ok and flag_ok:
            passed += 1
    
    print(f"\nRésultat: {passed}/{len(test_cases)} tests passés")
    return passed == len(test_cases)


def test_mapping():
    """Test TYPE MAPPING."""
    print("\n" + "="*70)
    print("TEST MAPPING")
    print("="*70)
    
    engine = get_scoring_engine()
    
    rule = {
        'pathology_type': 'MAPPING',
        'mapping': {
            'explosive': 1.0,
            'divergent': 1.0,
            'stable': 0.0,
            'converged': 0.0,
            'growing': 0.3,
            'shrinking': 0.3,
            'oscillating': 0.3
        },
        'weight': 3.0
    }
    
    test_cases = [
        ('explosive', 1.0, True, "Explosive → score max, flag"),
        ('divergent', 1.0, True, "Divergent → score max, flag"),
        ('stable', 0.0, False, "Stable → score 0"),
        ('converged', 0.0, False, "Converged → score 0"),
        ('growing', 0.3, False, "Growing → score 0.3"),
        ('unknown_value', 0.5, False, "Valeur non mappée → default 0.5"),
    ]
    
    passed = 0
    for value, expected_score, expected_flag, description in test_cases:
        result = engine.score_metric('test_metric', value, rule)
        
        score = result['score']
        flag = result['flag']
        
        score_ok = abs(score - expected_score) < 0.01
        flag_ok = flag == expected_flag
        
        status = "✓" if (score_ok and flag_ok) else "✗"
        
        print(f"  {status} {description}")
        print(f"      value='{value}', score={score:.3f} (expected {expected_score}), flag={flag} (expected {expected_flag})")
        
        if score_ok and flag_ok:
            passed += 1
    
    print(f"\nRésultat: {passed}/{len(test_cases)} tests passés")
    return passed == len(test_cases)


def test_aggregation():
    """Test agrégation métriques."""
    print("\n" + "="*70)
    print("TEST AGGREGATION")
    print("="*70)
    
    engine = get_scoring_engine()
    
    # Créer scores métriques synthétiques
    metric_scores = {
        'metric1': {
            'score': 0.2,
            'weight': 1.0,
            'flag': False
        },
        'metric2': {
            'score': 0.8,
            'weight': 2.0,
            'flag': False
        },
        'metric3': {
            'score': 0.5,
            'weight': 1.0,
            'flag': False
        }
    }
    
    # Test mode MAX
    print("\n  Mode: max")
    result_max = engine.aggregate_test_score(metric_scores, 'max')
    expected_max = 0.8
    
    if abs(result_max['test_score'] - expected_max) < 0.01:
        print(f"    ✓ test_score={result_max['test_score']:.3f} (expected {expected_max})")
    else:
        print(f"    ✗ test_score={result_max['test_score']:.3f} (expected {expected_max})")
        return False
    
    # Test mode WEIGHTED_MEAN
    print("\n  Mode: weighted_mean")
    result_wmean = engine.aggregate_test_score(metric_scores, 'weighted_mean')
    # (0.2*1 + 0.8*2 + 0.5*1) / (1+2+1) = 2.3 / 4 = 0.575
    expected_wmean = 0.575
    
    if abs(result_wmean['test_score'] - expected_wmean) < 0.01:
        print(f"    ✓ test_score={result_wmean['test_score']:.3f} (expected {expected_wmean})")
    else:
        print(f"    ✗ test_score={result_wmean['test_score']:.3f} (expected {expected_wmean})")
        return False
    
    print("\nRésultat: Agrégation OK")
    return True


def test_validation():
    """Test validation règles."""
    print("\n" + "="*70)
    print("TEST VALIDATION RÈGLES")
    print("="*70)
    
    engine = get_scoring_engine()
    
    test_cases = [
        # Valid rules
        ({
            'pathology_type': 'S1_COLLAPSE',
            'threshold_low': 0.1
        }, True, "S1_COLLAPSE valid"),
        
        ({
            'pathology_type': 'S2_EXPLOSION',
            'threshold_high': 1000.0
        }, True, "S2_EXPLOSION valid"),
        
        ({
            'pathology_type': 'MAPPING',
            'mapping': {'a': 1.0}
        }, True, "MAPPING valid"),
        
        # Invalid rules
        ({
            'pathology_type': 'UNKNOWN_TYPE'
        }, False, "Type inconnu → invalid"),
        
        ({
            'pathology_type': 'S1_COLLAPSE'
            # Missing threshold_low
        }, False, "S1_COLLAPSE sans threshold_low → invalid"),
        
        ({
            'pathology_type': 'MAPPING',
            'mapping': 'not_a_dict'
        }, False, "MAPPING avec mapping non dict → invalid"),
    ]
    
    passed = 0
    for rule, expected_valid, description in test_cases:
        is_valid, error = engine.validate_scoring_rule(rule)
        
        if is_valid == expected_valid:
            print(f"  ✓ {description}")
            passed += 1
        else:
            print(f"  ✗ {description}")
            print(f"      Expected valid={expected_valid}, got {is_valid}, error='{error}'")
    
    print(f"\nRésultat: {passed}/{len(test_cases)} tests passés")
    return passed == len(test_cases)


def main():
    parser = argparse.ArgumentParser(description="Tests validation scoring R0")
    parser.add_argument('--verbose', action='store_true', help="Mode verbose")
    args = parser.parse_args()
    
    print("\n" + "#"*70)
    print("# TESTS VALIDATION SCORING R0")
    print("#"*70)
    
    tests = [
        ("S1_COLLAPSE", test_s1_collapse),
        ("S2_EXPLOSION", test_s2_explosion),
        ("S3_PLATEAU", test_s3_plateau),
        ("MAPPING", test_mapping),
        ("AGGREGATION", test_aggregation),
        ("VALIDATION", test_validation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ ERREUR dans {name}: {e}")
            results.append((name, False))
    
    # Résumé
    print("\n" + "="*70)
    print("RÉSUMÉ")
    print("="*70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print("\n" + "="*70)
    if total_passed == total_tests:
        print(f"🎉 TOUS LES TESTS PASSENT ({total_passed}/{total_tests})")
        print("="*70)
        return 0
    else:
        print(f"❌ ÉCHECS DÉTECTÉS ({total_passed}/{total_tests} passent)")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())