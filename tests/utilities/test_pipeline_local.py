#!/usr/bin/env python3
# tests/test_pipeline_local.py
"""
Test pipeline complet Charter 5.4 - LOCAL (sans DB).

Objectifs:
1. Vérifier pipeline test → scoring → verdict
2. Vérifier séparation configs (traçabilité unique)
3. Vérifier cohérence données
"""

import numpy as np
from datetime import datetime
from typing import List, Dict
import json

# Imports modules
from tests.utilities.test_engine import TestEngine
from tests.utilities.scoring import score_observation
from tests.utilities.verdict_engine import (
    analyze_metric_patterns,
    compute_metric_quality,
    decide_verdict_from_patterns,
    generate_actionable_insights
)

# Imports tests
from tests import test_uni_001, test_sym_001


# =============================================================================
# MOCK DATA GENERATOR
# =============================================================================

def generate_mock_history(
    d_type: str,
    modifier: str,
    seed: int,
    behavior: str = "normal"
) -> List[np.ndarray]:
    """
    Génère historique mock pour tests.
    
    Args:
        d_type: Type D (SYM, ASY)
        modifier: Modifier (M0, M1, M3)
        seed: Seed
        behavior: normal | explosive | collapse | stable
    
    Returns:
        Liste 200 snapshots (matrices 10×10)
    """
    np.random.seed(seed)
    n_iter = 200
    size = 10
    
    history = []
    
    if behavior == "normal":
        # Évolution normale
        base = np.random.randn(size, size)
        for i in range(n_iter):
            noise = np.random.randn(size, size) * 0.1
            state = base + noise * (1 + i/n_iter)
            history.append(state)
    
    elif behavior == "explosive":
        # Explosion progressive
        base = np.random.randn(size, size)
        for i in range(n_iter):
            factor = np.exp(i / 50)  # Croissance exponentielle
            state = base * factor
            history.append(state)
    
    elif behavior == "collapse":
        # Collapse vers 0
        base = np.random.randn(size, size) * 10
        for i in range(n_iter):
            factor = np.exp(-i / 30)  # Décroissance exponentielle
            state = base * factor
            history.append(state)
    
    elif behavior == "stable":
        # Parfaitement stable
        base = np.random.randn(size, size)
        for i in range(n_iter):
            history.append(base.copy())
    
    return history


def generate_mock_runs(gamma_id: str = "GAM-TEST") -> List[Dict]:
    """
    Génère runs mock avec différents comportements.
    
    Returns:
        Liste runs: [
            {
                'exec_id': int,
                'run_id': str,
                'gamma_id': str,
                'd_base_id': str,
                'modifier_id': str,
                'seed': int,
                'history': List[np.ndarray],
                'expected_behavior': str
            }
        ]
    """
    runs = []
    exec_id = 1
    
    # Scénarios tests
    scenarios = [
        # Normal (majorité)
        ("SYM-001", "M0", 1, "normal"),
        ("SYM-001", "M0", 2, "normal"),
        ("SYM-001", "M0", 3, "normal"),
        ("ASY-001", "M0", 1, "normal"),
        ("ASY-001", "M0", 2, "normal"),
        
        # Explosive (pathologie systématique)
        ("SYM-001", "M3", 1, "explosive"),
        ("SYM-001", "M3", 2, "explosive"),
        ("ASY-001", "M3", 1, "explosive"),
        
        # Collapse (pathologie contextuelle)
        ("SYM-001", "M1", 1, "collapse"),
        
        # Stable (sain)
        ("ASY-001", "M1", 1, "stable"),
    ]
    
    for d_base, modifier, seed, behavior in scenarios:
        history = generate_mock_history(d_base, modifier, seed, behavior)
        
        runs.append({
            'exec_id': exec_id,
            'run_id': f"{gamma_id}_{d_base}_{modifier}_s{seed}",
            'gamma_id': gamma_id,
            'd_base_id': d_base,
            'modifier_id': modifier,
            'seed': seed,
            'history': history,
            'expected_behavior': behavior,
            'state_shape': history[0].shape
        })
        
        exec_id += 1
    
    return runs


# =============================================================================
# TEST PIPELINE
# =============================================================================

def test_pipeline_complete():
    """
    Test pipeline complet : test_engine → scoring → verdict.
    """
    print("\n" + "="*80)
    print("TEST PIPELINE COMPLET - LOCAL (sans DB)")
    print("="*80 + "\n")
    
    # Configuration
    gamma_id = "GAM-TEST"
    params_config_id = "params_default_v1"
    scoring_configs = [
        "scoring_pathologies_v1",
        "scoring_strict_v1"
    ]
    
    # Générer mock runs
    print("1. Génération mock runs...")
    runs = generate_mock_runs(gamma_id)
    print(f"   ✓ {len(runs)} runs générés")
    print(f"   Comportements: {set(r['expected_behavior'] for r in runs)}\n")
    
    # Tests à exécuter
    test_modules = {
        'UNIV-001': test_uni_001,
        'SYM-001': test_sym_001
    }
    
    # Initialiser engine
    engine = TestEngine()
    
    # Stockage résultats (en mémoire)
    all_observations = []
    all_scores = []
    
    # Phase 1 : Tests + Observations
    print("2. Phase TEST - Génération observations...")
    print("-" * 80)
    
    for run in runs:
        print(f"\n  Run: {run['run_id']} (behavior={run['expected_behavior']})")
        
        run_context = {
            'exec_id': run['exec_id'],
            'gamma_id': run['gamma_id'],
            'd_base_id': run['d_base_id'],
            'modifier_id': run['modifier_id'],
            'seed': run['seed'],
            'state_shape': run['state_shape']
        }
        
        for test_id, test_module in test_modules.items():
            # Exécuter test
            observation = engine.execute_test(
                test_module,
                run_context,
                run['history'],
                params_config_id
            )
            
            all_observations.append(observation)
            
            # Affichage compact
            status = observation['status']
            if status == 'SUCCESS':
                first_metric = list(observation['statistics'].keys())[0]
                final_val = observation['statistics'][first_metric]['final']
                print(f"    {test_id}: {status} (metric={final_val:.3f})")
            else:
                print(f"    {test_id}: {status}")
    
    print(f"\n  ✓ {len(all_observations)} observations générées\n")
    
    # Phase 2 : Scoring (multi-configs)
    print("3. Phase SCORING - Calcul scores pathologies...")
    print("-" * 80)
    
    for scoring_config_id in scoring_configs:
        print(f"\n  Config: {scoring_config_id}")
        
        config_scores = []
        
        for observation in all_observations:
            if observation['status'] != 'SUCCESS':
                continue
            
            test_id = observation['test_name']
            test_module = test_modules[test_id]
            
            # Scorer
            try:
                scores = score_observation(
                    observation,
                    test_module,
                    scoring_config_id
                )
                
                config_scores.append(scores)
                all_scores.append(scores)
                
            except Exception as e:
                print(f"    ✗ Erreur scoring {test_id}: {e}")
        
        print(f"    ✓ {len(config_scores)} scores calculés")
    
    print(f"\n  ✓ Total: {len(all_scores)} scores (toutes configs)\n")
    
    # Vérification séparation configs
    print("4. VÉRIFICATION séparation configs...")
    print("-" * 80)
    verify_config_separation(all_scores)
    
    # Phase 3 : Verdict (par config)
    print("\n5. Phase VERDICT - Analyse patterns...")
    print("-" * 80)
    
    for scoring_config_id in scoring_configs:
        print(f"\n  Config: {scoring_config_id}")
        
        # Filtrer scores pour cette config
        config_scores = [
            s for s in all_scores
            if s['config_scoring_id'] == scoring_config_id
        ]
        
        # Transformer format pour verdict_engine
        scores_data = transform_scores_for_verdict(config_scores, runs)
        
        # Analyser patterns
        patterns = analyze_metric_patterns(scores_data)
        metric_quality = compute_metric_quality(scores_data)
        
        # Décision
        verdict, reason = decide_verdict_from_patterns(patterns, metric_quality)
        insights = generate_actionable_insights(patterns, metric_quality)
        
        # Affichage
        print(f"\n    VERDICT: {verdict}")
        print(f"    RAISON:  {reason}")
        
        if patterns.get('non_discriminant'):
            print(f"\n    ⚠ Non discriminant: {patterns['non_discriminant']}")
        
        if patterns.get('over_discriminant'):
            print(f"    ⚠ Over discriminant: {patterns['over_discriminant']}")
        
        if patterns.get('systematic_failures'):
            print(f"    ❌ Échecs systématiques: {patterns['systematic_failures']}")
        
        if insights:
            print(f"\n    Insights actionnables:")
            for insight in insights[:3]:
                print(f"      - {insight}")
    
    # Résumé final
    print("\n" + "="*80)
    print("RÉSUMÉ TEST")
    print("="*80)
    print(f"Runs générés:         {len(runs)}")
    print(f"Observations:         {len(all_observations)}")
    print(f"Scores (multi-cfg):   {len(all_scores)}")
    print(f"Configs scoring:      {len(scoring_configs)}")
    print(f"Tests exécutés:       {len(test_modules)}")
    print("="*80 + "\n")
    
    return all_observations, all_scores


def verify_config_separation(all_scores: List[Dict]):
    """
    Vérifie séparation configs (traçabilité unique).
    
    Chaque score doit être unique par (exec_id, test_name, config_scoring_id).
    """
    print("\n  Vérification unicité scores...")
    
    seen = set()
    duplicates = []
    
    for score in all_scores:
        key = (
            score['exec_id'],
            score['test_name'],
            score['config_scoring_id']
        )
        
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    
    if duplicates:
        print(f"    ❌ ERREUR: {len(duplicates)} doublons détectés!")
        for dup in duplicates[:5]:
            print(f"       {dup}")
        raise AssertionError("Doublons dans scores (séparation configs échouée)")
    
    print(f"    ✓ {len(seen)} scores uniques")
    
    # Vérifier présence configs multiples
    configs_per_run = {}
    for score in all_scores:
        key = (score['exec_id'], score['test_name'])
        if key not in configs_per_run:
            configs_per_run[key] = set()
        configs_per_run[key].add(score['config_scoring_id'])
    
    multi_config = sum(1 for cfgs in configs_per_run.values() if len(cfgs) > 1)
    print(f"    ✓ {multi_config} runs avec multi-configs (attendu > 0)")
    
    if multi_config == 0:
        print("    ⚠ WARNING: Aucun run multi-config détecté")


def transform_scores_for_verdict(config_scores: List[Dict], runs: List[Dict]) -> List[Dict]:
    """
    Transforme scores en format verdict_engine.
    
    Args:
        config_scores: Scores filtrés pour une config
        runs: Runs originaux
    
    Returns:
        Liste format verdict_engine
    """
    # Créer mapping runs
    runs_map = {r['exec_id']: r for r in runs}
    
    transformed = []
    for score in config_scores:
        run = runs_map[score['exec_id']]
        
        transformed.append({
            'exec_id': score['exec_id'],
            'run_id': run['run_id'],
            'gamma_id': run['gamma_id'],
            'd_base_id': run['d_base_id'],
            'modifier_id': run['modifier_id'],
            'seed': run['seed'],
            'test_name': score['test_name'],
            'test_score': score['test_score'],
            'metric_scores': score['metric_scores'],
            'pathology_flags': score['pathology_flags'],
            'critical_metrics': score['critical_metrics'],
            'aggregation_mode': score['aggregation_mode']
        })
    
    return transformed


# =============================================================================
# TESTS UNITAIRES COMPOSANTS
# =============================================================================

def test_test_engine_isolation():
    """Test TestEngine isolé."""
    print("\n" + "="*80)
    print("TEST UNITAIRE - TestEngine")
    print("="*80 + "\n")
    
    engine = TestEngine()
    
    # Mock run
    history = generate_mock_history("SYM-001", "M0", 1, "normal")
    context = {
        'exec_id': 999,
        'gamma_id': 'GAM-UNIT',
        'd_base_id': 'SYM-001',
        'modifier_id': 'M0',
        'seed': 1,
        'state_shape': history[0].shape
    }
    
    # Exécuter
    observation = engine.execute_test(
        test_uni_001,
        context,
        history,
        'params_default_v1'
    )
    
    # Vérifications
    assert observation['status'] == 'SUCCESS', "Status doit être SUCCESS"
    assert observation['exec_id'] == 999, "exec_id doit être préservé"
    assert 'statistics' in observation, "Doit contenir statistics"
    assert 'evolution' in observation, "Doit contenir evolution"
    assert len(observation['statistics']) > 0, "Statistics non vide"
    
    print("✓ TestEngine isolation OK")
    print(f"  - Status: {observation['status']}")
    print(f"  - Métriques: {list(observation['statistics'].keys())}")
    print(f"  - exec_id préservé: {observation['exec_id']}")


def test_scoring_isolation():
    """Test Scoring isolé."""
    print("\n" + "="*80)
    print("TEST UNITAIRE - Scoring")
    print("="*80 + "\n")
    
    # Mock observation
    observation = {
        'exec_id': 888,
        'test_name': 'UNIV-001',
        'config_params_id': 'params_default_v1',
        'status': 'SUCCESS',
        'statistics': {
            'frobenius_norm': {
                'initial': 10.0,
                'final': 15000.0,  # Explosive
                'mean': 5000.0,
                'std': 2000.0
            }
        },
        'evolution': {
            'frobenius_norm': {
                'transition': 'explosive',
                'trend': 'increasing',
                'slope': 75.0
            }
        }
    }
    
    # Scorer
    scores = score_observation(
        observation,
        test_uni_001,
        'scoring_pathologies_v1'
    )
    
    # Vérifications
    assert scores['exec_id'] == 888, "exec_id préservé"
    assert scores['test_score'] >= 0.0 and scores['test_score'] <= 1.0, "Score [0,1]"
    assert 'metric_scores' in scores, "Doit contenir metric_scores"
    assert 'pathology_flags' in scores, "Doit contenir pathology_flags"
    
    print("✓ Scoring isolation OK")
    print(f"  - Test score: {scores['test_score']:.3f}")
    print(f"  - Flags: {scores['pathology_flags']}")
    print(f"  - Critical: {scores['critical_metrics']}")
    print(f"  - Métriques scorées: {list(scores['metric_scores'].keys())}")


# tests/test_pipeline_local.py
# Ligne ~505

def test_verdict_isolation():
    """Test Verdict isolé."""
    print("\n" + "="*80)
    print("TEST UNITAIRE - Verdict")
    print("="*80 + "\n")
    
    # Mock scores_data (format verdict_engine)
    scores_data = []
    
    # Scénario : 1 métrique en échec systématique (80% runs > 0.7)
    for seed in range(1, 11):
        if seed <= 8:  # 80% échecs
            score_val = 0.75 + (seed % 3) * 0.05
            flag_val = True
        else:  # 20% ok
            score_val = 0.3 + (seed % 2) * 0.2
            flag_val = False
        
        scores_data.append({
            'exec_id': seed,
            'run_id': f"GAM-001_SYM-001_M0_s{seed}",
            'gamma_id': 'GAM-001',
            'd_base_id': 'SYM-001',
            'modifier_id': 'M0',
            'seed': seed,
            'test_name': 'UNI-001',
            'test_score': score_val,
            'metric_scores': {
                'frobenius_norm': {
                    'score': score_val,
                    'flag': flag_val,
                    'pathology_type': 'S2_EXPLOSION',
                    'weight': 2.0
                }
            },
            'pathology_flags': ['frobenius_norm'] if flag_val else [],
            'critical_metrics': ['frobenius_norm'] if score_val >= 0.8 else []
        })
    
    # Analyser
    patterns = analyze_metric_patterns(scores_data)
    metric_quality = compute_metric_quality(scores_data)
    verdict, reason = decide_verdict_from_patterns(patterns, metric_quality)
    
    # Debug
    print(f"  Patterns:")
    print(f"    - systematic_failures: {patterns.get('systematic_failures', [])}")
    print(f"    - Métriques valides: {[k for k, v in metric_quality.items() if v['valid']]}")
    
    # Vérifications
    assert len(patterns.get('systematic_failures', [])) >= 1, \
        f"Doit détecter ≥1 échec systématique"
    
    # ⚠️ Avec seuil 1, REJECTED attendu
    assert verdict == 'REJECTED[R0]', \
        f"Doit être REJECTED[R0] avec échec systématique, reçu: {verdict}"
    
    print(f"\n  ✓ Verdict isolation OK")
    print(f"    - Verdict: {verdict}")
    print(f"    - Échecs: {patterns['systematic_failures']}")
    
    
# Ajouter aussi un test spécifique pour over_discriminant
def test_verdict_over_discriminant():
    """Test détection over_discriminant."""
    print("\n" + "="*80)
    print("TEST UNITAIRE - Verdict (over_discriminant)")
    print("="*80 + "\n")
    
    # Tous scores > 0.9 (métrique mal calibrée)
    scores_data = []
    for seed in range(1, 6):
        scores_data.append({
            'exec_id': seed,
            'run_id': f"GAM-001_SYM-001_M0_s{seed}",
            'gamma_id': 'GAM-001',
            'd_base_id': 'SYM-001',
            'modifier_id': 'M0',
            'seed': seed,
            'test_name': 'UNI-001',
            'test_score': 0.95,  # Tous identiques > 0.9
            'metric_scores': {
                'frobenius_norm': {
                    'score': 0.95,
                    'flag': True,
                    'pathology_type': 'S2_EXPLOSION',
                    'weight': 2.0
                }
            },
            'pathology_flags': ['frobenius_norm'],
            'critical_metrics': ['frobenius_norm']
        })
    
    patterns = analyze_metric_patterns(scores_data)
    
    # Vérification
    assert 'UNI-001.frobenius_norm' in patterns.get('over_discriminant', []), \
        "Doit détecter over_discriminant"
    
    # Systematic_failures ne doit PAS être détecté (skip par over_discriminant)
    assert 'UNI-001.frobenius_norm' not in patterns.get('systematic_failures', []), \
        "Ne doit PAS être dans systematic_failures (métrique mal calibrée)"
    
    print("  ✓ Over_discriminant détecté correctement")
    print(f"    - Pattern: {patterns['over_discriminant']}")
    print(f"    - Systematic skipped: OK")

# =============================================================================
# RAPPORT DÉTAILLÉ
# =============================================================================

def generate_detailed_report(all_observations: List[Dict], all_scores: List[Dict]):
    """Génère rapport détaillé résultats."""
    print("\n" + "="*80)
    print("RAPPORT DÉTAILLÉ")
    print("="*80 + "\n")
    
    # 1. Observations par test
    print("1. OBSERVATIONS PAR TEST")
    print("-" * 80)
    obs_by_test = {}
    for obs in all_observations:
        test = obs['test_name']
        if test not in obs_by_test:
            obs_by_test[test] = {'SUCCESS': 0, 'ERROR': 0, 'total': 0}
        obs_by_test[test][obs['status']] = obs_by_test[test].get(obs['status'], 0) + 1
        obs_by_test[test]['total'] += 1
    
    for test, counts in obs_by_test.items():
        print(f"  {test}:")
        print(f"    Total:   {counts['total']}")
        print(f"    SUCCESS: {counts['SUCCESS']}")
        print(f"    ERROR:   {counts.get('ERROR', 0)}")
    
    # 2. Scores par config
    print("\n2. SCORES PAR CONFIG")
    print("-" * 80)
    scores_by_config = {}
    for score in all_scores:
        cfg = score['config_scoring_id']
        if cfg not in scores_by_config:
            scores_by_config[cfg] = []
        scores_by_config[cfg].append(score['test_score'])
    
    for cfg, scores in scores_by_config.items():
        print(f"  {cfg}:")
        print(f"    Count:  {len(scores)}")
        print(f"    Mean:   {np.mean(scores):.3f}")
        print(f"    Std:    {np.std(scores):.3f}")
        print(f"    Min:    {np.min(scores):.3f}")
        print(f"    Max:    {np.max(scores):.3f}")
    
    # 3. Distribution scores
    print("\n3. DISTRIBUTION SCORES (toutes configs)")
    print("-" * 80)
    all_score_vals = [s['test_score'] for s in all_scores]
    
    bins = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    hist, _ = np.histogram(all_score_vals, bins=bins)
    
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        count = hist[i]
        pct = count / len(all_score_vals) * 100
        bar = "█" * int(pct / 2)
        print(f"  [{low:.1f}-{high:.1f}]: {count:3d} ({pct:5.1f}%) {bar}")
    
    # 4. Traçabilité
    print("\n4. TRAÇABILITÉ (échantillon)")
    print("-" * 80)
    for score in all_scores[:3]:
        print(f"  Score #{score['exec_id']}:")
        print(f"    test:     {score['test_name']}")
        print(f"    config:   {score['config_scoring_id']}")
        print(f"    score:    {score['test_score']:.3f}")
        print(f"    flags:    {score['pathology_flags']}")
        print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Point d'entrée principal."""
    print("\n" + "#"*80)
    print("# TEST PIPELINE CHARTER 5.4 - LOCAL")
    print("#"*80)
    
    try:
        # Tests unitaires
        test_test_engine_isolation()
        test_scoring_isolation()
        test_verdict_isolation()  # ← DÉCOMMENTER
        test_verdict_over_discriminant()
        
        # Test intégration complet
        all_observations, all_scores = test_pipeline_complete()
        
        # Rapport détaillé
        generate_detailed_report(all_observations, all_scores)
        
        print("\n" + "#"*80)
        print("# ✓ TOUS LES TESTS PASSENT")
        print("#"*80 + "\n")
        
        return 0
    
    except Exception as e:
        print("\n" + "#"*80)
        print("# ❌ ERREUR DÉTECTÉE")
        print("#"*80)
        print(f"\n{e}\n")
        
        import traceback
        traceback.print_exc()
        
        return 1
```

if __name__ == '__main__':
    import sys
    sys.exit(main())