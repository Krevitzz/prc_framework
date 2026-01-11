# tests/utilities/verdict_reporter.py
"""
Verdict Reporter - Orchestration génération rapports R0.

ARCHITECTURE :
- Orchestre verdict_engine (analyses globales) + gamma_profiling (profils individuels)
- Génère rapports structurés selon Charter R0 Section 3.1
- Séparation claire : analyses / profiling / reporting

RESPONSABILITÉS :
1. Charger observations (délégué verdict_engine)
2. Exécuter analyses globales (variance, interactions)
3. Exécuter profiling gamma (comportements individuels)
4. Fusionner résultats
5. Générer rapports multi-formats (JSON, TXT, CSV)
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

# Imports modules spécialisés
from .verdict_engine import (
    load_all_observations,
    filter_numeric_artifacts,
    generate_degeneracy_report,  # ⚠️ FIX : Pas diagnose_numeric_degeneracy
    diagnose_scale_outliers,
    print_degeneracy_report,
    print_scale_outliers_report,
    stratify_by_regime,
    analyze_regime,
    FACTORS,
    PROJECTIONS,
    MIN_SAMPLES_PER_GROUP,
    MIN_GROUPS,
    MIN_TOTAL_SAMPLES
)

from .gamma_profiling import (
    profile_all_gammas,
    rank_gammas_by_test,
    compare_gammas_summary
)

from .config_loader import get_loader


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def generate_verdict_report(
    params_config_id: str,
    verdict_config_id: str,
    output_dir: str = "reports/verdicts"
) -> dict:
    """
    Pipeline complet génération rapport verdict R0.
    
    ARCHITECTURE :
    1. Chargement observations (verdict_engine)
    2. Diagnostics numériques (verdict_engine)
    3. Analyses globales stratifiées (verdict_engine)
    4. Profiling gamma (gamma_profiling)
    5. Fusion résultats
    6. Génération rapports multi-formats
    
    Args:
        params_config_id: Config params utilisée (ex: 'params_default_v1')
        verdict_config_id: Config verdict (ex: 'verdict_default_v1')
        output_dir: Répertoire sortie rapports
    
    Returns:
        dict: Résultats complets (pour introspection)
        {
            'metadata': {...},
            'gamma_profiles': {...},
            'structural_patterns': {...},
            'comparisons': {...},
            'report_paths': {...}
        }
    """
    print(f"\n{'='*70}")
    print(f"VERDICT REPORTER R0 - GÉNÉRATION RAPPORTS")
    print(f"{'='*70}\n")
    
    print(f"Params config:  {params_config_id}")
    print(f"Verdict config: {verdict_config_id}\n")
    
    # =========================================================================
    # ÉTAPE 1 : CHARGEMENT + DIAGNOSTICS
    # =========================================================================
    
    print("1. Chargement observations...")
    observations = load_all_observations(params_config_id)
    print(f"   ✓ {len(observations)} observations chargées")
    
    # Filtrage artefacts numériques
    observations, rejection_stats = filter_numeric_artifacts(observations)
    
    if rejection_stats['rejected_observations'] > 0:
        print(f"   ⊘ Filtré {rejection_stats['rejected_observations']} observations "
              f"({rejection_stats['rejection_rate']*100:.1f}%) : artefacts numériques")
        for test, count in sorted(rejection_stats['rejected_by_test'].items()):
            print(f"      {test}: {count} invalides")
    print()
    
    # Diagnostics (informatifs uniquement)
    print("2. Diagnostics numériques...")
    # ⚠️ FIX : Ces fonctions prennent la liste complète, pas dict individuel
    from .verdict_engine import generate_degeneracy_report, diagnose_scale_outliers
    degeneracy_report = generate_degeneracy_report(observations)
    scale_report = diagnose_scale_outliers(observations)
    print(f"   ✓ Dégénérescences : {degeneracy_report['observations_with_flags']} observations")
    print(f"   ✓ Ruptures échelle : {scale_report['observations_with_outliers']} observations")
    print()
    
    # =========================================================================
    # ÉTAPE 2 : ANALYSES GLOBALES STRATIFIÉES (verdict_engine)
    # =========================================================================
    
    print("3. Analyses globales stratifiées...")
    
    # Stratification régimes
    obs_stable, obs_explosif = stratify_by_regime(observations)
    print(f"   Régime STABLE   : {len(obs_stable)} observations ({len(obs_stable)/len(observations)*100:.1f}%)")
    print(f"   Régime EXPLOSIF : {len(obs_explosif)} observations ({len(obs_explosif)/len(observations)*100:.1f}%)")
    
    # Analyses parallèles (3 strates)
    print("\n   Analyse GLOBAL...")
    results_global = analyze_regime(
        observations, 'GLOBAL',
        params_config_id, verdict_config_id
    )
    
    print("   Analyse STABLE...")
    # ⚠️ FIX : Vérifier que strate non vide
    if len(obs_stable) > 0:
        results_stable = analyze_regime(
            obs_stable, 'STABLE',
            params_config_id, verdict_config_id
        )
    else:
        results_stable = {
            'regime': 'STABLE',
            'n_observations': 0,
            'status': 'INSUFFICIENT_DATA',
            'message': 'Aucune observation dans strate STABLE'
        }
    
    print("   Analyse EXPLOSIF...")
    # ⚠️ FIX : Vérifier que strate non vide
    if len(obs_explosif) > 0:
        results_explosif = analyze_regime(
            obs_explosif, 'EXPLOSIF',
            params_config_id, verdict_config_id
        )
    else:
        results_explosif = {
            'regime': 'EXPLOSIF',
            'n_observations': 0,
            'status': 'INSUFFICIENT_DATA',
            'message': 'Aucune observation dans strate EXPLOSIF'
        }
    print()
    
    # =========================================================================
    # ÉTAPE 3 : PROFILING GAMMA (gamma_profiling)
    # =========================================================================
    
    print("4. Profiling gamma (comportements individuels)...")
    gamma_profiles = profile_all_gammas(observations)
    print(f"   ✓ {len(gamma_profiles)} gammas profilés")
    
    # Comparaisons inter-gammas
    comparisons = compare_gammas_summary(gamma_profiles)
    print(f"   ✓ Comparaisons : {len(comparisons['by_test'])} tests analysés")
    print()
    
    # =========================================================================
    # ÉTAPE 4 : FUSION RÉSULTATS
    # =========================================================================
    
    print("5. Fusion résultats...")
    
    # Compiler structure finale
    final_results = {
        'metadata': _compile_metadata(
            params_config_id,
            verdict_config_id,
            observations,
            rejection_stats,
            degeneracy_report,
            scale_report
        ),
        
        'gamma_profiles': _format_gamma_profiles(gamma_profiles),
        
        'structural_patterns': _compile_structural_patterns(
            results_global,
            results_stable,
            results_explosif
        ),
        
        'comparisons': comparisons,
        
        'diagnostics': {
            'numeric_artifacts': rejection_stats,
            'degeneracy': degeneracy_report,
            'scale_outliers': scale_report
        }
    }
    
    print(f"   ✓ Structure compilée")
    print()
    
    # =========================================================================
    # ÉTAPE 5 : GÉNÉRATION RAPPORTS
    # =========================================================================
    
    print("6. Génération rapports multi-formats...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = Path(output_dir) / f"{timestamp}_verdict_r0"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_paths = {}
    
    # 6a. Metadata
    _write_metadata(report_dir, final_results['metadata'])
    report_paths['metadata'] = str(report_dir / 'metadata.json')
    
    # 6b. Rapport humain principal
    _write_summary_report(report_dir, final_results)
    report_paths['summary'] = str(report_dir / 'summary.txt')
    
    # 6c. Gamma profiles (JSON + CSV)
    _write_gamma_profiles(report_dir, final_results['gamma_profiles'])
    report_paths['gamma_profiles_json'] = str(report_dir / 'gamma_profiles.json')
    report_paths['gamma_profiles_csv'] = str(report_dir / 'gamma_profiles.csv')
    
    # 6d. Comparaisons inter-gammas
    _write_comparisons(report_dir, final_results['comparisons'])
    report_paths['comparisons'] = str(report_dir / 'comparisons.json')
    
    # 6e. Structural patterns (analyses globales)
    _write_structural_patterns(report_dir, final_results['structural_patterns'])
    report_paths['structural_patterns'] = str(report_dir / 'structural_patterns.json')
    
    # 6f. Diagnostics détaillés
    _write_diagnostics(report_dir, final_results['diagnostics'])
    report_paths['diagnostics'] = str(report_dir / 'diagnostics.json')
    
    # 6g. Analyses stratifiées (CSVs)
    if results_global['status'] == 'SUCCESS':
        results_global['marginal_variance'].to_csv(
            report_dir / 'marginal_variance_global.csv',
            index=False
        )
        report_paths['marginal_variance_global'] = str(report_dir / 'marginal_variance_global.csv')
    
    if results_stable['status'] == 'SUCCESS' and not results_stable['marginal_variance'].empty:
        results_stable['marginal_variance'].to_csv(
            report_dir / 'marginal_variance_stable.csv',
            index=False
        )
        report_paths['marginal_variance_stable'] = str(report_dir / 'marginal_variance_stable.csv')
    
    if results_explosif['status'] == 'SUCCESS' and not results_explosif['marginal_variance'].empty:
        results_explosif['marginal_variance'].to_csv(
            report_dir / 'marginal_variance_explosif.csv',
            index=False
        )
        report_paths['marginal_variance_explosif'] = str(report_dir / 'marginal_variance_explosif.csv')
    
    print(f"   ✓ Rapports générés : {report_dir}")
    print()
    
    # =========================================================================
    # RÉSUMÉ FINAL
    # =========================================================================
    
    print("="*70)
    print("RAPPORT VERDICT R0 GÉNÉRÉ")
    print("="*70)
    print(f"Répertoire : {report_dir}")
    print(f"Gammas     : {len(gamma_profiles)}")
    print(f"Tests      : {len(comparisons['by_test'])}")
    print(f"Fichiers   : {len(report_paths)}")
    print("="*70 + "\n")
    
    final_results['report_paths'] = report_paths
    return final_results


# =============================================================================
# COMPILATION METADATA
# =============================================================================

def _compile_metadata(
    params_config_id: str,
    verdict_config_id: str,
    observations: list,
    rejection_stats: dict,
    degeneracy_report: dict,
    scale_report: dict
) -> dict:
    """Compile métadonnées rapport."""
    
    # Compter gammas/tests uniques
    gammas = set(obs['gamma_id'] for obs in observations)
    tests = set(obs['test_name'] for obs in observations)
    
    return {
        'generated_at': datetime.now().isoformat(),
        'engine_version': '5.5',
        'architecture': 'verdict_reporter_r0',
        'configs_used': {
            'params': params_config_id,
            'verdict': verdict_config_id
        },
        'data_summary': {
            'total_observations': len(observations),
            'valid_observations': rejection_stats['valid_observations'],
            'rejected_observations': rejection_stats['rejected_observations'],
            'rejection_rate': rejection_stats['rejection_rate'],
            'n_gammas': len(gammas),
            'n_tests': len(tests),
            'gammas_list': sorted(gammas),
            'tests_list': sorted(tests)
        },
        'quality_flags': {
            'observations_with_degeneracy': degeneracy_report['observations_with_flags'],
            'degeneracy_rate': degeneracy_report['flag_rate'],
            'observations_with_scale_outliers': scale_report['observations_with_outliers'],
            'scale_outlier_rate': scale_report['outlier_rate']
        },
        'analysis_parameters': {
            'factors_analyzed': FACTORS,
            'projections_analyzed': PROJECTIONS,
            'testability_thresholds': {
                'min_samples_per_group': MIN_SAMPLES_PER_GROUP,
                'min_groups': MIN_GROUPS,
                'min_total_samples': MIN_TOTAL_SAMPLES
            }
        }
    }


# =============================================================================
# FORMATAGE GAMMA PROFILES
# =============================================================================

def _format_gamma_profiles(gamma_profiles: dict) -> dict:
    """
    Formate gamma_profiles pour rapport.
    
    Structure Charter R0 :
    {
        'GAM-001': {
            'tests': {
                'SYM-001': {
                    'regime': 'CONSERVES_X',
                    'behavior': 'stable',
                    'timeline': 'early_deviation_then_saturation',
                    'confidence': 'high'
                }
            },
            'summary': {...}
        }
    }
    """
    formatted = {}
    
    for gamma_id, gamma_data in gamma_profiles.items():
        tests_formatted = {}
        
        for test_name, test_profile in gamma_data['tests'].items():
            prc = test_profile['prc_profile']
            
            tests_formatted[test_name] = {
                'regime': prc['regime'],
                'behavior': prc['behavior'],
                'timeline': prc['dominant_timeline']['timeline_compact'],
                'timeline_confidence': prc['dominant_timeline']['confidence'],
                'confidence': prc['confidence'],
                'n_runs': prc['n_runs'],
                'n_valid': prc['n_valid'],
                'pathologies': prc['pathologies'],
                'robustness': prc['robustness']
            }
        
        # Synthèse gamma (régime dominant)
        regime_counts = defaultdict(int)
        for test_prof in tests_formatted.values():
            regime_counts[test_prof['regime']] += 1
        
        dominant_regime = max(regime_counts.items(), key=lambda x: x[1])[0] if regime_counts else 'NO_DATA'
        
        formatted[gamma_id] = {
            'tests': tests_formatted,
            'summary': {
                'n_tests': len(tests_formatted),
                'dominant_regime': dominant_regime,
                'regime_distribution': dict(regime_counts)
            }
        }
    
    return formatted


# =============================================================================
# COMPILATION STRUCTURAL PATTERNS
# =============================================================================

def _compile_structural_patterns(
    results_global: dict,
    results_stable: dict,
    results_explosif: dict
) -> dict:
    """Compile patterns structuraux (analyses globales)."""
    
    return {
        'stratification': {
            'GLOBAL': {
                'n_observations': results_global['n_observations'],
                'status': results_global['status'],
                'verdict': results_global.get('verdict', 'N/A'),
                'reason': results_global.get('reason', 'N/A'),
                'patterns': results_global.get('patterns_global', {})
            },
            'STABLE': {
                'n_observations': results_stable['n_observations'],
                'status': results_stable['status'],
                'verdict': results_stable.get('verdict', 'N/A'),
                'reason': results_stable.get('reason', 'N/A'),
                'patterns': results_stable.get('patterns_global', {})
            },
            'EXPLOSIF': {
                'n_observations': results_explosif['n_observations'],
                'status': results_explosif['status'],
                'verdict': results_explosif.get('verdict', 'N/A'),
                'reason': results_explosif.get('reason', 'N/A'),
                'patterns': results_explosif.get('patterns_global', {})
            }
        }
    }


# =============================================================================
# GÉNÉRATION FICHIERS RAPPORTS
# =============================================================================

def _write_metadata(report_dir: Path, metadata: dict):
    """Écrit metadata.json."""
    with open(report_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


def _write_summary_report(report_dir: Path, results: dict):
    """Écrit rapport humain principal (ENRICHI R0+)."""
    
    metadata = results['metadata']
    gamma_profiles = results['gamma_profiles']
    structural = results['structural_patterns']
    comparisons = results['comparisons']
    
    with open(report_dir / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("VERDICT REPORT R0+ - POSTURE NON GAMMA-CENTRIQUE\n")
        f.write(f"{metadata['generated_at']}\n")
        f.write("="*80 + "\n\n")
        
        f.write("ARCHITECTURE RAPPORT:\n")
        f.write("  verdict_engine   : Analyses statistiques globales (variance, interactions)\n")
        f.write("  gamma_profiling  : Profils comportementaux individuels (régimes, timelines)\n")
        f.write("  verdict_reporter : Orchestration + génération rapports\n\n")
        
        # DATA SUMMARY
        f.write("="*80 + "\n")
        f.write("DATA SUMMARY\n")
        f.write("="*80 + "\n")
        data = metadata['data_summary']
        f.write(f"Total observations    : {data['total_observations']}\n")
        f.write(f"Valid observations    : {data['valid_observations']}\n")
        f.write(f"Rejected (artifacts)  : {data['rejected_observations']} ({data['rejection_rate']*100:.1f}%)\n")
        f.write(f"Gammas analyzed       : {data['n_gammas']}\n")
        f.write(f"Tests analyzed        : {data['n_tests']}\n\n")
        
        # QUALITY FLAGS
        f.write("QUALITY FLAGS:\n")
        quality = metadata['quality_flags']
        f.write(f"  Dégénérescences détectées : {quality['observations_with_degeneracy']} ({quality['degeneracy_rate']*100:.1f}%)\n")
        f.write(f"  Ruptures échelle          : {quality['observations_with_scale_outliers']} ({quality['scale_outlier_rate']*100:.1f}%)\n\n")
        
        # NOUVEAU : SYNTHÈSE RÉGIMES GLOBALE
        f.write("="*80 + "\n")
        f.write("SYNTHÈSE RÉGIMES (vue transversale)\n")
        f.write("="*80 + "\n")
        _write_regime_synthesis(f, gamma_profiles)
        f.write("\n")
        
        # NOUVEAU : SIGNATURES DYNAMIQUES
        f.write("="*80 + "\n")
        f.write("SIGNATURES DYNAMIQUES PAR GAMMA\n")
        f.write("="*80 + "\n")
        _write_dynamic_signatures(f, gamma_profiles)
        f.write("\n")
        
        # GAMMA PROFILES (résumé par gamma)
        f.write("="*80 + "\n")
        f.write("GAMMA PROFILES (régimes dominants par gamma)\n")
        f.write("="*80 + "\n")
        
        for gamma_id in sorted(gamma_profiles.keys()):
            profile = gamma_profiles[gamma_id]
            summary = profile['summary']
            f.write(f"\n{gamma_id}:\n")
            f.write(f"  Régime dominant : {summary['dominant_regime']}\n")
            f.write(f"  Tests profilés  : {summary['n_tests']}\n")
            f.write(f"  Distribution    : {summary['regime_distribution']}\n")
            
            # NOUVEAU : Propriétés conservées
            conserved = _extract_conserved_properties(profile)
            if conserved:
                f.write(f"  Propriétés conservées : {', '.join(conserved)}\n")
        
        f.write("\n")
        
        # COMPARISONS (enrichi avec contexte)
        f.write("="*80 + "\n")
        f.write("COMPARISONS INTER-GAMMAS (par propriété)\n")
        f.write("="*80 + "\n")
        _write_comparisons_enriched(f, comparisons, gamma_profiles)
        f.write("\n")
        
        # STRUCTURAL PATTERNS (analyses globales)
        f.write("="*80 + "\n")
        f.write("STRUCTURAL PATTERNS (analyses globales stratifiées)\n")
        f.write("="*80 + "\n")
        
        for regime_name in ['GLOBAL', 'STABLE', 'EXPLOSIF']:
            regime_data = structural['stratification'][regime_name]
            f.write(f"\nRÉGIME {regime_name}:\n")
            f.write(f"  Observations : {regime_data['n_observations']}\n")
            f.write(f"  Status       : {regime_data['status']}\n")
            
            if regime_data['status'] == 'SUCCESS':
                f.write(f"  Verdict      : {regime_data['verdict']}\n")
                f.write(f"  Raison       : {regime_data['reason']}\n")
                
                patterns = regime_data['patterns']
                f.write(f"  Patterns détectés:\n")
                for pattern_type, pattern_list in patterns.items():
                    if pattern_list:
                        f.write(f"    {pattern_type}: {len(pattern_list)} occurrences\n")
        
        f.write("\n")
        
        # FOOTER
        f.write("="*80 + "\n")
        f.write("CONSULTATION DÉTAILLÉE\n")
        f.write("="*80 + "\n")
        f.write("gamma_profiles.json       : Profils complets tous gammas × tests\n")
        f.write("gamma_profiles.csv        : Vue tabulaire pour analyse\n")
        f.write("comparisons.json          : Classements inter-gammas\n")
        f.write("structural_patterns.json  : Analyses globales (variance, interactions)\n")
        f.write("diagnostics.json          : Diagnostics numériques détaillés\n")
        f.write("marginal_variance_*.csv   : Données brutes analyses (3 strates)\n")
        f.write("="*80 + "\n")


def _write_gamma_profiles(report_dir: Path, gamma_profiles: dict):
    """Écrit gamma_profiles.json + CSV."""
    
    # JSON complet
    with open(report_dir / 'gamma_profiles.json', 'w', encoding='utf-8') as f:
        json.dump(gamma_profiles, f, indent=2)
    
    # CSV (vue tabulaire)
    rows = []
    for gamma_id, gamma_data in gamma_profiles.items():
        for test_name, test_data in gamma_data['tests'].items():
            rows.append({
                'gamma_id': gamma_id,
                'test_name': test_name,
                'regime': test_data['regime'],
                'behavior': test_data['behavior'],
                'timeline': test_data['timeline'],
                'timeline_confidence': test_data['timeline_confidence'],
                'confidence': test_data['confidence'],
                'n_runs': test_data['n_runs'],
                'n_valid': test_data['n_valid'],
                'pathology_numeric_instability': test_data['pathologies']['numeric_instability'],
                'pathology_oscillatory': test_data['pathologies']['oscillatory'],
                'pathology_collapse': test_data['pathologies']['collapse'],
                'pathology_trivial': test_data['pathologies']['trivial'],
                'robust_homogeneous': test_data['robustness']['homogeneous'],
                'robust_mixed_behavior': test_data['robustness']['mixed_behavior'],
                'robust_numerically_stable': test_data['robustness']['numerically_stable']
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(report_dir / 'gamma_profiles.csv', index=False)


def _write_comparisons(report_dir: Path, comparisons: dict):
    """Écrit comparisons.json."""
    with open(report_dir / 'comparisons.json', 'w', encoding='utf-8') as f:
        json.dump(comparisons, f, indent=2)


def _write_structural_patterns(report_dir: Path, structural: dict):
    """Écrit structural_patterns.json."""
    with open(report_dir / 'structural_patterns.json', 'w', encoding='utf-8') as f:
        json.dump(structural, f, indent=2)


def _write_diagnostics(report_dir: Path, diagnostics: dict):
    """Écrit diagnostics.json."""
    
    # ⚠️ FIX : Convertir clés tuple → string pour JSON
    diagnostics_serializable = _make_json_serializable(diagnostics)
    
    with open(report_dir / 'diagnostics.json', 'w', encoding='utf-8') as f:
        json.dump(diagnostics_serializable, f, indent=2)


def _make_json_serializable(obj):
    """
    Convertit récursivement tuples en strings pour sérialisation JSON.
    
    Problème : diagnose_scale_outliers() utilise des clés tuple
    (test_name, metric_name, proj_name) qui ne sont pas JSON-serializable.
    
    Solution : Convertir tuples → "test|metric|proj" strings.
    """
    if isinstance(obj, dict):
        return {
            (str(k) if isinstance(k, tuple) else k): _make_json_serializable(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        # Convertir tuples en strings lisibles
        return "|".join(str(x) for x in obj)
    else:
        return obj


# =============================================================================
# HELPERS SUMMARY ENRICHI
# =============================================================================

def _write_regime_synthesis(f, gamma_profiles: dict):
    """Écrit synthèse régimes transversale."""
    
    # Compter régimes globalement
    regime_counter = defaultdict(int)
    for gamma_data in gamma_profiles.values():
        for test_data in gamma_data['tests'].values():
            regime = test_data['regime']
            regime_counter[regime] += 1
    
    total = sum(regime_counter.values())
    
    f.write(f"Total profils : {total}\n\n")
    
    # Grouper par famille
    conservation = {k: v for k, v in regime_counter.items() if 'CONSERVES_' in k}
    pathologies = {k: v for k, v in regime_counter.items() if k in ['NUMERIC_INSTABILITY', 'OSCILLATORY_UNSTABLE', 'TRIVIAL', 'DEGRADING']}
    mixed = {k: v for k, v in regime_counter.items() if k.startswith('MIXED::')}
    other = {k: v for k, v in regime_counter.items() if k not in conservation and k not in pathologies and k not in mixed}
    
    if conservation:
        f.write("CONSERVATION (régimes sains):\n")
        for regime, count in sorted(conservation.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            f.write(f"  {regime:30s} : {count:3d} ({pct:5.1f}%)\n")
        f.write(f"  Total conservation : {sum(conservation.values())} ({sum(conservation.values())/total*100:.1f}%)\n\n")
    
    if pathologies:
        f.write("PATHOLOGIES:\n")
        for regime, count in sorted(pathologies.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            f.write(f"  {regime:30s} : {count:3d} ({pct:5.1f}%)\n")
        f.write(f"  Total pathologies : {sum(pathologies.values())} ({sum(pathologies.values())/total*100:.1f}%)\n\n")
    
    if mixed:
        f.write("MULTIMODALITÉ (MIXED::X):\n")
        for regime, count in sorted(mixed.items(), key=lambda x: -x[1])[:5]:
            pct = count / total * 100
            f.write(f"  {regime:30s} : {count:3d} ({pct:5.1f}%)\n")
        f.write(f"  Total multimodal : {sum(mixed.values())} ({sum(mixed.values())/total*100:.1f}%)\n\n")
    
    if other:
        f.write("AUTRES (SATURATES, UNCATEGORIZED):\n")
        for regime, count in sorted(other.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            f.write(f"  {regime:30s} : {count:3d} ({pct:5.1f}%)\n")


def _write_dynamic_signatures(f, gamma_profiles: dict):
    """Écrit signatures dynamiques par gamma."""
    
    for gamma_id in sorted(gamma_profiles.keys()):
        gamma_data = gamma_profiles[gamma_id]
        
        # Compter timelines dominantes
        timeline_counter = defaultdict(int)
        for test_data in gamma_data['tests'].values():
            timeline = test_data['timeline']
            timeline_counter[timeline] += 1
        
        # Timeline dominante
        if timeline_counter:
            dominant_timeline, count = max(timeline_counter.items(), key=lambda x: x[1])
            confidence = count / len(gamma_data['tests'])
            
            f.write(f"\n{gamma_id}:\n")
            f.write(f"  Timeline dominante : {dominant_timeline} ({confidence*100:.0f}% tests)\n")
            
            # Diversité timelines
            if len(timeline_counter) > 1:
                f.write(f"  Variantes ({len(timeline_counter)} timelines distinctes):\n")
                for tl, cnt in sorted(timeline_counter.items(), key=lambda x: -x[1])[:3]:
                    if tl != dominant_timeline:
                        f.write(f"    - {tl} ({cnt} tests)\n")


def _extract_conserved_properties(profile: dict) -> list:
    """Extrait propriétés conservées depuis profil gamma."""
    
    properties = set()
    
    for test_data in profile['tests'].values():
        regime = test_data['regime']
        
        if 'CONSERVES_SYMMETRY' in regime:
            properties.add('Symétrie')
        elif 'CONSERVES_NORM' in regime:
            properties.add('Norme')
        elif 'CONSERVES_PATTERN' in regime:
            properties.add('Pattern')
        elif 'CONSERVES_TOPOLOGY' in regime:
            properties.add('Topologie')
        elif 'CONSERVES_GRADIENT' in regime:
            properties.add('Gradient')
        elif 'CONSERVES_SPECTRUM' in regime:
            properties.add('Spectre')
    
    return sorted(properties)


def _write_comparisons_enriched(f, comparisons: dict, gamma_profiles: dict):
    """Écrit comparaisons enrichies avec contexte propriétés."""
    
    # Grouper tests par propriété
    tests_by_property = {
        'Symétrie': ['SYM-001'],
        'Norme': ['SPE-001', 'SPE-002', 'UNIV-001', 'UNIV-002'],
        'Pattern': ['PAT-001'],
        'Topologie': ['TOP-001'],
        'Gradient': ['GRA-001'],
        'Spectre': ['SPA-001']
    }
    
    by_test = comparisons['by_test']
    
    for property_name, test_list in tests_by_property.items():
        tests_in_data = [t for t in test_list if t in by_test]
        
        if not tests_in_data:
            continue
        
        f.write(f"\n{property_name.upper()}:\n")
        
        for test_name in tests_in_data:
            comp = by_test[test_name]
            f.write(f"\n  {test_name}:\n")
            f.write(f"    Meilleur : {comp['best_conservation']}\n")
            f.write(f"    Pire     : {comp['worst_conservation']}\n")
            f.write(f"    Classement : {', '.join(comp['ranking'][:5])}...\n")