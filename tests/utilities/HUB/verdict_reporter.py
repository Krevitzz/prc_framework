# tests/utilities/HUB/verdict_reporter.py
"""
Verdict Reporter - Orchestration génération rapports R0.

ARCHITECTURE MIGRÉE (Phase 3 - profiling_common) :
- ✅ Utilise profiling_common (profiling unifié tous axes)
- ✅ Templates Jinja2 (rapports standardisés)
- ✅ Délégation I/O → data_loading.py
- ✅ Délégation diagnostics → statistical_utils.py
- ✅ Délégation stratification → regime_utils.py
- ✅ Délégation formatage → report_writers.py

RESPONSABILITÉS CONSERVÉES :
- Orchestration pipeline complet (6 étapes)
- Compilation métadata
- Formatage structures gamma_profiles
- Compilation structural_patterns
- Coordination génération rapports
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

# ============================================================================
# IMPORTS MODULES REFACTORISÉS
# ============================================================================

# I/O et structuration données
from ..utils.data_loading import (
    load_all_observations
)

# Filtrage et diagnostics numériques
from ..utils.statistical_utils import (
    filter_numeric_artifacts,
    generate_degeneracy_report,
    diagnose_scale_outliers,
    print_degeneracy_report,
    print_scale_outliers_report
)

# Stratification régimes
from ..utils.regime_utils import (
    stratify_by_regime,
    extract_conserved_properties
)

# Analyses verdict et profiling
from .verdict_engine import (
    analyze_regime,
    FACTORS,
    PROJECTIONS,
    MIN_SAMPLES_PER_GROUP,
    MIN_GROUPS,
    MIN_TOTAL_SAMPLES
)

# ============================================================================
# MIGRATION PROFILING : gamma_profiling → profiling_common
# ============================================================================

from ..utils.profiling_common import (
    profile_all_gammas,
    compare_gammas_summary
)

# Note : rank_gammas_by_test() legacy supprimé
# → Utiliser cross_profiling.rank_entities_by_metric() si besoin

# ============================================================================

# Formatage et écriture rapports
from ..utils.report_writers import (
    write_json,
    write_header,
    write_regime_synthesis,
    write_dynamic_signatures,
    write_comparisons_enriched,
    write_consultation_footer,
    write_summary_axis  # Template Jinja2
)

# Configuration
from ..utils.config_loader import get_loader

# Orchestration profiling multi-axes
from .profiling_runner import run_all_profiling

# Cross-profiling (rankings, discriminant_power)
from ..utils.cross_profiling import compute_all_discriminant_powers

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
    
    ARCHITECTURE PHASE 4 (4 axes profiling) :
    1. Chargement observations (data_loading)
    2. Diagnostics numériques (statistical_utils)
    3. Analyses globales stratifiées (verdict_engine)
    4. Profiling multi-axes (profiling_runner) ← PHASE 4
    5. Génération rapports (report_writers - templates Jinja2)
    
    Args:
        params_config_id: Config params utilisée
        verdict_config_id: Config verdict
        output_dir: Répertoire sortie rapports
    
    Returns:
        dict: Résultats complets
        {
            'metadata': {...},
            'profiling_results': {      # ← PHASE 4
                'test': {...},
                'gamma': {...},
                'modifier': {...},
                'encoding': {...}
            },
            'structural_patterns': {...},
            'report_paths': {...}
        }
    """
    print(f"\n{'='*70}")
    print(f"VERDICT REPORTER R0 - GÉNÉRATION RAPPORTS (PHASE 4 - 4 AXES)")
    print(f"{'='*70}\n")
    
    print(f"Params config:  {params_config_id}")
    print(f"Verdict config: {verdict_config_id}\n")
    
    # =========================================================================
    # ÉTAPES 1-2 : CHARGEMENT + DIAGNOSTICS (INCHANGÉ)
    # =========================================================================
    
    print("1. Chargement observations...")
    observations = load_all_observations(params_config_id)
    print(f"   ✓ {len(observations)} observations chargées")
    
    observations, rejection_stats = filter_numeric_artifacts(observations)
    
    if rejection_stats['rejected_observations'] > 0:
        print(f"   ⊘ Filtré {rejection_stats['rejected_observations']} observations "
              f"({rejection_stats['rejection_rate']*100:.1f}%) : artefacts numériques")
        for test, count in sorted(rejection_stats['rejected_by_test'].items()):
            print(f"      {test}: {count} invalides")
    print()
    
    print("2. Diagnostics numériques...")
    degeneracy_report = generate_degeneracy_report(observations)
    scale_report = diagnose_scale_outliers(observations)
    print(f"   ✓ Dégénérescences : {degeneracy_report['observations_with_flags']} observations")
    print(f"   ✓ Ruptures échelle : {scale_report['observations_with_outliers']} observations")
    print()
    
    # =========================================================================
    # ÉTAPE 3 : ANALYSES GLOBALES STRATIFIÉES (INCHANGÉ)
    # =========================================================================
    
    print("3. Analyses globales stratifiées...")
    
    obs_stable, obs_explosif = stratify_by_regime(observations)
    print(f"   Régime STABLE   : {len(obs_stable)} observations ({len(obs_stable)/len(observations)*100:.1f}%)")
    print(f"   Régime EXPLOSIF : {len(obs_explosif)} observations ({len(obs_explosif)/len(observations)*100:.1f}%)")
    
    print("\n   Analyse GLOBAL...")
    results_global = analyze_regime(
        observations, 'GLOBAL',
        params_config_id, verdict_config_id
    )
    
    print("   Analyse STABLE...")
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
    # ÉTAPE 4 : PROFILING MULTI-AXES (PHASE 4 - MODIFIÉ)
    # =========================================================================
    
    print("4. Profiling multi-axes (4 axes : test, gamma, modifier, encoding)...")
    
    # Appel profiling_runner (orchestration 4 axes)
    profiling_results = run_all_profiling(
        observations,
        axes=['test', 'gamma', 'modifier', 'encoding']
    )
    
    print(f"   ✓ Test     : {profiling_results['test']['metadata']['n_entities']} entités profilées")
    print(f"   ✓ Gamma    : {profiling_results['gamma']['metadata']['n_entities']} entités profilées")
    print(f"   ✓ Modifier : {profiling_results['modifier']['metadata']['n_entities']} entités profilées")
    print(f"   ✓ Encoding : {profiling_results['encoding']['metadata']['n_entities']} entités profilées")
    print()
    
    # =========================================================================
    # ÉTAPE 5 : FUSION RÉSULTATS (PHASE 4 - MODIFIÉ)
    # =========================================================================
    
    print("5. Fusion résultats...")
    
    # Extraire entités disponibles (inline, pas fonction dédiée)
    gammas = sorted(set(obs['gamma_id'] for obs in observations))
    tests = sorted(set(obs['test_name'] for obs in observations))
    modifiers = sorted(set(obs['modifier_id'] for obs in observations))
    encodings = sorted(set(obs['d_encoding_id'] for obs in observations))
    
    final_results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'engine_version': '6.1',
            'architecture': 'verdict_reporter_r0_phase4_multiaxis',
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
                'n_modifiers': len(modifiers),
                'n_encodings': len(encodings),
                'gammas_list': gammas,
                'tests_list': tests,
                'modifiers_list': modifiers,
                'encodings_list': encodings
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
        },
        
        'profiling_results': profiling_results,  # ← PHASE 4 (4 axes)
        
        'structural_patterns': _compile_structural_patterns(
            results_global,
            results_stable,
            results_explosif
        ),
        
        'diagnostics': {
            'numeric_artifacts': rejection_stats,
            'degeneracy': degeneracy_report,
            'scale_outliers': scale_report
        }
    }
    
    print(f"   ✓ Structure compilée")
    print()
    
    # =========================================================================
    # ÉTAPE 6 : GÉNÉRATION RAPPORTS (PHASE 4 - MODIFIÉ)
    # =========================================================================
    
    print("6. Génération rapports multi-formats...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = Path(output_dir) / f"{timestamp}_verdict_r0"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_paths = {}
    
    # 6a. Metadata
    write_json(final_results['metadata'], report_dir / 'metadata.json')
    report_paths['metadata'] = str(report_dir / 'metadata.json')
    
    # 6b. Rapports par axe (4 × summary_*.txt via templates Jinja2)
    for axis in ['test', 'gamma', 'modifier', 'encoding']:
        print(f"   Génération summary_{axis}.txt (Jinja2)...")
        
        axis_results = profiling_results[axis]
        
        write_summary_axis(
            output_path=report_dir / f'summary_{axis}.txt',
            axis=axis,
            profiles=axis_results['profiles'],
            summary=axis_results['summary'],
            metadata=axis_results['metadata'],
            discriminant_powers=axis_results.get('discriminant_powers')  # Seulement axe test
        )
        report_paths[f'summary_{axis}'] = str(report_dir / f'summary_{axis}.txt')
    
    # 6c. Rapport global (synthèse multi-axes)
    print("   Génération summary_global.txt (Jinja2)...")
    from ..utils.report_writers import write_summary_global
    
    write_summary_global(
        output_path=report_dir / 'summary_global.txt',
        metadata=final_results['metadata'],
        profiling_results=profiling_results,
        structural_patterns=final_results['structural_patterns'],
        diagnostics=final_results['diagnostics']
    )
    report_paths['summary_global'] = str(report_dir / 'summary_global.txt')
    
    # 6d. Profils individuels (JSON par axe/entité)
    profiles_dir = report_dir / 'profiles'
    
    for axis in ['test', 'gamma', 'modifier', 'encoding']:
        axis_dir = profiles_dir / axis
        axis_dir.mkdir(parents=True, exist_ok=True)
        
        axis_profiles = profiling_results[axis]['profiles']
        
        for entity_id, entity_data in axis_profiles.items():
            entity_file = axis_dir / f"{entity_id}.json"
            write_json(entity_data, entity_file)
        
        report_paths[f'profiles_{axis}'] = str(axis_dir)
    
    # 6e. Structural patterns (analyses globales)
    write_json(final_results['structural_patterns'], report_dir / 'structural_patterns.json')
    report_paths['structural_patterns'] = str(report_dir / 'structural_patterns.json')
    
    # 6f. Diagnostics détaillés
    write_json(final_results['diagnostics'], report_dir / 'diagnostics.json')
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
    print("RAPPORT VERDICT R0 GÉNÉRÉ (PHASE 4 - 4 AXES PROFILING)")
    print("="*70)
    print(f"Répertoire : {report_dir}")
    print(f"Axes       : test, gamma, modifier, encoding")
    print(f"Entités    : {len(tests)} tests, {len(gammas)} gammas, "
          f"{len(modifiers)} modifiers, {len(encodings)} encodings")
    print(f"Fichiers   : {len(report_paths)}")
    print("="*70 + "\n")
    
    final_results['report_paths'] = report_paths
    return final_results


# =============================================================================
# COMPILATION METADATA (LOCAL - orchestration)
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
        'engine_version': '6.1',
        'architecture': 'verdict_reporter_r0_migrated_profiling_common',
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
# COMPILATION STRUCTURAL PATTERNS (LOCAL - spécifique verdict)
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


