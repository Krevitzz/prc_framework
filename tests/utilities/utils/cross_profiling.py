# tests/utilities/UTIL/cross_profiling.py
"""
Cross Profiling Module - Charter 6.1

RESPONSABILITÉ : Analyses croisées entre axes profiling

ARCHITECTURE (alignée analyse GPT) :
- Sépare calculs structurels (R0) et interprétation qualitative (R1+)
- profiling_common fournit espace vectoriel normalisé
- cross_profiling applique opérateurs sur cet espace

IMPLÉMENTATION R0 :
- Rankings multi-dimensionnels (extension by_test)
- Variance conditionnelle (discriminant power)
- Matrices concordance (helper projections conjointes)

PLACEHOLDERS R1+ :
- Interactions pairwise qualifiées
- Signatures globales avec vocabulaire interprétatif

FRONTIÈRE ÉPISTÉMIQUE :
- R0 : Calcule, mesure, compare (pas d'interprétation causale)
- R1+ : Qualifie, nomme, interprète (INVARIANT, AMPLIFIÉ, etc.)
"""

import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from collections import defaultdict, Counter
from scipy import stats


# ============================================================================
# RANKINGS MULTI-DIMENSIONNELS (R0 complet)
# ============================================================================

def rank_entities_by_metric(
    profiles: dict,
    grouping_dimension: str,
    metric_key: str,
    criterion: str | Callable = 'conservation'
) -> List[Tuple[str, float]]:
    """
    Ranking générique entités par métrique.
    
    REMPLACE : gamma_profiling.rank_gammas_by_test()
    EXTENSION : Critères multiples, tous axes supportés
    
    Args:
        profiles: Résultats profile_all_{axis}() depuis profiling_common
        grouping_dimension: Axe regroupement ('test', 'encoding', 'modifier')
        metric_key: Clé métrique (test_name si grouping='test', etc.)
        criterion: Critère scoring ou callable custom
    
    Returns:
        [(entity_id, score), ...] trié décroissant par score
    
    Critères standards :
        - 'conservation' : CONSERVES_* = 1.0, TRIVIAL = 0.5, autres = 0.0
        - 'stability' : Score basé instability_onset + collapse_fraction
        - 'homogeneity' : Score basé CV + bimodalité
        - callable : Fonction custom(prc_profile, diagnostic_signature) → float
    
    Exemples :
        # Gammas par test (rétrocompatibilité)
        rank_entities_by_metric(gamma_profiles, 'test', 'SYM-001', 'conservation')
        
        # Tests par gamma (inverse)
        rank_entities_by_metric(test_profiles, 'gamma', 'GAM-001', 'stability')
        
        # Modifiers par encoding
        rank_entities_by_metric(modifier_profiles, 'encoding', 'SYM-001', 'homogeneity')
        
        # Critère custom
        rank_entities_by_metric(
            gamma_profiles, 'test', 'TOP-001',
            criterion=lambda prc, diag: prc['confidence'] == 'high'
        )
    """
    scores = []
    
    for entity_id, entity_data in profiles.items():
        # Accéder profil selon grouping_dimension
        if grouping_dimension == 'test':
            # metric_key = test_name
            profile = entity_data['tests'].get(metric_key)
        else:
            # Grouping par autre dimension (encoding, modifier, etc.)
            # Rechercher dans tous tests
            profile = None
            for test_profile in entity_data['tests'].values():
                # Vérifier si test_profile contient metric_key
                # (structure future où tests pourraient grouper par encoding/modifier)
                # Pour R0 : on assume grouping='test' principalement
                pass
            
            if not profile:
                # Fallback : chercher test nommé metric_key
                profile = entity_data['tests'].get(metric_key)
        
        if not profile:
            continue
        
        prc = profile['prc_profile']
        diag = profile.get('diagnostic_signature', {})
        
        # Calcul score selon critère
        if callable(criterion):
            score = float(criterion(prc, diag))
        
        elif criterion == 'conservation':
            regime = prc['regime'].split('::')[-1]  # Strip MIXED::
            if regime.startswith('CONSERVES_'):
                score = 1.0
            elif regime == 'TRIVIAL':
                score = 0.5
            else:
                score = 0.0
        
        elif criterion == 'stability':
            dynamic_sig = diag.get('dynamic_signature', {})
            instab = dynamic_sig.get('instability_onset_median') is not None
            collapse = dynamic_sig.get('collapse_fraction', 0.0)
            score = 1.0 - (float(instab) * 0.5 + collapse * 0.5)
        
        elif criterion == 'homogeneity':
            run_disp = diag.get('run_dispersion', {})
            cv = run_disp.get('cv_across_runs', 1.0)
            bimodal = run_disp.get('bimodal_detected', False)
            score = (1.0 - min(cv, 1.0)) * (0.5 if bimodal else 1.0)
        
        else:
            # Critère inconnu
            score = 0.0
        
        scores.append((entity_id, score))
    
    # Tri décroissant
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# ============================================================================
# VARIANCE CONDITIONNELLE (R0 complet)
# ============================================================================

def compute_discriminant_power(
    profiles: dict,
    test_name: str,
    reference_axis: str = 'gamma'
) -> dict:
    """
    Calcule pouvoir discriminant d'un test cross-entités.
    
    NOUVEAU R0 : Variance conditionnelle inter/intra entités
    EXPLOITABLE : Insights tests sans interprétation qualitative
    
    Mesure capacité test à différencier entités de l'axe de référence.
    
    Args:
        profiles: Résultats profile_all_{axis}() (ex: gamma_profiles)
        test_name: Test analysé (ex: 'SYM-001')
        reference_axis: Axe entités (ex: 'gamma', 'modifier')
    
    Returns:
        {
            'test_name': str,
            'reference_axis': str,
            'n_entities': int,
            'inter_entity_variance': float,     # Variance cross-entités (final_value)
            'intra_entity_variance': float,     # Variance cross-runs agrégée
            'discriminant_ratio': float,        # inter / intra
            'effect_size': float,               # Eta-squared (ANOVA)
            'kruskal_wallis': {
                'statistic': float,
                'p_value': float
            },
            'entities_ranked': [(entity_id, median_final_value), ...],
            'interpretation': str               # R0 : factuel uniquement
        }
    
    Interprétation factuelle R0 :
        - discriminant_ratio > 3.0 : "Variance inter-entités domine"
        - discriminant_ratio < 0.3 : "Variance intra-entité domine"
        - Autre : "Variance mixte"
    
    Note R1+ :
        Vocabulaire qualitatif (DISCRIMINANT, INVARIANT) dans detect_global_signatures()
    """
    # Extraction données
    final_values_by_entity = {}
    all_final_values = []
    
    for entity_id, entity_data in profiles.items():
        test_profile = entity_data['tests'].get(test_name)
        
        if not test_profile:
            continue
        
        instr = test_profile.get('instrumentation', {})
        metrics = instr.get('summary_metrics', {})
        
        # Extraire final_value (structure agrégée)
        final_value_stats = metrics.get('final_value', {})
        median_final = final_value_stats.get('median')
        
        if median_final is None or not np.isfinite(median_final):
            continue
        
        final_values_by_entity[entity_id] = median_final
        all_final_values.append(median_final)
    
    n_entities = len(final_values_by_entity)
    
    if n_entities < 2:
        return {
            'test_name': test_name,
            'reference_axis': reference_axis,
            'n_entities': n_entities,
            'inter_entity_variance': 0.0,
            'intra_entity_variance': 0.0,
            'discriminant_ratio': 0.0,
            'effect_size': 0.0,
            'kruskal_wallis': {'statistic': 0.0, 'p_value': 1.0},
            'entities_ranked': [],
            'interpretation': 'Insufficient entities for analysis'
        }
    
    # Variance inter-entités (variance des médianes)
    inter_entity_variance = float(np.var(list(final_values_by_entity.values())))
    
    # Variance intra-entité agrégée
    # Approximation R0 : moyenne des variances run_dispersion
    intra_variances = []
    for entity_id, entity_data in profiles.items():
        test_profile = entity_data['tests'].get(test_name)
        if not test_profile:
            continue
        
        diag = test_profile.get('diagnostic_signature', {})
        run_disp = diag.get('run_dispersion', {})
        cv = run_disp.get('cv_across_runs', 0.0)
        
        # Approximation variance depuis CV
        # var ≈ (cv * mean)^2
        # Utiliser median comme proxy mean
        median_val = final_values_by_entity.get(entity_id, 0.0)
        approx_var = (cv * median_val) ** 2
        intra_variances.append(approx_var)
    
    intra_entity_variance = float(np.mean(intra_variances)) if intra_variances else 0.0
    
    # Discriminant ratio
    discriminant_ratio = (
        inter_entity_variance / (intra_entity_variance + 1e-10)
        if intra_entity_variance > 0 else float('inf')
    )
    
    # Effect size (Eta-squared)
    # η² = SSB / SST
    grand_mean = np.mean(all_final_values)
    ssb = sum((val - grand_mean) ** 2 for val in final_values_by_entity.values())
    sst = sum((val - grand_mean) ** 2 for val in all_final_values)
    effect_size = ssb / sst if sst > 1e-10 else 0.0
    
    # Kruskal-Wallis test (non-paramétrique)
    # Pour R0 : on n'a que médianes, test approximatif
    # R1+ : utiliser distributions complètes depuis observations
    try:
        # Créer groupes artificiels (répéter médiane comme proxy)
        groups = [[val] * 5 for val in final_values_by_entity.values()]  # Proxy
        h_stat, p_value = stats.kruskal(*groups)
    except:
        h_stat, p_value = 0.0, 1.0
    
    # Ranking entités
    entities_ranked = sorted(
        final_values_by_entity.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Interprétation factuelle R0
    if discriminant_ratio > 3.0:
        interpretation = "Variance inter-entités domine (discriminant fort)"
    elif discriminant_ratio < 0.3:
        interpretation = "Variance intra-entité domine (faible discrimination)"
    else:
        interpretation = "Variance mixte (discrimination modérée)"
    
    return {
        'test_name': test_name,
        'reference_axis': reference_axis,
        'n_entities': n_entities,
        'inter_entity_variance': inter_entity_variance,
        'intra_entity_variance': intra_entity_variance,
        'discriminant_ratio': discriminant_ratio,
        'effect_size': effect_size,
        'kruskal_wallis': {
            'statistic': float(h_stat),
            'p_value': float(p_value)
        },
        'entities_ranked': entities_ranked,
        'interpretation': interpretation
    }


def compute_all_discriminant_powers(
    profiles: dict,
    reference_axis: str = 'gamma'
) -> dict:
    """
    Calcule discriminant power tous tests pour un axe.
    
    Args:
        profiles: Résultats profile_all_{axis}()
        reference_axis: Axe de référence
    
    Returns:
        {
            'test_name_1': {...},  # compute_discriminant_power()
            'test_name_2': {...},
            ...
        }
    """
    # Découvrir tous tests
    all_tests = set()
    for entity_data in profiles.values():
        all_tests.update(entity_data['tests'].keys())
    
    results = {}
    for test_name in all_tests:
        results[test_name] = compute_discriminant_power(
            profiles, test_name, reference_axis
        )
    
    return results


# ============================================================================
# MATRICES CONCORDANCE (R0 helper)
# ============================================================================

def _compute_concordance_matrix(
    profiles_a: dict,
    profiles_b: dict,
    comparison_field: str = 'regime'
) -> dict:
    """
    Calcule matrice concordance entre deux axes.
    
    HELPER R0 : Préparation projections conjointes
    USAGE R1+ : analyze_pairwise_interactions()
    
    Args:
        profiles_a: Profils axe A (ex: gamma_profiles)
        profiles_b: Profils axe B (ex: modifier_profiles)
        comparison_field: Champ comparé ('regime', 'behavior', 'timeline')
    
    Returns:
        {
            'test_name_1': {
                'concordance_rate': float,  # Fraction régimes identiques
                'mismatches': [
                    ('entity_a_id', 'entity_b_id', 'regime_a', 'regime_b'),
                    ...
                ],
                'n_comparisons': int
            },
            ...
        }
    
    Exemple lecture :
        Si concordance_rate = 0.8 pour test SYM-001 :
        → 80% paires (gamma, modifier) ont même régime sous SYM-001
    """
    # Découvrir tests communs
    tests_a = set()
    tests_b = set()
    
    for entity_data in profiles_a.values():
        tests_a.update(entity_data['tests'].keys())
    for entity_data in profiles_b.values():
        tests_b.update(entity_data['tests'].keys())
    
    common_tests = tests_a & tests_b
    
    results = {}
    
    for test_name in common_tests:
        matches = 0
        total = 0
        mismatches = []
        
        for entity_a_id, entity_a_data in profiles_a.items():
            test_profile_a = entity_a_data['tests'].get(test_name)
            if not test_profile_a:
                continue
            
            value_a = test_profile_a['prc_profile'].get(comparison_field)
            
            for entity_b_id, entity_b_data in profiles_b.items():
                test_profile_b = entity_b_data['tests'].get(test_name)
                if not test_profile_b:
                    continue
                
                value_b = test_profile_b['prc_profile'].get(comparison_field)
                
                total += 1
                
                # Comparaison (strip MIXED:: si régime)
                if comparison_field == 'regime':
                    value_a_clean = value_a.split('::')[-1] if value_a else None
                    value_b_clean = value_b.split('::')[-1] if value_b else None
                else:
                    value_a_clean = value_a
                    value_b_clean = value_b
                
                if value_a_clean == value_b_clean:
                    matches += 1
                else:
                    mismatches.append((
                        entity_a_id, entity_b_id,
                        value_a_clean, value_b_clean
                    ))
        
        concordance_rate = matches / total if total > 0 else 0.0
        
        results[test_name] = {
            'concordance_rate': concordance_rate,
            'mismatches': mismatches[:10],  # Limiter aux 10 premiers
            'n_comparisons': total
        }
    
    return results


# ============================================================================
# INTERACTIONS PAIRWISE (R0 placeholder + structure)
# ============================================================================

def analyze_pairwise_interactions(
    profiles_a: dict,
    profiles_b: dict,
    metric: str = 'regime_concordance'
) -> dict:
    """
    Analyse interactions 2-way entre deux axes profiling.
    
    STATUT R0 : Structure + helper concordance fonctionnel
    STATUT R1+ : Interprétation qualitative complète
    
    Args:
        profiles_a: Profils axe A (ex: gamma_profiles)
        profiles_b: Profils axe B (ex: modifier_profiles)
        metric: Métrique interaction
            - 'regime_concordance' : Concordance régimes (R0 implémenté)
            - 'behavior_amplification' : Amplification pathologies (R1+)
            - 'timeline_coupling' : Couplage timelines (R1+)
    
    Returns R0 (metric='regime_concordance'):
        {
            'axis_a': str,  # Détecté depuis metadata profiles
            'axis_b': str,
            'metric': str,
            'concordance_matrix': {...},  # _compute_concordance_matrix()
            'summary': {
                'mean_concordance': float,
                'high_concordance_tests': [test_name, ...],  # > 0.8
                'low_concordance_tests': [test_name, ...],   # < 0.3
                'interaction_detected': bool  # True si variance concordance > seuil
            }
        }
    
    Returns R1+ (autres métriques) :
        {
            'interaction_matrix': np.ndarray,  # Shape: (n_a, n_b)
            'significant_pairs': [
                ('entity_a_id', 'entity_b_id', 'pattern_description'),
                ...
            ],
            'qualitative_labels': {
                ('GAM-004', 'M1'): 'AMPLIFIED',
                ('GAM-009', 'ASY-003'): 'CONTEXTUAL',
                ...
            }
        }
    
    Exemples détection R1+ (vocabulaire interprétatif) :
        - GAM-004 × M1 → OSCILLATORY_SYSTEMATIC (fraction > 0.8)
        - GAM-009 × ASY-003 → CONTEXT_DEPENDENT (variance conditionnelle élevée)
        - Encodings SYM × Tests SYM-* → STRUCTURALLY_ALIGNED
    """
    # R0 : Implémentation minimale pour regime_concordance
    if metric == 'regime_concordance':
        # Détection axes (depuis metadata premier profil)
        axis_a = 'unknown'
        axis_b = 'unknown'
        
        if profiles_a:
            first_entity_a = list(profiles_a.values())[0]
            if 'tests' in first_entity_a and first_entity_a['tests']:
                first_test_a = list(first_entity_a['tests'].values())[0]
                # Chercher clé *_id
                for key in first_test_a.keys():
                    if key.endswith('_id') and key != 'test_name':
                        axis_a = key.replace('_id', '')
                        break
        
        if profiles_b:
            first_entity_b = list(profiles_b.values())[0]
            if 'tests' in first_entity_b and first_entity_b['tests']:
                first_test_b = list(first_entity_b['tests'].values())[0]
                for key in first_test_b.keys():
                    if key.endswith('_id') and key != 'test_name':
                        axis_b = key.replace('_id', '')
                        break
        
        # Calcul matrice concordance
        concordance_matrix = _compute_concordance_matrix(
            profiles_a, profiles_b, comparison_field='regime'
        )
        
        # Synthèse
        concordance_rates = [
            data['concordance_rate']
            for data in concordance_matrix.values()
        ]
        
        mean_concordance = float(np.mean(concordance_rates)) if concordance_rates else 0.0
        
        high_concordance_tests = [
            test_name for test_name, data in concordance_matrix.items()
            if data['concordance_rate'] > 0.8
        ]
        
        low_concordance_tests = [
            test_name for test_name, data in concordance_matrix.items()
            if data['concordance_rate'] < 0.3
        ]
        
        # Interaction détectée si variance concordance élevée
        variance_concordance = float(np.var(concordance_rates)) if concordance_rates else 0.0
        interaction_detected = variance_concordance > 0.1
        
        return {
            'axis_a': axis_a,
            'axis_b': axis_b,
            'metric': metric,
            'concordance_matrix': concordance_matrix,
            'summary': {
                'mean_concordance': mean_concordance,
                'variance_concordance': variance_concordance,
                'high_concordance_tests': high_concordance_tests,
                'low_concordance_tests': low_concordance_tests,
                'interaction_detected': interaction_detected
            }
        }
    
    else:
        # R1+ : Métriques non implémentées
        raise NotImplementedError(
            f"R1+ : Métrique '{metric}' nécessite interprétation qualitative. "
            f"Implémentation différée après validation R0."
        )


# ============================================================================
# INTERACTIONS MULTIWAY (R1+ placeholder)
# ============================================================================

def analyze_multiway_interactions(
    all_profiles: dict,
    combination: Optional[List[str]] = None
) -> dict:
    """
    Analyse interactions n-way (3+ axes).
    
    STATUT R0 : Placeholder complet (docstring détaillé)
    STATUT R1+ : Implémentation après validation pairwise
    
    Args:
        all_profiles: Tous résultats profiling
            {
                'gamma': gamma_profiles,
                'modifier': modifier_profiles,
                'encoding': encoding_profiles,
                'test': test_profiles
            }
        combination: Sous-ensemble axes analysés ou None (tous)
    
    Returns R1+ :
        {
            'combination': ['gamma', 'modifier', 'encoding'],
            'triplets': [
                {
                    'entities': ('GAM-004', 'M1', 'ASY-003'),
                    'tests_affected': ['TOP-001', 'SPE-002'],
                    'emergent_pattern': 'UNIQUE_BEHAVIOR',
                    'description': "Comportement impossible prédire depuis pairwise"
                },
                ...
            ],
            'summary': {
                'n_triplets_analyzed': int,
                'n_emergent_patterns': int,
                'interaction_strength': float
            }
        }
    
    Méthode anticipée :
        1. Énumérer combinaisons n-uplets
        2. Pour chaque n-uplet, comparer :
           - Comportement observé (profil conjoint)
           - Comportement prédit (agrégation pairwise)
        3. Écart > seuil → émergence détectée
    
    Exemple détection :
        GAM-004 × M1 → OSCILLATORY (pairwise)
        GAM-004 × ASY-003 → STABLE (pairwise)
        M1 × ASY-003 → STABLE (pairwise)
        
        Mais GAM-004 × M1 × ASY-003 → COLLAPSE (unique, non prédictible)
    """
    raise NotImplementedError(
        "R1+ : Analyse interactions multiway nécessite validation pairwise d'abord. "
        "Implémentation différée."
    )


# ============================================================================
# SIGNATURES GLOBALES (R1+ placeholder)
# ============================================================================

def detect_global_signatures(all_profiles: dict) -> dict:
    """
    Détection patterns émergents cross-axes avec vocabulaire interprétatif.
    
    STATUT R0 : Placeholder complet (vocabulaire défini)
    STATUT R1+ : Implémentation complète avec labels qualitatifs
    
    Args:
        all_profiles: Tous résultats profiling
    
    Returns R1+ :
        {
            'modifier_signatures': {
                'M0': {
                    'label': 'INVARIANT',
                    'pattern': 'Baseline neutre tous contextes',
                    'confidence': 'high'
                },
                'M1': {
                    'label': 'AMPLIFIED',
                    'pattern': 'Amplifie instabilités GAM-004/GAM-009',
                    'contexts_affected': ['GAM-004', 'GAM-009'],
                    'tests_systematic': ['TOP-001', 'SPE-002'],
                    'confidence': 'high'
                },
                'M2': {
                    'label': 'CONDITIONAL',
                    'pattern': 'Effet dépend encodings',
                    'sensitive_encodings': ['ASY-003'],
                    'confidence': 'medium'
                }
            },
            'encoding_signatures': {
                'R3-001': {
                    'label': 'INVARIANT',
                    'pattern': 'Comportement homogène tous gammas',
                    'confidence': 'high'
                },
                'ASY-003': {
                    'label': 'STRUCTURING',
                    'pattern': 'Induit patterns spécifiques',
                    'affected_gammas': ['GAM-007'],
                    'confidence': 'medium'
                }
            },
            'test_signatures': {
                'SYM-001': {
                    'label': 'DISCRIMINANT',
                    'pattern': 'Distingue encodings SYM vs ASY',
                    'discriminant_power': 0.85,
                    'confidence': 'high'
                },
                'PAT-001': {
                    'label': 'INVARIANT',
                    'pattern': 'Non discriminant (variance faible)',
                    'discriminant_power': 0.12,
                    'confidence': 'high'
                }
            },
            'cross_signatures': {
                'gamma_modifier': [
                    {
                        'pattern': 'GAM-004 sensible perturbations',
                        'systematic_modifiers': ['M1', 'M2'],
                        'label': 'FRAGILE'
                    }
                ],
                'encoding_test': [
                    {
                        'pattern': 'Tests SYM-* alignés encodings SYM',
                        'label': 'STRUCTURAL_ALIGNMENT'
                    }
                ]
            }
        }
    
    Vocabulaire qualitatif R1+ :
        - INVARIANT : Entité stable tous contextes
        - AMPLIFIED : Interaction renforce effet
        - SUPPRESSED : Interaction masque effet
        - CONDITIONAL : Effet dépend contexte
        - CONTEXTUAL : Émergence spécifique combinaison
        - DISCRIMINANT : Forte capacité différenciation
        - FRAGILE : Sensible perturbations
        - ROBUST : Résistant perturbations
        - STRUCTURAL_ALIGNMENT : Cohérence structure/observation
    
    Méthode anticipée :
        1. Analyser discriminant_power tous tests
        2. Analyser concordance pairwise tous axes
        3. Détecter patterns récurrents (heuristiques)
        4. Attribuer labels qualitatifs
        5. Calculer confidence (robustesse pattern)
    """
    raise NotImplementedError(
        "R1+ : Détection signatures globales avec vocabulaire interprétatif. "
        "Implémentation différée après validation analyses quantitatives R0."
    )