# tests/utilities/regime_utils.py
"""
Regime Utilities - Stratification et classification régimes.

RESPONSABILITÉS :
- Stratification observations (stable/explosif)
- Classification régime comportemental
- Détection propriétés conservées
- Taxonomie régimes (CONSERVES_X, pathologies)

ARCHITECTURE :
- stratify_by_regime() : Séparation stable/explosif
- classify_regime() : Régime spécifique par test
- detect_conserved_property() : Déduction propriété

PRINCIPE R0 :
- Régimes SPÉCIFIQUES (CONSERVES_SYMMETRY vs CONSERVES_NORM)
- Qualificatif MIXED:: pour multimodalité
- Taxonomie extensible (nouveaux régimes faciles à ajouter)

UTILISATEURS :
- verdict_engine.py (stratification globale)
- gamma_profiling.py (classification individuelle)
"""

import numpy as np
from typing import List, Dict, Tuple


# =============================================================================
# STRATIFICATION STABLE/EXPLOSIF
# =============================================================================

def stratify_by_regime(
    observations: List[Dict],
    threshold: float = 1e50
) -> Tuple[List[Dict], List[Dict]]:
    """
    Stratifie observations en régimes stable/explosif.
    
    Critère : présence valeurs >threshold dans projections exploitées.
    Conserve TOUTES observations (aucun filtrage).
    
    Args:
        observations: Liste observations complètes
        threshold: Seuil magnitude extrême (défaut 1e50)
    
    Returns:
        (obs_stable, obs_explosif)
    
    Examples:
        >>> stable, explosif = stratify_by_regime(observations)
        >>> len(stable) / len(observations)
        0.85  # 85% stable
        >>> len(explosif) / len(observations)
        0.15  # 15% explosif
    """
    stable = []
    explosif = []
    
    for obs in observations:
        obs_data = obs.get('observation_data', {})
        stats = obs_data.get('statistics', {})
        evol = obs_data.get('evolution', {})
        
        has_extreme = False
        
        # Vérifier toutes projections exploitées
        for metric_stats in stats.values():
            for key in ['initial', 'final', 'mean', 'max']:
                val = metric_stats.get(key)
                if val is not None and abs(val) > threshold:
                    has_extreme = True
                    break
            if has_extreme:
                break
        
        if not has_extreme:
            for metric_evol in evol.values():
                for key in ['slope', 'relative_change']:
                    val = metric_evol.get(key)
                    if val is not None and abs(val) > threshold:
                        has_extreme = True
                        break
                if has_extreme:
                    break
        
        if has_extreme:
            explosif.append(obs)
        else:
            stable.append(obs)
    
    return stable, explosif


# =============================================================================
# CLASSIFICATION RÉGIME
# =============================================================================

def classify_regime(
    metrics: Dict,
    dynamic_sig: Dict,
    timeline_dist: Dict,
    dispersion: Dict,
    test_name: str
) -> str:
    """
    Classification régime R0 avec régimes SPÉCIFIQUES.
    
    Au lieu de CONSERVES_X générique, on détecte :
    - CONSERVES_SYMMETRY (SYM-*)
    - CONSERVES_NORM (SPE-*, UNIV-*)
    - CONSERVES_PATTERN (PAT-*)
    - CONSERVES_TOPOLOGY (TOP-*)
    - CONSERVES_GRADIENT (GRA-*)
    - CONSERVES_SPECTRUM (SPA-*)
    
    Régimes pathologiques :
    - NUMERIC_INSTABILITY, OSCILLATORY_UNSTABLE, TRIVIAL, 
      DEGRADING, SATURATES_HIGH, UNCATEGORIZED
    
    Si bimodal détecté → MIXED::{régime_base}
    
    Args:
        metrics: Retour aggregate_summary_metrics()
        dynamic_sig: Signatures dynamiques
        timeline_dist: Distribution timelines
        dispersion: Retour aggregate_run_dispersion()
        test_name: Nom du test (pour déduire propriété)
    
    Returns:
        str: Régime (ex: 'CONSERVES_SYMMETRY', 'MIXED::CONSERVES_NORM')
    
    Examples:
        >>> regime = classify_regime(metrics, dyn_sig, timeline, disp, 'SYM-001')
        >>> regime
        'CONSERVES_SYMMETRY'
        
        >>> regime = classify_regime(metrics, dyn_sig, timeline, disp, 'SPE-002')
        >>> regime
        'MIXED::CONSERVES_NORM'  # Multimodal
    """
    if not metrics:
        return "NO_DATA"
    
    final = metrics['final_value']['median']
    initial = metrics['initial_value']
    cv = metrics['cv']
    
    instability_onset = dynamic_sig.get('instability_onset_median')
    
    # PATHOLOGIES (prioritaires)
    if instability_onset is not None and instability_onset < 20 and final > 1e20:
        base_regime = "NUMERIC_INSTABILITY"
    elif dynamic_sig['oscillatory_fraction'] > 0.3:
        base_regime = "OSCILLATORY_UNSTABLE"
    elif cv < 0.01:
        base_regime = "TRIVIAL"
    
    # CONSERVATION (dépend du test)
    elif final < 2 * initial and cv < 0.1:
        base_regime = detect_conserved_property(test_name)
    
    # SATURATION
    elif 'saturation' in timeline_dist.get('dominant_timeline', '') and dynamic_sig['saturation_fraction'] > 0.7:
        if final > 10 * initial:
            base_regime = "SATURATES_HIGH"
        else:
            # Saturation mais pas croissance → Conservation
            base_regime = detect_conserved_property(test_name)
    
    # DEGRADING
    elif final < 0.5 * initial and dynamic_sig['collapse_fraction'] < 0.1:
        base_regime = "DEGRADING"
    
    else:
        base_regime = "UNCATEGORIZED"
    
    # Qualificatif multimodalité
    if dispersion['bimodal_detected']:
        return f"MIXED::{base_regime}"
    
    return base_regime


def detect_conserved_property(test_name: str) -> str:
    """
    Détermine propriété conservée selon préfixe test.
    
    Args:
        test_name: ID test (ex: 'SYM-001', 'SPE-002')
    
    Returns:
        str: Régime conservation spécifique
    
    Mapping :
        SYM-* → CONSERVES_SYMMETRY
        SPE-*, UNIV-* → CONSERVES_NORM
        PAT-* → CONSERVES_PATTERN
        TOP-* → CONSERVES_TOPOLOGY
        GRA-* → CONSERVES_GRADIENT
        SPA-* → CONSERVES_SPECTRUM
        Autre → CONSERVES_PROPERTY (fallback générique)
    
    Examples:
        >>> detect_conserved_property('SYM-001')
        'CONSERVES_SYMMETRY'
        >>> detect_conserved_property('UNIV-002')
        'CONSERVES_NORM'
    """
    if test_name.startswith('SYM-'):
        return "CONSERVES_SYMMETRY"
    elif test_name.startswith('SPE-') or test_name.startswith('UNIV-'):
        return "CONSERVES_NORM"
    elif test_name.startswith('PAT-'):
        return "CONSERVES_PATTERN"
    elif test_name.startswith('TOP-'):
        return "CONSERVES_TOPOLOGY"
    elif test_name.startswith('GRA-'):
        return "CONSERVES_GRADIENT"
    elif test_name.startswith('SPA-'):
        return "CONSERVES_SPECTRUM"
    else:
        return "CONSERVES_PROPERTY"  # Fallback générique


# =============================================================================
# EXTRACTION PROPRIÉTÉS CONSERVÉES
# =============================================================================

def extract_conserved_properties(profile: Dict) -> List[str]:
    """
    Extrait propriétés conservées depuis profil gamma.
    
    Args:
        profile: Profil gamma (avec tests)
    
    Returns:
        list[str]: Propriétés conservées
    
    Examples:
        >>> props = extract_conserved_properties(gamma_profile)
        >>> props
        ['Symétrie', 'Norme', 'Pattern']
    """
    properties = set()
    
    for test_data in profile.get('tests', {}).values():
        regime = test_data.get('regime', '')
        
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


# =============================================================================
# TAXONOMIE RÉGIMES (RÉFÉRENCE)
# =============================================================================

REGIME_TAXONOMY = {
    # Conservation (sains)
    'CONSERVES_SYMMETRY': {
        'family': 'conservation',
        'description': 'Asymétrie finale < 1e-6',
        'tests': ['SYM-*']
    },
    'CONSERVES_NORM': {
        'family': 'conservation',
        'description': 'Norme finale < 2× initiale',
        'tests': ['SPE-*', 'UNIV-*']
    },
    'CONSERVES_PATTERN': {
        'family': 'conservation',
        'description': 'Diversity/uniformity stables',
        'tests': ['PAT-*']
    },
    'CONSERVES_TOPOLOGY': {
        'family': 'conservation',
        'description': 'Euler characteristic stable',
        'tests': ['TOP-*']
    },
    'CONSERVES_GRADIENT': {
        'family': 'conservation',
        'description': 'Structure gradients conservée',
        'tests': ['GRA-*']
    },
    'CONSERVES_SPECTRUM': {
        'family': 'conservation',
        'description': 'Spectre valeurs propres stable',
        'tests': ['SPA-*']
    },
    
    # Pathologies
    'NUMERIC_INSTABILITY': {
        'family': 'pathology',
        'description': 'Instability onset < 20 && final > 1e20',
        'tests': ['Tous']
    },
    'OSCILLATORY_UNSTABLE': {
        'family': 'pathology',
        'description': 'Oscillatory fraction > 30%',
        'tests': ['Tous']
    },
    'TRIVIAL': {
        'family': 'pathology',
        'description': 'CV < 1% (aucune variation)',
        'tests': ['Tous']
    },
    'DEGRADING': {
        'family': 'pathology',
        'description': 'Final < 0.5 × initial (sans collapse)',
        'tests': ['Tous']
    },
    
    # Autres
    'SATURATES_HIGH': {
        'family': 'saturation',
        'description': 'Saturation + final > 10 × initial',
        'tests': ['Tous']
    },
    'UNCATEGORIZED': {
        'family': 'other',
        'description': 'Comportement non classifié',
        'tests': ['Tous']
    }
}


def get_regime_family(regime: str) -> str:
    """
    Retourne famille d'un régime.
    
    Args:
        regime: Nom régime (avec ou sans MIXED::)
    
    Returns:
        'conservation' | 'pathology' | 'saturation' | 'other'
    
    Examples:
        >>> get_regime_family('CONSERVES_SYMMETRY')
        'conservation'
        >>> get_regime_family('MIXED::CONSERVES_NORM')
        'conservation'
    """
    # Strip qualificatif MIXED::
    base_regime = regime.split('::')[-1] if '::' in regime else regime
    
    return REGIME_TAXONOMY.get(base_regime, {}).get('family', 'other')