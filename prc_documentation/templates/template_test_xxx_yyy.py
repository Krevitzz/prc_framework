# tests/utilities/template_test_xxx_yyy.py

import numpy as np
from typing import Optional, Tuple

# ============================================================================
# MÉTADONNÉES OBLIGATOIRES
# ============================================================================
TEST_ID = "XXX-NNN"  # Ex: "UNIV-001"
TEST_CATEGORY = "XXX"  # "UNIV" | "SYM" | "STR" | "BND" | "LOC" | etc.
REQUIRES_RANK = None | 2 | 3  # None = tous rangs, sinon spécifique
D_TYPES = ["SYM", "ASY", "R3"]  # Types D compatibles

# ============================================================================
# SPÉCIFICATION PARAMÈTRES
# ============================================================================
PARAMETERS_SPEC = {
    'critical': {
        'param_name': {
            'type': float | int | str,
            'description': "Description claire",
            'required': True,  # Test SKIPPED si absent
        },
    },
    'optional': {
        'param_name': {
            'type': float | int,
            'description': "Description",
            'default': valeur,  # Valeur par défaut
        },
    },
}

# ============================================================================
# SPÉCIFICATION SCORING
# ============================================================================
SCORING_SPEC = {
    'available_metrics': [
        'stat_initial',
        'stat_final',
        'stat_min',
        'stat_max',
        'stat_mean',
        'stat_std',
        'evolution_transition',
        'evolution_trend',
        'evolution_trend_coefficient',
    ],
    'critical_for_scoring': [
        'stat_final',  # Scoring échoue si ces métriques absentes
        'evolution_transition',
    ],
}

# ============================================================================
# SPÉCIFICATION THRESHOLDS (documentation)
# ============================================================================
THRESHOLDS_SPEC = {
    'recommended_thresholds': {
        'stat_final': 0.5,
        'evolution_transition': 0.7,
    },
    
    'interpretation': {
        'stat_final': "Description interprétation valeurs",
        'evolution_transition': "Description transitions",
    },
    
    'config_pass_criteria': {
        'description': "Critère pour qu'une config passe ce test",
        'threshold_global': 0.6,  # Score global config ≥ seuil
    },
    
    'verdict_implications': """
    Documentation textuelle :
    - Si test échoue : implications
    - Si test passe : implications
    - Combinaison avec autres tests
    """
}

# ============================================================================
# APPLICABILITÉ
# ============================================================================
def is_applicable(context: dict) -> Tuple[bool, str]:
    """
    Vérifie applicabilité AVANT exécution.
    
    Args:
        context: {
            'd_base_id': str,
            'gamma_id': str,
            'modifier_id': str,
            'seed': int,
            'state_shape': tuple,
        }
    
    Returns:
        (applicable: bool, reason: str)
    """
    # Vérification rang
    if REQUIRES_RANK is not None:
        if len(context['state_shape']) != REQUIRES_RANK:
            return False, f"Nécessite rang {REQUIRES_RANK}, reçu {len(context['state_shape'])}"
    
    # Vérification type D
    d_type = context['d_base_id'].split('_')[0]
    if d_type not in D_TYPES:
        return False, f"Type D {d_type} non supporté, accepte {D_TYPES}"
    
    # Vérifications spécifiques (ex: matrice carrée)
    if REQUIRES_RANK == 2:
        if context['state_shape'][0] != context['state_shape'][1]:
            return False, "Nécessite matrice carrée"
    
    return True, ""

# ============================================================================
# FORMULE DE CALCUL (LA SEULE PARTIE SPÉCIFIQUE À CE TEST)
# ============================================================================
def run_test(history: list, context: dict, config_params_id: str) -> dict:
    """
    Mesure propriété factuelle avec formule spécifique au test.
    
    AUCUN jugement, AUCUNE interprétation.
    Retourne observations pures + statistiques.
    
    Args:
        history: Liste états [D_0, ..., D_T]
        context: dict {d_base_id, gamma_id, seed, modifier_id, state_shape}
        config_params_id: Identifiant config paramètres (ex: "params_default")
    
    Returns:
        dict {
            'test_name': str,
            'config_params_id': str,
            'status': 'SUCCESS' | 'ERROR',  # Statut opérationnel uniquement
            'message': str,
            'statistics': {initial, final, min, max, mean, std},
            'evolution': {transition, trend, trend_coefficient},
            'metadata': {...}
        }
        
        OU None si NOT_APPLICABLE (géré en amont par batch_runner)
    """
    # -------------------------------------------------------------------------
    # CHARGEMENT PARAMÈTRES (depuis config/test_parameters_<config_id>.yaml)
    # -------------------------------------------------------------------------
    params = get_test_parameters(TEST_ID, config_params_id)
    explosion_threshold = params['explosion_threshold']
    stability_tolerance = params['stability_tolerance']
    # ... autres paramètres spécifiques
    
    # -------------------------------------------------------------------------
    # VALIDATION HISTORIQUE
    # -------------------------------------------------------------------------
    if not history or len(history) < 2:
        return {
            'test_name': TEST_ID,
            'config_params_id': config_params_id,
            'status': 'ERROR',
            'message': 'Historique insuffisant (< 2 états)',
            'statistics': {
                'initial': 0.0, 'final': 0.0,
                'min': 0.0, 'max': 0.0,
                'mean': 0.0, 'std': 0.0,
            },
            'evolution': {
                'transition': 'error',
                'trend': 'unknown',
                'trend_coefficient': 0.0,
            },
            'metadata': {'error_type': 'insufficient_history'}
        }
    
    # -------------------------------------------------------------------------
    # CALCUL TIMESERIES (LOCAL - PAS STOCKÉ)
    # -------------------------------------------------------------------------
    values = []
    for i, state in enumerate(history):
        val = _compute_test_metric(state)  # ← FORMULE SPÉCIFIQUE
        if val is None:
            return {
                'test_name': TEST_ID,
                'config_params_id': config_params_id,
                'status': 'ERROR',
                'message': f'NaN/Inf détecté à itération {i}',
                'statistics': {
                    'initial': values[0] if values else 0.0,
                    'final': 0.0,
                    'min': min(values) if values else 0.0,
                    'max': max(values) if values else 0.0,
                    'mean': float(np.mean(values)) if values else 0.0,
                    'std': float(np.std(values)) if values else 0.0,
                },
                'evolution': {
                    'transition': 'nan_detected',
                    'trend': 'unknown',
                    'trend_coefficient': 0.0,
                },
                'metadata': {
                    'error_type': 'numerical_error',
                    'first_error_iteration': i
                }
            }
        values.append(val)
    
    # -------------------------------------------------------------------------
    # STATISTIQUES (sorties standardisées)
    # -------------------------------------------------------------------------
    statistics = {
        'initial': values[0],
        'final': values[-1],
        'min': min(values),
        'max': max(values),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
    }
    
    # -------------------------------------------------------------------------
    # ÉVOLUTION/TENDANCE (sorties standardisées)
    # -------------------------------------------------------------------------
    if len(values) > 1:
        trend_coef = float(np.polyfit(range(len(values)), values, 1)[0])
    else:
        trend_coef = 0.0
    
    if abs(trend_coef) < 1e-8:
        trend = "stable"
    elif trend_coef > 0:
        trend = "increasing"
    else:
        trend = "decreasing"
    
    relative_change = abs(statistics['final'] - statistics['initial']) / \
                      (statistics['initial'] + 1e-10)
    
    if statistics['max'] > explosion_threshold:
        transition = "explosive"
    elif relative_change < stability_tolerance:
        transition = "stable"
    elif statistics['final'] > statistics['initial']:
        transition = "increasing"
    else:
        transition = "decreasing"
    
    # -------------------------------------------------------------------------
    # RETOUR FORMAT STANDARDISÉ
    # -------------------------------------------------------------------------
    return {
        'test_name': TEST_ID,
        'config_params_id': config_params_id,
        'status': 'SUCCESS',
        'message': f"Mesure: {statistics['initial']:.2f} → {statistics['final']:.2f}",
        
        'statistics': statistics,
        
        'evolution': {
            'transition': transition,
            'trend': trend,
            'trend_coefficient': trend_coef,
        },
        
        'metadata': {
            'explosion_threshold': explosion_threshold,
            'stability_tolerance': stability_tolerance,
            'relative_change': relative_change,
            'num_iterations': len(values),
        }
    }

