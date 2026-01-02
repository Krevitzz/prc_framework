# tests/utilities/scoring_engine.py
"""
Moteur de scoring pathologies R0.

Architecture Charter 5.4 - Section 12.8
"""

import numpy as np
from typing import Dict, Any, Union


class ScoringEngine:
    """
    Moteur de détection pathologies R0.
    
    Responsabilités:
    1. Scorer une métrique selon type pathologie (S1-S4, MAPPING)
    2. Agréger scores métriques en score test
    3. Traçabilité complète (flags, evidence)
    """
    
    VERSION = "5.4"
    
    PATHOLOGY_TYPES = [
        "S1_COLLAPSE",      # Seuil bas
        "S2_EXPLOSION",     # Seuil haut
        "S3_PLATEAU",       # Intervalle toxique
        "S4_INSTABILITY",   # Variation différentielle
        "MAPPING"           # Mapping catégoriel
    ]
    
    def __init__(self):
        pass
    
    def score_metric(
        self,
        metric_key: str,
        metric_value: Union[float, str],
        scoring_rule: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Calcule pathology_score pour une métrique.
        
        Args:
            metric_key: Nom métrique (ex: "stat_final")
            metric_value: Valeur observée
            scoring_rule: Config YAML règle scoring
            context: Contexte optionnel (historique, delta)
        
        Returns:
            {
                'metric_key': str,
                'value': float | str,
                'score': float [0,1],
                'flag': bool,
                'pathology_type': str,
                'weight': float,
                'evidence': dict
            }
        
        Raises:
            ValueError: Si pathology_type inconnu
        """
        pathology_type = scoring_rule.get('pathology_type')
        
        if pathology_type not in self.PATHOLOGY_TYPES:
            raise ValueError(
                f"Unknown pathology_type: {pathology_type}. "
                f"Expected one of: {self.PATHOLOGY_TYPES}"
            )
        
        weight = scoring_rule.get('weight', 1.0)
        mode = scoring_rule.get('mode', 'soft')
        
        # Dispatch selon type
        if pathology_type == "S1_COLLAPSE":
            score, flag = self._score_s1_collapse(metric_value, scoring_rule, mode)
        
        elif pathology_type == "S2_EXPLOSION":
            score, flag = self._score_s2_explosion(metric_value, scoring_rule, mode)
        
        elif pathology_type == "S3_PLATEAU":
            score, flag = self._score_s3_plateau(metric_value, scoring_rule, mode)
        
        elif pathology_type == "S4_INSTABILITY":
            score, flag = self._score_s4_instability(metric_value, scoring_rule, context)
        
        elif pathology_type == "MAPPING":
            score, flag = self._score_mapping(metric_value, scoring_rule)
        
        else:
            # Ne devrait jamais arriver (déjà validé)
            raise ValueError(f"Unhandled pathology_type: {pathology_type}")
        
        return {
            'metric_key': metric_key,
            'value': metric_value,
            'score': float(score),
            'flag': bool(flag),
            'pathology_type': pathology_type,
            'weight': float(weight),
            'evidence': {
                'threshold_config': scoring_rule,
                'context': context or {},
                'mode': mode
            }
        }
    
    def _score_s1_collapse(
        self,
        value: float,
        rule: Dict[str, Any],
        mode: str
    ) -> tuple[float, bool]:
        """
        TYPE S1: COLLAPSE (seuil bas).
        
        Args:
            value: Valeur métrique
            rule: {threshold_low, critical_low?, mode}
            mode: 'soft' | 'hard'
        
        Returns:
            (score, flag)
        """
        threshold = rule['threshold_low']
        critical = rule.get('critical_low', threshold * 0.1)
        
        if mode == 'hard':
            score = 1.0 if value < threshold else 0.0
        else:  # soft
            if value >= threshold:
                score = 0.0
            else:
                # Normaliser [critical, threshold] → [1, 0]
                if value <= critical:
                    score = 1.0
                else:
                    score = (threshold - value) / (threshold - critical)
        
        flag = value < critical
        
        return score, flag
    
    def _score_s2_explosion(
        self,
        value: float,
        rule: Dict[str, Any],
        mode: str
    ) -> tuple[float, bool]:
        """
        TYPE S2: EXPLOSION (seuil haut).
        
        Args:
            value: Valeur métrique
            rule: {threshold_high, critical_high?, mode}
            mode: 'soft' | 'hard'
        
        Returns:
            (score, flag)
        """
        threshold = rule['threshold_high']
        critical = rule.get('critical_high', threshold * 10)
        
        if mode == 'hard':
            score = 1.0 if value > threshold else 0.0
        else:  # soft
            if value <= threshold:
                score = 0.0
            else:
                # Normaliser [threshold, critical] → [0, 1]
                if value >= critical:
                    score = 1.0
                else:
                    score = (value - threshold) / (critical - threshold)
        
        flag = value > critical
        
        return score, flag
    
    def _score_s3_plateau(
        self,
        value: float,
        rule: Dict[str, Any],
        mode: str
    ) -> tuple[float, bool]:
        """
        TYPE S3: PLATEAU (intervalle toxique).
        
        Args:
            value: Valeur métrique
            rule: {interval_toxic: [min, max]}
            mode: Ignoré (toujours soft)
        
        Returns:
            (score, flag)
        """
        interval = rule['interval_toxic']
        interval_low, interval_high = interval[0], interval[1]
        
        if interval_low <= value <= interval_high:
            score = 1.0
            flag = True
        else:
            # Distance à l'intervalle
            if value < interval_low:
                dist = interval_low - value
            else:
                dist = value - interval_high
            
            # Score décroit avec distance (arbitraire: décroit linéairement)
            # Normaliser par largeur intervalle
            interval_width = interval_high - interval_low
            score = max(0, 1 - dist / max(interval_width, 0.1))
            flag = False
        
        return score, flag
    
    def _score_s4_instability(
        self,
        value: float,
        rule: Dict[str, Any],
        context: Dict[str, Any]
    ) -> tuple[float, bool]:
        """
        TYPE S4: INSTABILITY (variation différentielle).
        
        Args:
            value: Valeur métrique actuelle
            rule: {delta_critical}
            context: {delta?, value_prev?}
        
        Returns:
            (score, flag)
        """
        delta_critical = rule['delta_critical']
        
        # Récupérer delta du contexte
        if context and 'delta' in context:
            delta = abs(context['delta'])
        elif context and 'value_prev' in context:
            delta = abs(value - context['value_prev'])
        else:
            # Pas de contexte → assume delta=0 (pas d'instabilité)
            delta = 0.0
        
        # Normaliser delta par critical
        if delta <= 0:
            score = 0.0
        elif delta >= delta_critical:
            score = 1.0
        else:
            score = delta / delta_critical
        
        flag = delta > delta_critical
        
        return score, flag
    
    def _score_mapping(
        self,
        value: str,
        rule: Dict[str, Any]
    ) -> tuple[float, bool]:
        """
        TYPE MAPPING: Mapping catégoriel.
        
        Args:
            value: Valeur catégorielle (ex: "explosive")
            rule: {mapping: {cat: score, ...}}
        
        Returns:
            (score, flag)
        """
        mapping = rule['mapping']
        
        # Score depuis mapping (default 0.5 si non mappé)
        score = mapping.get(value, 0.5)
        
        # Flag si score >= 0.8 (seuil arbitraire critique)
        flag = score >= 0.8
        
        return score, flag
    
    def aggregate_test_score(
        self,
        metric_scores: Dict[str, Dict[str, Any]],
        aggregation_mode: str = "max"
    ) -> Dict[str, Any]:
        """
        Agrège scores métriques en score test.
        
        Args:
            metric_scores: {metric_key: score_dict}
            aggregation_mode: "max" | "weighted_mean" | "weighted_max"
        
        Returns:
            {
                'test_score': float [0,1],
                'pathology_flags': list[str],
                'critical_metrics': list[str],
                'aggregation_mode': str,
                'metric_scores': dict
            }
        
        Raises:
            ValueError: Si aggregation_mode inconnu
        """
        if not metric_scores:
            return {
                'test_score': 0.0,
                'pathology_flags': [],
                'critical_metrics': [],
                'aggregation_mode': aggregation_mode,
                'metric_scores': {}
            }
        
        # Agrégation selon mode
        if aggregation_mode == "max":
            # R0 default: une pathologie suffit
            test_score = max(m['score'] for m in metric_scores.values())
        
        elif aggregation_mode == "weighted_mean":
            # Future R1: pondération nuancée
            weighted_sum = sum(m['score'] * m['weight'] for m in metric_scores.values())
            weight_sum = sum(m['weight'] for m in metric_scores.values())
            test_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        
        elif aggregation_mode == "weighted_max":
            # Hybride: max des scores pondérés normalisés
            # Normaliser weights pour ne pas dépasser 1
            max_weight = max(m['weight'] for m in metric_scores.values())
            weighted_scores = [
                m['score'] * (m['weight'] / max_weight) 
                for m in metric_scores.values()
            ]
            test_score = max(weighted_scores) if weighted_scores else 0.0
        
        else:
            raise ValueError(
                f"Unknown aggregation_mode: {aggregation_mode}. "
                f"Expected: 'max', 'weighted_mean', 'weighted_max'"
            )
        
        # Flags et métriques critiques
        pathology_flags = [
            key for key, m in metric_scores.items() if m['flag']
        ]
        
        critical_metrics = [
            key for key, m in metric_scores.items() if m['score'] >= 0.8
        ]
        
        return {
            'test_score': float(test_score),
            'pathology_flags': pathology_flags,
            'critical_metrics': critical_metrics,
            'aggregation_mode': aggregation_mode,
            'metric_scores': metric_scores
        }
    
    def validate_scoring_rule(
        self,
        scoring_rule: Dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Valide une règle scoring.
        
        Args:
            scoring_rule: Config YAML règle
        
        Returns:
            (is_valid, error_message)
        """
        if 'pathology_type' not in scoring_rule:
            return False, "Missing 'pathology_type'"
        
        pathology_type = scoring_rule['pathology_type']
        
        if pathology_type not in self.PATHOLOGY_TYPES:
            return False, f"Unknown pathology_type: {pathology_type}"
        
        # Validation spécifique par type
        if pathology_type == "S1_COLLAPSE":
            if 'threshold_low' not in scoring_rule:
                return False, "S1_COLLAPSE requires 'threshold_low'"
        
        elif pathology_type == "S2_EXPLOSION":
            if 'threshold_high' not in scoring_rule:
                return False, "S2_EXPLOSION requires 'threshold_high'"
        
        elif pathology_type == "S3_PLATEAU":
            if 'interval_toxic' not in scoring_rule:
                return False, "S3_PLATEAU requires 'interval_toxic'"
            if len(scoring_rule['interval_toxic']) != 2:
                return False, "interval_toxic must be [min, max]"
        
        elif pathology_type == "S4_INSTABILITY":
            if 'delta_critical' not in scoring_rule:
                return False, "S4_INSTABILITY requires 'delta_critical'"
        
        elif pathology_type == "MAPPING":
            if 'mapping' not in scoring_rule:
                return False, "MAPPING requires 'mapping' dict"
            if not isinstance(scoring_rule['mapping'], dict):
                return False, "mapping must be dict"
        
        return True, ""


# Instance singleton
_engine = None

def get_scoring_engine() -> ScoringEngine:
    """Récupère instance singleton ScoringEngine."""
    global _engine
    if _engine is None:
        _engine = ScoringEngine()
    return _engine