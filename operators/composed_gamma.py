"""
Composed Gamma Factory - R1 Composition Support.

RESPONSABILITÉ:
- Factory create_composed_gamma() compatible discovery pattern
- Résolution instances gammas atomiques depuis active_entities
- Délégation ComposedGamma pour application séquentielle

CONFORMITÉ Charter 6.1:
- Factory pattern identique gammas atomiques
- PHASE/METADATA module-level (requis discovery)
- Signature factory extensible avec kwargs

Version: 1.0 (R1.1)
"""

from typing import List, Dict, Any, Optional

# =============================================================================
# METADATA MODULE (requis discovery)
# =============================================================================

PHASE = "R1"

METADATA = {
    'id': 'COMPOSED',
    'type': 'composed',
    'description': 'Gamma composition séquentielle',
    'family': 'composite',
    'd_applicability': []
}


# =============================================================================
# FACTORY (pattern standard)
# =============================================================================

def create_composed_gamma(
    sequence_gamma_ids: List[str],
    weights: Optional[List[float]] = None,
    seed: int = 42,
    active_gammas: List[Dict[str, Any]] = None
) -> 'ComposedGamma':
    """
    Factory composition gamma.
    
    Args:
        sequence_gamma_ids: IDs gammas atomiques ['GAM-001', 'GAM-002']
        weights: Poids application (None = uniforme)
        seed: Seed reproductibilité
        active_gammas: Entités gammas atomiques (depuis batch_runner)
    
    Returns:
        Instance ComposedGamma callable
    
    Raises:
        ValueError: Si gamma_id non trouvé
        TypeError: Si active_gammas None/vide
    """
    # Import classe composition (Option B workaround)
    from operators.composed_gamma_inline import ComposedGamma
    
    # Validation active_gammas
    if active_gammas is None or len(active_gammas) == 0:
        raise TypeError(
            "create_composed_gamma() requires active_gammas\n"
            "→ Called by batch_runner with atomic gammas"
        )
    
    # Résoudre instances gammas atomiques
    gamma_instances = []
    
    for gamma_id in sequence_gamma_ids:
        # Trouver entity correspondante
        gamma_entity = next(
            (g for g in active_gammas if g['id'] == gamma_id),
            None
        )
        
        if gamma_entity is None:
            available_ids = [g['id'] for g in active_gammas]
            raise ValueError(
                f"Gamma '{gamma_id}' not found in active_gammas\n"
                f"Available: {available_ids}"
            )
        
        # Créer instance via factory atomique
        gamma_module = gamma_entity['module']
        factory_name = gamma_entity['function_name']
        factory = getattr(gamma_module, factory_name)
        
        gamma_instance = factory(seed=seed)
        gamma_instances.append(gamma_instance)
    
    # Créer composition
    composed = ComposedGamma(
        sequence_gammas=gamma_instances,
        weights=weights,
        seed=seed
    )
    
    return composed