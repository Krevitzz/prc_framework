# tests/utilities/registries/__init__.py
"""
Registres de fonctions mathématiques réutilisables.

Architecture Charter 5.4 - Section 2
"""

from .base_registry import BaseRegistry, register_function
from .registry_manager import RegistryManager
from .post_processors import POST_PROCESSORS, get_post_processor, add_post_processor

# Registres disponibles (chargés dynamiquement par RegistryManager)
# - algebra_registry : Normes, asymétrie, traces
# - spectral_registry : FFT, wavelets (à créer)
# - spatial_registry : Convolutions, corrélations (à créer)
# - topological_registry : Invariants (à créer)
# - statistical_registry : Distributions, entropie (à créer)
# - graph_registry : Métriques graphes (à créer)
# - pattern_registry : Reconnaissance patterns (à créer)

__all__ = [
    'BaseRegistry',
    'register_function',
    'RegistryManager',
    'POST_PROCESSORS',
    'get_post_processor',
    'add_post_processor',
]