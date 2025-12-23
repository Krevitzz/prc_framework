# encodings/__init__.py

"""
Package encodings - Encodages spécifiques pour les tenseurs

Ce package contient des créateurs d'états D^(base) pour différents types de tenseurs.
Chaque module contient des présuppositions explicites sur la structure des tenseurs.

PRINCIPE FONDAMENTAL :
Ces présuppositions sont HORS du core - le kernel et state_preparation sont aveugles
à ces propriétés spécifiques.
"""

from .rank2_symmetric import create_identity, create_uniform, create_random

__all__ = [
    'create_identity',
    'create_uniform',
    'create_random',
]

__version__ = '1.0.0'
__author__ = 'PRC Encodings Team'
__description__ = 'Créateurs d\'états pour tenseurs rang 2 symétriques'

# Documentation des présuppositions spécifiques
__rank2_symmetric_assumptions__ = {
    "rank": 2,
    "symmetry": "C[i,j] = C[j,i]",
    "diagonal": "C[i,i] = 1",
    "bounds": "[-1, 1]"
}

# Note d'avertissement
__warning__ = "Ces créateurs imposent des présuppositions spécifiques - utiliser uniquement avec des tenseurs compatibles"