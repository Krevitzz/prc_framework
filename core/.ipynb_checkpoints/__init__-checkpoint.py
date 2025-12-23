# core/__init__.py

"""
Package core - Moteur d'exécution PRC universel

Ce package contient les composants fondamentaux du système PRC :
- kernel.py : Moteur d'exécution itératif aveugle
- state_preparation.py : Composition aveugle d'états

PRINCIPE FONDAMENTAL :
Chaque module est AVEUGLE : il ne connaît ni la dimension, 
ni la structure, ni l'interprétation des données qu'il manipule.
"""

from .kernel import run_kernel
from .state_preparation import prepare_state

__all__ = [
    'run_kernel',
    'prepare_state',
]
