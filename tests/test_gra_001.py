# tests/test_graph_001.py
"""
Propriétés graphe (interprétation adjacence).

Objectif :
- Analyser structure connectivité
- Détecter motifs réseau

Métriques :
- density : Densité connexions
- clustering_local : Transitivité locale
- degree_variance : Hétérogénéité degrés

Algorithmes utilisés :
- graph.density : Ratio arêtes/max
- graph.clustering_local : Clustering moyen
- graph.degree_variance : Variance degrés

Exclusions :
- Chemins plus courts : Trop coûteux pour R0
- Communautés : Nécessite algorithmes dédiés
"""

import numpy as np

TEST_ID = "GRA-001"
TEST_CATEGORY = "GRAPH"
TEST_VERSION = "5.5" 
TEST_WEIGHT = 1.0   
TEST_PHASE = None  # Applicable toutes phases
  
APPLICABILITY_SPEC = {
    "requires_rank": 2,
    "requires_square": True,
    "allowed_d_types": ["ALL"],
    "minimum_dimension": 5,
    "requires_even_dimension": False,
}

COMPUTATION_SPECS = {
    'density': {
        'registry_key': 'graph.density',
        'default_params': {
            'threshold': 0.1,  # Seuil arêtes
        },
        'post_process': 'round_4',
    },
    
    'clustering_local': {
        'registry_key': 'graph.clustering_local',
        'default_params': {
            'threshold': 0.1,
        },
        'post_process': 'round_4',
    },
    
    'degree_variance': {
        'registry_key': 'graph.degree_variance',
        'default_params': {
            'threshold': 0.1,
            'normalize': True,
        },
        'post_process': 'round_4',
    }
}
