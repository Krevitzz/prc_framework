# TESTS CATALOG

> Modules d'observation pure du système PRC  
> Ne retournent JAMAIS de verdict (PASS/FAIL)  

**RAPPEL** : Tests retournent observations (dict), pas verdicts (PASS/FAIL)

---

## TESTS UNIVERSELS (UNIV-*)

### UNIV-001 - Évolution Norme Frobenius
**Fichier** : `test_uni_001.py`  
**Objectif** : Mesurer stabilité globale tenseur sous action Γ  
**Catégorie** : UNIV  
**Version** : 5.5  
**Poids** : 1.0

**Applicabilité** :
- Rang : Tous
- Carré : Non requis
- Types D : Tous
- Dimension minimale : Aucune

**Métriques** :
- `frobenius_norm` : Discrimine explosions/effondrements/stabilité

**Algorithmes** :
- `algebra.matrix_norm` (frobenius) : Standard, robuste, O(n²)

**Exclusions** :
- Norme spectrale : Trop coûteuse (SVD), peu discriminante ici
- Norme nucléaire : Redondante avec Frobenius pour explosions

---

### UNIV-002 - Évolution Trace Normalisée
**Fichier** : `test_uni_002.py`  
**Objectif** : Mesurer stabilité trace (somme diagonale), détecter pathologies  
**Catégorie** : UNIV  
**Version** : 5.5  
**Poids** : 1.0

**Applicabilité** :
- Rang : 2
- Carré : Oui (requis)
- Types D : Tous
- Dimension minimale : Aucune

**Métriques** :
- `trace_normalized` : Trace / dimension (moyenne diagonale)
- `trace_absolute` : Trace brute

**Algorithmes** :
- `algebra.trace_value` : Calcul standard trace

**Exclusions** :
- Déterminant : Trop sensible petites variations
- Valeurs propres individuelles : Redondant avec tests spectraux

---

## TESTS SYMÉTRIE (SYM-*)

### SYM-001 - Évolution Asymétrie Matrices
**Fichier** : `test_sym_001.py`  
**Objectif** : Mesurer création/destruction/préservation symétrie  
**Catégorie** : SYM  
**Version** : 5.5  
**Poids** : 1.0

**Applicabilité** :
- Rang : 2
- Carré : Oui (requis)
- Types D : SYM, ASY uniquement
- Dimension minimale : Aucune

**Métriques** :
- `asymmetry_norm` : Discriminant principal (global), norme ||A - A^T||
- `asymmetry_norm_normalized` : Comparable entre tailles différentes

**Algorithmes** :
- `algebra.matrix_asymmetry` (frobenius) : Paramétrable

**Exclusions** :
- Trace asymétrie : Masque patterns spatiaux (trop agrégée)
- Max asymétrie : Sensible outliers, peu robuste

---

## TESTS SPECTRAUX (SPE-*)

### SPE-001 - Évolution Spectre Valeurs Propres
**Fichier** : `test_spe_001.py`  
**Objectif** : Observer évolution distribution spectrale, détecter concentration/dispersion  
**Catégorie** : SPECTRAL  
**Version** : 5.5  
**Poids** : 1.0

**Applicabilité** :
- Rang : 2
- Carré : Oui (requis)
- Types D : Tous
- Dimension minimale : Aucune

**Métriques** :
- `eigenvalue_max` : Plus grande valeur propre (dominance)
- `spectral_gap` : Écart λ₁ - λ₂ (séparation)

**Algorithmes** :
- `spectral.eigenvalue_max` (absolute=True) : Calcul λ_max
- `spectral.spectral_gap` (normalize=True) : Écart spectral relatif

**Exclusions** :
- Entropie spectrale : Redondant avec tests statistiques
- Valeurs propres individuelles : Trop détaillé pour R0

---

### SPE-002 - Rayon Spectral
**Fichier** : `test_spe_002.py`  
**Objectif** : Mesurer rayon spectral (stabilité itérations)  
**Catégorie** : SPECTRAL  
**Version** : 5.5  
**Poids** : 1.0

**Applicabilité** :
- Rang : 2
- Carré : Oui (requis)
- Types D : Tous
- Dimension minimale : 3 (gap nécessite ≥3 valeurs propres)

**Métriques** :
- `spectral_radius` : Rayon spectral

**Algorithmes** :
- `spectral.spectral_radius` : Calcul standard

**Exclusions** :
- (Voir SPE-001)

---

## TESTS PATTERN (PAT-*)

### PAT-001 - Diversité et Concentration Distribution
**Fichier** : `test_pat_001.py`  
**Objectif** : Mesurer dispersion valeurs, détecter émergence structures concentrées/uniformes  
**Catégorie** : PAT  
**Version** : 5.5  
**Poids** : 1.0

**Applicabilité** :
- Rang : Tous
- Carré : Non requis
- Types D : Tous
- Dimension minimale : Aucune

**Métriques** :
- `diversity_simpson` : Indice diversité Simpson
- `concentration_top10` : Concentration énergie dans top 10%
- `uniformity` : Proximité distribution uniforme

**Algorithmes** :
- `pattern.diversity` (bins=50) : Indice Simpson
- `pattern.concentration_ratio` (top_percent=0.1) : Ratio concentration
- `pattern.uniformity` (bins=50) : Distance à uniforme

**Exclusions** :
- Entropie Shannon : Redondant avec diversity
- Coefficient Gini : Approximé par concentration_ratio

---

## TESTS SPATIAUX (SPA-*)

### SPA-001 - Rugosité et Lissage Spatial
**Fichier** : `test_spa_001.py`  
**Objectif** : Mesurer complexité structure spatiale, détecter transitions rugosité/lissage  
**Catégorie** : SPATIAL  
**Version** : 5.5  
**Poids** : 1.0

**Applicabilité** :
- Rang : 2
- Carré : Non requis
- Types D : Tous
- Dimension minimale : 5 (gradients nécessitent espace)

**Métriques** :
- `gradient_magnitude` : Amplitude variations spatiales
- `laplacian_energy` : Rugosité (courbure locale)
- `smoothness` : Inverse rugosité normalisé

**Algorithmes** :
- `spatial.gradient_magnitude` (normalize=True) : Norme gradient moyen
- `spatial.laplacian_energy` (normalize=True) : Énergie laplacien
- `spatial.smoothness` : Score lissage

**Exclusions** :
- Variance locale : Corrélée avec gradient
- Détection contours : Trop spécifique pour R0

---

## TESTS GRAPHE (GRA-*)

### GRA-001 - Propriétés Graphe (Interprétation Adjacence)
**Fichier** : `test_gra_001.py`  
**Objectif** : Analyser structure connectivité, détecter motifs réseau  
**Catégorie** : GRAPH  
**Version** : 5.5  
**Poids** : 1.0

**Applicabilité** :
- Rang : 2
- Carré : Oui (requis)
- Types D : Tous
- Dimension minimale : 5

**Métriques** :
- `density` : Densité connexions
- `clustering_local` : Transitivité locale
- `degree_variance` : Hétérogénéité degrés

**Algorithmes** :
- `graph.density` (threshold=0.1) : Ratio arêtes/max
- `graph.clustering_local` (threshold=0.1) : Clustering moyen
- `graph.degree_variance` (threshold=0.1, normalize=True) : Variance degrés

**Exclusions** :
- Chemins plus courts : Trop coûteux pour R0
- Communautés : Nécessite algorithmes dédiés

---

## TESTS TOPOLOGIQUES (TOP-*)

### TOP-001 - Invariants Topologiques Simplifiés
**Fichier** : `test_top_001.py`  
**Objectif** : Observer changements topologiques, détecter création/destruction structures  
**Catégorie** : TOPOLOGICAL  
**Version** : 5.5  
**Poids** : 1.0

**Applicabilité** :
- Rang : 2
- Carré : Non requis
- Types D : Tous
- Dimension minimale : 10 (topologie nécessite espace)

**Métriques** :
- `connected_components` : Fragmentation
- `holes_count` : Nombre trous
- `euler_characteristic` : Invariant χ

**Algorithmes** :
- `topological.connected_components` (threshold=0.0, connectivity=1) : Comptage composantes
- `topological.holes_count` (threshold=0.0, min_hole_size=4) : Détection trous
- `topological.euler_characteristic` (threshold=0.0) : Calcul χ

**Exclusions** :
- Homologie persistante : Hors scope R0
- Betti numbers : Nécessite bibliothèques spécialisées

---

## STRUCTURE MODULE TEST

**Format standard** :
```python
"""
[Description objectif test]

Objectif :
- [Objectif 1]
- [Objectif 2]

Métriques :
- metric_1 : Description
- metric_2 : Description

Algorithmes utilisés :
- registry.key : Description

Exclusions :
- Alternative 1 : Raison exclusion
- Alternative 2 : Raison exclusion
"""

import numpy as np

TEST_ID = "CAT-NNN"
TEST_CATEGORY = "CATEGORY"
TEST_VERSION = "5.5"
TEST_WEIGHT = 1.0  # Non exploité R0, réservé R1+

APPLICABILITY_SPEC = {
    "requires_rank": None|2|3,
    "requires_square": True|False,
    "allowed_d_types": ["ALL"]|["SYM", "ASY"]|...,
    "minimum_dimension": None|int,
    "requires_even_dimension": False,
}

COMPUTATION_SPECS = {
    'metric_name': {
        'registry_key': 'category.function',
        'default_params': {
            'param1': value1,
            'param2': value2,
        },
        'post_process': 'round_4'|'round_6'|'abs'|None,
    },
    # ... autres métriques
}
```

---

## APPLICABILITY_SPEC

**Champs obligatoires** :
- `requires_rank` : None (tous) | 2 | 3
- `requires_square` : True | False
- `allowed_d_types` : ["ALL"] | ["SYM", "ASY"] | ...
- `minimum_dimension` : None | int
- `requires_even_dimension` : True | False

**Validation** : Effectuée par `utilities/applicability.py`

---

## COMPUTATION_SPECS

**Champs obligatoires par métrique** :
- `registry_key` : Référence fonction registre (ex: "algebra.matrix_norm")
- `default_params` : Paramètres par défaut (dict)
- `post_process` : Transformation post-calcul (optionnel)

**Post-processeurs disponibles** :
- `'round_4'` : Arrondi 4 décimales
- `'round_6'` : Arrondi 6 décimales
- `'round_2'` : Arrondi 2 décimales
- `'abs'` : Valeur absolue
- `None` : Aucune transformation

**Validation** : TOUTE métrique DOIT être analysable par patterns (R6-C)

---

## TEST_WEIGHT

**Statut R0** : Déclaré obligatoirement, **non exploité**

**Valeur par défaut** : 1.0

**Usage futur (R1+)** :
- Pondération patterns cross-tests
- Méta-analyses multi-tests

**Interdictions R0** :
- ❌ Pas de filtrage exécution
- ❌ Pas d'interprétation "qualité"
- ❌ Aucune logique conditionnelle basée sur weight

---

## CATÉGORIES TESTS

**UNIV** : Universels (applicables tous contextes)  
**SYM** : Symétrie  
**SPE** / SPECTRAL : Analyse spectrale  
**PAT** : Patterns distribution  
**SPA** / SPATIAL : Structure spatiale  
**GRA** / GRAPH : Propriétés graphe  
**TOP** / TOPOLOGICAL : Invariants topologiques

---

## RÈGLES CONCEPTION TESTS

**R-TEST-1** : Tests retournent observations (dict), JAMAIS verdicts (PASS/FAIL)

**R-TEST-2** : Toute métrique COMPUTATION_SPECS DOIT :
- Référencer registry_key existant
- Fournir default_params complets
- Être analysable par détection patterns (R6-A)

**R-TEST-3** : Docstring module DOIT contenir :
- Objectif clair (1-2 phrases)
- Liste métriques mesurées (justifiées)
- Algorithmes utilisés (registry_key précis)
- Exclusions (alternatives non retenues + pourquoi)

**R-TEST-4** : APPLICABILITY_SPEC DOIT être exhaustif (tous champs obligatoires)

**R-TEST-5** : Tests NE DOIVENT PAS :
- Hardcoder listes entités (gamma, D, modifier)
- Implémenter calculs inline (utiliser registres)
- Contenir logique verdict/classification
- Brancher selon gamma_id ou d_encoding_id

---

## FORMAT RETOUR TEST

**Structure observation** (générée par test_engine) :
```python
{
    'test_name': 'CAT-NNN',
    'gamma_id': 'GAM-XXX',
    'd_encoding_id': 'YYY-ZZZ',
    'modifier_id': 'MN',
    'seed': int,
    'status': 'SUCCESS'|'ERROR',
    
    # Si SUCCESS
    'statistics': {
        'initial': {'metric1': float, 'metric2': float, ...},
        'final': {...},
        'mean': {...},
        'max': {...},
        'min': {...}
    },
    'evolution': {
        'slope': {'metric1': float, ...},
        'relative_change': {...}
    },
    'timeline': {
        'events': [...],
        'compact': "string_descriptor"
    },
    
    # Si ERROR
    'error': {
        'type': 'ApplicabilityError'|'ComputationError',
        'message': str
    }
}
```

---

## DÉPENDANCES

**Autorisées** :
- NumPy
- `utilities/registries/*` (fonctions calcul)
- `utilities/applicability.py` (validation)

**Interdites** :
- core/ (tests observent, ne modifient pas)
- operators/, D_encodings/, modifiers/ (séparation stricte)
- Tout module HUB/UTIL (tests sont niveau base)

---

## EXTENSIONS FUTURES

**Checklist ajout nouveau test** :
- [ ] Définir objectif clair (non redondant tests existants)
- [ ] Identifier métriques discriminantes (justifiées)
- [ ] Vérifier registry_keys disponibles (ou créer fonctions registre)
- [ ] Documenter exclusions (alternatives considérées)
- [ ] Spécifier applicabilité complète (APPLICABILITY_SPEC)
- [ ] Fixer TEST_WEIGHT = 1.0 (défaut R0)
- [ ] Implémenter COMPUTATION_SPECS (toutes métriques analysables)
- [ ] Ajouter ID séquentiel (CAT-NNN)
- [ ] Mettre à jour ce catalogue

**Exemples extensions acceptables** :
- ✅ Tests rang 3 spécifiques (actuellement peu représentés)
- ✅ Tests corrélations temporelles (ordre supérieur)
- ✅ Tests conservation quantités physiques

**Exemples extensions REFUSÉES** :
- ❌ Tests retournant verdicts binaires (violé observation pure)
- ❌ Tests avec logique gamma-spécifique (violé aveuglement)
- ❌ Tests sans registry_key (calculs inline interdits)

---

**FIN TESTS CATALOG**