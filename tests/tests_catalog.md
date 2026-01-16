# tests_catalog.md

> Catalogue fonctionnel des tests d'observation  
> Responsabilité : Mesure propriétés D sans jugement normatif  
> Version : 6.0  
> Dernière mise à jour : 2025-01-15

---

## VUE D'ENSEMBLE

Le module `tests/` contient les **tests d'observation** qui mesurent l'évolution de D sous action γ.

**IDs catalogués** : 9 tests (UNIV-001, UNIV-002, SYM-001, SPE-001, SPE-002, PAT-001, SPA-001, GRA-001, TOP-001)

**Catégories** :
- **UNIV** : Universels (applicables tous contextes)
- **SYM** : Symétrie (préservation/création/destruction)
- **SPE** : Spectral (valeurs propres, stabilité)
- **PAT** : Pattern (diversité, concentration, uniformité)
- **SPA** : Spatial (rugosité, gradient, lissage)
- **GRA** : Graphe (connectivité, clustering)
- **TOP** : Topologique (composantes, trous, χ)

**Principe fondamental** :
- Tests **OBSERVENT** (pas de verdict PASS/FAIL)
- Format retour : dict v2 standardisé
- Patterns détectés ultérieurement (verdict_engine)
- Métriques via registries (algebra, spectral, pattern, etc.)

---

## SECTION 1 : CATÉGORIE UNIV (Universels)

### 1.1 UNIV-001 : Évolution norme Frobenius

**Fichier** : `test_uni_001.py`

**Objectif** :
- Mesurer stabilité globale tenseur sous action γ
- Discriminer explosions/effondrements/stabilité

**Métriques** :

| Métrique | Registry Key | Params défaut | Post-process | Pertinence |
|----------|--------------|---------------|--------------|------------|
| `frobenius_norm` | `algebra.matrix_norm` | `norm_type='frobenius'` | `round_4` | Détection explosions/collapse |

**Applicabilité** :
```python
APPLICABILITY_SPEC = {
    "requires_rank": None,           # Tout rang
    "requires_square": False,
    "allowed_d_types": ["ALL"],
    "minimum_dimension": None,
    "requires_even_dimension": False,
}
```

**Algorithmes utilisés** :
- `algebra.matrix_norm` : Norme standard Frobenius, robuste, O(N²)

**Exclusions** :
- Norme spectrale : Trop coûteuse, peu discriminante
- Norme nucléaire : Redondante avec Frobenius pour détection explosions

**Cas d'usage** :
- Test stabilité GAM-003, GAM-004, GAM-013
- Détection explosions numériques
- Baseline universelle tous γ

---

### 1.2 UNIV-002 : Évolution trace

**Fichier** : `test_uni_002.py`

**Objectif** :
- Mesurer stabilité trace (somme diagonale)
- Détection comportements pathologiques (explosion trace)

**Métriques** :

| Métrique | Registry Key | Params défaut | Post-process | Pertinence |
|----------|--------------|---------------|--------------|------------|
| `trace_normalized` | `algebra.trace_value` | `normalize=True` | `round_4` | Moyenne diagonale |
| `trace_absolute` | `algebra.trace_value` | `normalize=False` | `round_2` | Trace brute |

**Applicabilité** :
```python
APPLICABILITY_SPEC = {
    "requires_rank": 2,
    "requires_square": True,         # Trace nécessite matrices carrées
    "allowed_d_types": ["ALL"],
    "minimum_dimension": None,
    "requires_even_dimension": False,
}
```

**Algorithmes utilisés** :
- `algebra.trace_value` : Calcul standard trace

**Exclusions** :
- Déterminant : Trop sensible petites variations
- Valeurs propres individuelles : Redondant avec tests spectraux

**Cas d'usage** :
- Complément UNIV-001 (différente projection)
- Test stabilité diagonale

---

## SECTION 2 : CATÉGORIE SYM (Symétrie)

### 2.1 SYM-001 : Évolution asymétrie

**Fichier** : `test_sym_001.py`

**Objectif** :
- Mesurer création/destruction/préservation symétrie
- Discriminant principal (global)

**Métriques** :

| Métrique | Registry Key | Params défaut | Post-process | Pertinence |
|----------|--------------|---------------|--------------|------------|
| `asymmetry_norm` | `algebra.matrix_asymmetry` | `norm_type='frobenius'`, `normalize=False` | `round_6` | Asymétrie brute |
| `asymmetry_norm_normalized` | `algebra.matrix_asymmetry` | `norm_type='frobenius'`, `normalize=True` | `round_6` | Comparable entre tailles |

**Applicabilité** :
```python
APPLICABILITY_SPEC = {
    "requires_rank": 2,
    "requires_square": True,
    "allowed_d_types": ["SYM", "ASY"],  # Pas R3
    "minimum_dimension": None,
    "requires_even_dimension": False,
}
```

**Algorithmes utilisés** :
- `algebra.matrix_asymmetry` : Norme ||A - A^T||, paramétrable

**Exclusions** :
- Trace asymétrie : Masque patterns spatiaux (trop agrégée)
- Max asymétrie : Sensible outliers, peu robuste

**Cas d'usage** :
- Test préservation symétrie (GAM-001 sur SYM-*)
- Test création symétrie (GAM-012 sur ASY-*)
- Test robustesse bruit asymétrique (M1, M2 sur SYM-*)

---

## SECTION 3 : CATÉGORIE SPE (Spectral)

### 3.1 SPE-001 : Valeurs propres dominantes

**Fichier** : `test_spe_001.py`

**Objectif** :
- Observer évolution distribution spectrale
- Détecter concentration/dispersion énergie

**Métriques** :

| Métrique | Registry Key | Params défaut | Post-process | Pertinence |
|----------|--------------|---------------|--------------|------------|
| `eigenvalue_max` | `spectral.eigenvalue_max` | `absolute=True` | `round_4` | Plus grande valeur propre (dominance) |
| `spectral_gap` | `spectral.spectral_gap` | `normalize=True` | `round_6` | Écart λ₁ - λ₂ (séparation) |

**Applicabilité** :
```python
APPLICABILITY_SPEC = {
    "requires_rank": 2,
    "requires_square": True,
    "allowed_d_types": ["ALL"],
    "minimum_dimension": None,
    "requires_even_dimension": False,
}
```

**Algorithmes utilisés** :
- `spectral.eigenvalue_max` : Calcul λ_max
- `spectral.spectral_gap` : Écart spectral

**Exclusions** :
- Entropie spectrale : Redondant avec tests statistiques
- Valeurs propres individuelles : Trop détaillé pour R0

**Cas d'usage** :
- Test dominance spectrale
- Détection concentration énergie

---

### 3.2 SPE-002 : Rayon spectral

**Fichier** : `test_spe_002.py`

**Objectif** :
- Mesurer stabilité itérations (rayon spectral)

**Métriques** :

| Métrique | Registry Key | Params défaut | Post-process | Pertinence |
|----------|--------------|---------------|--------------|------------|
| `spectral_radius` | `spectral.spectral_radius` | - | `round_4` | Rayon spectral (stabilité itérations) |

**Applicabilité** :
```python
APPLICABILITY_SPEC = {
    "requires_rank": 2,
    "requires_square": True,
    "allowed_d_types": ["ALL"],
    "minimum_dimension": 3,          # Gap nécessite ≥3 valeurs propres
    "requires_even_dimension": False,
}
```

**Algorithmes utilisés** :
- `spectral.spectral_radius` : Rayon spectral

**Cas d'usage** :
- Test stabilité linéaire
- Prédiction convergence/divergence

---

## SECTION 4 : CATÉGORIE PAT (Pattern)

### 4.1 PAT-001 : Diversité et concentration

**Fichier** : `test_pat_001.py`

**Objectif** :
- Mesurer dispersion valeurs
- Détecter émergence structures concentrées ou uniformes

**Métriques** :

| Métrique | Registry Key | Params défaut | Post-process | Pertinence |
|----------|--------------|---------------|--------------|------------|
| `diversity_simpson` | `pattern.diversity` | `bins=50` | `round_4` | Indice diversité Simpson |
| `concentration_top10` | `pattern.concentration_ratio` | `top_percent=0.1` | `round_4` | Concentration énergie dans top 10% |
| `uniformity` | `pattern.uniformity` | `bins=50` | `round_4` | Proximité distribution uniforme |

**Applicabilité** :
```python
APPLICABILITY_SPEC = {
    "requires_rank": None,           # Tout rang
    "requires_square": False,
    "allowed_d_types": ["ALL"],
    "minimum_dimension": None,
    "requires_even_dimension": False,
}
```

**Algorithmes utilisés** :
- `pattern.diversity` : Indice Simpson
- `pattern.concentration_ratio` : Ratio concentration
- `pattern.uniformity` : Distance à uniforme

**Exclusions** :
- Entropie Shannon : Redondant avec diversity
- Coefficient Gini : Approximé par concentration_ratio

**Cas d'usage** :
- Test homogénéisation (GAM-002, GAM-007)
- Test émergence structures (GAM-013)
- Détection collapse/explosion patterns

---

## SECTION 5 : CATÉGORIE SPA (Spatial)

### 5.1 SPA-001 : Rugosité et lissage

**Fichier** : `test_spa_001.py`

**Objectif** :
- Mesurer complexité structure spatiale
- Détecter transitions rugosité/lissage

**Métriques** :

| Métrique | Registry Key | Params défaut | Post-process | Pertinence |
|----------|--------------|---------------|--------------|------------|
| `gradient_magnitude` | `spatial.gradient_magnitude` | `normalize=True` | `round_6` | Amplitude variations spatiales |
| `laplacian_energy` | `spatial.laplacian_energy` | `normalize=True` | `round_6` | Rugosité (courbure locale) |
| `smoothness` | `spatial.smoothness` | - | `round_4` | Inverse rugosité normalisé |

**Applicabilité** :
```python
APPLICABILITY_SPEC = {
    "requires_rank": 2,
    "requires_square": False,
    "allowed_d_types": ["ALL"],
    "minimum_dimension": 5,          # Gradients nécessitent espace
    "requires_even_dimension": False,
}
```

**Algorithmes utilisés** :
- `spatial.gradient_magnitude` : Norme gradient moyen
- `spatial.laplacian_energy` : Énergie laplacien
- `spatial.smoothness` : Score lissage

**Exclusions** :
- Variance locale : Corrélée avec gradient
- Détection contours : Trop spécifique pour R0

**Cas d'usage** :
- Test diffusion (GAM-002 → smoothness ↑)
- Test rugosité (GAM-013 → gradient ↑)

---

## SECTION 6 : CATÉGORIE GRA (Graphe)

### 6.1 GRA-001 : Propriétés graphe

**Fichier** : `test_gra_001.py`

**Objectif** :
- Analyser structure connectivité (interprétation adjacence)
- Détecter motifs réseau

**Métriques** :

| Métrique | Registry Key | Params défaut | Post-process | Pertinence |
|----------|--------------|---------------|--------------|------------|
| `density` | `graph.density` | `threshold=0.1` | `round_4` | Densité connexions |
| `clustering_local` | `graph.clustering_local` | `threshold=0.1` | `round_4` | Transitivité locale |
| `degree_variance` | `graph.degree_variance` | `threshold=0.1`, `normalize=True` | `round_4` | Hétérogénéité degrés |

**Applicabilité** :
```python
APPLICABILITY_SPEC = {
    "requires_rank": 2,
    "requires_square": True,
    "allowed_d_types": ["ALL"],
    "minimum_dimension": 5,
    "requires_even_dimension": False,
}
```

**Algorithmes utilisés** :
- `graph.density` : Ratio arêtes/max
- `graph.clustering_local` : Clustering moyen
- `graph.degree_variance` : Variance degrés

**Exclusions** :
- Chemins plus courts : Trop coûteux pour R0
- Communautés : Nécessite algorithmes dédiés

**Cas d'usage** :
- Test structures réseau
- Interprétation D comme matrice adjacence

---

## SECTION 7 : CATÉGORIE TOP (Topologique)

### 7.1 TOP-001 : Invariants topologiques

**Fichier** : `test_top_001.py`

**Objectif** :
- Observer changements topologiques
- Détecter création/destruction structures

**Métriques** :

| Métrique | Registry Key | Params défaut | Post-process | Pertinence |
|----------|--------------|---------------|--------------|------------|
| `connected_components` | `topological.connected_components` | `threshold=0.0`, `connectivity=1` | `round_2` | Fragmentation |
| `holes_count` | `topological.holes_count` | `threshold=0.0`, `min_hole_size=4` | `round_2` | Nombre trous |
| `euler_characteristic` | `topological.euler_characteristic` | `threshold=0.0` | `round_2` | Invariant χ |

**Applicabilité** :
```python
APPLICABILITY_SPEC = {
    "requires_rank": 2,
    "requires_square": False,
    "allowed_d_types": ["ALL"],
    "minimum_dimension": 10,         # Topologie nécessite espace
    "requires_even_dimension": False,
}
```

**Algorithmes utilisés** :
- `topological.connected_components` : Comptage composantes
- `topological.holes_count` : Détection trous
- `topological.euler_characteristic` : Calcul χ

**Exclusions** :
- Homologie persistante : Hors scope R0
- Betti numbers : Nécessite bibliothèques spécialisées

**Cas d'usage** :
- Test fragmentation
- Détection émergence trous topologiques

---

## SECTION 8 : MAPPING IDs ↔ TESTS

### 8.1 Table complète (catalogués)

| ID | Catégorie | Fichier | Métriques | Applicabilité |
|----|-----------|---------|-----------|---------------|
| UNIV-001 | UNIV | test_uni_001.py | frobenius_norm | Tout |
| UNIV-002 | UNIV | test_uni_002.py | trace_normalized, trace_absolute | Rang 2 carrée |
| SYM-001 | SYM | test_sym_001.py | asymmetry_norm, asymmetry_norm_normalized | Rang 2 carrée, SYM/ASY |
| SPE-001 | SPE | test_spe_001.py | eigenvalue_max, spectral_gap | Rang 2 carrée |
| SPE-002 | SPE | test_spe_002.py | spectral_radius | Rang 2 carrée, dim≥3 |
| PAT-001 | PAT | test_pat_001.py | diversity_simpson, concentration_top10, uniformity | Tout |
| SPA-001 | SPA | test_spa_001.py | gradient_magnitude, laplacian_energy, smoothness | Rang 2, dim≥5 |
| GRA-001 | GRA | test_gra_001.py | density, clustering_local, degree_variance | Rang 2 carrée, dim≥5 |
| TOP-001 | TOP | test_top_001.py | connected_components, holes_count, euler_characteristic | Rang 2, dim≥10 |

**Total** : 9 tests, 21 métriques

---

## SECTION 9 : STRUCTURE MODULE TEST

### 9.1 Format canonique

```python
"""
tests/test_category_nnn.py

[Titre descriptif]

Objectif :
- [Description phénomène mesuré]

Métriques :
- [nom_métrique_1] : [Pertinence]
- [nom_métrique_2] : [Pertinence]

Algorithmes utilisés :
- [registry_key_1] : [Justification]

Exclusions :
- [Alternatives non retenues] : [Pourquoi]
"""

import numpy as np

TEST_ID = "CAT-NNN"           # Format CAT-NNN (ex: UNIV-001)
TEST_CATEGORY = "CAT"         # UNIV, SYM, SPE, PAT, SPA, GRA, TOP
TEST_VERSION = "6.0"          # Version charter
TEST_WEIGHT = 1.0             # Importance épistémique (défaut 1.0)

APPLICABILITY_SPEC = {
    "requires_rank": int | None,          # 2, 3, ou None (tout)
    "requires_square": bool,              # Matrice carrée requise
    "allowed_d_types": List[str],         # ["SYM", "ASY", "R3"] ou ["ALL"]
    "minimum_dimension": int | None,      # Dimension minimale
    "requires_even_dimension": bool,      # Dimension paire requise
}

COMPUTATION_SPECS = {
    'nom_metrique_1': {
        'registry_key': 'registre.fonction',  # ex: 'algebra.matrix_norm'
        'default_params': {
            'param1': valeur1,
            'param2': valeur2,
        },
        'post_process': 'round_4',  # identity, round_N, abs, log, etc.
    },
    # 1 à 5 métriques maximum
}
```

### 9.2 Format retour (dict v2)

Voir Charter Section 4 pour structure complète. Résumé :

```python
{
    'run_metadata': {...},           # gamma_id, d_encoding_id, modifier_id, seed
    'test_name': str,
    'test_category': str,
    'test_version': str,
    'params_config_id': str,
    
    'status': 'SUCCESS' | 'ERROR' | 'NOT_APPLICABLE',
    'message': str,
    
    'statistics': {                  # Dict de dicts
        'metric_name': {
            'initial': float, 'final': float, 'min': float, 'max': float,
            'mean': float, 'std': float, 'median': float, 'q1': float, 'q3': float,
            'n_valid': int
        },
    },
    
    'evolution': {                   # Dict de dicts
        'metric_name': {
            'transition': str,       # stable, increasing, explosive, collapsing
            'trend': str,            # monotonic, oscillatory, chaotic
            'slope': float,
            'volatility': float,
            'relative_change': float
        },
    },
    
    'dynamic_events': {              # Dict de dicts
        'metric_name': {
            'deviation_onset': int | None,
            'instability_onset': int | None,
            'oscillatory': bool,
            'saturation': bool,
            'collapse': bool,
            'sequence': List[str],
            'sequence_timing': List[int],
            'sequence_timing_relative': List[float]
        },
    },
    
    'metadata': {...}                # engine_version, execution_time, etc.
}
```

---

## SECTION 10 : GRAPHE DE DÉPENDANCES

### 10.1 Relations inter-modules

```
test_*.py
    ├─ Appelé par : batch_runner.py (mode --test)
    ├─ Reçoit : history (List[np.ndarray]), params_config
    └─ Retourne : dict v2 (observations)
    
test_engine.py (HUB)
    ├─ Orchestre exécution tests
    ├─ Applique APPLICABILITY_SPEC
    ├─ Appelle registries (algebra, spectral, pattern, etc.)
    ├─ Calcule statistics, evolution, dynamic_events
    └─ Retourne dict v2

registries (UTIL)
    ├─ algebra_registry.py
    ├─ spectral_registry.py
    ├─ pattern_registry.py
    ├─ spatial_registry.py
    ├─ graph_registry.py
    └─ topological_registry.py
```

### 10.2 Flux typique exécution

```
1. batch_runner.py (mode --test)
   ↓
2. Charge history depuis db_raw
   ↓
3. Pour chaque test_module :
   ↓
4. test_engine.run_test(history, test_module, params_config)
   ↓
5. Vérification APPLICABILITY_SPEC
   ↓
6. Calcul métriques (via registries)
   ↓
7. Calcul statistics (initial, final, min, max, mean, std, ...)
   ↓
8. Calcul evolution (transition, trend, slope, volatility, ...)
   ↓
9. Détection dynamic_events (deviation_onset, oscillatory, collapse, ...)
   ↓
10. Retour dict v2
   ↓
11. Stockage db_results (TestObservations)
```

---

## SECTION 11 : INVARIANTS CRITIQUES

### 11.1 Tests observent (pas de jugement)

**R11-A** : Tests ne retournent JAMAIS :
- `"PASS"` / `"FAIL"`
- `"bon"` / `"mauvais"`
- Verdicts normatifs

**R11-B** : Status autorisés uniquement :
- `SUCCESS` : Test exécuté normalement
- `ERROR` : Erreur technique (→ ARRÊT BATCH, bug code)
- `NOT_APPLICABLE` : Test invalide pour contexte (non pénalisé)

---

### 11.2 Métriques analysables par patterns

**R11-C** : Toute métrique `COMPUTATION_SPECS` **doit** être analysable par patterns

Interdit :
```python
# ❌ INTERDIT (pas analysable)
COMPUTATION_SPECS = {
    'is_symmetric': {  # ❌ Booléen, pas métrique continue
        ...
    }
}
```

Autorisé :
```python
# ✅ CORRECT (métrique continue)
COMPUTATION_SPECS = {
    'asymmetry_norm': {  # ✅ Float, analysable
        ...
    }
}
```

---

### 11.3 Aucun skip silencieux

**R11-D** : Si métrique non calculable → ERROR fatal

```python
# ❌ INTERDIT
if not applicable:
    return None  # ❌ Skip silencieux

# ✅ CORRECT
if not applicable:
    return {
        'status': 'NOT_APPLICABLE',
        'message': 'Raison précise'
    }
```

---

### 11.4 Pas de paramètres hardcodés

**R11-E** : Tous paramètres dans `default_params`

```python
# ❌ INTERDIT
def compute_metric(state):
    threshold = 0.1  # ❌ Hardcodé
    ...

# ✅ CORRECT
COMPUTATION_SPECS = {
    'metric': {
        'default_params': {
            'threshold': 0.1  # ✅ Configurable
        }
    }
}
```

---

### 11.5 TEST_WEIGHT non exploité R0

**R11-F** : TEST_WEIGHT déclaré obligatoire, usage R1+

```python
TEST_WEIGHT = 1.0  # OBLIGATOIRE (défaut), usage futur
```

**Usage futur anticipé (R1+)** :
- Pondération patterns cross-tests
- Méta-analyses multi-tests

**Interdictions R0** :
- Pas de filtrage exécution
- Pas d'interprétation "qualité"

---

## SECTION 12 : EXTENSIONS FUTURES

### 12.1 Ajout nouveau test (checklist)

Avant d'ajouter un test :

- [ ] ID unique (CAT-NNN)
- [ ] Catégorie identifiée (UNIV, SYM, SPE, PAT, SPA, GRA, TOP, ou nouvelle)
- [ ] Objectif clair (docstring)
- [ ] 1 à 5 métriques (COMPUTATION_SPECS)
- [ ] Métriques continues analysables par patterns
- [ ] Registry keys valides (vérifier registries existants)
- [ ] Paramètres avec défauts raisonnables
- [ ] APPLICABILITY_SPEC complet
- [ ] TEST_VERSION = "6.0"
- [ ] TEST_WEIGHT = 1.0 (défaut)
- [ ] Algorithmes justifiés (docstring)
- [ ] Exclusions documentées (docstring)
- [ ] Ajouté à découverte automatique (naming `test_*.py`)
- [ ] Documenté dans ce catalogue

### 12.2 Extensions INTERDITES

❌ **Retour PASS/FAIL** :
```python
# INTERDIT
return {'status': 'PASS'}  # ❌
return {'status': 'FAIL'}  # ❌
```

❌ **Skip silencieux** :
```python
# INTERDIT
if not applicable:
    return None  # ❌
```

❌ **Hardcoding paramètres** :
```python
# INTERDIT
threshold = 0.1  # ❌ Dans code
```

❌ **Métriques non analysables** :
```python
# INTERDIT
'is_valid': bool  # ❌ Pas métrique continue
```

### 12.3 Extensions AUTORISÉES

✅ **Ajout métriques configurables** :
```python
# OK
COMPUTATION_SPECS = {
    'custom_metric': {
        'registry_key': 'algebra.custom_function',
        'default_params': {
            'param1': 0.5,
            'param2': 10,
        },
        'post_process': 'round_4',
    }
}
```

✅ **Ajout catégories tests** :
```python
# OK (si nouvelle classe phénomènes)
TEST_CATEGORY = "NEW_CAT"  # Ex: ENTROPY, CHAOS, etc.
```

✅ **Ajout contraintes applicabilité** :
```python
# OK
APPLICABILITY_SPEC = {
    "requires_rank": 3,
    "requires_symmetric": True,  # Nouvelle contrainte (si implémentée)
}
```

---

## SECTION 13 : NOTES TECHNIQUES

### 13.1 Coût computationnel tests

**Complexité par catégorie** :

| Catégorie | Complexité typique | Notes |
|-----------|-------------------|-------|
| UNIV | O(N²) | Normes, traces (optimal) |
| SYM | O(N²) | Asymétrie (transposée) |
| SPE | O(N³) | Valeurs propres (coûteux) |
| PAT | O(N² log N) | Histogrammes, tris |
| SPA | O(N²) | Gradients, laplacien |
| GRA | O(N² à N³) | Dépend algorithmes graphe |
| TOP | O(N² à N³) | Détection composantes, trous |

**Recommandations** :
- SPE, GRA, TOP : Limiter dimension (N ≤ 100) si possible
- Tests exécutés ~200 fois (snapshots) → optimiser calculs

### 13.2 Post-processing

**Post-processors disponibles** :
- `identity` : Aucune transformation
- `round_N` : Arrondi N décimales (ex: `round_4`)
- `abs` : Valeur absolue
- `log` : Logarithme (gestion log(0) nécessaire)
- `normalize` : Normalisation [0, 1] (custom)

**Usage** :
```python
'post_process': 'round_4'  # Arrondi 4 décimales
```

### 13.3 Gestion NaN/Inf

**Principe** : Registries doivent gérer cas dégénérés

**Cas typiques** :
- Division par zéro → NaN
- Overflow → Inf
- Valeurs propres matrices singulières → complexes/NaN

**Stratégie** :
- Registries retournent `np.nan` si calcul impossible
- test_engine filtre NaN dans `statistics` (`n_valid` comptabilise)
- Patterns détectent `NUMERIC_INSTABILITY` si fraction NaN élevée

### 13.4 Applicability vs Status

**Distinction critique** :

| Cas | APPLICABILITY_SPEC | Status retour |
|-----|-------------------|---------------|
| Test jamais applicable (rang incorrect) | Filtrage en amont | Test non exécuté |
| Test applicable mais erreur technique | Applicable | `ERROR` (→ ARRÊT BATCH) |
| Test applicable, exécuté normalement | Applicable | `SUCCESS` |
| Test applicable, contexte invalide ce run | Applicable | `NOT_APPLICABLE` |

**Exemple** :
- `requires_square=True` + state non-carrée → Test non exécuté (filtrage)
- Test exécuté, valeurs propres NaN → `ERROR` (bug registry)
- Test exécuté, tous snapshots identiques → `SUCCESS` (observation valide)

## SECTION 14 : TESTS ASSOCIÉS

### 14.1 Tests unitaires modules test

**Emplacement** : `tests/test_test_modules.py` (si existe)

**Scénarios minimaux** :
- Format retour dict v2 valide
- Status autorisés uniquement
- Métriques retournées cohérentes avec COMPUTATION_SPECS
- APPLICABILITY_SPEC respecté
- Pas de crash sur données pathologiques (NaN, Inf, zéros)

**Exemple** :
```python
def test_univ_001_format():
    import tests.test_uni_001 as test_module
    
    # History factice
    history = [np.random.randn(10, 10) for _ in range(200)]
    params_config = {}
    
    # Exécution
    result = test_engine.run_test(history, test_module, params_config)
    
    # Validations
    assert result['status'] in ['SUCCESS', 'ERROR', 'NOT_APPLICABLE']
    assert 'statistics' in result
    assert 'frobenius_norm' in result['statistics']
    assert 'evolution' in result
    assert 'dynamic_events' in result
```

### 14.2 Tests intégration

**Pipeline complet** :
```python
# Création D + gamma
D_base = create_correlation_matrix(50, seed=42)
gamma = create_gamma_hyp_001(beta=2.0)

# Exécution kernel
history = []
for i, state in run_kernel(D_base, gamma, max_iterations=200):
    history.append(state.copy())

# Exécution tests
import tests.test_uni_001 as test_module
result = test_engine.run_test(history, test_module, {})

# Vérifications
assert result['status'] == 'SUCCESS'
assert result['statistics']['frobenius_norm']['initial'] > 0
assert result['statistics']['frobenius_norm']['final'] > 0
```

---

## SECTION 15 : COMPARAISON CATÉGORIES

### 15.1 Matrice complémentarité

| Catégorie | Phénomène mesuré | Complémentaire à |
|-----------|------------------|------------------|
| UNIV | Stabilité globale | Toutes (baseline) |
| SYM | Préservation structure | SPE, PAT |
| SPE | Distribution énergie | UNIV, PAT |
| PAT | Dispersion valeurs | SPE, SPA |
| SPA | Complexité spatiale | PAT, GRA |
| GRA | Connectivité réseau | SPA, TOP |
| TOP | Invariants topologiques | GRA, SPA |

### 15.2 Cas d'usage recommandés

**UNIV (Universels)** :
- Baseline tous γ
- Détection explosions/collapse
- Validation stabilité numérique

**SYM (Symétrie)** :
- Test préservation propriétés structurelles
- Comparaison SYM vs ASY encodings
- Robustesse bruit asymétrique

**SPE (Spectral)** :
- Analyse stabilité linéaire
- Détection concentration énergie
- Prédiction convergence

**PAT (Pattern)** :
- Détection homogénéisation/fragmentation
- Émergence structures
- Collapse patterns

**SPA (Spatial)** :
- Test diffusion/lissage
- Détection rugosité
- Transitions spatiales

**GRA (Graphe)** :
- Interprétation réseau
- Clustering, connectivité
- Structures modulaires

**TOP (Topologique)** :
- Détection trous, fragmentation
- Invariants χ
- Changements topologiques

---

## ANNEXE A : TEMPLATE NOUVEAU TEST

```python
"""
tests/test_category_nnn.py

[Titre descriptif]

Objectif :
- [Description phénomène mesuré ligne 1]
- [Description phénomène mesuré ligne 2]

Métriques :
- [nom_métrique_1] : [Pertinence courte]
- [nom_métrique_2] : [Pertinence courte]

Algorithmes utilisés :
- [registry_key_1] : [Justification courte]
- [registry_key_2] : [Justification courte]

Exclusions :
- [Alternative non retenue 1] : [Raison exclusion]
- [Alternative non retenue 2] : [Raison exclusion]
"""

import numpy as np

TEST_ID = "CAT-NNN"           # Ex: UNIV-003, SYM-002
TEST_CATEGORY = "CAT"         # UNIV, SYM, SPE, PAT, SPA, GRA, TOP
TEST_VERSION = "6.0"          # Version charter actuelle
TEST_WEIGHT = 1.0             # Défaut 1.0, usage R1+

APPLICABILITY_SPEC = {
    "requires_rank": None,              # 2, 3, ou None
    "requires_square": False,           # True si nécessite matrice carrée
    "allowed_d_types": ["ALL"],         # ["SYM", "ASY", "R3"] ou ["ALL"]
    "minimum_dimension": None,          # int ou None
    "requires_even_dimension": False,   # True si nécessite dimension paire
}

COMPUTATION_SPECS = {
    'metric_name_1': {
        'registry_key': 'registry.function_name',  # Ex: 'algebra.matrix_norm'
        'default_params': {
            'param1': default_value1,
            'param2': default_value2,
        },
        'post_process': 'round_4',  # identity, round_N, abs, log
    },
    
    'metric_name_2': {
        'registry_key': 'registry.function_name_2',
        'default_params': {
            'param1': default_value1,
        },
        'post_process': 'round_6',
    },
    
    # Maximum 5 métriques recommandé
}
```

---

## ANNEXE B : REGISTRIES DISPONIBLES

### B.1 algebra_registry

**Fonctions clés** :
- `matrix_norm` : Normes (frobenius, spectral, nucléaire)
- `matrix_asymmetry` : ||A - A^T||
- `trace_value` : Trace (normalisée ou absolue)

**Signature standard** :
```python
def matrix_norm(state: np.ndarray, norm_type: str = 'frobenius') -> float
```

### B.2 spectral_registry

**Fonctions clés** :
- `eigenvalue_max` : Plus grande valeur propre
- `spectral_gap` : λ₁ - λ₂
- `spectral_radius` : max|λᵢ|

**Signature standard** :
```python
def eigenvalue_max(state: np.ndarray, absolute: bool = True) -> float
```

### B.3 pattern_registry

**Fonctions clés** :
- `diversity` : Indice Simpson
- `concentration_ratio` : Concentration top N%
- `uniformity` : Distance distribution uniforme

**Signature standard** :
```python
def diversity(state: np.ndarray, bins: int = 50) -> float
```

### B.4 spatial_registry

**Fonctions clés** :
- `gradient_magnitude` : Norme gradient moyen
- `laplacian_energy` : Énergie laplacien
- `smoothness` : Score lissage

**Signature standard** :
```python
def gradient_magnitude(state: np.ndarray, normalize: bool = True) -> float
```

### B.5 graph_registry

**Fonctions clés** :
- `density` : Densité connexions
- `clustering_local` : Clustering moyen
- `degree_variance` : Variance degrés

**Signature standard** :
```python
def density(state: np.ndarray, threshold: float = 0.1) -> float
```

### B.6 topological_registry

**Fonctions clés** :
- `connected_components` : Comptage composantes
- `holes_count` : Détection trous
- `euler_characteristic` : Calcul χ

**Signature standard** :
```python
def connected_components(state: np.ndarray, threshold: float = 0.0, connectivity: int = 1) -> int
```

**Note** : Voir `registries_catalog.md` pour documentation complète

---

## ANNEXE C : DYNAMIC_EVENTS

### C.1 Types événements

**Onsets (itération détection)** :
- `deviation_onset` : Première déviation significative
- `instability_onset` : Première instabilité détectée

**Flags booléens** :
- `oscillatory` : Oscillations détectées
- `saturation` : Saturation détectée
- `collapse` : Collapse détecté

**Séquence ordonnée** :
- `sequence` : Liste événements ordonnés (ex: `["deviation", "saturation"]`)
- `sequence_timing` : Itérations absolues (ex: `[15, 87]`)
- `sequence_timing_relative` : Normalisé [0,1] (ex: `[0.075, 0.435]`)

### C.2 Usage downstream

**Consommateurs** :
- `gamma_profiling.aggregate_dynamic_signatures()` : Signatures temporelles
- `timeline_utils.compute_timeline_descriptor()` : Timeline compacte
- `regime_utils.classify_regime()` : Classification régimes

**Exemple timeline** :
```python
# Input
sequence = ["deviation", "saturation"]
sequence_timing_relative = [0.075, 0.435]

# Processing
timeline_descriptor = compute_timeline_descriptor(sequence, sequence_timing_relative)

# Output
timeline_descriptor == "early_deviation_then_saturation"
```

### C.3 Détection événements

**Algorithmes (dans test_engine)** :

**deviation_onset** :
```python
# Détecte première déviation > threshold depuis initial
for i, value in enumerate(metric_values):
    if abs(value - initial) > threshold * abs(initial):
        deviation_onset = i
        break
```

**instability_onset** :
```python
# Détecte première variation > threshold
for i in range(1, len(metric_values)):
    if abs(metric_values[i] - metric_values[i-1]) > threshold:
        instability_onset = i
        break
```

**oscillatory** :
```python
# Compte changements de signe dérivée
sign_changes = count_sign_changes(np.diff(metric_values))
oscillatory = sign_changes > threshold_count
```

**saturation** :
```python
# Détecte plateau final
final_window = metric_values[-window_size:]
saturation = np.std(final_window) < threshold
```

**collapse** :
```python
# Détecte convergence vers zéro
collapse = abs(metric_values[-1]) < threshold and trend == "decreasing"
```

---

## ANNEXE D : HISTORIQUE MODIFICATIONS

| Date | Version | Changement |
|------|---------|------------|
| 2025-01-15 | 6.0.0 | Création catalogue initial (9 tests documentés) |
| - | - | Tests catalogués : UNIV-001/002, SYM-001, SPE-001/002, PAT-001, SPA-001, GRA-001, TOP-001 |
| - | - | 21 métriques totales, 7 catégories |

---

## ANNEXE E : INDEX ALPHABÉTIQUE TESTS

| Test | ID | Catégorie | Métriques | Fichier |
|------|-----|-----------|-----------|---------|
| Diversité et concentration | PAT-001 | PAT | diversity_simpson, concentration_top10, uniformity | test_pat_001.py |
| Évolution asymétrie | SYM-001 | SYM | asymmetry_norm, asymmetry_norm_normalized | test_sym_001.py |
| Évolution norme Frobenius | UNIV-001 | UNIV | frobenius_norm | test_uni_001.py |
| Évolution trace | UNIV-002 | UNIV | trace_normalized, trace_absolute | test_uni_002.py |
| Invariants topologiques | TOP-001 | TOP | connected_components, holes_count, euler_characteristic | test_top_001.py |
| Propriétés graphe | GRA-001 | GRA | density, clustering_local, degree_variance | test_gra_001.py |
| Rayon spectral | SPE-002 | SPE | spectral_radius | test_spe_002.py |
| Rugosité et lissage | SPA-001 | SPA | gradient_magnitude, laplacian_energy, smoothness | test_spa_001.py |
| Valeurs propres dominantes | SPE-001 | SPE | eigenvalue_max, spectral_gap | test_spe_001.py |

---

## ANNEXE F : COMPATIBILITÉ γ × TESTS

### F.1 Matrice recommandations

| Gamma | Tests critiques | Tests complémentaires |
|-------|----------------|----------------------|
| GAM-001 (saturation) | UNIV-001, PAT-001 | SYM-001, SPE-001 |
| GAM-002 (diffusion) | SPA-001, PAT-001 | UNIV-001, GRA-001 |
| GAM-003 (croissance) | UNIV-001 | SPE-001, PAT-001 |
| GAM-004 (décroissance) | UNIV-001, PAT-001 | SPE-001 |
| GAM-005 (oscillateur) | UNIV-001, SPE-001 | PAT-001 |
| GAM-006 (mémoire) | UNIV-001, PAT-001 | SYM-001, SPE-001 |
| GAM-007 (moyenne) | SPA-001, PAT-001 | UNIV-001, GRA-001 |
| GAM-008 (friction) | UNIV-001, PAT-001 | SPE-001 |
| GAM-009 (bruit additif) | UNIV-001, PAT-001 | SYM-001 |
| GAM-010 (bruit multiplicatif) | UNIV-001, PAT-001 | SPE-001 |
| GAM-012 (symétrie forcée) | SYM-001, UNIV-001 | SPE-001, PAT-001 |
| GAM-013 (hebbien) | UNIV-001, SPE-001 | PAT-001, GRA-001 |

**Légende** :
- **Critiques** : Tests essentiels détection comportement
- **Complémentaires** : Tests enrichissant profil

### F.2 Encodings × Tests

| Encoding | Tests spécifiques | Tests universels |
|----------|------------------|------------------|
| SYM-* | SYM-001, SPE-001 | UNIV-001, PAT-001 |
| ASY-* | SYM-001 (asymétrie) | UNIV-001, PAT-001 |
| R3-* | UNIV-001, PAT-001 | (SPE, SYM non applicables) |

### F.3 Modifiers × Tests

| Modifier | Tests sensibles | Phénomène attendu |
|----------|----------------|-------------------|
| M0 (baseline) | Tous | Référence sans perturbation |
| M1 (gaussien) | SYM-001, UNIV-001 | Brisure symétrie, augmentation norme |
| M2 (uniforme) | SYM-001, UNIV-001 | Brisure symétrie, augmentation norme |

---

## ANNEXE G : QUESTIONS FRÉQUENTES

### G.1 Pourquoi pas de verdict PASS/FAIL ?

**Réponse** : Tests observent phénomènes, ne jugent pas.

**Rationale** :
- Patterns détectés ultérieurement (verdict_engine)
- Évite biais normatifs précoces
- Permet analyses multi-dimensionnelles
- Un même résultat peut être "bon" pour un γ, "mauvais" pour un autre

**Exemple** :
- `frobenius_norm` → 0 : "mauvais" pour GAM-001 (collapse), "attendu" pour GAM-004 (décroissance)

---

### G.2 Différence ERROR vs NOT_APPLICABLE ?

**Réponse** :

| Status | Cause | Action pipeline | Pénalisation |
|--------|-------|----------------|--------------|
| `ERROR` | Bug code ou instabilité numérique | ARRÊT BATCH | Oui (doit être corrigé) |
| `NOT_APPLICABLE` | Test valide mais contexte invalide ce run | Continue batch | Non (légitime) |

**Exemples ERROR** :
- Registry lève exception (division par zéro non gérée)
- Valeurs propres NaN sur matrice non-singulière (bug algo)
- Import manquant

**Exemples NOT_APPLICABLE** :
- Tous snapshots identiques (pas d'évolution à mesurer)
- Matrice singulière pour test spectral (contexte invalide)
- Données hors bornes algorithme (ex: tous négatifs pour log)

---

### G.3 Pourquoi TEST_WEIGHT si non exploité R0 ?

**Réponse** :
- **Déclaration obligatoire** : Structure future stabilisée maintenant
- **Usage R1+** : Pondération patterns, méta-analyses
- **Évite refactor massif** : Tous tests déjà prêts pour R1

**Usage anticipé R1+** :
```python
# Pondération patterns (R1)
pattern_score = sum(
    test_weight * pattern_intensity 
    for test_weight, pattern_intensity in zip(weights, intensities)
)

# Méta-analyse (R1)
discriminant_power = compute_discriminant_power(
    test_results, 
    weights=test_weights
)
```

---

### G.4 Comment ajouter nouveau registry ?

**Réponse** :

**Étapes** :
1. Créer `tests/utilities/registries/my_registry.py`
2. Définir fonctions avec signature standard :
   ```python
   def my_function(state: np.ndarray, param1: type1 = default1) -> float:
       """Docstring complète."""
       # Implémentation
       return result
   ```
3. Enregistrer dans `registry_manager.py` :
   ```python
   from .my_registry import my_function
   
   REGISTRY['my_category.my_function'] = my_function
   ```
4. Documenter dans `registries_catalog.md`
5. Utiliser dans test :
   ```python
   COMPUTATION_SPECS = {
       'my_metric': {
           'registry_key': 'my_category.my_function',
           'default_params': {'param1': value1},
           'post_process': 'round_4',
       }
   }
   ```

---

### G.5 Limites métriques par test ?

**Réponse** :

| Nombre métriques | Recommandation | Rationale |
|-----------------|----------------|-----------|
| 1 | Acceptable | Test focalisé (ex: UNIV-001) |
| 2-3 | Recommandé | Bon équilibre |
| 4-5 | Maximum | Complexité gestion |
| 6+ | Interdit | Créer nouveau test (séparation responsabilités) |

**Principe** : Un test = un phénomène cohérent

**Contre-exemples** :
- ❌ Mélanger norme + symétrie + spectre dans un seul test
- ✅ Norme seule (UNIV-001), symétrie seule (SYM-001), spectre seul (SPE-001)

---

### G.6 Peut-on avoir plusieurs tests même catégorie ?

**Réponse** : Oui, si phénomènes distincts.

**Exemples légitimes** :
- UNIV-001 (norme Frobenius) vs UNIV-002 (trace)
- SPE-001 (valeurs propres dominantes) vs SPE-002 (rayon spectral)

**Critère** : Complémentarité, pas redondance

---

### G.7 Comment tester sur subset snapshots ?

**Réponse** : Non supporté R0, prévu R1+.

**Workaround R0** :
```python
# Filtrage manuel avant appel test_engine
history_subset = history[::10]  # 1 snapshot sur 10
result = test_engine.run_test(history_subset, test_module, params_config)
```

**Feature R1+** :
```python
COMPUTATION_SPECS = {
    'metric': {
        'sampling': {
            'method': 'uniform',  # uniform, adaptive, endpoints
            'rate': 10  # 1 snapshot sur 10
        }
    }
}
```

**FIN tests_catalog.md**