# CHARTER PRC 6.1 - ÉTAT OPÉRATIONNEL

> Document normatif unique décrivant l'architecture réelle du système PRC.  
> Basé sur l'état fonctionnel documenté dans les catalogues.  
> Version 6.1 prévaut sur toutes versions antérieures.

---

## SECTION 1 : FONDATIONS IMMUABLES

### Axiomes (A1-A3)
- **A1** : Dissymétrie informationnelle D irréductible
- **A2** : Mécanisme Γ agissant sur D
- **A3** : Aucun Γ stable ne peut annuler D complètement

### Hiérarchie des niveaux
```
L0 = ontologique → L1 = épistémique → L2 = théorique → L3 = opérationnel → L4 = documentaire
Règle : L(n) ne référence que L(≤n-1)
Exception explicite :
L3 peut consommer L4 à des fins descriptives ou déclaratives,
mais ne peut pas en dériver de règles nouvelles.
```

### Core = 2 fonctions aveugles
```python
prepare_state(base, modifiers) → np.ndarray  # Composition aveugle
run_kernel(state, gamma, ...) → Generator[iteration, state]  # Itération aveugle
```

**Règles K1-K5** : Le Core reste aveugle :
- Pas de validation sémantique
- Pas de classes State/Operator
- Pas de branchement dépendant de D ou Γ

---

## SECTION 2 : ARCHITECTURE RÉELLE

```
prc_framework/
├── core/                          # Exécution aveugle
│   ├── kernel.py                  # run_kernel()
│   ├── state_preparation.py       # prepare_state()
│   └── core_catalog.md
│
├── operators/                     # Définitions Γ
│   ├── gamma_hyp_*.py             # Classes gamma individuelles
│   └── gamma_catalog.md
│
├── D_encodings/                   # Création D^(base)
│   ├── rank2_symmetric.py         # SYM-001 à SYM-006
│   ├── rank2_asymmetric.py        # ASY-001 à ASY-004
│   ├── rank3_correlations.py      # R3-001 à R3-003
│   └── d_encoding_catalog.md
│
├── modifiers/                     # Transformations D
│   ├── noise.py                   # M1, M2 (bruit)
│   ├── constraints/               # D^(topo)
│   ├── plugins/                   # D^(plugin)
│   ├── domains/                   # D^(domaine)
│   └── modifier_catalog.md
│
├── prc_automation/                # Pipeline d'exécution
│   ├── batch_runner.py            # Point d'entrée unique
│   ├── init_databases.py
│   └── prc_database/
│       ├── prc_r0_raw.db          # Immuable, append-only
│       └── prc_r0_results.db      # Rejouable
│
├── tests/                         # Tests d'observation
│   ├── test_*.py                  # Modules de test (UNIV-001, SYM-001, etc.)
│   ├── tests_catalog.md            
│   ├── config/                    # Configurations YAML
│   └── utilities/                 # Moteurs d'analyse
│
└── reports/                       # Rapports générés
    └── verdicts/
        └── {timestamp}_analysis_{scope}/
```

**Séparation stricte maintenue** :
- `core/` : Exécution aveugle, zéro validation
- `D_encodings/` : Création D avec validation dimensionnelle
- `modifiers/` : Transformations D, passthrough
- `operators/` : Définitions Γ, passthrough
- `tests/` : Observation pure, pas de verdict
- `utilities/` : Moteurs d'analyse, pas d'exécution
- `prc_automation/` : point d'entrée

---

## SECTION 3 : PIPELINE OPÉRATIONNEL

### Workflow complet
```
1. Formulation HYP → 2. Spécification D/Γ → 3. Création TM
→ 4. Exécution kernel (db_raw) → 5. Tests observations (db_results)
→ 6. Profiling gamma → 7. Détection patterns → 8. Génération rapports
→ 9. Discussion → 10. Mise à jour STATUT
→ 11. CON si pertinent → 12. Nouvelle HYP[R1] si pertinent

Règle :
Les étapes 9 à 12 sont hors périmètre logiciel.
Aucun module PRC ne doit tenter de les automatiser.
```

### Bases de données

**prc_r0_raw.db** (immuable, append-only)

**prc_r0_results.db** (rejouable, versionné) 

---

## SECTION 4 : assistance coding

### **Mémoire LLMs** 
```
1. BIBLIOTHÈQUE PERMANENTE (contexte systématique toujours disponible)
   ├── charter_*.md                # Cadre architectural + interdictions
   ├── r0_status.md                # Travail en cours (ce qui change)
   └── functions_index.md          # liste de toutes les fonctionalités existantes

2. CATALOGUES (injection à la demande)
   ├── core_catalog.md
   ├── operators/gamma_catalog.md
   ├── D_encodings/d_encoding_catalog.md
   ├── modifiers/modifier_catalog.md
   ├── tests/tests_catalog.md
   └── tests/utilities/HUB_catalog.md, util_catalog.md, registries_catalog.md, PATTERNS.md

3. SOURCES CODE (injection conversation si besoin)
   ├── tests/utilities/*.py
   ├── operators/gamma_hyp_*.py
   └── ...
```



### Règles Directionnelles (Architecture)

#### Dépendances autorisées
```
core → operators, encodings, modifiers
prc_automation → core, tests, HUB
HUB → UTIL
UTIL → registries
registries → (rien)
```

#### Interdictions strictes
```
❌ UTIL → HUB
❌ HUB → core
❌ Toute dépendance circulaire
```

---

### Principes par Couche

#### HUB (Orchestration)
- **Rôle** : Orchestration, délégation stricte
- **Responsabilité** : Appeler UTIL/PROFILING, jamais réimplémenter
- **Exemples** : test_engine.py, verdict_reporter.py, profiling_runner.py

#### UTIL (Calculs spécialisés)
- **Rôle** : Fonctions stateless, réutilisables
- **Responsabilité** : Aucune orchestration, aucun I/O global
- **Exemples** : regime_utils.py, aggregation_utils.py, statistical_utils.py

#### registries (Fonctions mathématiques)
- **Rôle** : Fonctions pures (state → float)
- **Responsabilité** : Calculs atomiques, zéro dépendance interne
- **Exemples** : algebra_registry.py, spectral_registry.py

---

### Vérification Avant Merge

**Checklist obligatoire** :
- [ ] Pas de `from HUB import` dans UTIL
- [ ] Pas de duplication fonction (vérifier tables "AVANT DE CODER")
- [ ] Imports circulaires détectés : `python -m pycircular tests/`

---

### Exemples Violations Courantes

#### ❌ VIOLATION : UTIL importe HUB
```python
# Dans aggregation_utils.py (UTIL)
from ..HUB.verdict_engine import analyze_regime  # ❌ INTERDIT
```

**Correction** : HUB appelle UTIL, jamais l'inverse
```python
# Dans verdict_engine.py (HUB)
from .aggregation_utils import aggregate_summary_metrics  # ✅ OK
```

#### ❌ VIOLATION : Duplication fonction
```python
# Dans report_writers.py (UTIL)
def extract_conserved_properties(...):  # ❌ EXISTE DÉJÀ
    # Même code que regime_utils.py
```

**Correction** : Importer au lieu de dupliquer
```python
# Dans report_writers.py (UTIL)
from .regime_utils import extract_conserved_properties  # ✅ OK
```

---

### Ressources

**Avant de coder** : Consulter tables "AVANT DE CODER" dans :
- HUB_catalog.md
- UTIL_catalog.md (si créé)

**Registres disponibles** : `registries/README.md`

## SECTION 4.5 : Méthodologie Développement PRC Framework

TL;DR METHODOLOGY
- Pas de code sans objectif écrit
- Pas de code sans organigramme validé
- Interfaces avant implémentation
- Zéro duplication tolérée
- HUB orchestre, UTIL calcule, registries mesurent


### Principe Fondamental

**Ordre développement obligatoire** :

```
1. MÉTIER (objectif, contraintes)
     ↓
2. ORGANIGRAMME (architecture, dépendances)
     ↓
3. STRUCTURE (interfaces, formats)
     ↓
4. CODE (implémentation, tests)
```

**Validation** : Chaque étape complète avant suivante

---

### ÉTAPE 1 : MÉTIER

#### Objectif

Définir **QUOI** et **POURQUOI** avant **COMMENT**.

#### Livrables

**Pour un test** :
- Objectif clair (1-2 phrases)
- Métriques mesurées (liste justifiée)
- Algorithmes utilisés (registry_key précis)
- **Exclusions** (alternatives non retenues + pourquoi)

**Exemple test** :
```
Objectif : Mesurer stabilité globale tenseur sous Γ
Métriques : frobenius_norm (discrimine explosions/effondrements)
Algorithmes : algebra.matrix_norm (standard, robuste, O(n²))
Exclusions :
  - Norme spectrale : Trop coûteuse (SVD), peu discriminante ici
  - Norme nucléaire : Redondante avec Frobenius pour explosions
```

**Pour un gamma** :
- Mécanisme physique/mathématique
- Comportement attendu (convergence, stabilité, trivialité)
- Famille (markovian, non_markovian, stochastic, structural)
- Applicabilité D (SYM, ASY, R3)

**Exemple gamma** :
```
Mécanisme : Diffusion pure via Laplacien discret
Comportement attendu :
  - Convergence : Rapide (<500 iterations)
  - Stabilité : Von Neumann α < 0.25
  - Trivialité : Homogénéisation totale (attracteur uniforme)
Famille : markovian
Applicabilité : SYM, ASY (rang 2 uniquement)
```

#### Validation Étape 1
- [ ] Objectif consensuel (équipe)
- [ ] Métriques justifiées (pas arbitraires)
- [ ] Exclusions documentées (alternatives considérées)
- [ ] Revue par pair complète

---

### ÉTAPE 2 : ORGANIGRAMME

#### Objectif
Définir **ARCHITECTURE** avant code.

#### Livrables
**Diagramme modules** :
```
HUB (orchestration)
  ↓
PROFILING (analyses cross-entités)
  ↓
UTIL (calculs spécialisés)
  ↓
registries (fonctions pures)
```

**Graphe dépendances** :
- Acyclique obligatoire
- Respecter PRC_DEPENDENCY_RULES.md

**Flux données** :
```
observations (DB)
  ↓
data_loading.load_all_observations()
  ↓
profiling_runner.run_all_profiling()
  ↓
verdict_reporter.generate_verdict_report()
  ↓
rapports (JSON, TXT, CSV)
```

#### Validation Étape 2
- [ ] Zéro dépendance circulaire (vérifier `pycircular`)
- [ ] Modules UTIL purs (zéro dépendance interne PRC)
- [ ] HUB délègue strictement (pas de calcul inline)
- [ ] Conforme PRC_DEPENDENCY_RULES.md

---

### ÉTAPE 3 : STRUCTURE

#### Objectif
Définir **INTERFACES** avant implémentation.

#### Livrables
**Signatures fonctions** :
```python
def profile_all_gammas(observations: List[dict]) -> dict:
    """
    Profil comportemental tous gammas.
    
    Args:
        observations: Liste observations SUCCESS
    
    Returns:
        {
            'GAM-001': {
                'tests': {...},
                'n_tests': int,
                'n_total_runs': int
            },
            ...
        }
    """
    pass
```

**Formats retour** :
- Structure dict normalisée
- Clés standardisées (test_name, gamma_id, etc.)
- Types annotations complètes

**Conventions nommage** :
- Tests : `TEST_ID = "CAT-NNN"`
- Gammas : `GAM-NNN`
- Encodings : `SYM-NNN`, `ASY-NNN`, `R3-NNN`
- Registries : `registry.function_name`

#### Validation Étape 3
- [ ] Signatures types annotations complètes
- [ ] Format retour documenté (docstring)
- [ ] Respect conventions Charter
- [ ] Compatibilité backwards (si refactor)

---

### ÉTAPE 4 : CODE

#### Objectif
Implémenter **PROPREMENT** avec documentation.

#### Livrables

**Code source** :
```python
class SomeGamma:
    """
    Description mécanisme.
    
    ATTENDU:
    - Comportement convergence
    - Propriétés conservation
    
    AVEUGLEMENT:
    - Ne connaît pas dimension état
    - Ne connaît pas interprétation
    """
    
    def __init__(self, param: float):
        """
        Args:
            param: Description + contraintes
        
        Raises:
            AssertionError: Si param invalide
        """
        assert param > 0, "param doit être > 0"
        self.param = param
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique transformation Γ.
        
        Args:
            state: Tenseur état
        
        Returns:
            État transformé
        
        Raises:
            ValueError: Si contraintes non respectées
        """
        # Validation
        if state.ndim != 2:
            raise ValueError(f"Rang 2 requis, reçu {state.ndim}")
        
        # Algorithme
        result = ...
        
        return result
    
    def __repr__(self):
        return f"SomeGamma(param={self.param})"
```

**Docstrings obligatoires** :
- Module (header)
- Classe
- Méthodes publiques
- Exemples usage (si pertinent)

**Tests unitaires** :
```python
def test_some_gamma():
    """Test GAM-NNN comportement nominal."""
    
    # État test
    state = np.random.rand(10, 10)
    
    # Gamma
    gamma = SomeGamma(param=1.0)
    
    # Application
    result = gamma(state)
    
    # Assertions
    assert result.shape == state.shape
    assert np.all(np.isfinite(result))
    
    # Cas limites
    with pytest.raises(ValueError):
        gamma(np.random.rand(2, 2, 2))  # Rang 3
```

#### Validation Étape 4
- [ ] Docstrings complètes (module, classe, méthodes)
- [ ] Tests unitaires passent
- [ ] Validation contraintes (assertions)
- [ ] Exemples usage fonctionnels
- [ ] Revue code (pair)

---

### Règles Anti-Duplication (CRITIQUES)

#### Avant TOUTE Nouvelle Fonction

**Checklist obligatoire** :
1. ✅ Consulter table "⚠️ AVANT DE CODER" du catalogue concerné
2. ✅ Vérifier PRC_DEPENDENCY_RULES.md (dépendances autorisées)
3. ✅ Chercher dans `registries/README.md` (si calcul mathématique)
4. ✅ Chercher dans `UTIL_catalog.md` (si agrégation/stats)
5. ✅ Chercher dans `PROFILING_catalog.md` (si profiling)

#### Si Fonction EXISTE Déjà

**INTERDICTION ABSOLUE de la recréer**
```python
# ❌ INTERDIT
def extract_conserved_properties(...):
    # Duplication regime_utils.py
    pass

# ✅ OBLIGATOIRE
from .regime_utils import extract_conserved_properties
```

#### Si Fonction N'EXISTE PAS
**Créer dans le bon module** (selon PRC_DEPENDENCY_RULES.md) :

```python
# ✅ Calcul mathématique → registries
# algebra_registry.py
def compute_metric(state, ...) -> float:
    pass

# ✅ Agrégation statistique → UTIL/aggregation_utils
# aggregation_utils.py
def aggregate_metric(observations, ...) -> dict:
    pass

# ✅ Profiling → PROFILING/profiling_common
# profiling_common.py
def profile_entity(observations, ...) -> dict:
    pass
```

**Puis ajouter à table "⚠️ AVANT DE CODER" appropriée**

---

### Anti-Patterns (À Éviter)

#### ❌ Coder avant architecture

**Symptôme** : Code puis découverte duplication/incohérence

**Correction** : Étapes 1-2 obligatoires avant code

---

#### ❌ Dupliquer docstrings catalogue ↔ source

**Symptôme** : Même texte catalogue et source

**Correction** :
- Source = documentation primaire (docstring complète)
- Catalogue = synthèse/index (référence source)

---

#### ❌ Hardcoder listes entités

**Symptôme** : `GAMMA_IDS = ['GAM-001', 'GAM-002', ...]`

**Correction** : Découverte dynamique
```python
# ✅ Correct
from .discovery import discover_active_tests
tests = discover_active_tests()  # Découverte filesystem
```

---

#### ❌ Mélanger HUB/UTIL responsabilités

**Symptôme** : HUB fait calculs, UTIL fait orchestration

**Correction** :
- HUB = appelle UTIL/PROFILING
- UTIL = calcule, retourne
- PROFILING = profile, compare

---

#### ❌ Imports circulaires

**Symptôme** : A importe B, B importe A

**Correction** : Revoir architecture (Étape 2)

---

### Checklist Validation Globale

**Avant merge** :
- [ ] Étapes 1-4 complètes
- [ ] Docstrings sources complètes
- [ ] Tests unitaires passent
- [ ] Zéro duplication code (vérifier tables)
- [ ] Zéro dépendance circulaire (pycircular)
- [ ] Respect Charter strict
- [ ] PRC_DEPENDENCY_RULES.md respecté
- [ ] Catalogues mis à jour (si nécessaire)

---

### Ressources

**Avant de coder** :
- Tables "⚠️ AVANT DE CODER" (HUB)
- registries/README.md (fonctions disponibles)
- registries/PATTERNS.md (best practices)


## SECTION 5 : UTILITIES - ORGANISATION

### 5.1 Organisation actuelle
```
tests/utilities/
├── 📦 registries              # Fonctions pures (state → float)
│   ├── registries_catalog.md  # Liste l'existant
│   ├── PATTERNS.md             # Best practices, infos générales
│   ├── base_registry.py
│   ├── registry_manager.py
│   ├── post_processors.py
│   ├── algebra_registry.py
│   ├── spectral_registry.py
│   ├── statistical_registry.py
│   ├── spatial_registry.py
│   ├── pattern_registry.py
│   ├── topological_registry.py
│   └── graph_registry.py
│
├── ⭐ HUB (orchestration niveau système)
│   ├── HUB_catalog.md
│   ├── test_engine.py
│   ├── profiling_runner.py       # Orchestration profiling
│   ├── verdict_reporter.py       # Compilation + écriture
│   └── verdict_engine.py         # CONSERVÉ - Analyses globales   
│
└── ✅ UTIL (modules spécialisés) 
    ├── util_catalog.md
    ├── applicability.py
    ├── config_loader.py
    ├── discovery.py
    ├── registries/
    ├── aggregation_utils.py
    ├── data_loading.py
    ├── regime_utils.py
    ├── report_writers.py
    ├── statistical_utils.py
    ├── timeline_utils.py
    ├── profiling_common.py       # Moteur générique + API publique
    └── cross_profiling.py        

```

### Rôles des modules

**📦 registries** :registres de fonctions de calcul (algebra, graph, pattern, etc.)

**⭐ HUB** : Modules d'orchestration Orchestration, délégation stricte (pas de calcul inline)

- `test_engine.py` : Moteur d'exécution des tests avec détection d'événements dynamiques
- `verdict_engine.py` : Analyses statistiques multi-facteurs, détection patterns
- `verdict_reporter.py` : Orchestration complète génération rapports
- `profiling_runner.py` : Orchestration profiling multi-axes avec découverte automatique

**✅ UTIL** : Modules spécialisés (agrégations, classification, I/O)
- `applicability.py` : Validation des contraintes techniques (rang, dimension, etc.)
- `config_loader.py` : Singleton pour chargement configs (fusion global + spécifique)
- `discovery.py` : Découverte et validation structurelle des tests
- `aggregation_utils.py` : Agrégations statistiques (`aggregate_summary_metrics`, `aggregate_run_dispersion`)
- `data_loading.py` : Chargement observations (`load_all_observations`, `observations_to_dataframe`)
- `regime_utils.py` : Classification régimes (`classify_regime`, `stratify_by_regime`, `extract_conserved_properties`)
- `report_writers.py` : Formatage rapports (`write_json`, `write_regime_synthesis`, `write_dynamic_signatures`)
- `statistical_utils.py` : Outils stats (`compute_eta_squared`, `filter_numeric_artifacts`, `diagnose_scale_outliers`)
- `timeline_utils.py` : Construction timelines (`compute_timeline_descriptor`, `classify_timing`)
- `profiling_common.py` : Moteur générique profiling + API publique conventionnelle (profile_all_, compare__summary)
- `cross_profiling.py` : Rankings multi-dimensionnels + analyses interactions + détection signatures globales
    
### 5.2 Hiérarchie logique UTIL

**Organisation par niveaux de dépendance** (ordre logique, pas contrainte stricte) :

```
NIVEAU 1 (Données brutes)
├── data_loading.py           # DB → observations list
├── discovery.py              # Filesystem → test modules
└── config_loader.py          # YAML → config dict

NIVEAU 2 (Validation)
└── applicability.py          # Contraintes techniques

NIVEAU 3 (Transformations)
├── aggregation_utils.py      # Observations → métriques agrégées
├── timeline_utils.py         # Events → timeline descriptors
└── statistical_utils.py      # Observations → diagnostics

NIVEAU 4 (Classification)
├── regime_utils.py           # Métriques → régimes
├── profiling_common.py       # Observations → profils entités
└── cross_profiling.py        # Profils → analyses croisées

NIVEAU 5 (Output)
└── report_writers.py         # Données → fichiers formatés
```

**Principe** : Impossible générer rapport sans traiter données, impossible traiter sans charger.

### 5.3 Catégories FUNCTIONS_INDEX.md

**Pour orientation rapide avant coding** :

| Catégorie | Contenu | Fichiers typiques |
|-----------|---------|------------------|
| **Chargement données** | DB, filesystem, configs | data_loading, discovery, config_loader |
| **Validation** | Applicabilité, contraintes | applicability |
| **Transformations** | Agrégations, timelines, stats | aggregation_utils, timeline_utils, statistical_utils |
| **Classification** | Régimes, profiling | regime_utils, profiling_common, cross_profiling |
| **Output** | Writers, formatage | report_writers |
| **Registries** | Calculs purs (algebra, spectral, etc.) | *_registry.py |
| **Orchestration** | Moteurs exécution | HUB/* |

**Usage** :
1. Identifier catégorie pertinente (ex: besoin agrégation → "Transformations")
2. Consulter FUNCTIONS_INDEX.md section correspondante
3. Si fonction existe → utiliser
4. Si fonction absente → créer dans fichier approprié

### 5.4 Règles architecturales

**R5-A** (Source de vérité) :
Le catalogue définit ce qui est théoriquement disponible.
La base de données définit ce qui a été effectivement exécuté.
Le profiling ne travaille que sur les entités présentes dans les observations.

**R5-B** (Interdiction) :
Aucun module ne doit :
- Hardcoder une liste d'IDs
- Supposer l'exhaustivité des catalogues
- Inférer l'existence d'une entité non observée

**R5-C** (Principe fondateur) :
Le pipeline PRC a été initialement développé autour du gamma profiling.
Toute extension (modifier, encoding, test) doit :
- Réutiliser les abstractions existantes
- Ne jamais spécialiser le cœur
- Étendre uniquement par registres, hooks ou extensions déclarées

**R5-D** (Fonction privée → publique) :
Si une fonction privée (préfixe `_`) doit être réutilisée dans un autre module :
1. La rendre publique (supprimer `_`)
2. L'appeler depuis l'ancien ET le nouveau fichier
3. Mettre à jour FUNCTIONS_INDEX.md
4. **Ne jamais dupliquer** le code

### 5.5 Graphe dépendances

**Dépendances autorisées** :
```
HUB → UTIL, registries
UTIL (niveau N) → UTIL (niveau < N), registries
registries → (rien, fonctions pures)
```

**Interdictions strictes** :
```
❌ UTIL → HUB
❌ HUB → HUB (entre modules)
❌ registries → UTIL ou HUB
❌ Toute dépendance circulaire
```

*NOTE*: Sections 6–8 : Cartographie conceptuelle transversale (non implémentation)
Ces sections décrivent :
    -les formes des objets manipulés
    -les axes analytiques existants
    -les structures conceptuelles attendues

Elles ne constituent pas une source d’implémentation.
Toute implémentation doit :
    -Consulter functions_index.md
    -Consulter les catalogues pertinents
    -Réutiliser l’existant avant toute création
    
## SECTION 6 : INVARIANTS NORMATIFS

### Détection de patterns
**R6-A** : Patterns détectés sur observations **brutes + normalisées localement**

**Patterns détectés** :
1. **NON_DISCRIMINANT** : Toutes observations normalisées < seuil
2. **OVER_DISCRIMINANT** : Toutes observations normalisées > seuil  
3. **D_CORRELATED** : Variance inter-d_encoding_id > seuil
4. **MODIFIER_CORRELATED** : Variance inter-modifier_id > seuil
5. **SEED_UNSTABLE** : Variance inter-seeds > seuil (à D/modifier fixés)
6. **SYSTEMATIC_ANOMALY** : Fraction observations |normalized| > seuil dépasse ratio
7. **CONTEXTUAL_BEHAVIOR** : Combinaison D_CORRELATED ou MODIFIER_CORRELATED

### 6.1 Taxonomie régimes

**Régimes de conservation (sains)** :
```python
CONSERVES_SYMMETRY    # Test SYM-*
CONSERVES_NORM        # Test UNIV-*, SPE-*
CONSERVES_PATTERN     # Test PAT-*
CONSERVES_TOPOLOGY    # Test TOP-*
CONSERVES_GRADIENT    # Test SPA-*
CONSERVES_SPECTRUM    # Test GRA-*
```

**Régimes pathologiques** :
```python
NUMERIC_INSTABILITY   # Valeurs infinies/NaN
OSCILLATORY_UNSTABLE  # Oscillations non amorties
TRIVIAL               # Collapse vers zéro/constante
DEGRADING             # Dégradation progressive
```

**Autres régimes** :
```python
SATURATES_HIGH        # Saturation haute (tanh-like)
SATURATES_LOW         # Saturation basse
UNCATEGORIZED         # Non classifiable
```

**Qualificatif multimodal** :
```python
MIXED::{régime_base}  # Bimodalité détectée (ex: MIXED::CONSERVES_NORM)
```

**R6-B** : Régime déterminé par `regime_utils.classify_regime()` basé sur :
- Métriques statistiques
- Signature dynamique
- Distribution timeline
- Dispersion inter-runs
- Type de test

### 6.2 Normalisation robuste

**Méthode par défaut** (détection patterns) :
```python
normalized = (x - median) / IQR  # IQR = Q3 - Q1
```

**Propriétés** :
- Insensible valeurs extrêmes
- N'existe QUE dans cadre détection patterns
- Aucune donnée persistée normalisée

**Seuils typiques** (configurables via `verdict_config.yaml`) :
```yaml
patterns:
  non_discriminant:
    normalized_threshold: -2.0
  over_discriminant:
    normalized_threshold: 2.0
  seed_unstable:
    variance_threshold: 3.0
```

### 6.3 Stratification stable/explosif

**Critère** : Présence valeurs > threshold dans projections exploitées

**Seuil par défaut** : `1e50` (5 décades au-dessus P90 typique)

**Configurable via** :
```yaml
stratification:
  explosion_threshold: 1.0e50
```

**Projections inspectées** :
- `statistics` : initial, final, mean, max
- `evolution` : slope, relative_change

**Usage** : Analyses séparées GLOBAL / STABLE / EXPLOSIF pour isolation effets.

###  6.4 : RELATION PATTERNS ↔ VERDICTS

**R6.4-A** : Patterns = observations factuelles détectées
**R6.4-B** : Verdicts = décisions humaines post-rapports (HORS périmètre logiciel)

**Pipeline actuel** :
1. VerdictEngine détecte PATTERNS (D_CORRELATED, etc.)
2. VerdictReporter génère RAPPORTS (analysis_complete.json, summary.txt)
3. HUMAIN lit rapports → décide VERDICT (SURVIVES[R0], WIP[R0-open])
4. HUMAIN met à jour gamma_catalog.md avec STATUS

## SECTION 7 : PROFILING - PRINCIPES

### 7.1 Format retour unifié

**Règle R7.1-A** : Tous axes profiling DOIVENT retourner structure :
```python
{
    'profiles': dict,      # Profils individuels entités
    'summary': dict,       # Comparaisons cross-entités
    'metadata': dict,      # Infos exécution
    # Extensions spécifiques axe (optionnel)
}
```

**R7.1-B** : Règles découverte modules
- Modules profiling nommés : `*_profiling.py`
- Doivent exposer fonctions conventionnelles :
  - `profile_all_{axis}(observations) → dict`
  - `compare_{axis}_summary(profiles) → dict`
- Extensions détectées via introspection module

**R7.1-C **: Ordre découverte axes
```python
PROFILING_AXES = {
    'test': {...},      # Ordre déclaré explicite (dict standard Python 3.7+)
    'gamma': {...},
    'modifier': {...},
    'encoding': {...}
}
```
Rationale ordre :
    -test : Axe observation privilégié (lisibilité, comparabilité)
    -gamma, modifier, encoding : Axes causaux
    -⚠️ Axe test non structurellement dominant : présent par défaut mais pas obligatoire


**R5.1-D **: Tous axes garantis présents
    -Observations proviennent toujours de db_results
    -Chaque observation contient : test_name, gamma_id, modifier_id, d_encoding_id
    -Aucun axe ne peut être "manquant" (tous axes profiling exécutables)
    -Différence possible : nombre entités par axe (ex: 13 gammas, 3 modifiers)
  
### 7.2 Conventions nommage

**API publique conventionnelle** (détection automatique) :
```python
# Profiling individuel
profile_all_tests(observations) → dict
profile_all_gammas(observations) → dict
profile_all_modifiers(observations) → dict
profile_all_encodings(observations) → dict

# Comparaisons
compare_tests_summary(profiles) → dict
compare_gammas_summary(profiles) → dict
compare_modifiers_summary(profiles) → dict
compare_encodings_summary(profiles) → dict
```

**Mapping entity keys** (normalisation colonnes DB) :
```python
ENTITY_KEY_MAP = {
    'test': 'test_name',
    'gamma': 'gamma_id',
    'modifier': 'modifier_id',
    'encoding': 'd_encoding_id'
}
```

### 7.3 Délégation stricte

**Règle R7.3-A** : Modules profiling DOIVENT utiliser moteur générique :
```python
# Dans profiling_common.py
_profile_entity_axis(observations, axis, entity_key) → dict
```

**Règle R7.3-B** : HUB ne doit PAS implémenter profiling inline :
```python
# ❌ INTERDIT dans HUB
for gamma in gammas:
    regime = classify_regime(...)  # Réimplémentation

# ✅ CORRECT
results = profile_all_gammas(observations)
```

**Règle R7.3-C** : Aucun module ne doit duplicer code `profiling_common`.

### 8.1 Structure arbre (normative)
```
reports/verdicts/TIMESTAMP_analysis_global/
├── summary_global.txt              # Synthèse unifiée 
│
├── summaries/                      # Synthèses par axe
│   ├── summary_test.txt
│   ├── summary_gamma.txt
│   ├── summary_modifier.txt
│   └── summary_encoding.txt
│
├── analysis_complete.json          # Données complètes structurées
│
├── profiles/                       # Profils par axe/entité
│   ├── test/
│   │   └── {TEST-ID}.json
│   ├── gamma/
│   │   └── {GAM-ID}.json
│   ├── modifier/
│   │   └── {M-ID}.json
│   └── encoding/
│       └── {ENC-ID}.json
│
├── comparisons/                    # Rankings/interactions
│   ├── rankings_*.json
│   └── interactions_*.json         # Placeholders R0
│
└── diagnostics/                    # Diagnostics techniques
    ├── degeneracy_report.json
    └── scale_outliers.json
```

### 8.3 Formats output

**TXT** : Synthèses lisibles humain (summary_*.txt)
**JSON** : Données structurées réutilisables (analysis_complete.json, profiles/*.json)
**CSV** : Tableaux exportables (si pertinent, non systématique R0)

**Règle R8.3-A** : Tout rapport doit permettre reconstruction état complet via JSON.


### Templates Jinja2

**R8-C** : Templates génériques
```
templates/
├── summary_global.md.j2       # Synthèse multi-axes
├── summary_axis.md.j2         # Générique tous axes (sections conditionnelles)
└── profile_entity.json.j2     # Générique toute entité
```

**R8-D **: Sections conditionnelles autorisées
``` Jinja2
{% if discriminant_powers %}
## POUVOIR DISCRIMINANT
...
{% endif %}
```

Permet flexibilité axe test (discriminant_power) sans dupliquer templates

## SECTION 9 : EXTENSION DU SYSTÈME

### Principes d'extension
**R9-A** : Tout nouveau module DOIT respecter les I/O des catalogues existants
**R9-B** : Réutiliser les nomenclatures établies (IDs, catégories, paramètres)
**R9-C** : Suivre les patterns d'organisation observés dans les catalogues
**R9-D** (anti-hallucination): Si une information nécessaire à une implémentation n’est pas explicitement définie dans ce Charter,
    le développeur ou LLM doit demander le fichier source concerné
    (catalogue, module .py, config YAML).
    Toute supposition est interdite.

### Checklist de validation
Avant d'ajouter un nouveau module :
- [ ] IDs cohérents avec nomenclature existante
- [ ] I/O compatibles avec consommateurs/dépendants
- [ ] Aucun hardcoding (utiliser configs YAML)
- [ ] Documenté dans catalogue approprié
- [ ] Testé dans pipeline existant
- [ ] Respecte séparation stricte des responsabilités

## SECTION 9.5 : STRUCTURE CONFIGS YAML

### params_*.yaml (structure minimale)
```yaml
version: "1.0"           # OBLIGATOIRE
config_id: "params_xxx"  # OBLIGATOIRE
description: "..."       # OBLIGATOIRE

test_parameters:         # OBLIGATOIRE (peut être vide {})
  TEST-001:
    param1: value1
```

### verdict_*.yaml (structure minimale)
```yaml
version: "1.0"                    # OBLIGATOIRE
config_id: "verdict_xxx"          # OBLIGATOIRE
description: "..."                # OBLIGATOIRE

normalization:                    # OBLIGATOIRE
  default_method: "robust"
  scope: "per_test"

patterns:                         # OBLIGATOIRE (au moins 1 pattern)
  non_discriminant:
    normalized_threshold: -2.0
```

**Validation ConfigLoader** :
- Vérifie présence clés obligatoires
- Log warning si metadata incohérente
- Pas de validation contenu (délégué aux consommateurs)


## SECTION 10 : TEST_WEIGHT (État R0)

**R10-A** : Déclaration OBLIGATOIRE au niveau module
**R10-B** : Valeur par défaut : 1.0
**R10-C** : **Non exploité en R0** (réservé analyses R1+)

**Usage futur anticipé (R1+)** :
- Pondération patterns cross-tests
- Méta-analyses multi-tests

**Interdictions R0** :
- Pas de filtrage exécution
- Pas d'interprétation "qualité"
---

## SECTION 11 : NOMENCLATURE ACTUALISÉE

### IDs et catégories
```
# Encodings D
SYM-001 à SYM-006    # Symétrique rang 2
ASY-001 à ASY-004    # Asymétrique rang 2  
R3-001 à R3-003      # Rang 3

# Modifiers
M0                   # Baseline (aucune modification)
M1                   # Bruit gaussien (sigma=0.05)
M2                   # Bruit uniforme (amplitude=0.1)

# Gammas
GAM-001 à GAM-013    # Mécanismes Γ

# Tests
UNIV-001 à UNIV-002  # Universels
SYM-001              # Symétrie
SPE-001 à SPE-002    # Spectral
PAT-001              # Pattern
SPA-001              # Spatial
GRA-001              # Graphe
TOP-001              # Topologique
```

### Gestions fichiers:
#### Règle G1 : Une brique = un mécanisme = un fichier, pas un réglage

Critère : "Puis-je expliquer en une phrase ce que cette brique fait de différent ?"
- OUI → nouveau fichier
- NON → paramètre

#### Règle G2 : Métadonnées obligatoires complètes

Minimum :
- ID (attribut fonction)
- PHASE (variable module)
- METADATA['description']

Discovery lève CriticalDiscoveryError si absent.

#### Règle G3 : Dépréciation explicite

Jamais supprimer fichier sans :
- `_deprecated_` dans nom, OU
- `DEPRECATED = True` dans module


### Termes actualisés
| Terme | Définition | Remplace |
|-------|------------|----------|
| **d_encoding_id** | Identifiant encodage D | d_base_id |
| **régime** | Comportement global (CONSERVES_X, PATHOLOGY) | score, verdict |
| **timeline** | Séquence événements dynamiques | history pattern |
| **HUB** | Module d'orchestration (à refactoriser) | engine principal |
| **UTIL** | Module utilitaire spécialisé | helper function |
| **profil gamma** | Caractérisation comportementale Γ | gamma score |
| **profiling axis** | Dimension analyse (gamma, modifier, etc.) | - |
| **profiling runner** | Orchestrateur profiling multi-axes | - |
| **unified format** | Structure retour standard profiling | - |


### Statuts autorisés
```
WIP[R0-open]         # Exploration en cours
WIP[R0-closed]       # R0 terminé, ambigu → mise en attente  
SURVIVES[R0]         # Non éliminé à R0
REJECTED[R0]         # Éliminé comme autonome à R0
REJECTED[GLOBAL]     # Éliminé définitif tous rangs
```

## SECTION 12 : INTERDICTIONS CRITIQUES

### Téléologie
**R11-A** : TEST_WEIGHT ne doit jamais être :
- Utilisé pour filtrer/prioriser exécution
- Interprété comme "qualité" test

### Métriques
**R11-C** : Toute métrique `COMPUTATION_SPECS` **doit** être analysable par patterns
**R11-D** : Aucun skip silencieux autorisé → ERROR fatal si applicable

### Core
**R11-E** : Core reste aveugle :
- Aucune validation contenu (symétrie, bornes)
- Aucune classe State, Operator
- Aucun branchement dépendant de D ou Γ

### Tests
**R11-F** : Tests ne retournent jamais :
- "PASS"/"FAIL", "bon"/"mauvais"
- Classes/objets
- Paramètres hardcodés
- Jugements normatifs

### Pipeline
**R11-G** : Pipeline strictement séquentiel :
- Interdit : reruns dans le but d’influencer rétroactivement un verdict existant.
- Aucun mélange db_raw et db_results
- Aucune modification rétroactive db_raw

### Vocabulaire
**R11-H** : Nomenclature stricte :
- ❌ "matrice de corrélation" → ✅ "tenseur rang 2"
- ❌ "graphe" → ✅ "patterns dans tenseur"
- ❌ "position (i,j)" → ✅ "indice (i,j)"
- ❌ "d_base_id" → ✅ "d_encoding_id"

### Profiling

**R12-I** : Modules profiling DOIVENT :
    -Implémenter API conventionnelle profile_all_{axis}() et compare_{axis}_summary()
    -Utiliser moteur générique _profile_entity_axis() (pas réimplémenter)
    -Respecter format retour unifié (R5.1-A) strictement
    -Utiliser ENTITY_KEY_MAP pour normalisation noms colonnes DB

**R12-J** : Modules profiling NE DOIVENT PAS :
- Dupliquer code `profiling_common` (extraction obligatoire)
- Hardcoder listes entités (découverte dynamique depuis observations)
- Retourner formats hétérogènes (validation stricte)
- Implémenter verdicts décisionnels (rankings descriptifs uniquement)

### Orchestration

**R12-K** : `profiling_runner` NE DOIT PAS :
- Hardcoder liste axes disponibles (découverte automatique)
- Modifier données profiling (passthrough strict)
- Implémenter logique profiling (délégation totale modules)

**R12-L** : `verdict_reporter` NE DOIT PAS :
- Appeler directement modules profiling (passer par `profiling_runner`)
- Implémenter logique analytique (compilation + écriture uniquement)

**R12-M **: Architecture unifiée - Interdictions

    -❌ Créer modules {axe}_profiling.py séparés (tout dans profiling_common.py)
    -❌ Dupliquer logique profiling (un seul moteur _profile_entity_axis)
    -❌ Alias entity keys (utiliser exactement noms DB : d_encoding_id pas encoding_id)
    -❌ Extensions format retour hors discriminant_power pour tests (tout autre enrichissement → cross_profiling)


## SECTION 13 : GLOSSAIRE ACTUALISÉ

| Terme | Définition | Exemple |
|-------|------------|---------|
| **Registre** | Module fonctions réutilisables | `algebra_registry.py` |
| **registry_key** | Identifiant unique fonction | `"algebra.matrix_norm"` |
| **run_metadata** | Métadonnées exécution | `{gamma_id, d_encoding_id, modifier_id, seed}` |
| **history** | Liste snapshots (~200) | `[state_0, ..., state_199]` |
| **snapshot** | UN état à une itération | `np.ndarray (10, 10)` |
| **post_processor** | Transformation post-calcul | `"round_4"`, `"abs"` |
| **observation** | Dict retour test | `{'statistics': {...}, 'evolution': {...}}` |
| **pattern** | Régularité détectée cross-runs | `"D_CORRELATED"`, `"SYSTEMATIC_ANOMALY"` |
| **régime** | Comportement classifié | `"CONSERVES_SYMMETRY"`, `"NUMERIC_INSTABILITY"` |
| **timeline** | Séquence événements dynamiques | `"early_deviation_then_saturation"` |
| **profil gamma** | Caractérisation Γ | `{regime: "...", metrics: {...}, timeline: "..."}` |
| **HUB** | Module orchestration | `verdict_engine.py`, `gamma_profiling.py` |
| **UTIL** | Module utilitaire spécialisé | `aggregation_utils.py`, `timeline_utils.py` |
| **normalisation robuste** | `(x - median) / IQR` | Méthode par défaut |
| **IQR** | Écart interquartile | `Q3 - Q1` |
| **bimodalité** | Distribution à deux modes | `IQR_ratio > 3.0` |
| **timeline compacte** | Représentation textuelle timeline | `"early_deviation_then_saturation"` |
| **profiling axis** | Dimension analyse (test, gamma, modifier, encoding) |- |
| **entity key** | Nom colonne DB identifiant entité axe | 'gamma_id', 'd_encoding_id' |
| **discriminant power** | Pouvoir discriminant test cross-entities | variance inter-gammas | 
| **grouping dimension** | Axe regroupement pour ranking | 'test' dans rank gamma par test |

### 14.1 Obligation template + catalogue

**Règle R14-A** : Toute extension encodings/tests/gammas/modifiers DOIT :
1. Demander template approprié (structure fichier)
2. Consulter catalogue existant (nomenclature, conventions)
3. Documenter dans catalogue après création
4. Mettre à jour FUNCTIONS_INDEX.md si fonctions publiques

**Processus** :
```
1. Identifier type extension (encoding/test/gamma/modifier)
2. Demander : "Template pour nouveau {type}" + "{type}_catalog.md"
3. Suivre template strictement
4. Après implémentation : mettre à jour catalogue
5. Si fonctions réutilisables : extraire dans UTIL + FUNCTIONS_INDEX.md
```

**Templates disponibles** (à demander explicitement) :
- `Template nouveau encoding` : D_encodings/
- `Template nouveau test` : tests/
- `Template nouveau gamma` : operators/
- `Template nouveau modifier` : modifiers/

**Catalogues associés** :
- `d_encoding_catalog.md`
- `tests_catalog.md`
- `operators_catalog.md`
- `modifier_catalog.md`

**Interdiction** : Créer encoding/test/gamma/modifier sans consulter template + catalogue.

**FIN CHARTER PRC 6.1**

Ce charter est la **référence opérationnelle unique** pour l'architecture PRC 6.1.  
Il décrit l'état **réel** du système tel qu'implémenté et documenté dans les catalogues.  
Toute extension DOIT se conformer strictement à ces spécifications.