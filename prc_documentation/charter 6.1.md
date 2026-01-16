# CHARTER PRC 6.0 - ÉTAT OPÉRATIONNEL

> Document normatif unique décrivant l'architecture réelle du système PRC.  
> Basé sur l'état fonctionnel documenté dans les catalogues.  
> Version 6.0 prévaut sur toutes versions antérieures.

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

### Commandes batch_runner
```bash
# Mode 1 : Collecte données
batch_runner --brut --gamma GAM-001
# → Exécute kernel, stocke dans db_raw (immuable)

# Mode 2 : Application tests
batch_runner --test --gamma GAM-001 --params default_v1
# → Vérifie db_raw, auto-lance --brut si manquant
# → Applique tests, stocke dans db_results

# Mode 3 : Génération verdicts
batch_runner --verdict --params default_v1 --verdict default_v1
# → Vérifie db_results, auto-lance --test si manquant
# → Analyse patterns, génère rapports

# Raccourci complet
batch_runner --all --gamma GAM-001 --params default_v1 --verdict default_v1
```

### Bases de données

**prc_r0_raw.db** (immuable, append-only) :
```sql
Executions   -- Métadonnées runs (gamma_id, d_encoding_id, modifier_id, seed)
Snapshots    -- États sauvegardés (compressed)
Metrics      -- Métriques brutes par iteration
```

**prc_r0_results.db** (rejouable, versionné) :
```sql
TestObservations  -- Résultats tests (format dict v2)
-- Colonnes : observation_id, exec_id, test_name, params_config_id,
--            applicable, status, observation_data (JSON), computed_at
```

---

## SECTION 4 : TESTS - FORMAT ACTUEL

### Principe fondamental
**R4-A** : Tests OBSERVENT (pas de jugement). Patterns détectés ultérieurement.

### Structure d'un module test
```python
# tests/test_category_nnn.py
"""
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

TEST_ID = "CAT-NNN"           # Format CAT-NNN (ex: UNIV-001)
TEST_CATEGORY = "CAT"         # UNIV, SYM, SPE, etc.
TEST_VERSION = "6.0"          # Doit être "6.0"
TEST_WEIGHT = 1.0             # Obligatoire, importance épistémique

APPLICABILITY_SPEC = {
    "requires_rank": int | None,          # 2, 3, ou None (tout)
    "requires_square": bool,              # Matrice carrée requise
    "allowed_d_types": List[str],         # ["SYM", "ASY", "R3"]
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
        'post_process': 'round_4',  # Optionnel : identity, round_N, abs, log, etc.
    },
    # 1 à 5 métriques maximum
}
```

### Format de retour (dict v2)
```python
{
    # Traçabilité
    'run_metadata': {
        'gamma_id': str,
        'd_encoding_id': str,      # PAS d_base_id
        'modifier_id': str,
        'seed': int,
    },
    
    # Identification
    'test_name': str,
    'test_category': str,
    'test_version': str,
    'params_config_id': str,
    
    # Status (seulement 3 valeurs autorisées)
    'status': 'SUCCESS' | 'ERROR' | 'NOT_APPLICABLE',
    'message': str,  # Description si ERROR ou NOT_APPLICABLE
    
    # Résultats (dict de dicts)
    'statistics': {
        'metric_name': {
            'initial': float,
            'final': float,
            'min': float,
            'max': float,
            'mean': float,
            'std': float,
            'median': float,
            'q1': float,
            'q3': float,
            'n_valid': int,
        },
    },
    
    # Evolution (dict de dicts)
    'evolution': {
        'metric_name': {
            'transition': str,      # "stable", "increasing", "explosive", "collapsing"
            'trend': str,           # "monotonic", "oscillatory", "chaotic"
            'slope': float,         # Coefficient de régression linéaire
            'volatility': float,    # Écart-type des différences
            'relative_change': float,  # (final - initial) / max(|initial|, 1e-10)
        },
    },
    
    # Événements dynamiques 
    'dynamic_events': {
        'metric_name': {
            # Onsets (itération détection)
            'deviation_onset': int | None,
            'instability_onset': int | None,
        
            # Flags booléens
            'oscillatory': bool,
            'saturation': bool,
            'collapse': bool,
        
            # Séquence ordonnée
            'sequence': List[str],              # ["deviation", "saturation"]
            'sequence_timing': List[int],       # [15, 87] (itérations absolues)
            'sequence_timing_relative': List[float],  # [0.075, 0.435] (normalisé [0,1])
        },
    },
    
    # Métadonnées techniques
    'metadata': {
        'engine_version': '6.0',
        'execution_time_sec': float,
        'num_iterations_processed': int,
        'total_metrics': int,
        'successful_metrics': int,
        'computations': {
            'metric_name': {
                'registry_key': str,
                'params_used': dict,
                'has_post_process': bool,
            },
        }
    }
}
```

**Utilisation downstream** :
- `gamma_profiling.aggregate_dynamic_signatures()` lit ce format
- `timeline_utils.compute_timeline_descriptor()` génère timeline compacte

### Status autorisés
- `SUCCESS` : Test exécuté normalement
- `ERROR` : Erreur technique → ARRÊT BATCH (bug code)
- `NOT_APPLICABLE` : Test invalide pour contexte (non pénalisé)


---

## SECTION 5 : UTILITIES - ARCHITECTURE RÉELLE

### Organisation actuelle
```
tests/utilities/
├── 📦 registries
│   ├── registries_catalog.md
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

**⭐ HUB** : Modules d'orchestration 

- `test_engine.py` : Moteur d'exécution des tests avec détection d'événements dynamiques
- `verdict_engine.py` : Analyses statistiques multi-facteurs, détection patterns
- `verdict_reporter.py` : Orchestration complète génération rapports
- `profiling_runner.py` : Orchestration profiling multi-axes avec découverte automatique

**✅ UTIL** : Modules spécialisés
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

---
**R5-A** (Source de vérité)
Le catalogue définit ce qui est théoriquement disponible.
La base de données définit ce qui a été effectivement exécuté.
Le profiling ne travaille que sur les entités présentes dans les observations.

**R5-B** (Interdiction)
Aucun module ne doit :
    -hardcoder une liste d’IDs,
    -supposer l’exhaustivité des catalogues,
    -inférer l’existence d’une entité non observée.
    
**R5-C** (Principe fondateur)
Le pipeline PRC a été initialement développé autour du gamma profiling.
Toute extension (modifier, encoding, test) doit :
    -réutiliser les abstractions existantes,
    -ne jamais spécialiser le cœur,
    -étendre uniquement par registres, hooks ou extensions déclarées.
    
## SECTION 5.1 : PROFILING RUNNER

### Responsabilité
Orchestration exécution profiling multi-axes avec découverte automatique.

### Architecture découverte automatique
```python
# Découverte modules profiling disponibles
PROFILING_AXES = discover_profiling_modules()
# Retourne : {'gamma': module, 'modifier': module, ...}

# Exécution dynamique
run_all_profiling(observations, axes=['gamma', 'modifier'])
# Appelle automatiquement :
#   - gamma_profiling.profile_all_gammas()
#   - modifier_profiling.profile_all_modifiers()
```

### Format retour UNIFIÉ (règle stricte)

**R5.1-A** : Tous axes profiling DOIVENT retourner structure :
```python
{
    'profiles': dict,      # Profils individuels entités
    'summary': dict,       # Comparaisons cross-entités
    'metadata': dict,      # Infos exécution
    # Extensions spécifiques axe (optionnel, voir R5.1-B)
}
```

**R5.1-B** : Règles découverte modules
- Modules profiling nommés : `*_profiling.py`
- Doivent exposer fonctions conventionnelles :
  - `profile_all_{axis}(observations) → dict`
  - `compare_{axis}_summary(profiles) → dict`
- Extensions détectées via introspection module

**R5.1-D **: Ordre découverte axes
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


**R5.1-E **: Tous axes garantis présents
    -Observations proviennent toujours de db_results
    -Chaque observation contient : test_name, gamma_id, modifier_id, d_encoding_id
    -Aucun axe ne peut être "manquant" (tous axes profiling exécutables)
    -Différence possible : nombre entités par axe (ex: 13 gammas, 3 modifiers)


## SECTION 5.2 : PROFILING COMMON

### Responsabilité
Fonctions réutilisables tous modules profiling. Évite duplication code.

### Règles utilisation

**R5.2-A** : Moteur générique interne
```python
_profile_entity_axis(observations, axis, entity_key) → dict
# Moteur interne privé, logique partagée tous axes
```

**R5.2-B** : API publique conventionnelle
```python
# Fonctions découvrables automatiquement (naming convention stricte)
profile_all_tests(observations) → dict
profile_all_gammas(observations) → dict
profile_all_modifiers(observations) → dict
profile_all_encodings(observations) → dict

compare_tests_summary(profiles) → dict
compare_gammas_summary(profiles) → dict
compare_modifiers_summary(profiles) → dict
compare_encodings_summary(profiles) → dict
```
**R5.2-C ***: Mapping entity keys
```python
# Normalisation sur noms colonnes DB (pas d'alias)
ENTITY_KEY_MAP = {
    'test': 'test_name',
    'gamma': 'gamma_id',
    'modifier': 'modifier_id',
    'encoding': 'd_encoding_id'
}
```

**R5.2-D **: Axe test enrichi discriminant power
```python
# Structure identique autres axes + métrique additionnelle
profiles['UNIV-001'] = {
    'entities': {...},           # Standard (comme gamma/modifier/encoding)
    'discriminant_power': {      # AJOUT spécifique test
        'inter_entity_variance': float,
        'effect_size': float,
        'ranking_consistency': float
    },
    'n_entities': int,
    'n_total_runs': int
}
```

## SECTION 5.3 : CROSS PROFILING 

### Responsabilité
Analyse interactions entre axes profiling.

**Exemples cas d'usage** :
- Interaction gamma × modifier (GAM-004 × M1 → OSCILLATORY systématique)
- Interaction encoding × test (SYM vs ASY sous SYM-001)
- Interaction gamma × encoding × modifier (3-way)

**Emplacement** : `tests/utilities/PROFILING/cross_profiling.py`

**R5.3-A **: Rankings multi-dimensionnels
```python
rank_entities_by_metric(
    profiles: dict,
    grouping_dimension: str,   # 'test', 'encoding', 'modifier'
    metric_key: str,           # 'SYM-001', 'M1', 'GAM-004'
    criterion: str | callable
) → dict

# Remplace rank_gammas_by_test() (rétrocompatibilité via même logique)
```

**R5.3-B **: Interactions (placeholders R0)
```python
analyze_pairwise_interactions(profiles_a, profiles_b) → dict
analyze_multiway_interactions(all_profiles) → dict
detect_global_signatures(all_profiles) → dict

# R0 : Docstrings + structure retour définies, implémentation différée
```

## SECTION 6 : PATTERNS ET RÉGIMES

### Détection de patterns
**R6-A** : Patterns détectés sur observations **brutes + normalisées localement**

**Méthode de normalisation par défaut** :
```python
# Robust scaling (insensible aux valeurs extrêmes)
normalized = (x - median) / IQR  # IQR = Q3 - Q1
```
La normalisation robuste n’existe que dans le cadre de la détection de patterns.
Elle ne produit aucune donnée persistée.

**Patterns détectés** :
1. **NON_DISCRIMINANT** : Toutes observations normalisées < seuil
2. **OVER_DISCRIMINANT** : Toutes observations normalisées > seuil  
3. **D_CORRELATED** : Variance inter-d_encoding_id > seuil
4. **MODIFIER_CORRELATED** : Variance inter-modifier_id > seuil
5. **SEED_UNSTABLE** : Variance inter-seeds > seuil (à D/modifier fixés)
6. **SYSTEMATIC_ANOMALY** : Fraction observations |normalized| > seuil dépasse ratio
7. **CONTEXTUAL_BEHAVIOR** : Combinaison D_CORRELATED ou MODIFIER_CORRELATED

### Classification des régimes
**Taxonomie complète des régimes** :

```python
# Régimes de CONSERVATION (sains)
CONSERVES_SYMMETRY    # Test SYM-*
CONSERVES_NORM        # Test UNIV-*, SPE-*
CONSERVES_PATTERN     # Test PAT-*
CONSERVES_TOPOLOGY    # Test TOP-*
CONSERVES_GRADIENT    # Test SPA-*
CONSERVES_SPECTRUM    # Test GRA-*

# RÉGIMES PATHOLOGIQUES
NUMERIC_INSTABILITY   # Valeurs infinies/NaN
OSCILLATORY_UNSTABLE  # Oscillations non amorties
TRIVIAL               # Collapse vers zéro/constante
DEGRADING             # Dégradation progressive

# AUTRES RÉGIMES
SATURATES_HIGH        # Saturation haute (tanh-like)
SATURATES_LOW         # Saturation basse
UNCATEGORIZED         # Non classifiable

# QUALIFICATIF MULTIMODAL
MIXED::{régime_base}  # Bimodalité détectée (ex: MIXED::CONSERVES_NORM)
```

**R6-B** : Régime déterminé par `regime_utils.classify_regime()` basé sur :
- Métriques statistiques
- Signature dynamique
- Distribution timeline
- Dispersion inter-runs
- Type de test

## SECTION 6.1 : STRATIFICATION STABLE/EXPLOSIF

**Critère** : Présence valeurs > threshold dans projections exploitées

**Seuil par défaut** : `1e50` (5 décades au-dessus de P90 typique)

**Configurable via verdict_config.yaml** :
```yaml
stratification:
  explosion_threshold: 1.0e50  # Seuil détection explosions
```

**Projections inspectées** :
- statistics : initial, final, mean, max
- evolution : slope, relative_change

## SECTION 6.2 : RELATION PATTERNS ↔ VERDICTS

**R6.2-A** : Patterns = observations factuelles détectées
**R6.2-B** : Verdicts = décisions humaines post-rapports (HORS périmètre logiciel)

**Pipeline actuel** :
1. VerdictEngine détecte PATTERNS (D_CORRELATED, etc.)
2. VerdictReporter génère RAPPORTS (analysis_complete.json, summary.txt)
3. HUMAIN lit rapports → décide VERDICT (SURVIVES[R0], WIP[R0-open])
4. HUMAIN met à jour gamma_catalog.md avec STATUS

**Clarification terminologique** :
- Module nommé `verdict_engine` est legacy → devrait être `pattern_engine`
- Flag `--verdict` signifie "génération rapports patterns" (pas verdict décisionnel)


## SECTION 7 : PROFILING 

Toute fonction de classement est descriptive, jamais décisionnelle.

### Module `profiling_common.py`
**Responsabilité** : Profiling comportemental tous axes (gamma, modifier, encoding, test)

**Emplacement** : `tests/utilities/UTIL/profiling_common.py`  

**Dépendances** :
- `timeline_utils` : Événements dynamiques
- `aggregation_utils` : Agrégations statistiques
- `regime_utils` : Classification régimes

**Fonctions principales** :
```python
profile_all_gammas(observations) → dict
compare_gammas_summary(profiles) → dict
```

**Format retour** : Conforme R5.1-A (structure unifiée)  

**R7-A**
Aucun module de profiling n’a le droit d’implémenter :
    -un calcul de signature dynamique,
    -un calcul PRC,
    -un ranking cross-tests
sans passer par profiling_common.py.

## SECTION 8 : GÉNÉRATION DE RAPPORTS

### Module `verdict_reporter.py`
**Responsabilité** : Compilation résultats + génération rapports (simplifié)

**Pipeline actuel** :
```python
generate_verdict_report(params_config_id, verdict_config_id, output_dir) → dict
  │
  ├─ 1. CHARGEMENT + DIAGNOSTICS
  │    ↓ data_loading.load_all_observations()
  │
  ├─ 2. ANALYSES GLOBALES STRATIFIÉES
  │    ↓ verdict_engine.analyze_regime() × 3 (GLOBAL, STABLE, EXPLOSIF)
  │
  ├─ 3. PROFILING MULTI-AXES (NOUVEAU)
  │    ↓ profiling_runner.run_all_profiling(observations, axes=['gamma', 'modifier'])
  │    Returns: {'gamma': {...}, 'modifier': {...}}
  │
  ├─ 4. FUSION RÉSULTATS
  │    ↓ _compile_all_analyses(global_patterns, all_profiling)
  │
  └─ 5. GÉNÉRATION RAPPORTS
       ↓ report_writers.write_summary_global(...)       # NOUVEAU
       ↓ report_writers.write_summary_gamma(...)        # NOUVEAU (split existant)
       ↓ report_writers.write_summary_modifier(...)     # NOUVEAU
       ↓ report_writers.write_profiling_reports(...)    # NOUVEAU (générique)
       ↓ report_writers.write_json(...)                 # CONSERVÉ
```

**Simplification** : Délégation orchestration profiling à `profiling_runner`

### Formats de sortie

**Structure rapports** :
```
reports/verdicts/TIMESTAMP_analysis_global/
├── summary_global.txt              # NOUVEAU - Synthèse unifiée
│
├── summaries/                      # NOUVEAU - Synthèses par axe
│   ├── summary_gamma.txt
│   ├── summary_modifier.txt
│   ├── summary_encoding.txt        # Futur
│   └── summary_tests.txt           # Futur
│
├── analysis_complete.json          # CONSERVÉ - Données complètes
│
├── profiles/                       # NOUVEAU - Profils par axe/entité
│   ├── gamma/
│   │   ├── GAM-001.json
│   │   └── ...
│   ├── modifier/
│   │   ├── M0.json
│   │   ├── M1.json
│   │   └── M2.json
│   └── encoding/                   # Futur
│
├── comparisons/                    # ÉTENDU
│   ├── gamma_by_test.json
│   ├── modifier_by_test.json       # NOUVEAU
│   ├── modifier_signatures.json    # NOUVEAU
│   └── cross_analysis.json         # Futur
│
└── diagnostics/                    # CONSERVÉ
    ├── degeneracy_report.json
    └── scale_outliers.json
```

### Structure `summary_global.txt` (nouveau)

Template synthèse unifiée :
```markdown
# SYNTHÈSE GLOBALE R0

## CONFIGURATION
- Params: {params_config_id}
- Verdict: {verdict_config_id}
- Scope: {n_gammas} gammas × {n_modifiers} modifiers × {n_encodings} encodings × {n_tests} tests

## GAMMAS - ÉLÉMENTS CLÉS
- [Top 3 conservation]
- [Top 3 pathologiques]
- [Timeline dominante]

## MODIFIERS - ÉLÉMENTS CLÉS
- [Baseline validé]
- [Modifiers perturbateurs]
- [Sensibilités contextuelles]

## TESTS - ÉLÉMENTS CLÉS
- [Tests discriminants]
- [Tests non-discriminants]
- [Tests instables]

## PATTERNS STRUCTURELS
- [D_CORRELATED]
- [MODIFIER_CORRELATED]
- [CONTEXTUAL]

## RECOMMANDATIONS R0
- [Liste actions suggérées humain]
```

**R8-A** : `summary_global.txt` ne doit contenir QUE éléments clés
- Maximum 30 lignes
- Références détails → `summaries/{axe}.txt`

**R8-B** : `summary_{axe}.txt` format libre
- Chaque axe profiling peut définir structure
- Généré par `report_writers.write_summary_{axe}()`

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

### Structure rapports (vérification)

**VÉRIFIER** structure actuelle conforme :
```
reports/verdicts/TIMESTAMP_analysis_global/
├── summary_global.txt
├── summaries/
│   ├── summary_test.txt        # Ordre : test en premier
│   ├── summary_gamma.txt
│   ├── summary_modifier.txt
│   └── summary_encoding.txt
├── profiles/
│   ├── test/
│   ├── gamma/
│   ├── modifier/
│   └── encoding/
└── comparisons/
    ├── rankings_*.json         # Tous rankings cross_profiling
    └── interactions_*.json     # Placeholders R0
```

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
### SECTION 11.5 **Structure proposée** (optimisation tokens)
```
1. BIBLIOTHÈQUE PERMANENTE (contexte systématique toujours disponible)
   ├── charter_6.0.md              # Cadre architectural + interdictions
   ├── r0_status.md                # Travail en cours (ce qui change)
   └── catalogs_index.md           # Index catalogues (pas contenu complet)

2. CATALOGUES (injection à la demande)
   ├── core_catalog.md
   ├── operators/gamma_catalog.md
   ├── D_encodings/d_encoding_catalog.md
   ├── modifiers/modifier_catalog.md
   ├── tests/tests_catalog.md
   └── tests/utilities/HUB_catalog.md, util_catalog.md, registries_catalog.md

3. SOURCES CODE (injection conversation si besoin)
   ├── tests/utilities/*.py
   ├── operators/gamma_hyp_*.py
   └── ...
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
---

**FIN CHARTER PRC 6.1**

**Version** : 6.1.0  
**Date** : 2025-01-15  
**Statut** : PRODUCTION  
**Basé sur** : Catalogues opérationnels, état fonctionnel réel  

Ce charter est la **référence opérationnelle unique** pour l'architecture PRC 6.1.  
Il décrit l'état **réel** du système tel qu'implémenté et documenté dans les catalogues.  
Toute extension DOIT se conformer strictement à ces spécifications.