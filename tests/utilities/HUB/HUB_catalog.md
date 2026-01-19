# HUB CATALOG

> Modules d'orchestration niveau système  
> Délégation stricte, zéro calcul inline  

**INTERDICTIONS CRITIQUES** :
- ❌ Implémenter calculs inline (déléguer UTIL/registries)
- ❌ Dupliquer code UTIL/PROFILING (extraction obligatoire)
- ❌ Modifier observations (lecture seule)
- ❌ Hardcoder listes entités (découverte dynamique)

## MODULES D'ORCHESTRATION

### test_engine.py
**Responsabilité** : Génération observations pures à partir de runs  
**Version** : 5.5  
**Architecture** : Moteur exécution tests avec détection événements dynamiques

**Fonctions principales** :
- `TestEngine.execute_test()` : Exécute test sur history, retourne observation
- `detect_dynamic_events()` : Détecte événements dynamiques (deviation, instability, etc.)
- `compute_event_sequence()` : Construit séquence temporelle événements

**Événements dynamiques détectés** (R0) :
- `deviation_onset` : |val - initial| > 10% initial
- `instability_onset` : |diff| > P90(diffs) × 10
- `oscillatory` : sign_changes > 10% iterations
- `saturation` : std(last_20%) / mean < 5%
- `collapse` : retour brutal à ~0

**Délégations** :
- Validation specs → `RegistryManager`
- Calculs métriques → `registries/*`
- Post-processing → `post_processors.py`
- Config → `config_loader.py`

**Notes** :
- Zéro validation sémantique (observation pure)
- Cache computations (performances)
- Gestion erreurs granulaire (skip iterations invalides)
- Traçabilité complète via `exec_id`

### verdict_engine.py
**Responsabilité** : Analyses statistiques multi-facteurs (variance, interactions, patterns)  
**Version** : 5.5  
**Architecture** : Refactorisée Phase 2 (délégation stricte)

**Fonctions principales** :
- `analyze_marginal_variance()` : Variance marginale (η² par facteur)
- `analyze_oriented_interactions()` : Interactions orientées A|B ≠ B|A
- `analyze_metric_discrimination()` : Détection métriques non discriminantes
- `analyze_metric_correlations()` : Corrélations fortes entre métriques
- `interpret_patterns()` : Synthèse patterns globaux + par gamma
- `decide_verdict()` : Décision verdict (GLOBAL + PAR GAMMA)
- `analyze_regime()` : Pipeline complet sur strate (GLOBAL/STABLE/EXPLOSIF)
- `compute_verdict()` : Pipeline principal (point d'entrée)

**Facteurs analysés** :
```python
FACTORS = [
    'gamma_id',
    'd_encoding_id',
    'modifier_id',
    'seed',
    'test_name'
    # params_config_id EXCLU (trop corrélé test_name)
]
```

**Projections numériques** :
```python
PROJECTIONS = [
    'value_final',
    'value_mean',
    'slope',
    'volatility',
    'relative_change'
]
```

**Seuils testabilité** :
- MIN_SAMPLES_PER_GROUP = 2
- MIN_GROUPS = 2
- MIN_TOTAL_SAMPLES = 10

**Analyses effectuées** :
1. **Variance marginale** : η² (eta-squared) par facteur
2. **Interactions orientées** : Paires (A, B) testées séparément (permutations)
3. **Discrimination** : CV < 0.1 → non discriminant
4. **Corrélations** : Spearman > 0.8 → redondance

**Patterns détectés** :
- `marginal_dominant` : Facteur variance_ratio > 0.5
- `oriented_interactions` : VR(A|B) >> VR(A) marginal
- `non_discriminant` : Métriques CV < 0.1
- `redundant` : Paires corrélées > 0.8

**Délégations strictes** :
- I/O → `data_loading.py`
- Filtrage → `statistical_utils.py`
- Stratification → `regime_utils.py`
- Calculs η² → `statistical_utils.compute_eta_squared()`
- Tests Kruskal-Wallis → `statistical_utils.kruskal_wallis_test()`

**Notes** :
- Phase 2.1 : Correction calcul η² via fonction utilitaire
- Phase 2.2 : Critères testabilité interactions renforcés
- Stratification parallèle : 3 analyses (GLOBAL, STABLE, EXPLOSIF)
- Génération rapports stratifiés

### verdict_reporter.py
**Responsabilité** : Orchestration pipeline rapports complets R0  
**Version** : Refactorisée Phase 2.3  
**Architecture** : Coordination verdict_engine + profiling_runner + report_writers

**Fonction principale** :
- `generate_verdict_report()` : Pipeline complet 6 étapes

**Pipeline** :
```
1. Chargement observations (data_loading)
2. Diagnostics numériques (statistical_utils)
3. Analyses globales stratifiées (verdict_engine)
4. Profiling gamma (profiling_runner)  # ← NOUVEAU : profiling multi-axes
5. Fusion résultats (local - orchestration)
6. Génération rapports (report_writers)
```

**Rapports générés** :
- `metadata.json` : Métadonnées complètes
- `summary.txt` : Rapport humain principal (ENRICHI R0+)
- `gamma_profiles.json` + `.csv` : Profils comportementaux
- `comparisons.json` : Rankings inter-gammas
- `structural_patterns.json` : Analyses globales
- `diagnostics.json` : Diagnostics techniques
- `marginal_variance_{regime}.csv` : Analyses stratifiées (GLOBAL/STABLE/EXPLOSIF)

**Délégations strictes** :
- I/O → `data_loading.load_all_observations()`
- Filtrage → `statistical_utils.filter_numeric_artifacts()`
- Diagnostics → `statistical_utils.generate_degeneracy_report()`, `diagnose_scale_outliers()`
- Stratification → `regime_utils.stratify_by_regime()`
- Analyses → `verdict_engine.analyze_regime()`
- Profiling → `profiling_runner.run_all_profiling()`  # ← NOUVEAU
- Formatage → `report_writers.write_*()`

**Fonctions locales** (spécifiques verdict) :
- `_compile_metadata()` : Compilation métadonnées rapport
- `_format_gamma_profiles()` : Formatage structure gamma_profiles
- `_compile_structural_patterns()` : Compilation patterns globaux
- `_write_summary_report()` : Écriture rapport humain (partiellement délégué)
- `_write_gamma_profiles()` : Écriture JSON + CSV

**Notes** :
- Architecture refactorisée Phase 2.3 (délégation maximale)
- Posture non gamma-centrique (analyses globales + profils individuels)
- Rapports multi-formats (JSON, TXT, CSV)
- Traçabilité complète (configs, timestamps)

### profiling_runner.py
**Responsabilité** : Orchestration profiling multi-axes avec découverte automatique  
**Version** : 6.1  
**Architecture** : Délégation stricte profiling_common + cross_profiling

**Fonctions principales** :
- `run_all_profiling()` : Exécute profiling tous axes demandés
- `run_profiling_single_axis()` : Profiling un seul axe (helper)
- `discover_profiling_axes()` : Découverte axes disponibles
- `get_entity_profile()` : Extrait profil entité spécifique
- `get_test_profile_for_entity()` : Extrait profil test pour entité

**Axes profiling** (ordre DEFAULT_AXES_ORDER) :
1. `test` : Profiling tests (avec enrichissement discriminant_power)
2. `gamma` : Profiling gammas
3. `modifier` : Profiling modifiers
4. `encoding` : Profiling encodings

**Délégations strictes** :
- Profiling individuel → `profiling_common.profile_all_{axis}()`
- Comparaisons → `profiling_common.compare_{axis}_summary()`
- Enrichissement test → `cross_profiling.compute_all_discriminant_powers()`

**Notes** :
- Découverte automatique axes (R0 : hardcodé 4 axes, R1+ : introspection)
- Enrichissement spécifique axe test (discriminant_power)
- Format retour unifié strict (R7.1-A)
- Zéro calcul inline (orchestration pure)
- Helpers extraction pour reporting

## DÉPENDANCES AUTORISÉES

**HUB peut importer** :
```
HUB → UTIL (profiling_common, cross_profiling, data_loading, etc.)
HUB → registries (via RegistryManager)
HUB → config_loader
```

**INTERDICTIONS** :
```
❌ HUB → core (séparation stricte)
❌ HUB → operators, D_encodings, modifiers, tests (séparation stricte)
```

## RÈGLES ARCHITECTURALES

**R-HUB-1** : HUB orchestre, ne calcule jamais inline

**R-HUB-2** : Toute logique calcul → UTIL ou registries (extraction obligatoire)

**R-HUB-3** : Pas de duplication code UTIL/PROFILING (vérifier tables "AVANT DE CODER")

**R-HUB-4** : Observations lecture seule (jamais modifier)

**R-HUB-5** : Découverte dynamique entités (pas hardcoding listes)

**R-HUB-6** : Format retour unifié profiling (R7.1-A strict)

**R-HUB-7** : Délégation explicite documentée (via imports)

## WORKFLOW TYPIQUE

**Génération rapports complets** :
```
verdict_reporter.generate_verdict_report()
  ↓
  ├→ data_loading.load_all_observations()
  ├→ statistical_utils.filter_numeric_artifacts()
  ├→ verdict_engine.analyze_regime() × 3 (GLOBAL, STABLE, EXPLOSIF)
  ├→ profiling_runner.run_all_profiling()
  │    ↓
  │    ├→ profiling_common.profile_all_{axis}()
  │    ├→ profiling_common.compare_{axis}_summary()
  │    └→ cross_profiling.compute_all_discriminant_powers() (si test)
  └→ report_writers.write_*()
```

**Exécution test unique** :
```
TestEngine.execute_test()
  ↓
  ├→ config_loader.load()
  ├→ RegistryManager.validate_computation_spec()
  ├→ registries.{category}.{function}() × N snapshots
  ├→ post_processors.{method}()
  ├→ detect_dynamic_events()
  └→ compute_event_sequence()
```

## EXTENSIONS FUTURES

**Checklist ajout nouveau module HUB** :
- [ ] Définir responsabilité unique (orchestration stricte)
- [ ] Identifier dépendances UTIL/registries nécessaires
- [ ] Documenter délégations explicites
- [ ] Vérifier zéro calcul inline (extraction si besoin)
- [ ] Respecter format retour unifié (si profiling)
- [ ] Ajouter à ce catalogue
- [ ] Mettre à jour PRC_DEPENDENCY_RULES.md

**Exemples extensions acceptables** :
- ✅ Orchestrateur analyses temporelles (délègue UTIL)
- ✅ Reporter spécialisé (délègue report_writers)
- ✅ Profiling axes additionnels (délègue profiling_common)

**Exemples extensions REFUSÉES** :
- ❌ Module HUB avec calculs inline (violé délégation)
- ❌ Module HUB dupliquant UTIL (violé anti-duplication)
- ❌ Module HUB modifiant observations (violé lecture seule)

**FIN HUB CATALOG**