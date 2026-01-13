# Catalogue Utilities (Modules Utilitaires Tests) - R0 Annoté

## Vue d'ensemble

Modules de support pour l'exécution et l'analyse des tests PRC Charter 5.5.

**Architecture Globale Refactorisée** :
```
tests/utilities/
├── test_engine.py          # Exécution tests
├── applicability.py        # Validation applicabilité
├── config_loader.py        # Chargement configs YAML
├── discovery.py            # Discovery tests actifs
│
├── verdict_engine.py       # ⭐ HUB analyses globales (à refactoriser Phase 2)
├── gamma_profiling.py      # ⭐ HUB profiling gammas (à refactoriser Phase 2)
├── verdict_reporter.py     # ⭐ HUB orchestration rapports (à refactoriser Phase 2)
│
├── aggregation_utils.py    # ✅ Agrégations statistiques (nouveau)
├── data_loading.py         # ✅ I/O observations/DB (nouveau)
├── regime_utils.py         # ✅ Stratification/classification régimes (nouveau)
├── report_writers.py       # ✅ Formatage/écriture rapports (nouveau)
├── statistical_utils.py    # ✅ Outils statistiques réutilisables (nouveau)
├── timeline_utils.py       # ✅ Construction timelines (nouveau)
│
└── registries/             # Fonctions calcul métriques
    ├── registry_manager.py
    ├── algebra.py
    ├── graph.py
    ├── pattern.py
    ├── spatial.py
    ├── spectral.py
    ├── statistical.py
    └── topological.py
```

**LÉGENDE** :
- ⭐ **HUB** : Module orchestration (à refactoriser Phase 2 en délégant aux utils)
- ✅ **UTIL** : Module utilitaire spécialisé (nouvellement créé)
- 📦 **STABLE** : Module stable (pas de changement prévu)

---

## 🆕 Modules Utilitaires Spécialisés (Nouveaux)

### aggregation_utils.py ✅

**Responsabilité** : Agrégations statistiques inter-runs

**Fonctions** :

#### `aggregate_summary_metrics(observations, metric_name) → dict`
Agrège métriques statistiques inter-runs (médiane, q1, q3, cv).

**Utilisé par** :
- `gamma_profiling.py::aggregate_summary_metrics()` ← **DÉLÉGUER ICI (Phase 2)**

**Structure Retour** :
```python
{
    'final_value': {'median': float, 'q1': float, 'q3': float, 'mean': float, 'std': float},
    'initial_value': float,
    'mean_value': float,
    'cv': float
}
```

#### `aggregate_run_dispersion(observations, metric_name) → dict`
Calcule indicateurs multimodalité (IQR ratio, bimodal detection).

**Utilisé par** :
- `gamma_profiling.py::aggregate_run_dispersion()` ← **DÉLÉGUER ICI (Phase 2)**

**Structure Retour** :
```python
{
    'final_value_iqr_ratio': float,
    'cv_across_runs': float,
    'bimodal_detected': bool
}
```

**Notes** :
- Seuil bimodal : IQR_ratio > 3.0 (heuristique R0)
- Protection division par zéro : max(Q1, 1e-10)

---

### data_loading.py ✅

**Responsabilité** : I/O observations depuis DBs

**Fonctions** :

#### `load_all_observations(params_config_id) → List[dict]`
Charge observations SUCCESS avec métadonnées runs (double connexion DB).

**Actuellement dans** :
- `verdict_engine.py::load_all_observations()` ← **DÉJÀ DÉPLACÉ ICI**

**Double Connexion** :
- `prc_r0_results.db` : TestObservations (observation_data, status)
- `prc_r0_raw.db` : Executions (gamma_id, d_encoding_id, modifier_id, seed)

#### `observations_to_dataframe(observations) → DataFrame`
Convertit observations → DataFrame normalisé pour analyses stats.

**Actuellement dans** :
- `verdict_engine.py::observations_to_dataframe()` ← **DÉJÀ DÉPLACÉ ICI**

**Projections Extraites** :
- Numériques : value_final, value_initial, value_mean, value_std, slope, volatility, relative_change
- Catégorielles : transition, trend

#### `cache_observations(observations, cache_path)` & `load_cached_observations(cache_path)`
Cache/chargement observations (optimisation future).

**État** : Implémenté mais non utilisé (futur)

---

### regime_utils.py ✅

**Responsabilité** : Stratification et classification régimes

**Fonctions** :

#### `stratify_by_regime(observations, threshold=1e50) → Tuple[List, List]`
Stratifie observations en régimes stable/explosif.

**Actuellement dans** :
- `verdict_engine.py::stratify_by_regime()` ← **DÉJÀ DÉPLACÉ ICI**

**Critère** : Présence valeurs >threshold dans projections exploitées.

#### `classify_regime(metrics, dynamic_sig, timeline_dist, dispersion, test_name) → str`
Classification régime R0 avec régimes SPÉCIFIQUES.

**Actuellement dans** :
- `gamma_profiling.py::classify_regime()` ← **DÉJÀ DÉPLACÉ ICI**

**Régimes Retournés** :
- Conservation : CONSERVES_SYMMETRY, CONSERVES_NORM, CONSERVES_PATTERN, CONSERVES_TOPOLOGY, CONSERVES_GRADIENT, CONSERVES_SPECTRUM
- Pathologies : NUMERIC_INSTABILITY, OSCILLATORY_UNSTABLE, TRIVIAL, DEGRADING
- Autres : SATURATES_HIGH, UNCATEGORIZED
- Qualificatif : MIXED::{régime_base} si bimodal

#### `detect_conserved_property(test_name) → str`
Détermine propriété conservée selon préfixe test.

**Mapping** :
```python
SYM-* → CONSERVES_SYMMETRY
SPE-*, UNIV-* → CONSERVES_NORM
PAT-* → CONSERVES_PATTERN
TOP-* → CONSERVES_TOPOLOGY
GRA-* → CONSERVES_GRADIENT
SPA-* → CONSERVES_SPECTRUM
Autre → CONSERVES_PROPERTY (fallback)
```

#### `extract_conserved_properties(profile) → List[str]`
Extrait propriétés conservées depuis profil gamma.

**Actuellement dans** :
- `verdict_reporter.py::_extract_conserved_properties()` ← **DÉJÀ DÉPLACÉ ICI**

#### `get_regime_family(regime) → str`
Retourne famille d'un régime (conservation, pathology, saturation, other).

**Utilise** : Constante `REGIME_TAXONOMY` (référence taxonomie complète)

---

### report_writers.py ✅

**Responsabilité** : Formatage et écriture rapports structurés

**Fonctions** :

#### `write_json(data, filepath, indent=2)`
Écrit dict → JSON formaté (avec conversion tuples → strings).

**Utilise** : `_make_json_serializable(obj)` pour sérialisation sûre

#### `write_header(f, title, width=80, char='=')`
Écrit header section (= bordures).

#### `write_subheader(f, title, width=80, char='-')`
Écrit sous-header section (- bordures).

#### `write_key_value(f, key, value, indent=0)`
Écrit paire clé-valeur formatée.

#### `write_regime_synthesis(f, gamma_profiles, width=80)`
Écrit synthèse régimes transversale (aggregation tous gammas × tests).

**Actuellement dans** :
- `verdict_reporter.py::_write_regime_synthesis()` ← **DÉJÀ DÉPLACÉ ICI**

**Structure** :
```
CONSERVATION (régimes sains):
  CONSERVES_SYMMETRY : 120 (12.5%)
  ...
PATHOLOGIES:
  NUMERIC_INSTABILITY : 45 (4.7%)
  ...
MULTIMODALITÉ (MIXED::X):
  ...
```

#### `write_dynamic_signatures(f, gamma_profiles, width=80)`
Écrit signatures dynamiques par gamma (timelines dominantes).

**Actuellement dans** :
- `verdict_reporter.py::_write_dynamic_signatures()` ← **DÉJÀ DÉPLACÉ ICI**

#### `write_comparisons_enriched(f, comparisons, gamma_profiles, width=80)`
Écrit comparaisons enrichies avec contexte propriétés.

**Actuellement dans** :
- `verdict_reporter.py::_write_comparisons_enriched()` ← **DÉJÀ DÉPLACÉ ICI**

**Structure** :
```
SYMÉTRIE:
  SYM-001:
    Meilleur : GAM-001
    Pire     : GAM-013
    Classement : GAM-001, GAM-004, GAM-007...
```

#### `extract_conserved_properties_display(profile) → List[str]`
Extrait propriétés conservées (pour affichage).

**Actuellement dans** :
- `verdict_reporter.py::_extract_conserved_properties()` ← **DÉJÀ DÉPLACÉ ICI**

#### `write_consultation_footer(f, width=80, char='=')`
Écrit footer avec fichiers consultation.

---

### statistical_utils.py ✅

**Responsabilité** : Outils statistiques réutilisables

**Fonctions** :

#### `compute_eta_squared(groups) → Tuple[float, float, float]`
Calcule eta-squared (η²) : proportion variance expliquée.

**Actuellement dans** :
- `verdict_engine.py::compute_eta_squared()` ← **DÉJÀ DÉPLACÉ ICI**

**Formule** : η² = SSB / (SSB + SSW)

**Retourne** : (eta2, ssb, ssw)

#### `kruskal_wallis_test(groups) → Tuple[float, float]`
Test Kruskal-Wallis avec gestion erreurs.

**Nouveau** : Wrapper sécurisé autour `scipy.stats.kruskal()`

#### `is_numeric_valid(obs) → bool`
Détecte artefacts numériques (inf/nan) dans projections exploitées.

**Actuellement dans** :
- `verdict_engine.py::is_numeric_valid()` ← **DÉJÀ DÉPLACÉ ICI**

#### `filter_numeric_artifacts(observations) → Tuple[List, dict]`
Filtre observations avec artefacts numériques.

**Actuellement dans** :
- `verdict_engine.py::filter_numeric_artifacts()` ← **DÉJÀ DÉPLACÉ ICI**

**Log Rejets** : Stats par test pour traçabilité

#### `diagnose_numeric_degeneracy(obs) → List[str]`
Détecte dégénérescences numériques sur projections exploitées.

**Actuellement dans** :
- `verdict_engine.py::diagnose_numeric_degeneracy()` ← **DÉJÀ DÉPLACÉ ICI**

**Flags** : INFINITE_PROJECTION, NAN_PROJECTION, EXTREME_MAGNITUDE

#### `generate_degeneracy_report(observations) → dict`
Génère rapport diagnostique dégénérescences.

**Actuellement dans** :
- `verdict_engine.py::generate_degeneracy_report()` ← **DÉJÀ DÉPLACÉ ICI**

#### `print_degeneracy_report(report)`
Affiche rapport diagnostique dégénérescences (stdout).

**Actuellement dans** :
- `verdict_engine.py::print_degeneracy_report()` ← **DÉJÀ DÉPLACÉ ICI**

#### `diagnose_scale_outliers(observations) → dict`
Détecte ruptures d'échelle relatives par contexte.

**Actuellement dans** :
- `verdict_engine.py::diagnose_scale_outliers()` ← **DÉJÀ DÉPLACÉ ICI**

**Critère** : Valeur > P90 + 5 décades (facteur 1e5)

#### `print_scale_outliers_report(report)`
Affiche rapport ruptures d'échelle (stdout).

**Actuellement dans** :
- `verdict_engine.py::print_scale_outliers_report()` ← **DÉJÀ DÉPLACÉ ICI**

---

### timeline_utils.py ✅

**Responsabilité** : Construction timelines dynamiques compositionnels

**Configuration Globale** :
```python
TIMELINE_THRESHOLDS = {
    'early': 0.20,  # onset < 20% durée
    'mid':   0.60,  # 20% ≤ onset ≤ 60%
    'late':  0.60   # onset > 60%
}
```

**Fonctions** :

#### `classify_timing(onset_relative) → str`
Classifie timing selon seuils globaux.

**Actuellement dans** :
- `gamma_profiling.py::classify_timing()` ← **DÉJÀ DÉPLACÉ ICI**

**Retourne** : 'early' | 'mid' | 'late'

#### `compute_timeline_descriptor(sequence, sequence_timing_relative, oscillatory_global) → dict`
Génère descriptor timeline compositionnel.

**Actuellement dans** :
- `gamma_profiling.py::compute_timeline_descriptor()` ← **DÉJÀ DÉPLACÉ ICI**

**Structure Retour** :
```python
{
    'phases': [
        {'event': 'deviation', 'timing': 'early', 'onset_relative': 0.05},
        {'event': 'saturation', 'timing': 'late', 'onset_relative': 0.80}
    ],
    'timeline_compact': 'early_deviation_then_saturation',
    'n_phases': 2,
    'oscillatory_global': False
}
```

**Formats** :
- Aucun événement → 'no_significant_dynamics'
- 1 événement → '{timing}_{event}_only'
- 2+ événements → '{timing1}_{event1}_then_{event2}'
- Oscillatoire global → préfixe 'oscillatory_'

#### `extract_dynamic_events(observation, metric_name) → dict`
Extrait événements dynamiques depuis observation.

**Actuellement dans** :
- `gamma_profiling.py::extract_dynamic_events()` ← **DÉJÀ DÉPLACÉ ICI**

**Lit depuis** : `observation_data['dynamic_events'][metric_name]`

#### `extract_metric_timeseries(observation, metric_name) → Tuple[List, bool]`
Extrait série temporelle avec marqueur fallback.

**Actuellement dans** :
- `gamma_profiling.py::extract_metric_timeseries()` ← **DÉJÀ DÉPLACÉ ICI**

**Retourne** : (values, is_fallback)

**Fallback** : linspace(initial, final) si timeseries absent

---

## ⭐ Modules HUB (À Refactoriser Phase 2)

### verdict_engine.py ⭐

**Responsabilité Actuelle** : Analyses statistiques globales multi-facteurs

**État** : Hub monolithique (à décomposer Phase 2)

**Fonctions À Garder** :
```python
# === ANALYSES STATISTIQUES (cœur métier) ===
analyze_marginal_variance(df, factors, projections) → DataFrame
  ↓ Utilise : statistical_utils.compute_eta_squared()
  ↓ Utilise : statistical_utils.kruskal_wallis_test()

analyze_oriented_interactions(df, factors, projections, marginal) → DataFrame
  ↓ Utilise : statistical_utils.compute_eta_squared()
  ↓ Utilise : statistical_utils.kruskal_wallis_test()

analyze_metric_discrimination(df, projections) → DataFrame

analyze_metric_correlations(df, threshold=0.8) → DataFrame

# === SYNTHÈSE (pattern detection) ===
interpret_patterns(df, marginal, interactions, discrimination, correlations) 
  → (patterns_global, patterns_by_gamma)

decide_verdict(patterns_global, patterns_by_gamma) 
  → (verdict, reason, verdicts_by_gamma)

# === PIPELINE RÉGIME (orchestration) ===
analyze_regime(observations, regime_name, params_config_id, verdict_config_id) → dict
  ↓ Délègue : data_loading.observations_to_dataframe()
  ↓ Délègue : analyze_marginal_variance(), analyze_oriented_interactions(), etc.
  ↓ Délègue : interpret_patterns(), decide_verdict()
```

**Fonctions DÉPLACÉES** (✅ annotations pour Phase 2) :
```python
# === I/O & STRUCTURATION (→ data_loading.py) ===
load_all_observations(params_config_id) 
  ✅ DÉPLACÉ → data_loading.py

observations_to_dataframe(observations) 
  ✅ DÉPLACÉ → data_loading.py

# === FILTRAGE & DIAGNOSTICS (→ statistical_utils.py) ===
is_numeric_valid(obs) 
  ✅ DÉPLACÉ → statistical_utils.py

filter_numeric_artifacts(observations) 
  ✅ DÉPLACÉ → statistical_utils.py

diagnose_numeric_degeneracy(obs) 
  ✅ DÉPLACÉ → statistical_utils.py

generate_degeneracy_report(observations) 
  ✅ DÉPLACÉ → statistical_utils.py

print_degeneracy_report(report) 
  ✅ DÉPLACÉ → statistical_utils.py

diagnose_scale_outliers(observations) 
  ✅ DÉPLACÉ → statistical_utils.py

print_scale_outliers_report(report) 
  ✅ DÉPLACÉ → statistical_utils.py

# === STRATIFICATION (→ regime_utils.py) ===
stratify_by_regime(observations, threshold=1e50) 
  ✅ DÉPLACÉ → regime_utils.py

# === VARIANCE (→ statistical_utils.py) ===
compute_eta_squared(groups) 
  ✅ DÉPLACÉ → statistical_utils.py

# === RAPPORTS (→ report_writers.py ou verdict_reporter.py) ===
generate_stratified_report(...) 
  ⚠️ À REFACTORISER Phase 2 (déléguer à report_writers.py)
```

**Plan Refactorisation Phase 2** :
1. Importer fonctions depuis utils dans verdict_engine
2. Remplacer appels locaux par imports
3. Supprimer code dupliqué
4. Garder uniquement logique analyse métier

---

### gamma_profiling.py ⭐

**Responsabilité Actuelle** : Profiling comportemental gammas individuels

**État** : Hub semi-refactorisé (plusieurs fonctions déjà déplacées)

**Fonctions À Garder** :
```python
# === PROFILING (cœur métier) ===
profile_test_for_gamma(observations, test_name, gamma_id) → dict
  ↓ Utilise : aggregation_utils.aggregate_summary_metrics()
  ↓ Utilise : aggregation_utils.aggregate_run_dispersion()
  ↓ Utilise : aggregate_dynamic_signatures() [LOCAL pour l'instant]
  ↓ Utilise : regime_utils.classify_regime()
  ↓ Utilise : compute_prc_profile() [LOCAL]

profile_all_gammas(observations) → dict

rank_gammas_by_test(profiles, test_name, criterion) → List[Tuple]

compare_gammas_summary(profiles) → dict
  ↓ Utilise : regime_utils.extract_conserved_properties()
```

**Fonctions DÉPLACÉES** (✅ annotations pour Phase 2) :
```python
# === TIMELINES (→ timeline_utils.py) ===
classify_timing(onset_relative) 
  ✅ DÉPLACÉ → timeline_utils.py

compute_timeline_descriptor(sequence, timing_relative, oscillatory) 
  ✅ DÉPLACÉ → timeline_utils.py

extract_dynamic_events(observation, metric_name) 
  ✅ DÉPLACÉ → timeline_utils.py

extract_metric_timeseries(observation, metric_name) 
  ✅ DÉPLACÉ → timeline_utils.py

# === AGRÉGATIONS (→ aggregation_utils.py) ===
aggregate_summary_metrics(observations, metric_name) 
  ✅ DÉPLACÉ → aggregation_utils.py

aggregate_run_dispersion(observations, metric_name) 
  ✅ DÉPLACÉ → aggregation_utils.py

# === RÉGIMES (→ regime_utils.py) ===
classify_regime(metrics, dynamic_sig, timeline_dist, dispersion, test_name) 
  ✅ DÉPLACÉ → regime_utils.py
```

**Fonctions LOCALES** (à évaluer Phase 2) :
```python
# === AGRÉGATIONS SPÉCIALISÉES (candidat aggregation_utils ?) ===
aggregate_dynamic_signatures(observations, metric_name) → dict
  # Actuellement local
  # Phase 2 : Évaluer déplacement vers aggregation_utils
  ↓ Utilise : timeline_utils.compute_timeline_descriptor()

# === PROFIL PRC (cœur gamma_profiling) ===
compute_prc_profile(metrics, dynamic_sig, timeline_dist, dispersion, n_runs, n_valid, test_name) → dict
  # Reste local (spécifique gamma profiling)
  ↓ Utilise : regime_utils.classify_regime()
```

**Plan Refactorisation Phase 2** :
1. Importer fonctions depuis utils dans gamma_profiling
2. Évaluer déplacement `aggregate_dynamic_signatures` → aggregation_utils
3. Garder `compute_prc_profile` local (cœur métier profiling)

---

### verdict_reporter.py ⭐

**Responsabilité Actuelle** : Orchestration génération rapports R0

**État** : Hub orchestration (délègue déjà partiellement)

**Pipeline Actuel** :
```python
generate_verdict_report(params_config_id, verdict_config_id, output_dir) → dict
  │
  ├─ 1. CHARGEMENT + DIAGNOSTICS
  │    ↓ data_loading.load_all_observations()
  │    ↓ statistical_utils.filter_numeric_artifacts()
  │    ↓ statistical_utils.generate_degeneracy_report()
  │    ↓ statistical_utils.diagnose_scale_outliers()
  │
  ├─ 2. ANALYSES GLOBALES STRATIFIÉES
  │    ↓ regime_utils.stratify_by_regime()
  │    ↓ verdict_engine.analyze_regime() × 3 (GLOBAL, STABLE, EXPLOSIF)
  │
  ├─ 3. PROFILING GAMMA
  │    ↓ gamma_profiling.profile_all_gammas()
  │    ↓ gamma_profiling.compare_gammas_summary()
  │
  ├─ 4. FUSION RÉSULTATS
  │    ↓ _compile_metadata() [LOCAL]
  │    ↓ _format_gamma_profiles() [LOCAL]
  │    ↓ _compile_structural_patterns() [LOCAL]
  │
  └─ 5. GÉNÉRATION RAPPORTS
       ↓ _write_metadata() [LOCAL → report_writers ?]
       ↓ _write_summary_report() [LOCAL → report_writers]
       ↓ _write_gamma_profiles() [LOCAL → report_writers]
       ↓ _write_comparisons() [LOCAL → report_writers]
       ↓ _write_structural_patterns() [LOCAL → report_writers]
       ↓ _write_diagnostics() [LOCAL → report_writers]
```

**Fonctions LOCALES** (à évaluer Phase 2) :
```python
# === COMPILATION (candidat nouveau module ?) ===
_compile_metadata(...) → dict
  # Phase 2 : Garder local ou déplacer vers report_writers

_format_gamma_profiles(gamma_profiles) → dict
  # Phase 2 : Candidat report_writers

_compile_structural_patterns(results_global, results_stable, results_explosif) → dict
  # Phase 2 : Garder local (spécifique verdict)

# === ÉCRITURE (→ report_writers.py Phase 2) ===
_write_metadata(report_dir, metadata)
  ⚠️ À DÉPLACER → report_writers.write_json()

_write_summary_report(report_dir, results)
  ⚠️ À DÉCOMPOSER Phase 2
  ↓ Délègue : report_writers.write_header()
  ↓ Délègue : report_writers.write_regime_synthesis()
  ↓ Délègue : report_writers.write_dynamic_signatures()
  ↓ Délègue : report_writers.write_comparisons_enriched()

_write_gamma_profiles(report_dir, gamma_profiles)
  ⚠️ À DÉPLACER → report_writers.py

_write_comparisons(report_dir, comparisons)
  ⚠️ À DÉPLACER → report_writers.write_json()

_write_structural_patterns(report_dir, structural)
  ⚠️ À DÉPLACER → report_writers.write_json()

_write_diagnostics(report_dir, diagnostics)
  ⚠️ À DÉPLACER → report_writers.write_json()
  ↓ Utilise : report_writers._make_json_serializable()
```

**Fonctions DÉPLACÉES** (✅ annotations pour Phase 2) :
```python
# === HELPERS SUMMARY (→ report_writers.py) ===
_write_regime_synthesis(f, gamma_profiles) 
  ✅ DÉPLACÉ → report_writers.write_regime_synthesis()

_write_dynamic_signatures(f, gamma_profiles) 
  ✅ DÉPLACÉ → report_writers.write_dynamic_signatures()

_extract_conserved_properties(profile) 
  ✅ DÉPLACÉ → regime_utils.extract_conserved_properties()
  ✅ DÉPLACÉ → report_writers.extract_conserved_properties_display()

_write_comparisons_enriched(f, comparisons, gamma_profiles) 
  ✅ DÉPLACÉ → report_writers.write_comparisons_enriched()

_make_json_serializable(obj) 
  ✅ DÉPLACÉ → report_writers.py (utilisé par write_json)
```

**Plan Refactorisation Phase 2** :
1. Déléguer toutes écritures fichiers → report_writers
2. Garder logique compilation/orchestration dans verdict_reporter
3. Simplifier `_write_summary_report()` en composition de writers

---

## 📦 Modules Stables (Pas de Changement Prévu)

### test_engine.py 📦

**Classe** : `TestEngine`

**Responsabilité** : Moteur d'exécution tests avec registries

**État** : Stable (fonctionnel complet)

**Méthodes Clés** :
- `execute_test(test_module, run_metadata, history, params_config_id)` → dict

**Fonctions Détection Dynamique** :
- `detect_dynamic_events(values)` → dict
- `compute_event_sequence(events, n_iterations)` → dict

**Format Retour** :
```python
{
    'exec_id': int,
    'test_name': str,
    'status': 'SUCCESS' | 'ERROR',
    'statistics': {metric: {initial, final, mean, std, ...}},
    'evolution': {metric: {slope, volatility, transition, ...}},
    'dynamic_events': {metric: {events + sequence}},
    'timeseries': {metric: [val_0, ..., val_N]},
    'metadata': {...}
}
```

**Note** : Aucune fonction à déplacer (module autonome)

---

### applicability.py 📦

**Fonction** : `check(test_module, run_metadata)` → (bool, str)

**État** : Stable (validators extensibles)

**Validators Disponibles** :
- `requires_rank`, `requires_square`, `allowed_d_types`, `requires_even_dimension`, `minimum_dimension`

**Extensibilité** : `add_validator(name, validator)`

---

### config_loader.py 📦

**Classe** : `ConfigLoader` (singleton via `get_loader()`)

**État** : Stable (architecture configs fixée)

**Méthodes** :
- `load(config_type, config_id, test_id=None)` → dict
- `list_available(config_type)` → dict
- `clear_cache()`

**Stratégie Fusion** : Global + specific (merge récursif dicts)

---

### discovery.py 📦

**Fonction** : `discover_active_tests()` → Dict[str, module]

**État** : Stable (validation Charter 5.5)

**Validation Structure** : `validate_test_structure(module)`

**Attributs Requis** :
```python
REQUIRED_ATTRIBUTES = [
    'TEST_ID',           # Format CAT-NNN
    'TEST_CATEGORY',
    'TEST_VERSION',      # Doit être '5.5'
    'APPLICABILITY_SPEC',
    'COMPUTATION_SPECS'  # 1-5 métriques
]
```

---

### registries/ 📦

**registry_manager.py** : Gestion centralisée registries

**Registries Disponibles** :
- AlgebraRegistry (`algebra`)
- GraphRegistry (`graph`)
- PatternRegistry (`pattern`)
- SpatialRegistry (`spatial`)
- SpectralRegistry (`spectral`)
- StatisticalRegistry (`statistical`)
- TopologicalRegistry (`topological`)

**État** : Stable (registries complets)

---

##