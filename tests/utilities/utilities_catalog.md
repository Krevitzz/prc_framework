# Catalogue Utilities (Modules Utilitaires Tests) - R0 Final Refactorisé

## Vue d'ensemble

Modules de support pour l'exécution et l'analyse des tests PRC Charter 5.5.

**Architecture Globale Refactorisée (Phase 2 Complète)** :
```
tests/utilities/
├── test_engine.py          # 📦 Exécution tests (stable)
├── applicability.py        # 📦 Validation applicabilité (stable)
├── config_loader.py        # 📦 Chargement configs YAML (stable)
├── discovery.py            # 📦 Discovery tests actifs (stable)
│
├── verdict_engine.py       # ✅ HUB analyses globales (REFACTORISÉ Phase 2.1)
├── gamma_profiling.py      # ✅ HUB profiling gammas (REFACTORISÉ Phase 2.2)
├── verdict_reporter.py     # ✅ HUB orchestration rapports (REFACTORISÉ Phase 2.3)
│
├── aggregation_utils.py    # 🆕 Agrégations statistiques
├── data_loading.py         # 🆕 I/O observations/DB
├── regime_utils.py         # 🆕 Stratification/classification régimes
├── report_writers.py       # 🆕 Formatage/écriture rapports
├── statistical_utils.py    # 🆕 Outils statistiques réutilisables
├── timeline_utils.py       # 🆕 Construction timelines
│
└── registries/             # 📦 Fonctions calcul métriques (stable)
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
- 📦 **STABLE** : Module stable (pas de changement Phase 2)
- ✅ **REFACTORISÉ** : Hub refactorisé (délègue aux utils)
- 🆕 **NOUVEAU** : Module utilitaire spécialisé (créé Phase 2)

---

## 🆕 Modules Utilitaires Spécialisés

### aggregation_utils.py

**Responsabilité** : Agrégations statistiques inter-runs

**Statut** : Nouveau module (extraction depuis gamma_profiling)

**Fonctions Publiques** :

#### `aggregate_summary_metrics(observations: List[dict], metric_name: str) → dict`
Agrège métriques statistiques inter-runs (médiane, q1, q3, cv).

**Structure Retour** :
```python
{
    'final_value': {
        'median': float,
        'q1': float,
        'q3': float,
        'mean': float,
        'std': float
    },
    'initial_value': float,
    'mean_value': float,
    'cv': float  # Coefficient variation (std/mean)
}
```

**Utilisé par** :
- `gamma_profiling.profile_test_for_gamma()` : Instrumentation niveau 3

**Notes** :
- Protection division par zéro : `+ 1e-10`
- `initial_value` = médiane (robuste outliers)
- Retourne `{}` si aucune valeur finale disponible

---

#### `aggregate_run_dispersion(observations: List[dict], metric_name: str) → dict`
Calcule indicateurs multimodalité inter-runs.

**Structure Retour** :
```python
{
    'final_value_iqr_ratio': float,  # Q3 / Q1
    'cv_across_runs': float,          # std / |mean|
    'bimodal_detected': bool          # IQR ratio > 3.0
}
```

**Utilisé par** :
- `gamma_profiling.profile_test_for_gamma()` : Diagnostic signature niveau 2

**Principe Bimodalité** :
- IQR ratio > 3.0 → 2+ modes distincts probables
- Exemple : Q1=0.1, Q3=0.5 → ratio=5.0 → bimodal
- Heuristique R0 (pas test statistique formel)

**Notes** :
- Seuil bimodal : 3.0 (heuristique R0)
- Protection division par zéro Q1 : `max(Q1, 1e-10)`
- Retourne valeurs 0.0/False si < 2 valeurs finales

---

### data_loading.py

**Responsabilité** : I/O observations depuis DBs + conversion DataFrame

**Statut** : Nouveau module (extraction depuis verdict_engine)

**Fonctions Publiques** :

#### `load_all_observations(params_config_id: str, db_results_path='...', db_raw_path='...') → List[dict]`
Charge observations SUCCESS avec métadonnées runs (double connexion DB).

**Double Connexion** :
- `prc_r0_results.db` : TestObservations (observation_data, status)
- `prc_r0_raw.db` : Executions (gamma_id, d_encoding_id, modifier_id, seed)

**Structure Retour** :
```python
[
    {
        'observation_id': int,
        'exec_id': int,
        'run_id': str,
        'gamma_id': str,
        'd_encoding_id': str,
        'modifier_id': str,
        'seed': int,
        'test_name': str,
        'params_config_id': str,
        'observation_data': dict,  # JSON parsé
        'computed_at': str
    },
    ...
]
```

**Utilisé par** :
- `verdict_engine.compute_verdict()` : Chargement initial
- `verdict_reporter.generate_verdict_report()` : Pipeline étape 1

**Raises** :
- `ValueError` : Si aucune observation SUCCESS trouvée

---

#### `observations_to_dataframe(observations: List[dict]) → pd.DataFrame`
Convertit observations → DataFrame normalisé pour analyses stats.

**Projections Extraites** :
- **Numériques** : value_final, value_initial, value_mean, value_std, value_min, value_max, slope, volatility, relative_change
- **Catégorielles** : transition, trend

**Structure DataFrame** :
```python
# Colonnes identifiants
gamma_id, d_encoding_id, modifier_id, seed, test_name, params_config_id, metric_name

# Colonnes projections numériques
value_final, value_initial, value_mean, value_std, value_min, value_max
slope, volatility, relative_change

# Colonnes catégorielles
transition, trend
```

**Utilisé par** :
- `verdict_engine.analyze_regime()` : Conversion pour analyses
- `gamma_profiling` : (via verdict_engine indirectement)

**Notes** :
- Filtre lignes avec NaN dans TOUTES projections numériques
- Une ligne par (observation, metric)

---

#### `cache_observations(observations, cache_path)` & `load_cached_observations(cache_path)`
Cache/chargement observations (optimisation future).

**État** : Implémenté mais non utilisé (réservé futurs besoins)

---

### regime_utils.py

**Responsabilité** : Stratification, classification régimes, taxonomie

**Statut** : Nouveau module (extraction depuis verdict_engine + gamma_profiling)

**Fonctions Publiques** :

#### `stratify_by_regime(observations: List[dict], threshold=1e50) → Tuple[List[dict], List[dict]]`
Stratifie observations en régimes stable/explosif.

**Critère** : Présence valeurs >threshold dans projections exploitées.

**Retour** : `(obs_stable, obs_explosif)`

**Utilisé par** :
- `verdict_engine.compute_verdict()` : Stratification 3 strates
- `verdict_reporter.generate_verdict_report()` : Pipeline étape 2

**Notes** :
- Conservation TOUTES observations (aucun filtrage)
- Vérifie : statistics (initial, final, mean, max) + evolution (slope, relative_change)

---

#### `classify_regime(metrics, dynamic_sig, timeline_dist, dispersion, test_name) → str`
Classification régime R0 avec régimes SPÉCIFIQUES.

**Régimes Conservation** :
```python
CONSERVES_SYMMETRY    # SYM-* : asymétrie < 1e-6
CONSERVES_NORM        # SPE-*, UNIV-* : norme < 2× initial
CONSERVES_PATTERN     # PAT-* : diversity/uniformity stable
CONSERVES_TOPOLOGY    # TOP-* : euler_characteristic stable
CONSERVES_GRADIENT    # GRA-* : gradient structure conservée
CONSERVES_SPECTRUM    # SPA-* : spectre stable
```

**Régimes Pathologiques** :
```python
NUMERIC_INSTABILITY   # instability_onset < 20 && final > 1e20
OSCILLATORY_UNSTABLE  # oscillatory_fraction > 0.3
TRIVIAL               # cv < 0.01
DEGRADING             # final < 0.5 × initial (sans collapse)
```

**Régimes Autres** :
```python
SATURATES_HIGH        # saturation + final > 10 × initial
UNCATEGORIZED         # comportement non classifié
```

**Qualificatif Multimodalité** : `MIXED::{régime_base}` si bimodal détecté

**Utilisé par** :
- `gamma_profiling.compute_prc_profile()` : Classification comportementale

---

#### `detect_conserved_property(test_name: str) → str`
Détermine propriété conservée selon préfixe test.

**Mapping** :
```python
SYM-*           → CONSERVES_SYMMETRY
SPE-*, UNIV-*   → CONSERVES_NORM
PAT-*           → CONSERVES_PATTERN
TOP-*           → CONSERVES_TOPOLOGY
GRA-*           → CONSERVES_GRADIENT
SPA-*           → CONSERVES_SPECTRUM
Autre           → CONSERVES_PROPERTY (fallback)
```

**Utilisé par** :
- `classify_regime()` : Détection automatique propriété

---

#### `extract_conserved_properties(profile: dict) → List[str]`
Extrait propriétés conservées depuis profil gamma.

**Retour** : `['Symétrie', 'Norme', 'Pattern', ...]` (triées)

**Utilisé par** :
- `verdict_reporter._write_summary_report()` : Affichage propriétés par gamma

---

#### `get_regime_family(regime: str) → str`
Retourne famille d'un régime.

**Retour** : `'conservation'` | `'pathology'` | `'saturation'` | `'other'`

**Notes** :
- Strip qualificatif MIXED:: automatiquement
- Utilise constante `REGIME_TAXONOMY` (référence complète)

---

### report_writers.py

**Responsabilité** : Formatage et écriture rapports structurés

**Statut** : Nouveau module (extraction depuis verdict_reporter)

**Fonctions Publiques** :

#### `write_json(data: dict, filepath: Path, indent=2)`
Écrit dict → JSON formaté (avec conversion tuples → strings).

**Utilisé par** :
- `verdict_reporter` : Tous fichiers JSON (metadata, comparisons, structural_patterns, diagnostics)

**Notes** :
- Utilise `_make_json_serializable(obj)` pour sérialisation sûre
- Convertit tuples clés → strings (format `"test|metric|proj"`)

---

#### `write_header(f, title: str, width=80, char='=')`
Écrit header section (bordures =).

#### `write_subheader(f, title: str, width=80, char='-')`
Écrit sous-header section (bordures -).

#### `write_key_value(f, key: str, value, indent=0)`
Écrit paire clé-valeur formatée.

**Utilisé par** :
- `verdict_reporter._write_summary_report()` : Formatage sections TXT

---

#### `write_regime_synthesis(f, gamma_profiles: dict, width=80)`
Écrit synthèse régimes transversale (tous gammas × tests).

**Structure** :
```
CONSERVATION (régimes sains):
  CONSERVES_SYMMETRY : 120 (12.5%)
  CONSERVES_NORM     : 85  (8.9%)
  Total conservation : 205 (21.4%)

PATHOLOGIES:
  NUMERIC_INSTABILITY : 45 (4.7%)
  OSCILLATORY_UNSTABLE: 30 (3.1%)
  Total pathologies   : 75 (7.8%)

MULTIMODALITÉ (MIXED::X):
  MIXED::CONSERVES_NORM : 15 (1.6%)
  Total multimodal      : 20 (2.1%)
```

**Utilisé par** :
- `verdict_reporter._write_summary_report()` : Section synthèse régimes

---

#### `write_dynamic_signatures(f, gamma_profiles: dict, width=80)`
Écrit signatures dynamiques par gamma (timelines dominantes).

**Structure** :
```
GAM-001:
  Timeline dominante : early_deviation_then_saturation (75% tests)
  Variantes (3 timelines distinctes):
    - mid_instability_then_collapse (8 tests)
    - late_saturation_only (3 tests)
```

**Utilisé par** :
- `verdict_reporter._write_summary_report()` : Section signatures dynamiques

---

#### `write_comparisons_enriched(f, comparisons: dict, gamma_profiles: dict, width=80)`
Écrit comparaisons enrichies avec contexte propriétés.

**Structure** :
```
SYMÉTRIE:
  SYM-001:
    Meilleur : GAM-001
    Pire     : GAM-013
    Classement : GAM-001, GAM-004, GAM-007, GAM-009, GAM-012...

NORME:
  SPE-001:
    Meilleur : GAM-004
    ...
```

**Utilisé par** :
- `verdict_reporter._write_summary_report()` : Section comparaisons inter-gammas

**Groupement Propriétés** :
```python
tests_by_property = {
    'Symétrie': ['SYM-001'],
    'Norme': ['SPE-001', 'SPE-002', 'UNIV-001', 'UNIV-002'],
    'Pattern': ['PAT-001'],
    'Topologie': ['TOP-001'],
    'Gradient': ['GRA-001'],
    'Spectre': ['SPA-001']
}
```

---

#### `extract_conserved_properties_display(profile: dict) → List[str]`
Extrait propriétés conservées (pour affichage).

**Note** : Identique à `regime_utils.extract_conserved_properties` (duplication intentionnelle pour séparation formatage/logique)

---

#### `write_consultation_footer(f, width=80, char='=')`
Écrit footer avec fichiers consultation.

**Structure** :
```
================================================================================
CONSULTATION DÉTAILLÉE
================================================================================
gamma_profiles.json       : Profils complets tous gammas × tests
gamma_profiles.csv        : Vue tabulaire pour analyse
comparisons.json          : Classements inter-gammas
structural_patterns.json  : Analyses globales (variance, interactions)
diagnostics.json          : Diagnostics numériques détaillés
marginal_variance_*.csv   : Données brutes analyses (3 strates)
================================================================================
```

**Utilisé par** :
- `verdict_reporter._write_summary_report()` : Footer rapport principal

---

### statistical_utils.py

**Responsabilité** : Outils statistiques réutilisables

**Statut** : Nouveau module (extraction depuis verdict_engine)

**Fonctions Publiques** :

#### `compute_eta_squared(groups: List[np.ndarray]) → Tuple[float, float, float]`
Calcule eta-squared (η²) : proportion variance expliquée.

**Formule** :
```
η² = SSB / (SSB + SSW)

SSB (Sum of Squares Between) = Σ n_i × (mean_i - grand_mean)²
SSW (Sum of Squares Within)  = Σ Σ (x_ij - mean_i)²
SST (Sum of Squares Total)   = SSB + SSW
```

**Retour** : `(eta2, ssb, ssw)`

**Utilisé par** :
- `verdict_engine.analyze_marginal_variance()` : Calcul variance marginale
- `verdict_engine.analyze_oriented_interactions()` : Calcul variance conditionnelle

**Notes** :
- Retourne `(0.0, 0.0, 0.0)` si données insuffisantes
- Protection division par zéro : `sst > 1e-10`
- Filtre groupes vides automatiquement

---

#### `kruskal_wallis_test(groups: List[np.ndarray]) → Tuple[float, float]`
Test Kruskal-Wallis avec gestion erreurs.

**Retour** : `(statistic, p_value)`

**Raises** : `ValueError` si moins de 2 groupes ou données insuffisantes

**Utilisé par** :
- `verdict_engine.analyze_marginal_variance()` : Test significativité
- `verdict_engine.analyze_oriented_interactions()` : Test significativité

---

#### `is_numeric_valid(obs: dict) → bool`
Détecte artefacts numériques (inf/nan) dans projections exploitées.

**Vérifie** :
- statistics : initial, final, mean, std, min, max
- evolution : slope, volatility, relative_change

**Retour** : `True` si aucun artefact détecté

**Notes** : Fonction privée (usage interne `filter_numeric_artifacts`)

---

#### `filter_numeric_artifacts(observations: List[dict]) → Tuple[List[dict], dict]`
Filtre observations avec artefacts numériques.

**Retour** :
```python
(valid_observations, rejection_stats)

rejection_stats = {
    'total_observations': int,
    'valid_observations': int,
    'rejected_observations': int,
    'rejection_rate': float,
    'rejected_by_test': dict  # {test_name: count}
}
```

**Utilisé par** :
- `verdict_engine.compute_verdict()` : Filtrage initial
- `verdict_reporter.generate_verdict_report()` : Pipeline étape 1

**Notes** : Log rejets par test pour traçabilité

---

#### `diagnose_numeric_degeneracy(obs: dict) → List[str]`
Détecte dégénérescences numériques sur projections exploitées.

**Flags Détectés** :
```python
"metric:projection:INFINITE_PROJECTION"   # inf détecté
"metric:projection:NAN_PROJECTION"        # nan détecté
"metric:projection:EXTREME_MAGNITUDE"     # |valeur| > 1e50
```

**Inspecte** : value_final, value_mean, slope, volatility, relative_change

**Notes** : Flags par projection (pas global)

---

#### `generate_degeneracy_report(observations: List[dict]) → dict`
Génère rapport diagnostique dégénérescences.

**Structure Retour** :
```python
{
    'total_observations': int,
    'observations_with_flags': int,
    'flag_rate': float,
    'flag_counts': dict,              # {flag: count}
    'flags_by_test': dict,            # {test_name: {flag_type: count}}
    'flags_by_projection': dict       # {projection: count}
}
```

**Utilisé par** :
- `verdict_engine.compute_verdict()` : Diagnostics globaux
- `verdict_reporter.generate_verdict_report()` : Pipeline étape 2

---

#### `print_degeneracy_report(report: dict)`
Affiche rapport diagnostique dégénérescences (stdout).

**Format** :
```
================================================================================
DIAGNOSTIC DÉGÉNÉRESCENCES NUMÉRIQUES (projections exploitées)
================================================================================
Total observations:        4320
Observations flaggées:     150 (3.5%)

Dégénérescences par projection (variables analysées):
  value_final          :    80 occurrences ( 1.9%)
  slope                :    45 occurrences ( 1.0%)
  volatility           :    25 occurrences ( 0.6%)

Flags les plus fréquents:
  EXTREME_MAGNITUDE    :   120 occurrences ( 2.8%)
  INFINITE_PROJECTION  :    20 occurrences ( 0.5%)
  NAN_PROJECTION       :    10 occurrences ( 0.2%)

Dégénérescences par test:
TOP-001: 45 flags
  EXTREME_MAGNITUDE    :    40
  INFINITE_PROJECTION  :     5
...
```

**Utilisé par** :
- `verdict_engine.compute_verdict()` : Affichage diagnostics

---

#### `diagnose_scale_outliers(observations: List[dict]) → dict`
Détecte ruptures d'échelle relatives par contexte (test×métrique×projection).

**Critère** : Valeur > P90 + 5 décades (facteur 1e5)

**Structure Retour** :
```python
{
    'total_observations': int,
    'observations_with_outliers': int,
    'outlier_rate': float,
    'contexts_analyzed': int,          # Nb contextes avec ≥10 obs
    'contexts_with_outliers': int,
    'outliers_by_context': dict,       # {(test, metric, proj): [outliers]}
    'thresholds': dict                 # {(test, metric, proj): P90}
}
```

**Projections Analysées** : value_final, value_mean, slope, relative_change

**Utilisé par** :
- `verdict_engine.compute_verdict()` : Diagnostics échelle
- `verdict_reporter.generate_verdict_report()` : Pipeline étape 2

**Notes** :
- Min 10 observations par contexte pour calculer P90
- Outliers stockent gap en décades (log10)
- Clés contexte : `(test_name, metric_name, proj_name)`

---

#### `print_scale_outliers_report(report: dict)`
Affiche rapport ruptures d'échelle (stdout).

**Format** :
```
================================================================================
DIAGNOSTIC RUPTURES D'ÉCHELLE RELATIVES
================================================================================
Total observations:              4320
Contextes analysés (test×métrique×proj): 85
Observations avec outliers:      120 (2.8%)
Contextes ayant outliers:        12

Contextes avec ruptures d'échelle (>P90 + 5 décades):

TOP-001/euler_characteristic [value_final]:
   25 outliers ( 0.6%)
  Gap max: +7.3 décades, moyen: +6.1 décades
  Pires cas:
    obs#  123: 1.23e+12 (+7.3 décades)
    obs#  456: 5.67e+11 (+6.8 décades)
...
```

**Utilisé par** :
- `verdict_engine.compute_verdict()` : Affichage diagnostics

---

### timeline_utils.py

**Responsabilité** : Construction timelines dynamiques compositionnels

**Statut** : Nouveau module (extraction depuis gamma_profiling)

**Configuration Globale** :
```python
TIMELINE_THRESHOLDS = {
    'early': 0.20,  # onset < 20% durée
    'mid':   0.60,  # 20% ≤ onset ≤ 60%
    'late':  0.60   # onset > 60%
}
```

**Principe R0** :
- Toute notion temporelle est RELATIVE (jamais absolue)
- Seuils globaux uniques (pas de variation par test)
- Composition automatique : `{timing}_{event}_then_{event}`
- Descriptif pas causal ("then" pas "causes")

**Fonctions Publiques** :

#### `classify_timing(onset_relative: float) → str`
Classifie timing selon seuils globaux.

**Retour** : `'early'` | `'mid'` | `'late'`

**Utilisé par** :
- `compute_timeline_descriptor()` : Classification phases

---

#### `compute_timeline_descriptor(sequence, sequence_timing_relative, oscillatory_global) → dict`
Génère descriptor timeline compositionnel.

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

**Formats timeline_compact** :
- Aucun événement → `'no_significant_dynamics'`
- 1 événement → `'{timing}_{event}_only'`
- 2 événements → `'{timing1}_{event1}_then_{event2}'`
- 3+ événements → `'{timing1}_{event1}_to_{eventN}_complex'`
- Oscillatoire global → préfixe `'oscillatory_'`

**Utilisé par** :
- `gamma_profiling.aggregate_dynamic_signatures()` : Construction timelines

---

#### `extract_dynamic_events(observation: dict, metric_name: str) → dict`
Extrait événements dynamiques depuis observation.

**Lit depuis** : `observation_data['dynamic_events'][metric_name]`

**Structure Retour** :
```python
{
    'deviation_onset': int | None,
    'instability_onset': int | None,
    'oscillatory': bool,
    'saturation': bool,
    'collapse': bool,
    'sequence': ['deviation', 'instability', ...],
    'sequence_timing': [3, 7, ...],            # Absolus
    'sequence_timing_relative': [0.015, 0.035, ...],  # Relatifs [0,1]
    'saturation_onset_estimated': bool,
    'oscillatory_global': bool
}
```

**Utilisé par** :
- `gamma_profiling.aggregate_dynamic_signatures()` : Extraction événements

**Notes** :
- Retourne valeurs par défaut si dynamic_events absent
- Compatible fallback (test_engine avant enrichissement)

---

#### `extract_metric_timeseries(observation: dict, metric_name: str) → Tuple[List[float], bool]`
Extrait série temporelle avec marqueur fallback.

**Retour** : `(values, is_fallback)`
- `values` : liste valeurs ou None
- `is_fallback` : True si proxy linéaire utilisé

**Fallback** : `linspace(initial, final)` si timeseries absent

**Utilisé par** : Analyses temporelles avancées (futures)

---

## ✅ Modules HUB Refactorisés

### verdict_engine.py ✅

**Responsabilité** : Analyses statistiques globales multi-facteurs

**Statut** : Refactorisé Phase 2.1 (délègue aux utils)

**Configuration** :
```python
FACTORS = ['gamma_id', 'd_encoding_id', 'modifier_id', 'seed', 'test_name']
PROJECTIONS = ['value_final', 'value_mean', 'slope', 'volatility', 'relative_change']
MIN_SAMPLES_PER_GROUP = 2
MIN_GROUPS = 2
MIN_TOTAL_SAMPLES = 10
```

**Imports Refactorisés** :
```python
from .data_loading import load_all_observations, observations_to_dataframe
from .statistical_utils import compute_eta_squared, kruskal_wallis_test, filter_numeric_artifacts, ...
from .regime_utils import stratify_by_regime
```

**Fonctions Cœur Métier** (conservées) :

#### `analyze_marginal_variance(df, factors, projections) → DataFrame`
Analyse variance marginale (chaque facteur isolément).

**Utilise** :
- `statistical_utils.compute_eta_squared()` : Calcul η²
- `statistical_utils.kruskal_wallis_test()` : Test significativité

**Retour** : DataFrame avec colonnes :
- test_name, metric_name, projection, factor
- variance_ratio (η²), p_value, n_groups, significant

**Filtrage Testabilité** :
- Contexte global : ≥ MIN_TOTAL_SAMPLES observations
- Par groupe : ≥ MIN_SAMPLES_PER_GROUP observations
- Facteur : ≥ MIN_GROUPS niveaux distincts

---

#### `analyze_oriented_interactions(df, factors, projections, marginal_variance) → DataFrame`
Détecte interactions ORIENTÉES (A|B distinct de B|A).

**Critères Interaction Vraie** :
1. VR(A|B=b) significatif (η² > 0.3, p < 0.05)
2. VR(A|B=b) >> VR(A) marginal (ratio > 2.0)
3. Testabilité robuste (≥3 niveaux, ≥5 obs/groupe)

**Utilise** :
- `statistical_utils.compute_eta_squared()` : Calcul η² conditionnel
- `statistical_utils.kruskal_wallis_test()` : Test significativité

**Retour** : DataFrame avec colonnes :
- factor_varying, factor_context, context_value
- test_name, metric_name, projection
- vr_conditional, vr_marginal, interaction_strength
- p_value, n_groups

**Notes** :
- Paires orientées : `len(factors) × (len(factors) - 1)` = 20 paires
- Critères testabilité renforcés (Phase 2.2)

---

#### `analyze_metric_discrimination(df, projections) → DataFrame`
Détecte métriques non discriminantes (CV < 0.1).

**Retour** : DataFrame avec colonnes :
- test_name, metric_name, projection
- mean, std, cv, n_observations, non_discriminant

---

#### `analyze_metric_correlations(df, projection='value_final', threshold=0.8) → DataFrame`
Détecte corrélations fortes entre métriques (Spearman).

**Retour** : DataFrame avec colonnes :
- test_name, metric1, metric2, correlation, p_value, n_observations

---

#### `interpret_patterns(df, marginal, interactions, discrimination, correlations) → Tuple[dict, dict]`
Synthèse patterns détectés : GLOBAL + PAR GAMMA.

**Retour** : `(patterns_global, patterns_by_gamma)`

**Patterns Globaux** :
```python
{
    'marginal_dominant': [
        {'factor': 'gamma_id', 'n_metrics': 15, 'projections': [...], 'max_variance_ratio': 0.85}
    ],
    'oriented_interactions': [
        {'interaction': 'modifier_id | gamma_id', 'n_cases': 8, 'contexts_affected': [...], 'max_strength': 3.2}
    ],
    'non_discriminant': [{'n_metrics': 3, 'projections': [...]}],
    'redundant': [{'n_pairs': 5}]
}
```

**Drill-down Gamma** : Recalcule analyses sur sous-ensemble gamma strict

---

#### `decide_verdict(patterns_global, patterns_by_gamma) → Tuple[str, str, dict]`
Décision verdict final.

**Retour** : `(verdict_global, reason_global, verdicts_by_gamma)`

**Logique** :
- `SURVIVES[R0]` : Aucun pattern critique
- `WIP[R0-open]` : Patterns détectés

---

#### `analyze_regime(observations, regime_name, params_config_id, verdict_config_id) → dict`
Pipeline complet analyse sur une strate.

**Utilise** :
- `data_loading.observations_to_dataframe()`
- `analyze_marginal_variance()`, `analyze_oriented_interactions()`, etc.
- `interpret_patterns()`, `decide_verdict()`

**Retour** : Structure complète avec marginal_variance, oriented_interactions, patterns, verdict

---

#### `generate_stratified_report(...)`
Génère rapports 3 strates (GLOBAL, STABLE, EXPLOSIF).

**Fichiers générés** :
- `metadata.json`
- `summary.txt`
- `analysis_{regime}.json`
- `marginal_variance_{regime}.csv`
- `oriented_interactions_{regime}.csv`

---

#### `compute_verdict(params_config_id, verdict_config_id)`
**Point d'entrée principal** : Pipeline complet refactorisé.

**Étapes** :
1. Chargement observations (→ data_loading)
2. Filtrage artefacts (→ statistical_utils)
3. Diagnostics (→ statistical_utils)
4. Stratification (→ regime_utils)
5. Analyses parallèles 3 strates
6. Génération rapports

**Réduction code** : ~500 lignes → ~400 lignes (-20%)

---

### gamma_profiling.py ✅

**Responsabilité** : Profiling comportemental gammas individuels

**Statut** : Refactorisé Phase 2.2 (délègue aux utils)

**Imports Refactorisés** :
```python
from .timeline_utils import classify_timing, compute_timeline_descriptor, extract_dynamic_events, ...
from .aggregation_utils import aggregate_summary_metrics, aggregate_run_dispersion
from .regime_utils import classify_regime, extract_conserved_properties
```

**Fonctions Locales Conservées** :

#### `aggregate_dynamic_signatures(observations, metric_name) → dict`
Agrège signatures événements + timelines.

**Spécifique profiling** : Composition timeline + événements booléens

**Utilise** :
- `timeline_utils.extract_dynamic_events()` : Extraction événements
- `timeline_utils.compute_timeline_descriptor()` : Construction timeline

**Retour** :
```python
{
    'dynamic_signature': {
        'deviation_onset_median': float | None,
        'instability_onset_median': float | None,
        'oscillatory_fraction': float,
        'saturation_fraction': float,
        'collapse_fraction': float
    },
    'timeline_distribution': {
        'dominant_timeline': 'early_deviation_then_saturation',
        'timeline_confidence': 0.75,
        'timeline_variants': {'timeline1': 15, 'timeline2': 5, ...}
    }
}
```

---

#### `compute_prc_profile(metrics, dynamic_sig, timeline_dist, dispersion, n_runs, n_valid, test_name) → dict`
Génère profil PRC complet avec confidence heuristique.

**Cœur métier** profiling gamma

**Utilise** :
- `regime_utils.classify_regime()` : Régime + qualificatif MIXED

**Retour** :
```python
{
    'regime': 'CONSERVES_SYMMETRY' | 'MIXED::CONSERVES_NORM' | ...,
    'behavior': 'stable' | 'unstable' | 'degrading' | 'mixed',
    'dominant_timeline': {
        'timeline_compact': str,
        'confidence': float,
        'variants': dict
    },
    'robustness': {
        'homogeneous': bool,
        'mixed_behavior': bool,
        'numerically_stable': bool
    },
    'pathologies': {
        'numeric_instability': bool,
        'oscillatory': bool,
        'collapse': bool,
        'trivial': bool,
        'degrading': bool
    },
    'n_runs': int,
    'n_valid': int,
    'confidence': 'high' | 'medium' | 'low',
    'confidence_metadata': {
        'level': str,
        'criteria': dict,
        'rationale': str
    }
}
```

**Confidence Heuristique** :
- **high** : n_valid ≥ 20 && !bimodal && timeline_conf ≥ 0.7
- **medium** : n_valid ≥ 10
- **low** : n_valid < 10

---

#### `profile_test_for_gamma(observations, test_name, gamma_id) → dict`
Profil UN test sous UN gamma (3 niveaux).

**Utilise** :
- `aggregation_utils.aggregate_summary_metrics()` : Niveau 3 (instrumentation)
- `aggregation_utils.aggregate_run_dispersion()` : Niveau 2 (diagnostic)
- `aggregate_dynamic_signatures()` : Niveau 2 (local)
- `compute_prc_profile()` : Niveau 1 (cœur métier)

**Retour** :
```python
{
    'test_name': str,
    'gamma_id': str,
    'prc_profile': {...},           # Niveau 1
    'diagnostic_signature': {...},   # Niveau 2
    'instrumentation': {...}         # Niveau 3
}
```

**Marqueur Fallback** : `instrumentation.data_completeness.fallback_used`

---

#### `profile_all_gammas(observations) → dict`
Profil tous gammas × tous tests.

**Retour** :
```python
{
    'GAM-001': {
        'tests': {
            'SYM-001': {...},
            'SPE-001': {...}
        },
        'n_tests': int,
        'n_total_runs': int
    },
    ...
}
```

---

#### `rank_gammas_by_test(profiles, test_name, criterion='conservation') → List[Tuple]`
Classe gammas pour un test donné.

**Critères** :
- `'conservation'` : CONSERVES_* = 1.0, TRIVIAL = 0.5, autres = 0.0
- `'stability'` : Score basé instability_onset + collapse_fraction
- `'final_value'` : Score basé médiane finale (log10)

**Retour** : `[(gamma_id, score), ...]` (trié décroissant)

---

#### `compare_gammas_summary(profiles) → dict`
Synthèse comparative gammas.

**Utilise** :
- `regime_utils.extract_conserved_properties()` : Propriétés conservées

**Retour** :
```python
{
    'by_regime': {
        'CONSERVES_SYMMETRY': ['GAM-001', 'GAM-004'],
        'NUMERIC_INSTABILITY': ['GAM-013']
    },
    'by_test': {
        'SYM-001': {
            'best_conservation': 'GAM-001',
            'worst_conservation': 'GAM-013',
            'ranking': ['GAM-001', 'GAM-004', ...]
        }
    }
}
```

**Réduction code** : ~700 lignes → ~350 lignes (-50%)

---

### verdict_reporter.py ✅

**Responsabilité** : Orchestration génération rapports R0

**Statut** : Refactorisé Phase 2.3 (délègue aux utils)

**Imports Refactorisés** :
```python
from .data_loading import load_all_observations
from .statistical_utils import filter_numeric_artifacts, generate_degeneracy_report, ...
from .regime_utils import stratify_by_regime, extract_conserved_properties
from .verdict_engine import analyze_regime, FACTORS, PROJECTIONS, ...
from .gamma_profiling import profile_all_gammas, compare_gammas_summary
from .report_writers import write_json, write_header, write_regime_synthesis, ...
```

**Pipeline Principal** :

#### `generate_verdict_report(params_config_id, verdict_config_id, output_dir='reports/verdicts') → dict`
Pipeline complet génération rapport R0.

**Étapes** :
1. **Chargement + Diagnostics** (délégué)
   - `data_loading.load_all_observations()`
   - `statistical_utils.filter_numeric_artifacts()`
   - `statistical_utils.generate_degeneracy_report()`
   - `statistical_utils.diagnose_scale_outliers()`

2. **Analyses Globales Stratifiées** (délégué verdict_engine)
   - `regime_utils.stratify_by_regime()`
   - `verdict_engine.analyze_regime()` × 3 (GLOBAL, STABLE, EXPLOSIF)

3. **Profiling Gamma** (délégué gamma_profiling)
   - `gamma_profiling.profile_all_gammas()`
   - `gamma_profiling.compare_gammas_summary()`

4. **Fusion Résultats** (local - orchestration)
   - `_compile_metadata()`
   - `_format_gamma_profiles()`
   - `_compile_structural_patterns()`

5. **Génération Rapports** (délégué report_writers)
   - `report_writers.write_json()` : Tous fichiers JSON
   - `_write_summary_report()` : TXT (utilise report_writers pour sections)
   - `_write_gamma_profiles()` : JSON + CSV

**Retour** :
```python
{
    'metadata': {...},
    'gamma_profiles': {...},
    'structural_patterns': {...},
    'comparisons': {...},
    'diagnostics': {...},
    'report_paths': {...}
}
```

---

**Fonctions Locales Conservées** (orchestration) :

#### `_compile_metadata(...) → dict`
Compile métadonnées rapport (data_summary, quality_flags, analysis_parameters).

#### `_format_gamma_profiles(gamma_profiles) → dict`
Transforme structure pour rapports (extraction régime/behavior/timeline/confidence).

#### `_compile_structural_patterns(results_global, results_stable, results_explosif) → dict`
Compile patterns 3 strates (stratification.GLOBAL/STABLE/EXPLOSIF).

#### `_write_summary_report(report_dir, results)`
Écrit rapport TXT principal enrichi R0+.

**Utilise** :
- `report_writers.write_header()`
- `report_writers.write_regime_synthesis()`
- `report_writers.write_dynamic_signatures()`
- `report_writers.write_comparisons_enriched()`
- `report_writers.write_consultation_footer()`
- `regime_utils.extract_conserved_properties()`

#### `_write_gamma_profiles(report_dir, gamma_profiles)`
Écrit gamma_profiles.json + CSV tabulaire.

**Utilise** : `report_writers.write_json()`

**Réduction code** : ~600 lignes → ~400 lignes (-33%)

---

## 📦 Modules Stables (Pas de Changement)

### test_engine.py 📦

**Classe** : `TestEngine`

**Responsabilité** : Moteur d'exécution tests avec registries

**État** : Stable (fonctionnel complet Charter 5.5)

**Méthodes Clés** :
- `execute_test(test_module, run_metadata, history, params_config_id)` → dict
  - Valide COMPUTATION_SPECS via RegistryManager
  - Exécute formules sur snapshots
  - Applique post_processors
  - Calcule statistics/evolution/dynamic_events
  - Retourne observation format 5.5

**Fonctions Détection Dynamique** :
- `detect_dynamic_events(values)` → dict
  - Détecte : deviation_onset, instability_onset, oscillatory, saturation, collapse
  - Seuils relatifs (P90, 10% iterations, CV < 5%)
  
- `compute_event_sequence(events, n_iterations)` → dict
  - Génère séquence temporelle ordonnée
  - Calcule onsets relatifs [0, 1]
  - Structure : `{'sequence': [...], 'sequence_timing_relative': [...]}`

**Format Retour** :
```python
{
    'exec_id': int,
    'test_name': str,
    'status': 'SUCCESS' | 'ERROR',
    'statistics': {metric: {initial, final, mean, std, ...}},
    'evolution': {metric: {slope, volatility, transition, ...}},
    'dynamic_events': {metric: {events + sequence}},  # ⭐ Charter 5.5
    'timeseries': {metric: [val_0, ..., val_N]},      # Optionnel
    'metadata': {...}
}
```

**Note** : Aucune fonction à déplacer (module autonome)

---

### applicability.py 📦

**Fonction** : `check(test_module, run_metadata)` → (bool, str)

**Responsabilité** : Vérifier applicabilité test sur métadonnées run

**État** : Stable (validators extensibles)

**Validators Disponibles** :
```python
VALIDATORS = {
    'requires_rank': lambda meta, expected: ...,
    'requires_square': lambda meta, required: ...,
    'allowed_d_types': lambda meta, allowed: ...,
    'requires_even_dimension': lambda meta, required: ...,
    'minimum_dimension': lambda meta, min_dim: ...
}
```

**Extensibilité** : `add_validator(name, validator)`

**Exemple** :
```python
is_applicable, reason = check(test_sym_001, {
    'state_shape': (10, 10),
    'd_encoding_id': 'SYM-001',
    ...
})
# (True, "") si conforme
# (False, "requires_square = True non satisfait") sinon
```

---

### config_loader.py 📦

**Classe** : `ConfigLoader` (singleton via `get_loader()`)

**Responsabilité** : Chargement configs YAML avec fusion global/specific

**État** : Stable (architecture configs fixée)

**Méthodes** :
- `load(config_type, config_id, test_id=None)` → dict
  - config_type : 'params' | 'verdict'
  - Fusion auto global + specific
  - Cache pour performance
  
- `list_available(config_type)` → dict
  - `{'global': [...], 'tests': {test_id: [...]}}`
  
- `clear_cache()` : Vider cache

**Architecture Fichiers** :
```
tests/config/
├── global/
│   ├── params_default_v1.yaml
│   └── verdict_default_v1.yaml
└── tests/
    └── UNIV-001/
        └── params_custom_v1.yaml
```

**Stratégie Fusion** :
- Clés top-level : specific écrase global
- Dicts imbriqués : merge récursif
- Listes : specific remplace global

**Validation** :
- Métadata requise : version, config_id, description
- Warning si absente ou incohérente

---

### discovery.py 📦

**Fonction** : `discover_active_tests()` → Dict[str, module]

**Responsabilité** : Découverte automatique tests actifs (non `_deprecated`)

**État** : Stable (validation Charter 5.5)

**Validation Structure** : `validate_test_structure(module)`

**Attributs Requis Charter 5.5** :
```python
REQUIRED_ATTRIBUTES = [
    'TEST_ID',           # Format CAT-NNN
    'TEST_CATEGORY',
    'TEST_VERSION',      # Doit être '5.5'
    'APPLICABILITY_SPEC',
    'COMPUTATION_SPECS'  # 1-5 métriques
]
```

**Vérifications** :
- Types corrects
- TEST_VERSION == "5.5"
- TEST_ID format CAT-NNN (regex)
- COMPUTATION_SPECS : 1 ≤ len ≤ 5
- Chaque métrique : registry_key + default_params
- registry_key format 'registry.function'
- Pas de FORMULAS/is_applicable/compute_metric legacy

**Exemple** :
```python
tests = discover_active_tests()
# {'UNIV-001': <module>, 'SYM-001': <module>, ...}
```

---

### registries/ 📦

**registry_manager.py** : Gestion centralisée registries

**Classe** : `RegistryManager` (singleton)

**Responsabilité** : Gestion centralisée registries de fonctions calcul

**État** : Stable (registries complets)

**Méthodes** :
- `get_function(registry_key)` → Callable
  - registry_key format : "registry_name.function_name"
  - Ex : "algebra.frobenius_norm"
  
- `validate_computation_spec(spec)` → dict
  - Valide spec COMPUTATION_SPECS
  - Retourne : {function, params, post_process, registry_key}

**Registries Disponibles** :

| Registry | Clé | N Fonctions | Exemples |
|----------|-----|-------------|----------|
| AlgebraRegistry | `algebra` | 8 | frobenius_norm, trace, spectral_norm |
| GraphRegistry | `graph` | 6 | laplacian_energy, clustering_coefficient |
| PatternRegistry | `pattern` | 6 | local_diversity, pattern_uniformity |
| SpatialRegistry | `spatial` | 6 | center_of_mass, spatial_spread |
| SpectralRegistry | `spectral` | 6 | spectral_gap, spectral_entropy |
| StatisticalRegistry | `statistical` | 7 | entropy, kurtosis, skewness |
| TopologicalRegistry | `topological` | 6 | euler_characteristic, betti_numbers |

**Post-processors Disponibles** :
- `absolute` : np.abs()
- `log` : np.log(x + 1e-10)
- `normalize` : x / (np.linalg.norm(x) + 1e-10)

---

## 📊 Récapitulatif Architecture Refactorisée

### Flux de Données Complet

```
1. Discovery & Applicability (📦 stable)
   discover_active_tests() → {test_id: module}
   check(test_module, run_metadata) → (True, "")

2. Config Loading (📦 stable)
   get_loader().load('params', 'params_default_v1', 'UNIV-001')

3. Test Execution (📦 stable)
   TestEngine().execute_test(...)
   ├─ Validate COMPUTATION_SPECS via RegistryManager
   ├─ Execute on snapshots
   ├─ detect_dynamic_events() + compute_event_sequence()
   └─ Return observation dict

4. Verdict Analysis (✅ refactorisé)
   verdict_engine.compute_verdict(params_config_id, verdict_config_id)
   ├─ data_loading.load_all_observations()
   ├─ statistical_utils.filter_numeric_artifacts()
   ├─ regime_utils.stratify_by_regime()
   ├─ verdict_engine.analyze_regime() × 3
   └─ verdict_engine.generate_stratified_report()

5. Gamma Profiling (✅ refactorisé)
   gamma_profiling.profile_all_gammas(observations)
   ├─ profile_test_for_gamma() pour chaque gamma × test
   │  ├─ aggregation_utils.aggregate_summary_metrics()
   │  ├─ aggregation_utils.aggregate_run_dispersion()
   │  ├─ aggregate_dynamic_signatures() [local]
   │  │  ├─ timeline_utils.extract_dynamic_events()
   │  │  └─ timeline_utils.compute_timeline_descriptor()
   │  └─ compute_prc_profile()
   │     └─ regime_utils.classify_regime()
   └─ compare_gammas_summary()

6. Report Generation (✅ refactorisé)
   verdict_reporter.generate_verdict_report(...)
   ├─ Étape 1-3 : Délégation analyses (data_loading, verdict_engine, gamma_profiling)
   ├─ Étape 4 : Fusion résultats (local orchestration)
   └─ Étape 5 : Génération multi-format
      ├─ report_writers.write_json() : JSON files
      ├─ _write_summary_report()
      │  ├─ report_writers.write_header()
      │  ├─ report_writers.write_regime_synthesis()
      │  ├─ report_writers.write_dynamic_signatures()
      │  ├─ report_writers.write_comparisons_enriched()
      │  └─ report_writers.write_consultation_footer()
      └─ _write_gamma_profiles()
         └─ report_writers.write_json()
```

---

### Table Récapitulative Modules

| Module | Type | Lignes Avant | Lignes Après | Réduction | Responsabilité |
|--------|------|--------------|--------------|-----------|----------------|
| **verdict_engine** | ✅ Hub | ~500 | ~400 | -20% | Analyses statistiques globales |
| **gamma_profiling** | ✅ Hub | ~700 | ~350 | -50% | Profiling comportemental gammas |
| **verdict_reporter** | ✅ Hub | ~600 | ~400 | -33% | Orchestration rapports |
| **aggregation_utils** | 🆕 Util | - | ~150 | - | Agrégations statistiques |
| **data_loading** | 🆕 Util | - | ~200 | - | I/O observations/DB |
| **regime_utils** | 🆕 Util | - | ~250 | - | Classification régimes |
| **report_writers** | 🆕 Util | - | ~200 | - | Formatage rapports |
| **statistical_utils** | 🆕 Util | - | ~400 | - | Outils statistiques |
| **timeline_utils** | 🆕 Util | - | ~200 | - | Construction timelines |
| **test_engine** | 📦 Stable | ~600 | ~600 | 0% | Exécution tests |
| **applicability** | 📦 Stable | ~100 | ~100 | 0% | Validation applicabilité |
| **config_loader** | 📦 Stable | ~150 | ~150 | 0% | Chargement configs |
| **discovery** | 📦 Stable | ~100 | ~100 | 0% | Discovery tests |
| **registries/** | 📦 Stable | ~800 | ~800 | 0% | Fonctions calcul |

**Totaux** :
- Hubs refactorisés : ~1800 → ~1150 lignes (**-36%**)
- Utils créés : ~1400 lignes (nouveau code réutilisable)
- Modules stables : ~1750 lignes (inchangés)
- **Total framework : ~4300 lignes** (bien structuré, maintenable)

---

### Gains Architecture Phase 2

**Séparation des Responsabilités** :
- ✅ Logique métier (hubs) vs utilitaires réutilisables
- ✅ I/O séparé de la logique (data_loading)
- ✅ Diagnostics centralisés (statistical_utils)
- ✅ Formatage externalisé (report_writers)

**Réutilisabilité** :
- ✅ Utils utilisables par futurs modules (modifier_profiling, test_profiling)
- ✅ Fonctions statistiques génériques (compute_eta_squared, kruskal_wallis_test)
- ✅ Writers standardisés (write_json, write_regime_synthesis)

**Maintenabilité** :
- ✅ Modification utils sans toucher hubs
- ✅ Tests unitaires facilités (fonctions isolées)
- ✅ Documentation claire (1 responsabilité par module)

**Performance** :
- ✅ Cache data_loading (futur)
- ✅ Imports optimisés (pas de duplications)

---

**Version** : 5.5 - Phase 2 Complète  
**Date** : Post-Refactorisation  
**Architecture** : Charter R0 - Posture Non Gamma-Centrique  
**Statut** : ✅ Production Ready