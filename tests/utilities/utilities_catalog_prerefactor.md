# Catalogue Utilities (Modules Utilitaires Tests) - R0

## Vue d'ensemble

Modules de support pour l'exécution et l'analyse des tests PRC Charter 5.5.

**Architecture Globale** :
```
tests/utilities/
├── test_engine.py          # Exécution tests
├── applicability.py        # Validation applicabilité
├── config_loader.py        # Chargement configs YAML
├── discovery.py            # Discovery tests actifs
├── verdict_engine.py       # Analyses statistiques globales ⭐
├── gamma_profiling.py      # Profiling comportemental gammas
├── verdict_reporter.py     # Orchestration rapports
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

---

## Modules Principaux

### test_engine.py

**Classe** : `TestEngine`

**Responsabilité** : Moteur d'exécution tests avec registries

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
    'dynamic_events': {metric: {events + sequence}},  # ⭐ NOUVEAU
    'timeseries': {metric: [val_0, ..., val_N]},      # Optionnel
    'metadata': {...}
}
```

---

### verdict_engine.py ⭐

**Responsabilité** : Analyses statistiques globales multi-facteurs

**Architecture** :
- FACTORS : `['gamma_id', 'd_encoding_id', 'modifier_id', 'seed', 'test_name']`
- PROJECTIONS : `['value_final', 'value_mean', 'slope', 'volatility', 'relative_change']`
- Paires orientées : `len(FACTORS) × (len(FACTORS) - 1) = 20 paires`

**Fonctions Chargement** :
- `load_all_observations(params_config_id)` → List[dict]
  - Double connexion DB : TestObservations + Executions
  - Fusionne observations + métadonnées runs
  
- `observations_to_dataframe(observations)` → DataFrame
  - Normalise structure pour analyses statistiques
  - Projections numériques + catégorielles

**Fonctions Filtrage** :
- `filter_numeric_artifacts(observations)` → (valid_obs, stats)
  - Détecte inf/nan dans statistics + evolution
  - Log rejets par test
  
- `diagnose_numeric_degeneracy(obs)` → List[str]
  - Flags : EXTREME_MAGNITUDE, INFINITE_PROJECTION, NAN_PROJECTION
  - Par projection exploitée
  
- `diagnose_scale_outliers(observations)` → dict
  - Ruptures d'échelle relatives : > P90 + 5 décades
  - Contextuel : (test, metric, projection)

**Fonctions Stratification** :
- `stratify_by_regime(observations, threshold=1e50)` → (stable, explosif)
  - Critère : |projection| >= threshold
  - Conservation intégrale (aucun filtrage)
  
- `analyze_regime(observations, regime_name, ...)` → dict
  - Pipeline complet sur strate
  - Retourne : variance, interactions, patterns, verdict

**Analyses Statistiques** :

#### 1. Variance Marginale
```python
analyze_marginal_variance(df, factors, projections) -> DataFrame
```
- **Mesure** : η² (eta-squared) = SSB / (SSB + SSW)
- **Filtrage** : min_samples_per_group=2, min_groups=2, min_total_samples=10
- **Sortie** : variance_ratio, p_value (Kruskal-Wallis), n_groups, significant

#### 2. Interactions Orientées
```python
analyze_oriented_interactions(df, factors, projections, marginal) -> DataFrame
```
- **Principe** : A|B distinct de B|A (permutations pas combinations)
- **Détection** : VR_cond > 0.3 && VR_cond >> VR_marginal (ratio > 2.0)
- **Critères Robustesse** : 
  - VR_marginal ≥ 0.1 (marginal substantiel)
  - n_groups ≥ 3 (au moins 3 niveaux)
  - min_group_size ≥ 5 (robustesse statistique)
- **Sortie** : factor_varying, factor_context, context_value, interaction_strength

#### 3. Discrimination Métriques
```python
analyze_metric_discrimination(df, projections) -> DataFrame
```
- **Critère** : CV < 0.1 → non discriminant
- **Sortie** : mean, std, cv, non_discriminant

#### 4. Corrélations
```python
analyze_metric_correlations(df, threshold=0.8) -> DataFrame
```
- **Méthode** : Spearman
- **Sortie** : metric1, metric2, correlation, p_value

**Fonctions Synthèse** :
- `interpret_patterns(...)` → (patterns_global, patterns_by_gamma)
  - Détecte : marginal_dominant, oriented_interactions, non_discriminant, redundant
  
- `decide_verdict(patterns_global, patterns_by_gamma)` → (verdict, reason, by_gamma)
  - Logique : SURVIVES[R0] si aucun pattern critique, sinon WIP[R0-open]

**Utilitaire** :
- `compute_eta_squared(groups)` → (eta2, ssb, ssw)
  - Calcul proportion variance expliquée
  - Protection division par zéro

**Rapports** :
- `generate_stratified_report(...)` → Rapports 3 strates (GLOBAL, STABLE, EXPLOSIF)
- Fichiers : metadata.json, summary.txt, analysis_{regime}.json, CSVs

---

### gamma_profiling.py

**Responsabilité** : Profiling comportemental gammas individuels

**Configuration Timelines** :
```python
TIMELINE_THRESHOLDS = {
    'early': 0.20,  # onset < 20% durée
    'mid':   0.60,  # 20% ≤ onset ≤ 60%
    'late':  0.60   # onset > 60%
}
```

**Fonctions Timelines** :
- `classify_timing(onset_relative)` → 'early' | 'mid' | 'late'
- `compute_timeline_descriptor(sequence, timing_relative, oscillatory)` → dict
  - Structure : `{'phases': [...], 'timeline_compact': str, 'n_phases': int}`
  - Format : `{timing}_{event}_then_{event}`
  - Exemples : 'early_instability_then_collapse', 'mid_deviation_then_saturation'

**Fonctions Agrégation** :
- `aggregate_summary_metrics(observations, metric_name)` → dict
  - final_value : {median, q1, q3, mean, std}
  - initial_value, mean_value, cv
  
- `aggregate_run_dispersion(observations, metric_name)` → dict
  - final_value_iqr_ratio, cv_across_runs, bimodal_detected
  
- `aggregate_dynamic_signatures(observations, metric_name)` → dict
  - dynamic_signature : {onsets médians, fractions événements}
  - timeline_distribution : {dominant_timeline, confidence, variants}

**Classification Régime** :
```python
classify_regime(metrics, dynamic_sig, timeline_dist, dispersion, test_name) -> str
```

**Régimes Spécifiques** :
- `CONSERVES_SYMMETRY` (SYM-*) : asymétrie < 1e-6
- `CONSERVES_NORM` (SPE-*, UNIV-*) : norme < 2× initial
- `CONSERVES_PATTERN` (PAT-*) : diversity/uniformity stable
- `CONSERVES_TOPOLOGY` (TOP-*) : euler_characteristic stable
- `CONSERVES_GRADIENT` (GRA-*) : gradient structure conservée
- `CONSERVES_SPECTRUM` (SPA-*) : spectre stable

**Régimes Pathologiques** :
- `NUMERIC_INSTABILITY` : instability_onset < 20 && final > 1e20
- `OSCILLATORY_UNSTABLE` : oscillatory_fraction > 0.3
- `TRIVIAL` : cv < 0.01
- `DEGRADING` : final < 0.5 × initial
- `SATURATES_HIGH` : saturation + final > 10 × initial
- `UNCATEGORIZED` : autres cas

**Qualificatif Multimodalité** :
- Si `bimodal_detected` → `MIXED::{régime_base}`

**Profil PRC** :
```python
compute_prc_profile(...) -> dict
{
    'regime': str,
    'behavior': 'stable' | 'unstable' | 'degrading' | 'mixed',
    'dominant_timeline': {...},
    'robustness': {homogeneous, mixed_behavior, numerically_stable},
    'pathologies': {numeric_instability, oscillatory, collapse, trivial, degrading},
    'n_runs': int,
    'n_valid': int,
    'confidence': 'high' | 'medium' | 'low',
    'confidence_metadata': {...}
}
```

**Confidence Heuristique** :
- **high** : n_valid ≥ 20 && !bimodal && timeline_conf ≥ 0.7
- **medium** : n_valid ≥ 10
- **low** : n_valid < 10

**Fonctions Profiling** :
- `profile_test_for_gamma(observations, test_name, gamma_id)` → dict
  - 3 niveaux : PRC Profile, Diagnostic Signature, Instrumentation
  
- `profile_all_gammas(observations)` → dict
  - Profils tous gammas × tests
  
- `rank_gammas_by_test(profiles, test_name, criterion)` → List[(gamma_id, score)]
  - Critères : 'conservation', 'stability', 'final_value'
  
- `compare_gammas_summary(profiles)` → dict
  - by_regime : {regime: [gammas]}
  - by_test : {test: {best, worst, ranking}}

**Marqueurs Fallback** :
- `instrumentation.data_completeness.fallback_used` : True si dynamic_events absent

---

### verdict_reporter.py

**Responsabilité** : Orchestration génération rapports R0

**Pipeline** :
```python
generate_verdict_report(params_config_id, verdict_config_id) -> dict
```

**Étapes** :
1. **Chargement + Diagnostics** (verdict_engine)
   - load_all_observations()
   - filter_numeric_artifacts()
   - generate_degeneracy_report(), diagnose_scale_outliers()

2. **Analyses Globales Stratifiées** (verdict_engine)
   - stratify_by_regime()
   - analyze_regime() × 3 (GLOBAL, STABLE, EXPLOSIF)

3. **Profiling Gamma** (gamma_profiling)
   - profile_all_gammas()
   - compare_gammas_summary()

4. **Fusion Résultats**
   - Compile metadata, gamma_profiles, structural_patterns, comparisons

5. **Génération Rapports Multi-formats**
   - metadata.json
   - summary.txt (enrichi R0+)
   - gamma_profiles.json + CSV
   - comparisons.json
   - structural_patterns.json
   - diagnostics.json
   - marginal_variance_{regime}.csv

**Structure Finale** :
```json
{
    "metadata": {
        "generated_at": "ISO",
        "engine_version": "5.5",
        "data_summary": {...},
        "quality_flags": {...},
        "analysis_parameters": {...}
    },
    "gamma_profiles": {
        "GAM-001": {
            "tests": {"SYM-001": {...}},
            "summary": {...}
        }
    },
    "structural_patterns": {
        "stratification": {
            "GLOBAL": {...},
            "STABLE": {...},
            "EXPLOSIF": {...}
        }
    },
    "comparisons": {
        "by_regime": {...},
        "by_test": {...}
    },
    "diagnostics": {...},
    "report_paths": {...}
}
```

**Enrichissements R0+** :
- Synthèse régimes transversale (conservation vs pathologies vs mixed)
- Signatures dynamiques par gamma (timelines dominantes)
- Propriétés conservées détectées
- Comparaisons enrichies par propriété (Symétrie, Norme, Pattern, etc.)

---

### applicability.py

**Fonction** : `check(test_module, run_metadata)` → (bool, str)

**Responsabilité** : Vérifier applicabilité test sur métadonnées run

**Validators Disponibles** :
```python
VALIDATORS = {
    'requires_rank': lambda meta, expected: 
        expected is None or len(meta['state_shape']) == expected,
    
    'requires_square': lambda meta, required: 
        not required or (len(meta['state_shape']) == 2 and meta['state_shape'][0] == meta['state_shape'][1]),
    
    'allowed_d_types': lambda meta, allowed: 
        'ALL' in allowed or meta['d_encoding_id'].split('-')[0] in allowed,
    
    'requires_even_dimension': lambda meta, required: 
        not required or all(dim % 2 == 0 for dim in meta['state_shape']),
    
    'minimum_dimension': lambda meta, min_dim:
        min_dim is None or all(dim >= min_dim for dim in meta['state_shape'])
}
```

**Extensibilité** :
- `add_validator(name, validator)` : Ajouter validator custom

**Exemple** :
```python
from tests.utilities.applicability import check

is_applicable, reason = check(test_sym_001, {
    'state_shape': (10, 10),
    'd_encoding_id': 'SYM-001',
    ...
})
# (True, "") si conforme
# (False, "requires_square = True non satisfait") sinon
```

---

### config_loader.py

**Classe** : `ConfigLoader` (singleton via `get_loader()`)

**Responsabilité** : Chargement configs YAML avec fusion global/specific

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

### discovery.py

**Fonction** : `discover_active_tests()` → Dict[str, module]

**Responsabilité** : Découverte automatique tests actifs (non `_deprecated`)

**Validation Structure** :
- `validate_test_structure(module)` → None (raise si invalide)

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

---

## Registries (Sous-module)

### registry_manager.py

**Classe** : `RegistryManager` (singleton)

**Responsabilité** : Gestion centralisée registries de fonctions calcul

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

## Table Récapitulative

| Module | Type | Responsabilité Principale | Sortie Clé |
|--------|------|---------------------------|------------|
| test_engine | Exécution | Moteur tests + dynamic_events | observation dict |
| verdict_engine | Analyse | Analyses statistiques globales | variance, interactions, patterns |
| gamma_profiling | Analyse | Profiling comportemental gammas | régimes, timelines, PRC profiles |
| verdict_reporter | Orchestration | Génération rapports complets | multi-format reports |
| applicability | Validation | Vérifier applicabilité tests | (bool, reason) |
| config_loader | Configuration | Chargement configs YAML | dict fusionné |
| discovery | Discovery | Détection tests actifs | {test_id: module} |
| registry_manager | Registries | Gestion fonctions calcul | Callable |

---

## Flux de Données Typique

```
1. Discovery
   discover_active_tests() → {test_id: module}

2. Applicability Check
   check(test_module, run_metadata) → (True, "")

3. Config Loading
   get_loader().load('params', 'params_default_v1', 'UNIV-001')

4. Test Execution
   TestEngine().execute_test(...)
   ├─ Validate COMPUTATION_SPECS via RegistryManager
   ├─ Execute on snapshots
   ├─ detect_dynamic_events() + compute_event_sequence()
   └─ Return observation dict

5. Verdict Analysis
   verdict_engine.compute_verdict(params_config_id, verdict_config_id)
   ├─ load_all_observations()
   ├─ filter_numeric_artifacts()
   ├─ stratify_by_regime()
   ├─ analyze_regime() × 3
   └─ generate_stratified_report()

6. Gamma Profiling
   gamma_profiling.profile_all_gammas(observations)
   ├─ profile_test_for_gamma() pour chaque gamma × test
   ├─ classify_regime() avec propriété spécifique
   ├─ compute_timeline_descriptor()
   └─ compare_gammas_summary()

7. Report Generation
   verdict_reporter.generate_verdict_report(...)
   ├─ Fusion verdict_engine + gamma_profiling
   ├─ Compile metadata + diagnostics
   └─ Generate multi-format reports
```

---

## Utilisation Exemple

```python
# Discovery + Applicability
from tests.utilities.discovery import discover_active_tests
from tests.utilities.applicability import check

tests = discover_active_tests()
test_module = tests['UNIV-001']

is_applicable, reason = check(test_module, run_metadata)
if not is_applicable:
    print(f"Skip: {reason}")

# Execution
from tests.utilities.test_engine import TestEngine

engine = TestEngine()
result = engine.execute_test(
    test_module,
    run_metadata,
    history,
    'params_default_v1'
)

# Verdict + Profiling (après exécution batch)
from tests.utilities.verdict_reporter import generate_verdict_report

report = generate_verdict_report(
    params_config_id='params_default_v1',
    verdict_config_id='verdict_default_v1'
)

# Accès résultats
print(report['gamma_profiles']['GAM-001'])
print(report['structural_patterns']['stratification']['GLOBAL'])
```

---

## Évolutions Récentes (Charter 5.5)

### Test Engine
- ✅ Détection événements dynamiques (deviation, instability, oscillatory, saturation, collapse)
- ✅ Séquences temporelles avec onsets relatifs
- ✅ Timeseries optionnelles (lourd en stockage)

### Verdict Engine
- ✅ Correction η² (compute_eta_squared)
- ✅ Interactions orientées (permutations pas combinations)
- ✅ Filtrage testabilité renforcé
- ✅ Stratification stable/explosif (threshold 1e50)
- ✅ Diagnostics dégénérescences + ruptures échelle
- ✅ Analyses parallèles 3 strates

### Gamma Profiling
- ✅ Timelines compositionnels relatifs
- ✅ Régimes spécifiques par propriété (CONSERVES_SYMMETRY, etc.)
- ✅ Qualificatif MIXED::régime pour multimodalité
- ✅ Confidence metadata explicite
- ✅ Marqueur fallback timeseries

### Verdict Reporter
- ✅ Orchestration verdict_engine + gamma_profiling
- ✅ Rapports enrichis R0+ (synthèse régimes, signatures dynamiques)
- ✅ Multi-format (JSON, TXT, CSV)
- ✅ Comparaisons par propriété

---

**Version** : 5.5  
**Source** : `tests/utilities/*.py`, `tests/utilities/registries/*.py`  
**Architecture** : Charter R0 - Posture Non Gamma-Centrique