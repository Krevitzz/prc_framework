# HUB_catalog.md

> Catalogue fonctionnel des modules d'orchestration HUB  
> Responsabilité : Orchestration niveau système (tests, analyses, rapports)  
> Version : 6.0  
> Dernière mise à jour : 2025-01-15

---

## VUE D'ENSEMBLE

Le répertoire `tests/utilities/HUB/` contient les **modules d'orchestration** du système PRC.

**Modules catalogués** :
- `test_engine.py` : Moteur exécution tests + détection événements dynamiques
- `verdict_engine.py` : Analyses statistiques multi-facteurs (variance, interactions)
- `verdict_reporter.py` : Orchestration génération rapports complets
- `profiling_runner.py` : Orchestration profiling multi-axes (vide R0, prévu R1)

**Principe fondamental** :
- HUB = orchestration, pas calcul
- Délégation calculs → registries, UTIL, PROFILING
- Compilation résultats + coordination workflows
- Aucune logique métier dans HUB (sauf coordination)

---

## SECTION 1 : test_engine.py

### 1.1 Responsabilités

**Cœur métier** :
- Valider COMPUTATION_SPECS via RegistryManager
- Exécuter formules sur tous snapshots
- Appliquer post_processors
- Calculer statistics/evolution
- Détecter événements dynamiques (NEW R0)
- Retourner dict standardisé format v2

**Version** : 5.5

### 1.2 Classe TestEngine

**Signature** :
```python
class TestEngine:
    VERSION = "5.5"
    
    def __init__(self):
        self.registry_manager = RegistryManager()
        self.computation_cache: Dict[str, Dict] = {}
        self.config_loader = get_loader()
```

**Méthodes principales** :

#### execute_test()

```python
def execute_test(
    self,
    test_module,
    run_metadata: Dict[str, Any],
    history: List[np.ndarray],
    params_config_id: str
) -> Dict[str, Any]
```

**Workflow** :
1. Initialisation résultat (`_init_result`)
2. Chargement params YAML (`config_loader.load`)
3. Préparation computations (`_prepare_computations`)
4. Boucle exécution sur snapshots :
   - Appel registry functions
   - Application post_process
   - Stockage dans buffers
5. **Détection événements dynamiques** (NEW R0)
6. Calcul statistics/evolution (`_compile_results`)
7. Retour dict v2

**Paramètres** :
| Paramètre | Type | Description |
|-----------|------|-------------|
| `test_module` | module | Module test importé |
| `run_metadata` | dict | `{exec_id, gamma_id, d_encoding_id, modifier_id, seed, state_shape}` |
| `history` | `List[np.ndarray]` | Snapshots complets |
| `params_config_id` | str | ID config params |

**Retour** : Dict v2 (voir Charter Section 4)

**Gestion erreurs** :
- `status='ERROR'` → ARRÊT BATCH (bug code/registry)
- `status='NOT_APPLICABLE'` → Continue (contexte invalide ce run)
- `status='SUCCESS'` → Exécution normale

---

### 1.3 Détection événements dynamiques (NEW R0)

#### detect_dynamic_events()

```python
def detect_dynamic_events(values: np.ndarray) -> dict
```

**Événements détectés** :

| Événement | Critère | Type retour |
|-----------|---------|-------------|
| `deviation_onset` | \|val - initial\| > 10% \|initial\| | int \| None (itération) |
| `instability_onset` | \|diff\| > P90(diffs) × 10 | int \| None |
| `oscillatory` | sign_changes > 10% iterations | bool |
| `saturation` | std(last_20%) / mean < 5% | bool |
| `collapse` | any(\|last_10\| < 1e-10) AND max > 1.0 | bool |

**Retour** :
```python
{
    'deviation_onset': int | None,
    'instability_onset': int | None,
    'oscillatory': bool,
    'saturation': bool,
    'collapse': bool
}
```

---

#### compute_event_sequence()

```python
def compute_event_sequence(
    events: dict,
    n_iterations: int
) -> dict
```

**Rôle** : Construire séquence ordonnée + onsets relatifs

**Retour** :
```python
{
    'sequence': ['deviation', 'instability', 'saturation'],
    'sequence_timing': [15, 42, 160],  # Itérations absolues
    'sequence_timing_relative': [0.075, 0.21, 0.8],  # [0,1]
    'saturation_onset_estimated': bool,  # Si saturation = heuristique 80%
    'oscillatory_global': bool
}
```

**Heuristiques R0** :
- `saturation` onset estimé à 80% (pas d'onset ponctuel)
- `collapse` onset estimé à 90%
- `oscillatory` : comportement global (pas dans séquence)

---

#### patch_execute_test_dynamic_events()

```python
def patch_execute_test_dynamic_events(
    metric_buffers: Dict[str, List[float]],
    n_iterations: int
) -> Tuple[dict, dict]
```

**Rôle** : Wrapper calcul événements + timeseries pour tous metrics

**Retour** :
```python
(
    dynamic_events: {
        'metric_name': {
            'deviation_onset': ...,
            'sequence': [...],
            ...
        }
    },
    timeseries: {
        'metric_name': [val_0, ..., val_N]
    }
)
```

**Usage** : Appelé dans `execute_test()` après boucle snapshots

---

### 1.4 Méthodes utilitaires

#### _init_result()

```python
def _init_result(
    self,
    test_module,
    run_metadata: Dict,
    params_config_id: str
) -> Dict
```

**Rôle** : Initialise structure dict v2 avec `exec_id`

**Retour** : Dict avec clés :
- `exec_id`, `run_metadata`, `test_name`, `test_category`, `test_version`
- `config_params_id`, `status`, `message`
- `statistics`, `evolution`, `dynamic_events`, `timeseries`, `metadata`

---

#### _prepare_computations()

```python
def _prepare_computations(
    self,
    specs: Dict,
    config_id: str
) -> Dict
```

**Rôle** : Valider + cacher COMPUTATION_SPECS

**Workflow** :
1. Pour chaque métrique dans specs :
2. Vérifier cache (hash spec)
3. Si absent : valider via `registry_manager.validate_computation_spec()`
4. Cacher résultat
5. Retourner dict computations prêtes

**Gestion erreurs** : Skip métrique invalide (log warning, continue)

---

#### _compile_results()

```python
def _compile_results(
    self,
    result: Dict,
    buffers: Dict,
    skipped: Dict,
    computations: Dict,
    exec_time: float,
    params: dict,
    dynamic_events: dict,  # NEW
    timeseries: dict       # NEW
) -> Dict
```

**Rôle** : Compiler résultats finaux (statistics, evolution, metadata)

**Calculs** :

**Statistics** (par métrique) :
- initial, final, min, max, mean, std, median, q1, q3, n_valid

**Evolution** (par métrique, via `_analyze_evolution`) :
- transition : explosive, stable, growing, shrinking, oscillating
- trend : stable, increasing, decreasing
- slope, volatility, relative_change

**Metadata** :
- execution_time_sec
- num_iterations_processed
- total_metrics, successful_metrics
- skipped_iterations
- computations (registry_keys, params_used, has_post_process)

---

#### _analyze_evolution()

```python
def _analyze_evolution(self, values: List[float], params: dict) -> Dict
```

**Rôle** : Analyser évolution série temporelle

**Paramètres config** (depuis params YAML) :
- `explosion_threshold` : 1000.0 (défaut)
- `stability_tolerance` : 0.1
- `growth_factor` : 1.5
- `shrink_factor` : 0.5
- `epsilon` : 1e-10

**Logique** :

**Tendance** (slope linéaire) :
- abs(slope) < epsilon → "stable"
- slope > 0 → "increasing"
- slope < 0 → "decreasing"

**Transition** :
- max_val > explosion_threshold → "explosive"
- relative_change < stability_tolerance → "stable"
- final > initial × growth_factor → "growing"
- final < initial × shrink_factor → "shrinking"
- sinon → "oscillating"

**Retour** :
```python
{
    'transition': str,
    'trend': str,
    'slope': float,
    'volatility': float,
    'relative_change': float
}
```

---

### 1.5 Graphe dépendances test_engine

```
test_engine.py
    ├─ Appelé par : batch_runner.py (mode --test)
    ├─ Dépend de :
    │   ├─ registries/registry_manager.py (validation specs)
    │   ├─ config_loader.py (chargement params YAML)
    │   └─ (calculs internes, pas de dépendances UTIL)
    └─ Retourne : dict v2 → db_results (TestObservations)
```

---

## SECTION 2 : verdict_engine.py

### 2.1 Responsabilités

**Cœur métier** :
- Analyses variance marginale (η²)
- Analyses interactions orientées (A|B ≠ B|A)
- Discrimination métriques (CV)
- Corrélations
- Interprétation patterns
- Décision verdict (GLOBAL + par gamma)
- Pipeline régime complet (stratification)

**Version** : 5.5

**Architecture refactorisée (Phase 2.1-2.3)** :
- Délégation I/O → data_loading.py
- Délégation filtrage/diagnostics → statistical_utils.py
- Délégation stratification → regime_utils.py
- Cœur métier : analyses statistiques

---

### 2.2 Configuration

**Constantes globales** :
```python
FACTORS = [
    'gamma_id',
    'd_encoding_id',
    'modifier_id',
    'seed',
    'test_name'
]
# params_config_id EXCLU (trop corrélé test_name)

PROJECTIONS = [
    'value_final',
    'value_mean',
    'slope',
    'volatility',
    'relative_change'
]

# Seuils testabilité
MIN_SAMPLES_PER_GROUP = 2
MIN_GROUPS = 2
MIN_TOTAL_SAMPLES = 10
```

---

### 2.3 Fonction analyze_marginal_variance()

**Signature** :
```python
def analyze_marginal_variance(
    df: pd.DataFrame,
    factors: List[str],
    projections: List[str]
) -> pd.DataFrame
```

**Rôle** : Analyser variance marginale (chaque facteur isolément)

**Métrique** : η² (eta-squared)
```
η² = SSB / (SSB + SSW)
où SSB = variance inter-groupes
    SSW = variance intra-groupes
```

**Workflow** :
1. Pour chaque (projection, test_name, metric_name, factor) :
2. Filtrer testabilité (MIN_TOTAL_SAMPLES, MIN_GROUPS)
3. Grouper par factor
4. Calculer η² via `compute_eta_squared()` (statistical_utils)
5. Test Kruskal-Wallis (statistical_utils)
6. Stocker résultats

**Retour** : DataFrame colonnes :
- test_name, metric_name, projection, factor
- variance_ratio (η²), p_value, n_groups, significant

**Filtrage testabilité** :
- Contexte global : ≥ MIN_TOTAL_SAMPLES observations
- Par groupe : ≥ MIN_SAMPLES_PER_GROUP observations
- Facteur : ≥ MIN_GROUPS niveaux distincts

---

### 2.4 Fonction analyze_oriented_interactions()

**Signature** :
```python
def analyze_oriented_interactions(
    df: pd.DataFrame,
    factors: List[str],
    projections: List[str],
    marginal_variance: pd.DataFrame,
    min_interaction_strength: float = 2.0
) -> pd.DataFrame
```

**Rôle** : Détecter interactions ORIENTÉES (A|B ≠ B|A)

**Différence vs combinations** :
- Paires orientées : (A, B) ET (B, A) testées séparément
- Total : len(factors) × (len(factors) - 1) paires
- Interaction orientée : effet de A change selon contexte B

**Critères détection interaction** :
1. VR(A|B=b) significatif (η² > 0.3, p < 0.05)
2. VR(A|B=b) >> VR(A) marginal (ratio > min_interaction_strength)
3. Testabilité robuste (≥3 niveaux, ≥5 obs/groupe)
4. VR marginal substantiel (≥ 0.1)

**Workflow** :
1. Pour chaque paire orientée (factor_varying, factor_context) :
2. Pour chaque niveau context_value du factor_context :
3. Calculer VR(factor_varying | context_value)
4. Comparer avec VR(factor_varying) marginal
5. Si ratio > min_interaction_strength → interaction détectée

**Retour** : DataFrame colonnes :
- factor_varying, factor_context, context_value
- test_name, metric_name, projection
- vr_conditional, vr_marginal, interaction_strength
- p_value, n_groups

---

### 2.5 Fonction analyze_metric_discrimination()

**Signature** :
```python
def analyze_metric_discrimination(
    df: pd.DataFrame,
    projections: List[str]
) -> pd.DataFrame
```

**Rôle** : Détecter métriques non discriminantes (CV < 0.1)

**Métrique** : Coefficient de variation
```
CV = std / |mean|
```

**Critères** :
- CV < 0.1 → non_discriminant = True
- Nécessite ≥ 5 observations

**Retour** : DataFrame colonnes :
- test_name, metric_name, projection
- mean, std, cv, n_observations, non_discriminant

---

### 2.6 Fonction analyze_metric_correlations()

**Signature** :
```python
def analyze_metric_correlations(
    df: pd.DataFrame,
    projection: str = 'value_final',
    threshold: float = 0.8
) -> pd.DataFrame
```

**Rôle** : Détecter corrélations fortes entre métriques (même test)

**Métrique** : Corrélation Spearman (robuste outliers)

**Workflow** :
1. Pour chaque test_name :
2. Pivoter observations (index = runs, colonnes = metrics)
3. Calculer corrélations pairwise
4. Si |corr| > threshold → redondance détectée

**Retour** : DataFrame colonnes :
- test_name, metric1, metric2
- correlation, p_value, n_observations

---

### 2.7 Fonction interpret_patterns()

**Signature** :
```python
def interpret_patterns(
    df: pd.DataFrame,
    marginal_variance: pd.DataFrame,
    oriented_interactions: pd.DataFrame,
    discrimination: pd.DataFrame,
    correlations: pd.DataFrame
) -> Tuple[dict, dict]
```

**Rôle** : Synthèse patterns GLOBAL + PAR GAMMA

**Patterns globaux** :
- `marginal_dominant` : variance_ratio > 0.5, significant
- `oriented_interactions` : interactions vraies détectées
- `non_discriminant` : CV < 0.1
- `redundant` : corrélations > threshold

**Patterns par gamma** (drill-down RECALCULÉ) :
- Sous-ensemble strict gamma_df
- Recalcul analyses sur ce gamma uniquement
- Exclusion gamma_id des factors analysés

**Retour** :
```python
(
    patterns_global: {
        'marginal_dominant': [...],
        'oriented_interactions': [...],
        'non_discriminant': [...],
        'redundant': [...]
    },
    patterns_by_gamma: {
        'GAM-001': {
            'marginal_dominant': [...],
            'oriented_interactions': [...]
        }
    }
)
```

---

### 2.8 Fonction decide_verdict()

**Signature** :
```python
def decide_verdict(
    patterns_global: dict,
    patterns_by_gamma: dict
) -> Tuple[str, str, dict]
```

**Rôle** : Décision verdict GLOBAL + PAR GAMMA

**Logique verdict global** :
- Aucun pattern critique → "SURVIVES[R0]"
- Patterns critiques détectés → "WIP[R0-open]" + raisons

**Verdicts par gamma** :
- Identique logique (contexte gamma spécifique)

**Retour** :
```python
(
    verdict_global: str,
    reason_global: str,
    verdicts_by_gamma: {
        'GAM-001': {
            'verdict': str,
            'reason': str,
            'patterns': dict
        }
    }
)
```

---

### 2.9 Fonction analyze_regime()

**Signature** :
```python
def analyze_regime(
    observations: List[dict],
    regime_name: str,
    params_config_id: str,
    verdict_config_id: str
) -> dict
```

**Rôle** : Pipeline analyse complet sur une strate

**Workflow** : Identique pipeline global
1. Conversion observations → DataFrame
2. Filtrage testabilité
3. analyze_marginal_variance
4. analyze_oriented_interactions
5. analyze_metric_discrimination
6. analyze_metric_correlations
7. interpret_patterns
8. decide_verdict

**Retour** :
```python
{
    'regime': str,  # 'GLOBAL', 'STABLE', 'EXPLOSIF'
    'n_observations': int,
    'status': str,  # 'SUCCESS', 'INSUFFICIENT_DATA'
    'marginal_variance': pd.DataFrame,
    'oriented_interactions': pd.DataFrame,
    'discrimination': pd.DataFrame,
    'correlations': pd.DataFrame,
    'patterns_global': dict,
    'patterns_by_gamma': dict,
    'verdict': str,
    'reason': str,
    'verdicts_by_gamma': dict
}
```

---

### 2.10 Fonction compute_verdict() (PIPELINE)

**Signature** :
```python
def compute_verdict(
    params_config_id: str,
    verdict_config_id: str
) -> None
```

**Rôle** : Pipeline complet verdict exploratoire

**Workflow** :
1. Chargement observations (`load_all_observations`)
2. Filtrage artefacts (`filter_numeric_artifacts`)
3. Diagnostics (`generate_degeneracy_report`, `diagnose_scale_outliers`)
4. Stratification (`stratify_by_regime`)
5. Analyses parallèles 3 strates (`analyze_regime` × 3)
6. Génération rapports (`generate_stratified_report`)

**Affichage console** : Progression + résumés

---

### 2.11 Graphe dépendances verdict_engine

```
verdict_engine.py
    ├─ Appelé par : verdict_reporter.py, batch_runner.py (mode --verdict)
    ├─ Dépend de :
    │   ├─ data_loading.py (load_all_observations, observations_to_dataframe)
    │   ├─ statistical_utils.py (compute_eta_squared, filter_numeric_artifacts, etc.)
    │   ├─ regime_utils.py (stratify_by_regime)
    │   └─ config_loader.py
    └─ Retourne : dict analyses (marginal_variance, interactions, patterns, verdict)
```

---

## SECTION 3 : verdict_reporter.py

### 3.1 Responsabilités

**Cœur métier** :
- Orchestration pipeline complet (5 étapes)
- Compilation metadata
- Formatage structures gamma_profiles
- Compilation structural_patterns
- Coordination génération rapports

**Version** : 5.5

**Architecture refactorisée (Phase 2.3)** :
- Délégation I/O → data_loading.py
- Délégation diagnostics → statistical_utils.py
- Délégation stratification → regime_utils.py
- Délégation formatage → report_writers.py
- Délégation profiling → gamma_profiling.py
- Cœur : orchestration + compilation

---

### 3.2 Fonction generate_verdict_report() (PIPELINE PRINCIPAL)

**Signature** :
```python
def generate_verdict_report(
    params_config_id: str,
    verdict_config_id: str,
    output_dir: str = "reports/verdicts"
) -> dict
```

**Rôle** : Pipeline complet génération rapport verdict R0

**Workflow 5 étapes** :

**ÉTAPE 1** : Chargement + diagnostics
- `load_all_observations()` → data_loading
- `filter_numeric_artifacts()` → statistical_utils
- `generate_degeneracy_report()` → statistical_utils
- `diagnose_scale_outliers()` → statistical_utils

**ÉTAPE 2** : Analyses globales stratifiées
- `stratify_by_regime()` → regime_utils
- `analyze_regime()` × 3 (GLOBAL, STABLE, EXPLOSIF) → verdict_engine

**ÉTAPE 3** : Profiling gamma
- `profile_all_gammas()` → gamma_profiling
- `compare_gammas_summary()` → gamma_profiling

**ÉTAPE 4** : Fusion résultats
- `_compile_metadata()` (local)
- `_format_gamma_profiles()` (local)
- `_compile_structural_patterns()` (local)

**ÉTAPE 5** : Génération rapports
- `write_json()` → report_writers
- `_write_summary_report()` (partiellement délégué)
- `_write_gamma_profiles()` (partiellement délégué)

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

### 3.3 Fonctions compilation (LOCAL - orchestration)

#### _compile_metadata()

```python
def _compile_metadata(
    params_config_id: str,
    verdict_config_id: str,
    observations: list,
    rejection_stats: dict,
    degeneracy_report: dict,
    scale_report: dict
) -> dict
```

**Rôle** : Compiler métadonnées rapport

**Contenu** :
- generated_at, engine_version, architecture
- configs_used
- data_summary (total_observations, n_gammas, n_tests, etc.)
- quality_flags (degeneracy, scale_outliers)
- analysis_parameters (factors, projections, thresholds)

---

#### _format_gamma_profiles()

```python
def _format_gamma_profiles(gamma_profiles: dict) -> dict
```

**Rôle** : Formater gamma_profiles pour rapport

**Structure Charter R0** :
```python
{
    'GAM-001': {
        'tests': {
            'SYM-001': {
                'regime': 'CONSERVES_X',
                'behavior': 'stable',
                'timeline': 'early_deviation_then_saturation',
                'timeline_confidence': 'high',
                'confidence': 'high',
                'n_runs': 10,
                'n_valid': 10,
                'pathologies': {...},
                'robustness': {...}
            }
        },
        'summary': {
            'n_tests': 5,
            'dominant_regime': 'CONSERVES_NORM',
            'regime_distribution': {...}
        }
    }
}
```

---

#### _compile_structural_patterns()

```python
def _compile_structural_patterns(
    results_global: dict,
    results_stable: dict,
    results_explosif: dict
) -> dict
```

**Rôle** : Compiler patterns structuraux (analyses globales)

**Structure** :
```python
{
    'stratification': {
        'GLOBAL': {
            'n_observations': int,
            'status': str,
            'verdict': str,
            'reason': str,
            'patterns': dict
        },
        'STABLE': {...},
        'EXPLOSIF': {...}
    }
}
```

---

### 3.4 Fonctions génération fichiers

#### _write_summary_report()

```python
def _write_summary_report(report_dir: Path, results: dict)
```

**Rôle** : Écrire rapport humain principal (ENRICHI R0+)

**Sections** (partiellement déléguées) :
- Header (`write_header` → report_writers)
- Data summary
- Quality flags
- Regime synthesis (`write_regime_synthesis` → report_writers)
- Dynamic signatures (`write_dynamic_signatures` → report_writers)
- Gamma profiles (résumé)
- Comparisons (`write_comparisons_enriched` → report_writers)
- Structural patterns
- Footer (`write_consultation_footer` → report_writers)

---

#### _write_gamma_profiles()

```python
def _write_gamma_profiles(report_dir: Path, gamma_profiles: dict)
```

**Rôle** : Écrire gamma_profiles.json + CSV

**Fichiers générés** :
- `gamma_profiles.json` : Complet (`write_json` → report_writers)
- `gamma_profiles.csv` : Vue tabulaire (colonnes : gamma_id, test_name, regime, behavior, timeline, etc.)

---

### 3.5 Graphe dépendances verdict_reporter

```
verdict_reporter.py
    ├─ Appelé par : batch_runner.py (mode --verdict)
    ├─ Orchestre :
    │   ├─ data_loading.py (I/O observations)
    │   ├─ statistical_utils.py (filtrage, diagnostics)
    │   ├─ regime_utils.py (stratification)
    │   ├─ verdict_engine.py (analyses globales)
    │   ├─ gamma_profiling.py (profiling comportemental)
    │   └─ report_writers.py (formatage rapports)
    └─ Génère : Rapports multi-formats dans reports/verdicts/TIMESTAMP/
```

---

## SECTION 4 : profiling_runner.py

### 4.1 État R0

**Fichier** : Vide (placeholder R1)

**Responsabilité prévue R1** :
- Orchestration profiling multi-axes
- Découverte automatique modules profiling
- Validation format retour unifié
- Coordination exécution (gamma, modifier, encoding, test)

**Référence** : Voir Charter Section 5.1

---

## ANNEXE A : FORMAT RETOUR TEST_ENGINE (dict v2)

Voir `tests_catalog.md` Section 9.2 pour structure complète.

**Résumé clés essentielles** :
```python
{
    'exec_id': int,  # NEW (traçabilité DB)
    'run_metadata': {...},
    'test_name': str,
    'status': 'SUCCESS' | 'ERROR' | 'NOT_APPLICABLE',
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
            'n_valid': int
        }
    },
    'evolution': {
        'metric_name': {
            'transition': str,  # explosive, stable, growing, shrinking, oscillating
            'trend': str,  # stable, increasing, decreasing
            'slope': float,
            'volatility': float,
            'relative_change': float
        }
    },
    'dynamic_events': {  # NEW R0
        'metric_name': {
            'deviation_onset': int | None,
            'instability_onset': int | None,
            'oscillatory': bool,
            'saturation': bool,
            'collapse': bool,
            'sequence': List[str],
            'sequence_timing': List[int],
            'sequence_timing_relative': List[float]
        }
    },
    'timeseries': {  # NEW R0 (optionnel, lourd)
        'metric_name': [val_0, ..., val_N]
    },
    'metadata': {...}
}
```
## ANNEXE B : WORKFLOWS COMPLETS

### B.1 Workflow batch_runner --test

```
1. batch_runner.py (mode --test)
   ↓
2. Lecture db_raw (Executions, Snapshots)
   └─> history = [snapshot_0, ..., snapshot_N]
   ↓
3. Découverte tests applicables
   └─> discovery.discover_test_modules()
   └─> applicability.check_applicability(test_module, run_metadata)
   ↓
4. Pour chaque test applicable :
   ↓
5. test_engine.execute_test()
   ├─> Chargement params YAML (config_loader)
   ├─> Validation COMPUTATION_SPECS (registry_manager)
   ├─> Boucle snapshots :
   │   ├─> Appel registry functions
   │   ├─> Post-processing
   │   └─> Stockage buffers
   ├─> Détection événements dynamiques (detect_dynamic_events)
   ├─> Calcul statistics/evolution
   └─> Retour dict v2
   ↓
6. Stockage db_results (TestObservations)
   └─> observation_id, exec_id, test_name, params_config_id, observation_data (JSON)
```

---

### B.2 Workflow batch_runner --verdict

```
1. batch_runner.py (mode --verdict)
   ↓
2. verdict_reporter.generate_verdict_report()
   ↓
   ├─ ÉTAPE 1 : Chargement + diagnostics
   │   ├─> load_all_observations() (data_loading)
   │   ├─> filter_numeric_artifacts() (statistical_utils)
   │   ├─> generate_degeneracy_report() (statistical_utils)
   │   └─> diagnose_scale_outliers() (statistical_utils)
   │
   ├─ ÉTAPE 2 : Analyses globales stratifiées
   │   ├─> stratify_by_regime() (regime_utils)
   │   │   └─> obs_stable, obs_explosif
   │   ├─> analyze_regime(observations, 'GLOBAL') (verdict_engine)
   │   ├─> analyze_regime(obs_stable, 'STABLE') (verdict_engine)
   │   └─> analyze_regime(obs_explosif, 'EXPLOSIF') (verdict_engine)
   │       │
   │       └─> Pour chaque strate :
   │           ├─> analyze_marginal_variance()
   │           ├─> analyze_oriented_interactions()
   │           ├─> analyze_metric_discrimination()
   │           ├─> analyze_metric_correlations()
   │           ├─> interpret_patterns()
   │           └─> decide_verdict()
   │
   ├─ ÉTAPE 3 : Profiling gamma
   │   ├─> profile_all_gammas() (gamma_profiling)
   │   └─> compare_gammas_summary() (gamma_profiling)
   │
   ├─ ÉTAPE 4 : Fusion résultats
   │   ├─> _compile_metadata()
   │   ├─> _format_gamma_profiles()
   │   └─> _compile_structural_patterns()
   │
   └─ ÉTAPE 5 : Génération rapports
       ├─> write_json() × N (report_writers)
       ├─> _write_summary_report()
       │   ├─> write_header() (report_writers)
       │   ├─> write_regime_synthesis() (report_writers)
       │   ├─> write_dynamic_signatures() (report_writers)
       │   ├─> write_comparisons_enriched() (report_writers)
       │   └─> write_consultation_footer() (report_writers)
       └─> _write_gamma_profiles()
           ├─> write_json() (report_writers)
           └─> CSV export
```

---

### B.3 Workflow détection événements dynamiques

```
test_engine.execute_test()
   ↓
   [Boucle snapshots terminée]
   ↓
   metric_buffers = {
       'frobenius_norm': [val_0, ..., val_199],
       'asymmetry_norm': [val_0, ..., val_199]
   }
   ↓
patch_execute_test_dynamic_events(metric_buffers, n_iterations=200)
   ↓
   Pour chaque métrique :
       ↓
       detect_dynamic_events(values)
           ├─> Détecte deviation_onset (|val - initial| > 10%)
           ├─> Détecte instability_onset (|diff| > P90×10)
           ├─> Détecte oscillatory (sign_changes > 10% iterations)
           ├─> Détecte saturation (std(last_20%)/mean < 5%)
           ├─> Détecte collapse (|last_10| < 1e-10 AND max > 1.0)
           └─> Retourne events dict
       ↓
       compute_event_sequence(events, n_iterations)
           ├─> Construit sequence ordonnée
           ├─> Calcule onsets relatifs [0,1]
           ├─> Estime onsets heuristiques (saturation: 80%, collapse: 90%)
           └─> Retourne sequence dict
   ↓
Retourne (dynamic_events, timeseries)
   ↓
Intégration dans dict v2 :
   result['dynamic_events'] = dynamic_events
   result['timeseries'] = timeseries
```

---

### B.4 Workflow analyse variance marginale

```
analyze_marginal_variance(df, factors, projections)
   ↓
   Pour chaque (projection, test_name, metric_name, factor) :
       ↓
       Filtrage testabilité :
           ├─> len(group) >= MIN_TOTAL_SAMPLES (10)
           ├─> group[factor].nunique() >= MIN_GROUPS (2)
           └─> min_group_size >= MIN_SAMPLES_PER_GROUP (2)
       ↓
       Groupement par factor :
           factor_groups = [group1_values, group2_values, ...]
       ↓
       Test Kruskal-Wallis :
           statistic, p_value = kruskal_wallis_test(factor_groups)
       ↓
       Calcul η² :
           variance_ratio, ssb, ssw = compute_eta_squared(factor_groups)
               ├─> SSB = Σ n_k × (mean_k - grand_mean)²
               ├─> SSW = Σ Σ (x_ki - mean_k)²
               └─> η² = SSB / (SSB + SSW)
       ↓
       Stockage résultat :
           {test_name, metric_name, projection, factor,
            variance_ratio, p_value, n_groups, significant}
   ↓
Retourne DataFrame trié par variance_ratio (descendant)
```

---

### B.5 Workflow interactions orientées

```
analyze_oriented_interactions(df, factors, projections, marginal_variance)
   ↓
   Génération paires orientées :
       len(factors) × (len(factors) - 1) = N × (N-1) paires
       Ex: (gamma, modifier), (modifier, gamma), (gamma, seed), ...
   ↓
   Pour chaque (factor_varying, factor_context) :
       ↓
       Pour chaque context_value du factor_context :
           ↓
           Sous-ensemble : df[df[factor_context] == context_value]
           ↓
           Calcul VR(factor_varying | context_value) :
               ├─> Groupement par factor_varying dans ce contexte
               ├─> Calcul η² conditionnel
               └─> Test Kruskal-Wallis
           ↓
           Récupération VR(factor_varying) marginal :
               vr_marginal = marginal_index[(test, metric, proj, factor_varying)]
           ↓
           Calcul force interaction :
               interaction_strength = vr_conditional / vr_marginal
           ↓
           Critères détection :
               ├─> vr_conditional > 0.3
               ├─> vr_marginal > 0.1
               ├─> p_value < 0.05
               ├─> interaction_strength > 2.0
               ├─> n_groups >= 3
               └─> min_group_size >= 5
           ↓
           Si tous critères → stockage interaction
   ↓
Retourne DataFrame interactions orientées
```

---

## ANNEXE C : PATTERNS USAGE

### C.1 Test Engine - Usage basique

```python
from tests.utilities.HUB.test_engine import TestEngine

# Initialisation
engine = TestEngine()

# Exécution test
import tests.test_uni_001 as test_module

run_metadata = {
    'exec_id': 123,
    'gamma_id': 'GAM-001',
    'd_encoding_id': 'SYM-001',
    'modifier_id': 'M0',
    'seed': 42,
    'state_shape': (50, 50)
}

history = [np.random.randn(50, 50) for _ in range(200)]

result = engine.execute_test(
    test_module,
    run_metadata,
    history,
    params_config_id='params_default_v1'
)

# Exploitation résultat
if result['status'] == 'SUCCESS':
    print(f"Norme finale: {result['statistics']['frobenius_norm']['final']}")
    print(f"Transition: {result['evolution']['frobenius_norm']['transition']}")
    print(f"Événements: {result['dynamic_events']['frobenius_norm']['sequence']}")
```

---

### C.2 Verdict Engine - Analyse régime

```python
from tests.utilities.HUB.verdict_engine import analyze_regime
from tests.utilities.data_loading import load_all_observations

# Chargement observations
observations = load_all_observations('params_default_v1')

# Analyse régime GLOBAL
results = analyze_regime(
    observations,
    regime_name='GLOBAL',
    params_config_id='params_default_v1',
    verdict_config_id='verdict_default_v1'
)

# Exploitation patterns
if results['status'] == 'SUCCESS':
    print(f"Verdict: {results['verdict']}")
    print(f"Raison: {results['reason']}")
    
    # Variance marginale dominante
    dominant = results['marginal_variance'][
        results['marginal_variance']['variance_ratio'] > 0.5
    ]
    print(f"Facteurs dominants: {dominant['factor'].unique()}")
    
    # Interactions orientées
    if not results['oriented_interactions'].empty:
        print(f"Interactions détectées: {len(results['oriented_interactions'])}")
```

---

### C.3 Verdict Reporter - Génération rapport

```python
from tests.utilities.HUB.verdict_reporter import generate_verdict_report

# Pipeline complet
results = generate_verdict_report(
    params_config_id='params_default_v1',
    verdict_config_id='verdict_default_v1',
    output_dir='reports/verdicts'
)

# Exploitation
print(f"Rapport généré: {results['report_paths']['summary']}")
print(f"Gammas profilés: {len(results['gamma_profiles'])}")
print(f"Fichiers créés: {len(results['report_paths'])}")

# Accès gamma profiles
for gamma_id, profile in results['gamma_profiles'].items():
    summary = profile['summary']
    print(f"{gamma_id}: {summary['dominant_regime']} ({summary['n_tests']} tests)")
```

---

### C.4 Détection événements - Analyse manuelle

```python
from tests.utilities.HUB.test_engine import (
    detect_dynamic_events,
    compute_event_sequence
)

# Trajectoire métrique
values = np.array([1.0, 1.05, 1.2, 1.8, 2.5, 2.48, 2.49, 2.50])  # Exemple

# Détection
events = detect_dynamic_events(values)
print(f"Deviation onset: {events['deviation_onset']}")
print(f"Oscillatory: {events['oscillatory']}")
print(f"Saturation: {events['saturation']}")

# Séquence
seq_info = compute_event_sequence(events, n_iterations=len(values))
print(f"Séquence: {seq_info['sequence']}")
print(f"Timings relatifs: {seq_info['sequence_timing_relative']}")
```

---

## ANNEXE D : EXTENSIONS FUTURES

### D.1 Test Engine

**Extensions autorisées R1+** :

**Ajout post-processors** :
```python
# tests/utilities/registries/post_processors.py
def log_scale(value):
    """Post-processor logarithmique."""
    return np.log10(abs(value) + 1e-10)

# Enregistrement
POST_PROCESSORS['log_scale'] = log_scale

# Usage dans test
COMPUTATION_SPECS = {
    'metric': {
        'post_process': 'log_scale'
    }
}
```

**Ajout détection événements** :
```python
# Nouvel événement R1 : plateau
def detect_plateau(values, window=20, threshold=0.01):
    """Détecte plateau (variance faible sur fenêtre)."""
    for i in range(len(values) - window):
        segment = values[i:i+window]
        if np.std(segment) / (abs(np.mean(segment)) + 1e-10) < threshold:
            return i
    return None

# Intégration dans detect_dynamic_events()
```

---

### D.2 Verdict Engine

**Extensions autorisées R1+** :

**Ajout facteurs analysés** :
```python
# NOUVEAU facteur (si justifié)
FACTORS = [
    'gamma_id',
    'd_encoding_id',
    'modifier_id',
    'seed',
    'test_name',
    'dimension'  # NOUVEAU R1 (si pertinent)
]
```

**Ajout projections** :
```python
# NOUVELLE projection
PROJECTIONS = [
    'value_final',
    'value_mean',
    'slope',
    'volatility',
    'relative_change',
    'max_amplitude'  # NOUVEAU R1
]
```

**Ajout analyses** :
```python
# Nouvelle analyse R1 : interactions 3-way
def analyze_three_way_interactions(df, factors, projections):
    """
    Détecte interactions A × B × C.
    
    Logique : VR(A | B=b, C=c) >> VR(A | B=b)
    """
    # Implémentation...
    pass
```

---

### D.3 Verdict Reporter

**Extensions autorisées R1+** :

**Ajout sections rapport** :
```python
# Nouvelle section R1
def _write_hypothesis_tracking(report_dir, gamma_profiles):
    """
    Génère section tracking hypothèses (CON).
    
    Format :
    - Statut HYP par gamma
    - Évolution statuts (WIP → SURVIVES → ...)
    - Justifications décisions
    """
    pass

# Intégration dans generate_verdict_report()
```

**Ajout formats export** :
```python
# Export HTML interactif R1
def export_interactive_report(results, output_path):
    """
    Génère rapport HTML avec visualisations interactives.
    
    Technologies : Plotly, D3.js, ou similar
    """
    pass
```

---

### D.4 Profiling Runner (R1)

**Implémentation prévue R1** :

```python
# tests/utilities/HUB/profiling_runner.py

PROFILING_AXES = {
    'test': {
        'profile_func': profiling_common.profile_all_tests,
        'summary_func': profiling_common.compare_tests_summary
    },
    'gamma': {
        'profile_func': profiling_common.profile_all_gammas,
        'summary_func': profiling_common.compare_gammas_summary
    },
    'modifier': {
        'profile_func': profiling_common.profile_all_modifiers,
        'summary_func': profiling_common.compare_modifiers_summary
    },
    'encoding': {
        'profile_func': profiling_common.profile_all_encodings,
        'summary_func': profiling_common.compare_encodings_summary
    }
}

def run_all_profiling(observations, axes=None):
    """
    Orchestre profiling multi-axes.
    
    Args:
        observations: Liste observations
        axes: Liste axes à profiler (None = tous)
    
    Returns:
        {
            'test': {...},
            'gamma': {...},
            'modifier': {...},
            'encoding': {...}
        }
    """
    if axes is None:
        axes = list(PROFILING_AXES.keys())
    
    results = {}
    
    for axis in axes:
        print(f"Profiling {axis}...")
        
        axis_config = PROFILING_AXES[axis]
        profile_func = axis_config['profile_func']
        summary_func = axis_config['summary_func']
        
        # Profiling
        profiles = profile_func(observations)
        
        # Comparaisons
        summary = summary_func(profiles)
        
        # Validation format unifié (R5.1-A)
        _validate_profiling_output(profiles, axis)
        
        results[axis] = {
            'profiles': profiles,
            'summary': summary
        }
    
    return results
```

---

## ANNEXE E : INTERDICTIONS CRITIQUES

### E.1 Test Engine

**INTERDIT** :

❌ **Validation sémantique dans engine** :
```python
# INTERDIT
def execute_test(...):
    if not is_symmetric(snapshot):  # ❌
        raise ValueError("État non symétrique")
```

❌ **Branchement basé sur gamma_id** :
```python
# INTERDIT
if run_metadata['gamma_id'] == 'GAM-001':  # ❌
    # Traitement spécial
```

❌ **Modification run_metadata** :
```python
# INTERDIT
run_metadata['custom_field'] = value  # ❌ Mutation
```

❌ **Skip test silencieux** :
```python
# INTERDIT
if not applicable:
    return None  # ❌ Doit retourner dict avec status='NOT_APPLICABLE'
```

---

### E.2 Verdict Engine

**INTERDIT** :

❌ **Hardcoding listes entités** :
```python
# INTERDIT
GAMMAS_TO_ANALYZE = ['GAM-001', 'GAM-002']  # ❌ Découverte dynamique obligatoire
```

❌ **Modification observations** :
```python
# INTERDIT
def analyze_regime(observations, ...):
    observations.append(fake_obs)  # ❌ Mutation input
```

❌ **Verdict décisionnel automatique** :
```python
# INTERDIT
if variance_ratio > 0.8:
    return "REJECTED[R0]"  # ❌ Verdict = décision humaine post-rapport
```

❌ **Filtrage données basé sur verdict** :
```python
# INTERDIT
observations_filtered = [
    obs for obs in observations
    if should_keep(obs)  # ❌ Conservation intégrale obligatoire
]
```

---

### E.3 Verdict Reporter

**INTERDIT** :

❌ **Implémentation logique analytique** :
```python
# INTERDIT
def generate_verdict_report(...):
    # Calcul variance manuelle
    variance = np.var(values)  # ❌ Délégation verdict_engine obligatoire
```

❌ **Génération rapports sans orchestration** :
```python
# INTERDIT (bypass orchestration)
def quick_report(...):
    write_json(data, path)  # ❌ Sans compilation metadata/validation
```

❌ **Modification résultats analyses** :
```python
# INTERDIT
results_global['verdict'] = 'SURVIVES[R0]'  # ❌ Altération verdict
```

---

## ANNEXE F : HISTORIQUE MODIFICATIONS

| Date | Version | Module | Changement |
|------|---------|--------|------------|
| 2025-01-15 | 6.0.0 | Tous | Création catalogue initial |
| 2025-01-15 | 5.5 | test_engine | Ajout détection événements dynamiques |
| 2025-01-15 | 5.5 | verdict_engine | Refactorisation Phase 2.1 (délégation UTIL) |
| 2025-01-15 | 5.5 | verdict_engine | Correction interactions orientées (permutations) |
| 2025-01-15 | 5.5 | verdict_reporter | Refactorisation Phase 2.3 (orchestration pure) |
| - | - | profiling_runner | Placeholder R1 (vide R0) |

---

## ANNEXE G : QUESTIONS FRÉQUENTES

### G.1 Pourquoi test_engine ne valide pas la symétrie ?

**Réponse** : Séparation stricte responsabilités.

- **test_engine** : Exécution aveugle (registries + post-processing)
- **Tests** : Observation pure (pas de validation)
- **Validation sémantique** : Dans D_encodings/ uniquement

**Rationale** : test_engine doit rester générique (applicable tout type D, tout gamma)

---

### G.2 Différence variance marginale vs conditionnelle ?

**Réponse** :

| Type | Définition | Exemple |
|------|------------|---------|
| **Marginale** | Variance expliquée par facteur isolé | VR(gamma_id) = 0.6 |
| **Conditionnelle** | Variance expliquée dans contexte fixé | VR(gamma_id \| modifier=M1) = 0.8 |

**Interaction** : VR conditionnelle >> VR marginale

**Exemple concret** :
- VR(gamma) global = 0.4 (effet moyen)
- VR(gamma | modifier=M1) = 0.9 (effet fort avec bruit)
- → Interaction détectée : gamma amplifié par M1

---

### G.3 Pourquoi params_config_id exclu des FACTORS ?

**Réponse** : Corrélation structurelle avec test_name.

**Problème** :
- params_config_id détermine paramètres test
- test_name détermine quel test
- params_config_id × test_name = redondance quasi-totale

**Conséquence** : Inflation artificielle variance, interactions spurieuses

**Solution** : Analyse params via configurations différentes (ex: params_v1 vs params_v2)

---

### G.4 Détection événements : pourquoi heuristiques R0 ?

**Réponse** : Simplicité vs précision.

**R0 (actuel)** :
- `saturation` onset estimé 80% (heuristique)
- `collapse` onset estimé 90%
- Suffisant pour timelines qualitatives

**R1+ (prévu)** :
- Détection onsets exacts via algorithmes robustes
- Fenêtres glissantes, seuils adaptatifs
- Validation statistique onsets

**Trade-off R0** : Rapidité implémentation vs précision absolue

---

### G.5 Verdict Reporter : pourquoi 5 étapes distinctes ?

**Réponse** : Séparation concerns + traçabilité.

**Avantages** :
1. **Modularité** : Chaque étape testable indépendamment
2. **Traçabilité** : Logs clairs progression
3. **Debugging** : Isolation erreurs par étape
4. **Extensibilité** : Ajout étapes futures sans refactor

**Exemple** : Étape 3 (profiling) peut être désactivée sans casser pipeline

---

### G.6 Format dict v2 : pourquoi timeseries optionnel ?

**Réponse** : Trade-off stockage vs fonctionnalité.

**Timeseries** :
- Taille : N iterations × M métriques × 8 bytes (float64)
- Exemple : 200 it × 5 métriques × 8 = 8 KB / observation
- Pour 1000 observations → 8 MB

**R0** : Timeseries généré mais non stocké DB par défaut

**Usage R1+** :
- Visualisations trajectoires
- Analyses fréquentielles
- Détection patterns temporels fins

**Configuration** : Flag `store_timeseries` dans verdict_config.yaml

---

### G.7 Interactions orientées : combien de paires testées ?

**Réponse** : N × (N-1) paires (permutations, pas combinations)

**Calcul** :
```python
N = len(FACTORS) = 5  # gamma, encoding, modifier, seed, test
Paires = N × (N-1) = 5 × 4 = 20
```

**Exemples paires** :
- (gamma, modifier) : effet gamma change selon modifier
- (modifier, gamma) : effet modifier change selon gamma
- (gamma, seed) : effet gamma change selon seed
- etc.

**Total contextes testés** : 20 × nb_niveaux_contexte

**Ordre magnitude** : 20 × 10 = 200 contextes analysés (typique)



