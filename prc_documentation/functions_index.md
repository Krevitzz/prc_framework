## ⚠️ AVANT DE CODER

** PRC_AUTOMATION (batch_runner.py) **

### Fonctions publiques
| run_batch_brut | Publique | Collecte données (db_raw) |
| run_batch_test | Publique | Application tests (db_results) |
| run_batch_verdict | Publique | Génération verdicts |
| run_batch_all | Publique | Pipeline complet |
| parse_args | Publique | Parse arguments CLI |
| main | Publique | Point d'entrée principal |

### Fonctions helpers
| get_exec_ids_for_gamma | Module-level | Récupère exec_ids pour gamma |
| count_observations | Module-level | Compte observations SUCCESS |
| load_execution_context | Module-level | Charge contexte depuis db_raw |
| load_first_snapshot | Module-level | Charge premier snapshot (state_shape) |
| load_execution_history | Module-level | Charge history complète |
| store_test_observation | Module-level | Stocke observation db_results |

### Exceptions
| CriticalTestError | Exception | Erreurs critiques nécessitant arrêt |

**Fonctions existantes du core** :
| Fonction | Fichier | Responsabilité |
|----------|---------|----------------|
| `prepare_state()` | state_preparation.py | Composition séquentielle modifiers sur tenseur base |
| `run_kernel()` | kernel.py | Itération aveugle state_{n+1} = gamma(state_n) |

**Encodings existants** :
| ID | Fichier | Type | Propriétés clés |
|----|---------|------|-----------------|
| SYM-001 à SYM-006 | rank2_symmetric.py | Rang 2 symétrique | Diverses structures |
| create_circulant_asymmetric | Publique | Matrice circulante asymétrique |
| create_sparse_asymmetric | Publique | Matrice asymétrique sparse |
| ASY-001 à ASY-004 | rank2_asymmetric.py | Rang 2 asymétrique | Brisure symétrie |
| create_uniform | Publique | Corrélations uniformes |
| create_random | Publique | Corrélations aléatoires (helper) |
| R3-001 à R3-003 | rank3_correlations.py | Rang 3 | Couplages d'ordre supérieur |
| create_fully_symmetric_rank3 | Publique | Tenseur totalement symétrique (bonus) |
| create_diagonal_rank3 | Publique | Tenseur diagonal (bonus) |
| create_separable_rank3 | Publique | Tenseur séparable (bonus) |
| create_block_rank3 | Publique | Tenseur par blocs (bonus) |

**Modifiers existants** :
| ID | Fichier | Type | Paramètres clés |
|----|---------|------|-----------------|
| M0 | (implicite) | Baseline | Aucune modification |
| M1 | noise.py::add_gaussian_noise() | Bruit gaussien | sigma=0.05 |
| M2 | noise.py::add_uniform_noise() | Bruit uniforme | amplitude=0.1 |

**Gammas existants** :
| {GammaClass}.__init__ | Publique | Initialise paramètres + validation |
| {GammaClass}.__call__ | Publique | Applique transformation Δ |
| {GammaClass}.__repr__ | Publique | Représentation string |
| {GammaClass}.reset | Publique | Réinitialise mémoire (si non-markovien) |
| create_gamma_hyp_{NNN} | Publique | Factory création gamma |
| PARAM_GRID_PHASE1 | Global dict | Grille paramètres phase 1 |
| PARAM_GRID_PHASE2 | Global dict | Grille paramètres phase 2 |
| METADATA | Global dict | Métadonnées gamma |

| ID | Fichier | Famille | Applicabilité | Statut |
|----|---------|---------|---------------|--------|
| GAM-001 | gamma_hyp_001.py | markovian | SYM, ASY, R3 | WIP[R0-open] |
| GAM-002 | gamma_hyp_002.py | markovian | SYM, ASY | WIP[R0-open] |
| GAM-003 | gamma_hyp_003.py | markovian | SYM, ASY, R3 | WIP[R0-open] |
| GAM-004 | gamma_hyp_004.py | markovian | SYM, ASY, R3 | WIP[R0-open] |
| GAM-005 | gamma_hyp_005.py | markovian | SYM, ASY, R3 | WIP[R0-open] |
| GAM-006 | gamma_hyp_006.py | non_markovian | SYM, ASY, R3 | WIP[R0-open] |
| GAM-007 | gamma_hyp_007.py | non_markovian | SYM, ASY | WIP[R0-open] |
| GAM-008 | gamma_hyp_008.py | non_markovian | SYM, ASY, R3 | WIP[R0-open] |
| GAM-009 | gamma_hyp_009.py | stochastic | SYM, ASY, R3 | WIP[R0-open] |
| GAM-010 | gamma_hyp_010.py | stochastic | SYM, ASY, R3 | WIP[R0-open] |
| GAM-011 | gamma_hyp_011.py | (vide) | - | Non implémenté |
| GAM-012 | gamma_hyp_012.py | structural | SYM, ASY | WIP[R0-open] |
| GAM-013 | gamma_hyp_013.py | structural | SYM, ASY | WIP[R0-open] |

**Tests existants** :
| ID | Fichier | Catégorie | Applicabilité | Poids |
|----|---------|-----------|---------------|-------|
| UNIV-001 | test_uni_001.py | Universels | Tous rangs | 1.0 |
| UNIV-002 | test_uni_002.py | Universels | Rang 2 carré | 1.0 |
| SYM-001 | test_sym_001.py | Symétrie | Rang 2 carré (SYM/ASY) | 1.0 |
| SPE-001 | test_spe_001.py | Spectral | Rang 2 carré | 1.0 |
| SPE-002 | test_spe_002.py | Spectral | Rang 2 carré | 1.0 |
| PAT-001 | test_pat_001.py | Pattern | Tous rangs | 1.0 |
| SPA-001 | test_spa_001.py | Spatial | Rang 2 | 1.0 |
| GRA-001 | test_gra_001.py | Graphe | Rang 2 carré | 1.0 |
| TOP-001 | test_top_001.py | Topologique | Rang 2 | 1.0 |

**Modules HUB existants** :
| Module | Responsabilité | Dépendances autorisées |
|--------|----------------|------------------------|
| test_engine.py | Exécution tests + observations | registries, config_loader |
| detect_dynamic_events | Module-level | Détecte événements dynamiques sur trajectoire |
| compute_event_sequence | Module-level | Construit séquence ordonnée + onsets relatifs |
| patch_execute_test_dynamic_events | Module-level | Calcule dynamic_events + timeseries tous metrics |
| TestEngine._init_result | Privée | Initialise structure résultat avec exec_id |
| TestEngine._prepare_computations | Privée | Prépare et valide toutes spécifications |
| TestEngine._compile_results | Privée | Compile résultats finaux avec metadata |
| TestEngine._analyze_evolution | Privée | Analyse évolution série temporelle |
- `TestEngine.execute_test()` : Exécute test sur history, retourne observation

| verdict_engine.py | Analyses statistiques multi-facteurs | data_loading, statistical_utils, regime_utils |
| _compile_metadata | Privée | Compile métadonnées rapport |
| _format_gamma_profiles | Privée | Formate gamma_profiles pour rapport |
| _compile_structural_patterns | Privée | Compile patterns structuraux (3 strates) |
| _write_summary_report | Privée | Écrit rapport humain principal |
| _write_gamma_profiles | Privée | Écrit gamma_profiles.json + CSV |
- `analyze_marginal_variance()` : Variance marginale (η² par facteur)
- `analyze_oriented_interactions()` : Interactions orientées A|B ≠ B|A
- `analyze_metric_discrimination()` : Détection métriques non discriminantes
- `analyze_metric_correlations()` : Corrélations fortes entre métriques
- `interpret_patterns()` : Synthèse patterns globaux + par gamma
- `decide_verdict()` : Décision verdict (GLOBAL + PAR GAMMA)
- `analyze_regime()` : Pipeline complet sur strate (GLOBAL/STABLE/EXPLOSIF)
- `compute_verdict()` : Pipeline principal (point d'entrée)

| verdict_reporter.py | Orchestration rapports complets | verdict_engine, profiling_runner, report_writers |
- `generate_verdict_report()` : Pipeline complet 6 étapes
- `_compile_metadata()` : Compilation métadonnées rapport
- `_format_gamma_profiles()` : Formatage structure gamma_profiles
- `_compile_structural_patterns()` : Compilation patterns globaux
- `_write_summary_report()` : Écriture rapport humain (partiellement délégué)
- `_write_gamma_profiles()` : Écriture JSON + CSV

| profiling_runner.py | Orchestration profiling multi-axes | profiling_common, cross_profiling |
- `run_all_profiling()` : Exécute profiling tous axes demandés
- `run_profiling_single_axis()` : Profiling un seul axe (helper)
- `discover_profiling_axes()` : Découverte axes disponibles
- `get_entity_profile()` : Extrait profil entité spécifique
- `get_test_profile_for_entity()` : Extrait profil test pour entité

**Registres existants** :
### ALGEBRA (algebra_registry.py)
| Fonction | registry_key | Responsabilité | Applicabilité |
|----------|--------------|----------------|---------------|
| `compute_asymmetry()` | algebra.matrix_asymmetry | Norme partie anti-symétrique | 2D carré |
| `compute_norm()` | algebra.matrix_norm | Norme tenseur ordre N | Tous rangs |
| `compute_frobenius()` | algebra.frobenius_norm | Norme Frobenius (alias optimisé) | Tous rangs |
| `compute_trace()` | algebra.trace_value | Trace matrice | 2D carré |
| `compute_determinant()` | algebra.determinant_value | Déterminant | 2D carré |
| `compute_spectral_norm()` | algebra.spectral_norm | Norme spectrale (σ_max) | 2D |
| `compute_condition()` | algebra.condition_number | Conditionnement κ(A) | 2D carré |
| `compute_rank()` | algebra.rank_estimate | Rang effectif (SVD) | 2D |

### SPECTRAL (spectral_registry.py)
| Fonction | registry_key | Responsabilité | Applicabilité |
|----------|--------------|----------------|---------------|
| `compute_eigenvalue_max()` | spectral.eigenvalue_max | Plus grande valeur propre | 2D carré |
| `compute_eigenvalue_stats()` | spectral.eigenvalue_distribution | Stats distribution λ_i | 2D carré |
| `compute_spectral_gap()` | spectral.spectral_gap | Écart λ₁ - λ₂ | 2D carré |
| `compute_fft_power()` | spectral.fft_power | Puissance spectre FFT | Tous rangs |
| `compute_fft_entropy()` | spectral.fft_entropy | Entropie spectrale FFT | Tous rangs |
| `compute_spectral_radius()` | spectral.spectral_radius | Rayon spectral ρ(A) | 2D carré |

### STATISTICAL (statistical_registry.py)
| Fonction | registry_key | Responsabilité | Applicabilité |
|----------|--------------|----------------|---------------|
| `compute_entropy()` | statistical.entropy | Entropie Shannon | Tous rangs |
| `compute_kurtosis()` | statistical.kurtosis | Aplatissement distribution | Tous rangs |
| `compute_skewness()` | statistical.skewness | Asymétrie distribution | Tous rangs |
| `compute_variance()` | statistical.variance | Variance | Tous rangs |
| `compute_std_normalized()` | statistical.std_normalized | Coefficient variation (CV) | Tous rangs |
| `compute_correlation_mean()` | statistical.correlation_mean | Corrélation moyenne lignes/colonnes | 2D |
| `compute_sparsity()` | statistical.sparsity | Mesure parcimonie | Tous rangs |

### SPATIAL (spatial_registry.py)
| Fonction | registry_key | Responsabilité | Applicabilité |
|----------|--------------|----------------|---------------|
| `compute_gradient_magnitude()` | spatial.gradient_magnitude | Magnitude gradient moyen | 2D, 3D |
| `compute_laplacian_energy()` | spatial.laplacian_energy | Énergie laplacien (rugosité) | 2D |
| `compute_local_variance()` | spatial.local_variance | Variance locale moyenne | 2D |
| `compute_edge_density()` | spatial.edge_density | Densité contours détectés | 2D |
| `compute_spatial_autocorrelation()` | spatial.spatial_autocorrelation | Autocorrélation spatiale (Moran's I) | 2D |
| `compute_smoothness()` | spatial.smoothness | Mesure lissage | 2D |

### PATTERN (pattern_registry.py)
| Fonction | registry_key | Responsabilité | Applicabilité |
|----------|--------------|----------------|---------------|
| `compute_periodicity()` | pattern.periodicity | Détection périodicité | Tous rangs |
| `compute_symmetry_score()` | pattern.symmetry_score | Score symétrie radiale | 2D |
| `compute_clustering()` | pattern.clustering_coefficient | Clustering valeurs similaires | 2D |
| `compute_diversity()` | pattern.diversity | Diversité (indice Simpson) | Tous rangs |
| `compute_uniformity()` | pattern.uniformity | Uniformité distribution | Tous rangs |
| `compute_concentration()` | pattern.concentration_ratio | Ratio concentration (Gini-like) | Tous rangs |

### TOPOLOGICAL (topological_registry.py)
| Fonction | registry_key | Responsabilité | Applicabilité |
|----------|--------------|----------------|---------------|
| `compute_connected_components()` | topological.connected_components | Nombre composantes connexes | 2D |
| `compute_euler_characteristic()` | topological.euler_characteristic | Caractéristique Euler χ | 2D |
| `compute_perimeter_area_ratio()` | topological.perimeter_area_ratio | Ratio périmètre/aire | 2D |
| `compute_compactness()` | topological.compactness | Compacité (isoperimetric) | 2D |
| `compute_holes_count()` | topological.holes_count | Estimation nombre trous | 2D |
| `compute_fractal_dimension()` | topological.fractal_dimension | Dimension fractale (box-counting) | 2D |

### GRAPH (graph_registry.py)
| Fonction | registry_key | Responsabilité | Applicabilité |
|----------|--------------|----------------|---------------|
| `compute_density()` | graph.density | Densité graphe | 2D carré |
| `compute_degree_variance()` | graph.degree_variance | Variance degrés | 2D carré |
| `compute_clustering_local()` | graph.clustering_local | Coefficient clustering local | 2D carré |
| `compute_average_path_length()` | graph.average_path_length | Longueur chemin moyenne | 2D carré |
| `compute_centrality_concentration()` | graph.centrality_concentration | Concentration centralité | 2D carré |
| `compute_small_world()` | graph.small_world_coefficient | Coefficient petit-monde σ | 2D carré |

### POST-PROCESSORS (post_processors.py)

| get_post_processor | Publique | Récupère post-processor par clé |
| add_post_processor | Publique | Ajoute post-processor custom |
| POST_PROCESSORS | Global dict | Registre post-processors disponibles |
| Clé | Fonction | Responsabilité |
|-----|----------|----------------|
| `identity` | lambda x: x | Identité (pas de transformation) |
| `round_2` | round(x, 2) | Arrondi 2 décimales |
| `round_4` | round(x, 4) | Arrondi 4 décimales |
| `round_6` | round(x, 6) | Arrondi 6 décimales |
| `abs` | abs(x) | Valeur absolue |
| `log` | log(x + 1e-10) | Logarithme naturel |
| `log10` | log10(x + 1e-10) | Logarithme base 10 |
| `log1p` | log1p(x) | log(1 + x) |
| `clip_01` | clip(x, 0, 1) | Clipping [0, 1] |
| `clip_positive` | max(0, x) | Clipping ≥ 0 |
| `scientific_3` | format scientifique 3 décimales | Notation scientifique |

### RegistryManager (registry_manager.py)

| RegistryManager.__new__ | Publique | Singleton pattern |
| RegistryManager.__init__ | Publique | Initialise + charge registres |
| RegistryManager._load_all_registries | Privée | Charge dynamiquement tous *_registry.py |
| RegistryManager._validate_params | Privée | Valide paramètres contre signature |
- `get_function(registry_key)` : Récupère fonction (avec cache)
- `validate_computation_spec(spec)` : Valide COMPUTATION_SPECS
- `list_available_functions()` : Liste toutes fonctions par registre

### BaseRegistry (base_registry.py)

| BaseRegistry.__init__ | Publique | Initialise registre vide |
| BaseRegistry._register_all_functions | Privée | Découvre fonctions décorées |
| BaseRegistry.get_function | Publique | Récupère fonction par clé complète |
| BaseRegistry.list_functions | Publique | Liste fonctions avec documentation |

**Modules UTIL existants** :
### aggregation_utils.py (Agrégations statistiques)
| Fonction | Type | Responsabilité |
|----------|------|----------------|
| `aggregate_summary_metrics()` | Publique | Agrège métriques inter-runs (median, q1, q3, cv) |
| `aggregate_run_dispersion()` | Publique | Détecte multimodalité (IQR ratio, bimodal) |
| `compute_dominant_value()` | Publique | **Non implémenté** (placeholder futur) |
| `aggregate_event_counts()` | Publique | **Non implémenté** (placeholder futur) |

### applicability.py (Validation applicabilité)
| Fonction | Type | Responsabilité |
|----------|------|----------------|
| `check()` | Publique | Vérifie applicabilité test sur run_metadata |
| `add_validator()` | Publique | Ajoute validator custom |
| `VALIDATORS` | Global dict | Registre validators extensible |


### config_loader.py (Chargement configs YAML)
| Fonction | Type | Responsabilité |
|----------|------|----------------|
| ConfigLoader._load_global | Privée | Charge config global |
| ConfigLoader._load_specific | Privée | Charge config spécifique test |
| ConfigLoader._merge_configs | Privée | Fusionne configs (specific override global) |
| ConfigLoader._merge_dicts | Privée | Merge récursif dictionnaires |
| ConfigLoader._validate_config | Privée | Validation basique structure |
| ConfigLoader.list_available | Publique | Liste configs disponibles |
| ConfigLoader.clear_cache | Publique | Vide cache |
| get_loader | Publique | Récupère instance singleton |
| `ConfigLoader.__init__()` | Publique | Initialise cache configs |
| `ConfigLoader.load()` | Publique | Charge config avec fusion global+specific |
| `ConfigLoader.list_available()` | Publique | Liste configs disponibles |
| `ConfigLoader.clear_cache()` | Publique | Vide cache |

### cross_profiling.py (Analyses croisées)
| Fonction | Type | Responsabilité |
|----------|------|----------------|
| rank_entities_by_metric | Publique | Ranking générique entités par métrique |
| _compute_concordance_matrix | Privée | Calcule matrice concordance entre deux axes |
| `compute_discriminant_power()` | Publique | Variance conditionnelle cross-entités |
| `compute_all_discriminant_powers()` | Publique | Discriminant power tous tests |
| `analyze_pairwise_interactions()` | Publique | Interactions 2-way (R0: concordance, R1+: qualitatif) |
| `analyze_multiway_interactions()` | Publique | **Placeholder R1+** (interactions n-way) |
| `detect_global_signatures()` | Publique | **Placeholder R1+** (vocabulaire interprét.) |
| `_compute_concordance_matrix()` | Privée | Matrice concordance 2 axes |

### data_loading.py (I/O observations)
| Fonction | Type | Responsabilité |
|----------|------|----------------|
| `load_all_observations()` | Publique | Charge obs SUCCESS avec métadonnées (2 DBs) |
| `observations_to_dataframe()` | Publique | Convertit obs → DataFrame normalisé |
| `cache_observations()` | Publique | Cache obs disque (pickle) |
| `load_cached_observations()` | Publique | Charge cache observations |

| db_connection | Publique | Gestionnaire contexte DB (context manager) |
| decompress_snapshot | Publique | Décompresse snapshot gzip+pickle |
| extract_metric_data | Publique | Extrait statistics + evolution métrique |

### discovery.py (Découverte tests)
| Fonction | Type | Responsabilité |
|----------|------|----------------|
| `discover_active_tests()` | Publique | Découvre tests actifs (non _deprecated) |
| `validate_test_structure()` | Publique | Valide structure test 5.4 conforme |
| `REQUIRED_ATTRIBUTES` | Global list | Attributs requis architecture 5.5 |

### profiling_common.py (Profiling générique tous axes)
| Fonction | Type | Responsabilité |
|----------|------|----------------|
| _profile_test_for_entity | Privée | Profil UN test sous UNE entité (moteur) |
| _profile_entity_axis | Privée | Moteur générique profiling tous axes |
| _compare_entities_summary | Privée | Comparaisons cross-entities |
| ENTITY_KEY_MAP | Global dict | Mapping axes → clés DB |
| `aggregate_dynamic_signatures()` | Publique | Agrège événements + timelines compositionnels |
| `compute_prc_profile()` | Publique | Génère profil PRC avec confidence |
| `profile_all_tests()` | Publique | Profil comportemental tests (API découvrable) |
| `compare_tests_summary()` | Publique | Comparaisons inter-tests (API découvrable) |
| `profile_all_gammas()` | Publique | Profil comportemental gammas (API découvrable) |
| `compare_gammas_summary()` | Publique | Comparaisons inter-gammas (API découvrable) |
| `profile_all_modifiers()` | Publique | Profil comportemental modifiers (API découvrable) |
| `compare_modifiers_summary()` | Publique | Comparaisons inter-modifiers (API découvrable) |
| `profile_all_encodings()` | Publique | Profil comportemental encodings (API découvrable) |
| `compare_encodings_summary()` | Publique | Comparaisons inter-encodings (API découvrable) |

### regime_utils.py (Classification régimes)
| Fonction | Type | Responsabilité |
|----------|------|----------------|
| `stratify_by_regime()` | Publique | Stratifie obs stable/explosif |
| `classify_regime()` | Publique | Classification régime spécifique |
| `detect_conserved_property()` | Publique | Détermine propriété conservée |
| `extract_conserved_properties()` | Publique | Extrait propriétés depuis profil |
| `get_regime_family()` | Publique | Retourne famille régime |
| `REGIME_TAXONOMY` | Global dict | Taxonomie régimes (référence) |

### report_writers.py (Formatage rapports)
| Fonction | Type | Responsabilité |
|----------|------|----------------|
| `write_json()` | Publique | Écrit dict → JSON formaté |
| `write_header()` | Publique | Écrit header section TXT |
| `write_subheader()` | Publique | Écrit sous-header section |
| `write_key_value()` | Publique | Écrit paire clé-valeur |
| `write_regime_synthesis()` | Publique | Synthèse régimes transversale |
| `write_dynamic_signatures()` | Publique | Signatures dynamiques par gamma |
| `write_comparisons_enriched()` | Publique | Comparaisons enrichies contexte |
| `write_consultation_footer()` | Publique | Footer fichiers consultation |
| `_make_json_serializable()` | Privée | Convertit tuples → strings JSON |

### statistical_utils.py (Outils statistiques)
| Fonction | Type | Responsabilité |
|----------|------|----------------|
| `compute_eta_squared()` | Publique | Calcule η² (proportion variance expliquée) |
| `kruskal_wallis_test()` | Publique | Test Kruskal-Wallis avec gestion erreurs |
| `filter_numeric_artifacts()` | Publique | Filtre obs inf/nan avec stats rejets |
| `generate_degeneracy_report()` | Publique | Rapport dégénérescences projections |
| `print_degeneracy_report()` | Publique | Affiche rapport dégénérescences (stdout) |
| `diagnose_scale_outliers()` | Publique | Détecte ruptures échelle relatives |
| `print_scale_outliers_report()` | Publique | Affiche rapport outliers (stdout) |
| `is_numeric_valid()` | Privée | Détecte artefacts (inf/nan) observation |
| `diagnose_numeric_degeneracy()` | Privée | Flags dégénérescences par projection |

| compute_iqr | Publique | Calcule IQR et quartiles |
| robust_normalize | Publique | Normalisation (x-median)/IQR |

### timeline_utils.py (Timelines dynamiques)
| Fonction | Type | Responsabilité |
|----------|------|----------------|
| `classify_timing()` | Publique | Classifie timing (early/mid/late) |
| `compute_timeline_descriptor()` | Publique | Génère descriptor timeline compositionnel |
| `extract_dynamic_events()` | Publique | Extrait événements depuis observation |
| `extract_metric_timeseries()` | Publique | Extrait série temporelle avec fallback |
| `TIMELINE_THRESHOLDS` | Global dict | Seuils globaux relatifs |