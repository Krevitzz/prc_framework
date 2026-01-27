## ⚠️ AVANT DE CODER

## utils_databases.py
  # Initialiser bases R0
  python -m prc_automation.utils_databases --mode init --phase R0
  
  # Nettoyer entrées obsolètes
  python -m prc_automation.utils_databases --mode clean --phase R0
  
  # Backup avant modifications
  python -m prc_automation.utils_databases --mode backup --phase R0
  
  # Export schema uniquement (léger)
  python -m prc_automation.utils_databases --mode export_schema --phase R0
  
  # Export complet JSON (lourd)
  python -m prc_automation.utils_databases --mode export_json --phase R0
  
  # Export JSON avec BLOBs encodés
  python -m prc_automation.utils_databases --mode export_json --phase R0 --include-blobs
        """
** PRC_AUTOMATION (batch_runner.py) **

### Fonctions publiques
| Fonction | Type | Responsabilité |
|----------|------|----------------|
| run_batch_brut | Publique | Exécution kernel combinaisons manquantes (différentiel) |
| run_batch_test | Publique | Application tests sur exec_ids ciblés (différentiel) |
| run_batch_verdict | Publique | Génération rapports (toujours exécuté) |
| main | Publique | Pipeline 7 étapes (point d'entrée) |
| get_db_paths | Publique | Retourne chemins bases pour phase |

### Fonctions registry
| Fonction | Type | Responsabilité |
|----------|------|----------------|
| load_execution_registry | Publique | Charge registry JSON ou crée vide |
| update_execution_registry | Publique | Mise à jour registry (merge conservatif) |
| classify_new_files | Publique | Classification branch A/B/none |

### Fonctions helpers
| Fonction | Type | Responsabilité |
|----------|------|----------------|
| detect_missing_combinations | Publique | Détecte combinaisons non exécutées |
| insert_execution | Publique | Insère exécution dans db_raw |
| load_execution_context | Publique | Charge contexte depuis db_raw |
| load_first_snapshot | Publique | Charge premier snapshot (state_shape) |
| load_execution_history | Publique | Charge history complète |
| store_test_observation | Publique | Stocke observation db_results |

### Exceptions
| Exception | Type | Usage |
|-----------|------|-------|
| CriticalBatchError | Exception | Erreurs batch nécessitant arrêt |

### Configuration globale
| Constante | Type | Valeur/Description |
|-----------|------|-------------------|
| SEEDS | List[int] | [42, 123, 456, 789, 1011] |
| MAX_ITERATIONS | int | 2000 |
| SNAPSHOT_INTERVAL | int | 10 |
| DB_DIR | Path | ./prc_automation/prc_database |

## D_ENCODINGS - ENCODINGS DISPONIBLES

| ID | Type | Rang | Propriétés clés | Description courte |
|----|------|------|-----------------|-------------------|
| **SYM-001** | symmetric | 2 | identity, sparse | Matrice identité |
| **SYM-002** | symmetric | 2 | uniform_distribution | Aléatoire uniforme symétrisé |
| **SYM-003** | symmetric | 2 | gaussian_distribution | Aléatoire gaussien symétrisé |
| **SYM-004** | symmetric | 2 | positive_definite, unit_diagonal | Matrice corrélation (SPD) |
| **SYM-005** | symmetric | 2 | banded, sparse | Bande symétrique |
| **SYM-006** | symmetric | 2 | block_structure, hierarchical | Hiérarchique par blocs |
| **SYM-007** | symmetric | 2 | uniform_correlation, unit_diagonal | Corrélations uniformes |
| **SYM-008** | symmetric | 2 | parametric_normal, clipped | Aléatoire paramétrable clippé |
| **ASY-001** | asymmetric | 2 | bounded | Aléatoire asymétrique |
| **ASY-002** | asymmetric | 2 | triangular, sparse | Triangulaire inférieure |
| **ASY-003** | asymmetric | 2 | antisymmetric, zero_diagonal | Antisymétrique |
| **ASY-004** | asymmetric | 2 | linear_gradient | Gradient directionnel |
| **ASY-005** | asymmetric | 2 | circulant, periodic | Structure circulante |
| **ASY-006** | asymmetric | 2 | sparse, controlled_density | Sparse densité contrôlée |
| **R3-001** | random | 3 | no_symmetry | Aléatoire uniforme rang 3 |
| **R3-002** | partial_symmetric | 3 | partial_symmetry | Symétrique partiel (indices 2-3) |
| **R3-003** | local_coupling | 3 | sparse, geometric_locality | Couplages locaux |
| **R3-004** | fully_symmetric | 3 | full_symmetry, permutation_invariant | Symétrique total (6 permutations) |
| **R3-005** | diagonal | 3 | very_sparse, diagonal | Diagonal pur |
| **R3-006** | separable | 3 | factorized, low_rank | Séparable (produit externe) |
| **R3-007** | block_structure | 3 | hierarchical, modular | Par blocs 3D |

**Modifiers existants** :
| ID | Fichier | Type | Transformation |
|----|---------|------|----------------|
| **M0** | m0_baseline.py | baseline | D' = D |
| **M1** | m1_gaussian_noise.py | noise | D' = D + N(0,σ) |
| **M2** | m2_uniform_noise.py | noise | D' = D + U[-a,+a] |

**Gammas existants** :
| {GammaClass}.__init__ | Publique | Initialise paramètres + validation |
| {GammaClass}.__call__ | Publique | Applique transformation Gamma |
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
###test_engine.py | Exécution tests + observations | registries, config_loader |
| detect_dynamic_events | Module-level | Détecte événements dynamiques sur trajectoire |
| compute_event_sequence | Module-level | Construit séquence ordonnée + onsets relatifs |
| patch_execute_test_dynamic_events | Module-level | Calcule dynamic_events + timeseries tous metrics |
| TestEngine._init_result | Privée | Initialise structure résultat avec exec_id |
| TestEngine._prepare_computations | Privée | Prépare et valide toutes spécifications |
| TestEngine._compile_results | Privée | Compile résultats finaux avec metadata |
| TestEngine._analyze_evolution | Privée | Analyse évolution série temporelle |
- `TestEngine.execute_test()` : Exécute test sur history, retourne observation

### verdict_engine.py | Analyses statistiques multi-facteurs | data_loading, statistical_utils, regime_utils |
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

###verdict_reporter.py | Orchestration rapports complets | verdict_engine, profiling_runner, report_writers |
- `generate_verdict_report()` : Pipeline complet 6 étapes
- `_compile_metadata()` : Compilation métadonnées rapport
- `_compile_structural_patterns()` : Compilation patterns globaux
- `_write_summary_report()` : Écriture rapport humain (partiellement délégué)

### profiling_runner.py | Orchestration profiling multi-axes | profiling_common, cross_profiling |
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

### data_loading.py (Discovery + Applicabilité + I/O)
| Fonction | Type | Responsabilité |
|----------|------|----------------|
| discover_entities | Publique | Découvre entités actives (test/gamma/encoding/modifier) |
| check_applicability | Publique | Vérifie applicabilité test sur run_metadata |
| add_validator | Publique | Ajoute validator custom |
| load_all_observations | Publique | Charge obs SUCCESS (db_results uniquement) |
| observations_to_dataframe | Publique | Convertit obs → DataFrame normalisé |
| cache_observations | Publique | Cache obs disque (pickle) |
| load_cached_observations | Publique | Charge cache observations |
| db_connection | Publique | Context manager connexion DB |
| _discover_tests | Privée | Découvre tests actifs |
| _discover_gammas | Privée | Découvre gammas actifs |
| _discover_encodings | Privée | Découvre encodings actifs |
| _discover_modifiers | Privée | Découvre modifiers actifs |
| _validate_test_structure | Privée | Valide structure test 5.5 |
| VALIDATORS | Global dict | Registre validators extensible |

| db_connection | Publique | Gestionnaire contexte DB (context manager) |
| decompress_snapshot | Publique | Décompresse snapshot gzip+pickle |
| extract_metric_data | Publique | Extrait statistics + evolution métrique |

### profiling_common.py (Profiling générique tous axes)

| Fonction | Type | Responsabilité | Axes supportés |
|----------|------|----------------|----------------|
| `profile_all_tests()` | Publique | Profil comportemental tests | test |
| `compare_tests_summary()` | Publique | Comparaisons inter-tests | test |
| `profile_all_gammas()` | Publique | Profil comportemental gammas | gamma |
| `compare_gammas_summary()` | Publique | Comparaisons inter-gammas | gamma |
| `profile_all_modifiers()` | Publique | Profil comportemental modifiers | modifier |
| `compare_modifiers_summary()` | Publique | Comparaisons inter-modifiers | modifier |
| `profile_all_encodings()` | Publique | Profil comportemental encodings | encoding |
| `compare_encodings_summary()` | Publique | Comparaisons inter-encodings | encoding |
| `aggregate_dynamic_signatures()` | Publique | Agrège événements + timelines | Tous |
| `compute_prc_profile()` | Publique | Génère profil PRC complet | Tous |
| `_profile_test_for_entity()` | Privée | Profil UN test sous UNE entité | Tous |
| `_profile_entity_axis()` | Privée | Moteur générique profiling | Tous |
| `_compare_entities_summary()` | Privée | Comparaisons génériques | Tous |
| `ENTITY_KEY_MAP` | Constante | Mapping axes → clés DB | - |

**Convention naming** : `profile_all_{axis}()` + `compare_{axis}_summary()`

**Format retour** : Structure unifiée stricte (Charter R5.1-A)
```python
{
    'profiles': {...},       # Profils individuels
    'summary': {...},        # Comparaisons cross-entités
    'metadata': {...}        # Infos exécution
}
```

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