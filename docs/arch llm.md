# PRC — Architecture Map v4 (version LLM)

## Arborescence réelle — tree 2026-03-12 (corrigé)

```
analysing/
  clustering_peeling_v2.py · clustering_v2.py · cluster_namer_v2.py · concordance_lite.py
  hub_analysing_v2.py · outliers_v2.py · parquet_filter_v2.py · pipeline_v2.py · profiling_v2.py
  verdict_v2.py · visualizer_v2.py
  configs/ analysing_default.yaml · cluster_namer.yaml
atomics/
  atomics_inventory_v7.md
  D_encodings/ asy_* · r3_* · rn_* · sym_* (20 fichiers)
  modifiers/ m0_baseline.py · m1_gaussian_noise.py · m2_uniform_noise.py · plugins/ (biology, gravity, quantum, template)
  operators/ gam_* (15 fichiers)
batch.py
configs/
  phases/ aot_bench.yaml · poc/ (poc1.yaml, poc_bourrin_v7.yaml, poc_cache_bench.yaml, poc_perf.yaml, poc.yaml)
  pool_requirements.yaml
data/results/ (vide)
docs/ CHARTER_7_1.md · PHASES_GUIDE.md · prc_architecture_v3(1).html · requirements.txt
featuring/
  hub_featuring.py · jax_features_new.py
  configs/ features_applicability.yaml
running/
  batching_new.py · hub_running_new.py · kernel_new.py · plan_new.py · run_one_jax.py
utils/ data_loading_new.py · parquet_to_json.py
LICENSE
```

## Pipeline global — flux bout en bout

1. `batch.py` CLI → MODE 1 : run_phase() | MODE 2 : --verdict {phase} (run_verdict_from_parquet) | MODE 3 : --verdict (run_verdict_cross_phases)
2. `running.hub_running_new.run_phase()` → load_yaml → dry_run_stats → confirmation → registries
3. `running.plan_new.generate_kernel_groups()` → génère les groupes vmappables (streaming) à partir du YAML et des registries
4. `running.plan_new.split_into_batches()` → découpe chaque groupe en batches de taille ≤ batch_size
5. `running.batching_new.execute_batch()` (async) → features_async — construit les inputs batchés et appelle jit(vmap(`running.kernel_new._run_fn`)) via cache global _vmap_cache
6. `running.batching_new.sync_features_b()` → np.array() force sync GPU→CPU
7. `running.batching_new.rows_from_synced()` → construit les rows parquet avec statut (OK/EXPLOSION/NAN_ALL)
8. `running.hub_running_new` collecte rows_buffer → flush → open_parquet_writer/write_rows_to_parquet → `data/results/{phase}.parquet`
9. `analysing.hub_analysing_v2.run_verdict_from_parquet()` → lit parquet via `parquet_filter_v2.load_analysing_data()`, puis appelle `pipeline_v2.run_analysing_pipeline()` et écrit les rapports (`verdict_v2`)
10. `analysing.hub_analysing_v2.run_verdict_cross_phases()` → compare plusieurs phases via `concordance_lite`, sortie dans `data/results/reports/`

*Statuts run :* OK (features finies), EXPLOSION (Inf), NAN_ALL (toutes NaN non structurelles), INVALID (rank_constraint), FAIL (exception) — définis dans hub_running_new et batching_new.

---

## CLI + Running — orchestration

### `batch.py`
- **Lu** ✓
- Modes :
  - `python -m batch {phase}` → run phase
  - `python -m batch --verdict {phase}` → verdict depuis parquet
  - `python -m batch --verdict` → verdict cross-phases
- Args : --verbose, --auto-confirm, --cfg (chemin config analysing), --plot (PNG), --debug (save labels+trace)
- Imports : run_phase (running.hub_running_new), run_verdict_from_parquet, run_verdict_cross_phases (analysing.hub_analysing_v2)
- Output dir : data/results/ pour parquet ; data/results/reports/ pour rapports
- Observations :
  - yaml_path hardcodé sur configs/phases/poc/{phase}.yaml — pas flexible pour d'autres familles.
  - cfg_path résolu avec resolve() ; défaut analysing/configs/analysing_default.yaml (existe).
  - warnings filtrés : filterwarnings('ignore', message='.*SLASCLS.*') — masque warning LAPACK.
  - os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75' — contrôle mémoire JAX.

### `running/hub_running_new.py`
- **Lu** ✓
- Fonctions :
  - `run_phase(yaml_path, output_dir, auto_confirm, batch_size=256, flush_every=1000, aot_ram_gb=5.0, verbose) → Dict`
  - `dry_run_stats(run_config, batch_size) → Dict` métriques (zéro instanciation)
  - `_check_invalid(kernel_group, gamma_registry) → bool`
- Workflow :
  1. load_yaml → extrait phase, dmd_rank, aot_ram_gb
  2. dry_run_stats() → affichage + confirmation
  3. registries = discover_*_jax()
  4. generate_kernel_groups() + rolling_compile_window()
  5. Pour chaque kernel_group : split_into_batches()
  6. Pour chaque batch : _check_invalid() → si INVALID → make_nan_rows() direct
  7. Sinon : execute_batch() async → features_async
  8. Pendant que GPU compute, traite le batch précédent (pending) : sync_features_b() + rows_from_synced() → rows_buffer
  9. Flush tous les flush_every via open_parquet_writer/write_rows_to_parquet
  10. En finally : close_parquet_writer()
- Statuts run : OK, EXPLOSION, NAN_ALL, INVALID, FAIL
- Observations :
  - aot_ram_gb surchargeable via YAML.
  - dmd_rank lu du YAML (défaut 16).
  - rolling_compile_window utilise n_workers = max(1, os.cpu_count()-2).
  - **af-ram** : double boucle pending + batch suivant (nécessaire pour overlap CPU/GPU).
  - **af-ram** : rows_buffer + flush_every = 1000.
  - **af-dead** : _check_invalid utilise gamma_registry.get('metadata',{}).get('rank_constraint') — si absent, pas d'invalidation.

### `running/batching_new.py`
- **Lu** ✓
- Fonctions :
  - `execute_batch(batch, dmd_rank) → features_b dict {str: jnp.array(B,)}` async
  - `sync_features_b(features_b) → {str: np.ndarray(B,)}`
  - `rows_from_synced(batch, synced_features, phase) → List[Dict] rows parquet`
  - `make_nan_rows(batch, run_status, phase) → List[Dict] (INVALID/FAIL)`
  - `_build_batch_inputs(batch) → (D_b, gp_b, mp_b, keys_b)`
  - `_get_kernel_fn(batch, dmd_rank, gp_keys, mp_keys) → callable jit(vmap)`
  - `_compile_one(kg, dmd_rank, batch_size) → None` (peuple _vmap_cache)
  - `rolling_compile_window(kg_gen, dmd_rank, batch_size, n_workers, ram_gb) → Generator[kg]` (fenêtre glissante de pré-compilation)
- Cache global `_vmap_cache` : clé = (id(gamma_fn), id(mod_fn), n_dof, max_it, dmd_rank, is_differentiable, gamma_param_keys, mod_param_keys) → jit(vmap(_run_fn)) partagé entre tous les batches du groupe.
- Construction des inputs batchés :
  - D_b : encodings vmappables si enc_vmappable=True (vmap), sinon loop Python.
  - gp_b : gamma_params après prepare_params.
  - mp_b : mod_params.
  - keys_b : PRNGKeys batchées.
- Statuts run (déterminés dans `_run_status_from_features`) : OK, EXPLOSION, NAN_ALL.
- Observations :
  - FEATURE_NAMES et FEATURES_STRUCTURAL_NAN importés depuis jax_features_new.
  - rolling_compile_window utilise window_size = n_workers*2 ; ram_gb non utilisé.
  - **af-ext** : _vmap_cache est un point d'extension implicite.
  - **af-ram** : _build_enc_batch avec enc_vmappable=False fait une boucle Python.
  - **af-dead** : rolling_compile_window a paramètre ram_gb non utilisé.

### `running/kernel_new.py`
- **Lu** ✓
- Fonctions :
  - `_step_fn(carry, _) → (carry_next, measures)` — un pas de dynamique (dans lax.scan)
  - `_run_fn(gamma_fn, gamma_params, modifier_fn, modifier_params, D_initial, key, max_it, dmd_rank, is_differentiable) → dict features JAX` (vmappable)
  - `run_one_jax(...) → dict {str: float}` (interface pour exécution unitaire)
- Rôle : noyau compilé unique — applique modificateur, scan gamma, post-traitement.
  - `_run_fn` est vmappable, utilisée par batching_new via jit(vmap(...)).
  - `run_one_jax` est une version JIT pour usage unitaire.
- Phases internes :
  1. modifier_fn(D_initial, modifier_params, subkey_mod) → D_modified
  2. lax.scan sur max_it pas avec carry étendu (state, key, state_prev, sigmas_prev, A_k, P_k) — chaque pas appelle measure_state (depuis jax_features_new)
  3. `_post_scan_jax(signals, last_state, A_k_final)` → dict features (intégré)
- Observations :
  - dmd_rank argument statique → recompilation nécessaire si changé.
  - float() sur premier scalaire bloque implicitement.
  - **af-ram** : carry contient matrices n_dof×n_dof, mais en batch vmap le carry n'est pas stocké en batch.

### `running/plan_new.py`
- **Lu** ✓
- Fonctions :
  - `generate_kernel_groups(run_config, registries) → generator[Dict]`
  - `split_into_batches(kernel_group, batch_size) → List[Dict]`
  - `count_total_samples(run_config, registries) → Dict`
  - `_resolve_gamma_axis(...)`, `_resolve_gamma_sequence(...)`, `make_composed_gamma(...)`
- Rôle : transformer YAML + registries en kernel_groups vmappables (même clé de compilation XLA). Streaming.
- Contenu d'un kernel_group : gamma_id, gamma_fn, enc_id, enc_fn, mod_id, mod_fn, n_dof, rank_eff, max_it, is_differentiable, non_markovian, enc_vmappable, gamma_param_keys, mod_param_keys, prepare_params, phase, samples.
- Gestion des séquences pondérées : stocke sequence_fns et sequence_ids dans le dict composé.
- Observations :
  - **af-ext** : make_composed_gamma retourne nouvelle fonction → clé de cache distincte.
  - _get_rank_eff() : priorité à metadata['rank'], sinon params['rank'].
  - _seed_to_key() et _make_run_key() garantissent clés JAX déterministes.
  - **af-dead** : count_total_samples itère sur tous les groupes (peut être lourd).

### `running/run_one_jax.py`
- **Lu** ✓ (redondant avec kernel_new)
- Fonctions identiques à kernel_new.run_one_jax mais importe depuis hub_featuring.
- Observations :
  - Strictement redondant avec kernel_new.run_one_jax (même signature, imports différents).
  - **af-dead** : vestige probable, à supprimer si transition complète.

---

## Featuring — extraction intra-run

### `featuring/hub_featuring.py`
- **Lu** ✓
- `__all__ = ['measure_state', 'post_scan', 'FEATURE_NAMES']`
- Rôle : hub de routage pur — réexporte depuis `featuring.jax_features` (module non présent, probablement jax_features_new.py).
- Observations :
  - Import incohérent (jax_features vs jax_features_new).
  - **af-ext** : point d'entrée unique pour les kernels.

### `featuring/jax_features_new.py`
- **Lu** ✓
- Fonctions :
  - `measure_state(state, state_next, state_prev, gamma_fn, params, key, A_k, P_k, is_differentiable, sigmas_prev) → (measures, A_k_new, P_k_new, sigmas)`
  - `_post_scan_jax(signals, last_state, A_k) → dict features JAX` (vmappable)
  - `post_scan(signals, last_state, A_k, P_k) → dict {str: float}`
- Contient constantes `FEATURE_NAMES` (68 noms) et `FEATURES_STRUCTURAL_NAN` (liste des NaN par applicabilité).
- Organisation :
  - measure_state : appelée à chaque pas de scan.
  - _post_scan_jax : agrège signaux, calcule F6, F7, autocorr, santé.
  - post_scan : wrapper compatibilité pour run_one_jax.
- Points clés :
  - Utilisation de `lax.cond` pour branches conditionnelles (vmappable).
  - `FEATURES_STRUCTURAL_NAN` pour distinguer NaN structurels.
  - Santé : `health_has_inf` détecté via `jnp.stack`.
  - DMD streaming : `_dmd_streaming_update`.
- Observations :
  - **af-ext** : ajout d'une nouvelle feature en étendant les fonctions _fN_*.
  - _first_min_autocorr utilise `lax.dynamic_slice`.
  - FEATURE_NAMES_ORDERED_FOR_HEALTH dérivé manuellement.
  - **af-ram** : _approx_transfer_entropy O(T) par run.

### `featuring/configs/features_applicability.yaml`
- **Lu** ✓
- Source de vérité pour NaN structurels. Liste les features qui peuvent être NaN pour raisons d'applicabilité (ex: entanglement mode1 sur rang=2, features différentielles sur gamma non diff).
- Structure : chaque entrée indique condition, colonne parquet, valeur déclenchant.
- Observation : champ `parquet_val: registry` pour gammas non différentiables (résolution via registre).

---

## Analysing — clustering + verdict

### `analysing/clustering_peeling_v2.py`
- **Lu** ✓
- Fonctions :
  - `run_peeling(M_ortho, cfg, mcs_global=None, ms_global=None, M_2d=None, run_regimes=None, output_dir=None, label='peeling', save_debug=False, verbose=False) → Dict`
  - `_run_level(...)`, `_homogeneity_score(...)`, `_mcs_residual(...)`
- Rôle : peeling résiduel HDBSCAN. Plusieurs niveaux, adaptation des paramètres, score d'homogénéité.
- Principe :
  - Niveau 0 : config unique sur tous les points.
  - Niveaux suivants : batterie de configs sur le résidu.
  - Chaque cluster candidat évalué par score d'homogénéité (moyenne probabilités, silhouette, pureté régimes). Si score ≥ seuil (fonction du niveau) → extrait.
  - mcs adaptatif sur résidu.
- Détails :
  - `_run_level` essaie toutes les configs, choisit la meilleure selon score composite (nb clusters, bruit, ARI bootstrap).
  - Bootstrap ARI optionnel.
- Observations :
  - run_regimes optionnel pour pureté.
  - save_debug=True sauvegarde labels.npy et trace JSON.
  - **af-ram** : bootstrap ARI coûteux si n_iter grand ; désactivable.
  - **af-ram** : boucle sur configs à chaque niveau.

### `analysing/clustering_v2.py`
- **Lu** ✓
- Fonctions :
  - `prepare_matrix(data, corr_threshold=0.85, protected_features=None) → (M_ortho, kept_names, matrix_meta)`
  - `compute_projection(M_ortho, cfg=None, cache_path=None) → M_2d`
  - `run_clustering(M_ortho, feat_names, peeling_cfg, M_2d=None, output_dir=None, label='clustering', save_debug=False, verbose=False) → Optional[Dict]`
- Rôle : préparation matrice (normalisation, log-transform, robust scaling, orthogonalisation) et interface peeling.
- prepare_matrix :
  - Utilise `data.features_for_ml()` pour exclure health_*.
  - nan→0.0, log-transform si dynamique >1e6, RobustScaler, sélection orthogonale (corr<0.85, protège PROTECTED_FEATURES_V7).
- compute_projection : PCA(50) puis t-SNE ou UMAP selon seuil, cache .npy.
- run_clustering : appelle run_peeling avec paramètres config.
- Observations :
  - PROTECTED_FEATURES_V7 jamais éliminées.
  - En cas d'échec RobustScaler, utilise données brutes.
  - **af-ram** : calcul de corrélation O(F² n).

### `analysing/cluster_namer_v2.py`
- **Lu** ✓
- Fonctions :
  - `build_cluster_profile(cluster_mask, M_ortho, feat_names) → dict`
  - `build_layer_distribution(M_ortho, feat_names) → dict`
  - `ClusterNamer(cfg).name_cluster(profil, layer_dist, cluster_homogeneity, n, cluster_id) → dict`
  - `ClusterNamer.name_all(peeling_result, M_ortho, feat_names) → List[dict]`
  - `print_naming_report(...)`
- Rôle : nommage compositionnel des clusters piloté par YAML (cluster_namer.yaml).
- Architecture :
  - Profil : statistiques (médiane, IQR, etc.) des features du cluster.
  - Layer distribution : distribution globale pour percentiles.
  - Pour chaque slot, handler selon mode (zones, delta, threshold, uncalibrated) retourne terme + confiance.
  - Termes avec conf ≥ conf_min_name → nom primaire, ≥ conf_min_display → secondaire.
- Modes :
  - `zones` : exclusif avec bornes lo/hi (AMP, DMD).
  - `delta` : bidirectionnel autour de 0, omit_if_neutral (ENT, RNK, LYA).
  - `threshold` : seuil unique, omit_if_clean (CND).
  - `uncalibrated` : placeholder (SEQ).
- Observations :
  - **af-ext** : ajout d'un nouveau slot = entrée YAML ; nouveau mode = handler + _EVAL_DISPATCH.
  - Mode uncalibrated documenté comme dette technique (SEQ nécessite F8).
  - `_percentile_rank` utilise np.searchsorted vectorisé.
  - **af-ram** : build_cluster_profile itère sur features (~68).

### `analysing/concordance_lite.py`
- **Lu** ✓
- Fonctions :
  - `compute_kappa_stub(labels1, labels2) → float` (stub = 0.5)
  - `compute_dtw_stub(traj1, traj2) → float` (stub = distance euclidienne)
  - `run_concordance_cross_phases(phases_data) → Dict`
- Rôle : stub pour analyse de concordance entre phases (utilisé par hub_analysing_v2).
- Observations :
  - **af-dead** : code non fonctionnel, à remplacer par vraie implémentation (sklearn, dtaidistance).
  - **af-ext** : point d'extension clairement identifié.

### `analysing/hub_analysing_v2.py`
- **Lu** ✓
- Fonctions :
  - `scan_major_phases(results_dir) → List[Path]` (filtre r0.parquet, r1.parquet...)
  - `run_verdict_from_parquet(parquet_path, cfg_path=None, output_dir=None, label=None, plot=True, save_debug=False) → Dict`
  - `run_verdict_cross_phases(results_dir=None, cfg_path=None, output_dir=None, plot=False) → Dict`
- Rôle : hub de routage pour l'analyse. Charge config, filtre données via parquet_filter, exécute pipeline, écrit rapports (verdict_v2).
- Workflows détaillés dans le code.
- Observations :
  - apply_pool lu depuis cfg['pool_requirements']['apply'].
  - output_dir par défaut = data/results/reports/{label} si plot=True.
  - run_concordance_cross_phases importé de concordance_lite (stub).
  - **af-ext** : point d'entrée unique pour verdicts.

### `analysing/outliers_v2.py`
- **Lu** ✓
- Fonctions :
  - `detect_outliers(data, contamination=0.1) → (outlier_mask, stable_mask)`
  - `compute_atomic_recurrence(entity_arr, mask) → Dict[str, Dict]`
  - `analyze_outliers(data, contamination=0.1) → Dict`
- Rôle : détection outliers via IsolationForest, calcul de récurrence des atomiques parmi outliers.
- Utilise `data.features_for_ml()` pour exclure health_*. Remplace NaN par 0.0.
- Observations :
  - contamination paramètre depuis config.
  - **af-ram** : IsolationForest réentraîné à chaque appel.

### `analysing/parquet_filter_v2.py`
- **Lu** ✓
- Fonctions :
  - `load_analysing_data(parquet_path, scope=None, apply_pool=False, pool_path=None, verbose=True) → AnalysingData`
  - `features_for_ml(self) → (M, feat_names)` (sans health_*)
  - `_mask_seeds_one(data)`, `_mask_pool_requirements(data, req)`
- Rôle : chargement et filtrage du parquet vers format columnar `AnalysingData`.
- AnalysingData : M (n,F) float32 + meta arrays (gamma_ids, encoding_ids, modifier_ids, n_dofs, rank_effs, max_its, run_statuses, phases, seed_CIs, seed_runs).
- Filtres : pushdown pyarrow sur run_status, n_dof, rank_eff, modifier_id ; seeds:one (un échantillon par (gamma,encoding)) ; pool_requirements (n_dof min/max + deprecated).
- Observations :
  - seed_CI/seed_run None → -1 pour typage.
  - _mask_seeds_one vectorisé via np.unique.
  - **af-ram** : conversion toutes features en float32 coûteuse en mémoire.

### `analysing/pipeline_v2.py`
- **Lu** ✓
- Fonction principale : `run_analysing_pipeline(data, cfg, output_dir=None, label='verdict', plot=True, save_debug=False) → Dict`
- Orchestration complète :
  1. run_profiling(data)
  2. analyze_outliers(data, contamination)
  3. prepare_matrix(data, corr_threshold) → M_ortho
  4. Si plot : compute_projection(M_ortho, cfg['projection']) → M_2d
  5. run_clustering(M_ortho, feat_names, peeling_cfg, ...)
  6. ClusterNamer.from_yaml().name_all(...)
  7. Si plot : ClusterVisualizer(...).plot_all(...)
  8. Retourne dict avec résultats + metadata.
- Observations :
  - prepare_matrix appelée une seule fois.
  - data.n < 5 → résultat vide.
  - **af-obs** : M_2d conservé dans résultat (mémoire).
  - **af-ram** : M_ortho gardée en mémoire jusqu'à la fin.

### `analysing/verdict_v2.py`
- **Lu** ✓
- Fonctions :
  - `write_verdict_report(verdict_results, output_path) → None` (JSON)
  - `write_verdict_report_txt(verdict_results, output_path) → None` (TXT)
- Rôle : génération des rapports. Zéro calcul.
- JSON : sérialise types numpy, exclut 'M_2d'.
- TXT : format lisible avec header, outliers, clusters nommés, signatures, insights.
- Observations :
  - Sérialisation JSON avec `default=_serial`.
  - TXT affiche signatures vectorielles et slots non calibrés.

### `analysing/visualizer_v2.py`
- **Lu** ✓
- Classe `ClusterVisualizer` :
  - `plot_peeling_summary(output_dir, label)`
  - `plot_layer(layer_name, output_dir, label)`
  - `plot_signature_heatmap(output_dir, label)`
  - `plot_all(output_dir, label)`
- Rôle : génération PNG (clusters, niveaux, heatmap, vues par layer). Thème sombre.
- Observations :
  - Utilise `matplotlib.use('Agg')`.
  - `fix_str` (de utils.data_loading_new) pour noms.
  - **af-ext** : légendes slots continuous (LAG, PNN) en dur ; pourraient être externalisées.

### `analysing/configs/analysing_default.yaml`
- **Lu** ✓
- Configuration par défaut du pipeline :
  - hdbscan, scope, orthogonalization, pool_requirements, peeling, outliers, projection.
- Détails :
  - scope : run_status [OK,EXPLOSION], seeds all, n_dof all, rank_eff all, modifiers all.
  - peeling : homogeneity (threshold_base 0.55, step 0.05, max 0.80, weights), mcs_floor 8, ms_floor 3, max_levels 6, min_delta_extracted 1, residual_configs (cosine/eom, euclidean/leaf, cosine/leaf).
  - outliers : contamination 0.10.
  - projection : umap_threshold 50000.

### `analysing/configs/cluster_namer.yaml`
- **Lu** ✓
- Registre de nommage :
  - conf_min_name: 0.60, conf_min_display: 0.50, separator "·", heterogeneous_threshold 0.70.
  - slot_order : CND, AMP, ENT, RNK, LYA, DMD, LAG, PNN, SEQ.
  - Définition de chaque slot avec mode, features, termes, seuils.
- Exemples :
  - CND : mode threshold, termes CND! (inf_frac>0.2) et CND~ (condition_final>1e6).
  - AMP : mode zones avec termes AMP>>, AMP<<, AMP~0, AMP+, AMP++, AMP>>> (avec sentinel 148.413).
  - ENT, RNK : mode delta.
  - LYA : mode delta avec feature_std pour LYA~.
  - DMD : mode zones avec condition complex_pairs_gt_0.
  - LAG, PNN : mode delta (seuils provisoires).
  - SEQ : calibrated: false (placeholder).

---

## Utils — data loading, conversions

### `utils/data_loading_new.py`
- **Lu** ✓
- Fonctions :
  - `discover_gammas_jax()`, `discover_encodings_jax()`, `discover_modifiers_jax() → registries`
  - `load_yaml(path) → dict`
  - `open_parquet_writer(phase, output_dir, sample_row) → pq.ParquetWriter`
  - `write_rows_to_parquet(writer, rows, phase) → int`
  - `close_parquet_writer(writer)`
  - `read_parquet_rows(parquet_path, filters) → list[dict]`
  - `merge_configs(base, override) → dict`
  - `fix_str(s) → str`
- Rôle : module utilitaire central. Découverte des atomiques, lecture YAML, écriture/lecture parquet.
- Découverte : scan répertoires, import dynamique, vérifie METADATA et callable. Lève CriticalDiscoveryError si manquant. prepare_params optionnel.
- Parquet : écriture par batches (append) via writer. Paramètres sérialisés JSON. Lecture avec pushdown pyarrow optionnel.
- Observations :
  - read_parquet_rows utilisée par parquet_filter_v2 (découplé).
  - merge_configs : deep merge récursif.
  - fix_str corrige doubles encodages.

### `utils/parquet_to_json.py`
- **Non lu** (fichier non fourni)

---

## YAMLs de configuration — tous lus (sauf phases)

### `configs/pool_requirements.yaml`
- **Lu** ✓
- Contraintes globales : n_dof (min/max null), max_iterations (null), deprecated (gammas, encodings, modifiers vides). Version 1.0.
- Utilisé par parquet_filter_v2 si apply_pool=True.

### Autres YAMLs de phases (non fournis) :
- `configs/phases/aot_bench.yaml`
- `configs/phases/poc/poc.yaml`, poc1.yaml, poc_bourrin_v7.yaml, poc_cache_bench.yaml, poc_perf.yaml
→ Non lus (absents de l'audit).

---

## Atomiques — opérateurs, encodings, modificateurs
- D_encodings : 20 fichiers (asy, r3, rn, sym) — contrat METADATA+create()
- modifiers : m0, m1, m2 + plugins (biology, gravity, quantum, template)
- operators : 15 gammas (gam_001..015) — pas de g015_svd_truncation.py (supprimé)
- L'audit se concentre sur le pipeline ; ces composants sont supposés conformes au contrat d'interface. Leur découverte est assurée par `data_loading_new.py`.

---

## Synthèse audit

### Code mort & redondances
- `run_one_jax.py` strictement redondant avec `kernel_new.py`.
- `concordance_lite.py` stub non fonctionnel.
- `_check_invalid` dans `hub_running_new.py` inopérant si rank_constraint absent.
- `count_total_samples` dans `plan_new.py` lourd et rarement utilisé.
- `rolling_compile_window` a paramètre `ram_gb` non utilisé.
- Import incohérent dans `hub_featuring.py` (`featuring.jax_features` vs `jax_features_new`).

### RAM non essentielle
- Double `prepare_matrix` dans hub_analysing (t‑SNE puis clustering).
- Conservation de `M_2d` dans le résultat.
- Boucles Python dans `_build_enc_batch` si `enc_vmappable=False`.
- Matrices de carry DMD (`n_dof×n_dof`) multipliées par taille du batch.
- Conversion de toutes les features en float32 dans `parquet_filter`.

### Points d'extension
- `_vmap_cache` : support de nouveaux gamma/modifier.
- `cluster_namer.yaml` : ajout de slots sans modifier le code.
- `_EVAL_DISPATCH` : ajout de nouveaux modes de nommage.
- `concordance_lite` : stub pour vraie analyse cross‑phases.
- `features_applicability.yaml` : extension à d'autres conditions structurelles.

### Interdictions actives
- Zéro logique métier dans les hubs (running, analysing, featuring).
- Les registres (timeline, universal) doivent rester des fonctions pures.
- Pas de I/O dans les fonctions de features.
- Les paramètres doivent être externalisés dans les YAML, pas de hardcoding (exceptions : sentinelle 148.413, seuils d'alerte RAM).
- Le code doit suivre l'ordre Algo → Structure → Code.
