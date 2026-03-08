# AUDIT PRC v3 — 2026-03-04
> Référence cognitive LLM. Sources lues = vérité. Charter = axiomes.
> Flags : 🔴 code mort | 🟡 RAM | 🟢 extension | ⚠ violation P4 | ℹ obs

---

## PIPELINE ACTIF (flux réel)
```
batch.py CLI
  MODE1: run_batch() → runner.run_single() × N → hub_featuring.extract_features() → write_parquet() → run_verdict_intra()
  MODE2: run_verdict_from_parquet() → run_verdict_intra()
  MODE3: scan_major_phases() [filtre r+1chiffre → jamais poc/poc2/poc3, toujours vide]

run_verdict_intra():
  filter_rows_by_pool()           [🔴 chemin configs/constraints/ inexistant → fallback vide]
  run_profiling()
  analyze_outliers(contamination=0.1)  [⚠ hardcodé]
  classify_regimes_batch()        [🔴 DEPRECATED → supprimer étape 2]
  _build_run_regimes()            [→ vecteur pour namer uniquement]
  run_analysing(run_regimes=...)
    prepare_matrix(rows)          [🟡 pour t-SNE global, non réutilisé dans clustering]
    compute_tsne()                [🟡 plot=True par défaut, systématique]
    run_clustering_stratified()
      group_rows_by_layers()
      run_clustering(layer_rows)
        prepare_matrix(layer_rows) [🟡 2ème prepare_matrix]
        run_peeling()
    ClusterNamer.name_all()
      build_layer_distribution()  [🔴 aussi appelée dans hub_analysing juste avant]
    ClusterVisualizer.plot_all()
      plot_peeling_summary()
      plot_signature_heatmap()
      plot_layer('universal')     [🔴 hardcodé]
      plot_regimes()              [conditionnel run_regimes → s'annule à None]
  write_verdict_report_txt()
```

---

## FICHIERS — RÉSUMÉ PAR MODULE

### core/
**kernel.py** — générateur `run_kernel(initial_state, gamma, max_iter, convergence_check, record_history)`
- K1-K5 confirmés. record_history=False par défaut, runner collecte côté consommateur.
- ℹ convergence_check en signature, jamais passé par runner.py

**state_preparation.py** — `prepare_state(encoding_func, encoding_params, modifiers, modifier_configs)`
- Séquentiel pur, aveugle. encoding(**params) → modifier_1(state) → modifier_2...

### running/
**hub_running.py** — `run_batch(yaml_path, auto_confirm, output_dir, verbose)`
- Benchmark: run_single() × n_ranks sans extract_features → 🟡 temps consommé, résultat jeté
- ⚠ seuil RAM 15Go hardcodé
- 🔴 verbose sans effet sur skips (prints commentés)
- output_dir passé mais batch.py ne le fournit jamais → toujours data/results/

**compositions.py** — `generate_compositions(run_config) → List[Dict]`
- Axes résolus : gamma (all|random N|list|sequence) · encoding · modifier · n_dof (liste ✓)
- 🔴 max_iterations : scalaire, non normalisé, absent du product() → non itérable
- 🔴 seed_CI + seed_run : totalement absents
- 🟢 Étape 3 : patron exact = n_dof (l.368 normalise, l.421 product, l.427 inject)
- Composition output : {gamma_id, gamma_params, gamma_callable, encoding_id, encoding_params, encoding_callable, modifier_id, modifier_params, modifier_callable, n_dof, max_iterations, phase}

**runner.py** — `run_single(composition, verbose) → np.ndarray history (T, *dims)`
- prepare_state → gamma.reset() si attribut → run_kernel (générateur) → collect history
- NaN/Inf → break + warnings.warn si verbose
- RunnerError sur: prepare_state fail, tenseur vide, kernel crash, shapes incohérentes

### featuring/
**hub_featuring.py** — `extract_features(history, config) → {features, layers}`
- glob `configs/minimal/*.yaml` non récursif → 🔴 Nouveau dossier/ silencieux
- Seul layer actif en pratique : **timeline**

**extractor_lite.py** — `extract_for_layer(history, layer_name, layer_config)`
- import dynamique `featuring.registries.{layer_name}_lite`
- 🔴 registres dans Nouveau dossier/ → import Python impossible (espace dans chemin)
- 🟢 Nouveau layer = nouveau fichier zéro touche ici

**layers_lite.py** — `inspect_history() → {rank, shape, n_dof, is_square, is_symmetric, ...}`
- `check_applicability()` : AND logique conditions YAML
- `group_rows_by_layers()` : un run peut appartenir à plusieurs layers

**timeline_lite.py** — `extract(history, config) → ~149 features`
- 5 signaux (_MEASURE_FNS) : euclidean_norm · entropy · effective_rank · condition_number_svd · singular_value_spread
- Pipeline : signal[t]=fn(history[t]) → tag qualité (finite_ratio, norm_absolute) → normalisation sign(r)*log1p(|r|) → catch22+24 → ~26 features par signal
- Dérivées (5) : norm_ratio · entropy_delta · effective_rank_delta · log_condition_delta · spread_ratio
- Absolues (2) : norm_final · condition_number_svd_final
- Santé (2) : has_nan_inf · is_collapsed
- 🟡 _get_catch22_names() appel dummy pycatch22 à chaque extract() → devrait être constante module-level
- NaN = signal physique, préservé. Placeholder 0.0 appliqué dans clustering_lite.

### analysing/
**profiling_lite.py** — fusion de profiling/hub_profiling.py + aggregation_lite.py (supprimés)
- `run_profiling(rows) → {gamma, encoding, modifier, n_observations}`
- Agrège median/Q1/Q3/n_runs par entité pour toutes features numériques
- `find_feature_variants()` : rétrocompat legacy _initial/_final/_mean

**regimes_lite.py** — 🔴 DEPRECATED, transition vers cluster_namer
- `classify_regime(features, thresholds) → str` — consomme depuis dict (pas recalcul)
- 8 régimes : NUMERIC_INSTABILITY · EXPLOSION_PROGRESSIVE · EFFONDREMENT · CONSERVES_NORM · CROISSANCE_FAIBLE · CROISSANCE_FORTE · SATURATION · UNCATEGORIZED
- Features consommées : has_nan_inf · norm_ratio · norm_final · condition_number_svd_final · euclidean_norm__signal_finite_ratio
- ⚠ _LOG1P_SENTINEL=148.413 hardcodé dans classify_regime()
- 🔴 ASYMMETRY_BREAKING + TRIVIAL dans fallback defaults, absents de classify_regime()
- Colonne regime NON écrite en parquet (RAM only)

**outliers_lite.py** — `analyze_outliers(rows, contamination=0.1)`
- IsolationForest sur features communes (intersection, sans has_*/is_*)
- sanitize : NaN→0.0, inf→±1e15
- 🔴 _extract_common_features() : 3ème occurrence (aussi clustering_lite, cluster_namer)
- ⚠ contamination=0.1 hardcodé (3 occurrences : ici, verdict.py × 2)

**concordance_lite.py** — 🔴 STUB INTÉGRAL non fonctionnel
- compute_kappa_stub() → 0.5 fixe
- compute_dtw_stub() → norme euclidienne sur np.random.randn() (pas les données)
- run_concordance_cross_phases() : labels fictifs ['regime_a']*n
- 🟢 Étape 6 : sklearn.metrics.cohen_kappa_score déjà disponible

**clustering_lite.py** — helpers préparation + interface peeling
- `prepare_matrix(rows) → (M_ortho, kept_names, meta)`
  - sanitize (inf→±1e15, NaN→0.0) → log_transform (si dynamique>1e6) → RobustScaler → _select_orthogonal_features (corr<0.85)
  - Protected : norm_ratio · norm_final · entropy_delta · effective_rank_delta · log_condition_delta · spread_ratio · condition_number_svd_final · *signal_finite_ratio · *signal_norm_absolute
- `compute_tsne(M_ortho, cache_path, perplexity) → M_2d` — PCA(50)→TSNE(2), cache .npy optionnel
- `run_clustering_stratified(rows) → {layer: result}` — group_rows_by_layers() → run_clustering() par layer
- 🟡 layer_regimes construction O(n²) : identité objet Python sur grande liste

**clustering_peeling.py** — `run_peeling(M_ortho, cfg, ...) → {labels, extracted, residual_idx, trace, n_clusters, n_unresolved}`
- Niveau 0 : cosine/eom espace complet
- Niveaux suivants : batterie configs sur résidu (6 configs)
- _homogeneity_score : mean_proba×wp + silhouette×ws [+ regime_purity×wr si run_regimes]
  → 🟢 run_regimes=None → wr ignoré automatiquement, zéro touche pour suppression regimes
- mcs adaptatif résidu : sqrt(n_res/n_total) / density_factor
- Arrêt : résidu<floor | rien extrait 2 niveaux | delta<min_delta
- Sorties debug (.npy + .json) gatées à save_debug=True (ignore YAML save_trace/save_labels)
- 🔴 CLI _prepare() : duplique prepare_matrix() — double sanitization standalone
- 🔴 'regime' dans COMPOSITION_COLS CLI : vestige, colonne jamais écrite en parquet

**cluster_namer.py** — `ClusterNamer(cfg).name_all(peeling_result, all_features) → List[Dict]`
- 6 slots tous calibrés (VITESSE+TEXTURE calibrés poc3) :

| Slot | Feature | Termes | Notes |
|------|---------|--------|-------|
| AMPLITUDE | norm_ratio | EXPLOSION·COLLAPSE·CONSERVE·CROISSANCE_FAIBLE/FORTE·SATURATION | sentinel 148.413 |
| SANTÉ_NUM | has_nan_frac + cond_svd_final | INSTABLE·CONDITIONNÉ | omit_if_clean |
| VITESSE | CO_FirstMin_ac | RAPIDE(<10)·LENT(>88) | calibré poc3 |
| TEXTURE | MD_hrv_classic_pnn40 | LISSE(<0.05)·CHAOTIQUE(>0.60) | calibré poc3 |
| ENTROPIE | entropy_delta | CROISSANTE(>0.05)·DÉCROISSANTE(<-0.05) | omit_if_neutral |
| RANG | effective_rank_delta | EFFONDRÉ(<-5)·CROISSANT(>5) | omit_if_neutral |

- build_cluster_profile() : median/iqr/min/max/mean/nan_frac par feature
- 🔴 sys.stdout redirect hors __main__ (l.37-38) → effet de bord global à l'import
- 🔴 build_layer_distribution() appelée ici ET dans hub_analysing juste avant → double calcul
- ⚠ sentinel 148.413 hardcodé ici + regimes_lite

**hub_analysing.py** — `run_analysing(rows, stratified=True, run_regimes, ...) → Dict`
- Retour : {n_observations, strategy, clustering, clustering_stratified, named_clusters, layer_distribution, M_2d, metadata}
- 🔴 n_clusters ignoré en signature (HDBSCAN découvre seul, conservé rétrocompat)
- 🟡 plot=True par défaut → t-SNE + PNG systématiques
- 🟡 M_2d retourné dans dict → reste en RAM jusqu'à fin verdict
- 🟡 double prepare_matrix() : global (t-SNE) + par layer (clustering)
- clustering_unified = next(iter(clustering_stratified.values())) → premier layer seulement (rétrocompat)

**verdict.py**
- run_verdict_intra() : filter_pool → profiling → outliers → [regimes DEPRECATED] → run_analysing → metadata
- run_verdict_from_parquet() : reconstruit rows avec layers=['timeline'] hardcodé 🔴
- run_verdict_cross_phases() : scan_major_phases() filtre r+1chiffre → jamais poc* 🔴
- 🔴 load_pool_requirements() : cherche configs/constraints/pool_requirements.yaml → inexistant → fallback vide
- _fix_str() : corrige double-encodage cp1252/utf-8 (3ème occurrence)

**visualizer.py** — ClusterVisualizer.plot_all()
- 4 vues : peeling_summary · signature_heatmap · regimes (conditionnel) · layer('universal' hardcodé)
- Zéro calcul, matplotlib Agg. warnings.filterwarnings('ignore') module-level.

### utils/
**data_loading_lite.py**
- discover_gammas/encodings/modifiers() : scan glob → import dynamique → METADATA['id'] obligatoire (CriticalDiscoveryError)
- load_yaml(identifier|Path, mode) : Path direct OU identifier+merge default+override
- write_parquet(rows, phase, output_dir) : composition(6 cols) + features unpack → layers NON persisté
- _YAML_PATHS : operators · D_encodings · modifiers — featuring/thresholds/verdict en commentaire

---

## YAMLS ACTIFS

| Fichier | Clé | État |
|---------|-----|------|
| configs/phases/poc/poc3.yaml | phase:poc3 max_iterations:100(scalaire) n_dof:100(scalaire) axes:all | étape 3 non faite |
| featuring/configs/minimal/timeline.yaml | applicability:[] catch24:true 5 fonctions | seul yaml actif |
| analysing/configs/clustering_peeling.yaml | max_levels:4 weight_regime:0.2 threshold_base:0.65 step:0.05 | save_trace/labels ignorés (save_debug gate) |
| analysing/configs/cluster_namer.yaml | 6 slots calibrés conf_min_name:0.65 sentinel:148.413 | VITESSE+TEXTURE calibrés poc3 |
| configs/pool_requirements.yaml | toutes contraintes null | 🔴 chemin divergent avec verdict.py |

---

## ÉTAPES ROADMAP — PLAN CHIRURGICAL

### Étape 2 — Supprimer regimes_lite (PRÊT)
```
verdict.py:
  - supprimer imports l.30-34 (load_regime_thresholds, classify_regimes_batch, refine_conserves_norm_with_cv)
  - supprimer bloc "Régimes" l.153-162
  - run_analysing(run_regimes=None)
  - garder 'regimes': {} dans dict retour (rétrocompat write_verdict_report_txt)
  - garder plot_regimes dans visualizer inchangé (s'annule si run_regimes=None)

Supprimer ensuite:
  regimes_lite.py
  analysing/configs/regimes/  (3 yaml)

Zéro touche: clustering_peeling · hub_analysing · cluster_namer · visualizer
```

### Étape 3 — Axes optionnels (compositions.py)
```
Patron exact à reproduire (n_dof déjà fait):
  l.368 : max_its = _normalize_to_list(run_config.get('max_iterations', 200))
          seed_CIs = _normalize_to_list(run_config.get('seed_CI', [None]))
          seed_runs = _normalize_to_list(run_config.get('seed_run', [None]))
  l.421 : product(gammas, encodings, modifiers, n_dofs, max_its, seed_CIs, seed_runs)
  l.427 : encoding_params inject seed_CI si non None
  l.428 : gamma_params inject seed_run si non None
  l.433 : max_iterations: max_it (itéré)
```

### Étape 4a — Layer spatial
```
Créer: featuring/configs/minimal/spatial.yaml (avec applicability: conditions)
Créer: featuring/registries/spatial_lite.py exposant extract(history, config)
Zéro touche: hub_featuring · extractor_lite · layers_lite
Prérequis: renommer "Nouveau dossier/" (espace = import Python impossible)
```

### Corrections urgentes (avant étape 2)
```
1. pool_requirements.yaml : déplacer dans configs/constraints/ OU corriger verdict.py l.42
2. scan_major_phases() : ajouter filtre poc|r\d+ pour trouver les phases réelles
3. MODE 2 layers hardcodé : lire layers depuis parquet ou passer en param
```

---

## CODE MORT RÉEL (hors archives intentionnelles)

| Fichier | Problème |
|---------|----------|
| regimes_lite.py | DEPRECATED, étape 2 |
| concordance_lite.py | Stub intégral, résultat fictif |
| hub_running.py prints [SKIP] | Commentés, verbose sans effet skips |
| compositions.py seed_CI/seed_run | Absents, max_it non itérable |
| hub_analysing.py n_clusters param | Ignoré, mort en signature |
| clustering_peeling.py CLI 'regime' col | Vestige, jamais écrit en parquet |
| cluster_namer.py sys.stdout l.37-38 | Hors __main__, effet de bord import |
| verdict.py scan_major_phases() | Jamais de match sur projet actuel |
| verdict.py load_pool_requirements() | Chemin inexistant, fallback vide |
| MODE 2 layers=['timeline'] | Hardcodé, layers futurs ignorés |
| _extract_common_features() | ×3 (clustering_lite, outliers_lite, cluster_namer) |
| _fix_str() | ×3 (verdict, visualizer, cluster_namer CLI) |
| _INF_SENTINEL=1e15 | ×2 (clustering_lite, outliers_lite) |
| sentinel 148.413 | ×2 (regimes_lite, cluster_namer) |

## ARCHIVES INTENTIONNELLES (pas du code mort)
- `*_old.py` + `Nouveau dossier/` dans featuring/ : isolés par design, hors discovery

## VIOLATIONS P4 (hardcodés, devraient être YAML)
- seuil RAM 15Go (hub_running)
- contamination=0.1 (outliers_lite × 2 + verdict.py)
- _LOG1P_SENTINEL=148.413 (regimes_lite)
- sentinel 148.413 (cluster_namer)
