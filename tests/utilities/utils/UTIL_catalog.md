# UTIL CATALOG

> Modules utilitaires spécialisés  
> Délégation stricte, zéro duplication  

## RÈGLES ARCHITECTURALES

**R-UTIL-1** : Modules UTIL ne dépendent JAMAIS de HUB

**R-UTIL-2** : Délégation stricte (pas de duplication code)
```
✅ profiling_common → timeline_utils.compute_timeline_descriptor()
❌ profiling_common → implémentation timeline locale
```

**R-UTIL-3** : Fonctions privées préfixées `_` (usage interne uniquement)

**R-UTIL-4** : Pas de calculs inline HUB (extraction UTIL obligatoire)

**R-UTIL-5** : Global configs documentés (TIMELINE_THRESHOLDS, ENTITY_KEY_MAP, etc.)

## MODULES DÉTAILLÉS

### aggregation_utils.py
**Responsabilité** : Agrégations statistiques multi-runs

**Fonctions publiques** :
- `aggregate_summary_metrics(observations, metric_name)` : Agrège métriques (median, q1, q3, cv)
  - Retour : `{final_value: {median, q1, q3, mean, std}, initial_value, mean_value, cv}`
- `aggregate_run_dispersion(observations, metric_name)` : Détecte multimodalité
  - Retour : `{final_value_iqr_ratio, cv_across_runs, bimodal_detected}`
  - Seuil bimodal : IQR ratio > 3.0 (heuristique R0)

**Placeholders futurs** :
- `compute_dominant_value()` : Mode/médiane/moyenne catégorielle
- `aggregate_event_counts()` : Comptage événements boolénsa

**Notes** :
- Protection division par zéro (+ 1e-10)
- IQR ratio : Q3 / max(Q1, 1e-10)
- CV : std(final) / |mean(final)|

### applicability.py
**Responsabilité** : Validation applicabilité tests

**Fonctions publiques** :
- `check(test_module, run_metadata)` : Vérifie applicabilité
  - Retour : `(applicable: bool, reason: str)`
  - Validation sur metadata uniquement (pas sur state)
- `add_validator(name, validator)` : Ajoute validator custom

**Validators disponibles** (VALIDATORS dict) :
- `requires_rank` : None ou int (vérifie len(state_shape))
- `requires_square` : bool (vérifie shape[0] == shape[1])
- `allowed_d_types` : list (vérifie encoding dans liste)
- `requires_even_dimension` : bool (vérifie dim % 2 == 0)
- `minimum_dimension` : None ou int (vérifie dim >= min)

**Notes** :
- None/False = pas de contrainte
- Extensible via add_validator()

### config_loader.py
**Responsabilité** : Chargement configs YAML avec fusion

**Architecture** : Singleton pattern (get_loader())

**Workflow** :
```
ConfigLoader.load(config_type, config_id, test_id)
  ↓
  1. Load global (obligatoire)
  2. Load specific (optionnel)
  3. Merge (specific override global)
  4. Cache + Return
```

**Fonctions publiques** :
- `load(config_type, config_id, test_id, force_reload)` : Charge + fusionne
- `list_available(config_type)` : Liste configs disponibles
- `clear_cache()` : Vide cache

**Config types** : `'params'`, `'verdict'`

**Structure BASE_PATH** :
```
tests/config/
  ├── global/
  │   ├── params_default_v1.yaml
  │   └── verdict_default_v1.yaml
  └── tests/
      └── UNIV-001/
          └── params_custom_v1.yaml
```

### cross_profiling.py
**Responsabilité** : Analyses croisées entre axes profiling

**Architecture** : Séparation calculs (R0) / interprétation (R1+)

**Fonctions R0 (implémentées)** :
- `rank_entities_by_metric()` : Ranking générique entités
  - Critères : 'conservation', 'stability', 'homogeneity', callable
- `compute_discriminant_power()` : Variance inter/intra entités
  - Retour : `{inter_entity_variance, intra_entity_variance, discriminant_ratio, effect_size, kruskal_wallis, entities_ranked, interpretation}`
- `compute_all_discriminant_powers()` : Discriminant power tous tests
- `analyze_pairwise_interactions()` : Interactions 2-way
  - R0 : metric='regime_concordance' implémenté
  - R1+ : autres métriques placeholders

**Fonctions R1+ (placeholders)** :
- `analyze_multiway_interactions()` : Interactions n-way (3+ axes)
- `detect_global_signatures()` : Vocabulaire interprété (INVARIANT, AMPLIFIED, etc.)

**Notes** :
- Paires orientées : permutations() pas combinations()
- Interaction vraie : VR(A|B) >> VR(A) marginal

### data_loading.py
**Responsabilité** : I/O observations depuis DBs

**Double connexion** :
- `prc_r0_results.db` : TestObservations
- `prc_r0_raw.db` : Executions (métadonnées runs)

**Fonctions publiques** :
- `load_all_observations(params_config_id, db_results_path, db_raw_path)` : Charge obs + metadata
  - Retour : Liste dicts `{observation_id, exec_id, run_id, gamma_id, d_encoding_id, modifier_id, seed, test_name, params_config_id, observation_data, computed_at}`
- `observations_to_dataframe(observations)` : Convertit obs → DataFrame
  - Projections : value_final, value_initial, value_mean, slope, volatility, relative_change, transition, trend
- `cache_observations(observations, cache_path)` : Cache pickle
- `load_cached_observations(cache_path)` : Charge cache

**Notes** :
- Fusion exec_id (clé primaire double connexion)
- Filtre NaN (lignes sans aucune projection valide)

### discovery.py
**Responsabilité** : Découverte automatique tests actifs

**Fonctions publiques** :
- `discover_active_tests()` : Découvre tests (skip _deprecated)
  - Retour : `{test_id: module}`
- `validate_test_structure(module)` : Valide structure 5.5
  - Vérifie : Attributs requis, types, version 5.5, format TEST_ID, COMPUTATION_SPECS

**Attributs requis** (REQUIRED_ATTRIBUTES) :
- TEST_ID, TEST_CATEGORY, TEST_VERSION, APPLICABILITY_SPEC, COMPUTATION_SPECS

**Validations** :
- TEST_VERSION == '5.5'
- TEST_ID format : `^[A-Z]{3,4}-\d{3}$`
- COMPUTATION_SPECS : 1-5 métriques
- Chaque métrique : registry_key + default_params
- Pas de FORMULAS legacy

### profiling_common.py
**Responsabilité** : Profiling générique tous axes (gamma, modifier, encoding, test)

**Architecture unifiée** : Moteur générique avec API découvrable

**Fonctions publiques (API découvrable)** :
- `profile_all_tests(observations)` : Profil tests
- `compare_tests_summary(profiles)` : Comparaisons tests
- `profile_all_gammas(observations)` : Profil gammas
- `compare_gammas_summary(profiles)` : Comparaisons gammas
- `profile_all_modifiers(observations)` : Profil modifiers
- `compare_modifiers_summary(profiles)` : Comparaisons modifiers
- `profile_all_encodings(observations)` : Profil encodings
- `compare_encodings_summary(profiles)` : Comparaisons encodings

**Fonctions communes** :
- `aggregate_dynamic_signatures()` : Agrège événements + timelines
- `compute_prc_profile()` : Génère profil PRC complet

**Moteur interne** (privé) :
- `_profile_test_for_entity()` : Profil test sous entité
- `_profile_entity_axis()` : Générique tous axes
- `_compare_entities_summary()` : Comparaisons cross-entities

### regime_utils.py
**Responsabilité** : Stratification et classification régimes

**Fonctions publiques** :
- `stratify_by_regime(observations, threshold=1e50)` : Stable/explosif
  - Retour : `(obs_stable, obs_explosif)`
  - Conserve TOUTES observations (aucun filtrage)
- `classify_regime(metrics, dynamic_sig, timeline_dist, dispersion, test_name)` : Régime spécifique
  - Retour : str (ex: 'CONSERVES_SYMMETRY', 'MIXED::CONSERVES_NORM')
- `detect_conserved_property(test_name)` : Déduit propriété conservée
  - Mapping : SYM-* → CONSERVES_SYMMETRY, SPE-*/UNIV-* → CONSERVES_NORM, etc.
- `extract_conserved_properties(profile)` : Extrait propriétés depuis profil
- `get_regime_family(regime)` : Retourne famille régime

**Régimes disponibles** (REGIME_TAXONOMY) :
- **Conservation** : CONSERVES_SYMMETRY, CONSERVES_NORM, CONSERVES_PATTERN, CONSERVES_TOPOLOGY, CONSERVES_GRADIENT, CONSERVES_SPECTRUM
- **Pathologies** : NUMERIC_INSTABILITY, OSCILLATORY_UNSTABLE, TRIVIAL, DEGRADING
- **Autres** : SATURATES_HIGH, UNCATEGORIZED

**Qualificatif** : MIXED::{régime_base} si bimodal détecté

### report_writers.py
**Responsabilité** : Formatage rapports structurés

**Fonctions publiques** :
- `write_json(data, filepath, indent=2)` : JSON formaté
- `write_header(f, title, width, char)` : Header section TXT
- `write_subheader(f, title, width, char)` : Sous-header
- `write_key_value(f, key, value, indent)` : Paire clé-valeur
- `write_regime_synthesis(f, gamma_profiles, width)` : Synthèse régimes transversale
- `write_dynamic_signatures(f, gamma_profiles, width)` : Signatures dynamiques
- `write_comparisons_enriched(f, comparisons, gamma_profiles, width)` : Comparaisons contexte
- `write_consultation_footer(f, width, char)` : Footer fichiers

**Notes** :
- Sérialisation tuples → strings (`_make_json_serializable`)
- Formatage TXT lisible humains
- Sections standardisées (header, body, footer)

### statistical_utils.py
**Responsabilité** : Outils statistiques réutilisables

**Calculs variance** :
- `compute_eta_squared(groups)` : η² (proportion variance expliquée)
  - Formule : η² = SSB / (SSB + SSW)
  - Retour : `(eta2, ssb, ssw)`
- `kruskal_wallis_test(groups)` : Test non paramétrique
  - Retour : `(statistic, p_value)`

**Filtrage artefacts** :
- `filter_numeric_artifacts(observations)` : Filtre inf/nan
  - Retour : `(valid_obs, rejection_stats)`
  - Stats rejets par test

**Diagnostics** :
- `generate_degeneracy_report(observations)` : Rapport dégénérescences
  - Flags : INFINITE_PROJECTION, NAN_PROJECTION, EXTREME_MAGNITUDE
- `print_degeneracy_report(report)` : Affiche rapport (stdout)
- `diagnose_scale_outliers(observations)` : Ruptures échelle relatives
  - Critère : > P90 + 5 décades (facteur 1e5)
- `print_scale_outliers_report(report)` : Affiche rapport (stdout)

**Notes** :
- Protection division par zéro
- Filtrage groupes vides automatique
- Diagnostics contextuels (test×métrique×projection)

### timeline_utils.py
**Responsabilité** : Timelines dynamiques compositionnels

**Architecture** : Seuils globaux relatifs (TIMELINE_THRESHOLDS)

**Fonctions publiques** :
- `classify_timing(onset_relative)` : early/mid/late
  - Seuils : early < 0.20, mid ≤ 0.60, late > 0.60
- `compute_timeline_descriptor(sequence, sequence_timing_relative, oscillatory_global)` : Composition automatique
  - Format : `{timing}_{event}_then_{event}`
  - Retour : `{phases, timeline_compact, n_phases, oscillatory_global}`
- `extract_dynamic_events(observation, metric_name)` : Parse dynamic_events
- `extract_metric_timeseries(observation, metric_name)` : Série temporelle + fallback

**Principe** :
- Toute notion temporelle RELATIVE (jamais absolue)
- Composition automatique (pas patterns hardcodés)
- Descriptif pas causal ("then" pas "causes")

**Exemples timelines** :
- `early_instability_then_collapse`
- `mid_deviation_then_saturation`
- `oscillatory_early_deviation_only`

## DÉPENDANCES AUTORISÉES

```
UTIL Niveau 1:
  → NumPy, Pandas, SciPy

UTIL Niveau 2:
  → Niveau 1

UTIL Niveau 3:
  → Niveau 2 + timeline_utils, aggregation_utils, regime_utils

UTIL Niveau 4:
  → Niveau 3 + profiling_common

INTERDICTIONS:
  ❌ UTIL → HUB
  ❌ UTIL → core, operators, D_encodings, modifiers, tests
```

## EXTENSIONS FUTURES

**Checklist ajout nouveau module UTIL** :
- [ ] Identifier responsabilité unique
- [ ] Vérifier absence duplication (checker tables "AVANT DE CODER")
- [ ] Déterminer niveau hiérarchie (1-4)
- [ ] Documenter dépendances autorisées
- [ ] Implémenter fonctions publiques + privées
- [ ] Ajouter à ce catalogue (tables exhaustives)
- [ ] Mettre à jour FUNCTIONS_INDEX.md

**Exemples extensions acceptables** :
- ✅ Nouvelles agrégations statistiques (si réutilisables)
- ✅ Nouveaux formatters rapports (si standards)
- ✅ Utilitaires validation données

**Exemples extensions REFUSÉES** :
- ❌ Duplication code existant (vérifier "AVANT DE CODER")
- ❌ Fonctions spécifiques 1 cas d'usage (pas génériques)
- ❌ Dépendances UTIL → HUB (violé hiérarchie)

**FIN UTIL CATALOG**