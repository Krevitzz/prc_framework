# 📋 PLAN RÉORGANISATION REFACTOR_ML

**Date** : 16 février 2026  
**Objectif** : Découper REFACTOR_ML (7k lignes) en documents modulaires maintenables  
**Principe** : Charter 7.0 = philosophie globale, Refactor = philosophie locale + structure + code

---

## 🎯 STRUCTURE CIBLE

```
REFACTOR_ML/
├── 00_PHILOSOPHIE.md           # Vision globale, pourquoi ces choix
├── 01_ARCHITECTURE.md          # Structure modules + flux données
├── 02_DB_SCHEMA.md             # DB unique colonnes SQL + migrations
├── 03_FEATURING.md             # Hub + layers + projections + stats + events
├── 04_PROFILING.md             # Aggregation + régimes + timelines
├── 05_ANALYSING.md             # ML patterns + concordance
├── 06_PIPELINE.md              # Batch runner (axes YAML) + verdict
├── 07_AXES_COMPOSITION.md      # Discovery + YAML + dry-run
├── 08_UTILS_DATABASE.md        # Toolkit lecture/édition DB (NOUVEAU)
├── 09_MIGRATION.md             # Migration ancien → nouveau
└── ANNEXES/
    ├── CODE_FEATURING.md       # Code complet featuring (~2000 lignes)
    ├── CODE_PROFILING.md       # Code complet profiling (~1000 lignes)
    ├── CODE_ANALYSING.md       # Code complet analysing (~1000 lignes)
    ├── CODE_PIPELINE.md        # Code complet pipeline (~1000 lignes)
    ├── CODE_UTILS_DB.md        # Code complet utils/database.py (NOUVEAU)
    └── TESTS_VALIDATION.md     # Tests + benchmarks (~1000 lignes)
```

**Taille cible** : Documents principaux ~500-1000 lignes, Annexes ~1000-2000 lignes

---

## 📄 CONTENU DOCUMENTS PRINCIPAUX

### 00_PHILOSOPHIE.md (~800 lignes)

**Rôle** : Vision globale refactor, "étape ALGO" du charter

**Contenu** :
1. **Problématique** : Pourquoi refactor ?
   - Legacy tests modules complexes
   - Timeseries stockées (73% DB) jamais utilisées
   - Refactors répétés (R0→R1→R2)
   - DB séparées → concordance difficile

2. **Vision cible** : Base homogène cross-phases
   - R0→R1→R2→...→R_N (potentiellement centaines phases)
   - ROI : 4-5 semaines investissement > années refactors

3. **Principes immuables** :
   - **P1 : Intra-run / Inter-run**
     - Intra = tout ce qui peut (80% calculs, RAM, à la volée)
     - Inter = cross-runs uniquement (20% calculs, DB, contexte)
   
   - **P2 : DB unique partitionnée**
     - Phase colonne (R0, R1, R2, ...)
     - Colonnes SQL (axes + features, charge partielle)
     - Indexes critiques (phase, gamma_id)
   
   - **P3 : Featuring layers**
     - Inspection directe history (rank, dims)
     - Chemin if/elif (pas metadata)
     - History complète RAM (pas snapshots)
   
   - **P4 : Gestion erreurs 4 types**
     - Crash / Erreur logique / Erreur params / Explosion physique
     - Seule dernière = résultat PRC
   
   - **P5 : YAML partout**
     - Zéro hardcodé
     - Axes configurables
     - Discovery (all, liste, random)
     - Dry-run automatique

4. **Mapping responsabilités modules** :
   ```
   Featuring  : Extraction intra-run (history → features)
   Profiling  : Aggregation inter-run (features → profils)
   Analysing  : Patterns ML (features → insights)
   Pipeline   : Orchestration (YAML → runs → DB)
   Utils/DB   : Toolkit lecture/édition DB
   ```

5. **Flux données global** :
   ```
   YAML config → Batch runner
                   ↓
   Generate compositions (axes configurables)
                   ↓
   Dry-run (estimation temps/RAM/DB)
                   ↓
   Confirmation utilisateur (o/n)
                   ↓
   For each composition:
       Kernel → history (RAM)
       Featuring → features (~150)
       DB insert (1 write, colonnes SQL)
                   ↓
   Verdict (post-batch):
       Load DB (charge partielle colonnes)
       Profiling → profils gammas/encodings
       Analysing → clusters/outliers/concordance
       Rapport synthesis + recommendations
   ```

6. **Décisions architecture validées** :
   - DB colonnes SQL (pas JSON) → RAM critique
   - Régimes/Timelines intra-run → minimiser I/O
   - Dynamic events découpage → featuring + profiling
   - Utils/database.py → helpers lecture efficace

---

### 01_ARCHITECTURE.md (~600 lignes)

**Rôle** : Structure modules + dépendances + navigation

**Contenu** :
1. **Structure dossiers finale** (copie Charter 7.0 section 2.3)
2. **Dépendances modules** :
   ```
   core → atomics/
   batch_runner → core, featuring
   verdict → profiling, analysing, utils/database
   profiling/analysing → (rien pipeline, seulement DB)
   featuring.registries → (rien, isolation totale)
   utils/database → (DB uniquement)
   ```

3. **Interdictions strictes** :
   - ❌ Dépendances circulaires
   - ❌ featuring → profiling/analysing
   - ❌ registries → featuring

4. **Navigation docs** : Mapping tâche → module → doc
   ```
   Tâche : Ajouter feature
   → Module : featuring/registries/
   → Doc : 03_FEATURING.md + ANNEXES/CODE_FEATURING.md
   
   Tâche : Modifier seuils régimes
   → Module : configs/thresholds/
   → Doc : 04_PROFILING.md
   
   Tâche : Ajouter axe iteration
   → Module : configs/phases/
   → Doc : 07_AXES_COMPOSITION.md
   ```

5. **Checklist développement** :
   - [ ] Consulter catalogues AVANT coder (duplication)
   - [ ] Imports circulaires : `python -m pycircular prc/`
   - [ ] Params hardcodés : Tout YAML
   - [ ] Tests unitaires passent
   - [ ] DB migration si nouvel axe

---

### 02_DB_SCHEMA.md (~800 lignes)

**Rôle** : DB unique + schema colonnes SQL + migrations

**Contenu** :
1. **Décision architecture** : Colonnes SQL (pas JSON)
   - **Raison** : RAM critique (4 MB vs 600 MB charge partielle)
   - **Trade-off** : Migrations DB vs RAM explosion
   - **Scalabilité** : DB 2-3 To → JSON impossible

2. **Schema complet** :
   ```sql
   CREATE TABLE observations (
       -- Axes
       exec_id TEXT PRIMARY KEY,
       phase TEXT NOT NULL,
       gamma_id TEXT NOT NULL,
       d_encoding_id TEXT NOT NULL,
       modifier_id TEXT,
       seed INTEGER,
       DOF INTEGER,
       config_featuring TEXT,
       timestamp TEXT,
       
       -- Features (~150 colonnes)
       -- Algebra
       frobenius_norm_initial REAL,
       frobenius_norm_final REAL,
       frobenius_norm_mean REAL,
       frobenius_norm_std REAL,
       trace_final REAL,
       determinant_final REAL,
       condition_number_final REAL,
       
       -- Graph
       density_initial REAL,
       density_final REAL,
       density_mean REAL,
       clustering_coeff_mean REAL,
       
       -- Spectral
       eigenvalue_max_final REAL,
       eigenvalue_min_final REAL,
       spectral_gap_final REAL,
       
       -- Pattern
       symmetry_score_final REAL,
       sparsity_final REAL,
       
       -- Statistical
       entropy_final REAL,
       kurtosis_final REAL,
       
       -- Tensor (si rank 3)
       tucker_energy_final REAL,
       cp_rank_final INTEGER,
       
       -- Timeseries (si rank 2)
       autocorr_lag1_final REAL,
       
       -- Spatial (si 2D)
       gradient_magnitude_mean REAL,
       connected_components_final INTEGER,
       
       -- Dynamic events
       deviation_detected INTEGER,
       deviation_time REAL,
       saturation_detected INTEGER,
       saturation_time REAL,
       collapse_detected INTEGER,
       collapse_time REAL,
       instability_detected INTEGER,
       oscillation_detected INTEGER,
       
       -- Régimes
       regime TEXT,
       
       -- Timelines
       timeline_descriptor TEXT
   );
   
   -- Indexes critiques
   CREATE INDEX idx_phase ON observations(phase);
   CREATE INDEX idx_gamma_phase ON observations(gamma_id, phase);
   CREATE INDEX idx_regime ON observations(regime);
   CREATE INDEX idx_frobenius_final ON observations(frobenius_norm_final);
   CREATE INDEX idx_density_mean ON observations(density_mean);
   ```

3. **Volumétrie estimée** :
   ```
   R0 : 7,000 obs × 150 cols × 8 bytes ≈ 8 MB
   R1 : 57,000 obs ≈ 69 MB
   R2 : 200,000 obs ≈ 240 MB
   R3 : 500,000 obs ≈ 600 MB
   Total R0-R3 : ~1 GB
   ```

4. **Scripts migration auto** :
   ```python
   # utils/migrate_db.py
   def add_feature_column(feature_name, feature_type='REAL'):
       """Ajoute colonne feature automatiquement."""
       conn = sqlite3.connect('db_results.db')
       try:
           conn.execute(f"ALTER TABLE observations ADD COLUMN {feature_name} {feature_type}")
           logger.info(f"✓ Colonne '{feature_name}' ajoutée")
       except sqlite3.OperationalError:
           logger.info(f"Colonne '{feature_name}' existe déjà")
       conn.close()
   
   # Batch runner appel automatique
   for feature_name in new_features:
       add_feature_column(feature_name)
   ```

5. **Workflow ajout feature** :
   1. Développer registre fonction (featuring/registries/)
   2. Ajouter config YAML (configs/features/)
   3. Lancer batch_runner → migration auto colonne
   4. DB schema mis à jour automatiquement

6. **Stratégie indexes** :
   - Phase (toujours)
   - Gamma × Phase (queries fréquentes)
   - Features critiques (frobenius, density, regime)
   - Pas tout indexer (overhead insert)

---

### 03_FEATURING.md (~1000 lignes)

**Rôle** : Extraction intra-run (history → features)

**Contenu** :
1. **Principe** : 80% calculs totaux, RAM, à la volée
2. **Architecture hub** : `featuring/hub.py` orchestrateur
3. **Extractor principal** : `extract_features_ml(history, config)`
4. **Layers logique** :
   - universal (tout)
   - matrix_2d (rank 2)
   - matrix_square (rank 2 carrée)
   - tensor_3d (rank ≥3)
   - spatial_2d (analyses spatiales)
5. **Projections temporelles** : initial, final, mean, max, min, max_deviation
6. **Statistics scalaires** : initial, final, mean, std, slope, cv
7. **Dynamic events** : Détection + interprétation (deviation, saturation, collapse, instability, oscillation)
8. **Régimes** : Classification conservation/pathologie (intra-run, seuils YAML)
9. **Timelines** : Descriptors patterns temporels
10. **Registres** : 11 modules (algebra, graph, spectral, pattern, statistical, entropy, tensor, timeseries, spatial, topological)
11. **Pointeurs annexe** : CODE_FEATURING.md pour code complet

---

### 04_PROFILING.md (~600 lignes)

**Rôle** : Aggregation inter-run (features → profils)

**Contenu** :
1. **Principe** : 20% calculs totaux, DB, contexte cross-runs
2. **Architecture hub** : `profiling/hub.py`
3. **Aggregation** : median, IQR, bimodal detection
4. **Régimes distribution** : Fréquences conservation/pathologie/instabilité
5. **Timelines frequency** : Patterns dominants cross-runs
6. **Comparaisons timelines** : Interprétations cross-runs (pas re-calcul)
7. **Profils gammas** : Caractérisation comportement
8. **Profils encodings** : Patterns récurrents
9. **Pointeurs annexe** : CODE_PROFILING.md

---

### 05_ANALYSING.md (~600 lignes)

**Rôle** : Patterns ML (features → insights)

**Contenu** :
1. **Clustering** : HDBSCAN (clusters compositions similaires)
2. **Outliers** : IsolationForest (compositions anormales)
3. **Variance** : η², ANOVA (importance axes)
4. **Concordance cross-phases** : Kappa régimes, DTW timelines, trajectoires R0→R1
5. **Pointeurs annexe** : CODE_ANALYSING.md

---

### 06_PIPELINE.md (~800 lignes)

**Rôle** : Batch runner + verdict

**Contenu** :
1. **Batch runner** :
   - Generate compositions (axes YAML configurable)
   - Discovery (all, liste, random)
   - Dry-run estimation
   - Confirmation utilisateur
   - Loop runs (kernel → featuring → DB)

2. **Verdict** :
   - Load observations (charge partielle colonnes)
   - Profiling
   - Analysing
   - Rapport synthesis
   - Recommendations

3. **Pointeurs annexe** : CODE_PIPELINE.md

---

### 07_AXES_COMPOSITION.md (~800 lignes)

**Rôle** : Discovery + YAML + dry-run

**Contenu** :
1. **Axes configurables** : gamma_id, d_encoding_id, modifier_id, seed, DOF, config_featuring, ...
2. **Discovery** :
   - `all` : Discovery automatique fichiers
   - Liste explicite : `[GAM-001, GAM-002]`
   - `random: N` : Échantillon aléatoire
3. **Generate compositions** : Produit cartésien axes
4. **Dry-run** : Estimation temps/RAM/DB
5. **Validation implicite** : Compositions invalides ignorées gracefully
6. **Exemples YAML** : Production, exploration, développement
7. **Workflow ajout axe** : 4 étapes (YAML, DB migration, kernel, tests)

---

### 08_UTILS_DATABASE.md (~600 lignes) **NOUVEAU**

**Rôle** : Toolkit lecture/édition DB efficace

**Contenu** :
1. **Motivation** : DB 150 colonnes → helpers lecture partielle
2. **Fonctions** :
   - `load_observations(phase, columns=None)` : Charge partielle
   - `filter_by_axes(gamma_id=None, phase=None, ...)` : Filtres rapides
   - `aggregate_features(features, groupby)` : Agrégations
   - `add_feature_column(name, type)` : Migration auto
   - `get_schema()` : Inspection schema
   - `vacuum_db()` : Optimisation
   - `export_csv(query, filepath)` : Export résultats

3. **Exemples usage** :
   ```python
   # Charge seulement frobenius phase R6
   df = load_observations('R6', columns=['frobenius_norm_final', 'regime'])
   
   # Filtre gamma spécifique
   df = filter_by_axes(gamma_id='GAM-001', phase='R1')
   
   # Agrégation médiane par gamma
   medians = aggregate_features(['frobenius_norm_final'], groupby='gamma_id')
   ```

4. **Performances** : Benchmarks charge partielle (4 MB vs 600 MB)
5. **Pointeurs annexe** : CODE_UTILS_DB.md

---

### 09_MIGRATION.md (~600 lignes)

**Rôle** : Migration ancien → nouveau pipeline

**Contenu** :
1. **Workflow migration** : 5 étapes (backup, YAML équivalent, refactor, tests, migration DB)
2. **Validation consistance** : Script compare ancien vs nouveau
3. **Cibles validation** : Features correlation >0.95, Régimes agreement >0.90

---

## 📦 CONTENU ANNEXES

### ANNEXES/CODE_FEATURING.md (~2000 lignes)

**Contenu** : Code complet modules featuring
- hub.py
- extractor.py
- layers.py
- projections.py
- statistics.py
- dynamic_events.py
- registries/ (11 fichiers)

**Format** : Code + docstrings + exemples usage

---

### ANNEXES/CODE_PROFILING.md (~1000 lignes)

**Contenu** : Code complet profiling
- hub.py
- aggregation.py
- regimes.py
- timelines.py

---

### ANNEXES/CODE_ANALYSING.md (~1000 lignes)

**Contenu** : Code complet analysing
- hub.py
- clustering.py
- outliers.py
- variance.py
- concordance.py

---

### ANNEXES/CODE_PIPELINE.md (~1000 lignes)

**Contenu** : Code complet pipeline
- batch_runner.py
- verdict.py
- generate_compositions()
- dry_run_estimation()

---

### ANNEXES/CODE_UTILS_DB.md (~600 lignes) **NOUVEAU**

**Contenu** : Code complet utils/database.py
- load_observations()
- filter_by_axes()
- aggregate_features()
- add_feature_column()
- get_schema()
- vacuum_db()

---

### ANNEXES/TESTS_VALIDATION.md (~1000 lignes)

**Contenu** :
- Tests unitaires (featuring, profiling, analysing)
- Tests intégration (batch_runner, verdict)
- Benchmarks performance
- Validation migration

---

## 🔄 WORKFLOW PRODUCTION

### Étape 1 : Validation plan (UTILISATEUR)

**Questions restantes** :
- Niveau détail annexes : Code complet (~2000 lignes/annexe) ? OK ?
- Ordre production : 00_PHILOSOPHIE → 01_ARCHITECTURE → ... → Annexes ? OK ?

---

### Étape 2 : Production docs principaux

**Ordre** :
1. 00_PHILOSOPHIE.md (vision globale)
2. 01_ARCHITECTURE.md (structure)
3. 02_DB_SCHEMA.md (schema + migrations)
4. 03_FEATURING.md (extraction intra-run)
5. 04_PROFILING.md (aggregation inter-run)
6. 05_ANALYSING.md (ML patterns)
7. 06_PIPELINE.md (batch runner + verdict)
8. 07_AXES_COMPOSITION.md (discovery + YAML)
9. 08_UTILS_DATABASE.md (toolkit DB)
10. 09_MIGRATION.md (ancien → nouveau)

**Validation utilisateur** : Après chaque doc (pas bloc 10 docs d'un coup)

---

### Étape 3 : Production annexes code

**Ordre** :
1. CODE_FEATURING.md (extraction code REFACTOR_ML sections 4)
2. CODE_PROFILING.md (section 5)
3. CODE_ANALYSING.md (section 6)
4. CODE_PIPELINE.md (section 7)
5. CODE_UTILS_DB.md (nouveau, basé sur utils/database.py existant)
6. TESTS_VALIDATION.md (section 8)

**Validation utilisateur** : Après chaque annexe

---

### Étape 4 : Archivage REFACTOR_ML

```bash
$ mkdir -p legacy/refactors/
$ mv REFACTOR_ML_PARTIE_1.md legacy/refactors/REFACTOR_ML_v1_7k_lignes.md
```

---

## 📊 MÉTRIQUES SUCCÈS

| Critère | Avant (REFACTOR_ML) | Après (Réorganisation) |
|---------|---------------------|------------------------|
| **Lisibilité** | 1 fichier 7k lignes | 10 docs ~600 lignes |
| **Navigation** | Ctrl+F linéaire | Docs thématiques |
| **Maintenance** | Modifier 7k lignes | Modifier doc ciblé |
| **Obsolescence** | Tout ou rien | Docs indépendants |
| **Validation** | Bloc 7k lignes | Doc par doc |

---

**FIN PLAN** 

**Attends validation utilisateur avant production !**

# 00 - PHILOSOPHIE GÉNÉRALE REFACTOR ML

**Date** : 16 février 2026  
**Version** : 1.0 (post-audit Charter 7.0)  
**Rôle** : Vision globale refactor - Étape ALGO du Charter

---

## 🎯 OBJECTIF DE CE DOCUMENT

**Ce document EST** :
- La vision globale du refactor (pourquoi, comment conceptuellement)
- L'étape ALGO (langage courant, zéro code) avant structure/implémentation
- Le pont entre Charter 7.0 (philosophie globale PRC) et docs techniques

**Ce document N'EST PAS** :
- Un manuel d'implémentation (voir docs 01-09)
- Un catalogue de code (voir ANNEXES/)
- Une spécification exhaustive (voir docs modules)

**Principe navigation** :
- Lire ce doc EN PREMIER pour comprendre "où on va"
- Consulter docs spécifiques (01-09) selon tâche
- Annexes code si besoin détails implémentation

---

## 📊 SECTION 1 : PROBLÉMATIQUE

### 1.1 Situation actuelle (Legacy)

**Architecture pipeline actuel** :
```
tests/
├── test_gra_001.py      # 10 modules tests complexes
├── test_spe_001.py      # (graph, spectral, algebra, ...)
├── ...
└── utilities/
    ├── registries/      # Fonctions extraction (BIEN)
    ├── HUB/             # test_engine.py orchestrateur
    └── UTIL/            # aggregation, regime, timeline utils

Pipeline :
1. Kernel → history (RAM)
2. Tests (10 modules) → observation_data
   - test_name
   - statistics (scalaires)
   - timeseries (201 points temporels)
3. DB insert → db_results_r1.db
4. Verdict → load DB → profiling → analyses → rapport
```

### 1.2 Problèmes identifiés

**P1 : Tests = modules complexes métier** 
- 10 fichiers tests (graph, spectral, algebra, ...)
- Metadata applicability cauchemar (rank, dims, is_square, ...)
- Logique métier enfouie tests (pas réutilisable)
- Refactor à chaque phase (R0→R1 déjà 3 refactors)

**P2 : Timeseries stockées inutilement**
- DB R1 : 382 Mo total
- Timeseries : 280 Mo (73% DB)
- **Jamais réutilisées** après insertion (sauf peak ponctuel)
- Explosion I/O DB (lecture/écriture 201 points × features)

**P3 : DB séparées par phase**
- db_results_r0.db, db_results_r1.db, db_results_r2.db, ...
- Concordance cross-phases difficile (jointures, comparaisons)
- Duplication schema (migrations répétées)
- Pas de vue globale évolution phases

**P4 : Seuils arbitraires dispersés**
- Régimes : Seuils hardcodés utilities/regime.py
- Aggregation : Seuils hardcodés utilities/aggregation.py
- Timelines : Seuils hardcodés utilities/timeline.py
- Pas de configuration centralisée
- Calibration = modifier code Python (fragile)

**P5 : Refactors répétés**
- R0 → R1 : 3 refactors successifs
- Chaque phase = repenser architecture tests
- Coût temps : 2-3 semaines/refactor
- Accumulation dette technique

### 1.3 Conséquence critique

**Si pattern continue** :
- R2, R3, R4, ... R_N (potentiellement centaines phases)
- Refactor chaque phase = années perdues
- Dette technique exponentielle
- Impossible scaling scientifique

**Problème fondamental** : Architecture pas conçue pour **base homogène cross-phases**

---

## 🎯 SECTION 2 : VISION CIBLE

### 2.1 Principe directeur

**Base homogène cross-phases** :
- DB unique partitionnée (phase colonne)
- Pipeline unique extensible (axes configurables)
- Features standardisées (extraction identique R0-R_N)
- Concordance cross-phases native (même schema)

**Objectif** : Écrire pipeline UNE FOIS, fonctionne R0→R_N sans refactor

### 2.2 ROI refactor

**Investissement** : 4-5 semaines refactor complet
**Gain** : Zéro refactor R2, R3, ..., R_N (années économisées)

**Calcul simple** :
- Refactor actuel : 3 semaines/phase × 10 phases = 30 semaines
- Refactor ML : 5 semaines initiales + 0 semaines/phase × 10 = 5 semaines
- **Économie : 25 semaines** (6 mois)

**Bonus** :
- Concordance cross-phases (insights scientifiques)
- Scalabilité DB (millions observations)
- Maintenabilité pipeline (YAML vs code)

### 2.3 Cas usage futur

**Phase R2** : Compositions binaires gammas prometteurs
- Discovery R1 → sélection 5 gammas candidats
- R2 : Compositions binaires (GAM-001 + GAM-005, etc.)
- ~200,000 runs (vs 57,000 R1)

**Phase R3** : Mini-phases validation suspicions
- Gamma suspect (variance élevée, outliers R1/R2)
- Mini-phase : 100 runs ciblés (variations seed, DOF, noise)
- Si confirmé pathologique → éliminer pool combinatoire

**Phases R4+** : Dimensions supérieures, contraintes topologiques, etc.
- Mêmes features extraction (layers adaptatifs)
- Même DB (phase='R4')
- Même verdict (profiling + analysing)
- **Zéro refactor pipeline Python**

---

## 🏗️ SECTION 3 : PRINCIPES IMMUABLES

### 3.1 Principe 1 : Séparation Intra-run / Inter-run

**Motivation** : Minimiser I/O DB (bottleneck critique)

**INTRA-RUN** (featuring) :
- **Contexte** : History complète RAM (résolution maximale, à la volée)
- **Timing** : Pendant batch runner (kernel → featuring → DB)
- **Calculs** : Tout ce qui peut être calculé sur run unique
  - Features scalaires (projections temporelles, statistics)
  - Dynamic events (détection + interprétation timeline)
  - Régimes (classification conservation/pathologie)
  - Timelines descriptors (patterns temporels)
- **Output** : Features scalaires + interprétations → DB (1 write)
- **Principe** : **80% calculs totaux** (profiter history RAM complète)

**INTER-RUN** (profiling + analysing) :
- **Contexte** : Observations multiples DB (contexte cross-runs nécessaire)
- **Timing** : Post-batch (verdict)
- **Calculs** : Uniquement ce qui nécessite comparaisons cross-runs
  - Profiling : Aggregation (median, IQR), distributions régimes, fréquences timelines
  - Analysing : Clustering, outliers, variance, concordance cross-phases
- **Output** : Profils, insights, rapport
- **Principe** : **20% calculs totaux** (contexte cross-runs obligatoire)

**Règle d'or** : Si calculable **intra-run**, DOIT être calculé intra → minimiser I/O DB

**Exemple concret** :
```
❌ ANCIEN (inter-run) :
Kernel → history → DB (stocke timeseries 201 points)
Verdict → load timeseries → calcule mean/std/slope
→ I/O : Write 201 floats + Read 201 floats = lourd

✅ NOUVEAU (intra-run) :
Kernel → history → featuring calcule mean/std/slope → DB (stocke 3 floats)
Verdict → load 3 floats
→ I/O : Write 3 floats + Read 3 floats = léger (×67 réduction)
```

### 3.2 Principe 2 : DB unique partitionnée #obsolète utilise parquet désormais voir doc 02_parquet-schema

**Motivation** : Concordance cross-phases + scalabilité

**Architecture** :
- **DB unique** : `prc_databases/db_results.db`
- **Table unique** : `observations`
- **Partitionnement** : Colonne `phase` (R0, R1, R2, ...)
- **Schema** : Axes colonnes SQL + Features colonnes SQL
- **Indexes** : Phase, gamma_id × phase, regime

**Décision architecture critique** : **Colonnes SQL** (pas JSON)

**Raison** :
- **RAM** : Charge partielle efficace (4 MB vs 600 MB)
- **Scalabilité** : DB 2-3 To → JSON = explosion thermonucléaire RAM
- **Performance** : Queries 10-100× plus rapides (indexes natifs)

**Trade-off accepté** :
- ❌ Migrations DB si ajout feature (`ALTER TABLE ADD COLUMN`)
- ✅ Scripts migration automatiques (`utils/migrate_db.py`)
- ✅ Scalabilité garantie (millions observations)

**Exemple query efficace** :
```sql
-- Charge seulement 2 colonnes sur 150 (phase R6, 50,000 obs)
SELECT exec_id, frobenius_norm_final, regime 
FROM observations 
WHERE phase = 'R6' AND gamma_id = 'GAM-001';

-- RAM : 50,000 × 2 colonnes × 8 bytes = 800 KB
-- vs JSON : 50,000 × 150 colonnes × 8 bytes = 60 MB (×75 différence)
```

**Volumétrie estimée** :
```
R0 : 7,000 obs × 150 cols × 8 bytes ≈ 8 MB
R1 : 57,000 obs ≈ 69 MB
R2 : 200,000 obs ≈ 240 MB
R3 : 500,000 obs ≈ 600 MB
Total R0-R3 : ~1 GB (vs 1.5 GB ancien avec timeseries)
```

### 3.3 Principe 3 : Featuring layers (inspection directe)

**Motivation** : Éviter metadata applicability cauchemar

**Architecture** : Chemin logique if/elif (pas boucle for dynamique)

**Layers** :
- `universal` : Tout tenseur (frobenius_norm, entropy)
- `matrix_2d` : Rank 2 (density, clustering_coeff)
- `matrix_square` : Rank 2 carrée (trace, eigenvalue_max)
- `tensor_3d` : Rank ≥3 (tucker_energy, cp_rank)
- `spatial_2d` : Analyses spatiales 2D (gradient, connected_components)

**Inspection directe history** :
```python
rank = history.ndim - 1  # Premier dim = temps
dims = history.shape[1:]  # Dimensions spatiales
is_square = (rank == 2) and (dims[0] == dims[1])
is_cubic = (rank == 3) and (dims[0] == dims[1] == dims[2])

# Chemin if/elif prédéfini
if rank == 2:
    extract_matrix_2d_features()
    if is_square:
        extract_matrix_square_features()
elif rank == 3:
    extract_tensor_3d_features()
    if is_cubic:
        extract_matrix_square_features(mode='cubic')  # Slice plan médian
```

**Avantages** :
- ✅ Pas de metadata applicability complexe
- ✅ Chemin clair (suit logique YAML configs)
- ✅ Performance (pas calcul liste dynamique)
- ✅ Extensibilité (ajouter layer = ajouter if)

**Principe history RAM complète** :
- Pas de snapshots (201 points)
- History complète disponible **à la volée** pendant featuring
- Résolution temporelle maximale conservée
- Projections temporelles (initial, final, mean, max, ...) calculées sur history complète

### 3.4 Principe 4 : Gestion erreurs multi-niveaux

**Motivation** : Distinguer erreur pipeline vs explosion physique système

**4 types erreurs** :

| Type | Cause | Action | Exemple |
|------|-------|--------|---------|
| **Crash** | Bug code Python | ❌ Fix code (raise exception) | IndexError, TypeError |
| **Erreur logique** | Implémentation fausse | ❌ Fix code (debug) | Calcul mathématique faux |
| **Erreur params** | YAML sévère | ⚠️ Ajuster YAML (log warning) | Epsilon trop petit, bins=0 |
| **Explosion physique** | Système réel explosif | ✅ Résultat valide (return NaN) | Matrice singulière, collapse |

**Seule la dernière = résultat PRC valide**

Les 3 premières = **erreurs pipeline** à corriger (pas représentation PRC)

**Protection multi-niveaux** :

**Niveau 1 : Registres** (wrappers robustes)
```python
@register_function("condition_number")
def compute_condition(self, state, epsilon=1e-10):
    try:
        cond = np.linalg.cond(state)
        if np.isinf(cond):
            return float(1e10)  # Sentinelle matrice singulière
        return float(cond)
    except np.linalg.LinAlgError:
        return float(1e10)  # Explosion physique → signal
```

**Niveau 2 : Extractor** (validation state)
```python
# Avant calculs features
if not np.all(np.isfinite(state)):
    logger.warning("State contains NaN/Inf (explosion physique)")
    # Continue (pas raise), features retourneront NaN

# Protection variance nulle
if np.var(state) < 1e-15:
    logger.info("State variance nulle (matrice constante)")
    # Continue, certaines features NaN (OK)
```

**Niveau 3 : Verdict** (filtrage features inutiles)
```python
# Filtrer features >50% NaN
for col in observations.columns:
    nan_rate = observations[col].isna().mean()
    if nan_rate > 0.5:
        # Vérifier si params YAML trop sévères (faux positif)
        if is_parameter_too_severe(col, config):
            logger.warning(f"{col} : {nan_rate*100:.1f}% NaN, params sévères. Ajuster YAML.")
        else:
            # Feature réellement inutile (explosion physique systématique)
            useless_features.append(col)
```

**Principe** : NaN ≠ erreur, NaN = information (si explosion physique)

### 3.5 Principe 5 : YAML partout

**Motivation** : Zéro hardcodé, configuration externalisée

**Tout configurable YAML** :
- Seuils régimes (`configs/thresholds/regimes.yaml`)
- Params features (`configs/features/*.yaml`)
- Axes itération (`configs/phases/*.yaml`)
- Discovery (`all`, liste explicite, `random: N`)
- Dry-run automatique (estimation avant lancement)

**Axes configurables** :
```yaml
# configs/phases/default/r0.yaml
iteration_axes:
  gamma_id: all              # Discovery automatique
  d_encoding_id:             # Liste explicite
    - SYM-001
    - ASY-001
  modifier_id: all
  seed: [42, 123]            # Axe temporaire (calibration)
```

**Discovery flexible** :
- `all` : Discovery automatique tous fichiers disponibles
- Liste explicite : Bypass discovery (`[GAM-001, GAM-002]`)
- `random: N` : Échantillon aléatoire N fichiers (runs courts dev)

**Dry-run automatique** :
```bash
$ python batch_runner.py --phase r0

🔍 DRY-RUN ESTIMATION
─────────────────────────────────
Compositions : 7,605 runs
Temps estimé : 31.7h
RAM peak : 850 MB
DB finale : 9.1 GB
─────────────────────────────────

▶ Lancer batch ? (o/n) : n  # Annuler si explosion combinatoire
```

**Validation implicite** :
- Compositions invalides ignorées gracefully (pas crash)
- Kernel/Featuring adaptent params automatiquement
- Pas besoin système contraintes YAML complexe

**Exemple ajout axe** :
```yaml
# Nouvel axe : noise_level (expérimental)
iteration_axes:
  gamma_id: all
  noise_level: [0.01, 0.05, 0.1]  # ← Ajout 1 ligne YAML

# Pipeline Python : Zéro modification
# DB : Migration auto colonne noise_level
# Kernel : Extrait composition['noise_level']
```

---

## 🗺️ SECTION 4 : ARCHITECTURE GLOBALE

### 4.1 Mapping responsabilités modules

**Featuring** (`prc/featuring/`) :
- **Rôle** : Extraction intra-run (history → features)
- **Timing** : À la volée, RAM, pendant batch_runner
- **Responsabilité** : Calculer ~150 features scalaires depuis history
- **Output** : Dict features → DB (1 write)
- **Principe** : 80% calculs totaux (profiter history RAM)

**Profiling** (`prc/profiling/`) :
- **Rôle** : Aggregation inter-run (features → profils)
- **Timing** : Post-batch, DB, verdict
- **Responsabilité** : Aggregation (median, IQR), distributions régimes, fréquences timelines
- **Output** : Profils gammas/encodings/modifiers
- **Principe** : Contexte cross-runs nécessaire

**Analysing** (`prc/analysing/`) :
- **Rôle** : Patterns ML (features → insights)
- **Timing** : Post-batch, DB, verdict
- **Responsabilité** : Clustering, outliers, variance, concordance cross-phases
- **Output** : Clusters, outliers, η², trajectoires R0→R1
- **Principe** : 20% calculs totaux (contexte cross-runs)

**Pipeline** (`batch_runner.py`, `verdict.py`) :
- **Rôle** : Orchestration (YAML → runs → DB → rapport)
- **Timing** : Point entrée exécution
- **Responsabilité** : Generate compositions, dry-run, loop runs, verdict
- **Output** : DB remplie + rapport synthesis

**Utils/Database** (`prc/utils/database.py`) :
- **Rôle** : Toolkit lecture/édition DB efficace
- **Timing** : Utilisé par verdict (charge partielle)
- **Responsabilité** : Helpers load_observations(), filter_by_axes(), aggregate_features()
- **Output** : DataFrames pandas optimisés
- **Principe** : Charge partielle critique (4 MB vs 600 MB)

### 4.2 Flux données global

```
┌─────────────────────────────────────────────────────────────┐
│                    YAML CONFIG                              │
│  configs/phases/default/r0.yaml                             │
│  - iteration_axes (gamma, encoding, modifier, seed, ...)    │
│  - discovery (all, liste, random)                           │
│  - features (layers, projections, stats)                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                 BATCH RUNNER                                │
│  1. Generate compositions (produit cartésien axes)          │
│  2. Dry-run estimation (temps/RAM/DB)                       │
│  3. Confirmation utilisateur (o/n)                          │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│            FOR EACH COMPOSITION                             │
│                                                             │
│  ┌────────────────┐                                        │
│  │  KERNEL        │ composition dict                       │
│  │  core/kernel.py│ → history (T, n, m) RAM                │
│  └────────────────┘                                        │
│         ↓                                                   │
│  ┌────────────────┐                                        │
│  │  FEATURING     │ history complète RAM                   │
│  │  featuring/hub │ → features dict (~150 scalaires)       │
│  └────────────────┘                                        │
│         ↓                                                   │
│  ┌────────────────┐                                        │
│  │  DB INSERT     │ features + axes                        │
│  │  1 write       │ → observations table                   │
│  └────────────────┘                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                DB UNIQUE (partitionnée phase)               │
│  observations table                                         │
│  - Axes (colonnes) : exec_id, phase, gamma_id, ...         │
│  - Features (colonnes) : frobenius_norm_final, ...         │
│  - Indexes : phase, gamma_id × phase, regime               │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    VERDICT (post-batch)                     │
│                                                             │
│  ┌────────────────┐                                        │
│  │ LOAD DB        │ Charge partielle colonnes              │
│  │ utils/database │ (ex: frobenius_final, regime)          │
│  └────────────────┘                                        │
│         ↓                                                   │
│  ┌────────────────┐                                        │
│  │ PROFILING      │ Aggregation cross-runs                 │
│  │ profiling/hub  │ → profils gammas/encodings             │
│  └────────────────┘                                        │
│         ↓                                                   │
│  ┌────────────────┐                                        │
│  │ ANALYSING      │ Patterns ML                            │
│  │ analysing/hub  │ → clusters, outliers, concordance      │
│  └────────────────┘                                        │
│         ↓                                                   │
│  ┌────────────────┐                                        │
│  │ RAPPORT        │ Synthesis + recommendations            │
│  │ reports/       │ → synthesis_decision_ML.md             │
│  └────────────────┘                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Dépendances modules

**Autorisées** :
```
core → operators, encodings, modifiers
batch_runner → core, featuring
verdict → profiling, analysing, utils/database
profiling → utils/database (charge DB)
analysing → utils/database (charge DB)
```

**Interdictions strictes** :
```
❌ Dépendances circulaires (toujours)
❌ featuring → profiling/analysing (isolation intra-run)
❌ registries → featuring (isolation totale registres)
❌ profiling → featuring (séparation intra/inter)
```

### 4.4 Découpage logique featuring vs profiling

**Cas d'usage : Dynamic events**

**Problème** : Ancien pipeline stockait timeseries (201 points) pour analyser événements

**Nouvelle approche** : Découpage logique

**Featuring (intra-run)** :
```python
# featuring/dynamic_events.py
def detect_events(history):
    """Détecte événements timeline depuis history RAM complète."""
    events = {
        'deviation_detected': False,
        'deviation_time': None,
        'saturation_detected': False,
        'saturation_time': None,
        # ...
    }
    
    # Analyse history complète (résolution max)
    norms = np.linalg.norm(history, axis=(1,2))
    
    # Détection deviation
    if np.max(norms) > threshold:
        events['deviation_detected'] = True
        events['deviation_time'] = find_deviation_time(norms)
    
    # Détection saturation
    if is_saturated(norms):
        events['saturation_detected'] = True
        events['saturation_time'] = find_saturation_time(norms)
    
    return events  # → DB (booleans + floats)
```

**Profiling (inter-run)** :
```python
# profiling/timelines.py
def aggregate_timelines(observations):
    """Agrège fréquences événements cross-runs."""
    timelines_freq = {
        'early_deviation': 0,
        'late_saturation': 0,
        'deviation_then_collapse': 0,
        # ...
    }
    
    # Analyse patterns événements stockés DB
    for obs in observations:
        if obs['deviation_detected'] and obs['deviation_time'] < 0.2:
            timelines_freq['early_deviation'] += 1
        
        if obs['deviation_detected'] and obs['collapse_detected']:
            timelines_freq['deviation_then_collapse'] += 1
    
    return timelines_freq
```

**Pas doublon** : Découpage logique
- Featuring : Détection + interprétation (history → events)
- Profiling : Comparaisons fréquences (events → stats)

---

## ✅ SECTION 5 : DÉCISIONS ARCHITECTURE VALIDÉES

### 5.1 DB colonnes SQL (pas JSON)

**Décision** : Features stockées colonnes SQL (~150 colonnes)

**Raison** : RAM critique
- Charge partielle efficace (4 MB vs 600 MB)
- Scalabilité DB 2-3 To garantie
- Performance queries 10-100× plus rapides

**Trade-off** : Migrations DB si ajout feature
- Scripts migration automatiques (`utils/migrate_db.py`)
- Appel auto batch_runner (détection nouvelles features)

**Rejet JSON** : Charge complète blob features
- DB 1 GB → verdict charge 600 MB RAM (explosion)
- Incompatible scaling futur (millions observations)

### 5.2 Régimes/Timelines intra-run (featuring)

**Décision** : Calculés featuring (pas profiling)

**Raison** : Minimiser I/O DB (principe P1)
- Régimes : Classification conservation/pathologie depuis features scalaires
- Timelines : Interprétation événements depuis history RAM
- Seuils absolus YAML (pas adaptatifs, calibration manuelle)

**Output DB** : Colonnes `regime`, `timeline_descriptor`

**Profiling** : Comparaisons régimes/timelines cross-runs (pas re-calcul)

### 5.3 Dynamic events découpage logique

**Décision** : `featuring/dynamic_events.py` + `profiling/timelines.py`

**Raison** : Pas doublon, découpage responsabilités
- Featuring : Détection + interprétation (history → events dict)
- Profiling : Agrégation fréquences (events → stats cross-runs)

### 5.4 History complète RAM (pas snapshots)

**Décision** : History complète disponible featuring

**Raison** : Résolution temporelle maximale
- Pas de réduction snapshots (201 → N)
- Projections temporelles calculées résolution max
- À la volée (pas stockage)

### 5.5 Utils/database.py toolkit

**Décision** : Module helpers lecture DB efficace

**Raison** : DB 150 colonnes → charge partielle critique
- `load_observations(phase, columns)` : Charge seulement colonnes nécessaires
- `filter_by_axes()` : Filtres rapides gamma/phase/regime
- `aggregate_features()` : Agrégations optimisées

---

## 🎯 SECTION 6 : CAS USAGE CONCRETS

### 6.1 Workflow développement feature

**Étape 1 : Développer registre fonction**
```python
# featuring/registries/algebra_registry.py
@register_function("new_metric")
def compute_new_metric(self, state, param=1.0):
    """Nouvelle métrique algébrique."""
    # Calcul...
    return float(result)
```

**Étape 2 : Ajouter config YAML**
```yaml
# configs/features/algebra.yaml
algebra_features:
  - function: "algebra.new_metric"
    param: 1.5
    projections: [final, mean]
```

**Étape 3 : Lancer batch_runner**
```bash
$ python batch_runner.py --phase r0_dev
# Migration auto : ADD COLUMN new_metric_final, new_metric_mean
# Runs exécutés
# DB mise à jour automatiquement
```

**Résultat** : Feature disponible verdict (pas modification manuelle schema DB)

### 6.2 Workflow calibration seuils

**Objectif** : Calibrer epsilon régime conservation

**Étape 1 : Axe temporaire**
```yaml
# configs/phases/exploration/calibration.yaml
iteration_axes:
  gamma_id: [GAM-001]  # 1 gamma test
  epsilon: [1e-3, 1e-4, 1e-5, 1e-6]  # Axe temporaire
```

**Étape 2 : Runs**
```bash
$ python batch_runner.py --phase calibration
# 4 runs (1 gamma × 4 epsilon)
# DB : Colonne epsilon ajoutée auto
```

**Étape 3 : Analyse verdict**
```python
# Comparer régimes selon epsilon
df = load_observations('calibration', columns=['epsilon', 'regime'])
df.groupby('epsilon')['regime'].value_counts()

# Identifier epsilon optimal (balance conservation/pathologie)
epsilon_optimal = 1e-4
```

**Étape 4 : Fixer YAML production**
```yaml
# configs/thresholds/regimes.yaml
conservation:
  epsilon: 1e-4  # ← Valeur calibrée
```

**Étape 5 : Supprimer axe temporaire**
```yaml
# configs/phases/default/r0.yaml
iteration_axes:
  gamma_id: all
  # epsilon supprimé (calibration terminée)
```

**Résultat** : Axe epsilon reste DB (colonnes seed=42 runs passés), mais pas utilisé futures phases

### 6.3 Workflow concordance cross-phases

**Objectif** : Comparer régimes R0 vs R1

**Étape 1 : Charger observations**
```python
# utils/database.py
df_r0 = load_observations('R0', columns=['gamma_id', 'regime'])
df_r1 = load_observations('R1', columns=['gamma_id', 'regime'])
```

**Étape 2 : Concordance analysing**
```python
# analysing/concordance.py
kappa = cohen_kappa(df_r0['regime'], df_r1['regime'])
# kappa > 0.8 : Concordance forte (régimes stables)
# kappa < 0.5 : Concordance faible (régimes instables)
```

**Étape 3 : Trajectoires**
```python
# Identifier transitions R0→R1
transitions = pd.crosstab(df_r0['regime'], df_r1['regime'])
# Ex: CONSERVES_NORM (R0) → PATHOLOGICAL (R1) : Gamma suspect
```

**Résultat** : Insights scientifiques (stabilité régimes cross-phases)

---

## 🔬 SECTION 6BIS : MÉTHODOLOGIE EXPLORATION (RÉDUCTION AVANT EXPLORATION)

### 6bis.1 Principe anti-optimisation

**❌ Anti-pattern : Chercher "ce qui marche"**
```
R0 : Discovery 12 gammas → Identifier 5 "meilleurs"
R1 : Compositions sur 5 stables
R2 : Ternaires sur stables
  → Explosion combinatoire : 12×12×12 = 1,728 trios
  → Blindspots : Suspects (GAM-009, 010) jamais validés
```

**✅ Pattern correct : Éliminer "ce qui NE marche PAS"**
```
R0 : Discovery 12 gammas → Identifier 2 SUSPECTS
R1 : Mini-phases validation suspects (100-200 runs)
  → ÉLIMINER 2-4 gammas confirmés incompatibles
  → BASE RÉDUITE : 8-10 gammas
R2 : Compositions sur base propre
  → 10×10 = 100 paires (vs 144 = -30%)
```

**Économie** : -30,000 runs cumulés R2-R3

### 6bis.2 Workflow phases alternées

**Pattern** : Réduction ↔ Exploration
```
R0 : DISCOVERY (12 gammas atomiques)
R1 : RÉDUCTION (mini-phases validation suspects) → 8-10 gammas
R2 : EXPLORATION (compositions base réduite)
R3 : RÉDUCTION (validation paires suspectes) → 50 paires
R4 : EXPLORATION (ternaires base propre)
```

### 6bis.3 Mini-phases validation

**Objectif** : Confirmer incompatibilité AVANT éliminer

**Exemple YAML** :
```yaml
# configs/phases/validation/gam009.yaml
iteration_axes:
  gamma_id: [GAM-009]  # Suspect unique
  d_encoding_id: [SYM-001, SYM-003, ASY-001]  # Tolérants
  seed: [42, 123, 456, 789, 1011]
  config_threshold: [relaxed, default]

# Total : 1 × 3 × 5 × 2 = 30 runs
```

**Verdicts possibles** :
- **100% OSCILLATORY** → ÉLIMINER définitivement
- **Config-dependent** → RECONFIGURER (pas éliminer)
- **Stochastique** → AUGMENTER N seeds
- **Encoding-dependent** → RESTREINDRE combinaisons

**IMPORTANT** : Pipeline DÉTECTE + ANALYSE + RECOMMANDE  
Utilisateur VALIDE TOUJOURS (jamais auto-élimination)

### 6bis.4 Cas usage R0→R1

**R0 résultats** :
- 5 gammas STABLES (GAM-001, 002, 006, 008, 012)
- 2 gammas UNSTABLE (GAM-009, 010 = 100% OSCILLATORY)
- 4 gammas MIXED (GAM-003, 004, 005, 007)

**❌ Décision incorrecte** :
```
R1 : Compositions binaires 5 stables
  → Focus prématuré, blindspots suspects
```

**✅ Décision correcte** :
```
R1 : Mini-phases validation
  ├─ GAM-009 (30 runs) → 100% OSCILLATORY → ÉLIMINER
  ├─ GAM-010 (30 runs) → 100% OSCILLATORY → ÉLIMINER
  ├─ GAM-004 (30 runs) → 80% TRIVIAL → ÉLIMINER
  └─ GAM-005 (30 runs) → 80% TRIVIAL → ÉLIMINER
  
R2 : Compositions 8 gammas (vs 12)
  → 8×8 = 64 paires (vs 144 = -56%)
```

**Référence complète** : Lire `10_METHODOLOGIE_EXPLORATION.md`

---

## 📊 SECTION 7 : MÉTRIQUES SUCCÈS REFACTOR

### 7.1 Performances cibles

| Métrique | Ancien (Legacy) | Cible (Refactor ML) | Gain |
|----------|-----------------|---------------------|------|
| **I/O DB write** | 201 floats/feature | 1-3 floats/feature | ×67-200 réduction |
| **I/O DB read** | Timeseries complètes | Colonnes sélectives | ×150 réduction |
| **RAM verdict** | 2 GB (1 float) | 4 MB (charge partielle) | ×500 réduction |
| **Refactor phases** | 3 semaines/phase | 0 semaines/phase | Infini |
| **Migrations DB** | Manuelle (2h) | Auto (30s) | ×240 réduction |
| **Ajout feature** | Modifier code | 1 ligne YAML | Simplicité |
| **Calibration seuils** | Modifier code | Axe temporaire | Propre |

### 7.2 Critères validation

**Performances** :
- [ ] Batch runner R1 : <30h (vs 40h ancien)
- [ ] Verdict R1 : <10 min (vs 1h ancien)
- [ ] RAM verdict : <500 MB (vs 2 GB ancien)

**Consistance** :
- [ ] Features correlation ancien vs nouveau : >0.95
- [ ] Régimes agreement : >0.90

**Maintenabilité** :
- [ ] Ajout feature : <5 min (vs 2h ancien)
- [ ] Migration DB : Automatique (vs manuelle)
- [ ] Refactor phase R2 : 0h (vs 3 semaines)

---

## 🗺️ SECTION 8 : NAVIGATION DOCUMENTATION

### 8.1 Documents principaux (lecture séquentielle recommandée)

```
00_PHILOSOPHIE.md        ← Vous êtes ici (vision globale)
01_ARCHITECTURE.md       → Structure modules + dépendances
02_DB_SCHEMA.md          → Schema SQL + migrations
03_FEATURING.md          → Extraction intra-run
04_PROFILING.md          → Aggregation inter-run
05_ANALYSING.md          → Patterns ML
06_PIPELINE.md           → Batch runner + verdict
07_AXES_COMPOSITION.md   → Discovery + YAML
08_UTILS_DATABASE.md     → Toolkit lecture DB
09_MIGRATION.md          → Ancien → nouveau pipeline
```

### 8.2 Annexes code (consultation si besoin détails)

```
ANNEXES/CODE_FEATURING.md    → Code complet featuring
ANNEXES/CODE_PROFILING.md    → Code complet profiling
ANNEXES/CODE_ANALYSING.md    → Code complet analysing
ANNEXES/CODE_PIPELINE.md     → Code complet pipeline
ANNEXES/CODE_UTILS_DB.md     → Code complet utils/database
ANNEXES/TESTS_VALIDATION.md  → Tests + benchmarks
```

### 8.3 Mapping tâche → documentation

**Ajouter feature extraction** :
- Lire : 03_FEATURING.md (principe layers)
- Consulter : ANNEXES/CODE_FEATURING.md (registres existants)
- Référence : Charter 7.0 section 4.3 (layers)

**Modifier seuils régimes** :
- Lire : 04_PROFILING.md (régimes distribution)
- Éditer : `configs/thresholds/regimes.yaml`
- Calibrer : Axe temporaire (voir 07_AXES_COMPOSITION.md)

**Analyser concordance R0↔R1** :
- Lire : 05_ANALYSING.md (concordance)
- Consulter : ANNEXES/CODE_ANALYSING.md (code concordance)
- Utiliser : `utils/database.py` load_observations()

**Ajouter axe itération** :
- Lire : 07_AXES_COMPOSITION.md (workflow ajout axe)
- Éditer : `configs/phases/*/phase.yaml`
- Vérifier : Migration DB auto

**Optimiser queries DB** :
- Lire : 08_UTILS_DATABASE.md (toolkit)
- Consulter : ANNEXES/CODE_UTILS_DB.md (helpers)
- Référence : 02_DB_SCHEMA.md (indexes)

---

## ✅ SECTION 9 : CHECKLIST PRÉ-IMPLÉMENTATION

Avant commencer implémentation modules, valider :

**Architecture** :
- [ ] Principes P1-P5 compris
- [ ] Flux données global clair
- [ ] Mapping responsabilités modules validé

**DB Schema** :
- [ ] Colonnes SQL accepté (vs JSON)
- [ ] Migration auto scripts compris
- [ ] Volumétrie estimée réaliste

**Featuring** :
- [ ] Layers logique compris (inspection directe)
- [ ] History RAM complète principe validé
- [ ] Régimes/Timelines intra-run accepté

**Profiling/Analysing** :
- [ ] Séparation intra/inter claire
- [ ] Contexte cross-runs nécessité comprise
- [ ] Pas de re-calcul features (DB charge)

**Pipeline** :
- [ ] Axes YAML configurables principe validé
- [ ] Discovery flexible (all, liste, random)
- [ ] Dry-run automatique compris

**Utils/Database** :
- [ ] Charge partielle critique comprise
- [ ] Helpers lecture DB nécessité validée
- [ ] Performances RAM cibles claires

---

## 📝 CONCLUSION

**Ce refactor résout** :
- ✅ Timeseries stockées inutilement (73% DB)
- ✅ DB séparées par phase (concordance difficile)
- ✅ Tests modules complexes (metadata cauchemar)
- ✅ Seuils hardcodés dispersés (calibration fragile)
- ✅ Refactors répétés (3 semaines/phase)

**Ce refactor apporte** :
- ✅ Base homogène cross-phases (R0→R_N)
- ✅ Scalabilité DB (millions observations)
- ✅ Maintenabilité pipeline (YAML vs code)
- ✅ Performances RAM (×500 réduction verdict)
- ✅ ROI positif (5 semaines vs 30 semaines)

**Investissement** : 4-5 semaines refactor complet  
**Gain** : Années économisées R2→R_N (zéro refactor)

**Prochaine étape** : Lire `01_ARCHITECTURE.md` (structure modules détaillée)

---

**FIN 00_PHILOSOPHIE.md**

# 01 - ARCHITECTURE MODULES

**Date** : 16 février 2026  
**Version** : 1.0  
**Rôle** : Structure modules + dépendances + navigation

---

## 🎯 OBJECTIF DE CE DOCUMENT

**Ce document définit** :
- Structure dossiers finale (arborescence complète)
- Dépendances entre modules (autorisées/interdites)
- Navigation documentation (tâche → module → doc)
- Checklist développement (avant coder)

**Prérequis** : Avoir lu `00_PHILOSOPHIE.md` (principes immuables)

---

## 📁 SECTION 1 : STRUCTURE DOSSIERS FINALE

### 1.1 Arborescence complète

```
prc/
├── core/                           # Exécution aveugle (immuable)
│   ├── __init__.py
│   ├── kernel.py                   # run_kernel(composition, config)
│   ├── state_preparation.py        # prepare_state(base, modifiers)
│   └── core_catalog.md             # Documentation core
│
├── atomics/                        # Pool candidats γ, D, modifiers
│   ├── operators/                  # Gammas (GAM-001 à GAM-013)
│   │   ├── gamma_hyp_*.py
│   │   └── gamma_catalog.md
│   ├── D_encodings/                # Encodings D
│   │   ├── rank2_symmetric.py      # SYM-001 à SYM-006
│   │   ├── rank2_asymmetric.py     # ASY-001 à ASY-004
│   │   ├── rank3_correlations.py   # R3-001 à R3-003
│   │   └── d_encoding_catalog.md
│   └── modifiers/                  # Transformations D
│       ├── noise.py                # M1, M2
│       ├── constraints/            # D^(topo)
│       ├── plugins/                # D^(plugin)
│       ├── domains/                # D^(domaine)
│       └── modifier_catalog.md
│
├── featuring/                      # Extraction intra-run
│   ├── __init__.py
│   ├── hub.py                      # Orchestrateur featuring
│   ├── extractor.py                # extract_features_ml() principal
│   ├── layers.py                   # Logique layers (universal, matrix, tensor)
│   ├── projections.py              # Projections temporelles (initial, final, mean)
│   ├── statistics.py               # Statistics scalaires (mean, std, slope)
│   ├── dynamic_events.py           # Détection événements (deviation, saturation)
│   ├── featuring_catalog.md        # Documentation featuring
│   └── registries/                 # Pool fonctions extraction
│       ├── __init__.py
│       ├── base_registry.py        # Classe BaseRegistry
│       ├── registry_manager.py     # Singleton gestionnaire
│       ├── post_processors.py      # Helpers post-processing
│       ├── PATTERNS.md             # Patterns registres
│       ├── algebra_registry.py     # Normes, trace, déterminant
│       ├── graph_registry.py       # Densité, clustering, centralité
│       ├── spectral_registry.py    # Eigenvalues, FFT, spectral_gap
│       ├── spatial_registry.py     # Gradient, laplacien, correlation spatiale
│       ├── topological_registry.py # Connected components, Betti numbers
│       ├── pattern_registry.py     # Symétrie, sparsity, block structure
│       ├── statistical_registry.py # Moments, kurtosis, skewness
│       ├── entropy_registry.py     # Shannon, Rényi, von Neumann
│       ├── tensor_registry.py      # Tucker, CP decomposition, slices
│       ├── timeseries_registry.py  # Autocorr, FFT temporel, changepoints
│       └── registries_catalog.md   # Catalogue complet registres
│
├── profiling/                      # Aggregation inter-run
│   ├── __init__.py
│   ├── hub.py                      # Orchestrateur profiling
│   ├── aggregation.py              # Median, IQR, bimodal detection
│   ├── regimes.py                  # Distributions régimes (conservation, pathologie)
│   ├── timelines.py                # Fréquences événements cross-runs
│   └── profiling_catalog.md        # Documentation profiling
│
├── analysing/                      # Patterns ML verdict
│   ├── __init__.py
│   ├── hub.py                      # Orchestrateur analysing
│   ├── clustering.py               # HDBSCAN (clusters compositions)
│   ├── outliers.py                 # IsolationForest (compositions anormales)
│   ├── variance.py                 # η², ANOVA (importance axes)
│   ├── concordance.py              # Kappa régimes, DTW timelines, trajectoires R0→R1
│   └── analysing_catalog.md        # Documentation analysing
│
├── configs/                        # YAML centralisés
│   ├── phases/                     # Configs phases (axes itération)
│   │   ├── default/
│   │   │   ├── r0.yaml
│   │   │   ├── r1.yaml
│   │   │   └── r2.yaml
│   │   ├── laxe/                   # Seuils laxes (exploration)
│   │   │   └── r0.yaml
│   │   └── strict/                 # Seuils stricts (validation)
│   │       └── r0.yaml
│   ├── features/                   # Configs features (layers, params)
│   │   ├── default/
│   │   │   ├── layers.yaml         # Activation layers
│   │   │   ├── algebra.yaml        # Params algebra features
│   │   │   ├── graph.yaml
│   │   │   ├── spectral.yaml
│   │   │   ├── pattern.yaml
│   │   │   ├── statistical.yaml
│   │   │   ├── entropy.yaml
│   │   │   ├── tensor.yaml
│   │   │   ├── timeseries.yaml
│   │   │   └── spatial.yaml
│   │   ├── laxe/
│   │   └── strict/
│   ├── thresholds/                 # Seuils régimes, events, aggregation
│   │   ├── default/
│   │   │   ├── regimes.yaml        # Seuils conservation, pathologie
│   │   │   └── aggregation.yaml    # Seuils bimodal, outliers
│   │   ├── laxe/
│   │   └── strict/
│   └── verdict/                    # Configs verdict (profiling, analysing)
│       ├── default/
│       │   └── default.yaml
│       ├── laxe/
│       └── strict/
│
├── utils/                          # Utilitaires généraux transverses
│   ├── __init__.py
│   ├── database.py                 # Toolkit lecture/édition DB (NOUVEAU)
│   └── (autres helpers généraux)
│
├── prc_databases/                  # Bases de données
│   ├── db_results.db               # DB unique (partitionnée par phase)
│   └── (backups, archives)
│
├── tests/                          # Tests validation pipeline
│   ├── featuring/
│   │   ├── test_registries.py
│   │   ├── test_extractor.py
│   │   ├── test_projections.py
│   │   └── test_statistics.py
│   ├── profiling/
│   │   ├── test_aggregation.py
│   │   ├── test_regimes.py
│   │   └── test_timelines.py
│   ├── analysing/
│   │   ├── test_clustering.py
│   │   ├── test_outliers.py
│   │   ├── test_variance.py
│   │   └── test_concordance.py
│   ├── integration/
│   │   ├── test_batch_runner.py
│   │   └── test_verdict.py
│   └── benchmarks/
│       ├── bench_featuring.py
│       └── bench_verdict.py
│
├── reports/                        # Rapports (réécrits, pas accumulés)
│   ├── synthesis_decision_ML.md    # Rapport synthesis verdict
│   ├── analysis_complete_ML.json   # Données brutes analyses
│   └── visualizations/             # Graphiques
│
├── legacy/                         # Backup ancien pipeline
│   └── (archives refactors précédents)
│
├── batch_runner.py                 # Point entrée exécution
├── verdict.py                      # Point entrée verdict
├── charter_7_0.md                  # Hub cognitif (bibliothèque permanente)
└── README.md
```

### 1.2 Modules clés responsabilités

| Module | Rôle | Input | Output | Timing |
|--------|------|-------|--------|--------|
| **core/** | Exécution aveugle | composition dict | history (T, n, m) | Batch runner |
| **featuring/** | Extraction intra-run | history | features dict (~150) | Batch runner |
| **profiling/** | Aggregation inter-run | observations DB | profils gammas/encodings | Verdict |
| **analysing/** | Patterns ML | observations DB | clusters/outliers/concordance | Verdict |
| **utils/database** | Helpers lecture DB | queries | DataFrames optimisés | Verdict |
| **batch_runner** | Orchestration runs | YAML config | DB remplie | Point entrée |
| **verdict** | Orchestration verdict | DB | Rapport synthesis | Point entrée |

---

## 🔗 SECTION 2 : DÉPENDANCES MODULES

### 2.1 Graphe dépendances autorisées

```
batch_runner.py
    │
    ├──> core/kernel.py
    │       └──> atomics/operators/
    │       └──> atomics/D_encodings/
    │       └──> atomics/modifiers/
    │
    └──> featuring/hub.py
            └──> featuring/extractor.py
                    └──> featuring/layers.py
                    └──> featuring/projections.py
                    └──> featuring/statistics.py
                    └──> featuring/dynamic_events.py
                    └──> featuring/registries/*

verdict.py
    │
    ├──> utils/database.py
    │
    ├──> profiling/hub.py
    │       └──> profiling/aggregation.py
    │       └──> profiling/regimes.py
    │       └──> profiling/timelines.py
    │       └──> utils/database.py
    │
    └──> analysing/hub.py
            └──> analysing/clustering.py
            └──> analysing/outliers.py
            └──> analysing/variance.py
            └──> analysing/concordance.py
            └──> utils/database.py
```

### 2.2 Règles dépendances

**✅ AUTORISÉES** :
```python
# Core → atomics
from prc.atomics.operators.gamma_hyp_001 import gamma_function

# Batch runner → core, featuring
from prc.core.kernel import run_kernel
from prc.featuring.hub import extract_features_ml

# Verdict → profiling, analysing, utils
from prc.profiling.hub import run_profiling
from prc.analysing.hub import run_analysing
from prc.utils.database import load_observations

# Profiling/Analysing → utils/database
from prc.utils.database import filter_by_axes

# Registries → stdlib uniquement
import numpy as np
import scipy.stats
```

**❌ INTERDITES (crash si détectées)** :
```python
# ❌ Featuring → profiling/analysing (séparation intra/inter)
from prc.profiling.aggregation import compute_median  # INTERDIT

# ❌ Registries → featuring (isolation totale)
from prc.featuring.extractor import extract_features  # INTERDIT

# ❌ Profiling → featuring (séparation intra/inter)
from prc.featuring.projections import project_final  # INTERDIT

# ❌ Core → featuring (core aveugle)
from prc.featuring.hub import extract_features_ml  # INTERDIT

# ❌ Dépendances circulaires (toujours)
from prc.A import foo
from prc.B import bar  # où B importe A
```

### 2.3 Vérification imports circulaires

**Commande** :
```bash
$ python -m pycircular prc/

# ✅ Si OK :
No circular imports detected.

# ❌ Si erreur :
Circular import detected:
  prc.featuring.extractor -> prc.profiling.aggregation -> prc.featuring.projections
```

**Workflow** : Vérifier AVANT chaque commit

---

## 🗺️ SECTION 3 : NAVIGATION DOCUMENTATION

### 3.1 Mapping tâche → module → documentation

**Tâche : Ajouter feature extraction**
```
Module impliqué : featuring/registries/
Documentation :
  1. 03_FEATURING.md (principe layers, registres)
  2. ANNEXES/CODE_FEATURING.md (code registres existants)
  3. featuring/registries/PATTERNS.md (patterns implémentation)
Référence Charter : Section 4.3 (layers inspection directe)

Workflow :
  1. Consulter registries_catalog.md (vérifier duplication)
  2. Lire PATTERNS.md (patterns registres)
  3. Développer fonction registre
  4. Ajouter config YAML (configs/features/*.yaml)
  5. Tests unitaires (tests/featuring/test_registries.py)
```

**Tâche : Modifier seuils régimes**
```
Module impliqué : configs/thresholds/regimes.yaml
Documentation :
  1. 04_PROFILING.md (régimes distribution)
  2. profiling/profiling_catalog.md (seuils disponibles)
Référence Charter : Section 4.4 (YAML partout)

Workflow :
  1. Identifier régime à modifier (CONSERVES_NORM, PATHOLOGICAL, ...)
  2. Éditer configs/thresholds/regimes.yaml
  3. Optionnel : Calibration via axe temporaire (voir 07_AXES_COMPOSITION.md)
  4. Relancer verdict (pas batch runner)
```

**Tâche : Analyser concordance R0↔R1**
```
Module impliqué : analysing/concordance.py
Documentation :
  1. 05_ANALYSING.md (concordance cross-phases)
  2. ANNEXES/CODE_ANALYSING.md (code concordance)
  3. 08_UTILS_DATABASE.md (charge observations multi-phases)
Référence Charter : Section 4.6 (DB unique partitionnée)

Workflow :
  1. Charger observations R0 et R1 (utils/database.py)
  2. Calculer kappa régimes (analysing/concordance.py)
  3. Analyser trajectoires (transitions R0→R1)
  4. Rapport synthesis (verdict.py)
```

**Tâche : Ajouter axe itération**
```
Module impliqué : configs/phases/*.yaml, utils/database.py
Documentation :
  1. 07_AXES_COMPOSITION.md (workflow ajout axe)
  2. 02_DB_SCHEMA.md (migration DB)
  3. 06_PIPELINE.md (batch runner generate_compositions)
Référence Charter : Section 4.4 (axes configurables)

Workflow :
  1. Ajouter axe YAML (configs/phases/*/phase.yaml)
  2. Batch runner détecte axe → migration DB auto
  3. Kernel extrait composition[axe]
  4. Tests validation (tests/integration/test_axes_iteration.py)
```

**Tâche : Optimiser queries DB**
```
Module impliqué : utils/database.py
Documentation :
  1. 08_UTILS_DATABASE.md (toolkit)
  2. ANNEXES/CODE_UTILS_DB.md (helpers détaillés)
  3. 02_DB_SCHEMA.md (indexes disponibles)
Référence Charter : Section 4.6 (DB colonnes SQL)

Workflow :
  1. Identifier bottleneck (profiling queries)
  2. Ajouter index si nécessaire (ALTER TABLE)
  3. Utiliser load_observations(columns=...) charge partielle
  4. Benchmarks (tests/benchmarks/bench_verdict.py)
```

**Tâche : Débugger régime incorrect**
```
Module impliqué : featuring/extractor.py, configs/thresholds/
Documentation :
  1. 03_FEATURING.md (régimes calcul intra-run)
  2. 04_PROFILING.md (distributions régimes)
  3. featuring/featuring_catalog.md (features utilisées régimes)

Workflow :
  1. Identifier observations régime suspect
  2. Inspecter features scalaires (frobenius_norm_final, ...)
  3. Vérifier seuils YAML (configs/thresholds/regimes.yaml)
  4. Optionnel : Calibration seuils (axe temporaire)
  5. Tests régression (tests/featuring/test_regimes.py)
```

### 3.2 Documents référence rapide

**Architecture globale** :
- `00_PHILOSOPHIE.md` : Principes immuables, vision globale
- `01_ARCHITECTURE.md` : Structure modules (ce document)

**Implémentation modules** :
- `02_DB_SCHEMA.md` : Schema SQL, migrations, volumétrie
- `03_FEATURING.md` : Extraction intra-run (layers, registres)
- `04_PROFILING.md` : Aggregation inter-run (median, régimes)
- `05_ANALYSING.md` : Patterns ML (clustering, concordance)
- `06_PIPELINE.md` : Batch runner, verdict
- `07_AXES_COMPOSITION.md` : Discovery, YAML, dry-run
- `08_UTILS_DATABASE.md` : Toolkit lecture DB
- `09_MIGRATION.md` : Ancien → nouveau pipeline

**Code détaillé** :
- `ANNEXES/CODE_FEATURING.md` : Registres complets
- `ANNEXES/CODE_PROFILING.md` : Aggregation, régimes, timelines
- `ANNEXES/CODE_ANALYSING.md` : Clustering, outliers, variance
- `ANNEXES/CODE_PIPELINE.md` : Batch runner, verdict complets
- `ANNEXES/CODE_UTILS_DB.md` : Utils/database helpers
- `ANNEXES/TESTS_VALIDATION.md` : Tests, benchmarks

**Catalogues modules** :
- `core/core_catalog.md` : Fonctions kernel, state_preparation
- `atomics/gamma_catalog.md` : Gammas disponibles (GAM-001 à GAM-013)
- `atomics/d_encoding_catalog.md` : Encodings D (SYM, ASY, R3)
- `atomics/modifier_catalog.md` : Modifiers (noise, constraints, plugins)
- `featuring/featuring_catalog.md` : Features extraction disponibles
- `featuring/registries/registries_catalog.md` : Registres complets
- `profiling/profiling_catalog.md` : Aggregations, régimes, timelines
- `analysing/analysing_catalog.md` : Analyses ML disponibles

---

## ✅ SECTION 4 : CHECKLIST DÉVELOPPEMENT

### 4.1 Avant coder (OBLIGATOIRE)

**Consulter catalogues** :
- [ ] Vérifier fonction n'existe pas déjà (duplication)
- [ ] Lire patterns implémentation (registries/PATTERNS.md si featuring)
- [ ] Identifier dépendances nécessaires (stdlib vs PRC)

**Validation structure** :
- [ ] Définir ALGO (langage courant, zéro code)
- [ ] Définir STRUCTURE (squelette fonction, passages I/O)
- [ ] Validation utilisateur AVANT coder (Charter 7.0 section 3.1)

**Configuration** :
- [ ] Params externalisés YAML (pas hardcodé)
- [ ] Vérifier schema DB si nouvel axe (migration auto)

### 4.2 Pendant codage

**Imports** :
- [ ] Respecter dépendances autorisées (section 2.2)
- [ ] Featuring/registries : Stdlib uniquement (numpy, scipy)
- [ ] Pas imports circulaires

**Gestion erreurs** :
- [ ] Distinguer crash/erreur logique/erreur params/explosion physique
- [ ] Protection multi-niveaux (registres → extractor → verdict)
- [ ] NaN = signal explosion physique (pas erreur code)

**Tests** :
- [ ] Tests unitaires (pytest)
- [ ] Tests intégration si modification pipeline
- [ ] Benchmarks si optimisation performances

### 4.3 Avant commit

**Vérifications automatiques** :
```bash
# Imports circulaires
$ python -m pycircular prc/

# Tests unitaires
$ pytest tests/

# Coverage (optionnel)
$ pytest --cov=prc tests/
```

**Vérifications manuelles** :
- [ ] Pas de params hardcodés (grep hardcode suspects)
- [ ] Catalogues mis à jour si nouvelle fonction
- [ ] Documentation inline (docstrings)

**Checklist finale** :
- [ ] Code respecte Charter 7.0 principes
- [ ] Dépendances respectées (section 2.2)
- [ ] Tests passent
- [ ] Pas de warnings imports circulaires

---

## 🏗️ SECTION 5 : PATTERNS ORGANISATION CODE

### 5.1 Pattern registres (featuring)

**Localisation** : `featuring/registries/`

**Structure type** :
```python
# featuring/registries/algebra_registry.py

from .base_registry import BaseRegistry

class AlgebraRegistry(BaseRegistry):
    """Registre fonctions algébriques."""
    
    def __init__(self):
        super().__init__()
        self.layer = 'universal'  # Ou matrix_2d, matrix_square, tensor_3d
    
    @BaseRegistry.register_function("frobenius_norm")
    def compute_frobenius_norm(self, state, **kwargs):
        """
        Calcule norme Frobenius tenseur.
        
        Args:
            state (np.ndarray): Tenseur rank quelconque
        
        Returns:
            float: Norme Frobenius
        """
        return float(np.linalg.norm(state, 'fro'))
    
    @BaseRegistry.register_function("trace", layer="matrix_square")
    def compute_trace(self, state, **kwargs):
        """
        Calcule trace matrice carrée.
        
        Args:
            state (np.ndarray): Matrice carrée
        
        Returns:
            float: Trace
        """
        return float(np.trace(state))
```

**Principes** :
- Héritage `BaseRegistry`
- Décorateur `@register_function(registry_key)`
- Layer optionnel (validation applicability)
- Docstring obligatoire
- Return type explicite (float, int, bool)
- Protection NaN/Inf interne fonction

### 5.2 Pattern hub orchestration

**Localisation** : `featuring/hub.py`, `profiling/hub.py`, `analysing/hub.py`

**Structure type** :
```python
# featuring/hub.py

def extract_features_ml(history: np.ndarray, config: dict) -> dict:
    """
    Orchestrateur featuring (extraction intra-run).
    
    Args:
        history: Timeline états (T, *dims)
        config: Configuration YAML
    
    Returns:
        dict: Features scalaires (~150)
    """
    features = {}
    
    # 1. Validation input
    if not np.all(np.isfinite(history)):
        logger.warning("History contains NaN/Inf")
    
    # 2. Inspection history
    rank, dims, is_square, is_cubic = inspect_history(history)
    
    # 3. Extraction layers
    features.update(extract_universal_features(history, config))
    
    if rank == 2:
        features.update(extract_matrix_2d_features(history, config))
        if is_square:
            features.update(extract_matrix_square_features(history, config))
    
    elif rank == 3:
        features.update(extract_tensor_3d_features(history, config))
    
    # 4. Dynamic events
    features.update(detect_dynamic_events(history, config))
    
    # 5. Régimes
    features['regime'] = classify_regime(features, config)
    
    return features
```

**Principes** :
- Fonction unique point entrée
- Validation input systématique
- Orchestration séquentielle (layers → events → régimes)
- Return dict structuré
- Logging explicite (info, warning, error)

### 5.3 Pattern utils helpers

**Localisation** : `utils/database.py`

**Structure type** :
```python
# utils/database.py

def load_observations(phase: str, columns: list = None) -> pd.DataFrame:
    """
    Charge observations DB (charge partielle).
    
    Args:
        phase: Phase à charger (R0, R1, R2, ...)
        columns: Colonnes à charger (None = toutes)
    
    Returns:
        DataFrame: Observations filtrées
    """
    conn = sqlite3.connect('prc_databases/db_results.db')
    
    # Charge partielle colonnes
    if columns:
        cols_str = ', '.join(columns)
        query = f"SELECT {cols_str} FROM observations WHERE phase = ?"
    else:
        query = "SELECT * FROM observations WHERE phase = ?"
    
    df = pd.read_sql(query, conn, params=(phase,))
    conn.close()
    
    return df
```

**Principes** :
- Signature explicite (types hints)
- Docstring Args/Returns
- Charge partielle par défaut
- Gestion connexion DB (open/close)
- Return type pandas standard

---

## 📊 SECTION 6 : MÉTRIQUES QUALITÉ CODE

### 6.1 Complexité modules

**Cibles** :
- Registres : <50 lignes/fonction
- Hubs : <200 lignes totales
- Utils : <100 lignes/fonction
- Tests : <50 lignes/test

**Mesure** :
```bash
$ radon cc prc/ -a  # Complexité cyclomatique
A = 1-5 (simple)
B = 6-10 (complexe)
C = 11-20 (très complexe)
D = 21-50 (difficile maintenir)
```

**Cible** : >90% fonctions grade A-B

### 6.2 Coverage tests

**Cibles** :
- Registres : >90% coverage
- Hubs : >80% coverage
- Utils : >95% coverage
- Pipeline : >70% coverage

**Mesure** :
```bash
$ pytest --cov=prc --cov-report=html tests/
$ open htmlcov/index.html
```

### 6.3 Documentation inline

**Cibles** :
- Registres : 100% fonctions docstring
- Hubs : 100% fonctions docstring
- Utils : 100% fonctions docstring

**Vérification** :
```bash
$ pydocstyle prc/
```

---

## 🎯 SECTION 7 : WORKFLOW TYPE DÉVELOPPEMENT

### 7.1 Ajout feature extraction

**Étape 1 : Consulter catalogues**
```bash
$ cat featuring/registries/registries_catalog.md | grep "ma_feature"
# Si existe → utiliser existante
# Si pas existe → continuer
```

**Étape 2 : ALGO (validation utilisateur)**
```
Objectif : Calculer entropie Shannon distribution valeurs tenseur

Algo :
1. Aplatir tenseur → vecteur 1D
2. Créer histogramme (bins configurables YAML)
3. Normaliser histogramme → probabilités
4. Calculer entropie Shannon : H = -Σ p_i log(p_i)
5. Normaliser par log(bins) si demandé YAML
6. Retourner float
```

**Étape 3 : STRUCTURE (validation utilisateur)**
```python
# featuring/registries/entropy_registry.py
@register_function("shannon_entropy")
def compute_shannon_entropy(self, state, bins=50, normalize=True):
    """
    Calcule entropie Shannon distribution.
    
    Args:
        state: Tenseur rank quelconque
        bins: Nombre bins histogramme (config YAML)
        normalize: Normaliser [0,1] (config YAML)
    
    Returns:
        float: Entropie [0, log(bins)] ou [0, 1] si normalize
    """
    # CODE ici (après validation structure)
```

**Étape 4 : CODE (après validation)**
```python
@register_function("shannon_entropy")
def compute_shannon_entropy(self, state, bins=50, normalize=True):
    flat = state.flatten()
    counts, _ = np.histogram(flat, bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Éviter log(0)
    entropy = -np.sum(probs * np.log(probs))
    if normalize:
        entropy /= np.log(bins)
    return float(entropy)
```

**Étape 5 : Config YAML**
```yaml
# configs/features/default/entropy.yaml
entropy_features:
  - function: "entropy.shannon_entropy"
    bins: 50
    normalize: true
    projections: [final, mean]
```

**Étape 6 : Tests**
```python
# tests/featuring/test_entropy.py
def test_shannon_entropy():
    state = np.random.rand(10, 10)
    registry = EntropyRegistry()
    entropy = registry.compute_shannon_entropy(state, bins=50)
    assert 0 <= entropy <= 1  # Si normalize=True
```

**Étape 7 : Batch runner**
```bash
$ python batch_runner.py --phase r0_dev
# Migration DB auto : ADD COLUMN shannon_entropy_final, shannon_entropy_mean
```

### 7.2 Modification seuils régimes

**Étape 1 : Identifier seuil**
```yaml
# configs/thresholds/default/regimes.yaml
conservation:
  frobenius_norm_final:
    min_ratio: 0.95  # ← Modifier ici
    max_ratio: 1.05
```

**Étape 2 : Calibration (optionnel)**
```yaml
# configs/phases/exploration/calibration_regime.yaml
iteration_axes:
  gamma_id: [GAM-001]
  min_ratio: [0.90, 0.95, 0.98]  # Axe temporaire
```

**Étape 3 : Analyse**
```python
df = load_observations('calibration_regime')
df.groupby('min_ratio')['regime'].value_counts()
# Identifier min_ratio optimal
```

**Étape 4 : Fixer YAML production**
```yaml
# configs/thresholds/default/regimes.yaml
conservation:
  frobenius_norm_final:
    min_ratio: 0.95  # Valeur calibrée
```

**Étape 5 : Relancer verdict**
```bash
$ python verdict.py --phase r0
# Pas besoin relancer batch runner (features déjà DB)
```

---

## 📝 CONCLUSION

**Ce document définit** :
- ✅ Structure dossiers complète (arborescence finale)
- ✅ Dépendances modules (autorisées/interdites)
- ✅ Navigation documentation (tâche → module → doc)
- ✅ Checklist développement (avant/pendant/après)
- ✅ Patterns organisation (registres, hubs, utils)
- ✅ Workflows types (ajout feature, calibration)

**Prochaine étape** : Lire `02_DB_SCHEMA.md` (schema SQL + migrations)

---

**FIN 01_ARCHITECTURE.md**

# 02 - PARQUET SCHEMA & STOCKAGE

**Date** : 16 février 2026  
**Version** : 1.0  
**Rôle** : Schema données Parquet + stratégie stockage

---

## 🎯 OBJECTIF DE CE DOCUMENT

**Ce document définit** :
- Format stockage Parquet (colonnes DataFrame)
- Structure fichiers (1 fichier par phase)
- Checkpoints robustesse (crash safety)
- Volumétrie estimée (compression)
- Conventions colonnes (axes + features)

**Prérequis** : Avoir lu `00_PHILOSOPHIE.md` (principe DB unique partitionnée)

---

## 📁 SECTION 1 : STRUCTURE FICHIERS

### Organisation dossiers

```
prc_databases/
├── R0.parquet          # Phase R0 (5,364 obs)
├── R1.parquet          # Phase R1 (57,792 obs)
├── R2.parquet          # Phase R2 (estimé 200,000 obs)
├── R3.parquet          # Phase R3
├── checkpoints/        # Backups intermédiaires (temporaires)
│   ├── R1_checkpoint_1000.parquet
│   ├── R1_checkpoint_2000.parquet
│   └── ...
└── archives/           # Archivage long terme
    ├── R0_2026-02-10.parquet
    └── R1_2026-02-15.parquet
```

### Principe : 1 fichier par phase

**Avantages** :
- ✅ **Isolation phases** : R0, R1, R2 indépendants
- ✅ **Backup facile** : Copier fichier = archiver phase
- ✅ **Charge sélective** : Verdict charge seulement phases nécessaires
- ✅ **Pas corruption** : Pas append complexe (write once)

**Pattern écriture** :
```python
# Fin batch_runner
df = pd.DataFrame(observations)  # Liste dicts → DataFrame
df.to_parquet(f'prc_databases/{phase}.parquet')
```

**Pattern lecture** :
```python
# Verdict single phase
df = pd.read_parquet('prc_databases/R1.parquet')

# Verdict cross-phases
df_r0 = pd.read_parquet('prc_databases/R0.parquet')
df_r1 = pd.read_parquet('prc_databases/R1.parquet')
df_all = pd.concat([df_r0, df_r1], ignore_index=True)
```

---

## 📊 SECTION 2 : SCHEMA DATAFRAME

### Colonnes axes (standards)

| Colonne | Type | Description | Exemple |
|---------|------|-------------|---------|
| `exec_id` | str | Identifiant unique observation | 'R1_000042' |
| `phase` | str | Phase expérimentale | 'R0', 'R1', 'R2' |
| `timestamp` | str | Date/heure run | '2026-02-16 14:23:45' |
| `gamma_id` | str | Opérateur γ | 'GAM-001' |
| `d_encoding_id` | str | Encoding D | 'SYM-001', 'ASY-002', 'R3-001' |
| `modifier_id` | str | Modifier D (optionnel) | 'M1', 'M2', None |


### Colonnes axes (custom optionnels)

| Colonne | Type | Description | Phases |
|---------|------|-------------|--------|
| `seed` | int | Graine aléatoire | R0 (exploration) |
| `DOF` | int | Degrés liberté | R2+ (rank variants) |
| `config_featuring` | str | Config features | Calibration |
| `config_threshold` | str | Config seuils | Validation |
| `noise_level` | float | Niveau bruit | Si modifier noise |
| `gamma1_id` | str | Gamma 1 (compositions) | R1+ |
| `gamma2_id` | str | Gamma 2 (compositions) | R1+ |
| `gamma3_id` | str | Gamma 3 (ternaires) | R3+ |
| `run_duration` | float | Durée run (secondes) | testing|
**Flexibilité** : Nouveaux axes = nouvelles colonnes (pas migrations)

---

### Colonnes features (algebra)

| Feature | Type | Description |
|---------|------|-------------|
| `frobenius_norm_initial` | float64 | Norme Frobenius t=0 |
| `frobenius_norm_final` | float64 | Norme Frobenius t=T |
| `frobenius_norm_mean` | float64 | Norme moyenne temporelle |
| `frobenius_norm_std` | float64 | Écart-type temporel |
| `frobenius_norm_slope` | float64 | Pente régression linéaire |
| `frobenius_norm_cv` | float64 | Coefficient variation |
| `trace_initial` | float64 | Trace t=0 (matrix square) |
| `trace_final` | float64 | Trace t=T |
| `determinant_final` | float64 | Déterminant t=T |
| `condition_number_mean` | float64 | Conditionnement moyen |
| `matrix_rank_final` | int | Rang matrice |

### Colonnes features (graph)

| Feature | Type | Description |
|---------|------|-------------|
| `density_initial` | float64 | Densité graphe t=0 |
| `density_final` | float64 | Densité graphe t=T |
| `density_mean` | float64 | Densité moyenne |
| `clustering_coeff_mean` | float64 | Clustering coefficient |
| `transitivity_mean` | float64 | Transitivité |
| `avg_degree_mean` | float64 | Degré moyen |
| `max_degree_final` | int | Degré maximum |

### Colonnes features (spectral)

| Feature | Type | Description |
|---------|------|-------------|
| `eigenvalue_max_initial` | float64 | Eigenvalue max t=0 |
| `eigenvalue_max_final` | float64 | Eigenvalue max t=T |
| `eigenvalue_max_mean` | float64 | Eigenvalue max moyen |
| `eigenvalue_min_final` | float64 | Eigenvalue min t=T |
| `spectral_gap_mean` | float64 | Gap spectral moyen |
| `spectral_radius_final` | float64 | Rayon spectral |
| `fft_power_mean` | float64 | Puissance spectrale FFT |

### Colonnes features (pattern)

| Feature | Type | Description |
|---------|------|-------------|
| `symmetry_score_final` | float64 | Score symétrie |
| `sparsity_mean` | float64 | Sparsité moyenne |
| `diagonal_dominance_final` | float64 | Dominance diagonale |

### Colonnes features (statistical)

| Feature | Type | Description |
|---------|------|-------------|
| `mean_value_initial` | float64 | Moyenne valeurs t=0 |
| `mean_value_final` | float64 | Moyenne valeurs t=T |
| `std_value_mean` | float64 | Écart-type moyen |
| `variance_mean` | float64 | Variance moyenne |
| `skewness_final` | float64 | Asymétrie distribution |
| `kurtosis_final` | float64 | Kurtosis distribution |
| `percentile_25_final` | float64 | Percentile 25% |
| `percentile_75_final` | float64 | Percentile 75% |
| `iqr_final` | float64 | Intervalle interquartile |

### Colonnes features (entropy)

| Feature | Type | Description |
|---------|------|-------------|
| `shannon_entropy_final` | float64 | Entropie Shannon |
| `renyi_entropy_final` | float64 | Entropie Rényi |
| `von_neumann_entropy_final` | float64 | Entropie von Neumann |

### Colonnes features (dynamic events)

| Feature | Type | Description |
|---------|------|-------------|
| `deviation_detected` | int | Déviation détectée (0/1) |
| `deviation_time` | float64 | Temps déviation [0,1] |
| `deviation_magnitude` | float64 | Magnitude déviation |
| `saturation_detected` | int | Saturation détectée (0/1) |
| `saturation_time` | float64 | Temps saturation |
| `saturation_level` | float64 | Niveau saturation |
| `collapse_detected` | int | Collapse détecté (0/1) |
| `collapse_time` | float64 | Temps collapse |
| `instability_detected` | int | Instabilité détectée (0/1) |
| `instability_time` | float64 | Temps instabilité |
| `instability_magnitude` | float64 | Magnitude instabilité |
| `oscillation_detected` | int | Oscillation détectée (0/1) |
| `oscillation_frequency` | float64 | Fréquence oscillations |
| `oscillation_amplitude` | float64 | Amplitude oscillations |

### Colonnes métadata (classification)

| Feature | Type | Description |
|---------|------|-------------|
| `regime` | str | Régime classifié | 'CONSERVES_NORM', 'PATHOLOGICAL', ... |
| `regime_confidence` | float64 | Confiance classification (optionnel) |
| `timeline_descriptor` | str | Pattern temporel | 'stable', 'early_deviation', ... |

**Total colonnes** : ~160 (10 axes + 150 features)

---

## 📐 SECTION 3 : VOLUMÉTRIE ESTIMÉE

### Calcul taille observation

**Axes** (10 colonnes) :
- 7 colonnes str (20 bytes average) = 140 bytes
- 1 colonne int (8 bytes) = 8 bytes
- 2 colonnes float (8 bytes each) = 16 bytes
- **Total axes** : ~164 bytes/observation

**Features** (150 colonnes) :
- 140 colonnes float64 (8 bytes each) = 1,120 bytes
- 10 colonnes int (8 bytes each) = 80 bytes
- **Total features** : ~1,200 bytes/observation

**Total brut** : ~1,364 bytes/observation ≈ **1.33 KB**

### Volumétrie par phase

**Compression Parquet** : Facteur ~3× (colonnes homogènes)

| Phase | Observations | Brut (MB) | Parquet (MB) | Disk |
|-------|-------------|-----------|--------------|------|
| **R0** | 5,364 | 7.1 | **2.4** | ✅ Tiny |
| **R1** | 57,792 | 76.1 | **25.4** | ✅ Small |
| **R2** | 200,000 | 267.0 | **89.0** | ✅ Medium |
| **R3** | 500,000 | 667.5 | **222.5** | ✅ OK |
| **Total R0-R3** | ~763k | 1,018 MB | **339 MB** | ✅ Gérable |

**Comparaison SQL** :
- SQL (indexes 15%) : 1,018 × 1.15 = **1,171 MB**
- Parquet : **339 MB**
- **Gain** : -71% volumétrie

---

## 💾 SECTION 4 : CHECKPOINTS ROBUSTESSE

### Principe crash safety

**Problème** : Batch 57,792 runs = 10-15h → Risque crash

**Solution** : Checkpoints périodiques pendant batch

```python
# batch_runner.py

def run_batch(compositions, config, phase):
    observations = []  # RAM accumulation
    
    for i, composition in enumerate(compositions):
        # Run
        history = run_kernel(composition, config)
        features = extract_features_ml(history, config)
        observation = {
            'exec_id': f"{phase}_{i:06d}",
            'phase': phase,
            **composition,
            **features
        }
        observations.append(observation)
        
        # Checkpoint tous les 1000 runs
        if len(observations) % 1000 == 0:
            checkpoint_path = f'prc_databases/checkpoints/{phase}_checkpoint_{i:06d}.parquet'
            df_checkpoint = pd.DataFrame(observations)
            df_checkpoint.to_parquet(checkpoint_path)
            logger.info(f"✓ Checkpoint {i}: {len(observations)} observations")
    
    # Dump final
    df_final = pd.DataFrame(observations)
    df_final.to_parquet(f'prc_databases/{phase}.parquet')
    logger.info(f"✓ Phase {phase} saved: {len(observations)} observations")
    
    # Cleanup checkpoints (optionnel)
    shutil.rmtree('prc_databases/checkpoints/')
```

**Récupération crash** :
```python
# Si crash détecté
checkpoints = sorted(Path('prc_databases/checkpoints/').glob(f'{phase}_checkpoint_*.parquet'))
if checkpoints:
    # Charger dernier checkpoint
    last_checkpoint = checkpoints[-1]
    df_recovered = pd.read_parquet(last_checkpoint)
    
    # Extraire dernier exec_id
    last_exec_id = df_recovered['exec_id'].iloc[-1]  # Ex: 'R1_012000'
    last_idx = int(last_exec_id.split('_')[1])
    
    # Reprendre à last_idx + 1
    remaining_compositions = compositions[last_idx + 1:]
    logger.info(f"Recovered {last_idx} runs, resuming from {last_idx + 1}")
```

---

## 🔄 SECTION 5 : CONVENTIONS COLONNES

### Nommage features

**Pattern** : `{registry}_{metric}_{projection/statistic}`

**Exemples** :
```
frobenius_norm_initial     # Projection initial
frobenius_norm_final       # Projection final
frobenius_norm_mean        # Projection mean
frobenius_norm_std         # Statistic std
frobenius_norm_slope       # Statistic slope

density_mean               # Projection mean (graph)
eigenvalue_max_final       # Projection final (spectral)
```

**Registres préfixes** :
- `frobenius_norm_*`, `trace_*`, `determinant_*` → algebra
- `density_*`, `clustering_coeff_*`, `degree_*` → graph
- `eigenvalue_*`, `spectral_*`, `fft_*` → spectral
- `symmetry_*`, `sparsity_*`, `diagonal_*` → pattern
- `mean_value_*`, `std_value_*`, `skewness_*` → statistical
- `shannon_entropy_*`, `renyi_entropy_*` → entropy

### Valeurs spéciales

**NaN** : Feature non applicable ou calcul échoué
```python
# Exemple : eigenvalue sur matrice non-carrée
eigenvalue_max_final = np.nan  # Pas applicable rank ≠ 2 square
```

**None** : Axe optionnel absent
```python
modifier_id = None  # Pas de modifier appliqué
gamma3_id = None    # Composition binaire (pas ternaire)
```

**0/1** : Booléens events
```python
deviation_detected = 1   # Détecté
saturation_detected = 0  # Pas détecté
```

---

## ⚡ SECTION 6 : PERFORMANCE

### Write performance

**Benchmark R1** (57,792 observations) :
```python
# Mesure temps write
start = time.time()
df = pd.DataFrame(observations)  # Liste 57,792 dicts
df.to_parquet('R1.parquet')
duration = time.time() - start

# Résultat attendu : 1-2 secondes
print(f"Write: {duration:.2f}s")  # ~1.5s
```

**Comparaison SQL** :
- Parquet : 1-2s (write atomique)
- SQL INSERT : 5-10s (transactions batches)

### Read performance

**Benchmark R1** :
```python
# Charge complète
start = time.time()
df = pd.read_parquet('R1.parquet')
duration = time.time() - start
print(f"Read full: {duration:.2f}s")  # ~0.5s

# Charge partielle (10 colonnes)
start = time.time()
df = pd.read_parquet('R1.parquet', columns=['exec_id', 'gamma_id', 'frobenius_norm_final', ...])
duration = time.time() - start
print(f"Read partial: {duration:.2f}s")  # ~0.1s
```

**RAM verdict** :
- R1 full (160 cols) : 76 MB RAM
- R1 partial (10 cols) : 5 MB RAM
- R3 full (160 cols) : 267 MB RAM (gérable)

---

## 🔍 SECTION 7 : QUERIES & FILTRES

### Filtrage pandas

```python
# Charge phase
df = pd.read_parquet('R1.parquet')

# Filtres axes
df_gam001 = df[df['gamma_id'] == 'GAM-001']
df_conserves = df[df['regime'] == 'CONSERVES_NORM']
df_stable_gam001 = df[(df['gamma_id'] == 'GAM-001') & (df['regime'] == 'CONSERVES_NORM')]

# Sélection colonnes
df_features = df[['exec_id', 'gamma_id', 'frobenius_norm_final', 'density_mean']]

# Aggregation
median_frobenius = df.groupby('gamma_id')['frobenius_norm_final'].median()
regime_dist = df.groupby('gamma_id')['regime'].value_counts(normalize=True)
```

### Cross-phases

```python
# Concat phases
df_r0 = pd.read_parquet('R0.parquet')
df_r1 = pd.read_parquet('R1.parquet')
df_all = pd.concat([df_r0, df_r1], ignore_index=True)

# Filter cross-phases
df_gam001_all_phases = df_all[df_all['gamma_id'] == 'GAM-001']

# Concordance
regime_r0 = df_r0.groupby('gamma_id')['regime'].apply(lambda x: x.mode()[0])
regime_r1 = df_r1.groupby('gamma_id')['regime'].apply(lambda x: x.mode()[0])
concordance = (regime_r0 == regime_r1).mean()
```

---

## 📦 SECTION 8 : BACKUP & ARCHIVAGE

### Stratégie backup

**Backup AVANT** :
- Migrations schema (ajout colonnes)
- Suppressions massives
- Scripts manuels risqués

**Backup APRÈS** :
- Fin phase complète (archivage long terme)

```python
# Backup manuel
import shutil
from datetime import datetime

timestamp = datetime.now().strftime('%Y-%m-%d')
shutil.copy2('prc_databases/R1.parquet', f'prc_databases/archives/R1_{timestamp}.parquet')
```

**Rotation backups** :
- 3 derniers quotidiens (si modifications fréquentes)
- 1 backup/phase (archivage permanent)
- Compression archives >6 mois (gzip)

### Export formats

```python
# Export CSV (analyse externe)
df = pd.read_parquet('R1.parquet')
df.to_csv('exports/R1.csv', index=False)

# Export Excel (présentation)
df.to_excel('exports/R1.xlsx', index=False)

# Export JSON (interopérabilité)
df.to_json('exports/R1.json', orient='records')
```

---

## ✅ SECTION 9 : CHECKLIST STOCKAGE

### Avant batch runner

**Configuration** :
- [ ] Phase définie (R0, R1, R2, ...)
- [ ] Espace disque suffisant (estimation volumétrie)
- [ ] Checkpoints activés (crash safety)

### Pendant batch

**Monitoring** :
- [ ] Checkpoints créés régulièrement
- [ ] RAM stable (<500 MB)
- [ ] Pas erreurs write

### Après batch

**Validation** :
- [ ] Fichier phase créé (prc_databases/{phase}.parquet)
- [ ] Taille cohérente (estimation volumétrie)
- [ ] Chargement pandas réussit
- [ ] Schema complet (toutes colonnes attendues)

---

## 📝 CONCLUSION

**Ce document définit** :
- ✅ Format Parquet (1 fichier/phase)
- ✅ Schema DataFrame (160 colonnes)
- ✅ Volumétrie (339 MB R0-R3, -71% vs SQL)
- ✅ Checkpoints robustesse (crash safety)
- ✅ Performance (write 1-2s, read 0.5s)
- ✅ Conventions colonnes (nommage, valeurs spéciales)

**Avantages Parquet** :
- ✅ Code simple (pandas API)
- ✅ Pas migrations (schema flexible)
- ✅ Compression 3× (vs SQL)
- ✅ Performance 2-3× (vs SQL)
- ✅ Checkpoints faciles

**Prochaine étape** : Lire `08_UTILS_DATABASE.md` (helpers lecture/écriture)

---

**FIN 02_PARQUET_SCHEMA.md**
# 03 - FEATURING (EXTRACTION INTRA-RUN)

**Date** : 16 février 2026  
**Version** : 1.0  
**Rôle** : Extraction features intra-run (history → features dict)

---

## 🎯 OBJECTIF DE CE DOCUMENT

**Ce document définit** :
- Architecture featuring (hub, extractor, layers)
- Logique layers (inspection directe history)
- Projections temporelles (initial, final, mean, ...)
- Statistics scalaires (mean, std, slope, cv)
- Dynamic events (détection + interprétation)
- Régimes (classification intra-run)
- Timelines descriptors (patterns temporels)
- Registres (11 modules fonctions extraction)

**Prérequis** : Avoir lu `00_PHILOSOPHIE.md` (principe intra-run), `01_ARCHITECTURE.md` (structure), `02_DB_SCHEMA.md` (schema features)

---

## 🎯 SECTION 1 : PRINCIPE FEATURING

### 1.1 Rôle featuring

**Définition** : Extraction features intra-run = calculs sur run unique (history RAM complète)

**Timing** : À la volée, pendant batch_runner
```
Kernel → history (T, n, m) RAM
  ↓
Featuring → features dict (~150 scalaires)
  ↓
DB insert (1 write)
```

**Principe** : **80% calculs totaux** (profiter history RAM, minimiser I/O DB)

### 1.2 Input : History complète RAM

**Format** :
```python
history: np.ndarray
  shape: (T, *dims)
  T: Nombre steps temporels (config kernel, typ. ~201)
  dims: Dimensions spatiales (n, m) ou (n, m, p)
  
Exemples :
  - Rank 2 : (201, 100, 100) - Matrice 100×100 évoluant 201 steps
  - Rank 3 : (201, 50, 50, 50) - Tenseur 50³ évoluant 201 steps
```

**Principe clé** : History complète disponible (pas snapshots, résolution temporelle maximale)

### 1.3 Output : Features dict

**Format** :
```python
features: dict[str, float | int | bool | str]
  ~150 features scalaires
  
Exemple :
{
    # Algebra
    'frobenius_norm_initial': 8.234,
    'frobenius_norm_final': 8.145,
    'frobenius_norm_mean': 8.189,
    'frobenius_norm_std': 0.052,
    'frobenius_norm_slope': -0.0004,
    
    # Graph
    'density_mean': 0.452,
    'clustering_coeff_mean': 0.234,
    
    # Spectral
    'eigenvalue_max_final': 12.456,
    'spectral_gap_mean': 2.134,
    
    # Dynamic events
    'deviation_detected': 1,
    'deviation_time': 0.234,
    'saturation_detected': 0,
    
    # Régimes
    'regime': 'CONSERVES_NORM',
    
    # Timelines
    'timeline_descriptor': 'stable_then_slight_decay'
}
```

**Stockage DB** : 1 write (INSERT observation), colonnes SQL

---

## 🏗️ SECTION 2 : ARCHITECTURE FEATURING

### 2.1 Modules featuring

```
featuring/
├── hub.py                  # Orchestrateur (extract_features_ml)
├── extractor.py            # Logique extraction layers
├── layers.py               # Inspection history + dispatch layers
├── projections.py          # Projections temporelles (initial, final, mean)
├── statistics.py           # Statistics scalaires (std, slope, cv)
├── dynamic_events.py       # Détection événements timeline
└── registries/             # Pool fonctions extraction (11 modules)
    ├── base_registry.py
    ├── registry_manager.py
    ├── algebra_registry.py
    ├── graph_registry.py
    ├── spectral_registry.py
    ├── spatial_registry.py
    ├── topological_registry.py
    ├── pattern_registry.py
    ├── statistical_registry.py
    ├── entropy_registry.py
    ├── tensor_registry.py
    └── timeseries_registry.py
```

### 2.2 Flux extraction

```
extract_features_ml(history, config)
  │
  ├──> inspect_history(history)
  │      └──> rank, dims, is_square, is_cubic
  │
  ├──> extract_universal_features(history, config)
  │      └──> algebra.frobenius_norm, statistical.entropy, ...
  │
  ├──> IF rank == 2:
  │      ├──> extract_matrix_2d_features(history, config)
  │      │      └──> graph.density, spectral.fft_power, ...
  │      │
  │      └──> IF is_square:
  │             └──> extract_matrix_square_features(history, config)
  │                    └──> algebra.trace, spectral.eigenvalue_max, ...
  │
  ├──> ELIF rank == 3:
  │      ├──> extract_tensor_3d_features(history, config)
  │      │      └──> tensor.tucker_energy, tensor.cp_rank, ...
  │      │
  │      └──> IF is_cubic:
  │             └──> extract_matrix_square_features(history, config, mode='cubic')
  │                    └──> (slice plan médian, features square)
  │
  ├──> extract_spatial_2d_features(history, config)  # Si rank 2
  │      └──> spatial.gradient, topological.connected_components, ...
  │
  ├──> detect_dynamic_events(history, config)
  │      └──> deviation, saturation, collapse, instability, oscillation
  │
  ├──> classify_regime(features, config)
  │      └──> CONSERVES_NORM, PATHOLOGICAL, NUMERIC_INSTABILITY, ...
  │
  └──> interpret_timeline(features, events, config)
         └──> 'stable', 'early_deviation', 'collapse_recovery', ...
```

---

## 🔍 SECTION 3 : LAYERS (INSPECTION DIRECTE)

### 3.1 Principe layers

**Motivation** : Éviter metadata applicability cauchemar

**Approche** : Inspection directe history (rank, dims) → chemin if/elif prédéfini

**Layers disponibles** :
- `universal` : Tout tenseur (frobenius_norm, entropy, mean_value, ...)
- `matrix_2d` : Rank 2 (density, clustering_coeff, fft_power, ...)
- `matrix_square` : Rank 2 carrée (trace, eigenvalue_max, determinant, ...)
- `tensor_3d` : Rank ≥3 (tucker_energy, cp_rank, slice_correlation, ...)
- `spatial_2d` : Analyses spatiales 2D (gradient, laplacian, connected_components, ...)

### 3.2 Inspection history

```python
# featuring/layers.py

def inspect_history(history: np.ndarray) -> dict:
    """
    Inspecte history pour déterminer applicability layers.
    
    Args:
        history: Timeline états (T, *dims)
    
    Returns:
        dict: {
            'rank': int,
            'dims': tuple,
            'is_square': bool,
            'is_cubic': bool,
            'temporal_steps': int
        }
    """
    temporal_steps = history.shape[0]
    dims = history.shape[1:]  # Dimensions spatiales
    rank = len(dims)
    
    # Détection géométries spéciales
    is_square = (rank == 2) and (dims[0] == dims[1])
    is_cubic = (rank == 3) and (dims[0] == dims[1] == dims[2])
    
    return {
        'rank': rank,
        'dims': dims,
        'is_square': is_square,
        'is_cubic': is_cubic,
        'temporal_steps': temporal_steps
    }
```

### 3.3 Dispatch layers

```python
# featuring/extractor.py

def extract_features_ml(history: np.ndarray, config: dict) -> dict:
    """
    Extraction features principale (orchestrateur).
    
    Args:
        history: Timeline états (T, *dims)
        config: Configuration YAML featuring
    
    Returns:
        dict: Features scalaires (~150)
    """
    features = {}
    
    # 1. Validation input
    if not np.all(np.isfinite(history)):
        logger.warning("History contains NaN/Inf (explosion physique possible)")
    
    # 2. Inspection history
    info = inspect_history(history)
    rank = info['rank']
    is_square = info['is_square']
    is_cubic = info['is_cubic']
    
    # 3. LAYER universal (toujours applicable)
    if config['features']['layers']['universal']:
        features.update(extract_universal_features(history, config))
    
    # 4. LAYER rank-specific
    if rank == 2:
        # Matrix 2D features
        if config['features']['layers']['matrix_2d']:
            features.update(extract_matrix_2d_features(history, config))
        
        # Matrix square features (si carrée)
        if is_square and config['features']['layers']['matrix_square']:
            features.update(extract_matrix_square_features(history, config))
        
        # Spatial 2D features
        if config['features']['layers']['spatial_2d']:
            features.update(extract_spatial_2d_features(history, config))
    
    elif rank == 3:
        # Tensor 3D features
        if config['features']['layers']['tensor_3d']:
            features.update(extract_tensor_3d_features(history, config))
        
        # Cubic tensor : features square sur slice (astuce)
        if is_cubic and config['features']['layers']['matrix_square']:
            features.update(extract_matrix_square_features(
                history, config, mode='cubic'
            ))
    
    elif rank >= 4:
        # Tensor N-D features (futur)
        if config['features']['layers']['tensor_nd']:
            features.update(extract_tensor_nd_features(history, config))
    
    # 5. Dynamic events (toujours)
    events = detect_dynamic_events(history, config)
    features.update(events)
    
    # 6. Régimes (classification depuis features)
    regime = classify_regime(features, config)
    features['regime'] = regime
    
    # 7. Timelines descriptors (interprétation events)
    timeline = interpret_timeline(features, events, config)
    features['timeline_descriptor'] = timeline
    
    return features
```

### 3.4 Exemple cubic tensor (astuce slice)

**Problème** : Tensor 3D cubique (50×50×50), features square ?

**Solution** : Slice plan médian → matrice 2D carrée
```python
# featuring/extractor.py

def extract_matrix_square_features(
    history: np.ndarray,
    config: dict,
    mode: str = 'matrix'
) -> dict:
    """
    Extrait features matrix square.
    
    Args:
        history: (T, n, m) si mode='matrix', (T, n, n, n) si mode='cubic'
        config: Configuration
        mode: 'matrix' ou 'cubic'
    
    Returns:
        dict: Features square (trace, eigenvalue_max, ...)
    """
    features = {}
    
    if mode == 'cubic':
        # Slice plan médian (T, n, n, n) → (T, n, n)
        mid = history.shape[1] // 2
        history_slice = history[:, mid, :, :]
        logger.debug(f"Cubic mode: slice plan médian [{mid}, :, :]")
    else:
        history_slice = history
    
    # Extraction features square sur slice
    registry_manager = RegistryManager()
    
    for feature_config in config['features']['matrix_square']:
        registry_key = feature_config['function']
        function = registry_manager.get_function(registry_key)
        
        # Projections temporelles
        for projection in feature_config.get('projections', ['final']):
            state = project_temporal(history_slice, projection)
            
            try:
                value = function(state, **feature_config.get('params', {}))
                feature_name = f"{registry_key.replace('.', '_')}_{projection}"
                features[feature_name] = value
            
            except Exception as e:
                logger.error(f"Erreur {registry_key} ({projection}): {e}")
                continue
    
    return features
```

---

## 📊 SECTION 4 : PROJECTIONS TEMPORELLES

### 4.1 Principe projections

**Motivation** : Extraire vues temporelles history (résolution max → scalaires)

**Projections disponibles** :
- `initial` : État t=0
- `final` : État t=T-1
- `mean` : Moyenne temporelle (∫ state dt)
- `std` : Écart-type temporel
- `min` : Minimum temporel
- `max` : Maximum temporel
- `max_deviation` : Déviation max depuis initial

### 4.2 Implémentation projections

```python
# featuring/projections.py

import numpy as np

def project_temporal(history: np.ndarray, projection: str) -> np.ndarray:
    """
    Projette history selon dimension temporelle.
    
    Args:
        history: (T, *dims)
        projection: 'initial', 'final', 'mean', 'std', 'min', 'max', 'max_deviation'
    
    Returns:
        np.ndarray: État projeté (shape = dims)
    """
    if projection == 'initial':
        return history[0]
    
    elif projection == 'final':
        return history[-1]
    
    elif projection == 'mean':
        return np.mean(history, axis=0)
    
    elif projection == 'std':
        return np.std(history, axis=0)
    
    elif projection == 'min':
        return np.min(history, axis=0)
    
    elif projection == 'max':
        return np.max(history, axis=0)
    
    elif projection == 'max_deviation':
        # Déviation max depuis initial
        initial = history[0]
        deviations = np.abs(history - initial)
        return np.max(deviations, axis=0)
    
    else:
        raise ValueError(f"Projection inconnue : {projection}")


def compute_projection_statistics(
    history: np.ndarray,
    metric_function: callable,
    projections: list[str],
    **metric_params
) -> dict:
    """
    Calcule métrique sur projections temporelles.
    
    Args:
        history: (T, *dims)
        metric_function: Fonction extraction (ex: compute_frobenius_norm)
        projections: Liste projections ('initial', 'final', 'mean', ...)
        metric_params: Params fonction
    
    Returns:
        dict: {
            'metric_initial': float,
            'metric_final': float,
            'metric_mean': float,
            ...
        }
    """
    results = {}
    
    for projection in projections:
        state = project_temporal(history, projection)
        
        try:
            value = metric_function(state, **metric_params)
            results[projection] = value
        
        except Exception as e:
            logger.warning(f"Projection {projection} échoué : {e}")
            results[projection] = np.nan
    
    return results
```

### 4.3 Exemple usage projections

```python
# Exemple : frobenius_norm avec projections [initial, final, mean]

history = np.random.rand(201, 100, 100)  # Timeline 201 steps

# Projection initial
state_initial = project_temporal(history, 'initial')  # (100, 100)
frobenius_initial = np.linalg.norm(state_initial, 'fro')  # float

# Projection final
state_final = project_temporal(history, 'final')
frobenius_final = np.linalg.norm(state_final, 'fro')

# Projection mean
state_mean = project_temporal(history, 'mean')
frobenius_mean = np.linalg.norm(state_mean, 'fro')

# Output features
features = {
    'frobenius_norm_initial': frobenius_initial,
    'frobenius_norm_final': frobenius_final,
    'frobenius_norm_mean': frobenius_mean
}
```

---

## 📈 SECTION 5 : STATISTICS SCALAIRES

### 5.1 Principe statistics

**Motivation** : Capturer évolution temporelle métrique (slope, cv, ...)

**Statistics disponibles** :
- `initial` : Valeur t=0
- `final` : Valeur t=T-1
- `mean` : Moyenne temporelle
- `std` : Écart-type temporel
- `slope` : Pente régression linéaire
- `cv` : Coefficient variation (std/mean)

### 5.2 Implémentation statistics

```python
# featuring/statistics.py

import numpy as np
from scipy.stats import linregress

def compute_statistics(
    timeseries: np.ndarray,
    statistics: list[str] = ['initial', 'final', 'mean', 'std', 'slope']
) -> dict:
    """
    Calcule statistics scalaires timeseries métrique.
    
    Args:
        timeseries: Valeurs temporelles métrique (T,)
        statistics: Liste statistics à calculer
    
    Returns:
        dict: {
            'initial': float,
            'final': float,
            'mean': float,
            'std': float,
            'slope': float,
            'cv': float
        }
    """
    results = {}
    
    if 'initial' in statistics:
        results['initial'] = float(timeseries[0])
    
    if 'final' in statistics:
        results['final'] = float(timeseries[-1])
    
    if 'mean' in statistics:
        results['mean'] = float(np.mean(timeseries))
    
    if 'std' in statistics:
        results['std'] = float(np.std(timeseries))
    
    if 'slope' in statistics:
        # Régression linéaire
        t = np.arange(len(timeseries))
        slope, _, _, _, _ = linregress(t, timeseries)
        results['slope'] = float(slope)
    
    if 'cv' in statistics:
        # Coefficient variation
        mean_val = np.mean(timeseries)
        std_val = np.std(timeseries)
        if mean_val != 0:
            results['cv'] = float(std_val / np.abs(mean_val))
        else:
            results['cv'] = np.nan
    
    return results


def compute_metric_timeseries(
    history: np.ndarray,
    metric_function: callable,
    **metric_params
) -> np.ndarray:
    """
    Calcule timeseries métrique depuis history.
    
    Args:
        history: (T, *dims)
        metric_function: Fonction extraction
        metric_params: Params fonction
    
    Returns:
        np.ndarray: Timeseries (T,)
    """
    timeseries = []
    
    for t in range(history.shape[0]):
        state = history[t]
        
        try:
            value = metric_function(state, **metric_params)
            timeseries.append(value)
        
        except Exception as e:
            logger.warning(f"Métrique t={t} échoué : {e}")
            timeseries.append(np.nan)
    
    return np.array(timeseries)
```

### 5.3 Exemple usage statistics

```python
# Exemple : frobenius_norm statistics [initial, final, mean, std, slope]

history = np.random.rand(201, 100, 100)

# Timeseries frobenius_norm
frobenius_timeseries = []
for t in range(history.shape[0]):
    state = history[t]
    norm = np.linalg.norm(state, 'fro')
    frobenius_timeseries.append(norm)

frobenius_timeseries = np.array(frobenius_timeseries)  # (201,)

# Statistics
stats = compute_statistics(frobenius_timeseries, ['initial', 'final', 'mean', 'std', 'slope'])

# Output features
features = {
    'frobenius_norm_initial': stats['initial'],
    'frobenius_norm_final': stats['final'],
    'frobenius_norm_mean': stats['mean'],
    'frobenius_norm_std': stats['std'],
    'frobenius_norm_slope': stats['slope']
}
```

---

## 🔥 SECTION 6 : DYNAMIC EVENTS

### 6.1 Principe dynamic events

**Motivation** : Détecter événements significatifs timeline (deviation, saturation, collapse, ...)

**Événements disponibles** :
- `deviation` : Déviation significative depuis état initial
- `saturation` : Stabilisation métrique (plateau)
- `collapse` : Chute brutale métrique
- `instability` : Oscillations haute fréquence
- `oscillation` : Oscillations périodiques

**Output** : Booleans (detected) + metadata (time, magnitude, frequency, ...)

### 6.2 Implémentation detection events

```python
# featuring/dynamic_events.py

import numpy as np
from scipy.signal import find_peaks

def detect_dynamic_events(history: np.ndarray, config: dict) -> dict:
    """
    Détecte événements dynamiques timeline.
    
    Args:
        history: (T, *dims)
        config: Configuration seuils events
    
    Returns:
        dict: {
            'deviation_detected': int,
            'deviation_time': float,
            'saturation_detected': int,
            'saturation_time': float,
            ...
        }
    """
    events = {}
    thresholds = config['thresholds']['events']
    
    # Timeseries norme Frobenius (métrique référence)
    frobenius_timeseries = np.array([
        np.linalg.norm(history[t], 'fro') 
        for t in range(history.shape[0])
    ])
    
    # Normalisation temps [0, 1]
    t_normalized = np.linspace(0, 1, len(frobenius_timeseries))
    
    # 1. Deviation detection
    deviation_event = detect_deviation(
        frobenius_timeseries, 
        t_normalized, 
        threshold=thresholds['deviation_threshold']
    )
    events.update(deviation_event)
    
    # 2. Saturation detection
    saturation_event = detect_saturation(
        frobenius_timeseries, 
        t_normalized, 
        threshold=thresholds['saturation_threshold']
    )
    events.update(saturation_event)
    
    # 3. Collapse detection
    collapse_event = detect_collapse(
        frobenius_timeseries, 
        t_normalized, 
        threshold=thresholds['collapse_threshold']
    )
    events.update(collapse_event)
    
    # 4. Instability detection
    instability_event = detect_instability(
        frobenius_timeseries, 
        threshold=thresholds['instability_threshold']
    )
    events.update(instability_event)
    
    # 5. Oscillation detection
    oscillation_event = detect_oscillation(
        frobenius_timeseries, 
        threshold=thresholds['oscillation_threshold']
    )
    events.update(oscillation_event)
    
    return events


def detect_deviation(
    timeseries: np.ndarray, 
    t_normalized: np.ndarray, 
    threshold: float = 0.1
) -> dict:
    """
    Détecte déviation significative depuis initial.
    
    Args:
        timeseries: Valeurs temporelles (T,)
        t_normalized: Temps normalisé [0, 1]
        threshold: Seuil déviation relative (ex: 0.1 = 10%)
    
    Returns:
        dict: {
            'deviation_detected': 0 ou 1,
            'deviation_time': float [0, 1] ou None,
            'deviation_magnitude': float ou None
        }
    """
    initial_value = timeseries[0]
    
    # Déviation relative
    relative_deviation = np.abs(timeseries - initial_value) / np.abs(initial_value)
    
    # Premier dépassement seuil
    exceed_indices = np.where(relative_deviation > threshold)[0]
    
    if len(exceed_indices) > 0:
        first_exceed_idx = exceed_indices[0]
        return {
            'deviation_detected': 1,
            'deviation_time': float(t_normalized[first_exceed_idx]),
            'deviation_magnitude': float(relative_deviation[first_exceed_idx])
        }
    else:
        return {
            'deviation_detected': 0,
            'deviation_time': None,
            'deviation_magnitude': None
        }


def detect_saturation(
    timeseries: np.ndarray, 
    t_normalized: np.ndarray, 
    threshold: float = 0.01,
    window: int = 20
) -> dict:
    """
    Détecte saturation (plateau).
    
    Args:
        timeseries: Valeurs temporelles (T,)
        t_normalized: Temps normalisé [0, 1]
        threshold: Seuil variance relative (ex: 0.01 = 1%)
        window: Fenêtre glissante détection plateau
    
    Returns:
        dict: {
            'saturation_detected': 0 ou 1,
            'saturation_time': float [0, 1] ou None,
            'saturation_level': float ou None
        }
    """
    # Variance glissante
    for i in range(len(timeseries) - window):
        segment = timeseries[i:i+window]
        mean_val = np.mean(segment)
        variance_relative = np.std(segment) / np.abs(mean_val) if mean_val != 0 else np.inf
        
        if variance_relative < threshold:
            # Saturation détectée
            return {
                'saturation_detected': 1,
                'saturation_time': float(t_normalized[i]),
                'saturation_level': float(mean_val)
            }
    
    return {
        'saturation_detected': 0,
        'saturation_time': None,
        'saturation_level': None
    }


def detect_collapse(
    timeseries: np.ndarray, 
    t_normalized: np.ndarray, 
    threshold: float = 0.5
) -> dict:
    """
    Détecte collapse (chute brutale).
    
    Args:
        timeseries: Valeurs temporelles (T,)
        t_normalized: Temps normalisé [0, 1]
        threshold: Seuil chute relative (ex: 0.5 = 50%)
    
    Returns:
        dict: {
            'collapse_detected': 0 ou 1,
            'collapse_time': float [0, 1] ou None
        }
    """
    initial_value = timeseries[0]
    
    # Chute relative
    relative_drop = (initial_value - timeseries) / np.abs(initial_value)
    
    # Premier dépassement seuil
    exceed_indices = np.where(relative_drop > threshold)[0]
    
    if len(exceed_indices) > 0:
        first_exceed_idx = exceed_indices[0]
        return {
            'collapse_detected': 1,
            'collapse_time': float(t_normalized[first_exceed_idx])
        }
    else:
        return {
            'collapse_detected': 0,
            'collapse_time': None
        }


def detect_instability(
    timeseries: np.ndarray, 
    threshold: float = 0.2
) -> dict:
    """
    Détecte instabilité (oscillations haute fréquence).
    
    Args:
        timeseries: Valeurs temporelles (T,)
        threshold: Seuil écart-type relatif (ex: 0.2 = 20%)
    
    Returns:
        dict: {
            'instability_detected': 0 ou 1,
            'instability_time': float [0, 1] ou None,
            'instability_magnitude': float ou None
        }
    """
    # Dérivée seconde (approximation instabilité)
    second_derivative = np.diff(np.diff(timeseries))
    
    mean_val = np.mean(timeseries)
    std_second_derivative = np.std(second_derivative)
    
    instability_magnitude = std_second_derivative / np.abs(mean_val) if mean_val != 0 else 0
    
    if instability_magnitude > threshold:
        # Instabilité détectée
        # Temps premier pic instabilité
        peak_idx = np.argmax(np.abs(second_derivative))
        t_normalized = np.linspace(0, 1, len(timeseries))
        
        return {
            'instability_detected': 1,
            'instability_time': float(t_normalized[peak_idx]),
            'instability_magnitude': float(instability_magnitude)
        }
    else:
        return {
            'instability_detected': 0,
            'instability_time': None,
            'instability_magnitude': None
        }


def detect_oscillation(
    timeseries: np.ndarray, 
    threshold: float = 0.05
) -> dict:
    """
    Détecte oscillations périodiques.
    
    Args:
        timeseries: Valeurs temporelles (T,)
        threshold: Seuil amplitude relative (ex: 0.05 = 5%)
    
    Returns:
        dict: {
            'oscillation_detected': 0 ou 1,
            'oscillation_frequency': float ou None,
            'oscillation_amplitude': float ou None
        }
    """
    # Détection pics oscillations
    peaks, properties = find_peaks(timeseries, prominence=threshold * np.mean(timeseries))
    
    if len(peaks) >= 3:  # Au moins 3 pics (2 oscillations)
        # Fréquence moyenne oscillations
        peak_distances = np.diff(peaks)
        mean_distance = np.mean(peak_distances)
        frequency = 1.0 / mean_distance if mean_distance > 0 else 0
        
        # Amplitude oscillations
        amplitude = np.mean(properties['prominences'])
        
        return {
            'oscillation_detected': 1,
            'oscillation_frequency': float(frequency),
            'oscillation_amplitude': float(amplitude)
        }
    else:
        return {
            'oscillation_detected': 0,
            'oscillation_frequency': None,
            'oscillation_amplitude': None
        }
```

---

## 🏷️ SECTION 7 : RÉGIMES (CLASSIFICATION INTRA-RUN)

### 7.1 Principe régimes

**Motivation** : Classifier comportement run (conservation, pathologie, instabilité, ...)

**Régimes disponibles** :
- `CONSERVES_NORM` : Norme conservée (variation <5%)
- `SLIGHT_DECAY` : Décroissance lente (5-20%)
- `STRONG_DECAY` : Décroissance forte (>20%)
- `PATHOLOGICAL` : Explosion norme ou singularité
- `NUMERIC_INSTABILITY` : Oscillations incontrôlées
- `COLLAPSE` : Effondrement brutal

**Calcul** : Intra-run (features scalaires + seuils YAML absolus)

### 7.2 Implémentation classification

```python
# featuring/extractor.py (suite)

def classify_regime(features: dict, config: dict) -> str:
    """
    Classifie régime run depuis features.
    
    Args:
        features: Features scalaires déjà calculées
        config: Configuration seuils régimes
    
    Returns:
        str: Régime ('CONSERVES_NORM', 'PATHOLOGICAL', ...)
    """
    thresholds = config['thresholds']['regimes']
    
    # Extraire features critiques
    frobenius_initial = features.get('frobenius_norm_initial', 0)
    frobenius_final = features.get('frobenius_norm_final', 0)
    
    # Ratio conservation
    if frobenius_initial > 0:
        ratio = frobenius_final / frobenius_initial
    else:
        ratio = np.nan
    
    # Classification
    if np.isnan(ratio) or np.isinf(ratio):
        return 'PATHOLOGICAL'
    
    # Conservation norme
    if thresholds['conservation']['min_ratio'] <= ratio <= thresholds['conservation']['max_ratio']:
        return 'CONSERVES_NORM'
    
    # Décroissance lente
    elif thresholds['slight_decay']['min_ratio'] <= ratio < thresholds['conservation']['min_ratio']:
        return 'SLIGHT_DECAY'
    
    # Décroissance forte
    elif ratio < thresholds['slight_decay']['min_ratio']:
        # Vérifier collapse
        collapse_detected = features.get('collapse_detected', 0)
        if collapse_detected:
            return 'COLLAPSE'
        else:
            return 'STRONG_DECAY'
    
    # Explosion norme
    elif ratio > thresholds['conservation']['max_ratio']:
        return 'PATHOLOGICAL'
    
    # Instabilité numérique
    instability_detected = features.get('instability_detected', 0)
    if instability_detected:
        instability_magnitude = features.get('instability_magnitude', 0)
        if instability_magnitude > thresholds['instability']['magnitude_threshold']:
            return 'NUMERIC_INSTABILITY'
    
    # Par défaut (fallback)
    return 'UNKNOWN'
```

### 7.3 Configuration seuils régimes

```yaml
# configs/thresholds/default/regimes.yaml

conservation:
  min_ratio: 0.95
  max_ratio: 1.05

slight_decay:
  min_ratio: 0.80
  max_ratio: 0.95

strong_decay:
  min_ratio: 0.0
  max_ratio: 0.80

pathological:
  max_ratio: 1.50  # Explosion >50%

instability:
  magnitude_threshold: 0.20
```

---

## 📜 SECTION 8 : TIMELINES DESCRIPTORS

### 8.1 Principe timelines

**Motivation** : Interprétation patterns temporels (séquences événements)

**Descriptors disponibles** :
- `stable` : Pas d'événements
- `early_deviation` : Déviation t<0.2
- `late_saturation` : Saturation t>0.8
- `deviation_then_collapse` : Déviation suivie collapse
- `oscillatory` : Oscillations périodiques
- `unstable` : Instabilité haute fréquence

### 8.2 Implémentation interpretation

```python
# featuring/extractor.py (suite)

def interpret_timeline(features: dict, events: dict, config: dict) -> str:
    """
    Interprète timeline depuis événements détectés.
    
    Args:
        features: Features scalaires
        events: Dynamic events détectés
        config: Configuration
    
    Returns:
        str: Timeline descriptor
    """
    # Extraction événements
    deviation_detected = events.get('deviation_detected', 0)
    deviation_time = events.get('deviation_time', None)
    saturation_detected = events.get('saturation_detected', 0)
    saturation_time = events.get('saturation_time', None)
    collapse_detected = events.get('collapse_detected', 0)
    instability_detected = events.get('instability_detected', 0)
    oscillation_detected = events.get('oscillation_detected', 0)
    
    # Patterns composés
    if deviation_detected and collapse_detected:
        return 'deviation_then_collapse'
    
    if deviation_detected and saturation_detected:
        if deviation_time < 0.3 and saturation_time > 0.7:
            return 'early_deviation_late_saturation'
        else:
            return 'deviation_then_saturation'
    
    # Patterns simples
    if oscillation_detected:
        return 'oscillatory'
    
    if instability_detected:
        return 'unstable'
    
    if deviation_detected:
        if deviation_time < 0.2:
            return 'early_deviation'
        elif deviation_time > 0.8:
            return 'late_deviation'
        else:
            return 'mid_deviation'
    
    if saturation_detected:
        if saturation_time < 0.2:
            return 'early_saturation'
        elif saturation_time > 0.8:
            return 'late_saturation'
        else:
            return 'mid_saturation'
    
    if collapse_detected:
        return 'collapse'
    
    # Stable (pas d'événements)
    return 'stable'
```

---

## 🔧 SECTION 9 : REGISTRES (APERÇU)

### 9.1 Architecture registres

**11 registres** : Fonctions extraction pures (state → float)

```
registries/
├── algebra_registry.py      # Normes, trace, déterminant, condition
├── graph_registry.py         # Densité, clustering, centralité
├── spectral_registry.py      # Eigenvalues, spectral_gap, FFT
├── spatial_registry.py       # Gradient, laplacien, correlation
├── topological_registry.py   # Connected components, Betti numbers
├── pattern_registry.py       # Symétrie, sparsity, block structure
├── statistical_registry.py   # Moments, kurtosis, skewness
├── entropy_registry.py       # Shannon, Rényi, von Neumann
├── tensor_registry.py        # Tucker, CP decomposition
├── timeseries_registry.py    # Autocorr, FFT temporel
└── (spatial déjà listé)
```

**Détails complets** : Voir `ANNEXES/CODE_FEATURING.md`

### 9.2 Pattern registre

```python
# featuring/registries/algebra_registry.py

from .base_registry import BaseRegistry

class AlgebraRegistry(BaseRegistry):
    """Registre fonctions algébriques."""
    
    def __init__(self):
        super().__init__()
        self.layer = 'universal'  # Applicable tout tenseur
    
    @BaseRegistry.register_function("frobenius_norm")
    def compute_frobenius_norm(self, state, **kwargs):
        """Norme Frobenius tenseur."""
        return float(np.linalg.norm(state, 'fro'))
    
    @BaseRegistry.register_function("trace", layer="matrix_square")
    def compute_trace(self, state, **kwargs):
        """Trace matrice carrée."""
        return float(np.trace(state))
    
    @BaseRegistry.register_function("condition_number", layer="matrix_square")
    def compute_condition_number(self, state, epsilon=1e-10, **kwargs):
        """Conditionnement matrice."""
        try:
            cond = np.linalg.cond(state)
            if np.isinf(cond) or np.isnan(cond):
                return float(1e10)  # Sentinelle singularité
            return float(cond)
        except np.linalg.LinAlgError:
            return float(1e10)  # Explosion physique
```

---

## ✅ SECTION 10 : CHECKLIST FEATURING

### 10.1 Avant lancer batch runner

**Vérifications config** :
- [ ] Layers activés YAML (`configs/features/layers.yaml`)
- [ ] Registres configurés (`configs/features/*.yaml`)
- [ ] Seuils events définis (`configs/thresholds/events.yaml`)
- [ ] Seuils régimes définis (`configs/thresholds/regimes.yaml`)

**Vérifications code** :
- [ ] Registres fonctions testés unitairement
- [ ] Projections temporelles cohérentes
- [ ] Dynamic events seuils calibrés

### 10.2 Après batch runner

**Vérifications features** :
- [ ] Nombre features attendu (~150)
- [ ] Pas NaN anormaux (>50% observations)
- [ ] Régimes distribution cohérente (pas 100% PATHOLOGICAL)
- [ ] Timelines variété descriptors (pas tout 'stable')

---

## 📝 CONCLUSION

**Ce document définit** :
- ✅ Architecture featuring (hub, extractor, layers)
- ✅ Logique layers (inspection directe, dispatch if/elif)
- ✅ Projections temporelles (initial, final, mean, std, min, max, max_deviation)
- ✅ Statistics scalaires (initial, final, mean, std, slope, cv)
- ✅ Dynamic events (deviation, saturation, collapse, instability, oscillation)
- ✅ Régimes (classification intra-run, seuils YAML)
- ✅ Timelines descriptors (interprétation patterns événements)
- ✅ Registres (11 modules, détails ANNEXES)

**Principe clé** : 80% calculs totaux intra-run (profiter history RAM complète)

**Prochaine étape** : Lire `04_PROFILING.md` (aggregation inter-run)

---

**FIN 03_FEATURING.md**
# 04 - PROFILING (AGGREGATION INTER-RUN)

**Date** : 16 février 2026  
**Version** : 1.0  
**Rôle** : Aggregation inter-run (observations DB → profils)

---

## 🎯 OBJECTIF DE CE DOCUMENT

**Ce document définit** :
- Principe profiling (inter-run, contexte cross-runs)
- Aggregation cross-runs (median, IQR, bimodal detection)
- Distributions régimes (fréquences conservation/pathologie)
- Timelines frequency (patterns dominants cross-runs)
- Profils gammas/encodings/modifiers

**Prérequis** : Avoir lu `03_FEATURING.md` (features intra-run), `02_DB_SCHEMA.md` (schema DB)

---

## 🎯 SECTION 1 : PRINCIPE PROFILING

### 1.1 Rôle profiling

**Définition** : Aggregation inter-run = calculs cross-runs (contexte observations multiples)

**Timing** : Post-batch, verdict
```
DB observations (phase)
  ↓
Profiling → profils gammas/encodings
  ↓
Rapport synthesis
```

**Principe** : **20% calculs totaux** (contexte cross-runs nécessaire, pas intra-run)

### 1.2 Input : Observations DB

**Source** : DB unique partitionnée (charge partielle colonnes)

```python
# Exemple load observations phase R1
df = load_observations(
    phase='R1',
    columns=[
        'exec_id', 'gamma_id', 'd_encoding_id',
        'frobenius_norm_final', 'density_mean',
        'regime', 'timeline_descriptor'
    ]
)

# DataFrame pandas
#   exec_id | gamma_id | d_encoding_id | frobenius_norm_final | regime | timeline_descriptor
#   --------|----------|---------------|----------------------|--------|--------------------
#   r1_001  | GAM-001  | SYM-001       | 8.234                | CONSERVES_NORM | stable
#   r1_002  | GAM-001  | SYM-002       | 8.156                | CONSERVES_NORM | stable
#   ...
```

### 1.3 Output : Profils

**Format** :
```python
profil_gamma = {
    'gamma_id': 'GAM-001',
    'n_observations': 1200,
    
    # Aggregation features
    'frobenius_norm_final': {
        'median': 8.189,
        'iqr': 0.234,
        'q25': 8.056,
        'q75': 8.290,
        'bimodal': False
    },
    
    # Distribution régimes
    'regimes': {
        'CONSERVES_NORM': 0.85,      # 85% observations
        'SLIGHT_DECAY': 0.10,         # 10%
        'PATHOLOGICAL': 0.05          # 5%
    },
    
    # Timelines frequency
    'timelines': {
        'stable': 0.70,               # 70%
        'early_deviation': 0.15,      # 15%
        'deviation_then_saturation': 0.10,  # 10%
        'unstable': 0.05              # 5%
    }
}
```

---

## 📊 SECTION 2 : AGGREGATION CROSS-RUNS

### 2.1 Principe aggregation

**Motivation** : Caractériser comportement typique gamma/encoding (robuste outliers)

**Métriques aggregation** :
- `median` : Valeur médiane (robuste outliers)
- `iqr` : Intervalle interquartile (dispersion)
- `q25`, `q75` : Quartiles 25%, 75%
- `bimodal` : Détection distribution bimodale

**Choix median vs mean** : Median robuste outliers (explosions physiques)

### 2.2 Implémentation aggregation

```python
# profiling/aggregation.py

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

def aggregate_feature(
    observations: pd.DataFrame,
    feature_name: str,
    groupby: str = 'gamma_id'
) -> pd.DataFrame:
    """
    Agrège feature cross-runs.
    
    Args:
        observations: DataFrame observations
        feature_name: Colonne feature à agréger
        groupby: Colonne groupby (gamma_id, d_encoding_id, ...)
    
    Returns:
        DataFrame: {
            groupby: str,
            'median': float,
            'iqr': float,
            'q25': float,
            'q75': float,
            'bimodal': bool
        }
    """
    results = []
    
    for group_value, group_df in observations.groupby(groupby):
        values = group_df[feature_name].dropna()
        
        if len(values) == 0:
            continue
        
        # Aggregation
        median = float(np.median(values))
        q25 = float(np.percentile(values, 25))
        q75 = float(np.percentile(values, 75))
        iqr = q75 - q25
        
        # Détection bimodale
        bimodal = detect_bimodal(values)
        
        results.append({
            groupby: group_value,
            'median': median,
            'iqr': iqr,
            'q25': q25,
            'q75': q75,
            'bimodal': bimodal,
            'n_observations': len(values)
        })
    
    return pd.DataFrame(results)


def detect_bimodal(values: np.ndarray, threshold: float = 0.1) -> bool:
    """
    Détecte distribution bimodale.
    
    Args:
        values: Observations (N,)
        threshold: Seuil détection creux entre modes
    
    Returns:
        bool: True si bimodale
    """
    if len(values) < 20:
        return False  # Pas assez données
    
    try:
        # Kernel Density Estimation
        kde = gaussian_kde(values)
        x_range = np.linspace(values.min(), values.max(), 200)
        density = kde(x_range)
        
        # Détection pics (modes)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(density, prominence=threshold * density.max())
        
        # Bimodal si 2+ pics
        return len(peaks) >= 2
    
    except Exception as e:
        logger.warning(f"Détection bimodale échouée : {e}")
        return False


def aggregate_all_features(
    observations: pd.DataFrame,
    features_list: list[str],
    groupby: str = 'gamma_id'
) -> dict:
    """
    Agrège toutes features cross-runs.
    
    Args:
        observations: DataFrame observations
        features_list: Liste features à agréger
        groupby: Colonne groupby
    
    Returns:
        dict: {
            group_value: {
                feature_name: {
                    'median': float,
                    'iqr': float,
                    ...
                }
            }
        }
    """
    results = {}
    
    for group_value, group_df in observations.groupby(groupby):
        results[group_value] = {}
        
        for feature_name in features_list:
            if feature_name not in group_df.columns:
                continue
            
            values = group_df[feature_name].dropna()
            
            if len(values) == 0:
                continue
            
            results[group_value][feature_name] = {
                'median': float(np.median(values)),
                'iqr': float(np.percentile(values, 75) - np.percentile(values, 25)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75)),
                'bimodal': detect_bimodal(values),
                'n_observations': len(values)
            }
    
    return results
```

### 2.3 Exemple usage aggregation

```python
# Profil gamma GAM-001

observations_r1 = load_observations('R1')

# Aggregation frobenius_norm_final par gamma
agg_frobenius = aggregate_feature(
    observations_r1,
    feature_name='frobenius_norm_final',
    groupby='gamma_id'
)

# Résultat GAM-001
#   gamma_id | median | iqr   | q25   | q75   | bimodal | n_observations
#   GAM-001  | 8.189  | 0.234 | 8.056 | 8.290 | False   | 1200

# Aggregation toutes features
features_list = [
    'frobenius_norm_final',
    'density_mean',
    'eigenvalue_max_final',
    'spectral_gap_mean'
]

profil_gamma = aggregate_all_features(
    observations_r1,
    features_list,
    groupby='gamma_id'
)

# profil_gamma['GAM-001'] = {
#     'frobenius_norm_final': {'median': 8.189, 'iqr': 0.234, ...},
#     'density_mean': {'median': 0.452, 'iqr': 0.056, ...},
#     ...
# }
```

---

## 🏷️ SECTION 3 : DISTRIBUTIONS RÉGIMES

### 3.1 Principe distributions régimes

**Motivation** : Caractériser stabilité gamma (% conservation vs pathologie)

**Régimes** :
- `CONSERVES_NORM` : Conservation norme (stable)
- `SLIGHT_DECAY` : Décroissance lente
- `STRONG_DECAY` : Décroissance forte
- `PATHOLOGICAL` : Explosion/singularité
- `NUMERIC_INSTABILITY` : Oscillations incontrôlées
- `COLLAPSE` : Effondrement brutal

**Calcul** : Fréquences relatives régimes cross-runs

### 3.2 Implémentation distributions régimes

```python
# profiling/regimes.py

import pandas as pd

def compute_regime_distribution(
    observations: pd.DataFrame,
    groupby: str = 'gamma_id'
) -> pd.DataFrame:
    """
    Calcule distribution régimes par groupe.
    
    Args:
        observations: DataFrame observations (colonne 'regime')
        groupby: Colonne groupby
    
    Returns:
        DataFrame: {
            groupby: str,
            'regime': str,
            'frequency': float,
            'count': int
        }
    """
    results = []
    
    for group_value, group_df in observations.groupby(groupby):
        regime_counts = group_df['regime'].value_counts()
        total = len(group_df)
        
        for regime, count in regime_counts.items():
            frequency = count / total
            
            results.append({
                groupby: group_value,
                'regime': regime,
                'frequency': frequency,
                'count': count
            })
    
    return pd.DataFrame(results)


def get_dominant_regime(
    observations: pd.DataFrame,
    groupby: str = 'gamma_id'
) -> pd.DataFrame:
    """
    Identifie régime dominant par groupe.
    
    Args:
        observations: DataFrame observations
        groupby: Colonne groupby
    
    Returns:
        DataFrame: {
            groupby: str,
            'dominant_regime': str,
            'frequency': float
        }
    """
    regime_dist = compute_regime_distribution(observations, groupby)
    
    # Régime dominant (max frequency)
    dominant = regime_dist.loc[regime_dist.groupby(groupby)['frequency'].idxmax()]
    
    return dominant[[groupby, 'regime', 'frequency']].rename(
        columns={'regime': 'dominant_regime'}
    )


def classify_gamma_stability(
    observations: pd.DataFrame,
    gamma_id: str,
    thresholds: dict
) -> str:
    """
    Classifie stabilité gamma depuis distribution régimes.
    
    Args:
        observations: DataFrame observations gamma
        gamma_id: Identifiant gamma
        thresholds: Seuils classification
    
    Returns:
        str: 'STABLE', 'UNSTABLE', 'PATHOLOGICAL'
    """
    regime_counts = observations['regime'].value_counts(normalize=True)
    
    conserves_freq = regime_counts.get('CONSERVES_NORM', 0)
    pathological_freq = regime_counts.get('PATHOLOGICAL', 0)
    instability_freq = regime_counts.get('NUMERIC_INSTABILITY', 0)
    
    # Classification
    if pathological_freq > thresholds['pathological_max']:
        return 'PATHOLOGICAL'
    
    elif conserves_freq >= thresholds['stable_min']:
        return 'STABLE'
    
    elif instability_freq > thresholds['instability_max']:
        return 'UNSTABLE'
    
    else:
        return 'MODERATE'
```

### 3.3 Exemple usage distributions régimes

```python
# Distribution régimes phase R1

observations_r1 = load_observations('R1', columns=['gamma_id', 'regime'])

# Distribution par gamma
regime_dist = compute_regime_distribution(observations_r1, groupby='gamma_id')

# Résultat GAM-001
#   gamma_id | regime            | frequency | count
#   GAM-001  | CONSERVES_NORM    | 0.85      | 1020
#   GAM-001  | SLIGHT_DECAY      | 0.10      | 120
#   GAM-001  | PATHOLOGICAL      | 0.05      | 60

# Régime dominant
dominant = get_dominant_regime(observations_r1, groupby='gamma_id')

#   gamma_id | dominant_regime | frequency
#   GAM-001  | CONSERVES_NORM  | 0.85
#   GAM-002  | PATHOLOGICAL    | 0.60  # Suspect !

# Classification stabilité
thresholds = {
    'stable_min': 0.70,           # 70% conservation minimum
    'pathological_max': 0.10,     # 10% pathologie maximum
    'instability_max': 0.10       # 10% instabilité maximum
}

observations_gam001 = observations_r1[observations_r1['gamma_id'] == 'GAM-001']
stability = classify_gamma_stability(observations_gam001, 'GAM-001', thresholds)
# stability = 'STABLE' (85% conservation)
```

---

## 📜 SECTION 4 : TIMELINES FREQUENCY

### 4.1 Principe timelines frequency

**Motivation** : Identifier patterns temporels dominants cross-runs

**Timelines** :
- `stable` : Pas d'événements
- `early_deviation` : Déviation précoce
- `late_saturation` : Saturation tardive
- `deviation_then_collapse` : Séquence déviation → collapse
- `oscillatory` : Oscillations périodiques
- `unstable` : Instabilité haute fréquence

**Calcul** : Fréquences relatives timelines cross-runs

### 4.2 Implémentation timelines frequency

```python
# profiling/timelines.py

import pandas as pd

def compute_timeline_frequency(
    observations: pd.DataFrame,
    groupby: str = 'gamma_id'
) -> pd.DataFrame:
    """
    Calcule fréquences timelines par groupe.
    
    Args:
        observations: DataFrame observations (colonne 'timeline_descriptor')
        groupby: Colonne groupby
    
    Returns:
        DataFrame: {
            groupby: str,
            'timeline': str,
            'frequency': float,
            'count': int
        }
    """
    results = []
    
    for group_value, group_df in observations.groupby(groupby):
        timeline_counts = group_df['timeline_descriptor'].value_counts()
        total = len(group_df)
        
        for timeline, count in timeline_counts.items():
            frequency = count / total
            
            results.append({
                groupby: group_value,
                'timeline': timeline,
                'frequency': frequency,
                'count': count
            })
    
    return pd.DataFrame(results)


def get_dominant_timeline(
    observations: pd.DataFrame,
    groupby: str = 'gamma_id'
) -> pd.DataFrame:
    """
    Identifie timeline dominant par groupe.
    
    Args:
        observations: DataFrame observations
        groupby: Colonne groupby
    
    Returns:
        DataFrame: {
            groupby: str,
            'dominant_timeline': str,
            'frequency': float
        }
    """
    timeline_freq = compute_timeline_frequency(observations, groupby)
    
    # Timeline dominant (max frequency)
    dominant = timeline_freq.loc[timeline_freq.groupby(groupby)['frequency'].idxmax()]
    
    return dominant[[groupby, 'timeline', 'frequency']].rename(
        columns={'timeline': 'dominant_timeline'}
    )


def compare_timelines_cross_phases(
    observations_r0: pd.DataFrame,
    observations_r1: pd.DataFrame,
    groupby: str = 'gamma_id'
) -> pd.DataFrame:
    """
    Compare timelines dominants R0 vs R1.
    
    Args:
        observations_r0: Observations phase R0
        observations_r1: Observations phase R1
        groupby: Colonne groupby
    
    Returns:
        DataFrame: {
            groupby: str,
            'timeline_r0': str,
            'timeline_r1': str,
            'concordance': bool
        }
    """
    dominant_r0 = get_dominant_timeline(observations_r0, groupby)
    dominant_r1 = get_dominant_timeline(observations_r1, groupby)
    
    # Merge
    comparison = pd.merge(
        dominant_r0[[groupby, 'dominant_timeline']].rename(columns={'dominant_timeline': 'timeline_r0'}),
        dominant_r1[[groupby, 'dominant_timeline']].rename(columns={'dominant_timeline': 'timeline_r1'}),
        on=groupby,
        how='inner'
    )
    
    # Concordance
    comparison['concordance'] = comparison['timeline_r0'] == comparison['timeline_r1']
    
    return comparison
```

### 4.3 Exemple usage timelines frequency

```python
# Timelines frequency phase R1

observations_r1 = load_observations('R1', columns=['gamma_id', 'timeline_descriptor'])

# Fréquences par gamma
timeline_freq = compute_timeline_frequency(observations_r1, groupby='gamma_id')

# Résultat GAM-001
#   gamma_id | timeline                    | frequency | count
#   GAM-001  | stable                      | 0.70      | 840
#   GAM-001  | early_deviation             | 0.15      | 180
#   GAM-001  | deviation_then_saturation   | 0.10      | 120
#   GAM-001  | unstable                    | 0.05      | 60

# Timeline dominant
dominant = get_dominant_timeline(observations_r1, groupby='gamma_id')

#   gamma_id | dominant_timeline | frequency
#   GAM-001  | stable            | 0.70
#   GAM-005  | unstable          | 0.55  # Suspect !

# Comparaison cross-phases
observations_r0 = load_observations('R0', columns=['gamma_id', 'timeline_descriptor'])

comparison = compare_timelines_cross_phases(observations_r0, observations_r1, groupby='gamma_id')

#   gamma_id | timeline_r0 | timeline_r1 | concordance
#   GAM-001  | stable      | stable      | True
#   GAM-003  | stable      | unstable    | False  # Dégradation R0→R1 !
```

---

## 📊 SECTION 5 : PROFILS GAMMAS/ENCODINGS

### 5.1 Profil gamma complet

```python
# profiling/hub.py

def generate_gamma_profile(
    observations: pd.DataFrame,
    gamma_id: str,
    config: dict
) -> dict:
    """
    Génère profil complet gamma.
    
    Args:
        observations: DataFrame observations gamma
        gamma_id: Identifiant gamma
        config: Configuration seuils
    
    Returns:
        dict: Profil gamma
    """
    # Features critiques
    features_list = [
        'frobenius_norm_final',
        'density_mean',
        'eigenvalue_max_final',
        'spectral_gap_mean',
        'condition_number_mean'
    ]
    
    # Aggregation features
    features_agg = {}
    for feature_name in features_list:
        values = observations[feature_name].dropna()
        if len(values) > 0:
            features_agg[feature_name] = {
                'median': float(np.median(values)),
                'iqr': float(np.percentile(values, 75) - np.percentile(values, 25)),
                'bimodal': detect_bimodal(values)
            }
    
    # Distribution régimes
    regime_counts = observations['regime'].value_counts(normalize=True)
    regimes_dist = regime_counts.to_dict()
    
    # Timelines frequency
    timeline_counts = observations['timeline_descriptor'].value_counts(normalize=True)
    timelines_freq = timeline_counts.to_dict()
    
    # Classification stabilité
    stability = classify_gamma_stability(observations, gamma_id, config['thresholds']['stability'])
    
    # Profil complet
    return {
        'gamma_id': gamma_id,
        'n_observations': len(observations),
        'features': features_agg,
        'regimes': regimes_dist,
        'timelines': timelines_freq,
        'stability': stability,
        'dominant_regime': regime_counts.idxmax(),
        'dominant_timeline': timeline_counts.idxmax()
    }
```

### 5.2 Exemple profil gamma

```python
observations_r1 = load_observations('R1')
observations_gam001 = observations_r1[observations_r1['gamma_id'] == 'GAM-001']

profil_gam001 = generate_gamma_profile(observations_gam001, 'GAM-001', config)

# Résultat
{
    'gamma_id': 'GAM-001',
    'n_observations': 1200,
    
    'features': {
        'frobenius_norm_final': {
            'median': 8.189,
            'iqr': 0.234,
            'bimodal': False
        },
        'density_mean': {
            'median': 0.452,
            'iqr': 0.056,
            'bimodal': False
        },
        ...
    },
    
    'regimes': {
        'CONSERVES_NORM': 0.85,
        'SLIGHT_DECAY': 0.10,
        'PATHOLOGICAL': 0.05
    },
    
    'timelines': {
        'stable': 0.70,
        'early_deviation': 0.15,
        'deviation_then_saturation': 0.10,
        'unstable': 0.05
    },
    
    'stability': 'STABLE',
    'dominant_regime': 'CONSERVES_NORM',
    'dominant_timeline': 'stable'
}
```

### 5.3 Profils encodings/modifiers

**Même principe** : Aggregation par `d_encoding_id` ou `modifier_id`

```python
# Profil encoding SYM-001
observations_sym001 = observations_r1[observations_r1['d_encoding_id'] == 'SYM-001']
profil_sym001 = generate_encoding_profile(observations_sym001, 'SYM-001', config)

# Profil modifier M1
observations_m1 = observations_r1[observations_r1['modifier_id'] == 'M1']
profil_m1 = generate_modifier_profile(observations_m1, 'M1', config)
```

---

## 🔍 SECTION 6 : COMPARAISONS CROSS-RUNS

### 6.1 Comparaison gammas

```python
# profiling/hub.py

def compare_gammas(
    observations: pd.DataFrame,
    gamma_ids: list[str],
    feature_name: str
) -> pd.DataFrame:
    """
    Compare feature entre gammas.
    
    Args:
        observations: DataFrame observations
        gamma_ids: Liste gammas à comparer
        feature_name: Feature à comparer
    
    Returns:
        DataFrame: {
            'gamma_id': str,
            'median': float,
            'iqr': float,
            'rank': int
        }
    """
    results = []
    
    for gamma_id in gamma_ids:
        obs_gamma = observations[observations['gamma_id'] == gamma_id]
        values = obs_gamma[feature_name].dropna()
        
        if len(values) > 0:
            results.append({
                'gamma_id': gamma_id,
                'median': float(np.median(values)),
                'iqr': float(np.percentile(values, 75) - np.percentile(values, 25)),
                'n_observations': len(values)
            })
    
    df = pd.DataFrame(results)
    
    # Ranking (median décroissant)
    df['rank'] = df['median'].rank(ascending=False, method='dense').astype(int)
    
    return df.sort_values('rank')
```

### 6.2 Exemple comparaison gammas

```python
# Comparer conservation norme gammas

observations_r1 = load_observations('R1')
gamma_ids = ['GAM-001', 'GAM-002', 'GAM-003', 'GAM-004', 'GAM-005']

comparison = compare_gammas(
    observations_r1,
    gamma_ids,
    feature_name='frobenius_norm_final'
)

# Résultat
#   gamma_id | median | iqr   | n_observations | rank
#   GAM-001  | 8.189  | 0.234 | 1200           | 1  (meilleur conservation)
#   GAM-003  | 8.145  | 0.312 | 1150           | 2
#   GAM-002  | 7.456  | 0.567 | 1180           | 3
#   GAM-005  | 6.234  | 1.234 | 1100           | 4
#   GAM-004  | 3.456  | 2.345 | 980            | 5  (pire conservation)
```

---

## ✅ SECTION 7 : CHECKLIST PROFILING

### 7.1 Avant lancer verdict

**Vérifications DB** :
- [ ] Observations phase chargées (colonnes nécessaires)
- [ ] Features disponibles (pas 100% NaN)
- [ ] Régimes variété (pas mono-régime)
- [ ] Timelines variété (pas mono-timeline)

**Vérifications config** :
- [ ] Seuils stabilité définis (`configs/thresholds/stability.yaml`)
- [ ] Features critiques listées
- [ ] Groupby cohérent (gamma_id, d_encoding_id, ...)

### 7.2 Après profiling

**Vérifications profils** :
- [ ] Tous gammas profilés (pas manquants)
- [ ] Aggregations cohérentes (médians raisonnables)
- [ ] Distributions régimes somme = 1.0
- [ ] Timelines frequency somme = 1.0

**Vérifications stabilité** :
- [ ] Classification stabilité cohérente (pas 100% PATHOLOGICAL)
- [ ] Gammas STABLE identifiés (candidats R2)
- [ ] Gammas UNSTABLE identifiés (élimination ou mini-phases)

---

## 📝 CONCLUSION

**Ce document définit** :
- ✅ Principe profiling (inter-run, 20% calculs)
- ✅ Aggregation cross-runs (median, IQR, bimodal)
- ✅ Distributions régimes (fréquences conservation/pathologie)
- ✅ Timelines frequency (patterns dominants)
- ✅ Comparaisons cross-runs (gammas, encodings)
- ✅ Profils complets (features + régimes + timelines + stabilité)

**Principe clé** : Contexte cross-runs nécessaire (pas calculable intra-run)

**Prochaine étape** : Lire `05_ANALYSING.md` (patterns ML)

---

**FIN 04_PROFILING.md**
# 05 - ANALYSING (PATTERNS ML)

**Date** : 16 février 2026  
**Version** : 1.0  
**Rôle** : Patterns ML (observations DB → insights)

---

## 🎯 OBJECTIF DE CE DOCUMENT

**Ce document définit** :
- Principe analysing (patterns ML inter-run)
- Clustering (HDBSCAN compositions similaires)
- Outliers (IsolationForest compositions anormales)
- Variance (η², ANOVA importance axes)
- Concordance cross-phases (kappa régimes, DTW timelines, trajectoires R0→R1)

**Prérequis** : Avoir lu `04_PROFILING.md` (aggregation), `03_FEATURING.md` (features)

---

## 🎯 SECTION 1 : PRINCIPE ANALYSING

### 1.1 Rôle analysing

**Définition** : Patterns ML = analyses complexes nécessitant contexte cross-runs

**Timing** : Post-batch, verdict (après profiling)
```
DB observations (phase)
  ↓
Profiling → profils
  ↓
Analysing → insights ML
  ↓
Rapport synthesis
```

**Principe** : **20% calculs totaux** (patterns complexes, contexte cross-runs obligatoire)

### 1.2 Analyses disponibles

**Clustering** :
- HDBSCAN (clusters compositions similaires)
- Identification familles comportements
- Détection structures latentes

**Outliers** :
- IsolationForest (compositions anormales)
- Identification runs suspects
- Validation explosions physiques vs bugs

**Variance** :
- η² (effet taille axes)
- ANOVA (significativité axes)
- Importance relative gamma vs encoding vs modifier

**Concordance cross-phases** :
- Kappa régimes (stabilité R0↔R1)
- DTW timelines (similarité patterns)
- Trajectoires (transitions R0→R1)

---

## 🔍 SECTION 2 : CLUSTERING (HDBSCAN)

### 2.1 Principe clustering

**Motivation** : Identifier familles compositions similaires (au-delà groupby gamma/encoding)

**Algorithme** : HDBSCAN (Hierarchical Density-Based Spatial Clustering)
- Pas besoin spécifier nombre clusters
- Robuste outliers (label -1)
- Détecte clusters formes arbitraires

**Features utilisées** : Subset features critiques (5-10 features)

### 2.2 Implémentation clustering

```python
# analysing/clustering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN

def cluster_observations(
    observations: pd.DataFrame,
    features_list: list[str],
    min_cluster_size: int = 50,
    min_samples: int = 10
) -> pd.DataFrame:
    """
    Cluster observations HDBSCAN.
    
    Args:
        observations: DataFrame observations
        features_list: Features pour clustering (5-10)
        min_cluster_size: Taille minimum cluster
        min_samples: Samples minimum densité
    
    Returns:
        DataFrame: observations + colonne 'cluster_id'
    """
    # Sélection features
    X = observations[features_list].dropna()
    indices = X.index
    
    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Clustering HDBSCAN
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean'
    )
    
    labels = clusterer.fit_predict(X_scaled)
    
    # Ajouter labels
    observations_clustered = observations.copy()
    observations_clustered['cluster_id'] = -2  # Default (pas clustered)
    observations_clustered.loc[indices, 'cluster_id'] = labels
    
    return observations_clustered


def analyze_clusters(
    observations_clustered: pd.DataFrame,
    features_list: list[str]
) -> dict:
    """
    Analyse clusters identifiés.
    
    Args:
        observations_clustered: DataFrame avec cluster_id
        features_list: Features analysées
    
    Returns:
        dict: {
            cluster_id: {
                'n_observations': int,
                'dominant_gamma': str,
                'dominant_regime': str,
                'features_profile': dict
            }
        }
    """
    results = {}
    
    for cluster_id, cluster_df in observations_clustered.groupby('cluster_id'):
        if cluster_id == -1:
            continue  # Outliers HDBSCAN (traités séparément)
        
        # Profil cluster
        profile = {
            'n_observations': len(cluster_df),
            'dominant_gamma': cluster_df['gamma_id'].mode()[0] if len(cluster_df) > 0 else None,
            'dominant_regime': cluster_df['regime'].mode()[0] if 'regime' in cluster_df.columns else None,
            'features_profile': {}
        }
        
        # Features médians cluster
        for feature in features_list:
            if feature in cluster_df.columns:
                profile['features_profile'][feature] = {
                    'median': float(np.median(cluster_df[feature].dropna())),
                    'std': float(np.std(cluster_df[feature].dropna()))
                }
        
        results[int(cluster_id)] = profile
    
    return results
```

### 2.3 Exemple usage clustering

```python
# Clustering observations R1

observations_r1 = load_observations('R1')

# Features critiques clustering
features_list = [
    'frobenius_norm_final',
    'density_mean',
    'eigenvalue_max_final',
    'spectral_gap_mean',
    'condition_number_mean'
]

# Clustering
observations_clustered = cluster_observations(
    observations_r1,
    features_list,
    min_cluster_size=100
)

# Analyse clusters
clusters_profile = analyze_clusters(observations_clustered, features_list)

# Résultat
# Cluster 0 : 1200 obs, dominant GAM-001, CONSERVES_NORM, frobenius ~8.2
# Cluster 1 : 800 obs, dominant GAM-003, SLIGHT_DECAY, frobenius ~7.5
# Cluster 2 : 500 obs, dominant GAM-005, PATHOLOGICAL, frobenius ~15.3
# Outliers (-1) : 200 obs (compositions anormales)
```

---

## ⚠️ SECTION 3 : OUTLIERS (ISOLATIONFOREST)

### 3.1 Principe outliers

**Motivation** : Détecter compositions anormales (explosions physiques ou bugs code)

**Algorithme** : IsolationForest
- Détection anomalies espace features
- Robuste dimensions élevées
- Score outlier [-1, 1]

**Usage** : **DÉTECTION + ANALYSE + RECOMMANDATION** (jamais élimination automatique)

**IMPORTANT - Philosophie outliers** :
- ✅ Pipeline **DÉTECTE** outliers (IsolationForest)
- ✅ Pipeline **ANALYSE** causes (explosions physiques ? bugs ? config ?)
- ✅ Pipeline **RECOMMANDE** actions (mini-phase validation ? reconfiguration ?)
- ❌ Pipeline **N'ÉLIMINE JAMAIS** automatiquement
- ✅ Utilisateur **VALIDE** toute décision élimination (après incertitude levée)

**Exemple workflow** :
```
Outliers détectés → Analyse patterns → Recommendation mini-phase validation
                                    → Utilisateur décide (éliminer/reconfigurer/investiguer)
```

### 3.2 Implémentation outliers

```python
# analysing/outliers.py

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def detect_outliers(
    observations: pd.DataFrame,
    features_list: list[str],
    contamination: float = 0.05
) -> pd.DataFrame:
    """
    Détecte outliers IsolationForest.
    
    Args:
        observations: DataFrame observations
        features_list: Features pour détection
        contamination: Proportion outliers attendus (0.01-0.1)
    
    Returns:
        DataFrame: observations + colonnes 'is_outlier', 'outlier_score'
    """
    # Sélection features
    X = observations[features_list].dropna()
    indices = X.index
    
    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Isolation Forest
    clf = IsolationForest(
        contamination=contamination,
        random_state=42
    )
    
    predictions = clf.fit_predict(X_scaled)
    scores = clf.score_samples(X_scaled)
    
    # Ajouter résultats
    observations_outliers = observations.copy()
    observations_outliers['is_outlier'] = False
    observations_outliers['outlier_score'] = np.nan
    
    observations_outliers.loc[indices, 'is_outlier'] = (predictions == -1)
    observations_outliers.loc[indices, 'outlier_score'] = scores
    
    return observations_outliers


def analyze_outliers(
    observations_outliers: pd.DataFrame
) -> dict:
    """
    Analyse outliers détectés.
    
    Args:
        observations_outliers: DataFrame avec is_outlier
    
    Returns:
        dict: {
            'n_outliers': int,
            'outlier_rate': float,
            'outliers_by_gamma': dict,
            'outliers_by_regime': dict
        }
    """
    outliers = observations_outliers[observations_outliers['is_outlier']]
    
    return {
        'n_outliers': len(outliers),
        'outlier_rate': len(outliers) / len(observations_outliers),
        'outliers_by_gamma': outliers['gamma_id'].value_counts().to_dict(),
        'outliers_by_regime': outliers['regime'].value_counts().to_dict() if 'regime' in outliers.columns else {}
    }
```

### 3.3 Exemple usage outliers

```python
# Détection outliers R1

observations_r1 = load_observations('R1')

features_list = [
    'frobenius_norm_final',
    'density_mean',
    'condition_number_mean'
]

# Détection
observations_outliers = detect_outliers(
    observations_r1,
    features_list,
    contamination=0.05  # 5% outliers attendus
)

# Analyse
outliers_profile = analyze_outliers(observations_outliers)

# Résultat
{
    'n_outliers': 2850,           # 5% de 57,000
    'outlier_rate': 0.05,
    'outliers_by_gamma': {
        'GAM-005': 1200,          # Gamma suspect (42% outliers)
        'GAM-002': 800,
        'GAM-001': 300,
        ...
    },
    'outliers_by_regime': {
        'PATHOLOGICAL': 2000,     # 70% outliers sont pathologiques
        'NUMERIC_INSTABILITY': 500,
        'CONSERVES_NORM': 350     # Outliers "bons" (conservation extrême)
    }
}

# Inspection outliers gamma GAM-005
outliers_gam005 = observations_outliers[
    (observations_outliers['is_outlier']) & 
    (observations_outliers['gamma_id'] == 'GAM-005')
]

# Analyse causes
if len(outliers_gam005) > 0.3 * len(observations_outliers[observations_outliers['gamma_id'] == 'GAM-005']):
    # >30% observations GAM-005 sont outliers
    
    # RECOMMENDATION UTILISATEUR (pas action auto)
    recommendation = {
        'gamma_id': 'GAM-005',
        'outlier_rate': 0.42,
        'severity': 'HIGH',
        'action': 'MINI_PHASE_VALIDATION',
        'justification': '42% observations outliers, possible incompatibilité systématique',
        'next_steps': [
            'Mini-phase 100 runs (variations config/seed)',
            'Confirmer pathologie systématique ou artifact',
            'SI confirmé → Utilisateur valide élimination',
            'SI artifact → Reconfigurer seuils/params'
        ]
    }
    
    # Pipeline RECOMMANDE, utilisateur DÉCIDE
    print(f"⚠️  Recommendation: Mini-phase validation GAM-005")
    print(f"   Outlier rate: {recommendation['outlier_rate']:.0%}")
    print(f"   Action: {recommendation['action']}")
    # → Utilisateur lit rapport, décide next step
```

---

## 📊 SECTION 4 : VARIANCE (η², ANOVA)

### 4.1 Principe variance

**Motivation** : Quantifier importance relative axes (gamma vs encoding vs modifier)

**Métriques** :
- **η²** (eta-squared) : Proportion variance expliquée par axe
- **ANOVA** : Significativité effet axe (p-value)

**Interprétation η²** :
- 0.01 : Petit effet
- 0.06 : Effet moyen
- 0.14+ : Effet large

### 4.2 Implémentation variance

```python
# analysing/variance.py

import pandas as pd
import numpy as np
from scipy.stats import f_oneway

def compute_eta_squared(
    observations: pd.DataFrame,
    feature_name: str,
    factor: str
) -> float:
    """
    Calcule η² (proportion variance expliquée par facteur).
    
    Args:
        observations: DataFrame observations
        feature_name: Feature analysée
        factor: Axe (gamma_id, d_encoding_id, modifier_id)
    
    Returns:
        float: η² [0, 1]
    """
    # Variance totale
    total_variance = observations[feature_name].var()
    
    # Variance between groups
    group_means = observations.groupby(factor)[feature_name].mean()
    group_sizes = observations.groupby(factor)[feature_name].count()
    
    grand_mean = observations[feature_name].mean()
    
    between_variance = np.sum(group_sizes * (group_means - grand_mean) ** 2) / (len(group_means) - 1)
    
    # η²
    eta_squared = between_variance / total_variance if total_variance > 0 else 0
    
    return float(eta_squared)


def compute_anova(
    observations: pd.DataFrame,
    feature_name: str,
    factor: str
) -> dict:
    """
    Calcule ANOVA (significativité effet facteur).
    
    Args:
        observations: DataFrame observations
        feature_name: Feature analysée
        factor: Axe
    
    Returns:
        dict: {
            'f_statistic': float,
            'p_value': float,
            'significant': bool
        }
    """
    # Groupes
    groups = [
        group[feature_name].dropna().values
        for name, group in observations.groupby(factor)
    ]
    
    # ANOVA
    f_stat, p_value = f_oneway(*groups)
    
    return {
        'f_statistic': float(f_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05
    }


def analyze_axes_importance(
    observations: pd.DataFrame,
    features_list: list[str],
    axes: list[str] = ['gamma_id', 'd_encoding_id', 'modifier_id']
) -> pd.DataFrame:
    """
    Analyse importance relative axes.
    
    Args:
        observations: DataFrame observations
        features_list: Features analysées
        axes: Axes à comparer
    
    Returns:
        DataFrame: {
            'feature': str,
            'axe': str,
            'eta_squared': float,
            'p_value': float,
            'rank': int
        }
    """
    results = []
    
    for feature in features_list:
        for axe in axes:
            eta2 = compute_eta_squared(observations, feature, axe)
            anova = compute_anova(observations, feature, axe)
            
            results.append({
                'feature': feature,
                'axe': axe,
                'eta_squared': eta2,
                'p_value': anova['p_value'],
                'significant': anova['significant']
            })
    
    df = pd.DataFrame(results)
    
    # Ranking η² par feature
    df['rank'] = df.groupby('feature')['eta_squared'].rank(ascending=False, method='dense').astype(int)
    
    return df.sort_values(['feature', 'rank'])
```

### 4.3 Exemple usage variance

```python
# Analyse importance axes R1

observations_r1 = load_observations('R1')

features_list = [
    'frobenius_norm_final',
    'density_mean',
    'eigenvalue_max_final'
]

axes_importance = analyze_axes_importance(
    observations_r1,
    features_list,
    axes=['gamma_id', 'd_encoding_id', 'modifier_id']
)

# Résultat
#   feature              | axe              | eta_squared | p_value  | significant | rank
#   frobenius_norm_final | gamma_id         | 0.45        | <0.001   | True        | 1  (effet large)
#   frobenius_norm_final | d_encoding_id    | 0.12        | <0.001   | True        | 2  (effet moyen)
#   frobenius_norm_final | modifier_id      | 0.02        | 0.03     | True        | 3  (effet petit)
#   
#   density_mean         | gamma_id         | 0.38        | <0.001   | True        | 1
#   density_mean         | d_encoding_id    | 0.25        | <0.001   | True        | 2
#   density_mean         | modifier_id      | 0.01        | 0.15     | False       | 3

# Interprétation :
# - Gamma dominant (η² ~0.4) pour conservation norme et densité
# - Encoding effet moyen (η² ~0.15)
# - Modifier effet faible (η² <0.05)
```

---

## 🔄 SECTION 5 : CONCORDANCE CROSS-PHASES

### 5.1 Principe concordance

**Motivation** : Évaluer stabilité comportements R0→R1 (prédictibilité R2+)

**Métriques** :
- **Kappa régimes** : Agreement classification régimes R0 vs R1
- **DTW timelines** : Similarité patterns temporels
- **Trajectoires** : Transitions R0→R1 (conservation → pathologie ?)

### 5.2 Implémentation concordance

```python
# analysing/concordance.py

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def compute_kappa_regimes(
    observations_r0: pd.DataFrame,
    observations_r1: pd.DataFrame,
    groupby: str = 'gamma_id'
) -> pd.DataFrame:
    """
    Calcule kappa régimes R0 vs R1.
    
    Args:
        observations_r0: Observations phase R0
        observations_r1: Observations phase R1
        groupby: Colonne groupby
    
    Returns:
        DataFrame: {
            groupby: str,
            'kappa': float,
            'concordance': str
        }
    """
    results = []
    
    for group_value in observations_r0[groupby].unique():
        r0_group = observations_r0[observations_r0[groupby] == group_value]
        r1_group = observations_r1[observations_r1[groupby] == group_value]
        
        # Régimes dominants
        regime_r0 = r0_group['regime'].mode()[0] if len(r0_group) > 0 else None
        regime_r1 = r1_group['regime'].mode()[0] if len(r1_group) > 0 else None
        
        if regime_r0 is None or regime_r1 is None:
            continue
        
        # Agreement simple (régimes dominants)
        agreement = (regime_r0 == regime_r1)
        
        # Kappa (si données suffisantes distributions)
        if len(r0_group) > 10 and len(r1_group) > 10:
            # Sample équilibré pour kappa
            min_size = min(len(r0_group), len(r1_group), 100)
            r0_sample = r0_group['regime'].sample(min_size, random_state=42)
            r1_sample = r1_group['regime'].sample(min_size, random_state=42)
            
            kappa = cohen_kappa_score(r0_sample, r1_sample)
        else:
            kappa = 1.0 if agreement else 0.0
        
        # Classification concordance
        if kappa > 0.8:
            concordance = 'STRONG'
        elif kappa > 0.5:
            concordance = 'MODERATE'
        else:
            concordance = 'WEAK'
        
        results.append({
            groupby: group_value,
            'regime_r0': regime_r0,
            'regime_r1': regime_r1,
            'kappa': float(kappa),
            'concordance': concordance
        })
    
    return pd.DataFrame(results)


def analyze_trajectories(
    observations_r0: pd.DataFrame,
    observations_r1: pd.DataFrame,
    groupby: str = 'gamma_id'
) -> pd.DataFrame:
    """
    Analyse trajectoires régimes R0→R1.
    
    Args:
        observations_r0: Observations R0
        observations_r1: Observations R1
        groupby: Colonne groupby
    
    Returns:
        DataFrame: Matrice transitions
    """
    results = []
    
    for group_value in observations_r0[groupby].unique():
        r0_group = observations_r0[observations_r0[groupby] == group_value]
        r1_group = observations_r1[observations_r1[groupby] == group_value]
        
        regime_r0 = r0_group['regime'].mode()[0] if len(r0_group) > 0 else None
        regime_r1 = r1_group['regime'].mode()[0] if len(r1_group) > 0 else None
        
        if regime_r0 and regime_r1:
            results.append({
                groupby: group_value,
                'regime_r0': regime_r0,
                'regime_r1': regime_r1,
                'transition': f"{regime_r0} → {regime_r1}"
            })
    
    df = pd.DataFrame(results)
    
    # Matrice transitions
    transition_matrix = pd.crosstab(df['regime_r0'], df['regime_r1'])
    
    return transition_matrix


def compute_dtw_timelines(
    timeline_r0: np.ndarray,
    timeline_r1: np.ndarray
) -> float:
    """
    Calcule DTW (Dynamic Time Warping) entre timelines.
    
    Args:
        timeline_r0: Timeline R0 (T,)
        timeline_r1: Timeline R1 (T,)
    
    Returns:
        float: Distance DTW (plus faible = plus similaire)
    """
    distance, _ = fastdtw(timeline_r0, timeline_r1, dist=euclidean)
    return float(distance)
```

### 5.3 Exemple usage concordance

```python
# Concordance R0↔R1

observations_r0 = load_observations('R0', columns=['gamma_id', 'regime'])
observations_r1 = load_observations('R1', columns=['gamma_id', 'regime'])

# Kappa régimes
kappa_results = compute_kappa_regimes(observations_r0, observations_r1, groupby='gamma_id')

# Résultat
#   gamma_id | regime_r0      | regime_r1      | kappa | concordance
#   GAM-001  | CONSERVES_NORM | CONSERVES_NORM | 0.92  | STRONG
#   GAM-003  | CONSERVES_NORM | SLIGHT_DECAY   | 0.65  | MODERATE
#   GAM-005  | PATHOLOGICAL   | PATHOLOGICAL   | 0.88  | STRONG
#   GAM-007  | CONSERVES_NORM | PATHOLOGICAL   | 0.15  | WEAK  # Dégradation !

# Trajectoires
transition_matrix = analyze_trajectories(observations_r0, observations_r1, groupby='gamma_id')

# Matrice transitions
#                      | CONSERVES_NORM | SLIGHT_DECAY | PATHOLOGICAL
# CONSERVES_NORM       | 8              | 2            | 1  # Stabilité 8/11
# SLIGHT_DECAY         | 0              | 1            | 1
# PATHOLOGICAL         | 0              | 0            | 2

# Interprétation :
# - Gammas stables (CONSERVES → CONSERVES) : Candidats R2
# - Dégradations (CONSERVES → PATHOLOGICAL) : Éliminer ou mini-phases
```

---

## ✅ SECTION 6 : CHECKLIST ANALYSING

### 6.1 Avant lancer verdict

**Vérifications profiling** :
- [ ] Profiling terminé (profils gammas/encodings disponibles)
- [ ] Observations R0 et R1 chargées (concordance cross-phases)
- [ ] Features critiques sélectionnées (5-10 features)

**Vérifications config** :
- [ ] Paramètres clustering définis (min_cluster_size)
- [ ] Contamination outliers calibrée (0.01-0.1)
- [ ] Axes variance listés (gamma, encoding, modifier)

### 6.2 Après analysing

**Vérifications résultats** :
- [ ] Clusters identifiés (2-5 clusters typiques)
- [ ] Outliers cohérents (<10% observations)
- [ ] Variance η² cohérente (gamma dominant)
- [ ] Concordance R0↔R1 évaluée (kappa >0.7 souhaité)

**Actions verdict** :
- [ ] Gammas stables identifiés (candidats R2)
- [ ] Gammas instables identifiés (élimination ou mini-phases)
- [ ] Outliers analysés (explosions physiques vs bugs)

---

## 📝 CONCLUSION

**Ce document définit** :
- ✅ Principe analysing (patterns ML inter-run)
- ✅ Clustering HDBSCAN (familles compositions similaires)
- ✅ Outliers IsolationForest (compositions anormales)
- ✅ Variance η² & ANOVA (importance axes gamma vs encoding)
- ✅ Concordance cross-phases (kappa régimes, trajectoires R0→R1)

**Principe clé** : Patterns ML complexes nécessitant contexte cross-runs

**Prochaine étape** : Lire `06_PIPELINE.md` (batch runner + verdict)

---

**FIN 05_ANALYSING.md**
# 06 - PIPELINE (BATCH RUNNER & VERDICT)

**Date** : 16 février 2026  
**Version** : 1.0  
**Rôle** : Orchestration pipeline (batch runner + verdict)

---

## 🎯 OBJECTIF DE CE DOCUMENT

**Ce document définit** :
- Batch runner (génération compositions, dry-run, loop runs)
- Verdict (profiling + analysing + rapport)
- Generate compositions (axes YAML configurables)
- Dry-run estimation (temps/RAM/DB)
- Gestion erreurs pipeline

**Prérequis** : Avoir lu `03_FEATURING.md`, `04_PROFILING.md`, `05_ANALYSING.md`, `07_AXES_COMPOSITION.md`

---

## 🎯 SECTION 1 : ARCHITECTURE PIPELINE

### 1.1 Points entrée

**Batch runner** : `batch_runner.py`
```bash
$ python batch_runner.py --phase r0 --config-set default
```

**Verdict** : `verdict.py`
```bash
$ python verdict.py --phase r0 --config-set default
```

### 1.2 Flux complet

```
┌─────────────────────────────────────┐
│     BATCH RUNNER                    │
├─────────────────────────────────────┤
│ 1. Load config YAML                 │
│ 2. Generate compositions            │
│ 3. Dry-run estimation               │
│ 4. Confirmation utilisateur (o/n)   │
│ 5. Prepare DB (migrations)          │
│ 6. Loop runs:                       │
│    - Kernel → history               │
│    - Featuring → features           │
│    - DB insert                      │
│ 7. Logs progression                 │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│          VERDICT                    │
├─────────────────────────────────────┤
│ 1. Load observations DB             │
│ 2. Profiling:                       │
│    - Aggregation features           │
│    - Distributions régimes          │
│    - Timelines frequency            │
│ 3. Analysing:                       │
│    - Clustering                     │
│    - Outliers                       │
│    - Variance                       │
│    - Concordance cross-phases       │
│ 4. Rapport synthesis                │
│ 5. Recommendations                  │
└─────────────────────────────────────┘
```

---

## 🔄 SECTION 2 : BATCH RUNNER

### 2.1 Structure batch_runner.py

```python
# batch_runner.py

import argparse
import logging
import time
from pathlib import Path

from prc.core.kernel import run_kernel
from prc.featuring.hub import extract_features_ml
from prc.utils.database import insert_observation
from prc.utils.migrate_db import add_feature_column, add_axe_column
from prc.utils.config import load_config
from prc.pipeline.compositions import generate_compositions
from prc.pipeline.discovery import discover_atomics
from prc.pipeline.dry_run import estimate_batch

logger = logging.getLogger(__name__)


def main():
    """Point entrée batch runner."""
    # 1. Parse arguments
    parser = argparse.ArgumentParser(description='PRC Batch Runner')
    parser.add_argument('--phase', type=str, required=True, help='Phase (r0, r1, r2, ...)')
    parser.add_argument('--config-set', type=str, default='default', help='Config set (default, laxe, strict)')
    parser.add_argument('--no-confirm', action='store_true', help='Skip confirmation (auto yes)')
    args = parser.parse_args()
    
    # 2. Load config
    config = load_config(args.phase, args.config_set)
    
    logger.info(f"Starting batch runner - Phase: {args.phase}, Config: {args.config_set}")
    
    # 3. Discovery atomics
    discovery = discover_atomics(config)
    logger.info(f"Discovery: {len(discovery['gamma_ids'])} gammas, "
                f"{len(discovery['d_encoding_ids'])} encodings, "
                f"{len(discovery['modifier_ids'])} modifiers")
    
    # 4. Generate compositions
    compositions = list(generate_compositions(config, discovery))
    logger.info(f"Generated {len(compositions)} compositions")
    
    # 5. Dry-run estimation
    estimation = estimate_batch(compositions, config)
    
    print("\n🔍 DRY-RUN ESTIMATION")
    print("─" * 50)
    print(f"Compositions : {estimation['n_compositions']} runs")
    print(f"Temps estimé : {estimation['estimated_time_hours']:.1f}h")
    print(f"RAM peak     : {estimation['estimated_ram_mb']:.0f} MB")
    print(f"DB finale    : {estimation['estimated_db_mb']:.0f} MB")
    print("─" * 50)
    
    # 6. Confirmation
    if not args.no_confirm:
        response = input("\n▶ Lancer batch ? (o/n) : ").strip().lower()
        if response != 'o':
            logger.info("Batch annulé par utilisateur")
            return
    
    # 7. Prepare DB (migrations)
    prepare_database(config)
    
    # 8. Run batch
    run_batch(compositions, config, args.phase)
    
    logger.info("Batch runner terminé")


def prepare_database(config: dict):
    """Prépare DB (migrations auto axes/features)."""
    logger.info("Préparation DB...")
    
    # Migrations axes
    for axe_name in config['iteration_axes'].keys():
        if axe_name not in ['gamma_id', 'd_encoding_id', 'modifier_id']:
            axe_type = infer_axe_type(config['iteration_axes'][axe_name])
            add_axe_column(axe_name, axe_type)
    
    # Migrations features
    expected_features = get_expected_features(config)
    for feature_name, feature_type in expected_features.items():
        add_feature_column(feature_name, feature_type)
    
    logger.info("✓ DB préparée")


def run_batch(compositions: list[dict], config: dict, phase: str):
    """Exécute batch runs."""
    total = len(compositions)
    start_time = time.time()
    
    for i, composition in enumerate(compositions):
        # Progress
        if i % 10 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (total - i) / rate if rate > 0 else 0
            logger.info(f"Progress: {i}/{total} ({i/total*100:.1f}%) - "
                       f"ETA: {remaining/3600:.1f}h")
        
        try:
            # Kernel
            history = run_kernel(composition, config)
            
            # Featuring
            features = extract_features_ml(history, config)
            
            # Build observation
            observation = {
                'exec_id': f"{phase}_{i:06d}",
                'phase': phase,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                **composition,  # Axes
                **features      # Features
            }
            
            # DB insert
            insert_observation(observation)
        
        except Exception as e:
            logger.error(f"Run {i} failed: {e}")
            continue
    
    elapsed = time.time() - start_time
    logger.info(f"Batch terminé: {total} runs en {elapsed/3600:.1f}h")


def infer_axe_type(axe_values) -> str:
    """Infère type SQL axe depuis valeurs."""
    if isinstance(axe_values, list) and len(axe_values) > 0:
        sample = axe_values[0]
        if isinstance(sample, int):
            return 'INTEGER'
        elif isinstance(sample, float):
            return 'REAL'
    return 'TEXT'


def get_expected_features(config: dict) -> dict:
    """Retourne features attendues depuis config."""
    # Parse config features YAML → liste features attendues
    # Exemple : frobenius_norm avec projections [initial, final, mean]
    #   → frobenius_norm_initial, frobenius_norm_final, frobenius_norm_mean
    features = {}
    
    for layer in config['features']['layers']:
        if not config['features']['layers'][layer]:
            continue
        
        for feature_config in config['features'].get(layer, []):
            registry_key = feature_config['function']
            base_name = registry_key.replace('.', '_')
            
            projections = feature_config.get('projections', ['final'])
            for proj in projections:
                feature_name = f"{base_name}_{proj}"
                features[feature_name] = 'REAL'
    
    # Dynamic events
    features.update({
        'deviation_detected': 'INTEGER',
        'deviation_time': 'REAL',
        'saturation_detected': 'INTEGER',
        'saturation_time': 'REAL',
        'collapse_detected': 'INTEGER',
        'instability_detected': 'INTEGER',
        'oscillation_detected': 'INTEGER'
    })
    
    # Régimes, timelines
    features['regime'] = 'TEXT'
    features['timeline_descriptor'] = 'TEXT'
    
    return features


if __name__ == '__main__':
    main()
```

### 2.2 Gestion erreurs batch

**Types erreurs** :
1. **Erreur composition** : Composition invalide (ignorée gracefully)
2. **Erreur kernel** : Crash calcul (logged, skip)
3. **Erreur featuring** : Feature calcul échoué (logged, NaN)
4. **Erreur DB** : Insert échoué (logged, retry)

**Stratégie** :
```python
def run_batch_safe(compositions: list[dict], config: dict, phase: str):
    """Run batch avec gestion erreurs robuste."""
    success = 0
    failed = 0
    
    for i, composition in enumerate(compositions):
        try:
            # Run
            observation = run_single(composition, config, phase, i)
            insert_observation(observation)
            success += 1
        
        except KeyboardInterrupt:
            logger.warning("Batch interrompu par utilisateur")
            break
        
        except Exception as e:
            logger.error(f"Run {i} failed: {e}", exc_info=True)
            failed += 1
            
            # Arrêt si taux échec >50%
            if failed / (success + failed) > 0.5 and i > 10:
                logger.critical("Taux échec >50%, arrêt batch")
                raise RuntimeError("Batch failure rate too high")
    
    logger.info(f"Batch terminé: {success} success, {failed} failed")
```

---

## 📊 SECTION 3 : VERDICT

### 3.1 Structure verdict.py

```python
# verdict.py

import argparse
import logging
from pathlib import Path

from prc.utils.database import load_observations
from prc.profiling.hub import run_profiling
from prc.analysing.hub import run_analysing
from prc.utils.config import load_config
from prc.verdict.report import generate_report

logger = logging.getLogger(__name__)


def main():
    """Point entrée verdict."""
    # 1. Parse arguments
    parser = argparse.ArgumentParser(description='PRC Verdict')
    parser.add_argument('--phase', type=str, required=True, help='Phase (r0, r1, r2, ...)')
    parser.add_argument('--config-set', type=str, default='default', help='Config set')
    parser.add_argument('--compare-phase', type=str, help='Phase comparaison concordance (ex: r0)')
    args = parser.parse_args()
    
    # 2. Load config
    config = load_config(args.phase, args.config_set)
    
    logger.info(f"Starting verdict - Phase: {args.phase}")
    
    # 3. Load observations
    logger.info("Loading observations...")
    observations = load_observations(args.phase)
    logger.info(f"Loaded {len(observations)} observations")
    
    # 4. Profiling
    logger.info("Running profiling...")
    profiling_results = run_profiling(observations, config)
    logger.info("✓ Profiling terminé")
    
    # 5. Analysing
    logger.info("Running analysing...")
    analysing_results = run_analysing(observations, config)
    logger.info("✓ Analysing terminé")
    
    # 6. Concordance cross-phases (optionnel)
    concordance_results = None
    if args.compare_phase:
        logger.info(f"Computing concordance {args.compare_phase}↔{args.phase}...")
        observations_compare = load_observations(args.compare_phase)
        concordance_results = compute_concordance(
            observations_compare, observations, config
        )
        logger.info("✓ Concordance terminée")
    
    # 7. Generate rapport
    logger.info("Generating report...")
    report_path = generate_report(
        phase=args.phase,
        observations=observations,
        profiling_results=profiling_results,
        analysing_results=analysing_results,
        concordance_results=concordance_results,
        config=config
    )
    
    logger.info(f"✓ Rapport généré: {report_path}")
    logger.info("Verdict terminé")


if __name__ == '__main__':
    main()
```

### 3.2 Hub profiling

```python
# profiling/hub.py

def run_profiling(observations: pd.DataFrame, config: dict) -> dict:
    """
    Exécute profiling complet.
    
    Returns:
        dict: {
            'profiles_gamma': dict,
            'profiles_encoding': dict,
            'profiles_modifier': dict,
            'regime_distributions': dict,
            'timeline_frequencies': dict
        }
    """
    results = {}
    
    # Profils gammas
    profiles_gamma = {}
    for gamma_id in observations['gamma_id'].unique():
        obs_gamma = observations[observations['gamma_id'] == gamma_id]
        profiles_gamma[gamma_id] = generate_gamma_profile(obs_gamma, gamma_id, config)
    
    results['profiles_gamma'] = profiles_gamma
    
    # Profils encodings
    profiles_encoding = {}
    for encoding_id in observations['d_encoding_id'].unique():
        obs_encoding = observations[observations['d_encoding_id'] == encoding_id]
        profiles_encoding[encoding_id] = generate_encoding_profile(obs_encoding, encoding_id, config)
    
    results['profiles_encoding'] = profiles_encoding
    
    # Distributions régimes
    results['regime_distributions'] = compute_regime_distribution(observations, groupby='gamma_id')
    
    # Timelines frequency
    results['timeline_frequencies'] = compute_timeline_frequency(observations, groupby='gamma_id')
    
    return results
```

### 3.3 Hub analysing

```python
# analysing/hub.py

def run_analysing(observations: pd.DataFrame, config: dict) -> dict:
    """
    Exécute analysing complet.
    
    Returns:
        dict: {
            'clusters': dict,
            'outliers': dict,
            'variance': dict,
            'concordance': dict (si compare_phase fourni)
        }
    """
    results = {}
    
    # Features critiques
    features_list = config['analysing']['features_critical']
    
    # Clustering
    observations_clustered = cluster_observations(
        observations,
        features_list,
        min_cluster_size=config['analysing']['clustering']['min_cluster_size']
    )
    results['clusters'] = analyze_clusters(observations_clustered, features_list)
    
    # Outliers
    observations_outliers = detect_outliers(
        observations,
        features_list,
        contamination=config['analysing']['outliers']['contamination']
    )
    results['outliers'] = analyze_outliers(observations_outliers)
    
    # Variance
    results['variance'] = analyze_axes_importance(
        observations,
        features_list,
        axes=['gamma_id', 'd_encoding_id', 'modifier_id']
    )
    
    return results
```

---

## 📝 SECTION 4 : RAPPORT SYNTHESIS

### 4.1 Structure rapport

```markdown
# VERDICT PHASE R1 - SYNTHESIS DECISION ML

**Date** : 2026-02-16  
**Config** : default  
**Observations** : 57,000 runs

---

## 1. PROFILING GAMMAS

### Gammas STABLE (candidats R2)
- **GAM-001** : 85% CONSERVES_NORM, frobenius median 8.19 (IQR 0.23)
- **GAM-003** : 78% CONSERVES_NORM, frobenius median 8.14 (IQR 0.31)

### Gammas UNSTABLE (mini-phases validation)
- **GAM-007** : 45% NUMERIC_INSTABILITY, timeline dominant 'unstable'

### Gammas PATHOLOGICAL (éliminer pool)
- **GAM-005** : 60% PATHOLOGICAL, outliers 42%
- **GAM-009** : 55% PATHOLOGICAL, condition_number median 1e9

---

## 2. ANALYSING PATTERNS

### Clustering
- **Cluster 0** (n=1200) : GAM-001 dominant, conservation forte
- **Cluster 1** (n=800) : GAM-003/GAM-004, conservation modérée
- **Cluster 2** (n=500) : GAM-005, pathologique
- **Outliers** (-1, n=200) : 3.5% observations, majority GAM-009

### Variance axes
- **Gamma** : η²=0.45 (effet large, dominant)
- **Encoding** : η²=0.12 (effet moyen)
- **Modifier** : η²=0.02 (effet faible)

---

## 3. CONCORDANCE R0↔R1

### Stabilité régimes
- **Kappa global** : 0.78 (concordance forte)
- **Gammas stables** : GAM-001 (κ=0.92), GAM-003 (κ=0.85)
- **Dégradations** : GAM-007 (CONSERVES→UNSTABLE, κ=0.15)

### Trajectoires
- **Conservés** : 8/13 gammas (CONSERVES→CONSERVES)
- **Dégradés** : 3/13 gammas (CONSERVES→PATHOLOGICAL/UNSTABLE)
- **Pathologiques** : 2/13 gammas (PATHOLOGICAL→PATHOLOGICAL)

---

## 4. RECOMMENDATIONS

**PRINCIPE** : Réduction méthodique AVANT exploration (voir `10_METHODOLOGIE_EXPLORATION.md`)

### Phase R2 : Priorité RÉDUCTION (validation suspects)

**❌ NE PAS faire** : Compositions binaires stables immédiatement
- Focus prématuré sur "ce qui marche"
- Blindspots suspects jamais validés
- Explosion combinatoire garantie

**✅ FAIRE** : Mini-phases validation suspects AVANT compositions

**Suspects identifiés R1** :
- **GAM-005, GAM-009** : Pathologique confirmé R0+R1 (100% OSCILLATORY/TRIVIAL)
- **GAM-007** : Instabilité détectée R1 (possible artifact config)
- **GAM-004** : TRIVIAL dominant R0 (saturation haute)

**R2 = Mini-phases validation** (120-200 runs totaux) :
```yaml
iteration_axes:
  gamma_id: [GAM-005, GAM-007, GAM-009, GAM-004]
  d_encoding_id: [SYM-001, SYM-003, ASY-001]  # Encodings tolérants
  seed: [42, 123, 456, 789, 1011]
  config_threshold: [relaxed, default]

# Total : 4 × 3 × 5 × 2 = 120 runs
```

**Verdicts attendus** :
- Si 100% pathologique cross-configs → ÉLIMINER définitivement
- Si config-dependent → RECONFIGURER (restrictions acceptables)
- Si stochastique → AUGMENTER N seeds
- **IMPORTANT** : Utilisateur VALIDE toute élimination (jamais auto)

**Bénéfice** :
- Base réduite 8-10 gammas (vs 13)
- R3 compositions : 8×8 = 64 paires (vs 169 = -62%)

### Phase R3 : EXPLORATION base réduite

**APRÈS éliminations R2** :

**Candidats compositions** : GAM-001, GAM-002, GAM-003, GAM-006, GAM-008, GAM-012
- Stabilité R0↔R1 confirmée
- Conservation norme >75%
- Outliers <5%

**Compositions binaires** :
- 8×8 = 64 paires × 13 encodings × 3 modifiers = 2,496 compositions
- Vs 13×13 = 169 paires × 13 × 3 = 6,591 compositions
- **Économie** : -4,095 runs (-62%)

### Archivage (pas élimination silencieuse)

**Gammas pathologiques confirmés** (après validation utilisateur) :
- Marquer deprecated (pas supprimer)
- Archiver résultats DB (documentation pathologies)
- Exclure futures combinatoires automatiquement

---

## 5. MÉTRIQUES CLÉS

| Métrique | Valeur | Cible | Status |
|----------|--------|-------|--------|
| Success rate | 98.5% | >95% | ✅ |
| Conservation rate | 82% | >70% | ✅ |
| Pathological rate | 8% | <15% | ✅ |
| Outliers rate | 3.5% | <10% | ✅ |
| Kappa R0↔R1 | 0.78 | >0.70 | ✅ |

---

**CONCLUSION** : Phase R1 validée, R2 prête (5 gammas candidats)
```

### 4.2 Génération rapport

```python
# verdict/report.py

def generate_report(
    phase: str,
    observations: pd.DataFrame,
    profiling_results: dict,
    analysing_results: dict,
    concordance_results: dict,
    config: dict
) -> Path:
    """
    Génère rapport synthesis markdown.
    
    Returns:
        Path: Chemin rapport généré
    """
    report_path = Path(f"reports/synthesis_decision_{phase}.md")
    
    with open(report_path, 'w') as f:
        # Header
        f.write(f"# VERDICT PHASE {phase.upper()} - SYNTHESIS DECISION ML\n\n")
        f.write(f"**Date** : {time.strftime('%Y-%m-%d')}\n")
        f.write(f"**Observations** : {len(observations):,} runs\n\n")
        f.write("---\n\n")
        
        # Section 1 : Profiling gammas
        write_profiling_section(f, profiling_results)
        
        # Section 2 : Analysing patterns
        write_analysing_section(f, analysing_results)
        
        # Section 3 : Concordance (si disponible)
        if concordance_results:
            write_concordance_section(f, concordance_results)
        
        # Section 4 : Recommendations
        write_recommendations_section(f, profiling_results, analysing_results)
        
        # Section 5 : Métriques clés
        write_metrics_section(f, observations, profiling_results)
    
    return report_path
```

---

## ⚙️ SECTION 5 : CONFIGURATION YAML

### 5.1 Config phase

```yaml
# configs/phases/default/r1.yaml

phase: r1

# Axes itération
iteration_axes:
  gamma_id: all
  d_encoding_id: all
  modifier_id: all
  seed:
    - 42
    - 123
    - 456

# Discovery
discovery:
  gamma_ids: all
  d_encoding_ids: all
  modifier_ids: all

# Kernel config
kernel:
  n_steps: 200
  dt: 0.01

# Featuring config
features:
  layers:
    universal: true
    matrix_2d: true
    matrix_square: true
    tensor_3d: true
    spatial_2d: true
  
  # Features critiques (toutes layers)
  algebra:
    - function: "algebra.frobenius_norm"
      projections: [initial, final, mean, std]
      statistics: [initial, final, mean, std, slope]
  
  graph:
    - function: "graph.density"
      projections: [initial, final, mean]
      statistics: [mean, std]
  
  # ... (autres registres)

# Thresholds
thresholds:
  events:
    deviation_threshold: 0.10
    saturation_threshold: 0.01
    collapse_threshold: 0.50
    instability_threshold: 0.20
    oscillation_threshold: 0.05
  
  regimes:
    conservation:
      min_ratio: 0.95
      max_ratio: 1.05
    slight_decay:
      min_ratio: 0.80
    # ...
  
  stability:
    stable_min: 0.70
    pathological_max: 0.10
    instability_max: 0.10
```

### 5.2 Config verdict

```yaml
# configs/verdict/default/default.yaml

# Profiling config
profiling:
  features_critical:
    - frobenius_norm_final
    - density_mean
    - eigenvalue_max_final
    - spectral_gap_mean
    - condition_number_mean

# Analysing config
analysing:
  features_critical:
    - frobenius_norm_final
    - density_mean
    - eigenvalue_max_final
    - spectral_gap_mean
    - condition_number_mean
  
  clustering:
    min_cluster_size: 100
    min_samples: 10
  
  outliers:
    contamination: 0.05
  
  variance:
    axes:
      - gamma_id
      - d_encoding_id
      - modifier_id
```

---

## ✅ SECTION 6 : CHECKLIST PIPELINE

### 6.1 Avant batch runner

**Vérifications config** :
- [ ] YAML phase existe (`configs/phases/*/phase.yaml`)
- [ ] Axes itération définis
- [ ] Discovery cohérent (all vs liste)
- [ ] Features layers activés
- [ ] Thresholds calibrés

**Vérifications DB** :
- [ ] DB existe (`prc_databases/db_results.db`)
- [ ] Backup récent (<7 jours)
- [ ] Espace disque suffisant (estimation dry-run)

### 6.2 Pendant batch runner

**Monitoring** :
- [ ] Progress logs réguliers (10 runs)
- [ ] Taux succès >95%
- [ ] RAM stable (<max estimation)
- [ ] Pas erreurs répétées

### 6.3 Avant verdict

**Vérifications observations** :
- [ ] Observations phase chargées
- [ ] Features disponibles (pas 100% NaN)
- [ ] Régimes distribution cohérente
- [ ] Timelines variété descriptors

### 6.4 Après verdict

**Vérifications rapport** :
- [ ] Profils gammas générés (tous gammas)
- [ ] Clusters identifiés (2-5 clusters)
- [ ] Outliers analysés (<10%)
- [ ] Concordance R0↔R1 calculée (si applicable)
- [ ] Recommendations claires (candidats R2)

---

## 📝 CONCLUSION

**Ce document définit** :
- ✅ Batch runner (génération compositions, dry-run, loop runs)
- ✅ Verdict (profiling + analysing + rapport)
- ✅ Gestion erreurs robuste (retry, logs, taux échec)
- ✅ Configuration YAML complète
- ✅ Rapport synthesis markdown

**Principe clé** : Orchestration pipeline extensible (axes YAML, migrations auto)

**Prochaine étape** : Lire `07_AXES_COMPOSITION.md` (discovery + YAML détaillé)

---

**FIN 06_PIPELINE.md**
# 07 - AXES COMPOSITION & DISCOVERY

**Date** : 16 février 2026  
**Version** : 1.0  
**Rôle** : Axes itération configurables + discovery + dry-run

---

## 🎯 OBJECTIF DE CE DOCUMENT

**Ce document définit** :
- Principe axes configurables (YAML vs hardcodé)
- Discovery atomics (all, liste explicite, random)
- Generate compositions (produit cartésien axes)
- Dry-run estimation (temps/RAM/DB)
- Workflow ajout axe custom
- Validation implicite compositions

**Prérequis** : Avoir lu `00_PHILOSOPHIE.md` (principe YAML partout), `06_PIPELINE.md` (batch runner)

---

## 🎯 SECTION 1 : PRINCIPE AXES CONFIGURABLES

### 1.1 Motivation

**Problème ancien** : Axes hardcodés pipeline
```python
# ❌ Ancien (hardcodé)
for gamma_id in discovery['gamma_ids']:
    for d_encoding_id in discovery['d_encoding_ids']:
        for modifier_id in discovery['modifier_ids']:
            composition = {
                'gamma_id': gamma_id,
                'd_encoding_id': d_encoding_id,
                'modifier_id': modifier_id
            }
            # Run...
```

**Limitations** :
- Ordre axes figé (gamma → encoding → modifier)
- Ajout axe = modifier code Python
- Axes temporaires (seed, DOF) = complexité

**Solution** : Axes YAML configurables

### 1.2 Axes disponibles

**Axes standards** (toujours) :
- `gamma_id` : Opérateur γ (GAM-001, GAM-002, ...)
- `d_encoding_id` : Encoding D (SYM-001, ASY-001, R3-001, ...)
- `modifier_id` : Modifier D (M1, M2, ...)

**Axes temporaires** (exploration) :
- `seed` : Graine aléatoire (reproductibilité)
- `DOF` : Degrés liberté (si applicable rank)
- `config_featuring` : Configs features (calibration)
- `noise_level` : Niveau bruit (si modifier noise)
- `epsilon` : Seuils régimes (calibration)
- ... (extensible)

**Principe** : N'importe quel paramètre kernel/featuring devient axe

---

## 🔍 SECTION 2 : DISCOVERY ATOMICS

### 2.1 Modes discovery

**Mode 1 : `all`** - Discovery automatique fichiers
```yaml
iteration_axes:
  gamma_id: all              # Tous gammas disponibles
  d_encoding_id: all         # Tous encodings disponibles
  modifier_id: all           # Tous modifiers disponibles
```

**Mode 2 : Liste explicite** - Bypass discovery
```yaml
iteration_axes:
  gamma_id:
    - GAM-001
    - GAM-003
    - GAM-005
  d_encoding_id:
    - SYM-001
    - ASY-001
```

**Mode 3 : `random: N`** - Échantillon aléatoire
```yaml
iteration_axes:
  gamma_id:
    random: 3               # 3 gammas aléatoires
  d_encoding_id:
    random: 2               # 2 encodings aléatoires
```

### 2.2 Implémentation discovery

```python
# pipeline/discovery.py

from pathlib import Path
import random

def discover_atomics(config: dict) -> dict:
    """
    Découvre atomics disponibles.
    
    Args:
        config: Configuration phase
    
    Returns:
        dict: {
            'gamma_ids': list[str],
            'd_encoding_ids': list[str],
            'modifier_ids': list[str]
        }
    """
    discovery = {}
    
    # Discovery gammas
    discovery['gamma_ids'] = discover_axis(
        config['iteration_axes'].get('gamma_id', 'all'),
        base_path='prc/atomics/operators',
        pattern='gamma_hyp_*.py'
    )
    
    # Discovery encodings
    discovery['d_encoding_ids'] = discover_axis(
        config['iteration_axes'].get('d_encoding_id', 'all'),
        base_path='prc/atomics/D_encodings',
        pattern='*.py',
        exclude=['__init__.py']
    )
    
    # Discovery modifiers
    discovery['modifier_ids'] = discover_axis(
        config['iteration_axes'].get('modifier_id', 'all'),
        base_path='prc/atomics/modifiers',
        pattern='*.py',
        exclude=['__init__.py']
    )
    
    return discovery


def discover_axis(
    config_value,
    base_path: str,
    pattern: str,
    exclude: list = None
) -> list[str]:
    """
    Découvre fichiers axe.
    
    Args:
        config_value: 'all', liste, ou {'random': N}
        base_path: Chemin dossier atomics
        pattern: Pattern fichiers (ex: '*.py')
        exclude: Fichiers exclure
    
    Returns:
        list[str]: IDs découverts
    """
    exclude = exclude or []
    
    # Mode liste explicite
    if isinstance(config_value, list):
        return config_value
    
    # Mode random
    if isinstance(config_value, dict) and 'random' in config_value:
        n = config_value['random']
        all_files = discover_files(base_path, pattern, exclude)
        return random.sample(all_files, min(n, len(all_files)))
    
    # Mode 'all' (default)
    return discover_files(base_path, pattern, exclude)


def discover_files(
    base_path: str,
    pattern: str,
    exclude: list
) -> list[str]:
    """
    Découvre tous fichiers matching pattern.
    
    Returns:
        list[str]: IDs fichiers (stem sans extension)
    """
    base = Path(base_path)
    files = base.glob(pattern)
    
    ids = []
    for file in files:
        if file.name in exclude:
            continue
        
        # Extraire ID depuis filename
        # Ex: gamma_hyp_001.py → GAM-001
        # Ex: rank2_symmetric.py → SYM-001 (via parsing header)
        file_id = extract_id_from_file(file)
        if file_id:
            ids.append(file_id)
    
    return sorted(ids)


def extract_id_from_file(filepath: Path) -> str:
    """Extrait ID atomics depuis fichier."""
    # Logique extraction ID (parse header, filename, ...)
    # Implémentation dépend conventions nommage
    # Ex: gamma_hyp_001.py → GAM-001
    stem = filepath.stem
    if stem.startswith('gamma_hyp_'):
        num = stem.split('_')[-1]
        return f"GAM-{num}"
    # ... autres patterns
    return stem.upper()
```

### 2.3 Exemple discovery

```python
# Config YAML
iteration_axes:
  gamma_id: all
  d_encoding_id:
    random: 3
  modifier_id:
    - M1
    - M2

# Discovery résultat
discovery = {
    'gamma_ids': ['GAM-001', 'GAM-002', ..., 'GAM-013'],  # 13 fichiers
    'd_encoding_ids': ['SYM-002', 'ASY-004', 'R3-001'],   # 3 aléatoires
    'modifier_ids': ['M1', 'M2']                          # Liste explicite
}
```

---

## 🔄 SECTION 3 : GENERATE COMPOSITIONS

### 3.1 Principe produit cartésien

**Génération** : Produit cartésien tous axes configurés

```python
# pipeline/compositions.py

from itertools import product

def generate_compositions(config: dict, discovery: dict) -> Iterator[dict]:
    """
    Génère compositions (produit cartésien axes).
    
    Args:
        config: Configuration phase
        discovery: Discovery atomics
    
    Yields:
        dict: Composition {axe: valeur, ...}
    """
    # Extraction axes valeurs
    axes_values = {}
    
    for axe_name, axe_config in config['iteration_axes'].items():
        if axe_name in ['gamma_id', 'd_encoding_id', 'modifier_id']:
            # Axes standards (discovery)
            axes_values[axe_name] = discovery[axe_name + 's']
        else:
            # Axes custom (liste YAML)
            if isinstance(axe_config, list):
                axes_values[axe_name] = axe_config
            elif isinstance(axe_config, dict) and 'random' in axe_config:
                # Random déjà géré discovery pour standards
                # Pour custom, générer valeurs aléatoires
                axes_values[axe_name] = generate_random_values(axe_name, axe_config['random'])
    
    # Ordre axes (ordre insertion dict YAML)
    axes_order = list(config['iteration_axes'].keys())
    
    # Produit cartésien
    values_lists = [axes_values[axe] for axe in axes_order]
    
    for values_tuple in product(*values_lists):
        composition = dict(zip(axes_order, values_tuple))
        yield composition


def generate_random_values(axe_name: str, n: int) -> list:
    """Génère valeurs aléatoires axe custom."""
    # Exemple : seed
    if axe_name == 'seed':
        return [random.randint(0, 10000) for _ in range(n)]
    
    # Autres axes custom...
    return list(range(n))
```

### 3.2 Exemple génération

```yaml
# Config
iteration_axes:
  gamma_id:
    - GAM-001
    - GAM-003
  d_encoding_id:
    - SYM-001
    - ASY-001
  seed:
    - 42
    - 123
```

```python
# Compositions générées (produit cartésien)
[
    {'gamma_id': 'GAM-001', 'd_encoding_id': 'SYM-001', 'seed': 42},
    {'gamma_id': 'GAM-001', 'd_encoding_id': 'SYM-001', 'seed': 123},
    {'gamma_id': 'GAM-001', 'd_encoding_id': 'ASY-001', 'seed': 42},
    {'gamma_id': 'GAM-001', 'd_encoding_id': 'ASY-001', 'seed': 123},
    {'gamma_id': 'GAM-003', 'd_encoding_id': 'SYM-001', 'seed': 42},
    {'gamma_id': 'GAM-003', 'd_encoding_id': 'SYM-001', 'seed': 123},
    {'gamma_id': 'GAM-003', 'd_encoding_id': 'ASY-001', 'seed': 42},
    {'gamma_id': 'GAM-003', 'd_encoding_id': 'ASY-001', 'seed': 123}
]

# Total : 2 × 2 × 2 = 8 compositions
```

---

## 📊 SECTION 4 : DRY-RUN ESTIMATION

### 4.1 Principe dry-run

**Motivation** : Éviter explosions combinatoires silencieuses

**Métriques estimées** :
- Nombre compositions
- Temps estimé (run moyen × n_compositions)
- RAM peak (history max size)
- DB finale (n_compositions × size observation)

### 4.2 Implémentation dry-run

```python
# pipeline/dry_run.py

def estimate_batch(compositions: list[dict], config: dict) -> dict:
    """
    Estime ressources batch.
    
    Args:
        compositions: Liste compositions générées
        config: Configuration phase
    
    Returns:
        dict: {
            'n_compositions': int,
            'estimated_time_hours': float,
            'estimated_ram_mb': float,
            'estimated_db_mb': float
        }
    """
    n_compositions = len(compositions)
    
    # Temps estimation (run moyen)
    # Dépend DOF, rank, n_steps kernel
    dof_avg = estimate_average_dof(compositions, config)
    time_per_run = estimate_run_time(dof_avg, config)
    estimated_time_hours = (n_compositions * time_per_run) / 3600
    
    # RAM estimation (history peak)
    # Dépend rank, dims, n_steps
    estimated_ram_mb = estimate_ram_usage(dof_avg, config)
    
    # DB estimation
    # n_compositions × size observation (~1.3 KB)
    size_observation = 1.3  # KB
    estimated_db_mb = (n_compositions * size_observation) / 1024
    
    return {
        'n_compositions': n_compositions,
        'estimated_time_hours': estimated_time_hours,
        'estimated_ram_mb': estimated_ram_mb,
        'estimated_db_mb': estimated_db_mb
    }


def estimate_average_dof(compositions: list[dict], config: dict) -> int:
    """Estime DOF moyen compositions."""
    # Parse compositions pour extraire DOF moyen
    # Si axe DOF présent, moyenne valeurs
    # Sinon, valeur default config
    dof_values = [c.get('DOF', 100) for c in compositions]
    return int(np.mean(dof_values))


def estimate_run_time(dof: int, config: dict) -> float:
    """
    Estime temps run (secondes).
    
    Formule empirique :
    time = k * DOF^2 * n_steps
    k ≈ 1e-6 (calibré benchmarks)
    """
    n_steps = config['kernel']['n_steps']
    k = 1e-6
    return k * (dof ** 2) * n_steps


def estimate_ram_usage(dof: int, config: dict) -> float:
    """
    Estime RAM usage (MB).
    
    Formule :
    RAM = history size + overhead
    history = n_steps × DOF^2 × 8 bytes (float64)
    overhead ≈ 20% (featuring, kernel)
    """
    n_steps = config['kernel']['n_steps']
    history_size = n_steps * (dof ** 2) * 8  # bytes
    overhead = 1.2  # 20% overhead
    return (history_size * overhead) / (1024 ** 2)  # MB
```

### 4.3 Exemple dry-run

```python
# Config
iteration_axes:
  gamma_id: all            # 13
  d_encoding_id: all       # 13
  modifier_id: all         # 3
  seed: [42, 123, 456]     # 3

# Dry-run estimation
estimation = {
    'n_compositions': 1521,          # 13 × 13 × 3 × 3
    'estimated_time_hours': 6.3,     # 1521 × 15s / 3600
    'estimated_ram_mb': 320,         # History 100×100, 201 steps
    'estimated_db_mb': 1.9           # 1521 × 1.3 KB
}

# Affichage
🔍 DRY-RUN ESTIMATION
─────────────────────────────────
Compositions : 1,521 runs
Temps estimé : 6.3h
RAM peak     : 320 MB
DB finale    : 1.9 MB
─────────────────────────────────

▶ Lancer batch ? (o/n) :
```

---

## ⚙️ SECTION 5 : WORKFLOW AJOUT AXE CUSTOM

### 5.1 Étapes workflow

**Étape 1 : Ajouter axe YAML**
```yaml
# configs/phases/exploration/calibration.yaml
iteration_axes:
  gamma_id: [GAM-001]
  noise_level: [0.01, 0.05, 0.1]  # ← Nouvel axe
```

**Étape 2 : Migration DB auto**
```python
# Batch runner détecte axe custom → migration
add_axe_column('noise_level', 'REAL')
# ✓ Colonne 'noise_level' ajoutée (type=REAL)
```

**Étape 3 : Kernel extrait composition**
```python
# core/kernel.py
def run_kernel(composition: dict, config: dict):
    noise_level = composition.get('noise_level', 0.05)  # ← Extraire axe
    # Utiliser noise_level dans kernel
    D_noisy = D_base + np.random.randn(*D_base.shape) * noise_level
    # ...
```

**Étape 4 : Tests validation**
```python
# tests/integration/test_axes_custom.py
def test_noise_level_axis():
    config = {'iteration_axes': {'noise_level': [0.01, 0.05, 0.1]}}
    compositions = list(generate_compositions(config, discovery))
    assert len(compositions) == 3
    assert all('noise_level' in c for c in compositions)
```

**Résultat** : Axe ajouté sans modification pipeline Python ✅

### 5.2 Axes temporaires (calibration)

**Usage** : Établir baselines thresholds

**Workflow** :
1. Créer axe temporaire YAML (ex: `epsilon: [1e-3, 1e-4, 1e-5]`)
2. Runs calibration (50-100 runs)
3. Analyser verdict (optimal epsilon)
4. Fixer threshold YAML production
5. Supprimer axe temporaire YAML

**Résultat** : Axe reste DB (colonnes runs passés), mais pas futures phases

---

## ✅ SECTION 6 : VALIDATION IMPLICITE

### 6.1 Principe validation implicite

**Motivation** : Pas de système contraintes YAML complexe

**Approche** : Compositions invalides ignorées gracefully (kernel/featuring)

**Exemples** :
- DOF sur rank 2 → Kernel ignore DOF (pas applicable)
- Config_featuring incompatible → Featuring ajuste params
- Modifier incompatible encoding → Load échoue, skip composition

**Principe** : Validation métier dans code (pas config YAML)

### 6.2 Gestion compositions invalides

```python
# batch_runner.py (gestion erreurs)

def run_batch(compositions: list[dict], config: dict, phase: str):
    success = 0
    skipped = 0
    
    for composition in compositions:
        try:
            # Validation implicite kernel
            history = run_kernel(composition, config)
            
            # Featuring
            features = extract_features_ml(history, config)
            
            # DB insert
            insert_observation({**composition, **features})
            success += 1
        
        except ValidationError as e:
            # Composition invalide (skip gracefully)
            logger.debug(f"Composition skipped: {e}")
            skipped += 1
            continue
        
        except Exception as e:
            # Erreur réelle (logged)
            logger.error(f"Run failed: {e}")
            continue
    
    logger.info(f"Batch: {success} success, {skipped} skipped")
```

---

## ✅ SECTION 7 : CHECKLIST AXES

### 7.1 Avant ajout axe

**Vérifications** :
- [ ] Axe justifié (calibration ou exploration)
- [ ] Valeurs YAML cohérentes (type, range)
- [ ] Kernel supporte axe (extraction composition)
- [ ] Tests validation préparés

### 7.2 Après ajout axe

**Vérifications** :
- [ ] Migration DB réussie (colonne ajoutée)
- [ ] Compositions générées incluent axe
- [ ] Runs exécutés (pas skip 100%)
- [ ] DB stocke valeurs axe correctement

### 7.3 Cleanup axes temporaires

**Workflow** :
- [ ] Calibration terminée (threshold optimal identifié)
- [ ] Threshold fixé YAML production
- [ ] Axe supprimé YAML futures phases
- [ ] DB conserve colonnes (runs passés archivés)

---

## 📝 CONCLUSION

**Ce document définit** :
- ✅ Principe axes configurables (YAML vs hardcodé)
- ✅ Discovery (all, liste, random)
- ✅ Generate compositions (produit cartésien)
- ✅ Dry-run (temps/RAM/DB estimation)
- ✅ Workflow ajout axe (4 étapes, zéro modif pipeline)
- ✅ Validation implicite (compositions invalides skip)

**Principe clé** : Extensibilité pipeline sans modification code Python

**Prochaine étape** : Lire `08_UTILS_DATABASE.md` (toolkit lecture DB)

---

**FIN 07_AXES_COMPOSITION.md**
# 08 - UTILS DATABASE (PARQUET)

**Date** : 16 février 2026  
**Version** : 2.0 (Parquet)  
**Rôle** : Helpers lecture/écriture Parquet

---

## 🎯 OBJECTIF DE CE DOCUMENT

**Ce document définit** :
- Helpers load/save observations (Parquet)
- Filtres rapides (axes, régimes)
- Charge partielle colonnes (optimisation RAM)
- Cross-phases concat
- Export formats

**Prérequis** : Avoir lu `02_PARQUET_SCHEMA.md` (schema DataFrame)

---

## 📁 SECTION 1 : LOAD OBSERVATIONS

### Signature

```python
def load_observations(
    phase: str,
    columns: list[str] = None,
    filters: dict = None
) -> pd.DataFrame:
    """
    Charge observations phase (Parquet).
    
    Args:
        phase: Phase ('R0', 'R1', 'R2', ...)
        columns: Colonnes charger (None = toutes)
        filters: Filtres dict (ex: {'gamma_id': 'GAM-001'})
    
    Returns:
        DataFrame: Observations
    
    Examples:
        # Charge toutes observations R1
        df = load_observations('R1')
        
        # Charge 3 colonnes R1
        df = load_observations('R1', columns=['exec_id', 'frobenius_norm_final', 'regime'])
        
        # Charge avec filtre
        df = load_observations('R1', filters={'gamma_id': 'GAM-001'})
    """
    import pandas as pd
    from pathlib import Path
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Path phase
    path = Path(f'prc_databases/{phase}.parquet')
    
    if not path.exists():
        raise FileNotFoundError(f"Phase {phase} not found: {path}")
    
    # Load (avec colonnes optionnelles)
    if columns:
        df = pd.read_parquet(path, columns=columns)
        logger.info(f"Loaded {len(df)} observations ({len(columns)} columns) from {phase}")
    else:
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df)} observations (full) from {phase}")
    
    # Filtres post-load (pandas)
    if filters:
        for col, value in filters.items():
            if col not in df.columns:
                logger.warning(f"Filter column '{col}' not found, skipping")
                continue
            df = df[df[col] == value]
        logger.info(f"Filtered: {len(df)} observations remaining")
    
    return df
```

### Exemples usage

```python
# Exemple 1 : Charge complète R1
df = load_observations('R1')
# RAM : 76 MB (57,792 obs × 160 cols)

# Exemple 2 : Charge partielle 5 colonnes
df = load_observations('R1', columns=['exec_id', 'gamma_id', 'frobenius_norm_final', 'regime', 'timeline_descriptor'])
# RAM : 2.4 MB (57,792 obs × 5 cols)

# Exemple 3 : Charge + filtre gamma
df = load_observations('R1', filters={'gamma_id': 'GAM-001'})
# RAM : 6.3 MB (4,816 obs × 160 cols)

# Exemple 4 : Charge partielle + filtre
df = load_observations('R1', columns=['frobenius_norm_final', 'density_mean'], filters={'regime': 'CONSERVES_NORM'})
# RAM : 0.8 MB (47,400 obs × 2 cols)
```

---

## 📊 SECTION 2 : LOAD MULTIPLE PHASES

### Signature

```python
def load_multiple_phases(
    phases: list[str],
    columns: list[str] = None
) -> pd.DataFrame:
    """
    Charge plusieurs phases → concat.
    
    Args:
        phases: Liste phases (['R0', 'R1', ...])
        columns: Colonnes charger (None = toutes)
    
    Returns:
        DataFrame: Observations concaténées
    
    Examples:
        # Charge R0 + R1
        df = load_multiple_phases(['R0', 'R1'])
        
        # Charge R0 + R1 (colonnes partielles)
        df = load_multiple_phases(['R0', 'R1'], columns=['exec_id', 'gamma_id', 'regime'])
    """
    import pandas as pd
    import logging
    
    logger = logging.getLogger(__name__)
    
    dfs = []
    for phase in phases:
        df = load_observations(phase, columns=columns)
        dfs.append(df)
    
    df_all = pd.concat(dfs, ignore_index=True)
    logger.info(f"Concatenated {len(phases)} phases: {len(df_all)} total observations")
    
    return df_all
```

### Exemple usage

```python
# Cross-phases R0 + R1
df_all = load_multiple_phases(['R0', 'R1'])

# Concordance régimes
regime_r0 = df_all[df_all['phase'] == 'R0'].groupby('gamma_id')['regime'].apply(lambda x: x.mode()[0])
regime_r1 = df_all[df_all['phase'] == 'R1'].groupby('gamma_id')['regime'].apply(lambda x: x.mode()[0])

concordance = (regime_r0 == regime_r1).mean()
print(f"Concordance R0↔R1: {concordance:.2%}")
```

---

## 💾 SECTION 3 : SAVE OBSERVATIONS

### Signature

```python
def save_observations(
    observations: list[dict],
    phase: str,
    mode: str = 'overwrite'
) -> None:
    """
    Sauvegarde observations Parquet.
    
    Args:
        observations: Liste dicts observations
        phase: Phase ('R0', 'R1', ...)
        mode: 'overwrite' ou 'append' (append = réécriture complète)
    
    Examples:
        # Sauvegarde observations R1
        save_observations(observations_list, 'R1')
    """
    import pandas as pd
    from pathlib import Path
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Convert liste dicts → DataFrame
    df_new = pd.DataFrame(observations)
    
    path = Path(f'prc_databases/{phase}.parquet')
    
    if mode == 'append' and path.exists():
        # Append = load existant + concat + réécriture
        df_existing = pd.read_parquet(path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_parquet(path, index=False)
        logger.info(f"Appended {len(df_new)} observations to {phase} (total: {len(df_combined)})")
    
    else:
        # Overwrite
        df_new.to_parquet(path, index=False)
        logger.info(f"Saved {len(df_new)} observations to {phase}")
```

---

## 🔍 SECTION 4 : AGGREGATION HELPERS

```python
def aggregate_features(
    df: pd.DataFrame,
    features: list[str],
    groupby: str = 'gamma_id',
    aggregations: list[str] = ['median', 'mean', 'std']
) -> pd.DataFrame:
    """
    Agrège features par groupe.
    
    Examples:
        df_agg = aggregate_features(
            df,
            features=['frobenius_norm_final'],
            groupby='gamma_id',
            aggregations=['median', 'std']
        )
    """
    agg_dict = {}
    for feature in features:
        for agg in aggregations:
            if agg == 'median':
                agg_dict[f"{feature}_median"] = (feature, 'median')
            elif agg == 'mean':
                agg_dict[f"{feature}_mean"] = (feature, 'mean')
            elif agg == 'std':
                agg_dict[f"{feature}_std"] = (feature, 'std')
    
    df_agg = df.groupby(groupby).agg(**agg_dict).reset_index()
    return df_agg
```

---

## 📤 SECTION 5 : EXPORT

```python
def export_csv(df: pd.DataFrame, filepath: str) -> None:
    """Export CSV."""
    from pathlib import Path
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✓ Exported {len(df)} observations to {filepath}")
```

---

## 📝 CONCLUSION

**Ce document définit** :
- ✅ load_observations() : Charge partielle optimisée
- ✅ save_observations() : Write Parquet
- ✅ Aggregation helpers
- ✅ Code simplifié (pandas API)

**Performance** : Load <0.5s, RAM optimisée

**FIN 08_UTILS_DATABASE.md**
# 09 - MIGRATION (ANCIEN → NOUVEAU PIPELINE)

**Date** : 16 février 2026  
**Version** : 1.0  
**Rôle** : Migration ancien pipeline vers refactor ML

---

## 🎯 OBJECTIF DE CE DOCUMENT

**Ce document définit** :
- Workflow migration 5 étapes
- Backup ancien pipeline
- Création YAML équivalent ancien hardcodé
- Tests validation consistance
- Script compare pipelines

**Prérequis** : Avoir lu tous docs 00-08

---

## 🎯 SECTION 1 : WORKFLOW MIGRATION

### 1.1 Étapes migration

**Étape 1** : Backup ancien pipeline  
**Étape 2** : Créer YAML équivalent  
**Étape 3** : Refactorer batch_runner  
**Étape 4** : Tests validation  
**Étape 5** : Migration DB (si nécessaire)

**Durée estimée** : 1-2 semaines

---

## 📦 SECTION 2 : ÉTAPE 1 - BACKUP ANCIEN PIPELINE

### 2.1 Archivage code

```bash
# Créer dossier legacy
$ mkdir -p legacy/pipeline_v6/

# Backup fichiers critiques
$ cp batch_runner.py legacy/pipeline_v6/batch_runner_v6.py
$ cp verdict.py legacy/pipeline_v6/verdict_v6.py
$ cp -r tests/ legacy/pipeline_v6/tests_v6/

# Backup DB
$ cp prc_databases/db_results_r1.db legacy/pipeline_v6/db_results_r1_backup.db

# Git commit
$ git add legacy/
$ git commit -m "Backup pipeline v6 avant refactor ML"
```

### 2.2 Documentation ancien pipeline

```markdown
# legacy/pipeline_v6/README.md

## Pipeline V6 (Legacy)

**Date backup** : 2026-02-16  
**Dernière phase** : R1 (57,000 observations)

### Architecture
- Tests modules (10 fichiers)
- DB séparées par phase (r0, r1)
- Timeseries stockées (73% DB)

### Pour restaurer
```bash
$ cp legacy/pipeline_v6/batch_runner_v6.py batch_runner.py
$ python batch_runner.py --phase r1
```

### Raisons refactor
- Timeseries inutilisées (280 MB / 382 MB)
- DB séparées → concordance difficile
- Refactors répétés (3× R0→R1)
```

---

## ⚙️ SECTION 3 : ÉTAPE 2 - YAML ÉQUIVALENT

### 3.1 Identifier hardcodé ancien

**Ancien batch_runner.py** :
```python
# ❌ Hardcodé
for gamma_id in discovery['gamma_ids']:
    for d_encoding_id in discovery['d_encoding_ids']:
        for modifier_id in discovery['modifier_ids']:
            composition = {
                'gamma_id': gamma_id,
                'd_encoding_id': d_encoding_id,
                'modifier_id': modifier_id
            }
```

**Seuils hardcodés** :
```python
# tests/utilities/UTIL/regime.py
THRESHOLD_CONSERVATION = 0.95
THRESHOLD_PATHOLOGICAL = 1.50

# tests/utilities/UTIL/aggregation.py
BIMODAL_THRESHOLD = 0.1
```

### 3.2 Créer YAML équivalent

```yaml
# configs/phases/default/r1.yaml

phase: r1

# Axes itération (équivalent ancien)
iteration_axes:
  gamma_id: all
  d_encoding_id: all
  modifier_id: all

# Discovery
discovery:
  gamma_ids: all
  d_encoding_ids: all
  modifier_ids: all

# Features (mapper ancien tests modules)
features:
  layers:
    universal: true
    matrix_2d: true
    matrix_square: true
  
  algebra:
    - function: "algebra.frobenius_norm"
      projections: [initial, final, mean, std]
  
  graph:
    - function: "graph.density"
      projections: [initial, final, mean]
  
  # ... (autres modules)

# Thresholds (équivalent ancien hardcodé)
thresholds:
  regimes:
    conservation:
      min_ratio: 0.95
      max_ratio: 1.05
    pathological:
      max_ratio: 1.50
  
  aggregation:
    bimodal_threshold: 0.1
```

---

## 🔄 SECTION 4 : ÉTAPE 3 - REFACTORER BATCH_RUNNER

### 4.1 Nouveau batch_runner.py

```python
# batch_runner.py (nouveau)

from prc.core.kernel import run_kernel
from prc.featuring.hub import extract_features_ml
from prc.utils.database import insert_observation
from prc.pipeline.compositions import generate_compositions
from prc.pipeline.discovery import discover_atomics

def main():
    # Load config YAML
    config = load_config(args.phase, args.config_set)
    
    # Discovery
    discovery = discover_atomics(config)
    
    # Generate compositions
    compositions = list(generate_compositions(config, discovery))
    
    # Run batch
    for composition in compositions:
        history = run_kernel(composition, config)
        features = extract_features_ml(history, config)
        insert_observation({**composition, **features})
```

### 4.2 Migration tests → featuring

**Ancien** : `tests/test_gra_001.py`
```python
# tests/test_gra_001.py (ancien)
def test_graph_metrics(history):
    observation_data = {
        'test_name': 'graph_001',
        'statistics': {},
        'timeseries': []
    }
    
    # Calculs metrics
    density = compute_density(history[-1])
    clustering = compute_clustering_coeff(history[-1])
    
    observation_data['statistics']['density'] = density
    observation_data['timeseries'].append(density_timeseries)
    
    return observation_data
```

**Nouveau** : `featuring/registries/graph_registry.py`
```python
# featuring/registries/graph_registry.py (nouveau)
class GraphRegistry(BaseRegistry):
    @register_function("density")
    def compute_density(self, state, **kwargs):
        # Même calcul
        return float(density)
    
    @register_function("clustering_coeff")
    def compute_clustering_coeff(self, state, **kwargs):
        return float(clustering)
```

**Configuration** :
```yaml
# configs/features/graph.yaml
graph_features:
  - function: "graph.density"
    projections: [initial, final, mean]
  
  - function: "graph.clustering_coeff"
    projections: [mean]
```

---

## ✅ SECTION 5 : ÉTAPE 4 - TESTS VALIDATION

### 5.1 Principe tests validation

**Objectif** : Vérifier consistance ancien vs nouveau pipeline

**Métriques validation** :
- Features correlation : >0.95
- Régimes agreement : >0.90
- Timelines concordance : >0.80

### 5.2 Script compare_pipelines.py

```python
# tests/compare_pipelines.py

import pandas as pd
import numpy as np
from prc.utils.database import load_observations

def compare_pipelines(
    observations_old: pd.DataFrame,
    observations_new: pd.DataFrame
) -> dict:
    """
    Compare observations ancien vs nouveau pipeline.
    
    Args:
        observations_old: DataFrame ancien pipeline
        observations_new: DataFrame nouveau pipeline
    
    Returns:
        dict: {
            'features_correlation': float,
            'regimes_agreement': float,
            'n_observations': int,
            'features_compared': list[str]
        }
    """
    # Features communes
    common_features = [
        col for col in observations_old.columns
        if col in observations_new.columns
        and col.endswith(('_final', '_mean'))
        and observations_old[col].dtype in [np.float64, np.int64]
    ]
    
    # Features correlation
    correlations = []
    for feature in common_features:
        old_values = observations_old[feature].dropna()
        new_values = observations_new[feature].dropna()
        
        if len(old_values) > 0 and len(new_values) > 0:
            # Match observations (même gamma_id, d_encoding_id)
            merged = pd.merge(
                observations_old[['gamma_id', 'd_encoding_id', feature]].rename(columns={feature: 'old'}),
                observations_new[['gamma_id', 'd_encoding_id', feature]].rename(columns={feature: 'new'}),
                on=['gamma_id', 'd_encoding_id'],
                how='inner'
            )
            
            if len(merged) > 10:
                corr = merged['old'].corr(merged['new'])
                correlations.append((feature, corr))
    
    features_correlation = np.mean([c[1] for c in correlations]) if correlations else np.nan
    
    # Régimes agreement
    if 'regime' in observations_old.columns:
        # Mapper ancien régimes → nouveaux
        regime_mapping = {
            'conservation': 'CONSERVES_NORM',
            'pathological': 'PATHOLOGICAL',
            # ...
        }
        
        observations_old['regime_mapped'] = observations_old['regime'].map(regime_mapping)
        
        merged = pd.merge(
            observations_old[['gamma_id', 'regime_mapped']],
            observations_new[['gamma_id', 'regime']],
            on='gamma_id',
            how='inner'
        )
        
        regimes_agreement = (merged['regime_mapped'] == merged['regime']).mean()
    else:
        regimes_agreement = np.nan
    
    return {
        'features_correlation': features_correlation,
        'regimes_agreement': regimes_agreement,
        'n_observations': len(observations_new),
        'features_compared': [f[0] for f in correlations],
        'correlations_detail': correlations
    }


def main():
    """Exécute comparaison pipelines."""
    # Charger ancien (DB legacy)
    observations_old = load_observations_legacy('legacy/pipeline_v6/db_results_r1_backup.db')
    
    # Charger nouveau
    observations_new = load_observations('R1')
    
    # Comparaison
    results = compare_pipelines(observations_old, observations_new)
    
    # Affichage
    print("\n🔍 VALIDATION PIPELINE")
    print("─" * 50)
    print(f"Features correlation : {results['features_correlation']:.3f}")
    print(f"Régimes agreement    : {results['regimes_agreement']:.3f}")
    print(f"N observations       : {results['n_observations']}")
    print(f"Features comparées   : {len(results['features_compared'])}")
    print("─" * 50)
    
    # Validation
    if results['features_correlation'] > 0.95:
        print("✅ Features correlation > 0.95 (PASS)")
    else:
        print("❌ Features correlation < 0.95 (FAIL)")
    
    if results['regimes_agreement'] > 0.90:
        print("✅ Régimes agreement > 0.90 (PASS)")
    else:
        print("❌ Régimes agreement < 0.90 (FAIL)")
    
    # Détails correlations faibles
    print("\n📊 Correlations détaillées :")
    for feature, corr in sorted(results['correlations_detail'], key=lambda x: x[1]):
        if corr < 0.95:
            print(f"  ⚠️  {feature}: {corr:.3f} (faible)")


def load_observations_legacy(db_path: str) -> pd.DataFrame:
    """Charge observations ancien pipeline."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM observations", conn)
    conn.close()
    return df


if __name__ == '__main__':
    main()
```

### 5.3 Exécution tests validation

```bash
# Lancer validation
$ python tests/compare_pipelines.py

🔍 VALIDATION PIPELINE
──────────────────────────────────────────────────
Features correlation : 0.982
Régimes agreement    : 0.934
N observations       : 57000
Features comparées   : 45
──────────────────────────────────────────────────
✅ Features correlation > 0.95 (PASS)
✅ Régimes agreement > 0.90 (PASS)

📊 Correlations détaillées :
  ⚠️  eigenvalue_min_final: 0.923 (faible)
  ⚠️  condition_number_mean: 0.887 (faible)
```

---

## 🔧 SECTION 6 : ÉTAPE 5 - MIGRATION DB

### 6.1 Quand migrer DB ?

**Cas 1** : Axes temporaires ajoutés ancien pipeline
→ Migration colonne nécessaire

**Cas 2** : Features renommées
→ Migration renommage colonne

**Cas 3** : DB séparées → DB unique
→ Migration fusion tables

### 6.2 Script migration DB

```python
# utils/migrate_db_legacy.py

import sqlite3
import pandas as pd
from pathlib import Path

def migrate_legacy_to_new(
    legacy_dbs: list[Path],
    new_db: Path = Path('prc_databases/db_results.db')
):
    """
    Migre ancien DB séparées → DB unique partitionnée.
    
    Args:
        legacy_dbs: Liste DB anciennes (r0, r1, ...)
        new_db: DB unique nouveau pipeline
    """
    conn_new = sqlite3.connect(new_db)
    
    for legacy_db in legacy_dbs:
        phase = extract_phase_from_filename(legacy_db)
        
        # Charger observations legacy
        conn_legacy = sqlite3.connect(legacy_db)
        df = pd.read_sql("SELECT * FROM observations", conn_legacy)
        conn_legacy.close()
        
        # Mapper colonnes ancien → nouveau
        df_mapped = map_legacy_columns(df, phase)
        
        # Insert nouveau DB
        df_mapped.to_sql('observations', conn_new, if_exists='append', index=False)
        
        print(f"✓ Migré {phase}: {len(df_mapped)} observations")
    
    conn_new.close()
    print("✓ Migration terminée")


def extract_phase_from_filename(filepath: Path) -> str:
    """Extrait phase depuis filename DB."""
    # Ex: db_results_r1.db → R1
    stem = filepath.stem
    if '_r' in stem:
        return stem.split('_r')[-1].upper()
    return 'R0'


def map_legacy_columns(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    """Mapper colonnes ancien → nouveau."""
    # Renommages
    column_mapping = {
        'frobenius_norm': 'frobenius_norm_final',
        'density': 'density_mean',
        'regime_old': 'regime',
        # ...
    }
    
    df_mapped = df.rename(columns=column_mapping)
    
    # Ajouter phase
    df_mapped['phase'] = phase
    
    # Génération exec_id
    df_mapped['exec_id'] = [f"{phase}_{i:06d}" for i in range(len(df_mapped))]
    
    return df_mapped


if __name__ == '__main__':
    legacy_dbs = [
        Path('legacy/pipeline_v6/db_results_r0.db'),
        Path('legacy/pipeline_v6/db_results_r1.db')
    ]
    
    migrate_legacy_to_new(legacy_dbs)
```

---

## ✅ SECTION 7 : CHECKLIST MIGRATION

### 7.1 Avant migration

**Backup** :
- [ ] Code ancien sauvegardé (`legacy/`)
- [ ] DB anciennes sauvegardées
- [ ] Git commit avant refactor

**YAML** :
- [ ] Config équivalent ancien créé
- [ ] Seuils mappés hardcodé → YAML
- [ ] Axes itération définis

### 7.2 Pendant migration

**Refactor** :
- [ ] Batch runner refactoré (axes YAML)
- [ ] Tests migrés → featuring registres
- [ ] Utils refactorés → utils/database.py

**Validation** :
- [ ] Tests unitaires passent
- [ ] Imports circulaires vérifiés
- [ ] Params hardcodés supprimés

### 7.3 Après migration

**Tests validation** :
- [ ] Script compare_pipelines.py exécuté
- [ ] Features correlation >0.95
- [ ] Régimes agreement >0.90
- [ ] DB nouvelle validée (count, schema)

**Documentation** :
- [ ] README migration écrit
- [ ] Changelog mis à jour
- [ ] Legacy documenté

---

## 📝 CONCLUSION

**Ce document définit** :
- ✅ Workflow migration 5 étapes
- ✅ Backup ancien pipeline (code + DB)
- ✅ Création YAML équivalent ancien hardcodé
- ✅ Refactoring batch_runner (axes YAML)
- ✅ Tests validation (correlation >0.95, agreement >0.90)
- ✅ Migration DB (ancien séparées → unique partitionnée)

**Durée estimée** : 1-2 semaines

**Cibles validation** :
- Features correlation : >0.95 ✅
- Régimes agreement : >0.90 ✅
- Consistance observations : 100%

---

**FIN 09_MIGRATION.md**

---

**DOCUMENTS PRINCIPAUX TERMINÉS (9/9)** ✅
# 10 - MÉTHODOLOGIE EXPLORATION

**Date** : 16 février 2026  
**Version** : 1.0  
**Rôle** : Principe réduction méthodique avant exploration

---

## 🎯 OBJECTIF DE CE DOCUMENT

**Ce document définit** :
- Principe réduction avant exploration (anti-optimisation)
- Workflow phases alternées (réduction ↔ exploration)
- Mini-phases validation suspects
- Élimination méthodique incompatibilités
- Gestion explosion combinatoire

**Prérequis** : Avoir lu `00_PHILOSOPHIE.md` (vision globale)

---

## ⚠️ SECTION 1 : ANTI-PATTERN OPTIMISATION PRÉMATURÉE

### Problème : Chercher "ce qui marche"

**Pattern classique (INCORRECT)** :
```
R0 : Discovery 12 gammas
  ↓ Identifier "meilleurs"
R1 : Compositions sur 5 gammas stables
  ↓ Optimiser
R2 : Ternaires sur stables
  ↓ EXPLOSION
R3 : 12×12×12 = 1,728 trios
```

**Conséquences** :
- ❌ **Focalisation prématurée** : Focus 5 "bons", ignore 7 autres
- ❌ **Explosion combinatoire** : 12 gammas → 1,728 trios
- ❌ **Blindspots géants** : Suspects (GAM-009, 010) jamais validés
- ❌ **Micro-fente** : Optimise zone étroite, perd panorama

**Exemple concret** :
```
R0 : GAM-009 = 100% OSCILLATORY_UNSTABLE
  → Classé "pathologique"
  → Jamais re-testé

R1-R5 : Focus GAM-001, 002, 006 (stables)
  → Compositions multiples
  → Explosion combinatoire

GAM-009 : Incompatible VRAIMENT ou config trop stricte ?
  → JAMAIS SU (blindspot permanent)
```

---

## ✅ SECTION 2 : PRINCIPE RÉDUCTION MÉTHODIQUE

### Philosophie : "Éliminer ce qui NE marche PAS"

**Pattern correct** :
```
R0 : Discovery 12 gammas
  ↓ Identifier SUSPECTS
R1 : Mini-phases validation suspects (100-200 runs)
  ↓ Confirmer incompatibilités
  → ÉLIMINER 2-4 gammas
  ↓ BASE RÉDUITE (8-10 gammas)
R2 : Compositions sur base propre
  ↓ 10×10 = 100 paires (vs 144)
R3 : Ternaires base réduite
  ↓ 10×10×10 = 1,000 trios (vs 1,728)
```

**Bénéfices** :
- ✅ **Espace propre** : Incompatibilités éliminées méthodiquement
- ✅ **Réduction combinatoire** : -30% paires, -42% trios
- ✅ **Pas blindspots** : Suspects validés AVANT explorer
- ✅ **Panorama complet** : Cartographie incompatibilités exhaustive

**Économie cumulative** :
```
R2-R3 avec 12 gammas : 5,616 + 67,392 = 73,008 runs
R2-R3 avec 10 gammas : 3,900 + 39,000 = 42,900 runs
  → -30,000 runs économisés (-41%)
```

---

## 🔄 SECTION 3 : WORKFLOW PHASES ALTERNÉES

### Pattern : Réduction ↔ Exploration

```
R0 : DISCOVERY (large)
  ├─ 12 gammas atomiques
  ├─ 13 encodings
  └─ 3 modifiers
  → 5,364 observations

R1 : RÉDUCTION (validation suspects)
  ├─ Mini-phase GAM-009 (100 runs)
  ├─ Mini-phase GAM-010 (100 runs)
  ├─ Mini-phase GAM-004 (100 runs, TRIVIAL suspect)
  └─ Mini-phase GAM-005 (100 runs, TRIVIAL suspect)
  → 400 observations
  → ÉLIMINER 2-4 gammas confirmés incompatibles
  → BASE RÉDUITE : 8-10 gammas

R2 : EXPLORATION (compositions base réduite)
  ├─ Paires binaires (10×10 = 100)
  ├─ 13 encodings × 3 modifiers
  └─ 3,900 observations
  → Identifier paires suspectes (20-30 outliers)

R3 : RÉDUCTION (validation paires)
  ├─ Mini-phases 20 paires suspectes
  └─ 2,000 observations
  → ÉLIMINER paires pathologiques
  → BASE RÉDUITE : 50 paires validées

R4 : EXPLORATION (ternaires base propre)
  ├─ Trios depuis 50 paires
  └─ Espace exploration gérable
```

**Principe alternance** :
- **Réduction** : Nettoyer incompatibilités (mini-phases ciblées)
- **Exploration** : Explorer espace propre (compositions larges)
- **Itération** : Réduire → Explorer → Réduire → Explorer

---

## 🔬 SECTION 4 : MINI-PHASES VALIDATION

### Objectif : Confirmer incompatibilité

**Question** : GAM-009 = VRAIMENT incompatible ou config trop stricte ?

**Hypothèses tester** :
1. **Seuils trop stricts** → Tester config relaxed
2. **Encodings incompatibles** → Tester encodings tolérants
3. **Stochasticité** → Tester variations seed
4. **Paramétrisation** → Tester variations hyperparamètres

### Workflow mini-phase

```yaml
# configs/phases/validation/gam009.yaml

phase: validation_gam009

iteration_axes:
  gamma_id: [GAM-009]  # Suspect unique
  
  d_encoding_id:       # Encodings tolérants
    - SYM-001          # Plus stable R0
    - SYM-003
    - ASY-001
  
  seed:                # Variations stochastiques
    - 42
    - 123
    - 456
    - 789
    - 1011
  
  config_threshold:    # Axe temporaire (seuils)
    - relaxed          # Moins strict
    - default
    - strict

# Total : 1 × 3 × 5 × 3 = 45 runs
# Estimation : 15-30 minutes
```

**Verdicts possibles** :

**A) TOUJOURS pathologique** :
```
45 runs → 45 OSCILLATORY_UNSTABLE (100%)
  → CONFIRME incompatibilité
  → ÉLIMINER GAM-009 définitivement
```

**B) Config-dependent** :
```
15 runs relaxed  → 10 CONSERVES (66%)
15 runs default  → 5 CONSERVES (33%)
15 runs strict   → 0 CONSERVES (0%)
  → Incompatibilité = seuils trop stricts
  → RECONFIGURER (pas éliminer)
  → Garder GAM-009 avec config relaxed
```

**C) Stochastique** :
```
45 runs → 20 CONSERVES, 25 OSCILLATORY (variabilité haute)
  → Comportement aléatoire seeds
  → AUGMENTER N seeds (stabiliser variance)
  → Garder GAM-009, surveillance variance
```

**D) Encoding-dependent** :
```
SYM-001 : 15 runs → 12 CONSERVES (80%)
SYM-003 : 15 runs → 8 CONSERVES (53%)
ASY-001 : 15 runs → 2 CONSERVES (13%)
  → Incompatibilité = encodings asymétriques
  → RESTREINDRE GAM-009 + encodings symétriques
```

---

## 🎯 SECTION 5 : DÉCISION ÉLIMINATION

### Principe : Utilisateur valide TOUJOURS

**Pipeline DÉTECTE + ANALYSE + RECOMMANDE** :
```python
# Verdict mini-phase GAM-009
rapport = {
    'gamma_id': 'GAM-009',
    'n_runs': 45,
    'regimes': {
        'OSCILLATORY_UNSTABLE': 45,  # 100%
        'CONSERVES_NORM': 0
    },
    'analysis': "100% pathologique cross-configs/encodings/seeds",
    
    'recommendation': "ÉLIMINER définitivement",
    'confidence': 'HIGH',
    'justification': "Aucune configuration viable détectée"
}
```

**Utilisateur DÉCIDE** :
```bash
# Option 1 : Accepter recommendation
$ python manage_pool.py --eliminate GAM-009
# → Marque GAM-009 deprecated
# → Exclut futures compositions

# Option 2 : Rejeter (investigation supplémentaire)
$ python manage_pool.py --investigate GAM-009
# → Mini-phase étendue (variations paramètres)

# Option 3 : Reconfigurer
$ python manage_pool.py --reconfigure GAM-009 --config relaxed
# → Garde GAM-009 avec restrictions
```

**❌ JAMAIS auto-élimination** :
- Pipeline ne supprime JAMAIS automatiquement
- Toute élimination = validation humaine explicite
- Incertitude levée AVANT décision

---

## 📊 SECTION 6 : CAS USAGE CONCRETS

### Cas 1 : R0 → R1 validation suspects

**R0 résultats** :
- 5 gammas STABLES (GAM-001, 002, 006, 008, 012)
- 2 gammas UNSTABLE (GAM-009, 010 = 100% OSCILLATORY)
- 4 gammas MIXED (GAM-003, 004, 005, 007)

**❌ Décision incorrecte** :
```
R1 : Compositions binaires 5 stables
  → Focus prématuré
  → GAM-009, 010 jamais validés (blindspot)
```

**✅ Décision correcte** :
```
R1 : Mini-phases validation
  ├─ GAM-009 (45 runs) → 100% OSCILLATORY → ÉLIMINER
  ├─ GAM-010 (45 runs) → 100% OSCILLATORY → ÉLIMINER
  ├─ GAM-004 (45 runs) → 80% TRIVIAL → ÉLIMINER (saturation)
  └─ GAM-005 (45 runs) → 80% TRIVIAL → ÉLIMINER (saturation)
  
  → BASE RÉDUITE : 8 gammas (001, 002, 003, 006, 007, 008, 012 + ?)
  → R2 : Compositions 8×8 = 64 paires (vs 144 = -56%)
```

---

### Cas 2 : R2 → R3 validation paires

**R2 résultats** (compositions binaires) :
- 50 paires STABLES (conservation >70%)
- 10 paires MIXED (conservation 40-70%)
- 4 paires PATHOLOGICAL (outliers spectral instability)

**❌ Décision incorrecte** :
```
R3 : Ternaires direct sur 50 paires stables
  → 50×50×50 = 125,000 trios (explosion)
```

**✅ Décision correcte** :
```
R3 : Mini-phases validation paires suspectes
  ├─ 10 paires MIXED (100 runs each)
  ├─ 4 paires PATHOLOGICAL (50 runs each)
  └─ Total : 1,200 runs
  
  → ÉLIMINER 8 paires confirmées incompatibles
  → BASE RÉDUITE : 56 paires validées
  
R4 : Ternaires base réduite
  → Estimation combinatoire gérable
```

---

## ⚙️ SECTION 7 : CONFIGURATION MINI-PHASES

### Structure YAML

```yaml
# configs/phases/validation/suspects_r1.yaml

phase: validation_r1

# Description
description: "Validation gammas suspects R0 avant compositions R2"

# Cibles validation
validation_targets:
  - gamma_id: GAM-009
    reason: "100% OSCILLATORY_UNSTABLE R0"
    confidence: "LOW (possible config stricte)"
  
  - gamma_id: GAM-010
    reason: "90% OSCILLATORY_UNSTABLE R0"
    confidence: "LOW (possible encoding incompatible)"
  
  - gamma_id: GAM-004
    reason: "80% TRIVIAL R0"
    confidence: "MEDIUM (saturation haute)"
  
  - gamma_id: GAM-005
    reason: "80% TRIVIAL R0"
    confidence: "MEDIUM (saturation haute)"

# Compositions mini-phase
iteration_axes:
  gamma_id: [GAM-009, GAM-010, GAM-004, GAM-005]
  
  d_encoding_id:
    - SYM-001  # Tolérant
    - SYM-003
    - ASY-001
  
  seed: [42, 123, 456, 789, 1011]
  
  config_threshold:
    - relaxed
    - default

# Total : 4 × 3 × 5 × 2 = 120 runs
# Estimation temps : 1-2h

# Thresholds relaxed (tests moins stricts)
thresholds_relaxed:
  regimes:
    conservation:
      min_ratio: 0.90  # vs 0.95 default
      max_ratio: 1.10  # vs 1.05 default
  
  events:
    deviation_threshold: 0.15  # vs 0.10 default
    instability_threshold: 0.30  # vs 0.20 default
```

---

## 📋 SECTION 8 : CHECKLIST MINI-PHASE

### Avant lancer validation

**Hypothèses claires** :
- [ ] Suspects identifiés (régimes pathologiques R0)
- [ ] Raisons suspicion documentées
- [ ] Hypothèses testables formulées (seuils ? encodings ? seeds ?)

**Configuration mini-phase** :
- [ ] Axes variations définis (config, encodings, seeds)
- [ ] Estimation runs (<200 runs recommandé)
- [ ] Durée estimée acceptable (<3h)

### Après validation

**Analyse résultats** :
- [ ] Régimes distribution cohérente
- [ ] Hypothèses confirmées/infirmées
- [ ] Patterns identifiés (config-dependent ? stochastique ?)

**Décision utilisateur** :
- [ ] Éliminer confirmé incompatible
- [ ] Reconfigurer (restrictions acceptables)
- [ ] Investigation étendue (incertitude reste)

---

## 🎯 SECTION 9 : MÉTRIQUES RÉDUCTION

### Objectifs quantitatifs

**Taux élimination** :
- Cible : 20-30% pool initial (2-4 gammas sur 12)
- Acceptable : 10-40%
- Alerte : <10% (pas assez sélectif) ou >50% (trop strict)

**Combinatoire réduite** :
```
12 gammas → 10 gammas : -30% paires, -42% trios
12 gammas → 8 gammas  : -56% paires, -68% trios
```

**Confiance éliminations** :
- HIGH : 100% pathologique cross-configs (GAM-009)
- MEDIUM : >80% pathologique (GAM-004)
- LOW : <80% pathologique (investigation étendue)

### Métriques qualité

**Validité scientifique** :
- ✅ Incompatibilités confirmées méthodiquement
- ✅ Hypothèses testées explicitement
- ✅ Pas blindspots (suspects validés)

**Efficacité opérationnelle** :
- ✅ Réduction combinatoire >20%
- ✅ Runs validation <10% budget total
- ✅ Décisions rapides (<1 semaine/mini-phase)

---

## 📝 CONCLUSION

**Principe clé** : **RÉDUIRE avant EXPLORER**

**Anti-pattern** : Optimiser prématurément → Explosion + blindspots

**Pattern correct** :
1. Discovery large
2. **Mini-phases validation suspects**
3. **Élimination méthodique incompatibles**
4. Exploration base réduite propre
5. Itération (réduction ↔ exploration)

**Bénéfices** :
- ✅ Combinatoire gérable long terme
- ✅ Espace exploration propre (pas incompatibilités)
- ✅ Panorama complet (pas blindspots)
- ✅ Décisions validées (pas suppositions)

**Prochaine étape** : Implémenter mini-phases R1 validation suspects R0

---

**FIN 10_METHODOLOGIE_EXPLORATION.md**
# ANNEXE - CODE FEATURING COMPLET

**Date** : 16 février 2026  
**Rôle** : Code référence featuring (registres + hub + extractor)

---

## 🎯 UTILISATION ANNEXE

**Ce document contient** : Code complet implémentation featuring
**Usage** : Référence implémentation (copier/adapter si besoin)
**Validation** : Code validé Charter 7.0 + REFACTOR_ML

---

## 📁 SECTION 1 : HUB FEATURING

```python
# featuring/hub.py

import logging
import numpy as np
from typing import Dict

from .extractor import extract_features_ml
from .layers import inspect_history

logger = logging.getLogger(__name__)


def run_featuring(history: np.ndarray, config: dict) -> Dict:
    """
    Point entrée featuring (orchestrateur).
    
    Args:
        history: Timeline états (T, *dims)
        config: Configuration featuring
    
    Returns:
        dict: Features scalaires (~150)
    """
    # Validation input
    if not isinstance(history, np.ndarray):
        raise TypeError(f"History must be np.ndarray, got {type(history)}")
    
    if history.ndim < 2:
        raise ValueError(f"History must have rank ≥2 (got {history.ndim})")
    
    # Inspection
    info = inspect_history(history)
    logger.info(f"History: rank={info['rank']}, dims={info['dims']}, "
                f"is_square={info['is_square']}, steps={info['temporal_steps']}")
    
    # Extraction features
    features = extract_features_ml(history, config)
    
    logger.info(f"Extracted {len(features)} features")
    
    return features
```

---

## 📁 SECTION 2 : BASE REGISTRY

```python
# featuring/registries/base_registry.py

import numpy as np
import logging
from typing import Callable, Dict, Any

logger = logging.getLogger(__name__)


class BaseRegistry:
    """
    Classe base registres fonctions extraction.
    
    Pattern :
    - Fonctions pures (state → scalar)
    - Décorateur @register_function
    - Protection NaN/Inf interne
    - Layer optionnel (validation applicability)
    """
    
    def __init__(self):
        self.functions = {}
        self.layer = 'universal'  # Override subclass
    
    @staticmethod
    def register_function(name: str, layer: str = None):
        """
        Décorateur enregistrement fonction.
        
        Args:
            name: Registry key (ex: "frobenius_norm")
            layer: Layer optionnel (universal, matrix_square, ...)
        """
        def decorator(func: Callable):
            # Store metadata
            func._registry_name = name
            func._registry_layer = layer
            return func
        return decorator
    
    def get_function(self, name: str) -> Callable:
        """Retourne fonction enregistrée."""
        if name not in self.functions:
            raise ValueError(f"Function '{name}' not registered")
        return self.functions[name]
    
    def list_functions(self) -> list[str]:
        """Liste fonctions disponibles."""
        return list(self.functions.keys())
```

---

## 📁 SECTION 3 : ALGEBRA REGISTRY

```python
# featuring/registries/algebra_registry.py

import numpy as np
from .base_registry import BaseRegistry


class AlgebraRegistry(BaseRegistry):
    """Registre fonctions algébriques."""
    
    def __init__(self):
        super().__init__()
        self.layer = 'universal'
    
    @BaseRegistry.register_function("frobenius_norm")
    def compute_frobenius_norm(self, state: np.ndarray, **kwargs) -> float:
        """Norme Frobenius tenseur."""
        return float(np.linalg.norm(state, 'fro'))
    
    @BaseRegistry.register_function("trace", layer="matrix_square")
    def compute_trace(self, state: np.ndarray, **kwargs) -> float:
        """Trace matrice carrée."""
        return float(np.trace(state))
    
    @BaseRegistry.register_function("determinant", layer="matrix_square")
    def compute_determinant(self, state: np.ndarray, epsilon: float = 1e-10, **kwargs) -> float:
        """Déterminant matrice carrée."""
        try:
            det = np.linalg.det(state)
            if np.abs(det) < epsilon:
                return 0.0  # Singulière
            return float(det)
        except np.linalg.LinAlgError:
            return 0.0
    
    @BaseRegistry.register_function("condition_number", layer="matrix_square")
    def compute_condition_number(self, state: np.ndarray, **kwargs) -> float:
        """Conditionnement matrice."""
        try:
            cond = np.linalg.cond(state)
            if np.isinf(cond) or np.isnan(cond):
                return float(1e10)  # Sentinelle singularité
            return float(min(cond, 1e10))
        except np.linalg.LinAlgError:
            return float(1e10)
    
    @BaseRegistry.register_function("matrix_rank", layer="matrix_2d")
    def compute_matrix_rank(self, state: np.ndarray, **kwargs) -> int:
        """Rang matrice."""
        return int(np.linalg.matrix_rank(state))
```

---

## 📁 SECTION 4 : GRAPH REGISTRY

```python
# featuring/registries/graph_registry.py

import numpy as np
import networkx as nx
from .base_registry import BaseRegistry


class GraphRegistry(BaseRegistry):
    """Registre fonctions graphes."""
    
    def __init__(self):
        super().__init__()
        self.layer = 'matrix_2d'
    
    @BaseRegistry.register_function("density")
    def compute_density(self, state: np.ndarray, threshold: float = 0.0, **kwargs) -> float:
        """Densité graphe."""
        G = self._to_graph(state, threshold)
        return float(nx.density(G))
    
    @BaseRegistry.register_function("clustering_coeff")
    def compute_clustering_coeff(self, state: np.ndarray, threshold: float = 0.0, **kwargs) -> float:
        """Coefficient clustering moyen."""
        G = self._to_graph(state, threshold)
        return float(nx.average_clustering(G))
    
    @BaseRegistry.register_function("transitivity")
    def compute_transitivity(self, state: np.ndarray, threshold: float = 0.0, **kwargs) -> float:
        """Transitivité graphe."""
        G = self._to_graph(state, threshold)
        return float(nx.transitivity(G))
    
    @BaseRegistry.register_function("avg_degree")
    def compute_avg_degree(self, state: np.ndarray, threshold: float = 0.0, **kwargs) -> float:
        """Degré moyen."""
        G = self._to_graph(state, threshold)
        degrees = [d for n, d in G.degree()]
        return float(np.mean(degrees)) if degrees else 0.0
    
    @BaseRegistry.register_function("max_degree")
    def compute_max_degree(self, state: np.ndarray, threshold: float = 0.0, **kwargs) -> int:
        """Degré maximum."""
        G = self._to_graph(state, threshold)
        degrees = [d for n, d in G.degree()]
        return int(max(degrees)) if degrees else 0
    
    def _to_graph(self, state: np.ndarray, threshold: float) -> nx.Graph:
        """Convertit matrice → graphe."""
        adjacency = np.abs(state) > threshold
        G = nx.from_numpy_array(adjacency)
        return G
```

---

## 📁 SECTION 5 : SPECTRAL REGISTRY

```python
# featuring/registries/spectral_registry.py

import numpy as np
from scipy.fft import fft2
from .base_registry import BaseRegistry


class SpectralRegistry(BaseRegistry):
    """Registre fonctions spectrales."""
    
    def __init__(self):
        super().__init__()
        self.layer = 'matrix_square'
    
    @BaseRegistry.register_function("eigenvalue_max")
    def compute_eigenvalue_max(self, state: np.ndarray, **kwargs) -> float:
        """Valeur propre max (module)."""
        try:
            eigenvalues = np.linalg.eigvals(state)
            return float(np.max(np.abs(eigenvalues)))
        except np.linalg.LinAlgError:
            return np.nan
    
    @BaseRegistry.register_function("eigenvalue_min")
    def compute_eigenvalue_min(self, state: np.ndarray, **kwargs) -> float:
        """Valeur propre min (module)."""
        try:
            eigenvalues = np.linalg.eigvals(state)
            return float(np.min(np.abs(eigenvalues)))
        except np.linalg.LinAlgError:
            return np.nan
    
    @BaseRegistry.register_function("spectral_gap")
    def compute_spectral_gap(self, state: np.ndarray, **kwargs) -> float:
        """Gap spectral."""
        try:
            eigenvalues = np.linalg.eigvals(state)
            eig_abs = np.abs(eigenvalues)
            eig_sorted = np.sort(eig_abs)
            return float(eig_sorted[-1] - eig_sorted[-2])
        except (np.linalg.LinAlgError, IndexError):
            return np.nan
    
    @BaseRegistry.register_function("spectral_radius")
    def compute_spectral_radius(self, state: np.ndarray, **kwargs) -> float:
        """Rayon spectral."""
        return self.compute_eigenvalue_max(state, **kwargs)
    
    @BaseRegistry.register_function("fft_power", layer="matrix_2d")
    def compute_fft_power(self, state: np.ndarray, **kwargs) -> float:
        """Puissance spectrale FFT 2D."""
        fft_result = fft2(state)
        power = np.abs(fft_result) ** 2
        return float(np.sum(power))
```

---

## 📁 SECTION 6 : PATTERN REGISTRY

```python
# featuring/registries/pattern_registry.py

import numpy as np
from .base_registry import BaseRegistry


class PatternRegistry(BaseRegistry):
    """Registre patterns structures."""
    
    def __init__(self):
        super().__init__()
        self.layer = 'matrix_2d'
    
    @BaseRegistry.register_function("symmetry_score")
    def compute_symmetry_score(self, state: np.ndarray, **kwargs) -> float:
        """Score symétrie matrice."""
        if state.shape[0] != state.shape[1]:
            return 0.0
        
        diff = state - state.T
        return float(1.0 - np.linalg.norm(diff, 'fro') / np.linalg.norm(state, 'fro'))
    
    @BaseRegistry.register_function("sparsity")
    def compute_sparsity(self, state: np.ndarray, threshold: float = 1e-10, **kwargs) -> float:
        """Proportion éléments nuls."""
        zeros = np.abs(state) < threshold
        return float(np.mean(zeros))
    
    @BaseRegistry.register_function("diagonal_dominance")
    def compute_diagonal_dominance(self, state: np.ndarray, **kwargs) -> float:
        """Dominance diagonale."""
        if state.shape[0] != state.shape[1]:
            return 0.0
        
        diag = np.abs(np.diag(state))
        off_diag = np.abs(state - np.diag(np.diag(state)))
        row_sums = np.sum(off_diag, axis=1)
        
        dominance = diag - row_sums
        return float(np.mean(dominance > 0))
```

---

## 📁 SECTION 7 : STATISTICAL REGISTRY

```python
# featuring/registries/statistical_registry.py

import numpy as np
from scipy.stats import skew, kurtosis
from .base_registry import BaseRegistry


class StatisticalRegistry(BaseRegistry):
    """Registre statistiques."""
    
    def __init__(self):
        super().__init__()
        self.layer = 'universal'
    
    @BaseRegistry.register_function("mean_value")
    def compute_mean(self, state: np.ndarray, **kwargs) -> float:
        """Moyenne valeurs."""
        return float(np.mean(state))
    
    @BaseRegistry.register_function("std_value")
    def compute_std(self, state: np.ndarray, **kwargs) -> float:
        """Écart-type."""
        return float(np.std(state))
    
    @BaseRegistry.register_function("variance")
    def compute_variance(self, state: np.ndarray, **kwargs) -> float:
        """Variance."""
        return float(np.var(state))
    
    @BaseRegistry.register_function("skewness")
    def compute_skewness(self, state: np.ndarray, **kwargs) -> float:
        """Asymétrie."""
        flat = state.flatten()
        return float(skew(flat))
    
    @BaseRegistry.register_function("kurtosis")
    def compute_kurtosis(self, state: np.ndarray, **kwargs) -> float:
        """Kurtosis."""
        flat = state.flatten()
        return float(kurtosis(flat))
    
    @BaseRegistry.register_function("percentile_25")
    def compute_percentile_25(self, state: np.ndarray, **kwargs) -> float:
        """Percentile 25%."""
        return float(np.percentile(state, 25))
    
    @BaseRegistry.register_function("percentile_75")
    def compute_percentile_75(self, state: np.ndarray, **kwargs) -> float:
        """Percentile 75%."""
        return float(np.percentile(state, 75))
    
    @BaseRegistry.register_function("iqr")
    def compute_iqr(self, state: np.ndarray, **kwargs) -> float:
        """Intervalle interquartile."""
        q75 = np.percentile(state, 75)
        q25 = np.percentile(state, 25)
        return float(q75 - q25)
```

---

## 📁 SECTION 8 : ENTROPY REGISTRY

```python
# featuring/registries/entropy_registry.py

import numpy as np
from scipy.stats import entropy
from .base_registry import BaseRegistry


class EntropyRegistry(BaseRegistry):
    """Registre entropies."""
    
    def __init__(self):
        super().__init__()
        self.layer = 'universal'
    
    @BaseRegistry.register_function("shannon_entropy")
    def compute_shannon_entropy(self, state: np.ndarray, bins: int = 50, **kwargs) -> float:
        """Entropie Shannon distribution."""
        flat = state.flatten()
        counts, _ = np.histogram(flat, bins=bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return float(entropy(probs))
    
    @BaseRegistry.register_function("renyi_entropy")
    def compute_renyi_entropy(self, state: np.ndarray, alpha: float = 2.0, bins: int = 50, **kwargs) -> float:
        """Entropie Rényi."""
        flat = state.flatten()
        counts, _ = np.histogram(flat, bins=bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        
        if alpha == 1.0:
            return self.compute_shannon_entropy(state, bins)
        
        return float((1 / (1 - alpha)) * np.log(np.sum(probs ** alpha)))
    
    @BaseRegistry.register_function("von_neumann_entropy", layer="matrix_square")
    def compute_von_neumann_entropy(self, state: np.ndarray, **kwargs) -> float:
        """Entropie von Neumann."""
        try:
            eigenvalues = np.linalg.eigvalsh(state)
            # Normaliser
            eigenvalues = eigenvalues / np.sum(np.abs(eigenvalues))
            eigenvalues = eigenvalues[eigenvalues > 0]
            return float(-np.sum(eigenvalues * np.log(eigenvalues)))
        except np.linalg.LinAlgError:
            return np.nan
```

---

## 📁 SECTION 9 : EXTRACTOR PRINCIPAL

```python
# featuring/extractor.py

import numpy as np
import logging
from typing import Dict

from .layers import inspect_history
from .projections import project_temporal, compute_statistics
from .dynamic_events import detect_dynamic_events
from .registry_manager import RegistryManager

logger = logging.getLogger(__name__)


def extract_features_ml(history: np.ndarray, config: dict) -> Dict:
    """
    Extraction features principale (orchestrateur).
    
    Args:
        history: Timeline états (T, *dims)
        config: Configuration featuring
    
    Returns:
        dict: Features scalaires (~150)
    """
    features = {}
    
    # 1. Validation
    if not np.all(np.isfinite(history)):
        logger.warning("History contains NaN/Inf")
    
    # 2. Inspection
    info = inspect_history(history)
    rank = info['rank']
    is_square = info['is_square']
    is_cubic = info['is_cubic']
    
    # 3. Registry manager
    registry_mgr = RegistryManager()
    
    # 4. Extract universal features
    if config['features']['layers'].get('universal', True):
        features.update(extract_layer_features(
            history, 'universal', config, registry_mgr
        ))
    
    # 5. Extract rank-specific
    if rank == 2:
        if config['features']['layers'].get('matrix_2d', True):
            features.update(extract_layer_features(
                history, 'matrix_2d', config, registry_mgr
            ))
        
        if is_square and config['features']['layers'].get('matrix_square', True):
            features.update(extract_layer_features(
                history, 'matrix_square', config, registry_mgr
            ))
    
    elif rank == 3:
        if config['features']['layers'].get('tensor_3d', True):
            features.update(extract_layer_features(
                history, 'tensor_3d', config, registry_mgr
            ))
        
        if is_cubic and config['features']['layers'].get('matrix_square', True):
            # Astuce : slice plan médian
            mid = history.shape[1] // 2
            history_slice = history[:, mid, :, :]
            features.update(extract_layer_features(
                history_slice, 'matrix_square', config, registry_mgr, prefix='cubic_'
            ))
    
    # 6. Dynamic events
    events = detect_dynamic_events(history, config)
    features.update(events)
    
    # 7. Régimes
    regime = classify_regime(features, config)
    features['regime'] = regime
    
    # 8. Timelines
    timeline = interpret_timeline(features, events, config)
    features['timeline_descriptor'] = timeline
    
    return features


def extract_layer_features(
    history: np.ndarray,
    layer: str,
    config: dict,
    registry_mgr: RegistryManager,
    prefix: str = ''
) -> Dict:
    """Extrait features layer."""
    features = {}
    
    layer_config = config['features'].get(layer, [])
    
    for feature_config in layer_config:
        registry_key = feature_config['function']
        function = registry_mgr.get_function(registry_key)
        
        # Projections + statistics
        projections = feature_config.get('projections', ['final'])
        statistics = feature_config.get('statistics', [])
        
        base_name = prefix + registry_key.replace('.', '_')
        
        # Projections temporelles
        for projection in projections:
            try:
                state = project_temporal(history, projection)
                value = function(state, **feature_config.get('params', {}))
                
                feature_name = f"{base_name}_{projection}"
                features[feature_name] = float(value)
            
            except Exception as e:
                logger.error(f"{registry_key} ({projection}) failed: {e}")
                continue
        
        # Statistics temporelles
        if statistics:
            try:
                timeseries = compute_metric_timeseries(history, function, **feature_config.get('params', {}))
                stats = compute_statistics(timeseries, statistics)
                
                for stat_name, stat_value in stats.items():
                    feature_name = f"{base_name}_{stat_name}"
                    features[feature_name] = stat_value
            
            except Exception as e:
                logger.error(f"{registry_key} (statistics) failed: {e}")
    
    return features


def classify_regime(features: dict, config: dict) -> str:
    """Classifie régime."""
    thresholds = config['thresholds']['regimes']
    
    frobenius_initial = features.get('frobenius_norm_initial', 0)
    frobenius_final = features.get('frobenius_norm_final', 0)
    
    if frobenius_initial > 0:
        ratio = frobenius_final / frobenius_initial
    else:
        return 'PATHOLOGICAL'
    
    if np.isnan(ratio) or np.isinf(ratio):
        return 'PATHOLOGICAL'
    
    conservation = thresholds['conservation']
    if conservation['min_ratio'] <= ratio <= conservation['max_ratio']:
        return 'CONSERVES_NORM'
    
    slight_decay = thresholds['slight_decay']
    if slight_decay['min_ratio'] <= ratio < conservation['min_ratio']:
        return 'SLIGHT_DECAY'
    
    if ratio < slight_decay['min_ratio']:
        if features.get('collapse_detected', 0):
            return 'COLLAPSE'
        return 'STRONG_DECAY'
    
    if ratio > conservation['max_ratio']:
        return 'PATHOLOGICAL'
    
    if features.get('instability_detected', 0):
        return 'NUMERIC_INSTABILITY'
    
    return 'UNKNOWN'


def interpret_timeline(features: dict, events: dict, config: dict) -> str:
    """Interprète timeline."""
    deviation = events.get('deviation_detected', 0)
    deviation_time = events.get('deviation_time', None)
    saturation = events.get('saturation_detected', 0)
    collapse = events.get('collapse_detected', 0)
    instability = events.get('instability_detected', 0)
    oscillation = events.get('oscillation_detected', 0)
    
    if deviation and collapse:
        return 'deviation_then_collapse'
    
    if oscillation:
        return 'oscillatory'
    
    if instability:
        return 'unstable'
    
    if deviation:
        if deviation_time < 0.2:
            return 'early_deviation'
        elif deviation_time > 0.8:
            return 'late_deviation'
        return 'mid_deviation'
    
    if saturation:
        return 'saturation'
    
    if collapse:
        return 'collapse'
    
    return 'stable'
```

---

**FIN ANNEXE CODE_FEATURING.md**
# ANNEXE - CODE PROFILING COMPLET

**Date** : 16 février 2026  
**Rôle** : Code référence profiling (aggregation, régimes, timelines)

---

## 📁 SECTION 1 : HUB PROFILING

```python
# profiling/hub.py

import logging
import pandas as pd
from typing import Dict

from .aggregation import aggregate_all_features, detect_bimodal
from .regimes import compute_regime_distribution, classify_gamma_stability
from .timelines import compute_timeline_frequency

logger = logging.getLogger(__name__)


def run_profiling(observations: pd.DataFrame, config: dict) -> Dict:
    """
    Exécute profiling complet.
    
    Args:
        observations: DataFrame observations
        config: Configuration profiling
    
    Returns:
        dict: {
            'profiles_gamma': dict,
            'profiles_encoding': dict,
            'regime_distributions': DataFrame,
            'timeline_frequencies': DataFrame
        }
    """
    logger.info(f"Running profiling on {len(observations)} observations")
    
    results = {}
    
    # Features critiques
    features_list = config['profiling']['features_critical']
    
    # Profils gammas
    profiles_gamma = {}
    for gamma_id in observations['gamma_id'].unique():
        obs_gamma = observations[observations['gamma_id'] == gamma_id]
        profiles_gamma[gamma_id] = generate_gamma_profile(
            obs_gamma, gamma_id, features_list, config
        )
    
    results['profiles_gamma'] = profiles_gamma
    logger.info(f"Generated {len(profiles_gamma)} gamma profiles")
    
    # Profils encodings
    profiles_encoding = {}
    for encoding_id in observations['d_encoding_id'].unique():
        obs_encoding = observations[observations['d_encoding_id'] == encoding_id]
        profiles_encoding[encoding_id] = generate_encoding_profile(
            obs_encoding, encoding_id, features_list, config
        )
    
    results['profiles_encoding'] = profiles_encoding
    logger.info(f"Generated {len(profiles_encoding)} encoding profiles")
    
    # Distributions régimes
    results['regime_distributions'] = compute_regime_distribution(
        observations, groupby='gamma_id'
    )
    
    # Timelines frequency
    results['timeline_frequencies'] = compute_timeline_frequency(
        observations, groupby='gamma_id'
    )
    
    logger.info("Profiling terminé")
    
    return results


def generate_gamma_profile(
    observations: pd.DataFrame,
    gamma_id: str,
    features_list: list,
    config: dict
) -> Dict:
    """Génère profil gamma."""
    # Aggregation features
    features_agg = {}
    for feature_name in features_list:
        if feature_name not in observations.columns:
            continue
        
        values = observations[feature_name].dropna()
        if len(values) == 0:
            continue
        
        features_agg[feature_name] = {
            'median': float(np.median(values)),
            'iqr': float(np.percentile(values, 75) - np.percentile(values, 25)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75)),
            'bimodal': detect_bimodal(values),
            'n_observations': len(values)
        }
    
    # Distribution régimes
    regime_counts = observations['regime'].value_counts(normalize=True)
    regimes_dist = regime_counts.to_dict()
    
    # Timelines frequency
    timeline_counts = observations['timeline_descriptor'].value_counts(normalize=True)
    timelines_freq = timeline_counts.to_dict()
    
    # Classification stabilité
    stability = classify_gamma_stability(
        observations, gamma_id, config['thresholds']['stability']
    )
    
    return {
        'gamma_id': gamma_id,
        'n_observations': len(observations),
        'features': features_agg,
        'regimes': regimes_dist,
        'timelines': timelines_freq,
        'stability': stability,
        'dominant_regime': regime_counts.idxmax() if len(regime_counts) > 0 else None,
        'dominant_timeline': timeline_counts.idxmax() if len(timeline_counts) > 0 else None
    }


def generate_encoding_profile(
    observations: pd.DataFrame,
    encoding_id: str,
    features_list: list,
    config: dict
) -> Dict:
    """Génère profil encoding (similaire gamma)."""
    return generate_gamma_profile(observations, encoding_id, features_list, config)
```

---

## 📁 SECTION 2 : AGGREGATION

```python
# profiling/aggregation.py

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)


def aggregate_feature(
    observations: pd.DataFrame,
    feature_name: str,
    groupby: str = 'gamma_id'
) -> pd.DataFrame:
    """
    Agrège feature cross-runs.
    
    Returns:
        DataFrame: {
            groupby: str,
            'median': float,
            'iqr': float,
            'q25': float,
            'q75': float,
            'bimodal': bool,
            'n_observations': int
        }
    """
    results = []
    
    for group_value, group_df in observations.groupby(groupby):
        values = group_df[feature_name].dropna()
        
        if len(values) == 0:
            continue
        
        results.append({
            groupby: group_value,
            'median': float(np.median(values)),
            'iqr': float(np.percentile(values, 75) - np.percentile(values, 25)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75)),
            'bimodal': detect_bimodal(values),
            'n_observations': len(values)
        })
    
    return pd.DataFrame(results)


def detect_bimodal(values: np.ndarray, threshold: float = 0.1) -> bool:
    """Détecte distribution bimodale."""
    if len(values) < 20:
        return False
    
    try:
        kde = gaussian_kde(values)
        x_range = np.linspace(values.min(), values.max(), 200)
        density = kde(x_range)
        
        peaks, _ = find_peaks(density, prominence=threshold * density.max())
        
        return len(peaks) >= 2
    
    except Exception as e:
        logger.warning(f"Bimodal detection failed: {e}")
        return False


def aggregate_all_features(
    observations: pd.DataFrame,
    features_list: list,
    groupby: str = 'gamma_id'
) -> Dict:
    """Agrège toutes features cross-runs."""
    results = {}
    
    for group_value, group_df in observations.groupby(groupby):
        results[group_value] = {}
        
        for feature_name in features_list:
            if feature_name not in group_df.columns:
                continue
            
            values = group_df[feature_name].dropna()
            
            if len(values) == 0:
                continue
            
            results[group_value][feature_name] = {
                'median': float(np.median(values)),
                'iqr': float(np.percentile(values, 75) - np.percentile(values, 25)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75)),
                'bimodal': detect_bimodal(values),
                'n_observations': len(values)
            }
    
    return results
```

---

## 📁 SECTION 3 : RÉGIMES

```python
# profiling/regimes.py

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_regime_distribution(
    observations: pd.DataFrame,
    groupby: str = 'gamma_id'
) -> pd.DataFrame:
    """
    Calcule distribution régimes par groupe.
    
    Returns:
        DataFrame: {
            groupby: str,
            'regime': str,
            'frequency': float,
            'count': int
        }
    """
    results = []
    
    for group_value, group_df in observations.groupby(groupby):
        regime_counts = group_df['regime'].value_counts()
        total = len(group_df)
        
        for regime, count in regime_counts.items():
            frequency = count / total
            
            results.append({
                groupby: group_value,
                'regime': regime,
                'frequency': frequency,
                'count': count
            })
    
    return pd.DataFrame(results)


def get_dominant_regime(
    observations: pd.DataFrame,
    groupby: str = 'gamma_id'
) -> pd.DataFrame:
    """Identifie régime dominant par groupe."""
    regime_dist = compute_regime_distribution(observations, groupby)
    
    dominant = regime_dist.loc[regime_dist.groupby(groupby)['frequency'].idxmax()]
    
    return dominant[[groupby, 'regime', 'frequency']].rename(
        columns={'regime': 'dominant_regime'}
    )


def classify_gamma_stability(
    observations: pd.DataFrame,
    gamma_id: str,
    thresholds: dict
) -> str:
    """
    Classifie stabilité gamma.
    
    Returns:
        str: 'STABLE', 'UNSTABLE', 'PATHOLOGICAL', 'MODERATE'
    """
    regime_counts = observations['regime'].value_counts(normalize=True)
    
    conserves_freq = regime_counts.get('CONSERVES_NORM', 0)
    pathological_freq = regime_counts.get('PATHOLOGICAL', 0)
    instability_freq = regime_counts.get('NUMERIC_INSTABILITY', 0)
    
    if pathological_freq > thresholds['pathological_max']:
        return 'PATHOLOGICAL'
    
    elif conserves_freq >= thresholds['stable_min']:
        return 'STABLE'
    
    elif instability_freq > thresholds['instability_max']:
        return 'UNSTABLE'
    
    else:
        return 'MODERATE'
```

---

## 📁 SECTION 4 : TIMELINES

```python
# profiling/timelines.py

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_timeline_frequency(
    observations: pd.DataFrame,
    groupby: str = 'gamma_id'
) -> pd.DataFrame:
    """
    Calcule fréquences timelines par groupe.
    
    Returns:
        DataFrame: {
            groupby: str,
            'timeline': str,
            'frequency': float,
            'count': int
        }
    """
    results = []
    
    for group_value, group_df in observations.groupby(groupby):
        timeline_counts = group_df['timeline_descriptor'].value_counts()
        total = len(group_df)
        
        for timeline, count in timeline_counts.items():
            frequency = count / total
            
            results.append({
                groupby: group_value,
                'timeline': timeline,
                'frequency': frequency,
                'count': count
            })
    
    return pd.DataFrame(results)


def get_dominant_timeline(
    observations: pd.DataFrame,
    groupby: str = 'gamma_id'
) -> pd.DataFrame:
    """Identifie timeline dominant par groupe."""
    timeline_freq = compute_timeline_frequency(observations, groupby)
    
    dominant = timeline_freq.loc[timeline_freq.groupby(groupby)['frequency'].idxmax()]
    
    return dominant[[groupby, 'timeline', 'frequency']].rename(
        columns={'timeline': 'dominant_timeline'}
    )


def compare_timelines_cross_phases(
    observations_r0: pd.DataFrame,
    observations_r1: pd.DataFrame,
    groupby: str = 'gamma_id'
) -> pd.DataFrame:
    """
    Compare timelines dominants R0 vs R1.
    
    Returns:
        DataFrame: {
            groupby: str,
            'timeline_r0': str,
            'timeline_r1': str,
            'concordance': bool
        }
    """
    dominant_r0 = get_dominant_timeline(observations_r0, groupby)
    dominant_r1 = get_dominant_timeline(observations_r1, groupby)
    
    comparison = pd.merge(
        dominant_r0[[groupby, 'dominant_timeline']].rename(columns={'dominant_timeline': 'timeline_r0'}),
        dominant_r1[[groupby, 'dominant_timeline']].rename(columns={'dominant_timeline': 'timeline_r1'}),
        on=groupby,
        how='inner'
    )
    
    comparison['concordance'] = comparison['timeline_r0'] == comparison['timeline_r1']
    
    return comparison
```

---

**FIN ANNEXE CODE_PROFILING.md**
# ANNEXES - CODE COMPLET RESTANT

## CODE_ANALYSING.md - Patterns ML

```python
# analysing/hub.py
def run_analysing(observations: pd.DataFrame, config: dict) -> Dict:
    results = {}
    features_list = config['analysing']['features_critical']
    
    # Clustering
    observations_clustered = cluster_observations(
        observations, features_list,
        min_cluster_size=config['analysing']['clustering']['min_cluster_size']
    )
    results['clusters'] = analyze_clusters(observations_clustered, features_list)
    
    # Outliers  
    observations_outliers = detect_outliers(
        observations, features_list,
        contamination=config['analysing']['outliers']['contamination']
    )
    results['outliers'] = analyze_outliers(observations_outliers)
    
    # Variance
    results['variance'] = analyze_axes_importance(
        observations, features_list,
        axes=['gamma_id', 'd_encoding_id', 'modifier_id']
    )
    
    return results

# analysing/clustering.py
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN

def cluster_observations(observations, features_list, min_cluster_size=50):
    X = observations[features_list].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(X_scaled)
    observations['cluster_id'] = -2
    observations.loc[X.index, 'cluster_id'] = labels
    return observations

# analysing/outliers.py
from sklearn.ensemble import IsolationForest

def detect_outliers(observations, features_list, contamination=0.05):
    X = observations[features_list].dropna()
    clf = IsolationForest(contamination=contamination, random_state=42)
    predictions = clf.fit_predict(X)
    observations['is_outlier'] = False
    observations.loc[X.index, 'is_outlier'] = (predictions == -1)
    return observations

# analysing/variance.py
from scipy.stats import f_oneway

def compute_eta_squared(observations, feature_name, factor):
    total_variance = observations[feature_name].var()
    group_means = observations.groupby(factor)[feature_name].mean()
    group_sizes = observations.groupby(factor)[feature_name].count()
    grand_mean = observations[feature_name].mean()
    between_variance = np.sum(group_sizes * (group_means - grand_mean) ** 2) / (len(group_means) - 1)
    return float(between_variance / total_variance) if total_variance > 0 else 0.0

# analysing/concordance.py
from sklearn.metrics import cohen_kappa_score

def compute_kappa_regimes(observations_r0, observations_r1, groupby='gamma_id'):
    results = []
    for group_value in observations_r0[groupby].unique():
        r0_group = observations_r0[observations_r0[groupby] == group_value]
        r1_group = observations_r1[observations_r1[groupby] == group_value]
        
        regime_r0 = r0_group['regime'].mode()[0] if len(r0_group) > 0 else None
        regime_r1 = r1_group['regime'].mode()[0] if len(r1_group) > 0 else None
        
        if regime_r0 and regime_r1 and len(r0_group) > 10 and len(r1_group) > 10:
            min_size = min(len(r0_group), len(r1_group), 100)
            r0_sample = r0_group['regime'].sample(min_size, random_state=42)
            r1_sample = r1_group['regime'].sample(min_size, random_state=42)
            kappa = cohen_kappa_score(r0_sample, r1_sample)
        else:
            kappa = 1.0 if regime_r0 == regime_r1 else 0.0
        
        results.append({
            groupby: group_value,
            'regime_r0': regime_r0,
            'regime_r1': regime_r1,
            'kappa': float(kappa),
            'concordance': 'STRONG' if kappa > 0.8 else ('MODERATE' if kappa > 0.5 else 'WEAK')
        })
    return pd.DataFrame(results)
```

---

## CODE_PIPELINE.md - Batch runner complet

```python
# batch_runner.py - Point entrée
import argparse, logging, time
from prc.core.kernel import run_kernel
from prc.featuring.hub import extract_features_ml
from prc.utils.database import insert_observation
from prc.pipeline.compositions import generate_compositions
from prc.pipeline.discovery import discover_atomics
from prc.pipeline.dry_run import estimate_batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True)
    parser.add_argument('--config-set', default='default')
    parser.add_argument('--no-confirm', action='store_true')
    args = parser.parse_args()
    
    config = load_config(args.phase, args.config_set)
    discovery = discover_atomics(config)
    compositions = list(generate_compositions(config, discovery))
    
    estimation = estimate_batch(compositions, config)
    print(f"\\n🔍 DRY-RUN: {estimation['n_compositions']} runs, "
          f"{estimation['estimated_time_hours']:.1f}h")
    
    if not args.no_confirm and input("Lancer? (o/n): ") != 'o':
        return
    
    prepare_database(config)
    run_batch(compositions, config, args.phase)

def run_batch(compositions, config, phase):
    for i, composition in enumerate(compositions):
        try:
            history = run_kernel(composition, config)
            features = extract_features_ml(history, config)
            observation = {'exec_id': f"{phase}_{i:06d}", 'phase': phase,
                          **composition, **features}
            insert_observation(observation)
        except Exception as e:
            logger.error(f"Run {i} failed: {e}")

# verdict.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True)
    parser.add_argument('--compare-phase')
    args = parser.parse_args()
    
    config = load_config(args.phase, 'default')
    observations = load_observations(args.phase)
    
    profiling_results = run_profiling(observations, config)
    analysing_results = run_analysing(observations, config)
    
    concordance_results = None
    if args.compare_phase:
        observations_compare = load_observations(args.compare_phase)
        concordance_results = compute_concordance(observations_compare, observations, config)
    
    report_path = generate_report(args.phase, observations,
                                   profiling_results, analysing_results,
                                   concordance_results, config)
    print(f"✓ Rapport: {report_path}")
```

---

## CODE_UTILS_DB.md - Helpers DB

```python
# utils/database.py - Toolkit complet
import sqlite3, pandas as pd, numpy as np
from pathlib import Path

def load_observations(phase=None, columns=None, gamma_id=None, 
                     d_encoding_id=None, regime=None,
                     db_path='prc_databases/db_results.db'):
    """Charge observations (charge partielle optimisée)."""
    conn = sqlite3.connect(db_path)
    
    if columns:
        cols_str = ', '.join(columns)
        query = f"SELECT {cols_str} FROM observations"
    else:
        query = "SELECT * FROM observations"
    
    filters, params = [], []
    if phase:
        filters.append("phase = ?")
        params.append(phase)
    if gamma_id:
        filters.append("gamma_id = ?")
        params.append(gamma_id)
    if d_encoding_id:
        filters.append("d_encoding_id = ?")
        params.append(d_encoding_id)
    if regime:
        filters.append("regime = ?")
        params.append(regime)
    
    if filters:
        query += " WHERE " + " AND ".join(filters)
    
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df

def insert_observation(observation, db_path='prc_databases/db_results.db'):
    """Insert observation DB."""
    conn = sqlite3.connect(db_path)
    df = pd.DataFrame([observation])
    df.to_sql('observations', conn, if_exists='append', index=False)
    conn.close()

def add_feature_column(feature_name, feature_type='REAL', 
                       db_path='prc_databases/db_results.db'):
    """Ajoute colonne feature (si pas existe)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA table_info(observations)")
        columns = [row[1] for row in cursor.fetchall()]
        if feature_name in columns:
            conn.close()
            return False
        cursor.execute(f"ALTER TABLE observations ADD COLUMN {feature_name} {feature_type}")
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        conn.close()
        raise

def get_schema(db_path='prc_databases/db_results.db'):
    """Retourne schema DB."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(observations)")
    columns = cursor.fetchall()
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='observations'")
    indexes = cursor.fetchall()
    cursor.execute("SELECT COUNT(*) FROM observations")
    n_observations = cursor.fetchone()[0]
    conn.close()
    return {'columns': columns, 'indexes': indexes, 'n_observations': n_observations}

def vacuum_db(db_path='prc_databases/db_results.db'):
    """Optimise DB."""
    conn = sqlite3.connect(db_path)
    conn.execute("VACUUM")
    conn.close()
```

---

## TESTS_VALIDATION.md - Tests + Benchmarks

```python
# tests/featuring/test_registries.py
import pytest, numpy as np
from prc.featuring.registries.algebra_registry import AlgebraRegistry

def test_frobenius_norm():
    registry = AlgebraRegistry()
    state = np.random.rand(10, 10)
    result = registry.compute_frobenius_norm(state)
    expected = np.linalg.norm(state, 'fro')
    assert np.isclose(result, expected)

def test_trace():
    registry = AlgebraRegistry()
    state = np.random.rand(10, 10)
    result = registry.compute_trace(state)
    expected = np.trace(state)
    assert np.isclose(result, expected)

# tests/integration/test_batch_runner.py
def test_batch_runner_small():
    """Test batch runner 10 runs."""
    config = load_config('r0_test', 'default')
    config['iteration_axes'] = {
        'gamma_id': ['GAM-001'],
        'd_encoding_id': ['SYM-001'],
        'seed': [42, 123]
    }
    
    discovery = discover_atomics(config)
    compositions = list(generate_compositions(config, discovery))
    assert len(compositions) == 2
    
    for composition in compositions:
        history = run_kernel(composition, config)
        features = extract_features_ml(history, config)
        assert len(features) > 100
        assert 'regime' in features
        assert 'timeline_descriptor' in features

# tests/benchmarks/bench_database.py
import time
from prc.utils.database import load_observations

def benchmark_load_partial():
    start = time.time()
    df = load_observations('R1', columns=['exec_id', 'frobenius_norm_final', 'regime'])
    duration = time.time() - start
    print(f"Load partial (3 cols): {duration*1000:.1f} ms ({len(df)} obs)")
    assert duration < 0.1  # <100 ms

def benchmark_load_full():
    start = time.time()
    df = load_observations('R1')
    duration = time.time() - start
    print(f"Load full (150 cols): {duration*1000:.1f} ms ({len(df)} obs)")
    assert duration < 0.5  # <500 ms

# tests/compare_pipelines.py
def compare_pipelines(observations_old, observations_new):
    """Compare ancien vs nouveau pipeline."""
    common_features = [col for col in observations_old.columns
                       if col in observations_new.columns
                       and col.endswith(('_final', '_mean'))]
    
    correlations = []
    for feature in common_features:
        merged = pd.merge(
            observations_old[['gamma_id', feature]].rename(columns={feature: 'old'}),
            observations_new[['gamma_id', feature]].rename(columns={feature: 'new'}),
            on='gamma_id', how='inner'
        )
        if len(merged) > 10:
            corr = merged['old'].corr(merged['new'])
            correlations.append((feature, corr))
    
    features_correlation = np.mean([c[1] for c in correlations])
    
    # Régimes agreement
    regime_mapping = {'conservation': 'CONSERVES_NORM', 'pathological': 'PATHOLOGICAL'}
    observations_old['regime_mapped'] = observations_old['regime'].map(regime_mapping)
    merged = pd.merge(
        observations_old[['gamma_id', 'regime_mapped']],
        observations_new[['gamma_id', 'regime']],
        on='gamma_id', how='inner'
    )
    regimes_agreement = (merged['regime_mapped'] == merged['regime']).mean()
    
    return {
        'features_correlation': features_correlation,
        'regimes_agreement': regimes_agreement,
        'n_observations': len(observations_new)
    }
```

---

**FIN ANNEXES CODE COMPLET**

Total annexes : 6 fichiers (~7000 lignes code référence)
# INDEX REFACTOR_ML - VERSION 2.0 (PARQUET + MÉTHODOLOGIE)

**Date** : 16 février 2026  
**Version** : 2.0 (Parquet + Méthodologie exploration)  
**Total documents** : 10 principaux + 3 annexes = 13 fichiers

---

## 📋 CHANGEMENTS VERSION 2.0

### Ajouts

**10_METHODOLOGIE_EXPLORATION.md** (NOUVEAU) :
- Principe réduction avant exploration
- Anti-pattern optimisation prématurée
- Workflow phases alternées (réduction ↔ exploration)
- Mini-phases validation suspects
- Éliminations méthodiques (utilisateur valide toujours)

### Remplacements

**02_DB_SCHEMA.md → 02_PARQUET_SCHEMA.md** :
- Format Parquet (vs SQL)
- 1 fichier par phase (vs tables SQL)
- Compression 3× (339 MB vs 1,171 MB)
- Code simplifié (pandas API vs SQL)
- Pas migrations schema (flexibilité refactors)

**08_UTILS_DATABASE.md** (RÉÉCRIT) :
- Version Parquet simplifiée
- Helpers pandas natifs
- 300 lignes vs 700 SQL
- Performance 2-3× améliorée

### Patches

**00_PHILOSOPHIE.md** :
- Section 6bis : Méthodologie exploration (réduction avant exploration)

**06_PIPELINE.md** :
- Section 4 Recommendations : Shift méthodologique (validation suspects AVANT compositions)

**05_ANALYSING.md** :
- Section 3 Outliers : Clarification philosophie (détection + analyse + recommandation, jamais auto-élimination)

---

## 🎯 ORDRE LECTURE RECOMMANDÉ

**Pour comprendre architecture** :
1. `00_PHILOSOPHIE.md` (vision globale) ⚠️ PATCHÉ
2. `10_METHODOLOGIE_EXPLORATION.md` (workflow scientifique) ✅ NOUVEAU
3. `01_ARCHITECTURE.md` (structure modules)
4. `02_PARQUET_SCHEMA.md` (stockage données) ✅ NOUVEAU

**Pour implémenter modules** :
5. `03_FEATURING.md` (extraction intra-run)
6. `04_PROFILING.md` + `05_ANALYSING.md` (analyses inter-run) ⚠️ PATCHÉ
7. `06_PIPELINE.md` (orchestration) ⚠️ PATCHÉ
8. `07_AXES_COMPOSITION.md` (configurations)
9. `08_UTILS_DATABASE.md` (helpers) ✅ RÉÉCRIT

**Pour migration** :
10. `09_MIGRATION.md` (ancien → nouveau)

---

## 📊 MÉTRIQUES SUCCÈS v2.0

### Performance

| Métrique | Legacy | Refactor v2.0 | Gain |
|----------|--------|---------------|------|
| **Volumétrie DB** | 1,171 MB (SQL) | 339 MB (Parquet) | -71% |
| **RAM verdict** | 2 GB | 89 MB (charge partielle) | ×22 réduction |
| **Write perf** | 5-10s (SQL INSERT) | 1-2s (Parquet) | 2-5× |
| **Read perf** | 1s (SQL SELECT) | 0.5s (Parquet) | 2× |
| **Code utils** | 700 lignes (SQL) | 300 lignes (Pandas) | -57% |

### Combinatoire (méthodologie)

| Phase | Legacy (12 gammas) | v2.0 (10 gammas réduits) | Gain |
|-------|-------------------|---------------------------|------|
| **R2 paires** | 144 paires | 100 paires | -30% |
| **R3 ternaires** | 1,728 trios | 1,000 trios | -42% |
| **Économie cumulative** | 73,008 runs | 42,900 runs | **-30,108 runs (-41%)** |

---

**FIN INDEX v2.0**
