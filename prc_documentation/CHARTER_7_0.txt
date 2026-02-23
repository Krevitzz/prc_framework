# CHARTER PRC 7.0

> **Hub cognitif LLM** - Document normatif unique décrivant la philosophie et l'architecture PRC.  
> Version 7.0 prévaut sur toutes versions antérieures.

---

## PRÉAMBULE : Comment utiliser ce charter

### Ce qu'EST ce document
**Hub cognitif permanent** :
- Philosophie générale : "Où on va" (axiomes→code, principes immuables)
- Architecture globale : Structure modules (featuring/profiling/analysing)
- Méthodologie : Comment travailler (Algo→Structure→Code, validation obligatoire)
- Navigation : Où chercher documentation détaillée (catalogues, sources)

**Accessible en permanence** : Bibliothèque LLM, chaque conversation

### Ce qu'il N'EST PAS
**Pas un manuel technique** :
- ❌ Pas de détails implémentation (obsolètes au prochain refactor)
- ❌ Pas de code complet (sources demandées si besoin)
- ❌ Pas de métriques chiffrées (tests dédiés)
- ❌ Pas de catalogues fonctions (fichiers séparés)

**Principe** : Charter = "On va par là en marchant", Docs de travail = "Comment on marche"

### Comment l'utiliser
**En tant que LLM** :
1. **Lire en premier** : Poser bases (axiomes, principes, architecture)
2. **Consulter sections pertinentes** : Navigation doc selon tâche
3. **Demander docs de travail** : Catalogues, sources, refactor_ML.md si besoin détails
4. **Respecter méthodologie** : Algo→Structure→Code avec validation utilisateur obligatoire

**Règle** : Toujours avoir charter en mémoire + doc de travail spécifique tâche

---

## SECTION 1 : FONDATIONS IMMUABLES

#### 1.1 Axiomes PRC → Code
**Contenu** :
- **Axiomes A1-A3** :
  - A1 : Dissymétrie informationnelle D irréductible
  - A2 : Mécanisme Γ agissant sur D
  - A3 : Aucun Γ stable ne peut annuler D complètement

- **Logique expérimentale** :
  - Thèse pose gamma universel
  - Pipeline crée candidats atomiques Γ
  - Chaque phase teste candidats :
    - **Conserver** : Pas de preuve négative
    - **Explorer** : Verdict pointe comportement particulier
    - **Exclure** : Preuve incompatibilité axiome/candidat

- **Hiérarchie niveaux** :
  ```
  L0 (ontologique) → L1 (épistémique) → L2 (théorique) → L3 (opérationnel) → L4 (documentaire)
  Règle : L(n) référence uniquement L(≤n-1)
  Exception : L3 peut consommer L4 (descriptif), pas dériver règles
  ```

#### 1.2 Core aveugle
**Contenu** :
- 2 fonctions aveugles :
  ```
  prepare_state(base, modifiers) → np.ndarray
  run_kernel(composition, config) → np.ndarray  # history (T, *dims)
  ```
- **Règles K1-K5** : Core reste aveugle
  - Pas de validation sémantique
  - Pas de classes State/Operator
  - Pas de branchement dépendant D ou Γ

---

### SECTION 2 : ARCHITECTURE GLOBALE

#### 2.1 Organigramme flux
**Contenu** :
```
Batch runner (YAML-driven)
  ├─ Génération compositions (axes configurables)
  │   └─ Discovery (gamma, encoding, modifier) + axes temporaires (seed, DOF, ...)
  ├─ Dry-run (estimation temps/RAM/DB)
  └─ Confirmation (o/n)
  ↓
Kernel (core aveugle)
  ├─ Input : composition dict {'gamma_id': 'GAM-001', 'd_encoding_id': 'SYM-001', ...}
  └─ Output : history (RAM)
  ↓
FEATURING (intra-run, calculs à la volée)
  ├─ Registries (wrappers numpy/scipy)
  │   └─ Layers (universal, matrix_2d, matrix_square, tensor_3d, spatial_2d)
  ├─ Projections temporelles (initial, final, mean, max, min, max_deviation)
  ├─ Statistics scalaires (initial, final, mean, std, slope, cv)
  └─ Dynamic events (deviation, instability, saturation, collapse, oscillation)
  ↓
DB unique (axes colonnes + features JSON, partitionnée par phase)
  ↓
VERDICT (inter-run, post-batch)
  ├─ PROFILING (aggregation cross-runs)
  │   ├─ Aggregation (median, IQR, bimodal)
  │   ├─ Régimes (distribution, familles)
  │   └─ Timelines (fréquences, patterns)
  └─ ANALYSING (patterns ML)
      ├─ Clustering (HDBSCAN)
      ├─ Outliers (IsolationForest)
      ├─ Variance (η², ANOVA)
      └─ Concordance cross-phases (kappa, DTW, trajectoires)
  ↓
Rapports (synthesis + analysis)
```

#### 2.2 Philosophie modules
**Contenu** :

**FEATURING** :
- **Rôle** : Interception + traitement brut (intra-run)
- **Timing** : À la volée, RAM, pendant batch_runner
- **Responsabilité** : Extraire features scalaires depuis history
- **Output** : ~150 features scalaires/observation → DB
- **Principe** : 80% calculs totaux (profiter RAM, minimiser I/O)

**PROFILING** :
- **Rôle** : Agrégation cross-runs → profils
- **Timing** : Post-batch, DB, verdict
- **Responsabilité** : Aggregation (median, IQR, bimodal), régimes distribution, timelines fréquences
- **Output** : Profils gammas, encodings, modifiers
- **Principe** : Contexte cross-runs nécessaire

**ANALYSING** :
- **Rôle** : Patterns ML (inter-run)
- **Timing** : Post-batch, DB, verdict
- **Responsabilité** : Clustering, outliers, variance, concordance cross-phases
- **Output** : Clusters, outliers, η², trajectoires R0↔R1
- **Principe** : 20% calculs totaux (contexte cross-runs)

#### 2.3 Structure dossiers
**Contenu** :
```
prc/
├── core/                           # Exécution aveugle (immuable)
│   ├── kernel.py
│   ├── state_preparation.py
│   └── core_catalog.md
│
├── atomics/                        # Pool candidats (WIP, dépréciations)
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
│   ├── hub.py
│   ├── extractor.py
│   ├── layers.py
│   ├── projections.py
│   ├── statistics.py
│   ├── dynamic_events.py
│   ├── featuring_catalog.md
│   └── registries/
│       ├── __init__.py
│       ├── base_registry.py
│       ├── registry_manager.py
│       ├── post_processors.py
│       ├── PATTERNS.md
│       ├── algebra_registry.py
│       ├── graph_registry.py
│       ├── spectral_registry.py
│       ├── spatial_registry.py
│       ├── topological_registry.py
│       ├── pattern_registry.py
│       ├── statistical_registry.py
│       ├── entropy_registry.py     
│       ├── tensor_registry.py      
│       ├── timeseries_registry.py  
│       └── registries_catalog.md
│
├── profiling/                      # Aggregation DB → profils
│   ├── __init__.py
│   ├── hub.py
│   ├── aggregation.py
│   ├── regimes.py
│   ├── timelines.py
│   ├── dynamic_events.py
│   └── profiling_catalog.md
│
├── analysing/                      # Patterns ML verdict
│   ├── __init__.py
│   ├── hub.py
│   ├── clustering.py
│   ├── outliers.py
│   ├── variance.py
│   ├── interactions.py
│   ├── concordance.py
│   └── analysing_catalog.md
│
├── configs/                        # YAML centralisés
│   ├── phases/
│   │   ├── default/
│   │   │   ├── r0.yaml
│   │   │   ├── r1.yaml
│   │   │   └── r2.yaml
│   │   ├── laxe/
│   │   │   └── r0.yaml
│   │   └── strict/
│   │       └── r0.yaml
│   ├── features/
│   │   ├── default/
│   │   │   ├── layers.yaml
│   │   │   ├── algebra.yaml
│   │   │   ├── pattern.yaml
│   │   │   └── statistical.yaml
│   │   ├── laxe/
│   │   └── strict/
│   ├── thresholds/
│   │   ├── default/
│   │   │   ├── regimes.yaml
│   │   │   └── aggregation.yaml
│   │   ├── laxe/
│   │   └── strict/
│   └── verdict/
│       ├── default/
│       │   └── default.yaml
│       ├── laxe/
│       └── strict/
│
├── utils/                          # Utilitaires généraux transverses
│   ├── __init__.py
│   ├── data_loading.py             # DB queries, load observations
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
├── reports/                        # Rapports (réécrit, pas accumule)
│   ├── synthesis_decision_ML.md
│   ├── analysis_complete_ML.json
│   └── visualizations/
│
├── legacy/                         # Backup ancien pipeline
│   └── (archives refactors précédents)
│
├── batch_runner.py                 # Point entrée exécution
├── verdict.py                      # Point entrée verdict
├── charter_7_0.md                  # Ce document (hub cognitif)
└── README.md
```

---

### SECTION 3 : MÉTHODOLOGIE VALIDATION

#### 3.1 Processus obligatoire Algo → Structure → Code

**Étape 1 : ALGO (langage courant)**
- **Format** : Langage courant, logique métier, zéro code
- **Contenu** : Ce qu'on fait clairement
- **Exemple** :
  ```
  Objectif : Calculer entropie Shannon distribution valeurs tenseur
  
  Algo :
  1. Aplatir tenseur → vecteur 1D
  2. Créer histogramme (bins configurables)
  3. Normaliser histogramme → probabilités
  4. Calculer entropie Shannon : H = -Σ p_i log(p_i)
  5. Normaliser par log(bins) si demandé
  6. Retourner float
  ```

**Étape 2 : STRUCTURE (squelette ancré existant)**
- **Format** : Squelette fonction, passages I/O ancrés réalité code existant
- **Contenu** : Signature + dépendances + retour
- **Exemple** :
  ```python
  def compute_shannon_entropy(
      state: np.ndarray,          # ← On a déjà (from featuring)
      bins: int,                  # ← Param YAML (from config)
      normalize: bool             # ← Param YAML
  ):
      """
      Calcule entropie Shannon distribution.
      
      Utilise :
      - np.histogram (wrapper numpy existant)
      - scipy.stats.entropy (si disponible, sinon custom)
      - Protection epsilon log(0)
      
      Retour :
      - float : entropie [0, log(bins)] ou [0, 1] si normalize
      """
      # ALGO ici (étape 3)
      return float(entropy)
  ```

**Étape 3 : CODE (après validation structure)**
- **Format** : Code Python complet
- **Contenu** : Implémentation algo dans squelette validé

**VALIDATION OBLIGATOIRE** : Utilisateur valide **chaque étape** avant passage suivante

**Justification** : Non-programmeur détecte dérives métier avant qu'elles deviennent invisibles dans code

---

### SECTION 4 : PRINCIPES IMMUABLES

#### 4.1 Flux intra-run / inter-run
**Principe P1** :
- **Intra-run** : Calculs sur history unique (201 snapshots), à la volée, RAM
  - Featuring (registries, projections, statistics, events)
  - Output : Features scalaires → DB (1 write)
  - **80% calculs totaux** (profiter RAM, minimiser I/O)

- **Inter-run** : Calculs cross-runs (contexte observations multiples), verdict, DB
  - Profiling (aggregation, régimes, timelines)
  - Analysing (clustering, outliers, variance, concordance)
  - **20% calculs totaux** (contexte nécessaire)

**Règle** : TOUT ce qui peut être calculé **intra-run** SERA calculé intra → minimiser I/O DB

#### 4.2 Gestion erreurs (physique vs code)
**Principe P2** :

| Type | Cause | Action | Exemple |
|------|-------|--------|---------|
| **Code error** | Calcul non applicable | `raise ValueError` | eigenvalue sur non-carrée |
| **Aberrant** | Valeur hors domaine | `raise ValueError` | log(-1), sqrt(-5) |
| **Physique** | Explosion système réel | `return np.nan` (signal) | Matrice singulière, collapse |

**Niveaux protection** :
1. **Registres** : Wrappers robustes (try/except, sentinelles)
2. **Extractor** : Validation state (np.isfinite, variance non nulle)
3. **Verdict** : Filtrage features >50% NaN après vérif params sévères

**Règle** : NaN ≠ erreur, NaN = information (si explosion physique)

#### 4.3 Layers inspection directe
**Principe P3** :
- **Inspection directe** history (rank, dims, is_square, is_cubic)
- **Pas metadata applicability** (cauchemar compositions)
- **Chemin if/elif** prédéfini (pas boucle for dynamique)

**Layers** :
- `universal` : Tout tenseur (algebra.frobenius_norm, statistical.entropy)
- `matrix_2d` : Rank 2 (graph.density, spectral.fft_power)
- `matrix_square` : Rank 2 carrée (algebra.trace, spectral.eigenvalue_max)
- `tensor_3d` : Rank ≥3 (tensor.tucker_energy, tensor.cp_rank)
- `spatial_2d` : Analyses spatiales 2D (spatial.gradient, topological.connected_components)

**Astuce R3 cubique** : Features square applicables sur slice 2D (plan médian)

#### 4.4 YAML partout
**Principe P4** :
- **Zéro hardcodé** (seuils, params, configs)
- **Tout externalisé** : `configs/features/*.yaml`, `configs/thresholds/*.yaml`
- **Passage obligatoire** params via YAML

**Axes itération configurables** :
- **Ordre axes** : Défini YAML, pas hardcodé pipeline
- **Axes optionnels** : Ignorés automatiquement si vides
- **Nouveaux axes** : Ajout YAML sans modifier pipeline Python
- **Axes temporaires** : seed, config_featuring, DOF (phases exploratoires)
- **Extensibilité** : N'importe quel paramètre kernel devient axe itération

**Discovery** :
- `all` : Discovery automatique tous fichiers disponibles
- Liste explicite : Bypass discovery, utilise uniquement spécifiés
- `random: N` : Échantillon aléatoire N fichiers (runs courts)

**Validation** :
- **Dry-run automatique** : Estimation temps/RAM/DB avant lancement
- **Confirmation obligatoire** : Utilisateur valide (o/n) avant exécution
- **Validation implicite** : Compositions invalides ignorées naturellement (kernel/featuring)

**Exemple** :
```yaml
# configs/phases/default/r0.yaml
iteration_axes:
  gamma_id: all              # Discovery tous gammas
  d_encoding_id:             # Liste explicite
    - SYM-001
    - ASY-001
  modifier_id: all           # Discovery tous modifiers
  seed:                      # Axe temporaire (exploration)
    - 42
    - 123
```

**DB schema** :
- **Axes** : Colonnes SQL (queries rapides, indexes)
- **Features** : JSON (flexibilité refactors registres)
- **Migration** : Script ALTER TABLE si nouvel axe ajouté après coup

**Usage axes temporaires** : Établir baselines (thresholds, configs), puis supprimer axe après calibration.

#### 4.5 Wrappers robustes
**Principe P5** :
- **Préférence stdlib** : numpy, scipy, networkx (pas customs fragiles)
- **Registres = wrappers** : Fonctions pures (state → float)
- **Zéro dépendance interne** PRC

**Exemple** :
```python
# ✅ BON (wrapper numpy)
@register_function("frobenius_norm")
def compute_frobenius(self, state):
    return float(np.linalg.norm(state, 'fro'))

# ❌ MAUVAIS (custom fragile)
@register_function("custom_norm")
def compute_custom(self, state):
    # Réimplémente calcul norme à la main
    return sum(...)  # ← Fragile, pas testé
```

#### 4.6 DB unique partitionnée
**Principe P6** :
- **DB unique** SQLite : `db_results.db`
- **Table unique** : `observations`
- **Partitionnée** : Colonne `phase` (R0, R1, R2, ...)
- **Schema** : Axes = colonnes SQL, Features = JSON
- **Index phase** : Queries cross-phases rapides

```sql
CREATE TABLE observations (
    exec_id TEXT PRIMARY KEY,
    phase TEXT,
    
    -- AXES (colonnes SQL, standards + temporaires)
    gamma_id TEXT NOT NULL,
    d_encoding_id TEXT NOT NULL,
    modifier_id TEXT,
    seed INTEGER,
    DOF INTEGER,
    config_featuring TEXT,
    
    -- FEATURES (JSON, flexible refactors)
    features TEXT NOT NULL,  -- {'frobenius_norm_final': 8.2, ...}
    
    timestamp TEXT
);

CREATE INDEX idx_phase ON observations(phase);
CREATE INDEX idx_gamma_phase ON observations(gamma_id, phase);
```

**Rationale** : Axes colonnes (queries rapides), features JSON (flexibilité registres).

#### 4.7 Règles directionnelles

**Dépendances autorisées** :
```
core → operators, encodings, modifiers
batch_runner → core, featuring
verdict → profiling, analysing
profiling/analysing → (rien pipeline, seulement DB)
featuring.registries → (rien)
```

**Interdictions strictes** :
```
❌ Dépendances circulaires
❌ featuring → profiling/analysing
❌ registries → featuring (isolation totale)
```

#### 4.8 Checklist avant merge
- [ ] Pas de duplication fonction (consulter catalogues AVANT coder)
- [ ] Imports circulaires : `python -m pycircular prc/`
- [ ] Params hardcodés : Tout YAML
- [ ] Tests unitaires passent

---

### SECTION 5 : NAVIGATION DOCUMENTATION

#### 5.1 Bibliothèque permanente
**Accès systématique** (chaque conversation) :
```
charter_7_0.md              # Ce document (hub cognitif)
todo list en cours.md         
```

#### 5.2 Catalogues (injection demande)
**Structure** :
```
featuring/
  └── registries_catalog.md    # Toutes fonctions registres

profiling/
  └── profiling_catalog.md     # Fonctions aggregation, régimes, timelines

analysing/
  └── analysing_catalog.md     # Fonctions clustering, outliers, variance

configs/
  └── configs_catalog.md       # Structure YAML, params disponibles
```

**Usage** : Demander catalogue AVANT coder (vérifier fonction existe)

#### 5.3 Sources code (injection conversation)
**Si besoin détails** : Demander fichier source spécifique
```
featuring/registries/algebra_registry.py
profiling/aggregation.py
configs/features/layers.yaml
```

#### 5.4 Mapping Tâche → Module → Doc

**Exemple 1** :
```
Tâche : Ajouter nouvelle feature extraction
→ Module : featuring/registries/
→ Doc demander : registries_catalog.md, PATTERNS.md
→ Process : Algo → Structure → Code (validation chaque étape)
```

**Exemple 2** :
```
Tâche : Modifier seuils régimes
→ Module : configs/thresholds/regimes.yaml
→ Doc demander : configs_catalog.md
→ Process : Pas de code, éditer YAML uniquement
```

**Exemple 3** :
```
Tâche : Ajouter métrique concordance R0↔R1
→ Module : analysing/concordance.py
→ Doc demander : analysing_catalog.md, concordance.py source
→ Process : Algo → Structure → Code + tests
```

---

### SECTION 6 : INTERDICTIONS CRITIQUES

#### 6.1 Core aveugle
**R-CORE** : Core reste aveugle :
- Aucune validation contenu (symétrie, bornes)
- Aucune classe State, Operator
- Aucun branchement dépendant D ou Γ

#### 6.2 Featuring pureté
**R-FEAT-1** : Registres = fonctions pures (state → float)
- Zéro état global
- Zéro dépendance interne PRC
- Zéro I/O (DB, fichiers)

**R-FEAT-2** : Extraction intra-run uniquement
- Pas de contexte cross-runs (DB) dans featuring
- Pas de verdict dans featuring
- Output : Features scalaires → DB, c'est tout

#### 6.3 Profiling/Analysing inter-run
**R-PROF-1** : Pas de calculs intra-run
- Si calculable intra → mettre dans featuring
- Profiling = contexte cross-runs obligatoire

**R-ANAL-1** : Patterns ML uniquement
- Pas de calculs scalaires simples (profiling)
- Analysing = patterns complexes (clustering, outliers, η²)

#### 6.4 YAML obligatoire
**R-YAML-1** : Zéro hardcodé
- Seuils → `configs/thresholds/*.yaml`
- Params features → `configs/features/*.yaml`
- Configs phases → `configs/phases/*.yaml`

**R-YAML-2** : Passage params explicite
```python
# ❌ MAUVAIS
threshold = 0.1  # Hardcodé

# ✅ BON
threshold = config['thresholds']['regimes']['conservation']['final_ratio']
```

#### 6.5 Tests observations
**R-TEST-1** : Tests ne retournent jamais :
- "PASS"/"FAIL", "bon"/"mauvais"
- Classes/objets
- Params hardcodés
- Jugements normatifs

**R-TEST-2** : Tests = observations pures
- Retour : dict features scalaires
- Pas de verdict
- Pas de régime (calculé profiling)

#### 6.6 Nomenclature
**R-NOM-1** : Termes strictement interdits/remplacés :
- ❌ "matrice de corrélation" → ✅ "tenseur rang 2"
- ❌ "graphe" → ✅ "patterns dans tenseur"
- ❌ "position (i,j)" → ✅ "indice (i,j)"
- ❌ "d_base_id" → ✅ "d_encoding_id"
- ❌ "test verdict" → ✅ "observation" (tests), "verdict" (analysing)

---

### SECTION 7 : GLOSSAIRE

| Terme | Définition | Exemple |
|-------|------------|---------|
| **Intra-run** | Calculs history unique (RAM, à la volée) | Featuring |
| **Inter-run** | Calculs cross-runs (DB, contexte) | Profiling, Analysing |
| **Layer** | Catégorie features applicables | universal, matrix_2d, tensor_3d |
| **Projection** | Vue temporelle history | initial, final, mean, max |
| **Registry** | Module fonctions extraction | algebra_registry.py |
| **Dynamic event** | Event timeline | deviation, instability, saturation |
| **Régime** | Classification comportement | CONSERVES_NORM, NUMERIC_INSTABILITY |
| **Timeline** | Descriptor compositionnel events | "early_deviation_then_saturation" |
| **Concordance** | Stabilité cross-phases | kappa régimes, DTW timelines |
| **Wrapper** | Fonction encapsulation stdlib | `np.linalg.norm()` |
| **Hub** | Module orchestration | featuring/hub.py, verdict.py |
| **Registre** | Module fonctions réutilisables | algebra_registry.py |
| **registry_key** | Identifiant unique fonction | "algebra.matrix_norm" |
| **history** | Timeline états temporels (~201) | np.ndarray (201, n, m) |
| **composition** | Dict axes run | {'gamma_id': 'GAM-001', 'seed': 42} |
| **observation** | Features scalaires run unique | dict ~150 features |
| **profil gamma** | Caractérisation Γ cross-runs | régime dominant, timelines fréquences |
| **Axe** | Dimension itération configurable | gamma_id, seed, DOF |

---

**FIN CHARTER PRC 7.0**

Ce charter est la **référence cognitive unique** pour l'architecture PRC 7.0.  
Il décrit les **principes immuables** et la **philosophie générale** du système.  
Pour détails implémentation : consulter docs de travail (refactor_ML.md, catalogues).