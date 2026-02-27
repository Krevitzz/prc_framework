# CHARTER PRC 7.1

> **Hub cognitif LLM** — Document normatif décrivant la philosophie et l'architecture PRC.
> Version 7.1 prévaut sur toutes versions antérieures.

---

## PRÉAMBULE : Comment utiliser ce charter

**Ce qu'EST ce document** :
- La philosophie générale : "Où on va et pourquoi"
- Les principes immuables : Ce qui ne change pas entre refactors
- La méthodologie de travail : Algo→Structure→Code
- La navigation : Où chercher quand on a besoin de détails

**Ce qu'il N'EST PAS** :
- ❌ Pas une source de vérité sur le code (les fichiers sources le sont)
- ❌ Pas un manuel d'implémentation
- ❌ Pas un catalogue de fonctions
- ❌ Pas un état du projet (obsolète au moindre changement)

**Règle** : Charter = balise cognitive. Docs de travail + sources = réalité.

**En pratique** :
1. Lire le charter en premier (axiomes, principes, architecture)
2. Demander les sources et catalogues selon la tâche
3. Respecter Algo→Structure→Code avec validation utilisateur

---

## SECTION 0 : STRUCTURE PIPELINE

prc>tree /F /A | findstr /V "__pycache__" | findstr /V ".pyc"
C:.
|   .gitattributes
|   batch.py
|
+---analysing
|   |   clustering_lite.py
|   |   concordance_lite.py
|   |   hub_analysing.py
|   |   outliers_lite.py
|   |   regimes_lite.py
|   |   verdict.py
|   |
|   +---configs
|   |   \---regimes
|   |           regimes_default.yaml
|   |           regimes_laxe.yaml
|   |           regimes_strict.yaml
|   |
|
+---atomics
|   |   atomics_catalog.md
|   |
|   +---D_encodings
|   |   |   asy_001_random_asymmetric.py
|   |   |   asy_002_lower_triangular.py
|   |   |   asy_003_antisymmetric.py
|   |   |   asy_004_directional_gradient.py
|   |   |   asy_005_circulant.py
|   |   |   asy_006_sparse.py
|   |   |   r3_001_random_uniform.py
|   |   |   r3_002_partial_symmetric.py
|   |   |   r3_003_local_coupling.py
|   |   |   r3_004_fully_symmetric.py
|   |   |   r3_005_diagonal.py
|   |   |   r3_006_separable.py
|   |   |   r3_007_block_structure.py
|   |   |   sym_001_identity.py
|   |   |   sym_002_random_uniform.py
|   |   |   sym_003_random_gaussian.py
|   |   |   sym_004_correlation_matrix.py
|   |   |   sym_005_banded.py
|   |   |   sym_006_block_hierarchical.py
|   |   |   sym_007_uniform_correlation.py
|   |   |   sym_008_random_clipped.py
|   |   |
|   |   +---configs
|   |   |       D_encodings_default.yaml
|   |   |       D_encodings_laxe.yaml
|   |   |       D_encodings_strict.yaml
|   |   |
|   |
|   +---modifiers
|   |   |   m0_baseline.py
|   |   |   m1_gaussian_noise.py
|   |   |   m2_uniform_noise.py
|   |   |
|   |   +---configs
|   |   |       modifiers_default.yaml
|   |   |       modifiers_laxe.yaml
|   |   |       modifiers_strict.yaml
|   |   |
|   |   +---constraints
|   |   +---domains
|   |   +---plugins
|   |   |   +---biology
|   |   |   |       encoding.py
|   |   |   |
|   |   |   +---gravity
|   |   |   |       encoding.py
|   |   |   |
|   |   |   +---quantum
|   |   |   |   |   encoding.py
|   |   |   |   |
|   |   |   |   \---.ipynb_checkpoints
|   |   |   \---template
|   |   |           encoding.py
|   |   |
|   |
|   \---operators
|       |   gamma_hyp_001.py
|       |   gamma_hyp_002.py
|       |   gamma_hyp_003.py
|       |   gamma_hyp_004.py
|       |   gamma_hyp_005.py
|       |   gamma_hyp_006.py
|       |   gamma_hyp_007.py
|       |   gamma_hyp_008.py
|       |   gamma_hyp_009.py
|       |   gamma_hyp_010.py
|       |   gamma_hyp_012.py
|       |   gamma_hyp_013.py
|       |
|       +---configs
|       |       operators_default.yaml
|       |       operators_laxe.yaml
|       |       operators_strict.yaml
|       |
|
+---configs
|   |   pool_requirements.yaml
|   |
|   +---phases
|   |   \---poc
|   |           poc.yaml
|   |           poc2.yaml
|   |           poc_debug.yaml
|   |
|   \---tests
|           test_core.yaml
|
+---core
|   |   core_catalog.md
|   |   kernel.py
|   |   state_preparation.py
|   |
|
+---data
|   \---results
|           .gitkeep
|           poc.parquet
|           poc2.parquet
|
+---documentation
|       DIVERGENCES.md
|       PHASES_GUIDE.md
|       requirements.txt
|
+---featuring
|   |   extractor_lite.py
|   |   hub_featuring.py
|   |   layers_lite.py
|   |
|   +---configs
|   |   \---minimal
|   |           matrix_2d.yaml
|   |           matrix_square.yaml
|   |           symmetric_2d.yaml
|   |           tensor_3d.yaml
|   |           universal.yaml
|   |
|   +---registries
|   |   |   matrix_2d_lite.py
|   |   |   tensor_3d_lite.py
|   |   |   universal_lite.py
|   |   |
|   |
|
+---profiling
|   |   aggregation_lite.py
|   |   hub_profiling.py
|   |
|
+---reports
|       verdict_poc.json
|       verdict_poc.txt
|       verdict_poc2.json
|       verdict_poc2.txt
|
+---running
|   |   compositions.py
|   |   hub_running.py
|   |   runner.py
|   |
|
+---tests
|   |   test_analysing_lite.py
|   |   test_core.py
|   |   test_featuring_lite.py
|   |   test_hub_running.py
|   |   test_profiling_lite.py
|   |   test_registries_lite.py
|   |   test_runner.py
|   |   test_running_lite.py
|   |   test_verdict.py
|   |   test_verdict_integration.py
|   |
|
+---utils
|   |   database.py
|   |   data_loading_lite.py
|   |
|




## SECTION 1 : FONDATIONS IMMUABLES

### 1.1 Axiomes PRC

- **A1** : Dissymétrie informationnelle D irréductible
- **A2** : Mécanisme Γ agissant sur D
- **A3** : Aucun Γ stable ne peut annuler D complètement

**Hiérarchie niveaux** :
```
L0 (ontologique) → L1 (épistémique) → L2 (théorique) → L3 (opérationnel) → L4 (documentaire)
Règle : L(n) référence uniquement L(≤n-1)
Exception : L3 peut consommer L4 (descriptif), pas dériver règles
```

### 1.2 Logique expérimentale

Le pipeline crée des candidats atomiques Γ et les caractérise. 
Chaque phase teste des candidats :
    - **Conserver** : Pas de preuve négative
    - **Explorer** : Verdict pointe comportement particulier
    - **Exclure** : Preuve incompatibilité axiome/candidat

### 1.3 Core aveugle

Deux fonctions immuables, aveugles au domaine :
```
prepare_state(base, modifiers) → np.ndarray
run_kernel(composition, config) → np.ndarray  # history (T, *dims)
```

**Règles K1-K5 (inviolables)** :
- Aucune validation sémantique du contenu
- Aucune classe State ou Operator
- Aucun branchement dépendant de D ou Γ
- Aucune connaissance des atomics

---

## SECTION 2 : ARCHITECTURE GLOBALE

### 2.1 Flux pipeline

```
YAML config (phases, axes)
    ↓
Génération compositions (produit cartésien axes)
    ↓
Dry-run + confirmation utilisateur (o/n)
    ↓
Pour chaque composition :
    Kernel → history (RAM)
    Featuring → features scalaires (RAM)
    Écriture Parquet
    Verdict intra-run (en mémoire)
    ↓
Verdict inter-run optionnel (depuis Parquet) :
    Profiling → agrégation cross-runs
    Analysing → patterns ML
    ↓
Rapports
```

### 2.2 Responsabilités modules

**CORE** : Exécution aveugle. Ne connaît pas le domaine.

**ATOMICS** : Pool de candidats — gammas, encodings, modifiers. Chaque atomic expose `ID` et `create()`.

**FEATURING** : Extraction intra-run. Transforme history → features scalaires, en mémoire, pendant le batch.

**PROFILING** : Agrégation inter-run. Nécessite le contexte cross-runs. Opère depuis Parquet.

**ANALYSING** : Patterns ML. Clustering (stratifié par layer), outliers, variance, concordance cross-phases.

**RUNNING** : Orchestration batch. Génère compositions, pilote kernel + featuring + Parquet + verdict intra.

**CONFIGS** : YAML centralisés. Principe : un `configs/` global pour ce qui est transversal, un `configs/` par module pour ce qui lui est local.

### 2.3 Layers featuring

L'applicabilité est définie **par layer** dans le YAML du layer — pas par feature individuelle. 
Ajouter une feature = vérifier son layer, ajouter une fonction dans le registre correspondant. 
Le check applicabilité est hérité du layer entier.

| Layer | Condition |
|-------|-----------|
| `universal` | Tout tenseur |
| `matrix_2d` | rank == 2 |
| `matrix_square` | rank == 2 et carré |
| `tensor_3d` | rank ≥ 3 |
| `spatial_2d` | analyses spatiales 2D |

---

## SECTION 3 : MÉTHODOLOGIE VALIDATION

### 3.1 Processus Algo → Structure → Code

**VALIDATION OBLIGATOIRE à chaque étape — l'utilisateur valide avant de passer à la suivante.**

**Étape 1 — ALGO (langage courant)** :
- Ce qu'on fait, pourquoi, zéro code
- Permet à un non-programmeur de détecter les dérives métier tôt

**Étape 2 — STRUCTURE (squelette ancré)** :
- Signatures, I/O, dépendances, ancré sur le code existant réel
- Pas de code fonctionnel, juste la forme

**Étape 3 — CODE** :
- Implémentation dans le squelette validé
- Seulement après validation des deux étapes précédentes

### 3.2 Règles de travail

- **Toujours lire la documentation** avant de coder (catalogues, sources)
- **Jamais supposer** — demander les fichiers manquants
- **Jamais inventer un quick fix** sans vérification dans les sources
- **Toute déviation** aux principes de ce charter → discussion explicite avant implémentation

---

## SECTION 4 : PRINCIPES IMMUABLES

### P1 — Intra-run / Inter-run

- **Intra-run** : Tout ce qui opère sur une history unique en RAM — featuring, verdict intra.
- **Inter-run** : Tout ce qui nécessite le contexte cross-runs — profiling, analysing, verdict inter.

**Règle** : Ce qui peut être calculé intra-run l'est. L'inter-run est réservé à ce qui requiert explicitement plusieurs runs.

### P2 — Gestion erreurs

| Type | Cause | Action |
|------|-------|--------|
| Erreur code | Calcul non applicable | `raise ValueError` |
| Valeur aberrante | Hors domaine mathématique | `raise ValueError` |
| Explosion physique | Comportement système réel | `return np.nan` (signal, pas erreur) |

**NaN ≠ erreur. NaN = information** sur le comportement du candidat.

### P3 — Applicabilité par layer

L'applicabilité est une propriété du layer, définie en YAML. 
Un seul check par layer, pas par feature. 
Les registres d'un layer sont des fonctions pures (state → float), sans connaissance du reste du pipeline.

### P4 — YAML partout

Zéro hardcodé dans le code Python :
- Seuils → YAML analysing
- Params features → YAML featuring
- Configs phases + axes → YAML global

Tout paramètre kernel peut devenir un axe d'itération via YAML, sans modifier le pipeline Python.

### P5 — Wrappers robustes

Préférer les wrappers stdlib (numpy, scipy) aux implémentations customs. 
Les registres sont des fonctions pures sans dépendance interne PRC.

### P6 — Parquet par phase

Un fichier Parquet par phase. Une colonne par feature (charge partielle native). 
Écrit **avant** le verdict intra-run (resilience si crash verdict). Le verdict opère ensuite en mémoire sur les features déjà calculées.

### P7 — Clustering stratifié par layer

Pas de clustering global (réduit au plus petit dénominateur commun). Une passe par layer actif, sur des features homogènes. Paramétrable via YAML.

---

## SECTION 5 : INTERDICTIONS CRITIQUES

### Core
- ❌ Validation sémantique dans le core
- ❌ Classes State/Operator dans le core
- ❌ Branchements dépendant de D ou Γ

### Featuring
- ❌ Featuring → profiling ou analysing
- ❌ Registres → featuring (isolation totale)
- ❌ I/O dans les registres (Parquet, fichiers)
- ❌ Contexte cross-runs dans le featuring

### Général
- ❌ Dépendances circulaires
- ❌ Paramètres hardcodés (tout YAML)
- ❌ Code produit sans validation Algo→Structure→Code
- ❌ Déviation aux principes sans discussion explicite

### Nomenclature
- ❌ "matrice de corrélation" → ✅ "tenseur rang 2"
- ❌ "graphe" → ✅ "patterns dans tenseur"
- ❌ "d_encoding_id" → ✅ "encoding_id"
- ❌ "test verdict" → ✅ "observation" (featuring) / "verdict" (analysing)

---

## SECTION 6 : OBSERVATIONS DE VALIDATION

Les observations sont intégrées aux modules via le featuring : appeler un registre sur un state de test et observer la valeur retournée.

**Les observations ne retournent jamais** :
- PASS/FAIL, bon/mauvais
- Jugements normatifs
- Params hardcodés
- Classes ou objets

**Les observations retournent** : des valeurs numériques brutes, observables et interprétables.

---

## SECTION 7 : NAVIGATION DOCUMENTATION

### Bibliothèque permanente (chaque conversation)
```
CHARTER_7_1.md    # Ce document
```

### Catalogues (à demander selon tâche)
```
featuring/registries_catalog.md    # Fonctions registres disponibles
profiling/profiling_catalog.md     # Fonctions agrégation
analysing/analysing_catalog.md     # Clustering, outliers, variance
```

### Mapping tâche → doc

| Tâche | Demander |
|-------|----------|
| Ajouter une feature | registries_catalog.md + source du layer ciblé |
| Modifier seuils régimes | configs régimes du module analysing |
| Ajouter axe itération | configs phases global |
| Debugging pipeline | source runner.py + hub concerné |
| Concordance cross-phases | analysing_catalog.md + concordance.py |

---

## SECTION 8 : GLOSSAIRE

| Terme | Définition |
|-------|------------|
| **Intra-run** | Calculs sur history unique, en RAM |
| **Inter-run** | Calculs cross-runs, depuis Parquet |
| **Layer** | Catégorie features avec applicabilité commune |
| **Registre** | Module de fonctions pures (state → float) |
| **Projection** | Vue temporelle d'une feature (initial, final, mean, ...) |
| **Dynamic event** | Événement détecté sur la timeline (deviation, collapse, ...) |
| **Régime** | Classification comportement d'un run |
| **Timeline** | Descripteur compositionnel d'events |
| **Concordance** | Stabilité d'un verdict cross-phases |
| **Composition** | Dict axes d'un run `{gamma_id, encoding_id, ...}` |
| **History** | Séquence temporelle états `np.ndarray (T, *dims)` |
| **Observation** | Valeur numérique brute issue d'un registre |
| **Profil gamma** | Caractérisation Γ agrégée cross-runs |
| **Axe** | Dimension d'itération configurable via YAML |

---

**FIN CHARTER PRC 7.1**

*Référence cognitive — pas une source de vérité sur le code.*
*Pour les détails d'implémentation : sources et catalogues.*
