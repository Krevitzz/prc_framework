# DIVERGENCES — Charter 7.0 vs Nouveau Pipeline

> Tout écart intentionnel au Charter 7.0 doit être documenté ici
> **AVANT** d'être implémenté.
>
> Date création : 2026-02-17
> Version charter de référence : 7.0

---

## Règle de gestion

1. Identifier l'écart avec le charter
2. Le documenter dans ce fichier (ID, description, justification, impact)
3. Faire valider par l'utilisateur
4. Seulement ensuite implémenter

Aucun écart silencieux. Aucun "quick fix" non documenté.

---

## Divergences actives

### D1 — Format stockage résultats

| Champ | Valeur |
|-------|--------|
| **Référence charter** | Section 4.6 — DB unique SQLite `db_results.db` |
| **Nouveau pipeline** | Fichiers Parquet dans `prc/data/results/` (1 fichier par phase) |
| **Justification** | Volumétrie -71% (339 MB vs 1 171 MB), RAM verdict ×22 réduit (89 MB vs 2 GB), I/O 2-5× plus rapide, pas de migrations schema |
| **Impact** | `utils/database.py` utilise pandas/pyarrow au lieu de sqlite3. Queries via pandas filtering au lieu de SQL. |
| **Validé par** | Utilisateur — 2026-02-17 |

---

### D2 — Organisation orchestration batch

| Champ | Valeur |
|-------|--------|
| **Référence charter** | Section 2.3 — `batch_runner.py` et `verdict.py` à la racine |
| **Nouveau pipeline** | Module `prc/running/` contenant hub, compositions, discovery, runner, verdict |
| **Justification** | Séparation des responsabilités, testabilité unitaire, extensibilité (nouveaux axes sans modifier point d'entrée) |
| **Impact** | `batch.py` racine devient façade légère appelant `running/hub.py`. Charter Section 2.1 organigramme flux reste valide conceptuellement. |
| **Validé par** | Utilisateur — 2026-02-17 |

---

### D3 — Format colonnes features

| Champ | Valeur |
|-------|--------|
| **Référence charter** | Section 4.6 — `features TEXT NOT NULL` (JSON colonne unique) |
| **Nouveau pipeline** | Une colonne Parquet par feature (ex: `frobenius_norm_final`, `mean_value_final`, ...) |
| **Justification** | Charge partielle native (lire 3 colonnes sur 150 sans désérialiser JSON), RAM critique pour verdict, cohérent avec D1 (Parquet) |
| **Impact** | Pas de migration schema au sens SQL — ajout feature = nouvelle colonne Parquet. Profiling/Analysing lisent colonnes directement via pandas. |
| **Validé par** | Utilisateur — 2026-02-17 |

---

### D4 — METADATA atomics minimal

| Champ | Valeur |
|-------|--------|
| **Référence charter** | Section 2.3 — METADATA complet (id, name, family, form, parameters, expected_behavior, notes...) |
| **Nouveau pipeline** | METADATA réduit à `id` + `d_applicability` uniquement |
| **Justification** | Tout ce qui est chargé en RAM doit être nécessaire au pipeline. name, form, notes, expected_behavior sont de la doc humaine — leur place est dans les docstrings/commentaires du fichier, pas en mémoire à chaque run. |
| **Impact** | Discovery ne charge que `id` et `d_applicability`. Catalogues .md restent la référence doc humaine. |
| **Validé par** | Utilisateur — 2026-02-17 |

---

### D5 — PARAM_GRID supprimé des atomics

| Champ | Valeur |
|-------|--------|
| **Référence charter** | Section implicite — grilles paramètres dans les fichiers gamma |
| **Nouveau pipeline** | Valeurs par défaut dans `configs/atomics/{operators,D_encodings,modifiers}/defaults.yaml` |
| **Justification** | R-YAML-1 : zéro hardcodé. PARAM_GRID dans le Python est une violation directe du principe YAML partout. |
| **Impact** | Les fichiers atomics ne contiennent plus de grilles. `configs/atomics/*/defaults.yaml` contient les valeurs nominales. Le YAML de run surcharge si besoin. |
| **Validé par** | Utilisateur — 2026-02-17 |

---

### D6 — Convention callable unifiée `create()`

| Champ | Valeur |
|-------|--------|
| **Référence charter** | Section 2.3 — modifiers exposent `apply()`, gammas exposent `create_gamma_hyp_NNN()` |
| **Nouveau pipeline** | Tous les atomics (gammas, encodings, modifiers) exposent `create()` |
| **Justification** | Discovery identique pour les 3 types, pas de logique conditionnelle sur le nom du callable, cohérence totale. |
| **Impact** | `apply()` renommé `create()` dans tous les modifiers. Factories gammas `create_gamma_hyp_NNN()` renommées `create()`. `utils/data_loading.py` utilise `create` partout. |
| **Validé par** | Utilisateur — 2026-02-17 |

---

### D7 — Discovery dans `utils/data_loading.py`

| Champ | Valeur |
|-------|--------|
| **Référence charter** | Section 2.3 (D2) — `running/discovery.py` prévu |
| **Nouveau pipeline** | Discovery vit dans `utils/data_loading.py` (source unique de vérité pour tout chargement) |
| **Justification** | Discovery est une opération de chargement de données, pas d'orchestration. La séparer dans `running/` créait un couplage inutile et une confusion de responsabilités. |
| **Impact** | `running/discovery.py` stub supprimé ou vidé. `running/hub.py` appelle `utils/data_loading.discover_*()`. |
| **Validé par** | Utilisateur — 2026-02-17 |

---

### D8 — Deux axes seed distincts : `seed_CI` et `seed_run`

| Champ | Valeur |
|-------|--------|
| **Référence charter** | Section 4.4 — `seed` comme axe temporaire unique |
| **Nouveau pipeline** | Deux axes : `seed_CI` (conditions initiales → encodings) et `seed_run` (perturbations → gammas stochastiques + modifiers) |
| **Justification** | Un seed unique mélange deux sources de variance distinctes : sensibilité aux CI vs sensibilité aux perturbations en cours de run. Les séparer permet des analyses R1 propres. Pour R0 les deux peuvent être fixés à la même valeur. |
| **Impact** | `runner.py` passe `seed_CI` à `create()` des encodings, `seed_run` à `create()` des gammas et modifiers. Les atomics déterministes ignorent silencieusement les seeds. YAML de run expose les deux axes indépendamment. |
| **Note** | Pas une divergence au sens strict — le charter ne prescrit pas le nombre de seeds. Documenté ici pour traçabilité de la décision. |
| **Validé par** | Utilisateur — 2026-02-17 |

---

## Divergences archivées

_(aucune pour l'instant)_
