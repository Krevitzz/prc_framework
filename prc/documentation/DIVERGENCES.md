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

### D4 — Format fichiers atomics

| Champ | Valeur |
|-------|--------|
| **Référence charter** | Catalogs legacy — `PHASE`, `METADATA` dict complet, `PARAM_GRID`, `create_gamma_hyp_NNN()`, `d_applicability` |
| **Nouveau pipeline** | `ID = "XXX-NNN"` uniquement, `create()` uniforme, défauts dans signature, paramètres dans YAML, applicabilité par runtime catch |
| **Justification** | Séparation mécanique pure (Python) / configuration expérimentale (YAML). Supprime couplage fort entre fichiers atomics et phases d'exploration. Cohérence avec principe YAML partout (Charter 4.4). |
| **Impact** | Discovery cherche `ID` et `create()`/`apply()` — plus `PHASE` ni `METADATA`. Catalogs legacy (`gamma_catalog.md`, `d_encoding_catalog.md`, `modifier_catalog.md`) remplacés par `atomics/atomics_catalog.md`. Fichiers legacy dans `atomics/` sont brouillons à réécrire. |
| **Validé par** | Utilisateur — 2026-02-17 |

---

## Divergences archivées

_(aucune pour l'instant)_

---

## Extensions futures (questions ouvertes)

### E1 — Composition mathématique encodings

**Contexte :** Multi-lignes sous `encoding:` dans le YAML de run génère actuellement des alternatives à tester séparément. Contrairement aux gammas (opérateurs itératifs `D → D'`), les encodings sont des générateurs `∅ → D_0` — pas de sémantique de composition séquentielle évidente.

**Question :** Quelle opération mathématique pour composer deux encodings ?

**Candidats explorés :**
- **Moyenne pondérée** : `(w1·D1 + w2·D2)` — préserve rang/dimension, mais perd propriétés (SYM + ASY → ni SYM ni ASY)
- **Produit Hadamard** : `D1 ⊙ D2` (élément par élément) — préserve rang/dimension
- **Produit tensoriel** : `D1 ⊗ D2` — change le rang (incompatible gammas rang 2)
- **Produit Kronecker** : `D1 ⊗ D2` — préserve rang mais dimension explose (10×10 → 100×100)

**Décision :** Suspendue — nécessite exploration propriétés espace des possibles. Pour l'instant, multi-lignes encodings = alternatives seulement, pas de composition.

**Implémentation future :** `ComposedEncoding` analogue à `ComposedGamma`, une fois opération validée scientifiquement.

**Référence :** Discussion 2026-02-18, `compositions.py`

---

### E2 — Pondération axes de composition

**Contexte :** `weights` comme sous-axe optionnel permettant combinaisons linéaires pondérées de gammas composés. Implémenté pour gamma uniquement.

**Question ouverte :** Extension à d'autres axes (encodings, modifiers) une fois sémantique de composition définie (cf. E1).

**Format YAML proposé :**
```yaml
axes:
  gamma:
    - [GAM-001, GAM-009]
    - all
    weights:
      range: [0, 1]
      step: 0.1
```

**Explosion combinatoire :** 2×12 séquences × 11² poids = 2,904 compositions de gammas. Nécessite ménage atomics avant utilisation production.

**Référence :** Discussion 2026-02-18, `compositions.py`

