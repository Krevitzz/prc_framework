# PRC Batch Runner - Pipeline R0

Pipeline d'exploration exhaustive selon **Charte PRC 5.1 - Section 14**.

## Vue d'ensemble

Le batch runner est le **point d'entrée unique** pour l'exploration R0. Il gère trois modes avec **chaînage automatique** des dépendances :

```
--brut    → Collecte données runs (écrit db_raw)
--test    → Application tests + scoring (écrit db_results)
--verdict → Calcul verdicts agrégés (écrit db_results)
```

## Architecture des bases de données (Section 14.3)

### db_raw (données factuelles)
- **Fichier** : `prc_database/prc_r0_raw.db`
- **Tables** :
  - `Executions` - Métadonnées runs
  - `Snapshots` - États sauvegardés
  - `Metrics` - Métriques par itération
- **Propriétés** :
  - Append-only (jamais de UPDATE/DELETE)
  - Immuable (une fois écrit, jamais modifié)
  - Source de vérité unique

### db_results (analyses et verdicts)
- **Fichier** : `prc_database/prc_r0_results.db`
- **Tables** :
  - `TestObservations` - Observations brutes tests
  - `TestScores` - Scores calculés avec config
  - `GammaVerdicts` - Verdicts agrégés
- **Propriétés** :
  - Rejouable (peut être supprimée et reconstruite)
  - Versionnée (config_id, threshold_id)
  - Multiple verdicts coexistent

## Installation

### 1. Initialiser les bases de données

```bash
cd prc_automation
python init_databases.py
```

Pour réinitialiser complètement :
```bash
python init_databases.py --reset
```

### 2. Vérifier les configurations

**Pondérations tests** (Section 14.5) :
- `config/weights_default.yaml` ✓

**Seuils verdicts** (Section 14.6) :
- `config/thresholds_default.yaml` ✓

## Usage du pipeline

### MODE 1 : Collecte données (--brut)

Exécute le kernel pour tous (D, modifier, seed) applicables et stocke dans `db_raw`.

```bash
python batch_runner.py --brut --gamma GAM-001
```

**Ce qui se passe** :
1. Charge grille paramètres Phase 1 depuis `operators/gamma_hyp_001.py`
2. Filtre D applicables selon `d_applicability` du Γ
3. Exécute kernel pour chaque combinaison (params, D, modifier, seed)
4. Stocke métriques et snapshots dans `db_raw`
5. Skip les runs existants (append-only)

**Sorties** :
- `db_raw.Executions` : Métadonnées (status, iterations, temps, ...)
- `db_raw.Snapshots` : États sauvegardés tous les 10 itérations
- `db_raw.Metrics` : Métriques (norme, diversité, ...) par itération

### MODE 2 : Application tests (--test)

Applique tests applicables et calcule scores avec config spécifiée.

```bash
python batch_runner.py --test --gamma GAM-001 --config weights_default
```

**Chaînage automatique** :
- Si runs manquants dans `db_raw` → exécute `--brut` d'abord

**Ce qui se passe** :
1. Vérifie existence runs dans `db_raw`
2. Pour chaque run :
   - Charge historique depuis snapshots
   - Détermine tests applicables (matrice applicabilité)
   - Applique `run_all_applicable_tests()`
   - Calcule scores avec `score_all_observations()`
3. Stocke observations et scores dans `db_results`
4. Skip les runs déjà testés pour cette config

**Sorties** :
- `db_results.TestObservations` : Observations brutes (JSON)
- `db_results.TestScores` : Scores 0-1 + pondérations

**Rejouabilité** :
```bash
# Tester avec config différente SANS refaire les runs
python batch_runner.py --test --gamma GAM-001 --config weights_conservative
```

### MODE 3 : Calcul verdicts (--verdict)

Calcule verdicts agrégés selon les 3 critères et seuils YAML.

```bash
python batch_runner.py --verdict --gamma GAM-001 \
    --config weights_default \
    --thresholds thresholds_default
```

**Chaînage automatique** :
- Si scores manquants → exécute `--test` d'abord
- Si runs manquants → `--test` exécutera `--brut` d'abord

**Ce qui se passe** :
1. Vérifie existence scores dans `db_results`
2. Calcule les 3 critères (Section 7 Feuille de Route) :
   - **Majorité** : % configs PASS
   - **Robustesse** : % D avec ≥1 config viable
   - **Score global** : Moyenne pondérée /20
3. Charge seuils depuis `thresholds_default.yaml`
4. Applique logique verdict :
   - **SURVIVES[R0]** si OU logique satisfait
   - **FLAGGED_FOR_REVIEW** si ET logique insatisfait
   - **WIP[R0-closed]** par défaut
5. Stocke verdict dans `db_results.GammaVerdicts`

**Sorties** :
- `db_results.GammaVerdicts` : Verdict + 3 critères + raison

**Rejouabilité** :
```bash
# Recalculer verdict avec seuils différents SANS refaire scoring
python batch_runner.py --verdict --gamma GAM-001 \
    --config weights_default \
    --thresholds thresholds_strict
```

### MODE 4 : Chaîne complète (--all)

Exécute les 3 modes en séquence.

```bash
python batch_runner.py --all --gamma GAM-001 \
    --config weights_default \
    --thresholds thresholds_default
```

**Équivalent de** :
```bash
python batch_runner.py --brut --gamma GAM-001
python batch_runner.py --test --gamma GAM-001 --config weights_default
python batch_runner.py --verdict --gamma GAM-001 --config weights_default --thresholds thresholds_default
```

## Exemples d'utilisation

### Exploration complète d'un Γ

```bash
# 1. Collecte données
python batch_runner.py --brut --gamma GAM-001

# 2. Tester avec config nominale
python batch_runner.py --test --gamma GAM-001 --config weights_default

# 3. Verdict avec seuils nominaux
python batch_runner.py --verdict --gamma GAM-001 \
    --config weights_default \
    --thresholds thresholds_default

# 4. Comparer avec seuils stricts (SANS refaire runs ni scoring)
python batch_runner.py --verdict --gamma GAM-001 \
    --config weights_default \
    --thresholds thresholds_strict
```

### Tester plusieurs configs scoring

```bash
# 1. Données déjà collectées
# 2. Scorer avec config conservative
python batch_runner.py --test --gamma GAM-001 --config weights_conservative

# 3. Scorer avec config structurale
python batch_runner.py --test --gamma GAM-001 --config weights_structural

# 4. Comparer verdicts
python batch_runner.py --verdict --gamma GAM-001 --config weights_conservative --thresholds thresholds_default
python batch_runner.py --verdict --gamma GAM-001 --config weights_structural --thresholds thresholds_default
```

### Exploration batch de tous les Γ

```bash
# Boucle sur tous les Γ implémentés
for gamma in GAM-001 GAM-002 GAM-003 GAM-004 GAM-005 GAM-006 GAM-007 GAM-008 GAM-009 GAM-010 GAM-012 GAM-013; do
    python batch_runner.py --all --gamma $gamma \
        --config weights_default \
        --thresholds thresholds_default
done
```

## Requêtes sur les résultats

### Interroger db_raw

```bash
# Résumé global
python query_raw_data.py --summary

# Runs d'un Γ spécifique
python query_raw_data.py --gamma GAM-001

# Détails d'un run
python query_raw_data.py --run GAM-001_beta2.0_SYM-001_M0_s1

# Métriques temporelles
python query_raw_data.py --metrics GAM-001_beta2.0_SYM-001_M0_s1
```

### Interroger db_results

```bash
# TODO: Créer query_results.py
```

## Principe de non-reruns (Section 14.1)

**RÈGLE CRITIQUE** : Toute modification de scoring ou verdict doit être possible **SANS réexécuter les runs du kernel**.

### Modifications autorisées SANS reruns

✅ Changer pondérations tests → Modifier `weights_XXX.yaml` → Relancer `--test`

✅ Changer seuils verdicts → Modifier `thresholds_XXX.yaml` → Relancer `--verdict`

✅ Ajouter nouveau test → Implémenter dans `tests/utilities/` → Relancer `--test`

✅ Corriger bug scoring → Modifier `scoring.py` → Relancer `--test`

### Modifications nécessitant reruns

❌ Changer Γ (paramètres, formule) → Relancer `--brut`

❌ Changer D (générateur, params) → Relancer `--brut`

❌ Changer max_iterations → Relancer `--brut`

## Structure des fichiers

```
prc_automation/
├── batch_runner.py          # Point d'entrée unique (3 modes)
├── init_databases.py        # Initialisation bases
├── query_raw_data.py        # Requêtes db_raw
└── README.md               # Ce fichier

config/
├── weights_default.yaml     # Pondérations nominales
├── weights_conservative.yaml # Alternative (à créer)
├── thresholds_default.yaml  # Seuils nominaux
└── thresholds_strict.yaml   # Alternative (à créer)

prc_database/
├── schema_raw.sql           # Schéma db_raw
├── schema_results.sql       # Schéma db_results
├── prc_r0_raw.db           # Base données factuelles
└── prc_r0_results.db       # Base analyses/verdicts

tests/utilities/
├── __init__.py              # Exports + run_all_applicable_tests()
├── applicability.py         # Matrice applicabilité
├── scoring.py               # Fonctions scoring 0-1
├── test_norm.py             # Tests norme
├── test_symmetry.py         # Tests symétrie
├── test_diversity.py        # Tests diversité
└── test_convergence.py      # Tests convergence
```

## Conformité Charte PRC 5.1

### Section 14.1 : Principe de non-reruns ✓
- Modifications scoring/verdict SANS reruns kernel

### Section 14.2 : Architecture batch_runner ✓
- Un seul point d'entrée
- Trois modes avec chaînage automatique

### Section 14.3 : Séparation bases de données ✓
- db_raw : données factuelles, append-only
- db_results : analyses, rejouable

### Section 14.4 : Tests et scoring ✓
- Tests observent (observations brutes)
- Scoring interprète (contexte + pondérations)
- Séparation stricte

### Section 14.5 : Configs externes ✓
- Pondérations dans YAML
- Aucun hardcode dans le code

### Section 14.6 : Verdicts multiples ✓
- Plusieurs verdicts coexistent
- Identifiés par (gamma_id, config_id, threshold_id)

### Section 14.7 : Validation ⏳
- Scripts validation à créer

## Support

Pour questions/bugs :
1. Vérifier conformité Section 14 de la Charte
2. Consulter ce README
3. Examiner commentaires dans le code
4. Demander

## Licence

Conforme PRC Charter 5.1 - Section 14