# Implémentation Scoring & Verdicts R0

## 📋 Vue d'ensemble

Implémentation complète du système de scoring pathologies et verdicts R0 selon Charter 5.4 (Sections 12.8-12.9).

**Date**: 2025-01-02  
**Version**: 5.4  
**Statut**: ✅ Implémentation complète, tests à valider

---

## 📦 Fichiers créés/modifiés

### Nouveaux fichiers core

1. **tests/utilities/scoring_engine.py**
   - Moteur scoring pathologies (S1-S4, MAPPING)
   - Agrégation métriques → test_score
   - Validation règles scoring

2. **tests/utilities/verdict_engine.py** (complet)
   - Calcul verdict global gamma
   - Agrégation tous tests/runs
   - Génération rapport détaillé

3. **tests/utilities/scoring.py** (mis à jour)
   - Intégration scoring_engine
   - Conversion observations → scores

### Configs YAML

4. **tests/config/global/scoring_default_v1.yaml**
   - Règles scoring tous tests
   - Poids tests
   - Mode agrégation

5. **tests/config/global/thresholds_default_v1.yaml** (mis à jour)
   - Seuils verdicts R0
   - Logique OU/ET explicite

6. **tests/config/global/thresholds_strict_v1.yaml** (mis à jour)
   - Version stricte pour exploration ciblée

7. **tests/config/global/thresholds_lenient_v1.yaml** (mis à jour)
   - Version permissive pour exploration large

### Migration DB

8. **prc_automation/prc_database/schema_results_migration.sql**
   - Migration schema TestScores
   - Migration schema GammaVerdicts
   - Backup automatique

### Batch runner

9. **prc_automation/batch_runner.py** (snippet)
   - Fonctions run_batch_test (updated)
   - Fonctions run_batch_verdict (updated)
   - Stockage nouveau schema

### Tests validation

10. **tests/test_scoring_validation.py**
    - Tests unitaires scoring_engine
    - Tests S1-S4 + MAPPING
    - Tests agrégation
    - Golden set synthétique

---

## 🚀 Installation

### 1. Appliquer migration DB

```bash
# Backup DB actuelle
cp prc_database/prc_r0_results.db prc_database/prc_r0_results_backup.db

# Appliquer migration
sqlite3 prc_database/prc_r0_results.db < prc_automation/prc_database/schema_results_migration.sql

# Vérifier
sqlite3 prc_database/prc_r0_results.db ".schema TestScores"
sqlite3 prc_database/prc_r0_results.db ".schema GammaVerdicts"
```

### 2. Valider scoring_engine

```bash
# Lancer tests validation
python tests/test_scoring_validation.py

# Sortie attendue:
# 🎉 TOUS LES TESTS PASSENT (6/6)
```

### 3. Tester workflow complet

```bash
# 1. Collecte données (si pas déjà fait)
python batch_runner.py --brut --gamma GAM-001

# 2. Appliquer tests + scoring
python batch_runner.py --test \
  --gamma GAM-001 \
  --params params_default_v1 \
  --scoring scoring_default_v1

# 3. Calculer verdict
python batch_runner.py --verdict \
  --gamma GAM-001 \
  --params params_default_v1 \
  --scoring scoring_default_v1 \
  --thresholds thresholds_default_v1

# 4. Ou tout en une commande
python batch_runner.py --all \
  --gamma GAM-001 \
  --params params_default_v1 \
  --scoring scoring_default_v1 \
  --thresholds thresholds_default_v1
```

---

## 📊 Architecture scoring

### Types pathologies (S1-S4 + MAPPING)

```
S1_COLLAPSE      → Valeur trop faible (entropy, variance, rank)
S2_EXPLOSION     → Valeur trop élevée (norm, spectral_radius)
S3_PLATEAU       → Intervalle toxique (uniformity ≈ 1)
S4_INSTABILITY   → Variation brutale (Δmetric)
MAPPING          → Catégoriel (transition, trend)
```

### Échelle normalisée

```
Interne: pathology_score ∈ [0, 1]
  0 = sain (aucune pathologie)
  1 = pathologique (pathologie maximale)

Affichage: /20 (score × 20)
  0/20 = sain
  20/20 = pathologique
```

### Agrégation

```python
# R0 default: max (une pathologie suffit)
test_score = max(metric_pathology_scores)

# Future R1: weighted_mean (nuancé)
test_score = Σ(score_i × weight_i) / Σ(weight_i)
```

---

## ⚖️ Architecture verdicts

### Critères (3)

```python
majority_pct   = (n_configs_saines / n_total) × 100
robustness_pct = (n_D_viables / n_total_D) × 100
global_score   = mean(all_test_scores)
```

### Logique verdicts

```
SURVIVES[R0]   : ≥1 critère passe (OU logique, permissif)
REJECTED[R0]   : TOUS critères échouent (ET logique, strict)
WIP[R0-open]   : Ni SURVIVES ni REJECTED
```

### Seuils default

```yaml
threshold_test: 0.70  # Config saine si test_score < 0.70

survives:
  score_max: 0.40       # global_score < 0.40
  robustness_min: 0.60  # 60% D viables
  majority_min: 0.50    # 50% configs saines

rejected:
  score_min: 0.60       # global_score > 0.60
  robustness_max: 0.40  # <40% D viables
  majority_max: 0.20    # <20% configs saines
```

---

## 🔧 Utilisation avancée

### Créer config scoring custom

```yaml
# tests/config/tests/UNIV-001/scoring_custom_v1.yaml
version: "1.0"
config_id: "UNIV-001_scoring_custom_v1"

# Override seuils
stat_final:
  threshold_high: 500.0  # Plus strict
  critical_high: 5000.0
```

### Créer config thresholds custom

```yaml
# tests/config/global/thresholds_ultra_strict_v1.yaml
version: "1.0"
config_id: "thresholds_ultra_strict_v1"

threshold_test: 0.50  # Très strict

survives:
  score_max: 0.20
  robustness_min: 0.80
  majority_min: 0.70
```

### Comparer configs

```bash
# Config default
python batch_runner.py --verdict \
  --gamma GAM-001 \
  --thresholds thresholds_default_v1

# Config strict
python batch_runner.py --verdict \
  --gamma GAM-001 \
  --thresholds thresholds_strict_v1

# Config lenient
python batch_runner.py --verdict \
  --gamma GAM-001 \
  --thresholds thresholds_lenient_v1

# Comparer rapports
diff reports/verdict_GAM-001_*_default_*.txt \
     reports/verdict_GAM-001_*_strict_*.txt
```

---

## 📈 Workflow complet

```
1. db_raw existe (runs kernel)
   ↓
2. batch_runner --test
   → Applique tests
   → Score métriques (scoring_engine)
   → Stocke TestScores
   ↓
3. batch_runner --verdict
   → Agrège tous TestScores
   → Calcule 3 critères
   → Applique règles verdict
   → Génère rapport détaillé
   → Stocke GammaVerdicts
   ↓
4. Rapport dans reports/verdict_*.txt
```

---

## 🎯 Validation

### Tests obligatoires avant production

```bash
# 1. Tests unitaires scoring_engine
python tests/test_scoring_validation.py
# → Doit afficher: 🎉 TOUS LES TESTS PASSENT (6/6)

# 2. Test intégration (si runs disponibles)
python batch_runner.py --all \
  --gamma GAM-TEST \
  --params params_default_v1 \
  --scoring scoring_default_v1 \
  --thresholds thresholds_default_v1

# Vérifier rapport généré
cat reports/verdict_GAM-TEST_*.txt

# 3. Vérifier DB
sqlite3 prc_database/prc_r0_results.db "SELECT * FROM TestScores LIMIT 5"
sqlite3 prc_database/prc_r0_results.db "SELECT * FROM GammaVerdicts"
```

### Checklist validation

- [ ] Migration DB appliquée sans erreur
- [ ] Tests scoring_engine passent (6/6)
- [ ] TestScores contient données après --test
- [ ] GammaVerdicts contient verdict après --verdict
- [ ] Rapport généré dans reports/
- [ ] Rapport lisible et complet
- [ ] Verdicts cohérents (SURVIVES/REJECTED/WIP)

---

## 🐛 Dépannage

### Erreur: "Table TestScores has no column test_score"

```bash
# Migration non appliquée
sqlite3 prc_database/prc_r0_results.db < schema_results_migration.sql
```

### Erreur: "Unknown pathology_type"

```yaml
# Vérifier YAML scoring
# pathology_type doit être: S1_COLLAPSE | S2_EXPLOSION | S3_PLATEAU | S4_INSTABILITY | MAPPING
```

### Erreur: "Test XXX not found in scoring config"

```yaml
# Ajouter test dans scoring_default_v1.yaml
tests:
  XXX-001:
    scoring_rules:
      # ...
```

### Verdict toujours WIP

```bash
# Vérifier seuils thresholds
# Ajuster vers lenient si trop strict
python batch_runner.py --verdict --thresholds thresholds_lenient_v1
```

---

## 📚 Références

- **Charter 5.4** : Sections 12.8-12.9
- **scoring_engine.py** : Code complet types S1-S4
- **verdict_engine.py** : Algorithme verdict + rapport
- **scoring_default_v1.yaml** : Règles scoring tous tests

---

## 🔜 Prochaines étapes

### Immédiat (validation)

1. [ ] Lancer test_scoring_validation.py
2. [ ] Tester sur GAM-001 si runs disponibles
3. [ ] Vérifier rapport généré
4. [ ] Ajuster seuils si nécessaire

### Court terme (production)

1. [ ] Intégrer dans CI/CD
2. [ ] Automatiser génération rapports
3. [ ] Dashboard visualisation verdicts
4. [ ] Comparaison multi-configs

### Moyen terme (R1)

1. [ ] Affiner règles scoring (empirique)
2. [ ] Nouveaux types pathologies (S5+)
3. [ ] Verdicts composites (R1)
4. [ ] Métriques cross-domaines

---

**Statut**: ✅ Implémentation complète  
**Prêt pour**: Tests validation  
**Blocage**: Aucun  
**Contact**: Voir charter 5.4