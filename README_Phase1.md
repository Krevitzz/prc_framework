# PRC Framework - Phase 1 (Balayage Nominal) - Guide d'Utilisation

**Version**: 1.0.0  
**Date**: 2024-12-25  
**Statut**: EN COURS D'IMPLÉMENTATION

---

## 📋 Objectif Phase 1

**Balayage nominal exhaustif** : Tester tous les opérateurs Γ avec paramètres par défaut uniquement.

### Configuration Phase 1

- **Γ**: GAM-001 à GAM-014 (paramètres nominaux)
- **D**: 13 bases (SYM×6, ASY×4, R3×3)
- **Modifiers**: M0, M1, M2, M3
- **Seeds**: 1, 2, 3, 4, 5
- **Total estimé**: ~3,640 runs

### Objectifs

1. **Identifier Γ prometteurs** : Score ≥ 12/20, robustesse ≥ 60%
2. **Classifier Γ** : Prometteurs (60%+) / Ambigus (40-60%) / Échec (<40%)
3. **Sélectionner pour Phase 2** : 5-8 Γ pour exploration paramétrique complète

---

## 🗂️ Nouveaux Fichiers

### ✅ **CRÉÉS**

```
operators/
├── gamma_hyp_002.py          ✅ Diffusion pure
├── gamma_hyp_003.py          ✅ Croissance exponentielle
├── gamma_hyp_004.py          ✅ Décroissance exponentielle
├── gamma_hyp_005.py          ✅ Oscillateur harmonique
├── gamma_hyp_006.py          ✅ Saturation + mémoire
├── gamma_hyp_007.py          ✅ Régulation moyenne
├── gamma_hyp_008.py          ✅ Mémoire différentielle
├── gamma_hyp_009.py          ✅ Bruit additif
├── gamma_hyp_010.py          ✅ Bruit multiplicatif
├── gamma_hyp_011.py          ⏳ Branchement tensoriel (À créer)
├── gamma_hyp_012.py          ✅ Préservation symétrie
├── gamma_hyp_013.py          ✅ Hebbien
└── gamma_hyp_014.py          ⏳ Projection sous-espace (À créer)

prc_database/
└── schema.sql                ✅ Schéma SQLite complet

prc_automation/
├── batch_runner.py           ✅ Orchestration Phase 1
└── report_generator.py       ✅ Génération rapports

README_Phase1.md              ✅ Ce fichier
```

---

## 🚀 Installation et Setup

### 1. Vérifier Phase 0 complète

```bash
python validate_phase0.py
```

**Attendu** : ✓ PHASE 0 PRÊTE POUR EXÉCUTION

### 2. Initialiser la base de données

```bash
cd prc_framework
python prc_automation/batch_runner1.py --init-db
```

**Crée** : `prc_database/prc_r0_results.db` avec schéma complet

### 3. Vérifier les opérateurs implémentés

```python
from operators import OPERATOR_REGISTRY

for op_id, info in OPERATOR_REGISTRY.items():
    status = "✅" if info['implemented'] else "⏳"
    print(f"{status} {op_id}: {info['name']}")
```

---

## 🎯 Utilisation

### Mode 1 : Exécuter un Γ spécifique

```bash
python prc_automation/batch_runner1.py --phase 1 --gamma GAM-001 --verbose
```

**Exécute** :
- GAM-001 avec paramètres nominaux
- Sur tous les D applicables
- Tous modifiers M0-M3
- Tous seeds 1-5

**Durée estimée** : 
- GAM-001 : ~260 runs × 5s = 20-30 minutes
- GAM-002 : ~200 runs × 5s = 15-20 minutes (rang 2 seulement)

### Mode 2 : Exécuter tous les Γ implémentés

```bash
python prc_automation/batch_runner1.py --phase 1 --all
```

**Attention** : Avec 2 Γ implémentés, ~460 runs, 40-50 minutes

### Mode 3 : Reprendre après interruption

```bash
python prc_automation/batch_runner1.py --phase 1 --gamma GAM-001
```

**Comportement** : Saute automatiquement les runs déjà complétés (stockés en DB)

---

## 📊 Génération de Rapports

### Résumé global

```bash
python prc_automation/report_generator.py --summary
```

**Affiche** :
```
RAPPORT GLOBAL - PHASE 1

Statistiques globales:
  Total exécutions: 260
  Complétées: 254 (97.7%)
  Erreurs: 6 (2.3%)
  Temps moyen: 4.8s

Verdicts:
  PASS: 180 (70.9%)
  POOR: 45 (17.7%)
  REJECTED: 29 (11.4%)

RÉSUMÉ PAR Γ
──────────────────────────────────────────────────────────────────────
Γ            Runs     PASS     POOR     REJ      Score
──────────────────────────────────────────────────────────────────────
GAM-001      254      180      45       29       70.9%
GAM-002      200      50       80       70       25.0%
```

### Rapport détaillé pour un Γ

```bash
python prc_automation/report_generator.py --gamma GAM-001
```

**Affiche** :
```
RAPPORT DÉTAILLÉ - GAM-001

Configuration: {"beta": 2.0}
  Total runs: 260
  Complétés: 254
  Score global: 70.9%
  ✓ Classification: PROMETTEUR

MATRICE GAM-001 × D_base
──────────────────────────────────────────────────────────────────────
D_base       Runs     PASS     Taux
──────────────────────────────────────────────────────────────────────
SYM-001      20       18       90.0%
SYM-002      20       15       75.0%
ASY-001      20       8        40.0%
...
```

### Matrice complète Γ × D

```bash
python prc_automation/report_generator.py --matrix
```

**Affiche** :
```
MATRICE COMPLÈTE Γ × D (taux de succès)

Γ               SYM-001   SYM-002   ASY-001   R3-001    ...
──────────────────────────────────────────────────────────────────────
GAM-001         90.0%     75.0%     40.0%     60.0%     ...
GAM-002         25.0%     20.0%     15.0%     N/A       ...
```

### Tests échouant fréquemment

```bash
python prc_automation/report_generator.py --failing-tests
```

**Identifie** : Quels tests échouent systématiquement (>10% échec)

### Rapport complet

```bash
python prc_automation/report_generator.py --all
```

**Combine** : Tous les rapports ci-dessus

---

## 📈 Critères de Classification

### Score Global (0-100%)

Calculé comme : `score = n_pass / n_completed`

### Classification Γ

| Score | Classification | Action Phase 2 |
|-------|---------------|----------------|
| ≥ 60% | **PROMETTEUR** | Exploration grille complète |
| 40-60% | **AMBIGU** | Surveillance, exploration ciblée |
| < 40% | **FAIBLE** | Paramètres nominaux seulement |

### Robustesse

```
robustesse = (Nombre de D_base avec ≥1 config PASS) / (Nombre total D_base testés)
```

**Seuils** :
- Robustesse ≥ 60% : Universel
- Robustesse 40-60% : Spécialisé
- Robustesse < 40% : Fragile

### Critères Combinés pour Phase 2

**Sélection automatique** : Score ≥ 60% **ET** Robustesse ≥ 60%

**Sélection manuelle** : Analyse qualitative (patterns intéressants, complémentarité)

---

## 🐛 Debugging

### Problème : Base de données corrompue

```bash
rm prc_database/prc_r0_results.db
python prc_automation/batch_runner1.py --init-db
```

### Problème : Opérateur non implémenté

```python
from operators import OPERATOR_REGISTRY

print(OPERATOR_REGISTRY['GAM-003']['implemented'])
# False → À implémenter
```

**Solution** : Créer `operators/gamma_hyp_003.py` en suivant template de `gamma_hyp_001.py`

### Problème : Tous les runs en ERROR

**Causes possibles** :
1. Γ retourne NaN/Inf (explosion numérique)
2. Applicabilité D incorrecte (ex: GAM-002 sur R3)
3. Paramètres invalides (ex: alpha < 0)

**Diagnostic** :

```bash
# Exécuter 1 run en mode verbeux
python prc_automation/batch_runner1.py --phase 1 --gamma GAM-XXX --verbose
```

### Problème : Exécution très lente

**Optimisations** :

1. **Réduire max_iterations** (défaut: 2000)
   ```python
   # Dans batch_runner1.py
   MAX_ITERATIONS_DEFAULT = 500  # Pour tests rapides
   ```

2. **Réduire historique snapshots**
   ```python
   # Enregistrer tous les 20 itérations au lieu de 10
   if i % 20 == 0:
       history.append(state.copy())
   ```

3. **Désactiver tests coûteux**
   ```python
   # Dans tests/utilities/__init__.py, commenter LOC-001
   ```

---

## 📝 Prochaines Étapes

### Après Phase 1 complète

1. **Générer rapport final**
   ```bash
   python prc_automation/report_generator.py --all > reports/phase1_final.txt
   ```

2. **Analyser classification**
   - Identifier Γ prometteurs (score ≥ 60%)
   - Noter patterns d'échec communs
   - Sélectionner 5-8 Γ pour Phase 2

3. **Documenter findings**
   ```
   prc_documentation/logs/LOG-PHASE1-FINAL.txt
   ```

4. **Décider Phase 2**
   - Si ≥ 5 Γ prometteurs → Phase 2 (exploration grilles)
   - Si < 3 Γ prometteurs → Implémenter nouveaux Γ ou affiner existants
   - Si ≥ 10 Γ prometteurs → Prioriser par robustesse

### Implémenter Γ manquants

**Priorité haute** (familles critiques) :

1. `gamma_hyp_007.py` - Régulation moyenne (non-markovien)
2. `gamma_hyp_009.py` - Bruit additif (stochastique)
3. `gamma_hyp_013.py` - Hebbien (structurel)

**Template** :

```python
# Copier gamma_hyp_001.py ou gamma_hyp_006.py
# Modifier classe et formule Γ
# Définir PARAM_GRID_PHASE1 et PHASE2
# Ajouter METADATA
# Enregistrer dans operators/__init__.py
```

---

## 🔍 Requêtes SQL Utiles

### Trouver meilleurs Γ

```sql
SELECT gamma_id, AVG(
    CAST(SUM(CASE WHEN global_verdict = 'PASS' THEN 1 ELSE 0 END) AS REAL) 
    / COUNT(*)
) as score
FROM Executions
WHERE status = 'COMPLETED'
GROUP BY gamma_id
ORDER BY score DESC;
```

### Runs les plus lents

```sql
SELECT gamma_id, d_base_id, execution_time_seconds
FROM Executions
WHERE status = 'COMPLETED'
ORDER BY execution_time_seconds DESC
LIMIT 10;
```

### Tests bloquants par Γ

```sql
SELECT e.gamma_id, t.test_name, COUNT(*) as blocker_count
FROM Executions e
JOIN TestResults t ON e.id = t.exec_id
WHERE t.status = 'FAIL' AND t.blocking = 1
GROUP BY e.gamma_id, t.test_name
ORDER BY blocker_count DESC;
```

---

## ✅ Checklist Phase 1

Avant de passer à Phase 2 :

- [ ] ≥ 2 Γ implémentés et testés
- [ ] Base de données créée avec ≥ 500 runs
- [ ] Rapport final généré
- [ ] Classification tous Γ documentée
- [ ] ≥ 3 Γ avec score ≥ 60% identifiés
- [ ] Patterns d'échec analysés
- [ ] Contraintes CON-GAM-XXX générées si applicable
- [ ] Sélection Γ pour Phase 2 décidée

---

## 🆘 Support

### Documentation

- **Charte PRC 5.1** : `prc-charter-code - 5.1.txt` (Section 3.1)
- **Feuille de route** : `feuille de route.txt` (Section 5, PHASE 1)
- **Schema SQL** : `prc_database/schema.sql` (commentaires)

### Problèmes courants

**Imports manquants** :
```bash
# Vérifier structure
python -c "from prc_automation import batch_runner1; print('OK')"
```

**DB locked** :
```bash
# Fermer toutes connexions
pkill -f "python.*batch_runner1"
```

---

**🎉 Framework prêt pour Phase 1 progressive !**

**Prochaine étape** : Implémenter Γ restants (GAM-003 à GAM-014) selon priorités.

---

**Contacts** :  
- Framework: PRC Core Team  
- Phase 1: Voir Feuille de Route Section 5.1  
- Issues: Documenter avec ID Γ et logs DB