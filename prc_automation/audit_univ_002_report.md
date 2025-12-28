# AUDIT UNIV-002 - Rapport Exhaustif

**Date**: 2025-12-28 00:10:18
**Conformité**: Charte PRC 5.1 Section 14
**Méthode**: Audit post-hoc (aucun rerun)

**Données**: 2861 exécutions analysées

---

## RÉSUMÉ EXÉCUTIF

### Objectif

Déterminer si l'échec systématique de UNIV-002 est:

1. Un artefact de scoring/normalisation
2. Un biais de métrique (variance globale vs diversité structurelle)
3. Un effet de taille/base D
4. Un pattern dynamique réel

### Signaux forts détectés

1. **COLLAPSE RAPIDE dominant: 1776/2861 exécutions**
2. **SEUIL INFÉRIEUR SATURÉ: 57% des ratios < 0.3**



---

## 2. ANALYSE TEMPORELLE

### Observations

Exécutions avec séries temporelles: 2861/2861

Demi-vie moyenne: 27.3 itérations
Demi-vie médiane: 8.0 itérations
Demi-vie min/max: [1, 766]

Types de collapse:
  fast: 1776 (62.1%)
  slow: 41 (1.4%)
  plateau: 1044 (36.5%)
  oscillating: 790 (27.6%)


### ⚠️ Signaux forts

- **COLLAPSE RAPIDE dominant: 1776/2861 exécutions**


### ℹ️ Signaux faibles

- Oscillations détectées: 790 cas


### Figures

![audit2_temporal](figures_audit_univ_002\audit2_temporal.png)


---

## 3. VALIDATION SCORING

### Observations

Fonction de scoring actuelle:
  Corrélation avec score linéaire: 0.775
  Corrélation avec score sigmoïde: 0.627

Distribution vs seuils:
  Ratio < 0.3 (score ≈ 0): 1627 (56.9%)
  Ratio > 3.0 (score ≈ 1): 622 (21.7%)
  Ratio dans [0.3, 3.0]: 612 (21.4%)


### ⚠️ Signaux forts

- **SEUIL INFÉRIEUR SATURÉ: 57% des ratios < 0.3**


### Figures

![audit3_scoring](figures_audit_univ_002\audit3_scoring.png)


---

## 4. SENSIBILITÉ PARAMÈTRES

### Observations

Robustesse inter-graines (variance intra-paramètres):


GAM-001:
  Params {'beta': 2.0}: std_score=0.205, std_ratio=1.799 (n=260 seeds)

GAM-002:
  Params {'alpha': 0.05}: std_score=0.060, std_ratio=0.046 (n=200 seeds)

GAM-003:
  Params {'gamma': 0.05}: std_score=0.000, std_ratio=18695007107121466449185472512.000 (n=260 seeds)

GAM-004:
  Params {'gamma': 0.05}: std_score=0.000, std_ratio=0.000 (n=260 seeds)

GAM-005:
  Params {'omega': 0.7853981633974483}: std_score=0.000, std_ratio=0.000 (n=260 seeds)

GAM-006:
  Params {'alpha': 0.3, 'beta': 1.0}: std_score=0.081, std_ratio=0.035 (n=260 seeds)

GAM-007:
  Params {'epsilon': 0.1}: std_score=0.060, std_ratio=0.067 (n=200 seeds)

GAM-008:
  Params {'beta': 1.0, 'gamma': 0.3}: std_score=0.205, std_ratio=1.799 (n=260 seeds)

GAM-009:
  Params {'beta': 1.0, 'sigma': 0.01}: std_score=0.268, std_ratio=0.202 (n=260 seeds)

GAM-010:
  Params {'sigma': 0.05}: std_score=0.044, std_ratio=0.028 (n=260 seeds)

GAM-012:
  Params {'beta': 2.0}: std_score=0.250, std_ratio=1.802 (n=200 seeds)

GAM-013:
  Params {'eta': 0.01}: std_score=0.092, std_ratio=nan (n=181 seeds)


Effet des paramètres (si grille disponible):


GAM-001: Un seul jeu de paramètres (pas d'analyse possible)

GAM-002: Un seul jeu de paramètres (pas d'analyse possible)

GAM-003: Un seul jeu de paramètres (pas d'analyse possible)

GAM-004: Un seul jeu de paramètres (pas d'analyse possible)

GAM-005: Un seul jeu de paramètres (pas d'analyse possible)

GAM-006: Un seul jeu de paramètres (pas d'analyse possible)

GAM-007: Un seul jeu de paramètres (pas d'analyse possible)

GAM-008: Un seul jeu de paramètres (pas d'analyse possible)

GAM-009: Un seul jeu de paramètres (pas d'analyse possible)

GAM-010: Un seul jeu de paramètres (pas d'analyse possible)

GAM-012: Un seul jeu de paramètres (pas d'analyse possible)

GAM-013: Un seul jeu de paramètres (pas d'analyse possible)


### ℹ️ Signaux faibles

- GAM-001 avec params {'beta': 2.0} : variance élevée inter-seeds (std=0.205)
- GAM-008 avec params {'beta': 1.0, 'gamma': 0.3} : variance élevée inter-seeds (std=0.205)
- GAM-009 avec params {'beta': 1.0, 'sigma': 0.01} : variance élevée inter-seeds (std=0.268)
- GAM-012 avec params {'beta': 2.0} : variance élevée inter-seeds (std=0.250)


### Figures

![audit4_sensitivity](figures_audit_univ_002\audit4_sensitivity.png)


---

## 5. CORRÉLATIONS INTER-TESTS

### Observations

Corrélations UNIV-002 avec autres tests:


CONV-LYAPUNOV:
  Pearson: r=-0.192, p=0.0000
  Spearman: ρ=-0.807, p=0.0000
  n=2858 paires

BND-001:
  Pearson: r=-0.097, p=0.0000
  Spearman: ρ=-0.275, p=0.0000
  n=2861 paires

SYM-001:
  Pearson: r=-0.063, p=0.0021
  Spearman: ρ=-0.080, p=0.0001
  n=2381 paires

SYM-002:
  Pearson: r=-0.063, p=0.0021
  Spearman: ρ=-0.080, p=0.0001
  n=2381 paires

UNIV-001:
  Pearson: r=0.005, p=0.8162
  Spearman: ρ=-0.275, p=0.0000
  n=2381 paires

UNIV-003:
  Pearson: r=0.177, p=0.0000
  Spearman: ρ=0.096, p=0.0000
  n=2861 paires


### Figures

![audit5_correlations](figures_audit_univ_002\audit5_correlations.png)


---

## 6. LOCAL VS GLOBAL

### Observations

Audit local vs global nécessite snapshots complets.
Échantillonnage aléatoire de 20 exécutions pour analyse.

Nombre d'échantillons analysés: 16
Ratio moyen local/global: 0.705
Ratio médian local/global: 0.980
Écart-type ratios: 0.363


### Figures

![audit6_local_global](figures_audit_univ_002\audit6_local_global.png)


---

## 7. INVARIANCE D'ÉCHELLE

### Observations

Distribution par type de base:


ASY (n=953):
  Score moyen: 0.232 ± 0.285
  Ratio moyen: inf ± nan

R3 (n=480):
  Score moyen: 0.250 ± 0.283
  Ratio moyen: 3360146427270661592947290993154635346214912.000 ± 8890111815320353960891126852000806267256832.000

SYM (n=1428):
  Score moyen: 0.232 ± 0.274
  Ratio moyen: inf ± nan

ANOVA entre types de base:
  F=0.854, p=0.4257


Distribution par taille estimée:


Taille ≈20 DOF (n=480):
  Score moyen: 0.250 ± 0.283

Taille ≈50 DOF (n=2381):
  Score moyen: 0.232 ± 0.278


### Figures

![audit7_scale](figures_audit_univ_002\audit7_scale.png)


---

## TABLEAU DE DÉCISION

| Critère | Statut | Justification |
|---------|--------|---------------|
| UNIV-002 valide tel quel | ❌ | Signaux contradictoires détectés |
| UNIV-002 valide mais mal scoré | ⚠️ | Seuils de scoring inadaptés |
| UNIV-002 mesure autre chose | ⚠️ | Mesure collapse, pas diversité structurelle |
| UNIV-002 non concluant à R0 | ➖ | Données suffisamment informatives |


## RECOMMANDATIONS MÉTHODOLOGIQUES

### Actions immédiates

1. **Réviser fonction de scoring UNIV-002**
   - Ajuster seuils [0.3, 3.0] selon distributions observées
   - Tester scoring linéaire vs sigmoïde
   - Relancer `--verdict` avec nouveau scoring

2. **Clarifier définition UNIV-002**
   - Documenter explicitement: diversité globale vs structurelle
   - Envisager test complémentaire pour diversité locale
   - Dissocier 'variance' de 'diversité structurelle'

3. **Ne PAS ériger en contrainte L2**
   - Pattern observé dépend de l'instrumentation actuelle
   - Journaliser comme OBS-GAM-001 (L4), pas CON-GAM-001 (L2)
   - Validation nécessite tests R1 pour départager hypothèses

### Exploration R1

- Tester compositions Γ pour vérifier si UNIV-002 reste pertinent
- Comparer UNIV-002 avec métriques alternatives (entropie, structure locale)
- Analyser si compositions R1 dépassent limites observées R0

## INTERDICTIONS MÉTHODOLOGIQUES

❌ **NE PAS conclure** que "les mécanismes isolés échouent par nature"
   - Pattern observé dépend de l'opérateur UNIV-002 actuel
   - Généralisation hors R0 non validée

❌ **NE PAS proposer** de clôture R0
   - Instrumentation instable (tests plats, corrélations suspectes)
   - R0 'cohérent mais partiel', pas 'exhaustif'

❌ **NE PAS ériger** observation en contrainte
   - Passage L3→L2 nécessite validation instrumentale
   - Discussion JOUR obligatoire avant CON-GAM-XXX


---

## MÉTADONNÉES

- **Script**: audit_univ_002.py
- **Base db_raw**: prc_database\prc_r0_raw.db
- **Base db_results**: prc_database\prc_r0_results.db
- **Figures**: figures_audit_univ_002/
- **Conformité Charte**: Section 14 (rejouabilité sans reruns)