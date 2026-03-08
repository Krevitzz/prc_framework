# PHASES GUIDE — Structure évolutive PRC

> Documentation distinction phases majeures (x) vs sous-phases locales (x.y.z)
> Date : 2026-02-20

---

## PRINCIPES FONDAMENTAUX

### Phases majeures (x) : R0, R1, R2, ..., R_N
**Rôle** : Évolution pool atomics + architecture

**Caractéristiques** :
- Parquet écrit : `data/results/r{x}.parquet`
- Concordance inter-phases possible
- Pool requirements peut évoluer
- Axes itération stables (gamma, encoding, modifier, n_dof, max_iterations)

**Triggers phase x+1** :
- Atomics deprecated validés (≥3 gammas retirés)
- Contraintes pool identifiées (ex: n_dof > 100)
- Architecture modifiée (nouveaux layers, features)

**Exemples** :
```
R0 : Baseline exploration
  → Pool: 12 gammas, 21 encodings, 3 modifiers
  → n_dof: 50, max_iterations: 1000

R1 : Après calibration R0
  → Pool: 9 gammas (3 deprecated), 18 encodings
  → n_dof: 100 (requis), max_iterations: 200 (max)
  → Features: +5 nouvelles (matrix_2d layer)

R2 : Validation stabilité
  → Pool: 9 gammas, 18 encodings, 5 modifiers
  → Concordance R0↔R1↔R2 sur subset commun
```

---

### Sous-phases locales (x.y.z) : R0.1.0, R0.2.1, ...
**Rôle** : Calibration atomics individuels (affinage local)

**Caractéristiques** :
- **PAS de Parquet** (runs jetables)
- Modifications YAML atomics locales
- Modifications config featuring locales
- Objectif : valider/invalider atomics spécifiques

**Nomenclature** :
```
x.y.z
│ │ └─ Micro-ajustement (params atomics, bins entropy, ...)
│ └─── Itération calibration (tentative 1, 2, 3, ...)
└───── Phase majeure de référence
```

**Workflow type** :
```
Problème détecté : GAM-003 explose systématiquement (R0 → explosion 100%)

R0.1.0 : Test GAM-003 seul avec gamma=[0.01, 0.03, 0.05] (au lieu de 0.05 défaut)
  → Résultat : gamma=0.01 stable, mais comportement trivial (attracteur plat)
  
R0.1.1 : Test GAM-003 avec max_iterations=[100, 500, 1000]
  → Résultat : explosion même avec 100 iterations

R0.2.0 : Modification config featuring (bins entropy: 100 au lieu de 50)
  → Résultat : détection précoce explosion (iteration 20 au lieu de 50)

Décision : GAM-003 marqué deprecated
  → configs/atomics/operators/operators_default.yaml : gamma=0.05 restauré
  → pool_requirements.yaml : deprecated.gammas += ['GAM-003']
  → R1 lancé sans GAM-003
```

**Exemples cas d'usage** :
```
x.y.z → Calibration locale (PAS de Parquet)
  - Gamma instable → tests params gamma
  - Encoding pathologique → tests params encoding
  - Features bruités → tests bins/projections
  - Thresholds régimes → tests seuils

x → Phase majeure (Parquet + concordance)
  - Pool validé post-calibrations x.y.z
  - Analyses cross-runs complètes
  - Concordance inter-phases
```

---

## RÈGLES DÉCISION

### Quand créer sous-phase x.y.z ?
✅ **OUI** si :
- Calibration atomics individuel (1-3 gammas/encodings)
- Modification params locale temporaire
- Tests thresholds/configs featuring
- Runs jetables (<1000 observations)

❌ **NON** si :
- Changement pool (deprecated définitifs)
- Modification architecture (nouveaux layers)
- Runs production (≥1000 observations)
- Concordance inter-phases nécessaire

### Quand créer phase x+1 ?
✅ **OUI** si :
- Pool requirements modifié (deprecated, contraintes n_dof/max_it)
- Calibrations x.y.z terminées → pool stable
- Architecture évoluée (features, layers)
- Concordance avec x-1 nécessaire

❌ **NON** si :
- Juste 1-2 atomics deprecated (attendre batch)
- Calibration locale en cours
- Pool instable

---

## WORKFLOW COMPLET

### Cycle évolutif type
```
R0 (baseline exploration)
  ↓
R0.1.0, R0.1.1, ... (calibration GAM-003)
R0.2.0, R0.2.1, ... (calibration SYM-008)
  ↓
Décisions :
  - GAM-003 → deprecated
  - SYM-008 → params modifiés (sigma: 0.5 → 0.3)
  - n_dof > 100 validé
  ↓
R1 (pool stable + contraintes)
  ↓
R1.1.0, ... (affinage nouveaux comportements)
  ↓
R2 (validation stabilité)
  ↓
Concordance R0↔R1↔R2 (subset commun)
```

---

## CONCORDANCE INTER-PHASES

### Subset commun
**Problème** : R0 avec n_dof=50, R1 avec n_dof=100

**Solution** :
```python
# verdict.py — filtrage automatique

requirements = load_pool_requirements()  # n_dof.min = 100

r0_parquet = load('r0.parquet')  # 252 observations
r0_filtered = filter_rows_by_pool(r0_parquet, requirements)
  → 0 observations (toutes n_dof=50 < 100)

r1_parquet = load('r1.parquet')  # 180 observations
r1_filtered = filter_rows_by_pool(r1_parquet, requirements)
  → 180 observations (toutes valides)

concordance(r0_filtered, r1_filtered)
  → Subset commun : 0 lignes
  → Warning : "Concordance impossible (subset vide)"
```

**Validation minimum** :
```python
if len(subset_commun) < 10:
    warnings.warn(
        f"Subset commun trop petit ({len(subset_commun)}) — "
        f"concordance peu fiable"
    )
```

---

## COMMANDES CLI

### Batch runs
```bash
# Phase majeure (Parquet + verdict)
python -m batch r0

# Sous-phase locale (runs jetables, pas Parquet)
python -m batch r0.1.0 --no-parquet

# Verdict seul (sans reruns)
python -m batch --verdict r0
python -m batch --verdict r0,r1,r2  # Concordance inter-phases
```

### Structure fichiers
```
configs/phases/
├── r0/
│   └── r0.yaml              # Phase majeure
├── r0_local/
│   ├── r0.1.0.yaml          # Sous-phase calibration
│   ├── r0.1.1.yaml
│   └── r0.2.0.yaml
├── r1/
│   └── r1.yaml
└── ...

data/results/
├── r0.parquet               # Phase majeure uniquement
├── r1.parquet
└── r2.parquet
```

---

## BEST PRACTICES

### Nommage sous-phases
✅ **BON** :
- `r0.1.0` → Calibration GAM-003 tentative 1
- `r0.1.1` → Calibration GAM-003 tentative 2
- `r0.2.0` → Calibration SYM-008

❌ **MAUVAIS** :
- `r0_test` → Pas de structure
- `r0_bis` → Ambiguïté
- `r0_final` → Confusion avec phase majeure

### Documentation sous-phases
Chaque YAML sous-phase doit avoir :
```yaml
# configs/phases/r0_local/r0.1.0.yaml

description: |
  Calibration GAM-003 — tests gamma=[0.01, 0.03, 0.05]
  Objectif : Identifier si gamma=0.01 stabilise sans trivialiser
  
target_atomics:
  - GAM-003

modifications:
  - operators_default.yaml : gamma: [0.01, 0.03, 0.05]
  
expected_outcome: |
  Si gamma=0.01 stable ET non-trivial → modifier default
  Sinon → marquer deprecated
```

### Traçabilité décisions
```yaml
# pool_requirements.yaml

deprecated:
  gammas:
    - GAM-003:
        reason: "Explosion systématique (R0.1.0, R0.1.1)"
        date: "2026-02-15"
        calibrations_tested: ["r0.1.0", "r0.1.1"]
```

---

**FIN PHASES GUIDE**
