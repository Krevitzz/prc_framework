# PRC Framework - Phase 0 (Calibration) - Guide d'Utilisation

**Version**: 1.0.0  
**Date**: 2024-12-24  
**Statut**: OPÉRATIONNEL pour TM-GAM-001

---

## 📦 Fichiers Générés

### ✅ **COMPLETS** (Phase 0 prêt)

```
operators/
├── __init__.py                  ✅ Package avec registre
└── gamma_hyp_001.py             ✅ Saturation pure (β)

tests/utilities/
├── __init__.py                  ✅ Package avec helpers
├── test_symmetry.py             ✅ TEST-SYM-001, 002, 003
├── test_norm.py                 ✅ TEST-UNIV-001, TEST-BND-001, STR-002
├── test_diversity.py            ✅ TEST-UNIV-002 + compléments
└── test_convergence.py          ✅ TEST-UNIV-003 + compléments

tests/
└── TM-GAM-001.py                ✅ Toy Model complet (Phase 0)
```

### ⏳ **TODO** (Phase 1+)

- `operators/gamma_hyp_002.py` à `gamma_hyp_014.py` (13 opérateurs)
- `tests/TM-GAM-002.py` à `TM-GAM-014.py` (25 Toy Models)
- `prc_database/schema.sql` (pour stockage résultats)
- `prc_automation/batch_runner.py` (orchestration complète)

---

## 🚀 Installation

### Prérequis

```bash
python >= 3.8
numpy >= 1.20.0
```

### Structure du projet

```
prc_framework/
├── core/                        ✅ Existant
│   ├── __init__.py
│   ├── kernel.py
│   └── state_preparation.py
│
├── D_encodings/                 ✅ Existant
│   ├── __init__.py
│   ├── rank2_symmetric.py
│   ├── rank2_asymmetric.py
│   └── rank3_correlations.py
│
├── modifiers/
│   ├── __init__.py
│   └── noise.py                 ✅ Existant
│
├── operators/                   ✅ NOUVEAU
│   ├── __init__.py
│   └── gamma_hyp_001.py
│
└── tests/
    ├── utilities/               ✅ NOUVEAU
    │   ├── __init__.py
    │   ├── test_symmetry.py
    │   ├── test_norm.py
    │   ├── test_diversity.py
    │   └── test_convergence.py
    │
    └── TM-GAM-001.py            ✅ NOUVEAU
```

---

## 🎯 Phase 0 : Calibration

### Objectif

Valider le pipeline complet sur **1 hypothèse Γ** avec **36 runs** :

- **Γ**: PureSaturationGamma (β ∈ {1.0, 2.0})
- **D**: SYM-001, SYM-002, ASY-001
- **Modifiers**: M0 (aucun), M1 (bruit gaussien)
- **Seeds**: 1, 2, 3
- **Total**: 2 × 3 × 2 × 3 = **36 exécutions**

### Usage

#### 1️⃣ Test rapide (1 run)

```bash
cd prc_framework/tests
python TM-GAM-001.py --single
```

**Sortie attendue** :
```
RUN: GAM-001_beta2.0_DSYM-001_MM0_seed42
✓ D^(base) généré: SYM-001, shape=(50, 50)
✓ Modifier: aucun (M0)
✓ Γ créé: PureSaturationGamma(beta=2.0)
✓ Exécution kernel (max_iter=200)...
✓ Kernel terminé: 200 iterations, 21 snapshots
✓ Application des tests...

RÉSUMÉ DES TESTS:
  ✓ UNIV-001: Évolution contrôlée: stable, max=50.00
  ✓ UNIV-002: Diversité préservée: ratio=0.85
  ✗ UNIV-003: Convergence triviale vers identity (dist=1.2e-02)
  ✓ SYM-001: Symétrie préservée (max asym: 2.3e-07)
  ...

✓ VERDICT: PASS (7/11 tests réussis)
```

#### 2️⃣ Phase 0 complète (36 runs)

```bash
python TM-GAM-001.py --phase 0
```

**Durée estimée** : 5-10 minutes (selon machine)

**Sortie finale** :
```
# RAPPORT FINAL - PHASE 0
Total runs: 36
Complétés: 36 (100%)
Erreurs: 0 (0%)

Verdicts globaux:
  PASS: 18 (50%)
  POOR: 10 (28%)
  REJECTED: 8 (22%)

Matrice β × D_base:
β         SYM-001        SYM-002        ASY-001
1.0       67% (12)       50% (12)       42% (12)
2.0       83% (12)       58% (12)       33% (12)
```

---

## 🔬 Tests Disponibles

### Tests Universels (tous D, tous Γ)

| ID | Nom | Mesure | Verdict |
|----|-----|--------|---------|
| **UNIV-001** | Norme Frobenius | `‖D‖_F` évolution | PASS si stable/oscillant |
| **UNIV-002** | Diversité | `σ(D)` évolution | PASS si ratio ∈ [0.3, 3.0] |
| **UNIV-003** | Convergence | `‖D_{t+1} - D_t‖` | PASS si non-triviale |

### Tests Symétrie (rang 2)

| ID | Nom | Applicable | Verdict |
|----|-----|-----------|---------|
| **SYM-001** | Préservation | D symétrique | PASS si asymétrie < 1e-4 |
| **SYM-002** | Création | D asymétrique | PASS si symétrie créée |
| **SYM-003** | Évolution | Tous | NEUTRAL (observation) |

### Tests Structure

| ID | Nom | Mesure |
|----|-----|--------|
| **STR-002** | Spectre | `max(|λ_i|)` |
| **BND-001** | Bornes | `min/max(D)` |

### Tests Complémentaires

- `DIV-ENTROPY`: Entropie distribution
- `DIV-UNIFORM`: Coefficient variation
- `CONV-LYAPUNOV`: Exposant Lyapunov

---

## 📊 Interprétation des Résultats

### Status des Tests

- ✅ **PASS**: Test réussi selon critères
- ❌ **FAIL**: Test échoué (peut être bloquant)
- ⚪ **NEUTRAL**: Observation sans verdict

### Verdict Global par Run

- **PASS**: Majorité tests PASS, aucun blocker
- **POOR**: Plus de FAIL que de PASS
- **REJECTED**: Tests bloquants échoués (explosion, trivialité)
- **NEUTRAL**: Ambiguïté

### Exemple d'Analyse

```python
# Si TM-GAM-001 donne:
# - β=1.0 : 50% PASS
# - β=2.0 : 75% PASS

# Interprétation:
# - β=2.0 plus robuste que β=1.0
# - Saturation forte favorise stabilité
# - Mais risque trivialité (convergence vers signes)
```

---

## 🛠️ Personnalisation

### Ajouter un nouveau D^(base)

```python
# Dans TM-GAM-001.py, modifier PHASE0_CONFIG:
'd_bases': {
    'SYM-001': (create_identity, {'n_dof': 50}),
    'SYM-007': (mon_nouveau_generateur, {'n_dof': 50, ...}),  # NOUVEAU
}
```

### Modifier les paramètres Γ

```python
# Dans TM-GAM-001.py:
'beta_values': [0.5, 1.0, 2.0, 5.0],  # Ajouter valeurs
```

### Changer max_iterations

```python
'max_iterations': 500,  # Plus d'itérations si convergence lente
```

---

## 🔍 Debugging

### Problème : NaN ou Inf détecté

```
❌ FAIL: Explosion numérique détectée
```

**Causes possibles** :
- `β` trop grand (> 10)
- D initial mal conditionné
- Γ instable

**Solutions** :
- Réduire `β`
- Vérifier D avec `D_encodings.verification_tests`
- Ajouter saturation dans Γ

### Problème : Convergence triviale

```
✗ UNIV-003: Convergence triviale vers identity
```

**Interprétation** :
- Γ trop faible (collapse vers attracteur trivial)
- D initial trop proche identité

**Solutions** :
- Augmenter `β` ou force de Γ
- Tester sur D plus diversifié (SYM-002 au lieu SYM-001)

### Problème : Tous tests FAIL

```
VERDICT: REJECTED (8 tests bloquants échoués)
```

**Actions** :
1. Vérifier implémentation Γ avec `operators.validate_operator("GAM-001")`
2. Tester D avec `D_encodings.run_verification_suite(...)`
3. Réduire `max_iterations` pour identifier échec précoce

---

## 📈 Prochaines Étapes

### Après Phase 0 réussie

1. **Analyser résultats** :
   - Matrice β × D : identifier patterns
   - Tests échouant systématiquement
   - Configurations robustes

2. **Documenter findings** :
   - Créer `prc_documentation/logs/LOG-GAM-001-phase0.txt`
   - Noter observations (trivialité, robustesse, etc.)

3. **Décider Phase 1** :
   - Si GAM-001 prometteur → Grille complète Phase 2
   - Si GAM-001 rejeté → Passer à GAM-002

### Implémenter Prochains Γ

**Priorité pour Phase 1** :

1. `operators/gamma_hyp_002.py` - Diffusion pure
2. `operators/gamma_hyp_006.py` - Mémoire ordre-1
3. `operators/gamma_hyp_009.py` - Bruit additif

**Template à suivre** : `gamma_hyp_001.py`

---

## 🆘 Support

### Problèmes courants

**Import errors** :
```bash
# Vérifier structure:
python -c "from operators import PureSaturationGamma; print('OK')"
python -c "from tests.utilities import test_symmetry_preservation; print('OK')"
```

**Tests incompatibles** :
- Vérifier `d_base_id` commence par "SYM", "ASY", ou "R3"
- Tester avec `get_applicable_tests(d_base_type)`

### Documentation

- **Charte PRC 5.1** : `prc-charter-code - 5.1.txt`
- **Feuille de route R0** : `feuille de route.txt`
- **Encodings** : `D_encodings/README_catalogue.txt`

---

## ✅ Checklist Phase 0

Avant de passer à Phase 1, vérifier :

- [ ] TM-GAM-001 exécute sans erreurs
- [ ] 36 runs complétés avec succès
- [ ] Rapport final généré avec matrice β × D
- [ ] Au moins 1 configuration PASS
- [ ] Tests de validation passent : `operators.validate_all_operators()`
- [ ] Documentation créée : LOG + notes observations
- [ ] Code versionné (git commit)

---

**🎉 Phase 0 complète ! Framework opérationnel pour exploration R0.**

**Prochaine étape** : Implémenter GAM-002 à GAM-014 pour Phase 1.

---

**Contacts** :  
- Framework: PRC Core Team  
- Support: Voir documentation Charte PRC 5.1  
- Issues: Documenter avec ID du test/opérateur échouant