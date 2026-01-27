# MODIFIER CATALOG

> Transformations D appliquées sur tenseur base  
> Composition: D^(base) + modifiers → D^(final)  
> Architecture PHASE 10: 1 fichier = 1 modifier

**Version**: 2.0 (PHASE 10)  
**Date**: 2026-01-23

---

## 📐 ARCHITECTURE

### Convention nommage
```
modifiers/
├── m0_baseline.py
├── m1_gaussian_noise.py
├── m2_uniform_noise.py
└── m{N}_descriptif.py
```

### Structure fichier (OBLIGATOIRE)
```python
PHASE = "R0"  # OBLIGATOIRE

METADATA = {
    'id': 'M{N}',              # OBLIGATOIRE
    'type': '...',             # OBLIGATOIRE
    'description': '...',      # OBLIGATOIRE
    'properties': [...],
    'usage': '...'
}

def apply(state: np.ndarray, seed: int = None, **kwargs) -> np.ndarray:
    """..."""
    pass
```

### Discovery
- Pattern: `m*.py`
- Skip: `*_deprecated_*`
- Validation: `PHASE` présent, `METADATA['id']` présent, `apply()` présent
- Extraction ID: `METADATA['id']`

---

## 📋 MODIFIERS DISPONIBLES

### M0 - Baseline
**Fichier**: `m0_baseline.py`  
**Fonction**: `apply(state, seed=None)`  
**Transformation**: D' = D (identité)  
**Propriétés**: Déterministe, identité  
**Usage**: Référence pure, aucune modification

**Exemple**:
```python
from modifiers.m0_baseline import apply
D_final = apply(D_base)  # Aucune modification
```

---

### M1 - Bruit Gaussien
**Fichier**: `m1_gaussian_noise.py`  
**Fonction**: `apply(state, seed=None, sigma=0.05)`  
**Transformation**: D' = D + N(0, σ)  
**Paramètres**:
- `sigma`: Écart-type du bruit (0.05 par défaut)
- `seed`: Graine aléatoire (reproductibilité)

**Propriétés**: 
- Bruit centré, amplitude σ
- Distribution normale
- Préserve structure moyenne

**Usage**: Perturbation gaussienne, test robustesse

**Exemple**:
```python
from modifiers.m1_gaussian_noise import apply
D_noisy = apply(D_base, seed=42, sigma=0.05)
```

---

### M2 - Bruit Uniforme
**Fichier**: `m2_uniform_noise.py`  
**Fonction**: `apply(state, seed=None, amplitude=0.1)`  
**Transformation**: D' = D + U[-amplitude, +amplitude]  
**Paramètres**:
- `amplitude`: Amplitude du bruit (0.1 par défaut)
- `seed`: Graine aléatoire (reproductibilité)

**Propriétés**:
- Bruit uniforme, bornes [-amplitude, +amplitude]
- Distribution équiprobable
- Perturbation non gaussienne

**Usage**: Perturbation uniforme, test robustesse non-gaussienne

**Exemple**:
```python
from modifiers.m2_uniform_noise import apply
D_noisy = apply(D_base, seed=42, amplitude=0.1)
```

---

## 🔄 COMPOSITION SÉQUENTIELLE

### Via prepare_state (core)
```python
from prc_framework.core.state_preparation import prepare_state

# Single modifier
D_final = prepare_state(D_base, [modifier1])

# Multiple modifiers (ordre compte)
D_final = prepare_state(D_base, [modifier1, modifier2, modifier3])
# Équivalent à:
# state = D_base.copy()
# state = modifier1(state)
# state = modifier2(state)
# state = modifier3(state)
# D_final = state
```

### Propriétés composition
- **Non commutatif**: Ordre d'application compte
- **Séquentiel**: Chaque modifier reçoit résultat du précédent
- **Aveugle**: Core applique sans validation

---

## ✅ VALIDATION

### Modifiers DOIVENT
- Retourner `np.ndarray` de **même shape** que input
- Être **déterministes** si `seed` fourni
- Ne pas lever d'exception (sauf erreur critique)

### Validation effectuée
**HORS du core** (responsabilité modifier)

---

## 🚫 DÉPENDANCES

**Autorisées**: NumPy uniquement  
**Interdites**: 
- `core/` (pas de dépendance circulaire)
- `operators/`, `D_encodings/`, `tests/`, `utilities/`

---

## 📏 RÈGLES ARCHITECTURALES (PHASE 10)

### Règle G1 - Une brique = un mécanisme
**Critère**: "Puis-je expliquer en une phrase ce que cette brique fait de différent ?"
- OUI → nouveau fichier
- NON → paramètre du même fichier

**Exemple BON**:
- `m1_gaussian_noise.py` (distribution gaussienne)
- `m2_uniform_noise.py` (distribution uniforme)
→ Mécanismes différents

**Exemple MAUVAIS** (À ÉVITER):
- `m1_gaussian_noise.py`
- `m1_gaussian_noise_low_sigma.py`
→ Variation paramétrique, PAS nouveau mécanisme

### Règle G2 - Métadonnées obligatoires
**Minimum OBLIGATOIRE**:
- `PHASE = "R0"`
- `METADATA['id']`
- `METADATA['description']`

Discovery lève `CriticalDiscoveryError` si absent.

### Règle G3 - Dépréciation explicite
**Jamais supprimer fichier sans**:
- `_deprecated_` dans nom fichier, OU
- `DEPRECATED = True` dans module

**Protection**: Reproductibilité, lecture historique, backfills

---

## 🔮 EXTENSIONS FUTURES (R1+)

### Extensions acceptables
- ✅ Bruit corrélé spatialement
- ✅ Clipping bornes [-1, 1]
- ✅ Normalisation (rescaling)
- ✅ Projection sur sous-espace
- ✅ Contraintes topologiques (périodiques, etc.)

### Extensions REFUSÉES
- ❌ Validation propriétés (symétrie, etc.) → Responsabilité encodings
- ❌ Adaptation selon gamma → Violé aveuglement core
- ❌ Optimisation état → Pas une transformation passive

### Structure anticipée R1+
```
modifiers/
├── m0_baseline.py
├── m1_gaussian_noise.py
├── m2_uniform_noise.py
├── constraints/
│   ├── periodic.py
│   └── bounded.py
├── plugins/
│   └── domain_specific.py
└── domains/
    └── ...
```

---

## 📊 RÉCAPITULATIF

| ID | Fichier | Type | Transformation |
|----|---------|------|----------------|
| **M0** | m0_baseline.py | baseline | D' = D |
| **M1** | m1_gaussian_noise.py | noise | D' = D + N(0,σ) |
| **M2** | m2_uniform_noise.py | noise | D' = D + U[-a,+a] |
| **TOTAL** | 3 modifiers R0 | | |

---

## 🔍 NOTES ARCHITECTURALES

### Séparation stricte
- **Modifiers** transforment D (D → D')
- **Encodings** créent D^(base)
- **Core** compose séquentiellement (prepare_state)
- **Tests** observent résultat

### Principe fondamental
Modifiers sont **transformations pures** sans connaissance contexte (gamma, test, encoding).

### Pattern apply() vs Factory
**PHASE 10 change**: Passage de pattern Factory à fonction directe `apply()`

**Ancien (noise.py)**:
```python
def add_gaussian_noise(sigma, seed):
    def modifier(state):
        ...
    return modifier
```

**Nouveau (m1_gaussian_noise.py)**:
```python
def apply(state, seed=None, sigma=0.05):
    ...
    return state + noise
```

**Justification**:
- Cohérence avec architecture unifiée
- Paramètres explicites dans signature
- Discovery uniforme (même pattern que encodings)

---

## ✅ CHECKLIST AJOUT NOUVEAU MODIFIER

- [ ] Créer fichier `m{N}_descriptif.py`
- [ ] Ajouter `PHASE = "R0"`
- [ ] Ajouter `METADATA` complet (id, type, description)
- [ ] Implémenter `apply(state, seed=None, **kwargs)`
- [ ] Retourner `np.ndarray` shape identique
- [ ] Documenter transformation mathématique
- [ ] Tester reproductibilité (seed fixe)
- [ ] Mettre à jour ce catalogue

---

**FIN MODIFIER CATALOG v2.0**