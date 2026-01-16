# modifier_catalog.md

> Catalogue fonctionnel des modificateurs D  
> Responsabilité : Transformation D^(base) → D^(final)  
> Version : 6.0  
> Dernière mise à jour : 2025-01-15

---

## VUE D'ENSEMBLE

Le module `modifiers/` transforme les tenseurs de base D^(base) en états finaux D^(final) via composition séquentielle.

**Modules existants** :
- `noise.py` : Ajout bruit (gaussien, uniforme)
- `constraints/` : Contraintes topologiques (à documenter)
- `plugins/` : Extensions custom (à documenter)
- `domains/` : Domaines spécialisés (à documenter)

**IDs catalogués** :
- **M0** : Baseline (aucune modification)
- **M1** : Bruit gaussien (sigma=0.05)
- **M2** : Bruit uniforme (amplitude=0.1)

**Principe fondamental** :
- Modifiers = **factories** retournant `Callable[[np.ndarray], np.ndarray]`
- Application séquentielle dans `prepare_state()`
- ✅ Transformations pures (pas d'effet de bord)
- ❌ Aucune validation sémantique (passthrough)

---

## SECTION 1 : noise.py

### 1.1 add_gaussian_noise()

**Signature** :
```python
def add_gaussian_noise(sigma: float = 0.01, seed: int = None) -> Callable[[np.ndarray], np.ndarray]
```

**Type** : Factory (retourne fonction)

**Forme mathématique** :
```
D' = D + N(0, σ)
où N ~ N(0, σ) gaussien, shape identique à D
```

**Paramètres** :
| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `sigma` | `float` | 0.01 | Écart-type du bruit gaussien |
| `seed` | `int` | None | Graine aléatoire (reproductibilité) |

**Retour** : `Callable[[np.ndarray], np.ndarray]`
- Fonction prenant tenseur → retourne tenseur bruité

**Comportement** :
1. Crée générateur aléatoire (np.random.RandomState si seed, sinon np.random)
2. Retourne closure qui :
   - Génère bruit N(0, σ) même shape que state
   - Ajoute bruit à state
   - Retourne state + bruit

**Cas d'usage** :
```python
# Création modifier
modifier = add_gaussian_noise(sigma=0.05, seed=42)

# Application directe
D_noisy = modifier(D_base)

# Via prepare_state (usage standard)
D_final = prepare_state(D_base, [
    add_gaussian_noise(sigma=0.05, seed=42)
])
```

**ID catalogué** :
- **M1** : `add_gaussian_noise(sigma=0.05)` (sans seed spécifié)

**Consommateurs** :
- `core/state_preparation.py` (via prepare_state)
- `prc_automation/batch_runner.py` (construction pipeline)

**Notes techniques** :
- Bruit gaussien : distribution normale centrée
- σ typique : 0.01 à 0.1 (1% à 10% de l'échelle)
- Reproductibilité : seed fixe nécessaire

---

### 1.2 add_uniform_noise()

**Signature** :
```python
def add_uniform_noise(amplitude: float = 0.01, seed: int = None) -> Callable[[np.ndarray], np.ndarray]
```

**Type** : Factory (retourne fonction)

**Forme mathématique** :
```
D' = D + U[-a, +a]
où U ~ U[-amplitude, +amplitude] uniforme
```

**Paramètres** :
| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `amplitude` | `float` | 0.01 | Amplitude bruit (bornes ±amplitude) |
| `seed` | `int` | None | Graine aléatoire |

**Retour** : `Callable[[np.ndarray], np.ndarray]`

**Comportement** :
1. Crée générateur aléatoire
2. Retourne closure qui :
   - Génère bruit U[-amplitude, +amplitude] même shape
   - Ajoute bruit à state
   - Retourne state + bruit

**Cas d'usage** :
```python
# Création modifier
modifier = add_uniform_noise(amplitude=0.1, seed=42)

# Application
D_noisy = modifier(D_base)

# Via prepare_state
D_final = prepare_state(D_base, [
    add_uniform_noise(amplitude=0.1, seed=42)
])
```

**ID catalogué** :
- **M2** : `add_uniform_noise(amplitude=0.1)` (sans seed spécifié)

**Différence vs gaussien** :
- Gaussien : queues lourdes (valeurs extrêmes rares mais possibles)
- Uniforme : bornes strictes (jamais au-delà ±amplitude)

---

## SECTION 2 : BASELINE (M0)

### 2.1 Définition M0

**ID** : M0  
**Nom** : Baseline (aucune modification)

**Implémentation** :
```python
# M0 : Pas de modifier dans liste
D_final = prepare_state(D_base, modifiers=None)
# ou
D_final = prepare_state(D_base, modifiers=[])
```

**Usage** :
- Référence pour comparaison
- Test D^(base) pur (sans perturbation)
- Isoler effet γ seul

**Note** : M0 n'est PAS une fonction, c'est l'absence de modification

---

## SECTION 3 : AUTRES MODULES (À DOCUMENTER)

### 3.1 constraints/ (structure attendue)

**Responsabilité supposée** : Contraintes topologiques D^(topo)

**Fichiers potentiels** :
- `periodic.py` : Conditions périodiques
- `boundary.py` : Conditions aux bords
- `symmetry.py` : Forçage symétries

**Format attendu** :
```python
def apply_periodic_constraint(...) -> Callable:
    """Factory appliquant contrainte périodique."""
    def modifier(state):
        # Applique contrainte
        return constrained_state
    return modifier
```

**Statut** : À compléter après inspection fichiers

---

### 3.2 plugins/ (structure attendue)

**Responsabilité supposée** : Extensions custom D^(plugin)

**Usage potentiel** :
- Transformations ad-hoc
- Expérimentations
- Modifiers non génériques

**Statut** : À compléter après inspection fichiers

---

### 3.3 domains/ (structure attendue)

**Responsabilité supposée** : Transformations domaine-spécifiques D^(domaine)

**Usage potentiel** :
- Projections sous-espaces
- Changements base
- Restrictions dimensionnelles

**Statut** : À compléter après inspection fichiers

---

## SECTION 4 : MAPPING IDs ↔ MODIFIERS

### 4.1 Table complète (catalogués)

| ID | Module | Factory | Paramètres défaut | Usage |
|----|--------|---------|-------------------|-------|
| M0 | - | (aucun) | - | Baseline (référence) |
| M1 | noise | add_gaussian_noise | sigma=0.05 | Bruit gaussien 5% |
| M2 | noise | add_uniform_noise | amplitude=0.1 | Bruit uniforme ±10% |

### 4.2 Implémentation mapping (suggestion)

```python
# modifiers/__init__.py
from .noise import add_gaussian_noise, add_uniform_noise

MODIFIER_REGISTRY = {
    'M0': None,  # Baseline (pas de modifier)
    'M1': lambda seed=None: add_gaussian_noise(sigma=0.05, seed=seed),
    'M2': lambda seed=None: add_uniform_noise(amplitude=0.1, seed=seed),
}

def get_modifier(modifier_id: str, seed: int = None):
    """
    Retourne modifier depuis ID.
    
    Args:
        modifier_id: ID modifier (ex: "M1")
        seed: Graine aléatoire (optionnel)
    
    Returns:
        Callable ou None (si M0)
    """
    if modifier_id not in MODIFIER_REGISTRY:
        raise ValueError(f"Unknown modifier_id: {modifier_id}")
    
    factory = MODIFIER_REGISTRY[modifier_id]
    if factory is None:
        return None  # M0
    return factory(seed=seed)

def get_modifiers_list(modifier_id: str, seed: int = None):
    """
    Retourne liste modifiers pour prepare_state().
    
    Args:
        modifier_id: ID modifier
        seed: Graine aléatoire
    
    Returns:
        List[Callable] ou [] (si M0)
    """
    modifier = get_modifier(modifier_id, seed=seed)
    return [] if modifier is None else [modifier]
```

**Usage** :
```python
# Pipeline typique
from modifiers import get_modifiers_list

D_base = create_random_uniform(50, seed=42)
modifiers = get_modifiers_list('M1', seed=123)
D_final = prepare_state(D_base, modifiers)
```

---

## SECTION 5 : GRAPHE DE DÉPENDANCES

### 5.1 Relations inter-modules

```
noise.py
    ├─ Appelé par : batch_runner.py (construction pipeline)
    ├─ Reçoit : paramètres (sigma/amplitude, seed)
    └─ Retourne : Callable[[np.ndarray], np.ndarray]
    
constraints/ (à documenter)
    ├─ Appelé par : batch_runner.py
    └─ Retourne : Callable

plugins/ (à documenter)
    ├─ Appelé par : batch_runner.py
    └─ Retourne : Callable

domains/ (à documenter)
    ├─ Appelé par : batch_runner.py
    └─ Retourne : Callable
```

### 5.2 Flux typique application modifiers

```
1. batch_runner.py lit modifier_id (ex: "M1")
   ↓
2. Mapping vers factory :
   "M1" → add_gaussian_noise(sigma=0.05, seed=...)
   ↓
3. Appel factory → retourne Callable
   ↓
4. Liste modifiers : [callable_1, callable_2, ...]
   ↓
5. prepare_state(D_base, modifiers)
   ↓
6. Application séquentielle :
   D → modifier_1(D) → modifier_2(...) → D_final
```

---

## SECTION 6 : INVARIANTS CRITIQUES

### 6.1 Pattern factory obligatoire

**R6.1-A** : Tous modifiers DOIVENT être factories

```python
# ✅ CORRECT (factory)
def add_noise(sigma):
    def modifier(state):
        return state + noise
    return modifier

# ❌ INTERDIT (fonction directe)
def add_noise(state, sigma):  # ❌
    return state + noise
```

**Rationale** :
- Paramètres figés à création (sigma, seed, etc.)
- Signature uniforme : `Callable[[np.ndarray], np.ndarray]`
- Compatible `prepare_state(base, [mod1, mod2, ...])`

---

### 6.2 Passthrough strict (aucune validation)

**R6.2-A** : Modifiers ne valident RIEN

```python
# ✅ CORRECT
def add_noise(sigma):
    def modifier(state):
        return state + noise  # Applique transformation, c'est tout
    return modifier

# ❌ INTERDIT
def add_noise(sigma):
    def modifier(state):
        if not is_symmetric(state):  # ❌ Validation sémantique
            raise ValueError(...)
        return state + noise
    return modifier
```

**Principe** :
- Validation sémantique → `tests/`
- Validation dimensionnelle → `D_encodings/`
- Modifiers → transformations aveugles uniquement

---

### 6.3 Transformations pures (sans effet de bord)

**R6.3-A** : Modifiers ne mutent JAMAIS l'entrée

```python
# ✅ CORRECT
def modifier(state):
    noise = np.random.randn(*state.shape)
    return state + noise  # Nouvelle copie

# ❌ INTERDIT
def modifier(state):
    state += noise  # ❌ Mute l'entrée
    return state
```

**Note** : `prepare_state()` copie déjà base, mais modifiers doivent rester purs

---

### 6.4 Gestion seed (reproductibilité)

**R6.4-A** : Seed DOIT être optionnel

```python
# ✅ CORRECT
def add_noise(sigma, seed=None):
    rng = np.random.RandomState(seed) if seed else np.random
    ...

# ❌ INTERDIT
def add_noise(sigma, seed):  # ❌ seed obligatoire
    ...
```

**R6.4-B** : Si seed=None → comportement non déterministe acceptable

**Usage** :
- Tests/debugging : seed fixe
- Production : seed=None (ou seed dérivé de exec_id)

---

## SECTION 7 : EXTENSIONS FUTURES

### 7.1 Ajout nouveau modifier (checklist)

Avant d'ajouter un modifier :

- [ ] ID unique (M3, M4, ..., ou nom descriptif si non catalogué)
- [ ] Pattern factory respecté
- [ ] Signature : `def factory(..., seed=None) -> Callable`
- [ ] Retour : `Callable[[np.ndarray], np.ndarray]`
- [ ] Paramètres avec défauts raisonnables
- [ ] Seed optionnel (reproductibilité)
- [ ] Transformation pure (pas de mutation)
- [ ] Aucune validation sémantique
- [ ] Docstring complète (forme mathématique, usage)
- [ ] Ajouté à MODIFIER_REGISTRY (si catalogué)
- [ ] Documenté dans ce catalogue

### 7.2 Extensions INTERDITES

❌ **Validation sémantique dans modifiers** :
```python
# INTERDIT
def modifier(state):
    if not is_positive_definite(state):  # ❌
        raise ValueError(...)
```

❌ **Mutation état entrée** :
```python
# INTERDIT
def modifier(state):
    state += noise  # ❌ Mute entrée
    return state
```

❌ **Retour dict/tuple** :
```python
# INTERDIT
def modifier(state):
    return {'data': state_modified, 'metadata': {...}}  # ❌
```

❌ **Branchement basé sur type D** :
```python
# INTERDIT
def modifier(state):
    if encoding_id == "SYM-001":  # ❌ Modifier ne connaît pas encoding_id
        # Traitement spécial
```

### 7.3 Extensions AUTORISÉES

✅ **Ajout paramètres configurables** :
```python
# OK
def add_noise(sigma, clip_bounds=None, seed=None):
    def modifier(state):
        noisy = state + noise
        if clip_bounds:
            noisy = np.clip(noisy, *clip_bounds)
        return noisy
    return modifier
```

✅ **Composition modifiers** :
```python
# OK
def add_noise_and_smooth(sigma, smooth_window, seed=None):
    noise_mod = add_gaussian_noise(sigma, seed)
    smooth_mod = apply_smoothing(smooth_window)
    
    def modifier(state):
        return smooth_mod(noise_mod(state))
    return modifier
```

✅ **Modifiers paramétriques complexes** :
```python
# OK
def apply_domain_transform(transform_matrix):
    def modifier(state):
        # Projection, rotation, etc.
        return transform_matrix @ state @ transform_matrix.T
    return modifier
```

---

## SECTION 8 : NOTES TECHNIQUES

### 8.1 Ordre application (non-commutatif)

**Important** : L'ordre des modifiers est significatif

```python
# Ordre 1
D1 = prepare_state(base, [add_noise(0.1), apply_constraint()])

# Ordre 2
D2 = prepare_state(base, [apply_constraint(), add_noise(0.1)])

# Généralement : D1 ≠ D2
```

**Convention** :
- Bruit appliqué APRÈS contraintes (si ordre non spécifié)
- Sauf si contrainte doit "corriger" bruit

### 8.2 Amplitude bruit vs échelle D

**Guidelines** :
| Échelle D | σ gaussien | amplitude uniforme |
|-----------|------------|-------------------|
| [-1, 1] | 0.01 - 0.05 | 0.05 - 0.1 |
| [0, 1] | 0.005 - 0.02 | 0.02 - 0.05 |
| Libre | 1% - 5% de std(D) | 2% - 10% de range(D) |

**M1** : σ=0.05 → ~5% perturbation (échelle [-1,1])  
**M2** : amplitude=0.1 → ±10% perturbation (échelle [-1,1])

### 8.3 Gaussien vs Uniforme

**Bruit gaussien** :
- Queues lourdes (outliers possibles)
- Concentré autour moyenne
- Perturbation "naturelle"

**Bruit uniforme** :
- Bornes strictes
- Distribution plate
- Perturbation "contrôlée"

**Choix** :
- Gaussien : test robustesse outliers
- Uniforme : test perturbation bornée

### 8.4 Reproductibilité multi-niveaux

**Seeds hiérarchiques** :
```python
# Seed global execution
exec_seed = 42

# Seed D_base
D_base = create_random_uniform(50, seed=exec_seed)

# Seed modifier (dérivé)
modifier_seed = exec_seed + 1000
modifiers = [add_gaussian_noise(sigma=0.05, seed=modifier_seed)]

D_final = prepare_state(D_base, modifiers)
```

**Convention batch_runner** :
- `seed` (DB) : seed execution globale
- Seed modifier : dérivé ou None (selon config)

---

## SECTION 9 : TESTS ASSOCIÉS

### 9.1 Tests unitaires modifiers

**Emplacement** : `tests/test_modifiers.py` (si existe)

**Scénarios minimaux** :
- Shape préservé : `assert modified.shape == original.shape`
- Pureté : `original` non muté après application
- Reproductibilité : même seed → même résultat
- Composition : ordre application significatif

**Exemple** :
```python
def test_add_gaussian_noise():
    D = np.eye(10)
    modifier = add_gaussian_noise(sigma=0.1, seed=42)
    
    # Shape préservé
    D_noisy = modifier(D)
    assert D_noisy.shape == D.shape
    
    # Original non muté
    assert np.allclose(D, np.eye(10))
    
    # Reproductibilité
    modifier2 = add_gaussian_noise(sigma=0.1, seed=42)
    D_noisy2 = modifier2(D)
    assert np.allclose(D_noisy, D_noisy2)
```

### 9.2 Tests intégration

**Pipeline complet** :
```python
# Création D
D_base = create_correlation_matrix(50, seed=42)

# Application modifiers
modifiers = [add_gaussian_noise(sigma=0.05, seed=123)]
D_final = prepare_state(D_base, modifiers)

# Vérifications
assert D_final.shape == D_base.shape
assert not np.allclose(D_final, D_base)  # Modifier a agi
```

---

## SECTION 10 : COMPARAISON M0 vs M1 vs M2

### 10.1 Matrice décision

| Critère | M0 | M1 | M2 |
|---------|----|----|-----|
| Perturbation | Aucune | Gaussienne | Uniforme |
| Amplitude typique | - | σ=0.05 (5%) | ±0.1 (±10%) |
| Bornes strictes | - | Non | Oui |
| Outliers possibles | - | Oui (rares) | Non |
| Usage principal | Baseline | Robustesse outliers | Perturbation contrôlée |

### 10.2 Cas d'usage recommandés

**M0** (Baseline) :
- Établir référence
- Isoler effet γ pur
- Comparer vs modifiers

**M1** (Gaussien) :
- Tester robustesse bruit naturel
- Simuler incertitudes mesure
- Détecter fragilité numérique

**M2** (Uniforme) :
- Tester perturbation bornée
- Simuler erreurs quantification
- Détecter sensibilité paramètres

---

## ANNEXE A : TEMPLATE NOUVEAU MODIFIER

```python
"""
modifiers/my_category.py

Description catégorie modifiers.
"""

import numpy as np
from typing import Callable


def my_modifier_factory(param1: float = default_value,
                       param2: int = default_value,
                       seed: int = None) -> Callable[[np.ndarray], np.ndarray]:
    """
    Factory retournant fonction qui applique transformation X.
    
    FORME MATHÉMATIQUE:
        D' = f(D, param1, param2)
    
    Args:
        param1: Description param1
        param2: Description param2
        seed: Graine aléatoire (reproductibilité)
    
    Returns:
        Callable appliquant transformation
    
    Exemple:
        modifier = my_modifier_factory(param1=0.5, seed=42)
        D_transformed = modifier(D_base)
    """
    # Setup (ex: générateur aléatoire)
    rng = np.random.RandomState(seed) if seed is not None else np.random
    
    # Closure
    def modifier(state: np.ndarray) -> np.ndarray:
        """Applique transformation X à state."""
        # Implémentation transformation
        # ...
        return transformed_state
    
    return modifier
```

---

## ANNEXE B : HISTORIQUE MODIFICATIONS

| Date | Version | Changement |
|------|---------|------------|
| 2025-01-15 | 6.0.0 | Création catalogue initial (noise.py documenté) |

---

## ANNEXE C : INDEX ALPHABÉTIQUE MODIFIERS

| Modifier | ID | Module | Paramètres clés |
|----------|-----|--------|----------------|
| (baseline) | M0 | - | - |
| add_gaussian_noise | M1 | noise | sigma=0.05 |
| add_uniform_noise | M2 | noise | amplitude=0.1 |

**Note** : constraints/, plugins/, domains/ à documenter après inspection

---

**FIN modifier_catalog.md**