# Catalogue Modifiers R0

## Vue d'ensemble

Modificateurs appliqués à l'état D avant exécution du mécanisme Γ.

**Architecture** :
```python
D_final = prepare_state(D_base, [modifier])
```

---

## Inventaire

### M0 - Baseline

**Description** : Aucune modification

**Implémentation** :
```python
D_final = prepare_state(D_base, [])  # Liste vide
```

**Paramètres** : Aucun

---

### M1 - Bruit Gaussien

**Description** : Perturbation additive N(0, σ)

**Implémentation** :
```python
from modifiers import add_gaussian_noise
D_final = prepare_state(D_base, [add_gaussian_noise(sigma=0.05, seed=42)])
```

**Paramètres** :
- `sigma` (float) : Amplitude du bruit (défaut 0.01)
- `seed` (int, optionnel) : Graine aléatoire

**Valeur R0** : `sigma=0.05`

---

### M2 - Bruit Uniforme

**Description** : Perturbation additive U[-a, +a]

**Implémentation** :
```python
from modifiers import add_uniform_noise
D_final = prepare_state(D_base, [add_uniform_noise(amplitude=0.1, seed=42)])
```

**Paramètres** :
- `amplitude` (float) : Amplitude max (défaut 0.01)
- `seed` (int, optionnel) : Graine aléatoire

**Valeur R0** : `amplitude=0.1`

---

### M3 - (Non implémenté)

**Statut** : Réservé

---

## Nomenclature

| ID | Fonction | Paramètres R0 |
|----|----------|---------------|
| M0 | `[]` | Aucun |
| M1 | `add_gaussian_noise` | `sigma=0.05` |
| M2 | `add_uniform_noise` | `amplitude=0.1` |
| M3 | N/A | N/A |

---

## Utilisation
```python
from modifiers import add_gaussian_noise, add_uniform_noise
from core.state_preparation import prepare_state

# M0
D = prepare_state(D_base, [])

# M1
D = prepare_state(D_base, [add_gaussian_noise(sigma=0.05, seed=42)])

# M2
D = prepare_state(D_base, [add_uniform_noise(amplitude=0.1, seed=42)])
```

---

**Version** : 1.0.0  
**Code** : `modifiers/noise.py`