# Registries - Patterns & Best Practices

**Version** : 6.1

---

## Convention Nommage

### Format registry_key

**Standard** : `registry.function_name`

**Exemples** :
```python
'algebra.matrix_norm'
'spectral.eigenvalue_max'
'pattern.diversity'
```

**Interdictions** :
```python
# ❌ Incorrect
'matrix_norm'  # Manque registry
'algebra_matrix_norm'  # Underscore au lieu de point
'Algebra.MatrixNorm'  # CamelCase
```

---

## Gestion Erreurs

### Spécificités PRC

**Epsilon standard** :
```python
EPSILON = 1e-10

# Protection division par zéro
if abs(denominator) < EPSILON:
    return np.nan
```

**Valeurs sentinelles** :
```python
# État singulier (matrice non inversible)
if np.linalg.cond(state) > 1e10:
    return 1e10  # Sentinelle "singular"

# Non applicable (contrainte non respectée)
if state.ndim != 2:
    return np.nan
```

---

## Post-processors Standards

### Disponibles

| Post-processor | Transformation | Usage |
|---------------|----------------|-------|
| `identity` | x → x | Aucune |
| `round_2` | x → round(x, 2) | Entiers/comptages |
| `round_4` | x → round(x, 4) | Métriques standard |
| `round_6` | x → round(x, 6) | Haute précision |
| `abs` | x → \|x\| | Valeur absolue |
| `log` | x → log₁₀(x) | Échelle logarithmique |

### Usage dans COMPUTATION_SPECS

```python
COMPUTATION_SPECS = {
    'metric_name': {
        'registry_key': 'algebra.trace_value',
        'default_params': {'normalize': True},
        'post_process': 'round_4',  # ← Ici
    }
}
```

---

## Validation Contraintes

### Pattern validation 2D

```python
def compute_metric(state: np.ndarray, ...) -> float:
    """Métrique rang 2."""
    
    # Validation rang
    if state.ndim != 2:
        raise ValueError(
            f"Métrique applicable rang 2 uniquement, reçu rang {state.ndim}"
        )
    
    # Calcul
    ...
```

### Pattern validation carrée

```python
def compute_metric(state: np.ndarray, ...) -> float:
    """Métrique matrice carrée."""
    
    # Validation rang 2 carré
    if state.ndim != 2:
        raise ValueError(f"Attendu rang 2, reçu {state.ndim}")
    
    if state.shape[0] != state.shape[1]:
        raise ValueError(
            f"Matrice carrée requise, reçu {state.shape}"
        )
    
    # Calcul
    ...
```

### Pattern validation dimension minimale

```python
def compute_metric(state: np.ndarray, min_dim: int = 5, ...) -> float:
    """Métrique nécessitant espace suffisant."""
    
    # Validation dimension
    if any(dim < min_dim for dim in state.shape):
        raise ValueError(
            f"Dimension minimale {min_dim} requise, reçu {state.shape}"
        )
    
    # Calcul
    ...
```

---

## Gestion NaN/Inf

### Filtrage robuste

```python
def compute_metric(state: np.ndarray, ...) -> float:
    """Métrique robuste valeurs infinies."""
    
    # Filtrer NaN/Inf
    values = state.flatten()
    valid_values = values[np.isfinite(values)]
    
    # Vérifier données valides
    if len(valid_values) == 0:
        return np.nan  # Aucune donnée valide
    
    # Calcul
    result = np.mean(valid_values)
    return float(result)
```

---

## Protection Division Par Zéro

### Pattern standard

```python
def compute_metric(state: np.ndarray, ...) -> float:
    """Métrique avec division."""
    
    numerator = ...
    denominator = ...
    
    # Protection division
    if abs(denominator) < 1e-10:
        return np.nan  # Division par zéro
    
    result = numerator / denominator
    return float(result)
```

### Pattern coefficient variation

```python
def coefficient_variation(state: np.ndarray) -> float:
    """CV = σ / |μ|."""
    
    mean_val = np.mean(state)
    std_val = np.std(state)
    
    # Protection
    if abs(mean_val) < 1e-10:
        return np.nan
    
    cv = std_val / abs(mean_val)
    return float(cv)
```

---

## Normalisation

### Pattern normalisation état

```python
def compute_metric(state: np.ndarray, normalize: bool = False) -> float:
    """Métrique avec normalisation optionnelle."""
    
    # Calcul brut
    raw_value = ...
    
    # Normalisation
    if normalize:
        # Selon dimension
        n_elements = state.size
        normalized_value = raw_value / n_elements
        return float(normalized_value)
    
    return float(raw_value)
```

---

## Tests Unitaires

### Template test fonction registry

```python
# Dans tests/test_registries_validation.py

def test_metric_name():
    """Test algebra.metric_name."""
    
    # État test
    state = np.array([[1.0, 0.5], [0.5, 1.0]])
    
    # Appel fonction
    from registries.algebra_registry import metric_name
    result = metric_name(state, param=value)
    
    # Assertions
    assert isinstance(result, float)
    assert np.isfinite(result)
    assert result >= 0  # Si métrique positive
    
    # Cas limites
    state_singular = np.zeros((2, 2))
    result_singular = metric_name(state_singular)
    assert result_singular == 1e10 or np.isnan(result_singular)
    
    # Validation contraintes
    state_3d = np.random.rand(2, 2, 2)
    with pytest.raises(ValueError):
        metric_name(state_3d)
```

---

## Anti-patterns (À Éviter)

### ❌ Hardcoder valeurs

```python
# ❌ Mauvais
def compute_metric(state):
    threshold = 0.1  # Hardcodé
    return (state > threshold).sum()
```

```python
# ✅ Correct
def compute_metric(state, threshold: float = 0.1):
    return (state > threshold).sum()
```

### ❌ Dépendances internes PRC

```python
# ❌ Mauvais
from ..regime_utils import classify_regime

def compute_metric(state):
    regime = classify_regime(...)  # Dépendance interne
    return ...
```

```python
# ✅ Correct
def compute_metric(state, param1, param2):
    # Calcul pur, zéro dépendance interne
    result = ...
    return result
```

### ❌ Modificar état

```python
# ❌ Mauvais
def compute_metric(state):
    state[0, 0] = 0  # Modifie état !
    return np.mean(state)
```

```python
# ✅ Correct
def compute_metric(state):
    state_copy = state.copy()  # Copie si modification nécessaire
    state_copy[0, 0] = 0
    return np.mean(state_copy)
