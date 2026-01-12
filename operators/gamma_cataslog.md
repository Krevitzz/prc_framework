# Catalogue Gammas R0

## Inventaire

### GAM-001 - Saturation Pure Pointwise
- **Classe** : `PureSaturationGamma`
- **Forme** : `T_{n+1}[i,j] = tanh(β·T_n[i,j])`
- **Paramètres** : `beta=2.0` (défaut)
- **Applicabilité** : SYM, ASY, R3

### GAM-002 - Diffusion Pure
- **Classe** : `PureDiffusionGamma`
- **Forme** : `T_{n+1}[i,j] = T_n[i,j] + α·(∑_voisins - 4·T_n[i,j])`
- **Paramètres** : `alpha=0.05` (défaut)
- **Applicabilité** : SYM, ASY (rang 2 uniquement)

### GAM-003 - Croissance Exponentielle
- **Classe** : `ExponentialGrowthGamma`
- **Forme** : `T_{n+1}[i,j] = T_n[i,j] · exp(γ)`
- **Paramètres** : `gamma=0.05` (défaut)
- **Applicabilité** : SYM, ASY, R3
- **Note** : Conçu pour échouer (test détection explosions)

### GAM-004 - Décroissance Exponentielle
- **Classe** : `ExponentialDecayGamma`
- **Forme** : `T_{n+1}[i,j] = T_n[i,j] · exp(-γ)`
- **Paramètres** : `gamma=0.05` (défaut)
- **Applicabilité** : SYM, ASY, R3

### GAM-005 - Oscillateur Harmonique
- **Classe** : `HarmonicOscillatorGamma`
- **Forme** : `T_{n+1} = cos(ω)·T_n - sin(ω)·T_{n-1}`
- **Paramètres** : `omega=π/4` (défaut)
- **Applicabilité** : SYM, ASY, R3
- **Type** : Non-markovien (stocke T_{n-1})

### GAM-006 - Saturation + Mémoire
- **Classe** : `MemorySaturationGamma`
- **Forme** : `T_{n+1} = tanh(β·T_n + α·(T_n - T_{n-1}))`
- **Paramètres** : `beta=1.0`, `alpha=0.3` (défaut)
- **Applicabilité** : SYM, ASY, R3
- **Type** : Non-markovien (stocke T_{n-1})

### GAM-007 - Régulation Moyenne Glissante
- **Classe** : `SlidingAverageGamma`
- **Forme** : `T_{n+1}[i,j] = (1-ε)·T_n[i,j] + ε·mean(voisins_8)`
- **Paramètres** : `epsilon=0.1` (défaut)
- **Applicabilité** : SYM, ASY (rang 2 uniquement)

### GAM-008 - Mémoire Différentielle
- **Classe** : `DifferentialMemoryGamma`
- **Forme** : `T_{n+1} = tanh(T_n + γ·(T_n - T_{n-1}) + β·T_n)`
- **Paramètres** : `gamma=0.3`, `beta=1.0` (défaut)
- **Applicabilité** : SYM, ASY, R3
- **Type** : Non-markovien (stocke T_{n-1})

### GAM-009 - Saturation + Bruit Additif
- **Classe** : `StochasticSaturationGamma`
- **Forme** : `T_{n+1} = tanh(β·T_n) + σ·ε, ε ~ N(0,1)`
- **Paramètres** : `beta=1.0`, `sigma=0.01`, `seed=42` (défaut)
- **Applicabilité** : SYM, ASY, R3
- **Type** : Stochastique

### GAM-010 - Bruit Multiplicatif
- **Classe** : `MultiplicativeNoiseGamma`
- **Forme** : `T_{n+1} = tanh(T_n · (1 + σ·ε)), ε ~ N(0,1)`
- **Paramètres** : `sigma=0.05`, `seed=42` (défaut)
- **Applicabilité** : SYM, ASY, R3
- **Type** : Stochastique

### GAM-012 - Préservation Symétrie Forcée
- **Classe** : `ForcedSymmetryGamma`
- **Forme** : `T_{n+1} = (F(T_n) + F(T_n)^T) / 2, F = tanh(β·)`
- **Paramètres** : `beta=2.0` (défaut)
- **Applicabilité** : SYM, ASY (rang 2 uniquement)

### GAM-013 - Renforcement Hebbien
- **Classe** : `HebbianReinforcementGamma`
- **Forme** : `T_{n+1}[i,j] = T_n[i,j] + η·∑_k T_n[i,k]·T_n[k,j]`
- **Paramètres** : `eta=0.01` (défaut)
- **Applicabilité** : SYM, ASY (rang 2 carré uniquement)

## Table Récapitulative

| ID | Famille | Type | Rang 2 | Rang 3 | Carré Requis |
|----|---------|------|--------|--------|--------------|
| GAM-001 | Markovian | Déterministe | ✅ | ✅ | ❌ |
| GAM-002 | Markovian | Déterministe | ✅ | ❌ | ❌ |
| GAM-003 | Markovian | Déterministe | ✅ | ✅ | ❌ |
| GAM-004 | Markovian | Déterministe | ✅ | ✅ | ❌ |
| GAM-005 | Non-markovian | Déterministe | ✅ | ✅ | ❌ |
| GAM-006 | Non-markovian | Déterministe | ✅ | ✅ | ❌ |
| GAM-007 | Markovian | Déterministe | ✅ | ❌ | ❌ |
| GAM-008 | Non-markovian | Déterministe | ✅ | ✅ | ❌ |
| GAM-009 | Markovian | Stochastique | ✅ | ✅ | ❌ |
| GAM-010 | Markovian | Stochastique | ✅ | ✅ | ❌ |
| GAM-012 | Structural | Déterministe | ✅ | ❌ | ❌ |
| GAM-013 | Structural | Déterministe | ✅ | ❌ | ✅ |

## Utilisation
```python
from operators.gamma_hyp_001 import PureSaturationGamma

gamma = PureSaturationGamma(beta=2.0)
state_next = gamma(state)
```

### Gammas Non-Markoviens
```python
# GAM-005, GAM-006, GAM-008 nécessitent reset() entre runs
gamma = MemorySaturationGamma(beta=1.0, alpha=0.3)

# Run 1
for state in history:
    state_next = gamma(state)

# Avant Run 2
gamma.reset()
```

### Gammas Stochastiques
```python
# GAM-009, GAM-010 : fixer seed pour reproductibilité
gamma = StochasticSaturationGamma(beta=1.0, sigma=0.01, seed=42)
```

**Source** : `operators/gamma/gamma_hyp_*.py`