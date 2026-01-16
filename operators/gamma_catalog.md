# operators_catalog.md

> Catalogue fonctionnel des opérateurs γ (gamma)  
> Responsabilité : Définitions mécanismes Δ agissant sur D  
> Version : 6.0  
> Dernière mise à jour : 2025-01-15

---

## VUE D'ENSEMBLE

Le module `operators/` définit les **mécanismes γ** qui transforment l'état D itérativement.

**IDs catalogués** : GAM-001 à GAM-013 (11 opérateurs documentés, GAM-011 absent)

**Familles** :
- **markovian** : État suivant dépend uniquement de l'état courant
- **non_markovian** : Mémoire explicite (stocke états précédents)
- **stochastic** : Comportement non déterministe (bruit)
- **structural** : Préservation/création structures (symétrie, clusters)

**Principe fondamental** :
- γ = **classe callable** avec méthode `__call__(state) → state_next`
- Factory `create_gamma_hyp_XXX()` retourne instance
- Core reste aveugle (pas de validation dans γ)
- γ gère sa propre mémoire (si non-markovien)

---

## SECTION 1 : FAMILLE MARKOVIAN

### 1.1 GAM-001 : Saturation pure pointwise

**Fichier** : `gamma_hyp_001.py`

**Classe** : `PureSaturationGamma`

**Signature** :
```python
def create_gamma_hyp_001(beta: float = 2.0) -> Callable[[np.ndarray], np.ndarray]
```

**Forme mathématique** :
```
T_{n+1}[i,j] = tanh(β · T_n[i,j])
```

**Paramètres** :
| Paramètre | Type | Range | Défaut | Description |
|-----------|------|-------|--------|-------------|
| `beta` | `float` | (0, +∞) | 2.0 | Force de saturation |

**Présuppositions** :
- Aucune (applicable tout rang, toute dimension)

**Applicabilité D** : SYM, ASY, R3 (tous encodages)

**Comportement attendu** :
- **Convergence** : Rapide (typiquement < 200 iterations)
- **Attracteurs** : [-1, 1] (bornes tanh)
- **Diversité** : Perte progressive (homogénéisation)
- **Trivial** : Oui (convergence vers constante ou zéro)

**Grilles paramètres** :
```python
PARAM_GRID_PHASE1 = {'nominal': {'beta': 2.0}}

PARAM_GRID_PHASE2 = {
    'beta_low': {'beta': 0.5},
    'beta_nominal': {'beta': 1.0},
    'beta_high': {'beta': 2.0},
    'beta_very_high': {'beta': 5.0},
}
```

**Notes** :
- Mécanisme le plus simple (pointwise, sans couplage)
- Test baseline pour saturation
- Préserve symétrie si D symétrique (pointwise)

---

### 1.2 GAM-002 : Diffusion pure (Laplacien)

**Fichier** : `gamma_hyp_002.py`

**Classe** : `PureDiffusionGamma`

**Signature** :
```python
def create_gamma_hyp_002(alpha: float = 0.05) -> Callable[[np.ndarray], np.ndarray]
```

**Forme mathématique** :
```
T_{n+1}[i,j] = T_n[i,j] + α·∇²T_n[i,j]

où ∇²T_n[i,j] = (T_n[i-1,j] + T_n[i+1,j] + T_n[i,j-1] + T_n[i,j+1]) - 4·T_n[i,j]
```

**Paramètres** :
| Paramètre | Type | Range | Défaut | Description |
|-----------|------|-------|--------|-------------|
| `alpha` | `float` | (0, 0.25) | 0.05 | Coefficient diffusion (stabilité Von Neumann) |

**Présuppositions** :
- **Rang 2 uniquement** (voisinage 4-connexe 2D)
- Conditions limites périodiques (np.roll)

**Applicabilité D** : SYM, ASY (pas R3)

**Comportement attendu** :
- **Convergence** : Rapide (< 500 iterations)
- **Attracteurs** : Uniforme (homogénéisation totale)
- **Diversité** : Perte totale
- **Trivial** : Oui

**Validation** :
```python
if state.ndim != 2:
    raise ValueError(f"Applicable uniquement rang 2, reçu {state.ndim}")
```

**Grilles paramètres** :
```python
PARAM_GRID_PHASE1 = {'nominal': {'alpha': 0.05}}

PARAM_GRID_PHASE2 = {
    'alpha_low': {'alpha': 0.01},
    'alpha_nominal': {'alpha': 0.05},
    'alpha_high': {'alpha': 0.1},
}
```

**Notes** :
- Voisinage 4-connexe (haut, bas, gauche, droite)
- Stabilité : α < 0.25 (condition Von Neumann)
- Lisse toute structure initiale

---

### 1.3 GAM-003 : Croissance exponentielle

**Fichier** : `gamma_hyp_003.py`

**Classe** : `ExponentialGrowthGamma`

**Signature** :
```python
def create_gamma_hyp_003(gamma: float = 0.05) -> Callable[[np.ndarray], np.ndarray]
```

**Forme mathématique** :
```
T_{n+1}[i,j] = T_n[i,j] · exp(γ)
```

**Paramètres** :
| Paramètre | Type | Range | Défaut | Description |
|-----------|------|-------|--------|-------------|
| `gamma` | `float` | (0, +∞) | 0.05 | Taux de croissance exponentielle |

**Applicabilité D** : SYM, ASY, R3

**Comportement attendu** :
- **Convergence** : Jamais (divergence exponentielle)
- **Attracteurs** : Aucun (explosion)
- **Diversité** : Explosion
- **Trivial** : Non (mais échec attendu)
- **Expected failure** : True

**Grilles paramètres** :
```python
PARAM_GRID_PHASE1 = {'nominal': {'gamma': 0.05}}

PARAM_GRID_PHASE2 = {
    'gamma_low': {'gamma': 0.01},
    'gamma_nominal': {'gamma': 0.05},
    'gamma_high': {'gamma': 0.1},
}
```

**Notes** :
- **CONÇU POUR ÉCHOUER**
- Validation détection explosions
- Explosion typique < 100 iterations (γ=0.05)
- Attendu : REJECTED[GLOBAL]

---

### 1.4 GAM-004 : Décroissance exponentielle

**Fichier** : `gamma_hyp_004.py`

**Classe** : `ExponentialDecayGamma`

**Signature** :
```python
def create_gamma_hyp_004(gamma: float = 0.05) -> Callable[[np.ndarray], np.ndarray]
```

**Forme mathématique** :
```
T_{n+1}[i,j] = T_n[i,j] · exp(-γ)
```

**Paramètres** :
| Paramètre | Type | Range | Défaut | Description |
|-----------|------|-------|--------|-------------|
| `gamma` | `float` | (0, +∞) | 0.05 | Taux de décroissance exponentielle |

**Applicabilité D** : SYM, ASY, R3

**Comportement attendu** :
- **Convergence** : Rapide (< 500 iterations)
- **Attracteurs** : Zéro (trivial)
- **Diversité** : Perte totale
- **Trivial** : Oui

**Grilles paramètres** :
```python
PARAM_GRID_PHASE1 = {'nominal': {'gamma': 0.05}}

PARAM_GRID_PHASE2 = {
    'gamma_low': {'gamma': 0.01},
    'gamma_nominal': {'gamma': 0.05},
    'gamma_high': {'gamma': 0.1},
}
```

**Notes** :
- Convergence exponentielle vers 0
- Temps caractéristique : 1/γ iterations
- Attendu : REJECTED[R0] pour trivialité

---

### 1.5 GAM-007 : Régulation moyenne glissante

**Fichier** : `gamma_hyp_007.py`

**Classe** : `SlidingAverageGamma`

**Signature** :
```python
def create_gamma_hyp_007(epsilon: float = 0.1) -> Callable[[np.ndarray], np.ndarray]
```

**Forme mathématique** :
```
T_{n+1}[i,j] = (1-ε)·T_n[i,j] + ε·mean(voisins_8(T_n[i,j]))

Voisins 8-connexe : haut-gauche, haut, haut-droite, gauche, droite, bas-gauche, bas, bas-droite
```

**Paramètres** :
| Paramètre | Type | Range | Défaut | Description |
|-----------|------|-------|--------|-------------|
| `epsilon` | `float` | [0, 1] | 0.1 | Force régulation (0=identité, 1=moyenne pure) |

**Présuppositions** :
- **Rang 2 uniquement** (voisinage 2D)
- Conditions limites périodiques

**Applicabilité D** : SYM, ASY (pas R3)

**Comportement attendu** :
- **Convergence** : Moyenne (500-1000 iterations)
- **Attracteurs** : Uniforme (plus lent que GAM-002)
- **Diversité** : Perte progressive
- **Trivial** : Oui

**Validation** :
```python
if state.ndim != 2:
    raise ValueError(f"Applicable uniquement rang 2, reçu {state.ndim}")
```

**Grilles paramètres** :
```python
PARAM_GRID_PHASE1 = {'nominal': {'epsilon': 0.1}}

PARAM_GRID_PHASE2 = {
    'epsilon_low': {'epsilon': 0.05},
    'epsilon_nominal': {'epsilon': 0.1},
    'epsilon_high': {'epsilon': 0.2},
}
```

**Notes** :
- Voisinage 8-connexe (diagonales incluses)
- Plus doux que diffusion Laplacienne (GAM-002)
- Implémentation O(N²) (boucles Python, peut être lent)

---

## SECTION 2 : FAMILLE NON_MARKOVIAN

### 2.1 GAM-005 : Oscillateur harmonique

**Fichier** : `gamma_hyp_005.py`

**Classe** : `HarmonicOscillatorGamma`

**Signature** :
```python
def create_gamma_hyp_005(omega: float = np.pi / 4) -> Callable[[np.ndarray], np.ndarray]
```

**Forme mathématique** :
```
T_{n+1} = cos(ω)·T_n - sin(ω)·T_{n-1}
```

**Paramètres** :
| Paramètre | Type | Range | Défaut | Description |
|-----------|------|-------|--------|-------------|
| `omega` | `float` | (0, π) | π/4 | Fréquence angulaire (période = 2π/ω) |

**Mémoire** :
- Stocke `T_{n-1}` dans `self._previous_state`
- Première itération : comportement identité

**Applicabilité D** : SYM, ASY, R3

**Comportement attendu** :
- **Convergence** : Jamais (oscillations périodiques)
- **Attracteurs** : Cycle périodique
- **Diversité** : Conservation (théorique)
- **Trivial** : Non

**Méthodes** :
```python
def reset(self):
    """Réinitialise la mémoire."""
    self._previous_state = None
```

**Grilles paramètres** :
```python
PARAM_GRID_PHASE1 = {'nominal': {'omega': np.pi / 4}}

PARAM_GRID_PHASE2 = {
    'omega_slow': {'omega': np.pi / 8},
    'omega_nominal': {'omega': np.pi / 4},
    'omega_fast': {'omega': np.pi / 2},
}
```

**Notes** :
- Non-markovien ordre 1
- Conservation énergie théorique (norme constante)
- Période : 2π/ω iterations
- Intéressant pour test détection périodicité

---

### 2.2 GAM-006 : Saturation + mémoire ordre-1

**Fichier** : `gamma_hyp_006.py`

**Classe** : `MemorySaturationGamma`

**Signature** :
```python
def create_gamma_hyp_006(beta: float = 1.0, alpha: float = 0.3) -> Callable[[np.ndarray], np.ndarray]
```

**Forme mathématique** :
```
T_{n+1} = tanh(β·T_n + α·(T_n - T_{n-1}))

où (T_n - T_{n-1}) = vélocité
```

**Paramètres** :
| Paramètre | Type | Range | Défaut | Description |
|-----------|------|-------|--------|-------------|
| `beta` | `float` | (0, +∞) | 1.0 | Force de saturation |
| `alpha` | `float` | [0, 1] | 0.3 | Poids de la mémoire (inertie) |

**Mémoire** :
- Stocke `T_{n-1}` dans `self._previous_state`
- Première itération : `tanh(β·T_n)` (markovien)

**Applicabilité D** : SYM, ASY, R3

**Comportement attendu** :
- **Convergence** : Plus lent que markovien (inertie)
- **Attracteurs** : Non-triviaux possibles
- **Diversité** : Possible préservation avec α adéquat
- **Trivial** : Non

**Méthodes** :
```python
def reset(self):
    """Réinitialise la mémoire (entre runs)."""
    self._previous_state = None
```

**Grilles paramètres** :
```python
PARAM_GRID_PHASE1 = {'nominal': {'beta': 1.0, 'alpha': 0.3}}

PARAM_GRID_PHASE2 = {
    'mem_weak_low_sat': {'beta': 1.0, 'alpha': 0.1},
    'mem_weak_high_sat': {'beta': 2.0, 'alpha': 0.1},
    'mem_mid_low_sat': {'beta': 1.0, 'alpha': 0.3},
    'mem_mid_high_sat': {'beta': 2.0, 'alpha': 0.3},
    'mem_strong_low_sat': {'beta': 1.0, 'alpha': 0.5},
    'mem_strong_high_sat': {'beta': 2.0, 'alpha': 0.5},
}
```

**Notes** :
- Inertie peut éviter attracteurs triviaux
- Appeler `reset()` entre runs différents

---

### 2.3 GAM-008 : Mémoire différentielle

**Fichier** : `gamma_hyp_008.py`

**Classe** : `DifferentialMemoryGamma`

**Signature** :
```python
def create_gamma_hyp_008(gamma: float = 0.3, beta: float = 1.0) -> Callable[[np.ndarray], np.ndarray]
```

**Forme mathématique** :
```
T_{n+1} = tanh(T_n + γ·(T_n - T_{n-1}) + β·T_n)
```

**Paramètres** :
| Paramètre | Type | Range | Défaut | Description |
|-----------|------|-------|--------|-------------|
| `gamma` | `float` | [0, 1] | 0.3 | Poids inertie (vélocité) |
| `beta` | `float` | (0, +∞) | 1.0 | Force saturation |

**Mémoire** :
- Stocke `T_{n-1}` dans `self._previous_state`
- Première itération : `tanh(β·T_n)` (markovien)

**Applicabilité D** : SYM, ASY, R3

**Comportement attendu** :
- **Convergence** : Oscillations amorties possibles
- **Attracteurs** : Non-triviaux possibles
- **Diversité** : Maintien possible avec γ adéquat
- **Trivial** : Non

**Méthodes** :
```python
def reset(self):
    self._previous_state = None
```

**Grilles paramètres** :
```python
PARAM_GRID_PHASE1 = {'nominal': {'gamma': 0.3, 'beta': 1.0}}

PARAM_GRID_PHASE2 = {
    'low_inertia_low_sat': {'gamma': 0.1, 'beta': 1.0},
    'low_inertia_high_sat': {'gamma': 0.1, 'beta': 2.0},
    'high_inertia_low_sat': {'gamma': 0.5, 'beta': 1.0},
    'high_inertia_high_sat': {'gamma': 0.5, 'beta': 2.0},
}
```

**Notes** :
- Similaire GAM-006 mais avec terme β additionnel
- Combine inertie + saturation + friction
- Oscillations amorties si bien paramétré

---

## SECTION 3 : FAMILLE STOCHASTIC

### 3.1 GAM-009 : Saturation + bruit additif

**Fichier** : `gamma_hyp_009.py`

**Classe** : `StochasticSaturationGamma`

**Signature** :
```python
def create_gamma_hyp_009(beta: float = 1.0, sigma: float = 0.01, seed: int = None) -> Callable[[np.ndarray], np.ndarray]
```

**Forme mathématique** :
```
T_{n+1} = tanh(β·T_n) + σ·ε
où ε ~ N(0, 1)
```

**Paramètres** :
| Paramètre | Type | Range | Défaut | Description |
|-----------|------|-------|--------|-------------|
| `beta` | `float` | (0, +∞) | 1.0 | Force de saturation |
| `sigma` | `float` | [0, +∞) | 0.01 | Amplitude bruit gaussien |
| `seed` | `int` | ℕ | None | Graine aléatoire (reproductibilité) |

**Stochasticité** :
- Utilise `np.random.RandomState(seed)` si seed fourni
- Sinon `np.random` (non déterministe)

**Applicabilité D** : SYM, ASY, R3

**Comportement attendu** :
- **Convergence** : Équilibre stochastique possible
- **Attracteurs** : Distribution stationnaire
- **Diversité** : Maintien possible avec σ adéquat
- **Trivial** : Non

**Méthodes** :
```python
def reset(self):
    pass  # API consistente (pas de reset nécessaire)
```

**Grilles paramètres** :
```python
PARAM_GRID_PHASE1 = {'nominal': {'beta': 1.0, 'sigma': 0.01, 'seed': 42}}

PARAM_GRID_PHASE2 = {
    'low_sat_low_noise': {'beta': 1.0, 'sigma': 0.01, 'seed': 42},
    'high_sat_low_noise': {'beta': 2.0, 'sigma': 0.01, 'seed': 42},
    'low_sat_high_noise': {'beta': 1.0, 'sigma': 0.05, 'seed': 42},
    'high_sat_high_noise': {'beta': 2.0, 'sigma': 0.05, 'seed': 42},
}
```

**Notes** :
- Processus stochastique (non-déterministe)
- Fixer seed pour reproductibilité
- Balance déterminisme (β) / exploration (σ)
- TEST-UNIV-004 (sensibilité CI) particulièrement pertinent
- Moyenner sur plusieurs seeds pour analyses

---

### 3.2 GAM-010 : Bruit multiplicatif

**Fichier** : `gamma_hyp_010.py`

**Classe** : `MultiplicativeNoiseGamma`

**Signature** :
```python
def create_gamma_hyp_010(sigma: float = 0.05, seed: int = None) -> Callable[[np.ndarray], np.ndarray]
```

**Forme mathématique** :
```
T_{n+1} = tanh(T_n · (1 + σ·ε))
où ε ~ N(0, 1)
```

**Paramètres** :
| Paramètre | Type | Range | Défaut | Description |
|-----------|------|-------|--------|-------------|
| `sigma` | `float` | [0, +∞) | 0.05 | Amplitude bruit multiplicatif |
| `seed` | `int` | ℕ | None | Graine aléatoire |

**Stochasticité** :
- Utilise `np.random.RandomState(seed)` si seed fourni

**Applicabilité D** : SYM, ASY, R3

**Comportement attendu** :
- **Convergence** : Variable (dépend σ)
- **Attracteurs** : Structures amplifiées ou chaos
- **Diversité** : Possible augmentation (amplification)
- **Trivial** : Non

**Méthodes** :
```python
def reset(self):
    pass
```

**Grilles paramètres** :
```python
PARAM_GRID_PHASE1 = {'nominal': {'sigma': 0.05, 'seed': 42}}

PARAM_GRID_PHASE2 = {
    'sigma_low': {'sigma': 0.01, 'seed': 42},
    'sigma_nominal': {'sigma': 0.05, 'seed': 42},
    'sigma_high': {'sigma': 0.1, 'seed': 42},
}
```

**Notes** :
- Bruit multiplicatif : amplifie proportionnellement
- Différent de GAM-009 (bruit additif)
- Saturation nécessaire pour stabilité
- Risque avalanche si σ > 0.2
- Peut créer hétérogénéité (riches plus riches)

---

## SECTION 4 : FAMILLE STRUCTURAL

### 4.1 GAM-012 : Préservation symétrie forcée

**Fichier** : `gamma_hyp_012.py`

**Classe** : `ForcedSymmetryGamma`

**Signature** :
```python
def create_gamma_hyp_012(beta: float = 2.0) -> Callable[[np.ndarray], np.ndarray]
```

**Forme mathématique** :
```
T_{n+1} = (F(T_n) + F(T_n)^T) / 2
où F(X) = tanh(β·X)
```

**Paramètres** :
| Paramètre | Type | Range | Défaut | Description |
|-----------|------|-------|--------|-------------|
| `beta` | `float` | (0, +∞) | 2.0 | Force de saturation |

**Présuppositions** :
- **Rang 2 uniquement** (transposée matricielle)

**Applicabilité D** : SYM, ASY (pas R3)

**Comportement attendu** :
- **Convergence** : Similaire GAM-001 mais symétrique
- **Attracteurs** : Symétriques garantis
- **Diversité** : Possible perte comme saturation pure
- **Trivial** : Possible (comme GAM-001)

**Validation** :
```python
if state.ndim != 2:
    raise ValueError(f"Applicable uniquement rang 2, reçu {state.ndim}")
```

**Grilles paramètres** :
```python
PARAM_GRID_PHASE1 = {'nominal': {'beta': 2.0}}

PARAM_GRID_PHASE2 = {
    'beta_low': {'beta': 1.0},
    'beta_nominal': {'beta': 2.0},
    'beta_high': {'beta': 5.0},
}
```

**Notes** :
- Force symétrie de manière artificielle
- TEST-SYM-001 devrait toujours PASS
- TEST-SYM-002 : peut créer symétrie depuis ASY
- Robuste au bruit asymétrique (M1, M2)
- Question : forçage aide-t-il non-trivialité ?

---

### 4.2 GAM-013 : Renforcement hebbien

**Fichier** : `gamma_hyp_013.py`

**Classe** : `HebbianReinforcementGamma`

**Signature** :
```python
def create_gamma_hyp_013(eta: float = 0.01) -> Callable[[np.ndarray], np.ndarray]
```

**Forme mathématique** :
```
T_{n+1}[i,j] = T_n[i,j] + η·Σ_k T_n[i,k]·T_n[k,j]
ou en notation matricielle :
T_{n+1} = T_n + η·(T_n @ T_n)
```

**Paramètres** :
| Paramètre | Type | Range | Défaut | Description |
|-----------|------|-------|--------|-------------|
| `eta` | `float` | [0, 0.1] | 0.01 | Taux d'apprentissage hebbien |

**Présuppositions** :
- **Rang 2 carrée uniquement** (produit matriciel T @ T)

**Applicabilité D** : SYM, ASY (matrices carrées uniquement)

**Comportement attendu** :
- **Convergence** : Instable (risque explosion)
- **Attracteurs** : Structures émergentes ou explosion
- **Diversité** : Augmentation (clusters)
- **Trivial** : Non

**Validation** :
```python
if state.ndim != 2:
    raise ValueError(f"Applicable uniquement rang 2, reçu {state.ndim}")
if state.shape[0] != state.shape[1]:
    raise ValueError(f"Nécessite matrice carrée, reçu {state.shape}")
```

**Grilles paramètres** :
```python
PARAM_GRID_PHASE1 = {'nominal': {'eta': 0.01}}

PARAM_GRID_PHASE2 = {
    'eta_very_low': {'eta': 0.001},
    'eta_low': {'eta': 0.01},
    'eta_high': {'eta': 0.05},
}
```

**Notes** :
- **INSTABLE sans régulation additionnelle**
- Produit matriciel T @ T (coûteux : O(N³))
- Risque explosion si η trop grand ou D mal conditionné
- Intéressant si combiné avec saturation (voir GAM-103[R1])
- Peut créer structures hiérarchiques
- TEST-UNIV-001 (norme) critique pour détecter explosions

---

## SECTION 5 : MAPPING IDs ↔ GAMMAS

### 5.1 Table complète (catalogués)

| ID | Famille | Classe | Paramètres défaut | Applicabilité |
|----|---------|--------|-------------------|---------------|
| GAM-001 | markovian | PureSaturationGamma | beta=2.0 | SYM, ASY, R3 |
| GAM-002 | markovian | PureDiffusionGamma | alpha=0.05 | SYM, ASY |
| GAM-003 | markovian | ExponentialGrowthGamma | gamma=0.05 | SYM, ASY, R3 |
| GAM-004 | markovian | ExponentialDecayGamma | gamma=0.05 | SYM, ASY, R3 |
| GAM-005 | non_markovian | HarmonicOscillatorGamma | omega=π/4 | SYM, ASY, R3 |
| GAM-006 | non_markovian | MemorySaturationGamma | beta=1.0, alpha=0.3 | SYM, ASY, R3 |
| GAM-007 | non_markovian | SlidingAverageGamma | epsilon=0.1 | SYM, ASY |
| GAM-008 | non_markovian | DifferentialMemoryGamma | gamma=0.3, beta=1.0 | SYM, ASY, R3 |
| GAM-009 | stochastic | StochasticSaturationGamma | beta=1.0, sigma=0.01, seed=42 | SYM, ASY, R3 |
| GAM-010 | stochastic | MultiplicativeNoiseGamma | sigma=0.05, seed=42 | SYM, ASY, R3 |
| GAM-012 | structural | ForcedSymmetryGamma | beta=2.0 | SYM, ASY |
| GAM-013 | structural | HebbianReinforcementGamma | eta=0.01 | SYM, ASY (carrée) |

**Note** : GAM-011 absent (non implémenté ou réservé R1+)

### 5.2 Implémentation mapping (suggestion)

```python
# operators/__init__.py
from .gamma_hyp_001 import create_gamma_hyp_001
from .gamma_hyp_002 import create_gamma_hyp_002
from .gamma_hyp_003 import create_gamma_hyp_003
from .gamma_hyp_004 import create_gamma_hyp_004
from .gamma_hyp_005 import create_gamma_hyp_005
from .gamma_hyp_006 import create_gamma_hyp_006
from .gamma_hyp_007 import create_gamma_hyp_007
from .gamma_hyp_008 import create_gamma_hyp_008
from .gamma_hyp_009 import create_gamma_hyp_009
from .gamma_hyp_010 import create_gamma_hyp_010
from .gamma_hyp_012 import create_gamma_hyp_012
from .gamma_hyp_013 import create_gamma_hyp_013

GAMMA_REGISTRY = {
    'GAM-001': create_gamma_hyp_001,
    'GAM-002': create_gamma_hyp_002,
    'GAM-003': create_gamma_hyp_003,
    'GAM-004': create_gamma_hyp_004,
    'GAM-005': create_gamma_hyp_005,
    'GAM-006': create_gamma_hyp_006,
    'GAM-007': create_gamma_hyp_007,
    'GAM-008': create_gamma_hyp_008,
    'GAM-009': create_gamma_hyp_009,
    'GAM-010': create_gamma_hyp_010,
    'GAM-012': create_gamma_hyp_012,
    'GAM-013': create_gamma_hyp_013,
}

def get_gamma(gamma_id: str, **params):
    """
    Retourne instance gamma depuis ID et paramètres.
    
    Args:
        gamma_id: ID gamma (ex: "GAM-001")
        **params: Paramètres spécifiques au gamma
    
    Returns:
        Instance callable gamma
    """
    if gamma_id not in GAMMA_REGISTRY:
        raise ValueError(f"Unknown gamma_id: {gamma_id}")
    
    factory = GAMMA_REGISTRY[gamma_id]
    return factory(**params)
```

**Usage** :
```python
from operators import get_gamma

# Création gamma avec paramètres défaut
gamma = get_gamma('GAM-001')

# Création avec paramètres custom
gamma = get_gamma('GAM-006', beta=1.5, alpha=0.4)

# Utilisation dans kernel
for i, state in run_kernel(D_initial, gamma, max_iterations=1000):
    pass
```

---

## SECTION 6 : GRAPHE DE DÉPENDANCES

### 6.1 Relations inter-modules

```
gamma_hyp_*.py
    ├─ Appelé par : batch_runner.py (construction pipeline)
    ├─ Reçoit : paramètres (beta, alpha, sigma, etc.)
    └─ Retourne : Instance callable (classe avec __call__)
    
Classe Gamma
    ├─ Méthode __call__(state) → state_next
    ├─ Méthode reset() (si non-markovien ou stochastique)
    ├─ Méthode __repr__() (affichage)
    └─ Attributs : paramètres + mémoire (si applicable)
```

### 6.2 Flux typique exécution

```
1. batch_runner.py lit gamma_id (ex: "GAM-006")
   ↓
2. Mapping vers factory :
   "GAM-006" → create_gamma_hyp_006
   ↓
3. Appel factory avec paramètres :
   gamma = create_gamma_hyp_006(beta=1.0, alpha=0.3)
   ↓
4. Retour instance callable
   ↓
5. run_kernel(D_initial, gamma, ...)
   ↓
6. Boucle kernel :
   for i, state in run_kernel(...):
       # kernel appelle gamma(state) → state_next
```

---

## SECTION 7 : INVARIANTS CRITIQUES

### 7.1 Pattern classe callable obligatoire

**R7.1-A** : Tous gammas DOIVENT être classes avec `__call__`

```python
# ✅ CORRECT
class MyGamma:
    def __init__(self, param):
        self.param = param
    
    def __call__(self, state):
        return transformed_state

# ❌ INTERDIT (fonction directe)
def my_gamma(state, param):  # ❌
    return transformed_state
```

**Rationale** :
- Paramètres stockés dans instance
- Mémoire gérée par instance (si non-markovien)
- API uniforme : `gamma(state) → state_next`

---

### 7.2 Validation technique (pas sémantique)

**R7.2-A** : Gammas peuvent valider contraintes techniques

```python
# ✅ AUTORISÉ (validation rang)
def __call__(self, state):
    if state.ndim != 2:
        raise ValueError("Applicable uniquement rang 2")
    return transformed_state

# ✅ AUTORISÉ (validation shape carrée)
def __call__(self, state):
    if state.shape[0] != state.shape[1]:
        raise ValueError("Nécessite matrice carrée")
    return transformed_state

# ❌ INTERDIT (validation sémantique)
def __call__(self, state):
    if not is_symmetric(state):  # ❌
        raise ValueError("État non symétrique")
    return transformed_state
```

**Principe** :
- Validation dimensionnelle → autorisée (contraintes applicabilité)
- Validation sémantique → interdite (reste dans tests/)

---

### 7.3 Gestion mémoire (non-markovien)

**R7.3-A** : Mémoire stockée dans `self._previous_state`

**R7.3-B** : Méthode `reset()` obligatoire

```python
# ✅ CORRECT
class MyNonMarkovianGamma:
    def __init__(self, param):
        self.param = param
        self._previous_state = None
    
    def __call__(self, state):
        if self._previous_state is None:
            # Première itération
            result = markovian_behavior(state)
        else:
            # Itérations suivantes
            result = use_memory(state, self._previous_state)
        
        self._previous_state = state.copy()
        return result
    
    def reset(self):
        self._previous_state = None
```

**R7.3-C** : Appeler `reset()` entre runs différents

---

### 7.4 Gestion stochasticité

**R7.4-A** : Seed optionnel, générateur local

```python
# ✅ CORRECT
class MyStochasticGamma:
    def __init__(self, sigma, seed=None):
        self.sigma = sigma
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
    
    def __call__(self, state):
        noise = self.rng.randn(*state.shape)
        return state + noise * self.sigma
```

**R7.4-B** : Ne pas utiliser `np.random.seed()` global

---

### 7.5 Méthode __repr__

**R7.5-A** : Tous gammas DOIVENT avoir `__repr__`

```python
# ✅ CORRECT
def __repr__(self):
    return f"MyGamma(param1={self.param1}, param2={self.param2})"
```

**Usage** : Debugging, logs, traçabilité

---

## SECTION 8 : EXTENSIONS FUTURES

### 8.1 Ajout nouveau gamma (checklist)

Avant d'ajouter un gamma :

- [ ] ID unique (GAM-NNN)
- [ ] Famille identifiée (markovian, non_markovian, stochastic, structural)
- [ ] Classe avec `__call__(state) → state_next`
- [ ] Factory `create_gamma_hyp_NNN(**params) → instance`
- [ ] Paramètres avec défauts raisonnables
- [ ] Validation technique (si contraintes applicabilité)
- [ ] Méthode `reset()` (si non-markovien ou stochastique)
- [ ] Méthode `__repr__()`
- [ ] Docstring classe complète (forme mathématique, comportement attendu)
- [ ] PARAM_GRID_PHASE1 et PHASE2 définis
- [ ] METADATA complet
- [ ] Ajouté à GAMMA_REGISTRY
- [ ] Documenté dans ce catalogue

### 8.2 Extensions INTERDITES

❌ **Validation sémantique dans gamma** :
```python
# INTERDIT
def __call__(self, state):
    if not is_symmetric(state):  # ❌
        raise ValueError(...)
```

❌ **Mutation état entrée** :
```python
# INTERDIT
def __call__(self, state):
    state += delta  # ❌ Mute entrée
    return state
```

❌ **Branchement basé sur metadata externe** :
```python
# INTERDIT
def __call__(self, state):
    if encoding_id == "SYM-001":  # ❌ Gamma ne connaît pas encoding_id
        # Traitement spécial
```

❌ **Seed global** :
```python
# INTERDIT
def __init__(self, sigma, seed):
    np.random.seed(seed)  # ❌ Pollue état global
```

### 8.3 Extensions AUTORISÉES

✅ **Ajout paramètres configurables** :
```python
# OK
class MyGamma:
    def __init__(self, param1, param2, clip_bounds=None):
        self.param1 = param1
        self.param2 = param2
        self.clip_bounds = clip_bounds
```

✅ **Composition interne** :
```python
# OK
class ComposedGamma:
    def __init__(self, gamma1, gamma2):
        self.gamma1 = gamma1
        self.gamma2 = gamma2
    
    def __call__(self, state):
        return self.gamma2(self.gamma1(state))
```

✅ **Métriques internes (pas validation)** :
```python
# OK (si pas validation)
class MyGamma:
    def __call__(self, state):
        self._last_norm = np.linalg.norm(state)  # Mémorise (pas valide)
        return transformed_state
```

---

## SECTION 9 : NOTES TECHNIQUES

### 9.1 Stabilité numérique

**Guidelines par famille** :

| Famille | Stabilité | Précautions |
|---------|-----------|-------------|
| markovian (saturation) | Haute | tanh borne automatiquement |
| markovian (diffusion) | Moyenne | α < 0.25 (Von Neumann) |
| markovian (exponentiel) | Faible | Explosion/collapse attendus |
| non_markovian | Variable | Dépend balance mémoire/saturation |
| stochastic | Moyenne | σ trop grand → instabilité |
| structural | Variable | Dépend mécanisme spécifique |

### 9.2 Coût computationnel

**Complexité par gamma** :

| Gamma | Complexité | Notes |
|-------|------------|-------|
| GAM-001 | O(N²) | Pointwise, optimal |
| GAM-002 | O(N²) | Laplacien vectorisé |
| GAM-003, GAM-004 | O(N²) | Pointwise |
| GAM-005, GAM-006, GAM-008 | O(N²) | + copie mémoire |
| GAM-007 | O(N²) boucles | Lent (Python loops) |
| GAM-009, GAM-010 | O(N²) | + génération bruit |
| GAM-012 | O(N²) | + transposée |
| GAM-013 | O(N³) | Produit matriciel |

**Recommandations** :
- GAM-007 : Vectoriser si possible (R1+)
- GAM-013 : Limiter dimension (N ≤ 100)

### 9.3 Reproductibilité

**Niveaux seed** :

```python
# Seed global execution
exec_seed = 42

# Seed D_base
D_base = create_random_uniform(50, seed=exec_seed)

# Seed modifier
modifiers = [add_gaussian_noise(sigma=0.05, seed=exec_seed + 1000)]

# Seed gamma (si stochastique)
gamma = create_gamma_hyp_009(beta=1.0, sigma=0.01, seed=exec_seed + 2000)
```

**Convention batch_runner** :
- `seed` (DB) : seed execution globale
- Seed gamma stochastique : dérivé ou fixe (selon config)

### 9.4 Comparaison gammas similaires

**Saturation** :
- GAM-001 : Pure pointwise
- GAM-006 : + mémoire ordre-1
- GAM-008 : + mémoire + friction
- GAM-009 : + bruit additif
- GAM-012 : + forçage symétrie

**Bruit** :
- GAM-009 : Bruit additif (exploration uniforme)
- GAM-010 : Bruit multiplicatif (amplification sélective)

**Diffusion** :
- GAM-002 : Laplacien 4-connexe
- GAM-007 : Moyenne 8-connexe

---

## SECTION 10 : TESTS ASSOCIÉS

### 10.1 Tests unitaires gammas

**Emplacement** : `tests/test_operators.py` (si existe)

**Scénarios minimaux** :
- Shape préservé : `assert gamma(state).shape == state.shape`
- Pureté : `state` non muté après appel
- Reproductibilité (stochastique) : même seed → même résultat
- Reset (non-markovien) : `gamma.reset()` réinitialise mémoire
- Validation applicabilité : erreur si rang incorrect

**Exemple** :
```python
def test_gamma_hyp_001():
    gamma = create_gamma_hyp_001(beta=2.0)
    state = np.random.randn(10, 10)
    
    # Shape préservé
    state_next = gamma(state)
    assert state_next.shape == state.shape
    
    # Bornes tanh
    assert np.all(state_next >= -1) and np.all(state_next <= 1)
    
    # Pureté
    state_copy = state.copy()
    gamma(state)
    assert np.allclose(state, state_copy)
```

### 10.2 Tests intégration

**Pipeline complet** :
```python
# Création D
D_base = create_correlation_matrix(50, seed=42)
D_final = prepare_state(D_base, [add_gaussian_noise(sigma=0.05, seed=123)])

# Création gamma
gamma = get_gamma('GAM-001', beta=2.0)

# Exécution kernel
history = []
for i, state in run_kernel(D_final, gamma, max_iterations=1000):
    history.append(state.copy())
    if i >= 200:
        break

# Vérifications
assert len(history) == 201
assert all(s.shape == D_final.shape for s in history)
```

---

## SECTION 11 : METADATA STRUCTURE

### 11.1 Format METADATA dict

**Tous fichiers gamma_hyp_*.py contiennent** :

```python
METADATA = {
    'id': 'GAM-XXX',
    'name': 'Nom descriptif',
    'family': 'markovian' | 'non_markovian' | 'stochastic' | 'structural',
    'form': 'Équation LaTeX ou pseudo-code',
    'parameters': {
        'param_name': {
            'type': 'float' | 'int',
            'range': 'Description range',
            'nominal': valeur_défaut,
            'description': 'Description rôle'
        },
        # ... autres paramètres
    },
    'd_applicability': ['SYM', 'ASY', 'R3'],  # Liste types D applicables
    'expected_behavior': {
        'convergence': 'Description',
        'diversity': 'Description',
        'attractors': 'Description',
        'trivial': bool,
        'expected_failure': bool  # Optionnel (ex: GAM-003)
    },
    'notes': [
        'Note 1',
        'Note 2',
        # ...
    ]
}
```

### 11.2 Usage METADATA

**Consommateurs** :
- `batch_runner.py` : Validation applicabilité
- `gamma_profiling.py` : Enrichissement profils
- Documentation auto-générée (R1+)

---

## ANNEXE A : TEMPLATE NOUVEAU GAMMA

```python
"""
operators/gamma_hyp_NNN.py

HYP-GAM-NNN: [Titre descriptif]

FORME: [Équation]

ATTENDU: [Comportement attendu]
"""

import numpy as np
from typing import Callable


class MyGamma:
    """
    Γ [description courte].
    
    Mécanisme:
    - [Point 1]
    - [Point 2]
    
    ATTENDU:
    - [Comportement 1]
    - [Comportement 2]
    """
    
    def __init__(self, param1: float = default1, param2: float = default2):
        """
        Args:
            param1: Description param1
            param2: Description param2
        """
        # Validations
        assert condition1, "Message erreur"
        assert condition2, "Message erreur"
        
        # Attributs
        self.param1 = param1
        self.param2 = param2
        
        # Mémoire (si non-markovien)
        self._previous_state = None
        
        # Générateur aléatoire (si stochastique)
        # self.rng = np.random.RandomState(seed) if seed is not None else np.random
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique transformation γ.
        
        Args:
            state: État courant
        
        Returns:
            État suivant
        
        Raises:
            ValueError: Si contraintes applicabilité non respectées
        """
        # Validations techniques (si nécessaire)
        if state.ndim != 2:
            raise ValueError("Applicable uniquement rang 2")
        
        # Logique transformation
        # ...
        
        return transformed_state
    
    def reset(self):
        """Réinitialise la mémoire (si applicable)."""
        self._previous_state = None
    
    def __repr__(self):
        return f"MyGamma(param1={self.param1}, param2={self.param2})"


def create_gamma_hyp_NNN(param1: float = default1, 
                         param2: float = default2) -> Callable[[np.ndarray], np.ndarray]:
    """Factory pour GAM-NNN."""
    return MyGamma(param1=param1, param2=param2)


# ============================================================================
# GRILLES DE PARAMÈTRES
# ============================================================================

PARAM_GRID_PHASE1 = {
    'nominal': {'param1': default1, 'param2': default2}
}

PARAM_GRID_PHASE2 = {
    'config1': {'param1': value1, 'param2': value2},
    'config2': {'param1': value3, 'param2': value4},
}


# ============================================================================
# MÉTADONNÉES
# ============================================================================

METADATA = {
    'id': 'GAM-NNN',
    'name': '[Nom descriptif]',
    'family': 'markovian',  # ou non_markovian, stochastic, structural
    'form': '[Équation LaTeX]',
    'parameters': {
        'param1': {
            'type': 'float',
            'range': '[range]',
            'nominal': default1,
            'description': '[Description]'
        },
        'param2': {
            'type': 'float',
            'range': '[range]',
            'nominal': default2,
            'description': '[Description]'
        }
    },
    'd_applicability': ['SYM', 'ASY', 'R3'],
    'expected_behavior': {
        'convergence': '[Description]',
        'diversity': '[Description]',
        'attractors': '[Description]',
        'trivial': bool
    },
    'notes': [
        '[Note 1]',
        '[Note 2]'
    ]
}
```

---

## ANNEXE B : HISTORIQUE MODIFICATIONS

| Date | Version | Changement |
|------|---------|------------|
| 2025-01-15 | 6.0.0 | Création catalogue initial (11 gammas documentés) |

---

## ANNEXE C : INDEX ALPHABÉTIQUE GAMMAS

| Gamma | ID | Famille | Fichier |
|-------|-----|---------|---------|
| DifferentialMemoryGamma | GAM-008 | non_markovian | gamma_hyp_008.py |
| ExponentialDecayGamma | GAM-004 | markovian | gamma_hyp_004.py |
| ExponentialGrowthGamma | GAM-003 | markovian | gamma_hyp_003.py |
| ForcedSymmetryGamma | GAM-012 | structural | gamma_hyp_012.py |
| HarmonicOscillatorGamma | GAM-005 | non_markovian | gamma_hyp_005.py |
| HebbianReinforcementGamma | GAM-013 | structural | gamma_hyp_013.py |
| MemorySaturationGamma | GAM-006 | non_markovian | gamma_hyp_006.py |
| MultiplicativeNoiseGamma | GAM-010 | stochastic | gamma_hyp_010.py |
| PureDiffusionGamma | GAM-002 | markovian | gamma_hyp_002.py |
| PureSaturationGamma | GAM-001 | markovian | gamma_hyp_001.py |
| SlidingAverageGamma | GAM-007 | non_markovian | gamma_hyp_007.py |
| StochasticSaturationGamma | GAM-009 | stochastic | gamma_hyp_009.py |

---

**FIN operators_catalog.md**