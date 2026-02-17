# ATOMICS CATALOG

> Référence normative unique — gammas, encodings, modifiers  
> Version 1.0 — Nouveau pipeline PRC (divergence D4 vs legacy)  
> Date : 2026-02-17

---

## 1. PRINCIPES

### 1.1 Pourquoi ce format

Le format legacy mélangeait deux responsabilités dans chaque fichier atomic :

- **La mécanique pure** (algo, paramètres, comportement) → appartient au fichier Python
- **La configuration expérimentale** (grilles paramètres, phase, applicabilité déclarée) → appartient au YAML

Ce format sépare proprement ces deux responsabilités.

### 1.2 Ce qui disparaît du legacy

| Élément legacy | Remplacé par |
|----------------|-------------|
| `PHASE = "R0"` | Implicite — présence du fichier suffit |
| `METADATA` dict complet | `ID = "XXX-NNN"` uniquement |
| `PARAM_GRID_PHASE1/2` | YAML de configuration |
| `d_applicability` | Catch `ValueError` runtime (principe layers) |
| `create_gamma_hyp_NNN()` | `create()` uniforme |
| `CriticalDiscoveryError` sur PHASE | `CriticalDiscoveryError` sur ID et create/apply |

### 1.3 Règles communes aux 3 types

- **Un fichier = un mécanisme** (règle G1 legacy conservée)
- **Dépréciation explicite** : `_deprecated_` dans le nom de fichier (règle G3 legacy conservée)
- **Dépendances** : NumPy uniquement — zéro import interne PRC
- **Défauts dans la signature** : comportement sans YAML toujours défini
- **Applicabilité par runtime** : `ValueError` si incompatible, loggé et skipé par le runner

---

## 2. FORMAT FICHIER

### 2.1 Gammas (`atomics/operators/`)

#### Convention nommage
```
gamma_hyp_NNN.py     # NNN = numéro séquentiel (001, 002, ...)
```

#### Structure obligatoire

```python
"""
operators/gamma_hyp_NNN.py

Description courte du mécanisme.
Forme : T_{n+1} = f(T_n)
"""

import numpy as np

# ── Identifiant discovery (OBLIGATOIRE) ──────────────────────────────────────
ID = "GAM-NNN"

# ── Classe gamma ─────────────────────────────────────────────────────────────
class SomeGamma:
    """
    Description mécanisme.

    Forme : T_{n+1} = ...
    Famille : markovian | non_markovian | stochastic | structural
    """

    def __init__(self, param1: float = default1, param2: float = default2):
        self.param1 = param1
        self.param2 = param2

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Applique transformation.

        Raises:
            ValueError : Si rang ou dimensions incompatibles.
        """
        # Validation rang si nécessaire
        # if state.ndim != 2:
        #     raise ValueError(f"GAM-NNN requiert rang 2, reçu {state.ndim}")

        return ...  # transformation

    def __repr__(self):
        return f"SomeGamma(param1={self.param1})"


# ── Factory (OBLIGATOIRE) ─────────────────────────────────────────────────────
def create(param1: float = default1, param2: float = default2) -> SomeGamma:
    """
    Factory — instancie gamma avec paramètres.

    Args:
        param1 : Description (défaut = default1)
        param2 : Description (défaut = default2)

    Returns:
        Instance SomeGamma callable.
    """
    return SomeGamma(param1=param1, param2=param2)
```

#### Exemple minimal — GAM-001

```python
"""
operators/gamma_hyp_001.py

Saturation pure pointwise.
Forme : T_{n+1}[i,j] = tanh(β · T_n[i,j])
"""

import numpy as np

ID = "GAM-001"

class PureSaturationGamma:
    """
    Saturation pointwise via tanh.

    Forme : T_{n+1} = tanh(β · T_n)
    Famille : markovian
    """

    def __init__(self, beta: float = 2.0):
        self.beta = beta

    def __call__(self, state: np.ndarray) -> np.ndarray:
        return np.tanh(self.beta * state)

    def __repr__(self):
        return f"PureSaturationGamma(beta={self.beta})"


def create(beta: float = 2.0) -> PureSaturationGamma:
    """
    Args:
        beta : Force de saturation (défaut = 2.0)
    """
    return PureSaturationGamma(beta=beta)
```

#### Gammas non-markoviens — gestion mémoire

Les gammas non-markoviens stockent un état interne. La méthode `reset()` **doit être présente** mais le runner n'a pas besoin de l'appeler — chaque run instancie un nouveau gamma via `create()`, donc la mémoire repart vierge automatiquement.

`reset()` reste utile pour usage manuel hors runner.

```python
class NonMarkovianGamma:
    def __init__(self, beta: float = 1.0, alpha: float = 0.3):
        self.beta = beta
        self.alpha = alpha
        self._prev: np.ndarray | None = None

    def __call__(self, state: np.ndarray) -> np.ndarray:
        if self._prev is None:
            # Première itération : fallback markovien
            result = np.tanh(self.beta * state)
        else:
            result = np.tanh(self.beta * state + self.alpha * (state - self._prev))
        self._prev = state.copy()
        return result

    def reset(self):
        """Réinitialise mémoire — utile hors runner."""
        self._prev = None


def create(beta: float = 1.0, alpha: float = 0.3) -> NonMarkovianGamma:
    return NonMarkovianGamma(beta=beta, alpha=alpha)
```

#### Gammas stochastiques — gestion seed

Le seed est géré par `prepare_state()` en amont (centralisation). Les gammas stochastiques utilisent `np.random` global — déjà seeded par le core.

```python
class StochasticGamma:
    def __init__(self, beta: float = 1.0, sigma: float = 0.01):
        self.beta = beta
        self.sigma = sigma

    def __call__(self, state: np.ndarray) -> np.ndarray:
        noise = np.random.randn(*state.shape) * self.sigma
        return np.tanh(self.beta * state) + noise


def create(beta: float = 1.0, sigma: float = 0.01) -> StochasticGamma:
    return StochasticGamma(beta=beta, sigma=sigma)
```

---

### 2.2 Encodings (`atomics/D_encodings/`)

#### Convention nommage
```
sym_NNN_descriptif.py    # Rang 2 symétrique
asy_NNN_descriptif.py    # Rang 2 asymétrique
r3_NNN_descriptif.py     # Rang 3
```

#### Structure obligatoire

```python
"""
D_encodings/xxx_NNN_descriptif.py

Description courte.
Forme : ...
"""

import numpy as np

# ── Identifiant discovery (OBLIGATOIRE) ──────────────────────────────────────
ID = "XXX-NNN"

# ── Factory (OBLIGATOIRE) ─────────────────────────────────────────────────────
def create(n_dof: int, param1: float = default1, **kwargs) -> np.ndarray:
    """
    Crée tenseur D^(base).

    Args:
        n_dof  : Dimension du tenseur
        param1 : Description (défaut = default1)

    Returns:
        np.ndarray de shape appropriée
    """
    return ...
```

#### Exemple minimal — ASY-004

```python
"""
D_encodings/asy_004_directional_gradient.py

Matrice avec gradient directionnel.
Forme : A_ij = gradient·(i-j) + U[-noise, +noise]
"""

import numpy as np

ID = "ASY-004"

def create(n_dof: int, gradient: float = 0.1,
           noise_amplitude: float = 0.2) -> np.ndarray:
    """
    Args:
        n_dof            : Dimension
        gradient         : Pente du gradient (défaut = 0.1)
        noise_amplitude  : Amplitude bruit additif (défaut = 0.2)
    """
    i_idx, j_idx = np.meshgrid(range(n_dof), range(n_dof), indexing='ij')
    A = gradient * (i_idx - j_idx)
    A += np.random.uniform(-noise_amplitude, noise_amplitude, (n_dof, n_dof))
    return A
```

---

### 2.3 Modifiers (`atomics/modifiers/`)

#### Convention nommage
```
m0_baseline.py
m1_descriptif.py
mN_descriptif.py
```

#### Structure obligatoire

```python
"""
modifiers/mN_descriptif.py

Description courte.
Transformation : D' = ...
"""

import numpy as np

# ── Identifiant discovery (OBLIGATOIRE) ──────────────────────────────────────
ID = "MN"

# ── Fonction principale (OBLIGATOIRE) ─────────────────────────────────────────
def apply(state: np.ndarray, param1: float = default1) -> np.ndarray:
    """
    Applique transformation.

    Args:
        state  : Tenseur d'entrée
        param1 : Description (défaut = default1)

    Returns:
        np.ndarray même shape que state
    """
    return ...
```

#### Exemple minimal — M2

```python
"""
modifiers/m2_uniform_noise.py

Bruit uniforme additif.
Transformation : D' = D + U[-amplitude, +amplitude]
"""

import numpy as np

ID = "M2"

def apply(state: np.ndarray, amplitude: float = 0.1) -> np.ndarray:
    """
    Args:
        state     : Tenseur d'entrée
        amplitude : Amplitude bruit (défaut = 0.1)
    """
    noise = np.random.uniform(-amplitude, amplitude, size=state.shape)
    return state + noise
```

---

## 3. DISCOVERY

### 3.1 Algorithme commun

```
Pour chaque type (gamma, encoding, modifier) :

1. Glob fichiers selon pattern
2. Skip si '_deprecated_' dans nom fichier
3. Import module (importlib)
4. Lire ID  → CriticalDiscoveryError si absent
5. Trouver create() (gammas/encodings) ou apply() (modifiers)
   → CriticalDiscoveryError si absent
6. Enregistrer {id → callable} dans registre discovery
```

### 3.2 Patterns glob

| Type | Pattern | Exemple |
|------|---------|---------|
| Gammas | `gamma_hyp_*.py` | `gamma_hyp_001.py` |
| Encodings SYM | `sym_*.py` | `sym_002_random_uniform.py` |
| Encodings ASY | `asy_*.py` | `asy_004_directional_gradient.py` |
| Encodings R3 | `r3_*.py` | `r3_001_random_uniform.py` |
| Modifiers | `m*.py` | `m1_gaussian_noise.py` |

### 3.3 CriticalDiscoveryError

Levée (pas warn) si :
- `ID` absent du module
- `create()` absent (gammas/encodings)
- `apply()` absent (modifiers)

### 3.4 Dépréciation

Jamais supprimer un fichier. Renommer avec `_deprecated_` :
```
gamma_hyp_011_deprecated_exp_growth.py   # skipé par discovery
```

---

## 4. CONFIGURATION YAML

### 4.1 Principe défauts / override

Si un atomic n'est pas configuré dans le YAML avec des `params`, les **défauts de la signature `create()`** s'appliquent — une seule composition est générée.

Si `params` est spécifié dans le YAML, le runner génère **une composition par combinaison**.

### 4.2 Syntaxe YAML

```yaml
# configs/phases/r0.yaml

axes:
  gamma:
    - id: GAM-001                    # défauts → beta=2.0, 1 composition
    - id: GAM-001
      params:
        beta: [0.5, 1.0, 2.0, 5.0]  # liste explicite → 4 compositions
    - id: GAM-009
      params:
        beta: [1.0, 2.0]
        sigma: [0.01, 0.05]          # produit cartésien → 4 compositions

  encoding:
    - id: SYM-002                    # défaut
    - id: ASY-004
      params:
        gradient: [0.1, 0.5]         # 2 compositions

  modifier:
    - id: M0                         # baseline, pas de params
    - id: M1
      params:
        sigma: [0.01, 0.05]

  n_dof: [10, 20, 50]                # axe global
  seed: [42, 123, 456]               # axe global
  max_iterations: 500                # scalaire → même valeur pour tous
```

### 4.3 Résolution par le runner

```
Pour chaque combinaison (gamma_config, encoding_config, modifier_config, n_dof, seed) :
    gamma    = gamma_create(**gamma_params ou défauts)
    state    = prepare_state(encoding_create, encoding_params, modifiers, seed)
    history  = run_kernel(state, gamma, max_iterations)
    features = extract(history)
    write(features → Parquet)
```

---

## 5. APPLICABILITÉ RUNTIME

### 5.1 Principe

Pas de pré-validation par métadonnées. Même principe que les layers en featuring :
- Le runner **essaie** la composition
- Si incompatible → `ValueError` levée naturellement par l'atomic
- Le runner **logge et skipe** — pas d'arrêt du batch

### 5.2 Où lever ValueError

Dans `__call__()` du gamma ou dans `create()` de l'encoding, si la contrainte est vérifiable :

```python
# Gamma rang 2 seulement
def __call__(self, state: np.ndarray) -> np.ndarray:
    if state.ndim != 2:
        raise ValueError(
            f"GAM-013 requiert rang 2, reçu rang {state.ndim}"
        )
    ...

# Gamma matrice carrée seulement
def __call__(self, state: np.ndarray) -> np.ndarray:
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError(
            f"GAM-013 requiert matrice carrée, reçu shape {state.shape}"
        )
    ...
```

### 5.3 Ce que le runner fait

```python
try:
    history = run_kernel(state, gamma, ...)
    features = extract(history)
    write(features)
except ValueError as e:
    log_skipped(composition_id, reason=str(e))
    continue
```

---

## 6. INVENTAIRE

### Gammas (operators/)

| ID | Fichier | Famille | Forme | Défauts |
|----|---------|---------|-------|---------|
| GAM-001 | gamma_hyp_001.py | markovian | tanh(β·T) | β=2.0 |
| GAM-002 | gamma_hyp_002.py | markovian | Diffusion Laplacienne | α=0.05 |
| GAM-003 | gamma_hyp_003.py | markovian | T·exp(γ) | γ=0.05 |
| GAM-004 | gamma_hyp_004.py | markovian | T·exp(-γ) | γ=0.05 |
| GAM-005 | gamma_hyp_005.py | non_markovian | Oscillateur harmonique | ω=π/4 |
| GAM-006 | gamma_hyp_006.py | non_markovian | tanh(β·T + α·ΔT) | β=1.0, α=0.3 |
| GAM-007 | gamma_hyp_007.py | non_markovian | Moyenne glissante 8-voisins | ε=0.1 |
| GAM-008 | gamma_hyp_008.py | non_markovian | Mémoire différentielle | γ=0.3, β=1.0 |
| GAM-009 | gamma_hyp_009.py | stochastic | tanh(β·T) + σ·ε | β=1.0, σ=0.01 |
| GAM-010 | gamma_hyp_010.py | stochastic | tanh(T·(1+σ·ε)) | σ=0.05 |
| GAM-012 | gamma_hyp_012.py | structural | Symétrie forcée | β=2.0 |
| GAM-013 | gamma_hyp_013.py | structural | Hebbien T+η·T@T | η=0.01 |

**Note GAM-002, GAM-007, GAM-012** : rang 2 uniquement → ValueError si rang 3  
**Note GAM-013** : matrice carrée uniquement → ValueError si non-carré  
**Note GAM-003** : conçu pour exploser (test robustesse)

---

### Encodings (D_encodings/)

#### Symétrique (SYM-*)

| ID | Fichier | Forme | Défauts |
|----|---------|-------|---------|
| SYM-001 | sym_001_identity.py | I_ij = δ_ij | — |
| SYM-002 | sym_002_random_uniform.py | (B+Bᵀ)/2, U[-1,1] | — |
| SYM-003 | sym_003_random_gaussian.py | (B+Bᵀ)/2, N(0,σ) | σ=0.3 |
| SYM-004 | sym_004_correlation_matrix.py | C·Cᵀ normalisé | — |
| SYM-005 | sym_005_banded.py | Bande sparse | bandwidth=3, amplitude=0.5 |
| SYM-006 | sym_006_block_hierarchical.py | Blocs intra/inter | n_blocks=10, intra=0.7, inter=0.1 |
| SYM-007 | sym_007_uniform_correlation.py | C_ij = ρ (i≠j), 1 (diag) | correlation=0.5 |
| SYM-008 | sym_008_random_clipped.py | N(mean,std) clippé [-1,1] | mean=0.0, std=0.3 |

#### Asymétrique (ASY-*)

| ID | Fichier | Forme | Défauts |
|----|---------|-------|---------|
| ASY-001 | asy_001_random_asymmetric.py | A_ij ~ U[-1,1] | — |
| ASY-002 | asy_002_lower_triangular.py | Triangulaire inf. | — |
| ASY-003 | asy_003_antisymmetric.py | A = -Aᵀ | — |
| ASY-004 | asy_004_directional_gradient.py | gradient·(i-j) + bruit | gradient=0.1, noise=0.2 |
| ASY-005 | asy_005_circulant.py | Circulant décalé | — |
| ASY-006 | asy_006_sparse.py | Sparse asymétrique | density=0.2 |

#### Rang 3 (R3-*)

| ID | Fichier | Forme | Défauts |
|----|---------|-------|---------|
| R3-001 | r3_001_random_uniform.py | T_ijk ~ U[-1,1] | — |
| R3-002 | r3_002_partial_symmetric.py | T_ijk = T_ikj | — |
| R3-003 | r3_003_local_coupling.py | Sparse local | radius=2 |
| R3-004 | r3_004_fully_symmetric.py | Symétrie totale 6 permut. | — |
| R3-005 | r3_005_diagonal.py | T_ijk ≠ 0 ssi i=j=k | — |
| R3-006 | r3_006_separable.py | u_i·v_j·w_k | — |
| R3-007 | r3_007_block_structure.py | Blocs 3D | n_blocks=4 |

---

### Modifiers (modifiers/)

| ID | Fichier | Transformation | Défauts |
|----|---------|---------------|---------|
| M0 | m0_baseline.py | D' = D (identité) | — |
| M1 | m1_gaussian_noise.py | D' = D + N(0,σ) | σ=0.05 |
| M2 | m2_uniform_noise.py | D' = D + U[-a,+a] | amplitude=0.1 |

---

## 7. CHECKLIST AJOUT NOUVEL ATOMIC

### Gamma
- [ ] Fichier `gamma_hyp_NNN.py` créé
- [ ] `ID = "GAM-NNN"` présent
- [ ] Classe avec `__init__`, `__call__`, `__repr__`
- [ ] `ValueError` si rang/dimension incompatible
- [ ] `reset()` présent si non-markovien
- [ ] `create(**params)` avec défauts explicites
- [ ] Défauts choisis = comportement nominal attendu
- [ ] Ajouté à l'inventaire section 6

### Encoding
- [ ] Fichier `{sym|asy|r3}_NNN_descriptif.py` créé
- [ ] `ID = "XXX-NNN"` présent
- [ ] `create(n_dof, **params)` avec défauts explicites
- [ ] Retourne `np.ndarray` shape correcte
- [ ] Ajouté à l'inventaire section 6

### Modifier
- [ ] Fichier `mN_descriptif.py` créé
- [ ] `ID = "MN"` présent
- [ ] `apply(state, **params)` avec défauts explicites
- [ ] Retourne `np.ndarray` même shape que input
- [ ] Ajouté à l'inventaire section 6

---

**FIN ATOMICS CATALOG v1.0**
