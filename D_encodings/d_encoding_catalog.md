# Catalogue Encodings Rang 2

## Famille SYM (Symétrique)

### SYM-001 - Identité
- **Fonction** : `create_identity(n_dof)`
- **Propriétés** : Symétrique, SPD, Sparse, Diag=1

### SYM-002 - Aléatoire Uniforme
- **Fonction** : `create_random_uniform(n_dof, seed)`
- **Propriétés** : Symétrique, Bornes [-1,1]

### SYM-003 - Aléatoire Gaussienne
- **Fonction** : `create_random_gaussian(n_dof, sigma, seed)`
- **Propriétés** : Symétrique, N(0,σ)
- **Paramètres** : `sigma=0.3` (défaut)

### SYM-004 - Corrélation SPD
- **Fonction** : `create_correlation_matrix(n_dof, seed)`
- **Propriétés** : Symétrique, SPD, Diag=1

### SYM-005 - Bande
- **Fonction** : `create_banded(n_dof, bandwidth, amplitude, seed)`
- **Propriétés** : Symétrique, Sparse, Diag=1
- **Paramètres** : `bandwidth=3`, `amplitude=0.5` (défaut)

### SYM-006 - Hiérarchique Blocs
- **Fonction** : `create_block_hierarchical(n_dof, n_blocks, intra_corr, inter_corr, seed)`
- **Propriétés** : Symétrique, Structure modulaire
- **Paramètres** : `n_blocks=10`, `intra_corr=0.7`, `inter_corr=0.1` (défaut)

## Famille ASY (Asymétrique)

### ASY-001 - Aléatoire Asymétrique
- **Fonction** : `create_random_asymmetric(n_dof, seed)`
- **Propriétés** : Asymétrique, Bornes [-1,1]

### ASY-002 - Triangulaire Inférieure
- **Fonction** : `create_lower_triangular(n_dof, seed)`
- **Propriétés** : Asymétrique, Sparse, Triangulaire

### ASY-003 - Antisymétrique
- **Fonction** : `create_antisymmetric(n_dof, seed)`
- **Propriétés** : A = -A^T, Diag=0

### ASY-004 - Gradient Directionnel
- **Fonction** : `create_directional_gradient(n_dof, gradient, noise_amplitude, seed)`
- **Propriétés** : Asymétrique, Gradient linéaire
- **Paramètres** : `gradient=0.1`, `noise_amplitude=0.2` (défaut)

## Encodings Bonus (Non-R0)

### Circulant Asymétrique
- **Fonction** : `create_circulant_asymmetric(n_dof, seed)`
- **Propriétés** : Asymétrique, Structure circulante

### Sparse Asymétrique
- **Fonction** : `create_sparse_asymmetric(n_dof, density, seed)`
- **Propriétés** : Asymétrique, Sparse
- **Paramètres** : `density=0.2` (défaut)

### Uniforme (Legacy)
- **Fonction** : `create_uniform(n_dof, correlation)`
- **Propriétés** : Symétrique, Corrélation uniforme
- **Paramètres** : `correlation=0.5` (défaut)

### Random (Legacy)
- **Fonction** : `create_random(n_dof, mean, std, seed)`
- **Propriétés** : Symétrique, Gaussienne, Bornes [-1,1]
- **Paramètres** : `mean=0.0`, `std=0.3` (défaut)

## Table Récapitulative

| ID | Symétrie | SPD | Sparse | Diag=1 | Bornes |
|----|----------|-----|--------|--------|--------|
| SYM-001 | ✅ | ✅ | ✅ | ✅ | N/A |
| SYM-002 | ✅ | ❌ | ❌ | ❌ | [-1,1] |
| SYM-003 | ✅ | ❌ | ❌ | ❌ | Non borné |
| SYM-004 | ✅ | ✅ | ❌ | ✅ | [-1,1] |
| SYM-005 | ✅ | ❌ | ✅ | ✅ | [-1,1] |
| SYM-006 | ✅ | ❌ | ❌ | ✅ | [-1,1] |
| ASY-001 | ❌ | ❌ | ❌ | ❌ | [-1,1] |
| ASY-002 | ❌ | ❌ | ✅ | ❌ | [-1,1] |
| ASY-003 | ❌ | ❌ | ❌ | ✅ (=0) | [-1,1] |
| ASY-004 | ❌ | ❌ | ❌ | ❌ | Non borné |

## Utilisation
```python
from D_encodings import get_generator_by_id

# Par ID
gen = get_generator_by_id("SYM-002")
D_base = gen(n_dof=50, seed=42)

# Import direct
from D_encodings.rank2_symmetric import create_random_uniform
D_base = create_random_uniform(n_dof=50, seed=42)
```

**Source** : `D_encodings/rank2_symmetric.py`, `D_encodings/rank2_asymmetric.py`

# Catalogue Encodings Rang 3

## Encodings Principaux

### R3-001 - Aléatoire Uniforme
- **Fonction** : `create_random_rank3(n_dof, seed)`
- **Propriétés** : Aucune symétrie, Bornes [-1,1]

### R3-002 - Symétrie Partielle
- **Fonction** : `create_partial_symmetric_rank3(n_dof, seed)`
- **Propriétés** : T[i,j,k] = T[i,k,j], Bornes [-1,1]

### R3-003 - Couplages Locaux
- **Fonction** : `create_local_coupling_rank3(n_dof, radius, seed)`
- **Propriétés** : Sparse, Localité géométrique
- **Paramètres** : `radius=2` (défaut)

## Encodings Bonus

### Symétrique Complet
- **Fonction** : `create_fully_symmetric_rank3(n_dof, seed)`
- **Propriétés** : Invariant permutations indices

### Diagonal
- **Fonction** : `create_diagonal_rank3(n_dof, seed)`
- **Propriétés** : T[i,j,k] ≠ 0 ssi i=j=k

### Séparable
- **Fonction** : `create_separable_rank3(n_dof, seed)`
- **Propriétés** : T[i,j,k] = u[i]·v[j]·w[k], Rang tensoriel = 1

### Blocs
- **Fonction** : `create_block_rank3(n_dof, n_blocks, seed)`
- **Propriétés** : Structure hiérarchique
- **Paramètres** : `n_blocks=4` (défaut)

## Table Récapitulative

| ID | Symétrie | Sparse | Bornes | Coût Mémoire (N=20) |
|----|----------|--------|--------|---------------------|
| R3-001 | ❌ | ❌ | [-1,1] | 8000 éléments |
| R3-002 | Partielle | ❌ | [-1,1] | 8000 éléments |
| R3-003 | ❌ | ✅ | [-1,1] | ~800 éléments (radius=2) |

## Utilisation
```python
from D_encodings import get_generator_by_id

# Par ID
gen = get_generator_by_id("R3-001")
D_base = gen(n_dof=20, seed=42)

# Import direct
from D_encodings.rank3_correlations import create_random_rank3
D_base = create_random_rank3(n_dof=20, seed=42)
```

**Note Mémoire** : N=20 recommandé (8KB), N=30 acceptable (216KB), N>50 déconseillé (>1MB)

**Source** : `D_encodings/rank3_correlations.py`