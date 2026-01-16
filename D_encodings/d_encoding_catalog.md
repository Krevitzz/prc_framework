# d_encoding_catalog.md

> Catalogue fonctionnel des encodages D^(base)  
> Responsabilité : Création tenseurs initiaux avec présuppositions explicites  
> Version : 6.0  
> Dernière mise à jour : 2025-01-15

---

## VUE D'ENSEMBLE

Le module `D_encodings/` crée les **tenseurs de base D^(base)** avec présuppositions explicites (rang, symétrie, bornes, structure).

**Modules** :
- `rank2_symmetric.py` : Matrices symétriques (6 encodages + utilitaires)
- `rank2_asymmetric.py` : Matrices asymétriques (4 encodages + bonus)
- `rank3_correlations.py` : Tenseurs rang 3 (3 encodages + bonus)

**Principe fondamental** :
- ✅ Validation dimensionnelle ICI (rang, shape, symétrie)
- ✅ Présuppositions sémantiques EXPLICITES (docstrings)
- ❌ Aucune validation dans core (séparation stricte)

---

## SECTION 1 : rank2_symmetric.py

### 1.1 Encodages catalogués (SYM-001 à SYM-006)

#### SYM-001 : create_identity()

**Signature** :
```python
def create_identity(n_dof: int) -> np.ndarray
```

**Présuppositions** :
- Rang 2 (matrice)
- Symétrie : C[i,j] = C[j,i]
- Diagonale : C[i,i] = 1
- Hors-diagonale : C[i,j] = 0 (i≠j)

**Propriétés** :
- Symétrique, définie positive, sparse
- Point fixe trivial pour nombreux γ
- Test stabilité minimale

**Paramètres** :
| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `n_dof` | `int` | - | Nombre de degrés de liberté |

**Retour** : `np.ndarray` shape (n_dof, n_dof)

**Cas d'usage** :
```python
D_base = create_identity(50)
# Matrice identité 50×50
```

---

#### SYM-002 : create_random_uniform()

**Signature** :
```python
def create_random_uniform(n_dof: int, seed: int = None) -> np.ndarray
```

**Forme mathématique** :
```
A = (B + B^T) / 2
B_ij ~ U[-1, 1]
```

**Présuppositions** :
- Rang 2, symétrique
- Bornes : [-1, 1]
- Distribution uniforme

**Propriétés** :
- Diversité maximale
- Générique (pas de structure particulière)

**Paramètres** :
| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `n_dof` | `int` | - | Nombre de degrés de liberté |
| `seed` | `int` | None | Graine aléatoire (reproductibilité) |

**Retour** : `np.ndarray` shape (n_dof, n_dof)

**Cas d'usage** :
```python
D_base = create_random_uniform(50, seed=42)
# Test diversité maximale, générique
```

---

#### SYM-003 : create_random_gaussian()

**Signature** :
```python
def create_random_gaussian(n_dof: int, sigma: float = 0.3, 
                          seed: int = None) -> np.ndarray
```

**Forme mathématique** :
```
A = (B + B^T) / 2
B_ij ~ N(0, σ=0.3)
```

**Présuppositions** :
- Rang 2, symétrique
- Distribution gaussienne
- Non bornée a priori (peut dépasser [-1, 1])

**Propriétés** :
- Test continuité
- Distribution normale (vs uniforme)

**Paramètres** :
| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `n_dof` | `int` | - | Nombre de degrés de liberté |
| `sigma` | `float` | 0.3 | Écart-type distribution gaussienne |
| `seed` | `int` | None | Graine aléatoire |

**Retour** : `np.ndarray` shape (n_dof, n_dof)

**Cas d'usage** :
```python
D_base = create_random_gaussian(50, sigma=0.5, seed=42)
# Test distribution gaussienne
```

---

#### SYM-004 : create_correlation_matrix()

**Signature** :
```python
def create_correlation_matrix(n_dof: int, seed: int = None) -> np.ndarray
```

**Forme mathématique** :
```
A = C·C^T normalisée
C_ij ~ N(0, 1)
Normalisation : D^(-1/2) · A · D^(-1/2) où D = diag(A)
```

**Présuppositions** :
- Rang 2, symétrique
- Définie positive (SPD)
- Diagonale : C[i,i] = 1
- Bornes : [-1, 1] (garanties par normalisation)

**Propriétés** :
- Matrice de corrélation valide
- SPD (valeurs propres > 0)
- Test positivité définie

**Paramètres** :
| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `n_dof` | `int` | - | Nombre de degrés de liberté |
| `seed` | `int` | None | Graine aléatoire |

**Retour** : `np.ndarray` shape (n_dof, n_dof)

**Cas d'usage** :
```python
D_base = create_correlation_matrix(50, seed=42)
# Test positivité définie, matrice de corrélation
```

---

#### SYM-005 : create_banded()

**Signature** :
```python
def create_banded(n_dof: int, bandwidth: int = 3, 
                  amplitude: float = 0.5, seed: int = None) -> np.ndarray
```

**Forme mathématique** :
```
A_ij ≠ 0 ssi |i-j| ≤ bandwidth
A_ij ~ U[-amplitude, amplitude] (i≠j)
A_ii = 1
```

**Présuppositions** :
- Rang 2, symétrique
- Structure bande (sparse)
- Diagonale : C[i,i] = 1
- Localité structurelle

**Propriétés** :
- Sparse (couplages locaux uniquement)
- Test localité

**Paramètres** :
| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `n_dof` | `int` | - | Nombre de degrés de liberté |
| `bandwidth` | `int` | 3 | Largeur de bande |
| `amplitude` | `float` | 0.5 | Amplitude valeurs hors-diagonale |
| `seed` | `int` | None | Graine aléatoire |

**Retour** : `np.ndarray` shape (n_dof, n_dof)

**Cas d'usage** :
```python
D_base = create_banded(50, bandwidth=5, amplitude=0.7, seed=42)
# Test localité structurelle
```

---

#### SYM-006 : create_block_hierarchical()

**Signature** :
```python
def create_block_hierarchical(n_dof: int, n_blocks: int = 10,
                              intra_corr: float = 0.7,
                              inter_corr: float = 0.1,
                              seed: int = None) -> np.ndarray
```

**Forme mathématique** :
```
Blocs : intra_corr (forte corrélation interne)
Inter : inter_corr (faible corrélation externe)
+ Bruit gaussien N(0, 0.05) symétrisé
Clip final dans [-1, 1], diagonale = 1
```

**Présuppositions** :
- Rang 2, symétrique
- Structure modulaire (blocs)
- n_dof doit être divisible par n_blocks
- Diagonale : C[i,i] = 1

**Propriétés** :
- Structure hiérarchique
- Test préservation modularité

**Paramètres** :
| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `n_dof` | `int` | - | Nombre de degrés de liberté (divisible par n_blocks) |
| `n_blocks` | `int` | 10 | Nombre de blocs |
| `intra_corr` | `float` | 0.7 | Corrélation intra-bloc |
| `inter_corr` | `float` | 0.1 | Corrélation inter-blocs |
| `seed` | `int` | None | Graine aléatoire |

**Retour** : `np.ndarray` shape (n_dof, n_dof)

**Validation** : Lève `AssertionError` si n_dof % n_blocks ≠ 0

**Cas d'usage** :
```python
D_base = create_block_hierarchical(100, n_blocks=10, seed=42)
# Test structure modulaire hiérarchique
```

---

### 1.2 Fonctions utilitaires (non cataloguées)

#### create_uniform()

**Signature** :
```python
def create_uniform(n_dof: int, correlation: float = 0.5) -> np.ndarray
```

**Forme** :
```
C[i,j] = correlation (i≠j)
C[i,i] = 1
```

**Usage** : Corrélations uniformes (tous DOF également corrélés)

**Validation** : Assert correlation ∈ [-1, 1]

---

#### create_random()

**Signature** :
```python
def create_random(n_dof: int, mean: float = 0.0, std: float = 0.3, 
                  seed: int = None) -> np.ndarray
```

**Forme** :
```
C ~ N(mean, std) symétrisé
Diagonale = 1
Clip dans [-1, 1]
```

**Usage** : Corrélations aléatoires gaussiennes (legacy, remplacé par SYM-003)

---

## SECTION 2 : rank2_asymmetric.py

### 2.1 Encodages catalogués (ASY-001 à ASY-004)

#### ASY-001 : create_random_asymmetric()

**Signature** :
```python
def create_random_asymmetric(n_dof: int, seed: int = None) -> np.ndarray
```

**Forme mathématique** :
```
A_ij ~ U[-1, 1] indépendants
```

**Présuppositions** :
- Rang 2, asymétrique (général)
- C[i,j] ≠ C[j,i]
- Bornes : [-1, 1]

**Propriétés** :
- Asymétrie générique
- Pas de contrainte diagonale

**Paramètres** :
| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `n_dof` | `int` | - | Nombre de degrés de liberté |
| `seed` | `int` | None | Graine aléatoire |

**Retour** : `np.ndarray` shape (n_dof, n_dof)

**Cas d'usage** :
```python
D_base = create_random_asymmetric(50, seed=42)
# Test asymétrie générique
```

---

#### ASY-002 : create_lower_triangular()

**Signature** :
```python
def create_lower_triangular(n_dof: int, seed: int = None) -> np.ndarray
```

**Forme mathématique** :
```
A_ij = U[-1, 1] si i > j
A_ij = 0 sinon
```

**Présuppositions** :
- Rang 2, asymétrique
- Triangulaire inférieure stricte
- Sparse (50% zéros)

**Propriétés** :
- Orientation directionnelle
- Test structure triangulaire

**Paramètres** :
| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `n_dof` | `int` | - | Nombre de degrés de liberté |
| `seed` | `int` | None | Graine aléatoire |

**Retour** : `np.ndarray` shape (n_dof, n_dof)

**Cas d'usage** :
```python
D_base = create_lower_triangular(50, seed=42)
# Test orientation directionnelle, structure triangulaire
```

---

#### ASY-003 : create_antisymmetric()

**Signature** :
```python
def create_antisymmetric(n_dof: int, seed: int = None) -> np.ndarray
```

**Forme mathématique** :
```
A = -A^T
B_ij ~ U[-1, 1] pour i > j
A = B - B^T
```

**Présuppositions** :
- Rang 2, antisymétrique (cas spécial asymétrique)
- A[i,j] = -A[j,i]
- Diagonale nulle : A[i,i] = 0

**Propriétés** :
- Conservation antisymétrie
- Structure algébrique particulière

**Paramètres** :
| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `n_dof` | `int` | - | Nombre de degrés de liberté |
| `seed` | `int` | None | Graine aléatoire |

**Retour** : `np.ndarray` shape (n_dof, n_dof)

**Cas d'usage** :
```python
D_base = create_antisymmetric(50, seed=42)
# Test conservation antisymétrie
```

---

#### ASY-004 : create_directional_gradient()

**Signature** :
```python
def create_directional_gradient(n_dof: int, gradient: float = 0.1,
                                noise_amplitude: float = 0.2,
                                seed: int = None) -> np.ndarray
```

**Forme mathématique** :
```
A_ij = gradient·(i - j) + U[-noise_amplitude, +noise_amplitude]
```

**Présuppositions** :
- Rang 2, asymétrique
- Gradient linéaire directionnel
- Brisure symétrie structurée

**Propriétés** :
- Gradient directionnel
- Asymétrie avec structure

**Paramètres** :
| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `n_dof` | `int` | - | Nombre de degrés de liberté |
| `gradient` | `float` | 0.1 | Pente du gradient |
| `noise_amplitude` | `float` | 0.2 | Amplitude bruit additif |
| `seed` | `int` | None | Graine aléatoire |

**Retour** : `np.ndarray` shape (n_dof, n_dof)

**Cas d'usage** :
```python
D_base = create_directional_gradient(50, gradient=0.2, seed=42)
# Test brisure symétrie avec structure
```

---

### 2.2 Fonctions bonus (non cataloguées)

#### create_circulant_asymmetric()

**Forme** : Matrice circulante (chaque ligne = décalage précédente)  
**Usage** : Structure périodique asymétrique

#### create_sparse_asymmetric()

**Forme** : Asymétrique sparse (densité paramétrable)  
**Usage** : Structures creuses asymétriques

---

## SECTION 3 : rank3_correlations.py

### 3.1 Encodages catalogués (R3-001 à R3-003)

#### R3-001 : create_random_rank3()

**Signature** :
```python
def create_random_rank3(n_dof: int, seed: int = None) -> np.ndarray
```

**Forme mathématique** :
```
T_ijk ~ U[-1, 1]
```

**Présuppositions** :
- Rang 3 (tenseur N×N×N)
- Aucune symétrie
- Bornes : [-1, 1]

**Propriétés** :
- Générique rang 3
- Pas de structure particulière

**Paramètres** :
| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `n_dof` | `int` | - | Dimension tenseur (N×N×N) |
| `seed` | `int` | None | Graine aléatoire |

**Retour** : `np.ndarray` shape (n_dof, n_dof, n_dof)

**Notes** :
- Mémoire : O(N³)
- N=20 → 8000 éléments
- N=30 → 27000 éléments

**Cas d'usage** :
```python
D_base = create_random_rank3(20, seed=42)
# Test générique rang 3 (prudence mémoire)
```

---

#### R3-002 : create_partial_symmetric_rank3()

**Signature** :
```python
def create_partial_symmetric_rank3(n_dof: int, seed: int = None) -> np.ndarray
```

**Forme mathématique** :
```
T_ijk = T_ikj
T_sym = (T + T^(0,2,1)) / 2
```

**Présuppositions** :
- Rang 3
- Symétrie partielle (indices j,k)
- Valeurs ~ U[-1, 1]

**Propriétés** :
- Symétrie sur 2 indices
- Test symétries partielles

**Paramètres** :
| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `n_dof` | `int` | - | Dimension tenseur |
| `seed` | `int` | None | Graine aléatoire |

**Retour** : `np.ndarray` shape (n_dof, n_dof, n_dof)

**Cas d'usage** :
```python
D_base = create_partial_symmetric_rank3(20, seed=42)
# Test symétries partielles
```

---

#### R3-003 : create_local_coupling_rank3()

**Signature** :
```python
def create_local_coupling_rank3(n_dof: int, radius: int = 2,
                               seed: int = None) -> np.ndarray
```

**Forme mathématique** :
```
T_ijk ≠ 0 ssi |i-j| + |j-k| + |k-i| ≤ 2·radius
T_ijk ~ U[-1, 1] (si non-nul)
```

**Présuppositions** :
- Rang 3
- Sparse (couplages locaux uniquement)
- Localité géométrique 3-corps

**Propriétés** :
- Sparse
- Test localité 3-corps

**Paramètres** :
| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `n_dof` | `int` | - | Dimension tenseur |
| `radius` | `int` | 2 | Rayon de localité (2·radius = distance max) |
| `seed` | `int` | None | Graine aléatoire |

**Retour** : `np.ndarray` shape (n_dof, n_dof, n_dof)

**Cas d'usage** :
```python
D_base = create_local_coupling_rank3(20, radius=3, seed=42)
# Test localité 3-corps
```

---

### 3.2 Fonctions bonus (non cataloguées)

#### create_fully_symmetric_rank3()

**Forme** : T invariant par toute permutation (i,j,k)  
**Usage** : Symétrie complète (6 permutations)

#### create_diagonal_rank3()

**Forme** : T_ijk ≠ 0 ssi i=j=k  
**Usage** : Structure diagonale rang 3 (très sparse)

#### create_separable_rank3()

**Forme** : T_ijk = u_i · v_j · w_k (produit externe)  
**Usage** : Structure factorisée (rang tensoriel = 1)

#### create_block_rank3()

**Forme** : Structure par blocs 3D  
**Usage** : Modularité rang 3

---

## SECTION 4 : GRAPHE DE DÉPENDANCES

### 4.1 Relations inter-modules

```
rank2_symmetric.py
    ├─ Appelé par : batch_runner.py (via prepare_state)
    ├─ Reçoit : n_dof, paramètres encodage, seed
    └─ Retourne : np.ndarray (n_dof, n_dof)

rank2_asymmetric.py
    ├─ Appelé par : batch_runner.py (via prepare_state)
    ├─ Reçoit : n_dof, paramètres encodage, seed
    └─ Retourne : np.ndarray (n_dof, n_dof)

rank3_correlations.py
    ├─ Appelé par : batch_runner.py (via prepare_state)
    ├─ Reçoit : n_dof, paramètres encodage, seed
    └─ Retourne : np.ndarray (n_dof, n_dof, n_dof)
```

### 4.2 Flux typique création D^(base)

```
1. batch_runner.py lit d_encoding_id (ex: "SYM-001")
   ↓
2. Mapping vers fonction :
   "SYM-001" → rank2_symmetric.create_identity
   "ASY-003" → rank2_asymmetric.create_antisymmetric
   "R3-001" → rank3_correlations.create_random_rank3
   ↓
3. Appel fonction avec paramètres (n_dof, seed, ...)
   ↓
4. Retour D^(base) : np.ndarray
   ↓
5. Passage à prepare_state(D^(base), modifiers)
```

---

## SECTION 5 : MAPPING IDs ↔ FONCTIONS

### 5.1 Table complète (catalogués uniquement)

| ID | Module | Fonction | Rang | Symétrie |
|----|--------|----------|------|----------|
| SYM-001 | rank2_symmetric | create_identity | 2 | Symétrique |
| SYM-002 | rank2_symmetric | create_random_uniform | 2 | Symétrique |
| SYM-003 | rank2_symmetric | create_random_gaussian | 2 | Symétrique |
| SYM-004 | rank2_symmetric | create_correlation_matrix | 2 | Symétrique |
| SYM-005 | rank2_symmetric | create_banded | 2 | Symétrique |
| SYM-006 | rank2_symmetric | create_block_hierarchical | 2 | Symétrique |
| ASY-001 | rank2_asymmetric | create_random_asymmetric | 2 | Asymétrique |
| ASY-002 | rank2_asymmetric | create_lower_triangular | 2 | Asymétrique |
| ASY-003 | rank2_asymmetric | create_antisymmetric | 2 | Antisymétrique |
| ASY-004 | rank2_asymmetric | create_directional_gradient | 2 | Asymétrique |
| R3-001 | rank3_correlations | create_random_rank3 | 3 | Aucune |
| R3-002 | rank3_correlations | create_partial_symmetric_rank3 | 3 | Partielle |
| R3-003 | rank3_correlations | create_local_coupling_rank3 | 3 | Aucune |

### 5.2 Implémentation mapping (suggestion)

```python
# D_encodings/__init__.py (à créer si absent)
from .rank2_symmetric import (
    create_identity,
    create_random_uniform,
    create_random_gaussian,
    create_correlation_matrix,
    create_banded,
    create_block_hierarchical
)
from .rank2_asymmetric import (
    create_random_asymmetric,
    create_lower_triangular,
    create_antisymmetric,
    create_directional_gradient
)
from .rank3_correlations import (
    create_random_rank3,
    create_partial_symmetric_rank3,
    create_local_coupling_rank3
)

ENCODING_REGISTRY = {
    'SYM-001': create_identity,
    'SYM-002': create_random_uniform,
    'SYM-003': create_random_gaussian,
    'SYM-004': create_correlation_matrix,
    'SYM-005': create_banded,
    'SYM-006': create_block_hierarchical,
    'ASY-001': create_random_asymmetric,
    'ASY-002': create_lower_triangular,
    'ASY-003': create_antisymmetric,
    'ASY-004': create_directional_gradient,
    'R3-001': create_random_rank3,
    'R3-002': create_partial_symmetric_rank3,
    'R3-003': create_local_coupling_rank3,
}

def get_encoding(encoding_id: str):
    """Retourne fonction d'encodage depuis ID."""
    if encoding_id not in ENCODING_REGISTRY:
        raise ValueError(f"Unknown encoding_id: {encoding_id}")
    return ENCODING_REGISTRY[encoding_id]
```

---

## SECTION 6 : INVARIANTS CRITIQUES

### 6.1 Règles validation (dans D_encodings, PAS dans core)

**V1** : Validation dimensionnelle autorisée
```python
# OK dans D_encodings
assert n_dof % n_blocks == 0, "n_dof doit être divisible par n_blocks"
```

**V2** : Validation sémantique autorisée
```python
# OK dans D_encodings
assert -1.0 <= correlation <= 1.0, "correlation doit être dans [-1, 1]"
```

**V3** : Présuppositions EXPLICITES (docstrings)
```python
"""
PRÉSUPPOSITIONS EXPLICITES:
- Rang 2 (matrice)
- Symétrie C[i,j] = C[j,i]
- Bornes [-1, 1]
"""
```

**V4** : Retour brut (np.ndarray uniquement)
```python
# OK
return A  # np.ndarray

# INTERDIT
return {'data': A, 'metadata': {...}}  # ❌
```

### 6.2 Séparation stricte responsabilités

| Responsabilité | Où | Pas ailleurs |
|----------------|-----|--------------|
| Création D^(base) | D_encodings/ | ✅ |
| Validation dimensionnelle | D_encodings/ | ✅ |
| Présuppositions explicites | D_encodings/ (docstrings) | ✅ |
| Composition D | core/state_preparation.py | ❌ |
| Itération γ | core/kernel.py | ❌ |
| Tests sémantiques | tests/ | ❌ |

---

## SECTION 7 : EXTENSIONS FUTURES

### 7.1 Ajout nouveau encodage (checklist)

Avant d'ajouter un encodage :

- [ ] ID unique (format : CAT-NNN)
- [ ] Rang clair (2 ou 3)
- [ ] Présuppositions EXPLICITES (docstrings)
- [ ] Paramètres avec valeurs par défaut raisonnables
- [ ] Seed aléatoire optionnel (reproductibilité)
- [ ] Retour np.ndarray brut (pas de dict)
- [ ] Validation dimensionnelle (si pertinent)
- [ ] Ajouté à ENCODING_REGISTRY
- [ ] Documenté dans ce catalogue
- [ ] Cas d'usage identifié

### 7.2 Extensions INTERDITES

❌ **Ajout validation dans core** :
```python
# INTERDIT (validation reste dans D_encodings)
def prepare_state(base, modifiers):
    if not is_symmetric(base):  # ❌
        raise ValueError(...)
```

❌ **Ajout métadonnées retour** :
```python
# INTERDIT
def create_identity(n_dof):
    return {  # ❌
        'data': np.eye(n_dof),
        'properties': {'symmetric': True}
    }
```

❌ **Branchement dans core basé sur encodage** :
```python
# INTERDIT
def run_kernel(state, gamma):
    if encoding_id == "SYM-001":  # ❌
        # Fast path spécifique
```

### 7.3 Extensions AUTORISÉES

✅ **Ajout paramètres configurables** :
```python
# OK
def create_custom(n_dof: int, param1: float = 0.5, 
                  param2: int = 10, seed: int = None):
    ...
```

✅ **Ajout encodages bonus (non catalogués)** :
```python
# OK (si documenté comme "bonus" dans docstring)
def create_experimental_structure(n_dof, ...):
    """
    Structure expérimentale (bonus, non catalogué).
    ...
    """
```

---

## SECTION 8 : NOTES TECHNIQUES

### 8.1 Gestion mémoire rang 3

**Complexité** : O(N³)

**Tailles typiques** :
| n_dof | Éléments | Mémoire (float64) |
|-------|----------|-------------------|
| 10 | 1,000 | ~8 KB |
| 20 | 8,000 | ~64 KB |
| 30 | 27,000 | ~216 KB |
| 50 | 125,000 | ~1 MB |
| 100 | 1,000,000 | ~8 MB |

**Recommandations** :
- Rang 3 : n_dof ≤ 30 (usage courant)
- Rang 3 sparse : n_dof ≤ 50 (si applicable)
- Éviter n_dof > 50 sans justification

### 8.2 Reproductibilité (seeds)

**Convention** :
- Toutes fonctions acceptent `seed: int = None`
- Si `None` : comportement non déterministe
- Si `int` : reproductibilité garantie

**Usage** :
```python
# Reproductible
D1 = create_random_uniform(50, seed=42)
D2 = create_random_uniform(50, seed=42)
assert np.allclose(D1, D2)  # True

# Non déterministe
D3 = create_random_uniform(50)
D4 = create_random_uniform(50)
assert np.allclose(D3, D4)  # False (probabilité ~0)
```

**Note** : `np.random.seed()` est global → préférer `np.random.Generator` (R1+)

### 8.3 Validation entrées

**Principes** :
- Validation technique autorisée (dimensions, types)
- Validation sémantique autorisée (bornes, contraintes)
- Pas de validation coûteuse (calcul valeurs propres, etc.)

**Exemples autorisés** :
```python
# OK - validation technique
assert isinstance(n_dof, int), "n_dof doit être int"
assert n_dof > 0, "n_dof doit être positif"

# OK - validation contraintes
assert -1.0 <= correlation <= 1.0, "correlation dans [-1,1]"
assert n_dof % n_blocks == 0, "n_dof divisible par n_blocks"

# INTERDIT - validation coûteuse
eigvals = np.linalg.eigvals(A)  # ❌
assert np.all(eigvals > 0), "Matrice non SPD"  # ❌
```

### 8.4 Conventions nommage

**Paramètres standards** :
- `n_dof` : nombre degrés de liberté (dimension)
- `seed` : graine aléatoire
- `sigma` : écart-type gaussien
- `amplitude` : amplitude valeurs
- `correlation` : coefficient corrélation

**Variables internes** :
- `A`, `B`, `C` : matrices
- `T` : tenseur rang 3
- `i`, `j`, `k` : indices
- `u`, `v`, `w` : vecteurs

### 8.5 Types retour

**Strictement** :
- ✅ `np.ndarray` uniquement
- ❌ Pas de `dict`, `tuple`, `class`

**Shapes attendus** :
- Rang 2 : `(n_dof, n_dof)`
- Rang 3 : `(n_dof, n_dof, n_dof)`

**Dtype** :
- Par défaut : `float64` (numpy default)
- Pas de conversion explicite nécessaire

---

## SECTION 9 : TESTS ASSOCIÉS

### 9.1 Tests unitaires encodages

**Emplacement** : `tests/test_d_encodings.py` (si existe)

**Scénarios minimaux par encodage** :
- Shape correct : `assert D.shape == (n_dof, n_dof)`
- Symétrie (si applicable) : `assert np.allclose(D, D.T)`
- Antisymétrie (si applicable) : `assert np.allclose(D, -D.T)`
- Diagonale (si spécifié) : `assert np.allclose(np.diag(D), 1.0)`
- Bornes (si spécifié) : `assert np.all(D >= -1) and np.all(D <= 1)`
- Reproductibilité : deux appels même seed → tenseurs identiques

**Tests applicabilité** :
- Vérifiés dans `tests/utilities/UTIL/applicability.py`
- Basés sur `APPLICABILITY_SPEC` modules tests

### 9.2 Tests intégration

**Pipeline complet** :
```python
# 1. Création D^(base)
D_base = create_correlation_matrix(50, seed=42)

# 2. Application modifiers
D_final = prepare_state(D_base, [add_noise(sigma=0.05)])

# 3. Exécution kernel
for i, state in run_kernel(D_final, gamma, max_iterations=1000):
    pass

# 4. Tests observations
results = test_engine.run_test(history, test_module)
```

---

## SECTION 10 : CHECKLIST AJOUT ENCODAGE

Avant d'ajouter un nouveau `create_xxx()` :

### 10.1 Identification

- [ ] ID unique choisi (CAT-NNN)
- [ ] Catégorie claire (SYM, ASY, R3)
- [ ] Nom descriptif fonction (`create_xxx`)
- [ ] Rang défini (2 ou 3)

### 10.2 Implémentation

- [ ] Signature avec paramètres défaut raisonnables
- [ ] Paramètre `seed: int = None` présent
- [ ] Validation entrées (si pertinent)
- [ ] Forme mathématique claire (commentaires)
- [ ] Retour `np.ndarray` brut (pas dict/tuple)
- [ ] Shape correct : (n_dof, n_dof) ou (n_dof, n_dof, n_dof)

### 10.3 Documentation

- [ ] Docstring complète :
  - Description une ligne
  - Section PRÉSUPPOSITIONS EXPLICITES
  - Section FORME (math)
  - Section USAGE
  - Section PROPRIÉTÉS
  - Args avec types et descriptions
  - Returns avec shape
- [ ] Commentaires code si logique complexe

### 10.4 Intégration

- [ ] Ajouté à `ENCODING_REGISTRY` (si catalogué)
- [ ] Documenté dans ce catalogue (Section 1/2/3)
- [ ] Mapping ID ↔ fonction (Section 5)
- [ ] Cas d'usage identifié
- [ ] Tests écrits (optionnel R0, recommandé R1+)

### 10.5 Validation

- [ ] Exécuté manuellement avec plusieurs n_dof
- [ ] Vérifié shape retour
- [ ] Vérifié propriétés attendues (symétrie, bornes, etc.)
- [ ] Testé reproductibilité (même seed → même résultat)
- [ ] Testé dans pipeline complet (prepare_state → kernel)

---

## ANNEXE A : FORMULES MATHÉMATIQUES

### A.1 Symétrisation matrice

```python
# Méthode standard
A_sym = (A + A.T) / 2

# Propriété : A_sym[i,j] = A_sym[j,i]
```

### A.2 Antisymétrisation matrice

```python
# Méthode standard
A_anti = (A - A.T) / 2

# Propriétés :
# - A_anti[i,j] = -A_anti[j,i]
# - A_anti[i,i] = 0 (diagonale nulle)
```

### A.3 Normalisation corrélation (SPD)

```python
# Garantir matrice corrélation valide
A = C @ C.T  # Produit → SPD
D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(A)))
A_corr = D_inv_sqrt @ A @ D_inv_sqrt

# Propriétés :
# - A_corr[i,i] = 1
# - A_corr SPD (valeurs propres > 0)
# - A_corr[i,j] ∈ [-1, 1]
```

### A.4 Symétrisation partielle tenseur

```python
# Symétrie indices 1-2 (j,k)
T_sym = (T + np.transpose(T, (0, 2, 1))) / 2

# Symétrie complète (toutes permutations)
T_full = (T +
          np.transpose(T, (0, 2, 1)) +
          np.transpose(T, (1, 0, 2)) +
          np.transpose(T, (1, 2, 0)) +
          np.transpose(T, (2, 0, 1)) +
          np.transpose(T, (2, 1, 0))) / 6.0
```

---

## ANNEXE B : HISTORIQUE MODIFICATIONS

| Date | Version | Changement |
|------|---------|------------|
| 2025-01-15 | 6.0.0 | Création catalogue initial |

---

## ANNEXE C : INDEX ALPHABÉTIQUE FONCTIONS

| Fonction | ID | Module | Rang |
|----------|-----|--------|------|
| create_antisymmetric | ASY-003 | rank2_asymmetric | 2 |
| create_banded | SYM-005 | rank2_symmetric | 2 |
| create_block_hierarchical | SYM-006 | rank2_symmetric | 2 |
| create_correlation_matrix | SYM-004 | rank2_symmetric | 2 |
| create_directional_gradient | ASY-004 | rank2_asymmetric | 2 |
| create_identity | SYM-001 | rank2_symmetric | 2 |
| create_local_coupling_rank3 | R3-003 | rank3_correlations | 3 |
| create_lower_triangular | ASY-002 | rank2_asymmetric | 2 |
| create_partial_symmetric_rank3 | R3-002 | rank3_correlations | 3 |
| create_random | - | rank2_symmetric | 2 |
| create_random_asymmetric | ASY-001 | rank2_asymmetric | 2 |
| create_random_gaussian | SYM-003 | rank2_symmetric | 2 |
| create_random_rank3 | R3-001 | rank3_correlations | 3 |
| create_random_uniform | SYM-002 | rank2_symmetric | 2 |
| create_uniform | - | rank2_symmetric | 2 |

**Légende** :
- ID avec valeur : encodage catalogué
- ID "-" : fonction utilitaire/bonus (non cataloguée)

---

**FIN d_encoding_catalog.md**