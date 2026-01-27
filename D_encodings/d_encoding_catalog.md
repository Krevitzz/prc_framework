# D_ENCODINGS CATALOG

> Création d'états D^(base) pour tenseurs rang 2 et rang 3  
> Présuppositions explicites HORS du core  
> Architecture PHASE 10: 1 fichier = 1 encoding

**Version**: 2.0 (PHASE 10)  
**Date**: 2026-01-23

---

## 📐 ARCHITECTURE

### Convention nommage
```
D_encodings/
├── sym_NNN_descriptif.py     # Rang 2 symétrique
├── asy_NNN_descriptif.py     # Rang 2 asymétrique
└── r3_NNN_descriptif.py      # Rang 3
```

### Structure fichier (OBLIGATOIRE)
```python
PHASE = "R0"  # OBLIGATOIRE

METADATA = {
    'id': 'XXX-NNN',           # OBLIGATOIRE
    'rank': 2 ou 3,            # OBLIGATOIRE
    'type': '...',             # OBLIGATOIRE
    'description': '...',      # OBLIGATOIRE
    'properties': [...],
    'usage': '...'
}

def create(n_dof: int, seed: int = None, **kwargs) -> np.ndarray:
    """..."""
    pass
```

### Discovery
- Pattern: `{sym,asy,r3}_*.py`
- Skip: `*_deprecated_*`
- Validation: `PHASE` présent, `METADATA['id']` présent, `create()` présent
- Extraction ID: `METADATA['id']`

---

## 📋 TENSEURS RANG 2 SYMÉTRIQUE (SYM-*)

### SYM-001 - Identité
**Fichier**: `sym_001_identity.py`  
**Fonction**: `create(n_dof, seed=None)`  
**Forme**: I_ij = δ_ij  
**Interprétation**: DOF indépendants (corrélations nulles)  
**Propriétés**: Symétrique, définie positive, sparse  
**Usage**: Test stabilité minimale, point fixe trivial

---

### SYM-002 - Aléatoire Uniforme
**Fichier**: `sym_002_random_uniform.py`  
**Fonction**: `create(n_dof, seed=None)`  
**Forme**: A = (B + B^T)/2, B_ij ~ U[-1,1]  
**Propriétés**: Symétrique, bornes [-1,1]  
**Usage**: Test diversité maximale, générique

---

### SYM-003 - Aléatoire Gaussienne
**Fichier**: `sym_003_random_gaussian.py`  
**Fonction**: `create(n_dof, seed=None, sigma=0.3)`  
**Forme**: A = (B + B^T)/2, B_ij ~ N(0, σ=0.3)  
**Propriétés**: Symétrique, non bornée a priori  
**Paramètres**:
- `sigma`: Écart-type distribution (0.3 par défaut)

**Usage**: Test continuité, distribution normale

---

### SYM-004 - Matrice de Corrélation (SPD)
**Fichier**: `sym_004_correlation_matrix.py`  
**Fonction**: `create(n_dof, seed=None)`  
**Forme**: A = C·C^T normalisée, C_ij ~ N(0,1)  
**Propriétés**: Symétrique, semi-définie positive, diag=1  
**Usage**: Test positivité définie

---

### SYM-005 - Bande Symétrique
**Fichier**: `sym_005_banded.py`  
**Fonction**: `create(n_dof, seed=None, bandwidth=3, amplitude=0.5)`  
**Forme**: A_ij ≠ 0 ssi |i-j| ≤ bandwidth, valeurs ~ U[-amplitude, amplitude]  
**Propriétés**: Symétrique, sparse, bande  
**Paramètres**:
- `bandwidth`: Largeur de bande (3 par défaut)
- `amplitude`: Amplitude valeurs hors-diagonale (0.5 par défaut)

**Usage**: Test localité structurelle

---

### SYM-006 - Hiérarchique par Blocs
**Fichier**: `sym_006_block_hierarchical.py`  
**Fonction**: `create(n_dof, seed=None, n_blocks=10, intra_corr=0.7, inter_corr=0.1)`  
**Forme**: Blocs denses intra (corrélation forte), sparse inter (corrélation faible)  
**Propriétés**: Symétrique, structure blocs  
**Paramètres**:
- `n_blocks`: Nombre de blocs (10 par défaut)
- `intra_corr`: Corrélation intra-bloc (0.7 par défaut)
- `inter_corr`: Corrélation inter-blocs (0.1 par défaut)

**Contrainte**: n_dof divisible par n_blocks  
**Usage**: Test préservation structure modulaire

---

## 📋 TENSEURS RANG 2 ASYMÉTRIQUE (ASY-*)

### ASY-001 - Aléatoire Asymétrique
**Fichier**: `asy_001_random_asymmetric.py`  
**Fonction**: `create(n_dof, seed=None)`  
**Forme**: A_ij ~ U[-1,1] indépendants  
**Propriétés**: Asymétrique, bornes [-1,1]  
**Usage**: Test asymétrie générique

---

### ASY-002 - Triangulaire Inférieure
**Fichier**: `asy_002_lower_triangular.py`  
**Fonction**: `create(n_dof, seed=None)`  
**Forme**: A_ij = U[-1,1] si i > j, sinon 0  
**Propriétés**: Asymétrique, sparse, triangulaire  
**Usage**: Test orientation directionnelle

---

### ASY-003 - Antisymétrique
**Fichier**: `asy_003_antisymmetric.py`  
**Fonction**: `create(n_dof, seed=None)`  
**Forme**: A = -A^T, A_ij ~ U[-1,1] pour i>j  
**Propriétés**: Antisymétrique (cas spécial asymétrique), diagonale nulle  
**Usage**: Test conservation antisymétrie

---

### ASY-004 - Gradient Directionnel
**Fichier**: `asy_004_directional_gradient.py`  
**Fonction**: `create(n_dof, seed=None, gradient=0.1, noise_amplitude=0.2)`  
**Forme**: A_ij = gradient·(i-j) + U[-noise, +noise]  
**Propriétés**: Asymétrique, gradient linéaire  
**Paramètres**:
- `gradient`: Pente du gradient (0.1 par défaut)
- `noise_amplitude`: Amplitude bruit additif (0.2 par défaut)

**Usage**: Test brisure symétrie avec structure

---

## 📋 TENSEURS RANG 3 (R3-*)

### R3-001 - Aléatoire Uniforme
**Fichier**: `r3_001_random_uniform.py`  
**Fonction**: `create(n_dof, seed=None)`  
**Forme**: T_ijk ~ U[-1,1]  
**Propriétés**: Aucune symétrie, bornes [-1,1]  
**Usage**: Test générique rang 3  
**Note**: Coût mémoire O(N³) - N=20 → 8000 éléments

---

### R3-002 - Symétrique Partiel
**Fichier**: `r3_002_partial_symmetric.py`  
**Fonction**: `create(n_dof, seed=None)`  
**Forme**: T_ijk = T_ikj, valeurs ~ U[-1,1]  
**Propriétés**: Symétrie sur indices (j,k)  
**Usage**: Test symétries partielles

---

### R3-003 - Couplages Locaux
**Fichier**: `r3_003_local_coupling.py`  
**Fonction**: `create(n_dof, seed=None, radius=2)`  
**Forme**: T_ijk ≠ 0 ssi |i-j|+|j-k|+|k-i| ≤ 2*radius  
**Propriétés**: Sparse, localité géométrique  
**Paramètres**:
- `radius`: Rayon de localité (2 par défaut → 2*radius=4)

**Usage**: Test localité 3-corps

---

## 🔍 PRÉSUPPOSITIONS EXPLICITES

### Rang 2 Symétrique
- C[i,j] = C[j,i]
- Diagonale C[i,i] = 1 (selon type)
- Bornes [-1, 1] (sauf SYM-003)

### Rang 2 Asymétrique
- C[i,j] ≠ C[j,i] (général)
- Pas de contrainte diagonale (sauf spécifié)
- Bornes [-1, 1] (sauf spécifié)

### Rang 3
- Aucune symétrie par défaut (sauf spécifié)
- Bornes [-1, 1] (sauf spécifié)
- Coût mémoire O(N³)

---

## ✅ VALIDATION

### Validation discovery (automatique)
- `PHASE` présent et == "R0"
- `METADATA['id']` présent
- `create()` fonction présente
- Cohérence `METADATA['id']` avec nom fichier

### Validation dimensionnelle (runtime)
Tous les encodings DOIVENT:
- Retourner `np.ndarray` de shape correcte
- Respecter bornes déclarées (si spécifié)
- Garantir propriétés structurelles (symétrie, etc.)

Validation effectuée HORS du core (dans D_encodings/ même).

---

## 🚫 DÉPENDANCES

**Autorisées**: NumPy uniquement  
**Interdites**: Tout module PRC (core/, operators/, tests/, utilities/)

---

## 📏 RÈGLES ARCHITECTURALES (PHASE 10)

### Règle G1 - Une brique = un mécanisme
**Critère**: "Puis-je expliquer en une phrase ce que cette brique fait de différent ?"
- OUI → nouveau fichier
- NON → paramètre du même fichier

**Exemple BON**:
- `sym_002_random_uniform.py` (distribution uniforme)
- `sym_003_random_gaussian.py` (distribution gaussienne)
→ Mécanismes différents

**Exemple MAUVAIS** (À ÉVITER):
- `sym_002_random_uniform.py`
- `sym_002_random_uniform_low_variance.py`
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

### SYM-007 - Corrélations Uniformes
**Fichier**: `sym_007_uniform_correlation.py`  
**Fonction**: `create(n_dof, seed=None, correlation=0.5)`  
**Forme**: C[i,j] = correlation (i≠j), C[i,i] = 1  
**Interprétation**: Tous DOF également corrélés  
**Propriétés**: Symétrique, corrélation uniforme, diagonale=1  
**Paramètres**:
- `correlation`: Valeur uniforme dans [-1,1] (0.5 par défaut)

**Usage**: Test corrélations égales entre tous DOF

---

### SYM-008 - Aléatoire Clippé Paramétrable
**Fichier**: `sym_008_random_clipped.py`  
**Fonction**: `create(n_dof, seed=None, mean=0.0, std=0.3)`  
**Forme**: C = (N(mean,std) + N(mean,std)^T)/2, clippé [-1,1], diag=1  
**Propriétés**: Symétrique, distribution paramétrable, clippé, diagonale=1  
**Paramètres**:
- `mean`: Moyenne corrélations (0.0 par défaut)
- `std`: Écart-type (0.3 par défaut)

**Usage**: Test distribution normale paramétrable avec clipping

---

## 📋 TENSEURS RANG 2 ASYMÉTRIQUE (ASY-*) (suite)

### ASY-005 - Circulant
**Fichier**: `asy_005_circulant.py`  
**Fonction**: `create(n_dof, seed=None)`  
**Forme**: Chaque ligne = précédente décalée d'un cran  
**Propriétés**: Asymétrique, structure circulante, périodique  
**Usage**: Test structure périodique asymétrique

---

### ASY-006 - Sparse Contrôlé
**Fichier**: `asy_006_sparse.py`  
**Fonction**: `create(n_dof, seed=None, density=0.2)`  
**Forme**: density% des éléments non-nuls, asymétrique  
**Propriétés**: Asymétrique, sparse, densité contrôlée  
**Paramètres**:
- `density`: Densité valeurs non-nulles (0.2 = 20% par défaut)

**Usage**: Test structures creuses asymétriques

---

## 📋 TENSEURS RANG 3 (R3-*) (suite)

### R3-004 - Symétrique Total
**Fichier**: `r3_004_fully_symmetric.py`  
**Fonction**: `create(n_dof, seed=None)`  
**Forme**: T_ijk = T_jik = T_ikj = T_jki = T_kij = T_kji  
**Propriétés**: Symétrie totale, invariant 6 permutations  
**Usage**: Test symétrie complète

---

### R3-005 - Diagonal
**Fichier**: `r3_005_diagonal.py`  
**Fonction**: `create(n_dof, seed=None)`  
**Forme**: T_ijk ≠ 0 ssi i=j=k  
**Propriétés**: Très sparse, diagonal  
**Usage**: Test structure diagonale rang 3

---

### R3-006 - Séparable
**Fichier**: `r3_006_separable.py`  
**Fonction**: `create(n_dof, seed=None)`  
**Forme**: T_ijk = u_i · v_j · w_k (produit externe)  
**Propriétés**: Séparable, factorisé, rang tensoriel=1  
**Usage**: Test structure factorisée

---

### R3-007 - Par Blocs
**Fichier**: `r3_007_block_structure.py`  
**Fonction**: `create(n_dof, seed=None, n_blocks=4)`  
**Forme**: Structure par blocs dans les 3 dimensions  
**Propriétés**: Structure hiérarchique, modulaire  
**Paramètres**:
- `n_blocks`: Nombre de blocs par dimension (4 par défaut)

**Contrainte**: n_dof divisible par n_blocks  
**Usage**: Test modularité rang 3

---

## 📊 RÉCAPITULATIF

| Catégorie | Count | Fichiers |
|-----------|-------|----------|
| **Symétrique** | 8 | sym_001 à sym_008 |
| **Asymétrique** | 6 | asy_001 à asy_006 |
| **Rang 3** | 7 | r3_001 à r3_007 |
| **TOTAL** | 21 | encodings R0 |

---

## 🔄 NOTES ARCHITECTURALES

### Séparation stricte
- Encodings définissent **QUOI** créer (structure, propriétés)
- Core applique **COMMENT** transformer (prepare_state, run_kernel)
- Tests observent **RÉSULTAT** sans juger

### Principe fondamental
Les présuppositions (symétrie, bornes) sont des **contraintes métier** HORS du core aveugle.

### Évolution R1+
Nouveaux encodings R1 possibles:
- Compositions encodings R0
- Structures rang 4+
- Encodings domaine-spécifiques

**Règle**: Toujours respecter G1-G3 (garde-fous).

---

**FIN D_ENCODING CATALOG v2.0**