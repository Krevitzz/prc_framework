# D_ENCODINGS CATALOG

> Création d'états D^(base) pour tenseurs rang 2 et rang 3  
> Présuppositions explicites HORS du core  

**RAPPEL** : Encodings définissent présuppositions (symétrie, bornes, etc.) HORS du core

## TENSEURS RANG 2 SYMÉTRIQUES (SYM-*)

### SYM-001 - Identité
**Fichier** : `rank2_symmetric.py::create_identity()`  
**Forme** : I_ij = δ_ij  
**Interprétation** : DOF indépendants (corrélations nulles)  
**Propriétés** : Symétrique, définie positive, sparse  
**Usage** : Test stabilité minimale, point fixe trivial

### SYM-002 - Aléatoire Uniforme
**Fichier** : `rank2_symmetric.py::create_random_uniform()`  
**Forme** : A = (B + B^T)/2, B_ij ~ U[-1,1]  
**Propriétés** : Symétrique, bornes [-1,1]  
**Usage** : Test diversité maximale, générique

### SYM-003 - Aléatoire Gaussienne
**Fichier** : `rank2_symmetric.py::create_random_gaussian()`  
**Forme** : A = (B + B^T)/2, B_ij ~ N(0, σ=0.3)  
**Propriétés** : Symétrique, non bornée a priori  
**Usage** : Test continuité, distribution normale

### SYM-004 - Matrice de Corrélation (SPD)
**Fichier** : `rank2_symmetric.py::create_correlation_matrix()`  
**Forme** : A = C·C^T normalisée, C_ij ~ N(0,1)  
**Propriétés** : Symétrique, définie positive, diag=1  
**Usage** : Test positivité définie

### SYM-005 - Bande Symétrique
**Fichier** : `rank2_symmetric.py::create_banded()`  
**Forme** : A_ij ≠ 0 ssi |i-j| ≤ bandwidth, valeurs ~ U[-amplitude, amplitude]  
**Propriétés** : Symétrique, sparse, bande  
**Paramètres** : bandwidth=3, amplitude=0.5  
**Usage** : Test localité structurelle

### SYM-006 - Hiérarchique par Blocs
**Fichier** : `rank2_symmetric.py::create_block_hierarchical()`  
**Forme** : Blocs denses intra (corrélation forte), sparse inter (corrélation faible)  
**Propriétés** : Symétrique, structure blocs  
**Paramètres** : n_blocks=10, intra_corr=0.7, inter_corr=0.1  
**Contrainte** : n_dof divisible par n_blocks  
**Usage** : Test préservation structure modulaire

## TENSEURS RANG 2 ASYMÉTRIQUES (ASY-*)

### ASY-001 - Aléatoire Asymétrique
**Fichier** : `rank2_asymmetric.py::create_random_asymmetric()`  
**Forme** : A_ij ~ U[-1,1] indépendants  
**Propriétés** : Asymétrique, bornes [-1,1]  
**Usage** : Test asymétrie générique

### ASY-002 - Triangulaire Inférieure
**Fichier** : `rank2_asymmetric.py::create_lower_triangular()`  
**Forme** : A_ij = U[-1,1] si i > j, sinon 0  
**Propriétés** : Asymétrique, sparse, triangulaire  
**Usage** : Test orientation directionnelle

### ASY-003 - Antisymétrique
**Fichier** : `rank2_asymmetric.py::create_antisymmetric()`  
**Forme** : A = -A^T, A_ij ~ U[-1,1] pour i>j  
**Propriétés** : Antisymétrique (cas spécial asymétrique), diagonale nulle  
**Usage** : Test conservation antisymétrie

### ASY-004 - Gradient Directionnel
**Fichier** : `rank2_asymmetric.py::create_directional_gradient()`  
**Forme** : A_ij = gradient·(i-j) + U[-noise, +noise]  
**Propriétés** : Asymétrique, gradient linéaire  
**Paramètres** : gradient=0.1, noise_amplitude=0.2  
**Usage** : Test brisure symétrie avec structure

## TENSEURS RANG 3 (R3-*)

### R3-001 - Aléatoire Uniforme
**Fichier** : `rank3_correlations.py::create_random_rank3()`  
**Forme** : T_ijk ~ U[-1,1]  
**Propriétés** : Aucune symétrie  
**Usage** : Test générique rang 3  
**Note** : Coût mémoire N³ (ex: N=20 → 8000 éléments)

### R3-002 - Symétrique Partiel
**Fichier** : `rank3_correlations.py::create_partial_symmetric_rank3()`  
**Forme** : T_ijk = T_ikj, valeurs ~ U[-1,1]  
**Propriétés** : Symétrie sur 2 indices (j,k)  
**Usage** : Test symétries partielles

### R3-003 - Couplages Locaux
**Fichier** : `rank3_correlations.py::create_local_coupling_rank3()`  
**Forme** : T_ijk ≠ 0 ssi |i-j|+|j-k|+|k-i| ≤ 2*radius  
**Propriétés** : Sparse, localité géométrique  
**Paramètres** : radius=2 (donc 2*radius=4)  
**Usage** : Test localité 3-corps

## ENCODINGS BONUS 

**rank2_symmetric.py** :
- `create_uniform()` : Corrélations uniformes
- `create_random()` : Aléatoire avec mean/std paramétrables

**rank2_asymmetric.py** :
- `create_circulant_asymmetric()` : Structure circulante
- `create_sparse_asymmetric()` : Sparse avec densité paramétrable

**rank3_correlations.py** :
- `create_fully_symmetric_rank3()` : Symétrie totale (6 permutations)
- `create_diagonal_rank3()` : T_ijk ≠ 0 ssi i=j=k
- `create_separable_rank3()` : Produit externe u_i · v_j · w_k
- `create_block_rank3()` : Structure par blocs 3D

## PRÉSUPPOSITIONS EXPLICITES

**Rang 2 Symétrique** :
- C[i,j] = C[j,i]
- Diagonale C[i,i] = 1 (selon type)
- Bornes [-1, 1]

**Rang 2 Asymétrique** :
- C[i,j] ≠ C[j,i] (général)
- Pas de contrainte diagonale (sauf spécifié)
- Bornes [-1, 1] (sauf spécifié)

**Rang 3** :
- Aucune symétrie par défaut (sauf spécifié)
- Bornes [-1, 1] (sauf spécifié)
- Coût mémoire O(N³)

## VALIDATION DIMENSIONNELLE

**Tous les encodings DOIVENT** :
- Retourner `np.ndarray` de shape attendue
- Respecter bornes déclarées (si spécifié)
- Garantir propriétés structurelles (symétrie, etc.)

**Validation effectuée HORS du core** (dans D_encodings/ même)

## DÉPENDANCES

**Autorisées** : NumPy uniquement  
**Interdites** : Tout module PRC (core/, operators/, tests/, utilities/)

## NOTES ARCHITECTURALES

**Séparation stricte** :
- Encodings définissent **QUOI** créer (structure, propriétés)
- Core applique **COMMENT** transformer (prepare_state, run_kernel)
- Tests observent **RÉSULTAT** sans juger

**Principe** : Les présuppositions (symétrie, bornes) sont des **contraintes métier** HORS du core aveugle.

**FIN D_ENCODING CATALOG**