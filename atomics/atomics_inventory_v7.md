# ATOMICS INVENTORY PRC v7 — JAX
> Inventaire stabilisé avant réécriture JAX
> Remplace atomics_catalog.md pour le pipeline prc_framework/
> Date : 2026-03-06

---

## DÉCISIONS ARCHITECTURALES

### Rang comme axe d'itération

Le rang n'est pas un axe YAML global — il est un **paramètre d'encoding** pour les
encodings paramétriques (RN-*). La raison : SYM-*/ASY-* n'ont pas de sens en rang
arbitraire (symétrie, antisymétrie, corrélation sont des concepts rang-2).

```yaml
# Dans le YAML de run
encoding:
  - id: RN-001
    params:
      rank: [2, 3, 4, 5]    # → 4 compositions distinctes
  - id: SYM-002              # → toujours rang 2, pas de paramètre rank
```

Le rang effectif est dérivé de la shape du tenseur de sortie.
Il entre dans la clé de groupement vmap : `(gamma_id, encoding_id, n_dof, rank_eff, max_it)`.

### Signature JAX unifiée (roadmap section 2.1)

```python
# Encoding
METADATA = {'id': 'XXX-NNN', 'rank': 2, 'stochastic': bool}
def create(n_dof: int, params: dict, key: jax.Array) -> jnp.ndarray: ...

# Gamma
METADATA = {'id': 'GAM-NNN', 'family': str, 'rank_constraint': None|2|'square',
            'differentiable': bool, 'stochastic': bool, 'non_markovian': bool}
def apply(state: jnp.ndarray, params: dict, key: jax.Array) -> jnp.ndarray: ...

# Modifier
METADATA = {'id': 'MN', 'stochastic': bool}
def apply(state: jnp.ndarray, params: dict, key: jax.Array) -> jnp.ndarray: ...
```

`key` partout — ignorée si `stochastic: False`.
`non_markovian: True` → kernel étend le carry avec `prev_state`.
`rank_constraint` → ValueError levée dans `apply()` si incompatible.

### Sort des R3-* numpy

| R3 | Sort | Raison |
|----|------|--------|
| R3-001 random uniform | → RN-001 rank=3 | supersédé |
| R3-002 partial symmetric | → RN-002 rank=3 | supersédé |
| R3-003 local coupling | **conservé** | structure spatiale rank-3 spécifique |
| R3-004 fully symmetric | → RN-002 + symétrie complète | supersédé |
| R3-005 diagonal | → RN-003 rank=3 | supersédé |
| R3-006 separable | → RN-004 rank=3 | supersédé |
| R3-007 block structure | **conservé** | structure bloc 3D spécifique |

R3-003 et R3-007 conservés en JAX — les autres sont des cas particuliers de RN-*.

---

## ENCODINGS

### SYM-* — Symétrique rang 2 (inchangés)

| ID | Fichier JAX | Forme | Params | Stochastique |
|----|-------------|-------|--------|--------------|
| SYM-001 | sym_001_identity.py | I_ij = δ_ij | — | Non |
| SYM-002 | sym_002_random_uniform.py | (B+Bᵀ)/2, U[-1,1] | — | Oui (key) |
| SYM-003 | sym_003_random_gaussian.py | (B+Bᵀ)/2, N(0,σ) | sigma=0.3 | Oui (key) |
| SYM-004 | sym_004_correlation_matrix.py | C·Cᵀ normalisé | — | Oui (key) |
| SYM-005 | sym_005_banded.py | Bande sparse | bandwidth=3, amplitude=0.5 | Non |
| SYM-006 | sym_006_block_hierarchical.py | Blocs intra/inter | n_blocks=10, intra=0.7, inter=0.1 | Oui (key) |
| SYM-007 | sym_007_uniform_correlation.py | C_ij=ρ (i≠j), 1 (diag) | correlation=0.5 | Non |
| SYM-008 | sym_008_random_clipped.py | N(mean,std) clippé [-1,1] | mean=0.0, std=0.3 | Oui (key) |

**Note** : `rank` figé à 2. Stochastique = key consommée pour la génération initiale
(pas dans lax.scan — les encodings ne sont appelés qu'une fois avant le scan).

### ASY-* — Asymétrique rang 2 (inchangés)

| ID | Fichier JAX | Forme | Params | Stochastique |
|----|-------------|-------|--------|--------------|
| ASY-001 | asy_001_random_asymmetric.py | A_ij ~ U[-1,1] | — | Oui (key) |
| ASY-002 | asy_002_lower_triangular.py | Triangulaire inf. | — | Oui (key) |
| ASY-003 | asy_003_antisymmetric.py | (B-Bᵀ)/2 | — | Oui (key) |
| ASY-004 | asy_004_directional_gradient.py | gradient·(i-j) + bruit | gradient=0.1, noise=0.2 | Oui (key) |
| ASY-005 | asy_005_circulant.py | Circulant décalé | — | Non |
| ASY-006 | asy_006_sparse.py | Sparse asymétrique | density=0.2 | Oui (key) |

### RN-* — Rang paramétrique (NOUVEAUX)

Encodings dont `rank` est un paramètre explicite → couvre rang 2, 3, 4, 5+.
Shape de sortie : `(n_dof,) * rank`.

| ID | Fichier JAX | Forme | Params | Stochastique | Remplace |
|----|-------------|-------|--------|--------------|---------|
| RN-001 | rn_001_random_uniform.py | T_i₁..iᵣ ~ U[-1,1] | rank=3 | Oui (key) | R3-001 |
| RN-002 | rn_002_partial_symmetric.py | T sym sur axes 0,1 | rank=3 | Oui (key) | R3-002, R3-004 |
| RN-003 | rn_003_diagonal.py | T_i..i ≠ 0 ssi i₁=..=iᵣ | rank=3 | Non | R3-005 |
| RN-004 | rn_004_separable.py | u₁⊗u₂⊗..⊗uᵣ | rank=3 | Oui (key) | R3-006 |

**Note RN-004** : entanglement entropy = 0 par construction → point zéro de référence F3.
**Note RN-002** : symétrie sur les deux premiers axes uniquement (T_ij.. = T_ji..).
  Symétrie complète toutes permutations → coûteux, non prioritaire.

### R3-* conservés — Rang 3 structures spécifiques

| ID | Fichier JAX | Forme | Params | Justification |
|----|-------------|-------|--------|---------------|
| R3-003 | r3_003_local_coupling.py | Sparse local radius | radius=2 | Structure spatiale, pas généralisable proprement |
| R3-007 | r3_007_block_structure.py | Blocs 3D | n_blocks=4 | Structure bloc 3D, spécifique |

---

## GAMMAS

### Légende METADATA

```
family           : markovian | non_markovian | stochastic | structural
rank_constraint  : None (tout rang) | 2 (rang 2) | 'square' (carré rang 2)
differentiable   : True → jacfwd F4 disponible
stochastic       : True → key consommée dans apply()
non_markovian    : True → carry étendu avec prev_state dans kernel
```

### Inventaire complet

| ID | Forme | Family | rank_constraint | diff | stoch | non_markov | Rôle v7 |
|----|-------|--------|-----------------|------|-------|------------|---------|
| GAM-001 | tanh(β·T) | markovian | None | ✓ | ✗ | ✗ | Baseline saturation |
| GAM-002 | Diffusion Laplacienne | markovian | 2 | ✓ | ✗ | ✗ | Transport rang-2 |
| GAM-003 | T·exp(+γ) | markovian | None | ✓ | ✗ | ✗ | Test explosion / robustesse NaN |
| GAM-004 | T·exp(-γ) | markovian | None | ✓ | ✗ | ✗ | Decay, annihilation D |
| GAM-005 | Oscillateur harmonique | non_markovian | None | ✓ | ✗ | ✓ | Régime oscillatoire |
| GAM-006 | tanh(β·T + α·ΔT) | non_markovian | None | ✓ | ✗ | ✓ | Saturation + momentum |
| GAM-007 | Moyenne 8-voisins | non_markovian | 2 | ✓ | ✗ | ✗ | Diffusion spatiale locale |
| GAM-008 | Mémoire différentielle | non_markovian | None | ✓ | ✗ | ✓ | Dynamique inertielle |
| GAM-009 | tanh(β·T) + σ·ε | stochastic | None | ✗ | ✓ | ✗ | Saturation stochastique |
| GAM-010 | tanh(T·(1+σ·ε)) | stochastic | None | ✗ | ✓ | ✗ | Perturbation multiplicative |
| GAM-011 | W⊗T (tensordot mode-0) | markovian | None | ✓ | ✗ | ✗ | **Linéaire rang-agnostique — validation F7** |
| GAM-012 | (T+Tᵀ)/2·tanh(β) | structural | 2 | ✓ | ✗ | ✗ | Symétrisation forcée |
| GAM-013 | T + η·T@T (Hebbien) | structural | square | ✓ | ✗ | ✗ | Apprentissage associatif |
| GAM-014 | U@T@Uᵀ (orthogonal) | markovian | 2 | ✓ | ✗ | ✗ | **Null model A3 — conserve S_VN** |
| GAM-015 | SVD_k(T) / σ₁ | structural | None | ✗ | ✗ | ✗ | **Attracteur holographique — réduit rang** |

### Notes gammas nouveaux

**GAM-011 — Linéaire tensordot**
```
T_{n+1} = W ⊗₀ T  =  tensordot(W, T, axes=([1],[0]))
W : (n_dof, n_dof), généré une fois, passé dans params
scale : contrôle ||W|| / ρ(W)
  scale < 1 → contraction     scale = 1 → isométrie     scale > 1 → expansion
differentiable: True  (tensordot linéaire → jacfwd analytique)
rank_constraint: None  (tensordot mode-0 valide sur tout rang)
Rôle F7 : spectre DMD = valeurs propres de W → validation analytique possible
```

**GAM-014 — Orthogonal**
```
T_{n+1} = U @ T @ Uᵀ   (rang 2)
U : matrice orthogonale (n_dof, n_dof), QR d'une matrice aléatoire, figée
trace_J = 0 par construction  (transformation isométrique)
S_VN conservé par construction → null model A3 direct
differentiable: True  (linéaire en T)
rank_constraint: 2
Rôle F4 : trace_J = 0 → point de référence contraction/expansion
Rôle A3 : Γ qui ne dissipe PAS D → cas limite de résistance maximale
```

**GAM-015 — SVD Truncation**
```
T_{n+1} = SVD_k(mode0(T)) / σ₁   (projection rang-k normalisée)
k : rang de troncature (paramètre)  défaut = n_dof // 2
mode-0 unfolding → reconstruction → reshape original
differentiable: False  (SVD truncation non-smooth)
stochastic: False
rank_constraint: None  (mode-0 unfolding valide sur tout rang)
Rôle F2 : S_VN décroît garantie → attracteur holographique candidat contrôlé
Rôle Q3 : produit des structures émergentes stables par construction
```

---

## MODIFIERS (inchangés)

| ID | Fichier JAX | Transformation | Params | Stochastique |
|----|-------------|---------------|--------|--------------|
| M0 | m0_baseline.py | D' = D | — | Non |
| M1 | m1_gaussian_noise.py | D' = D + N(0,σ) | sigma=0.05 | Oui (key) |
| M2 | m2_uniform_noise.py | D' = D + U[-a,+a] | amplitude=0.1 | Oui (key) |

---

## AUDIT VMAP

### Clé de groupement

Un groupe vmappable est défini par :
```
(gamma_id, encoding_id, n_dof, rank_eff, max_it)
```
où `rank_eff` = `len(D.shape)` — dérivé de la shape de sortie de l'encoding.

Au sein d'un groupe, vmap porte sur :
```
params numériques (beta, sigma, scale, k, ...)
seed_CI  → PRNGKey encoding
seed_run → PRNGKey gamma stochastique
```

### Compatibilités encoding × gamma

```
rank_eff = 2 (SYM-*, ASY-*, RN-* rank=2)
  Gammas compatibles : tous sauf rank_constraint=None exclusif rang 3+ (aucun)
  GAM-002 ✓  GAM-007 ✓  GAM-012 ✓  GAM-013 (square uniquement) ✓
  GAM-011 ✓  GAM-014 ✓  GAM-015 ✓

rank_eff = 3 (R3-003, R3-007, RN-* rank=3)
  Gammas compatibles : rank_constraint=None uniquement
  GAM-001 ✓  GAM-003 ✓  GAM-004 ✓  GAM-005 ✓  GAM-006 ✓
  GAM-008 ✓  GAM-009 ✓  GAM-010 ✓  GAM-011 ✓  GAM-015 ✓
  GAM-002 ✗  GAM-007 ✗  GAM-012 ✗  GAM-013 ✗  GAM-014 ✗

rank_eff ≥ 4 (RN-* rank=4,5,...)
  Gammas compatibles : rank_constraint=None uniquement
  GAM-001 ✓  GAM-003 ✓  GAM-004 ✓  GAM-005 ✓  GAM-006 ✓
  GAM-008 ✓  GAM-009 ✓  GAM-010 ✓  GAM-011 ✓  GAM-015 ✓
  GAM-002 ✗  GAM-007 ✗  GAM-012 ✗  GAM-013 ✗  GAM-014 ✗
```

### Gammas non-markoviens — carry étendu

GAM-005, GAM-006, GAM-008 requièrent `prev_state` dans le carry.
Le kernel détecte `METADATA['non_markovian'] = True` en Python avant compilation.
Deux versions compilées du kernel : carry standard vs carry étendu.
Zéro branchement dans lax.scan — décision faite en Python à la construction.

### Différentiabilité — F4 jacfwd

| Gammas differentiable=True | Condition jacfwd pratique |
|---------------------------|--------------------------|
| GAM-001,002,003,004,005,006,007,008,011,012,013,014 | n_dof ≤ 50 (J = n_dof² × n_dof²) |
| GAM-009, GAM-010 | ✗ stochastique → jacfwd invalide |
| GAM-015 | ✗ SVD truncation non-smooth |

Hutchinson (F4.1) disponible pour tous — jacfwd complet limité aux petits n_dof.

### Recompilations XLA anticipées

| Axe | Cause recompilation | Mitigation |
|-----|---------------------|------------|
| gamma_id | Fonction différente | Groupement — 1 compile par gamma |
| encoding_id | Fonction différente | Groupement — 1 compile par encoding |
| n_dof | Shape différente | Groupement — 1 compile par n_dof |
| rank_eff | Shape différente | Groupement — 1 compile par rank |
| max_it | Longueur scan | Groupement — 1 compile par max_it |
| params numériques | Valeurs scalaires | vmap — 0 recompilation |
| seeds | PRNGKey | vmap — 0 recompilation |
| non_markovian | Carry shape | 2 versions kernel max |

Nombre max de compilations :
```
n_gamma × n_encoding × n_dof × n_rank × n_max_it × n_kernel_versions
= 15 × 20 × 4 × 4 × 6 × 2 = ~57 600 groupes théoriques max
En pratique (contraintes rang) : << 10 000 groupes réels
```
Chaque compilation est amortie sur N runs du groupe via vmap.

---

## RÉSUMÉ INVENTAIRE

| Catégorie | Existants numpy | JAX réécrits | JAX nouveaux | Total JAX |
|-----------|----------------|--------------|--------------|-----------|
| SYM-* | 8 | 8 | 0 | 8 |
| ASY-* | 6 | 6 | 0 | 6 |
| RN-* | 0 | 0 | 4 | 4 |
| R3-* conservés | 7 | 2 | 0 | 2 |
| R3-* supersédés | 5 | 0 | — | 0 |
| Gammas | 12 | 12 | 3 | 15 |
| Modifiers | 3 | 3 | 0 | 3 |
| **TOTAL** | **34** | **31** | **7** | **38** |

---

**FIN ATOMICS INVENTORY PRC v7**
*Document vivant — mis à jour à chaque réécriture atomic.*
