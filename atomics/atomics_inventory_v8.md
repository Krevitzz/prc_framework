# ATOMICS INVENTORY PRC v8 — JAX
> Inventaire final après Phase A (audit et mise à niveau des atomics)
> Date : 2026-03-17

---

## 1. DÉCISIONS ARCHITECTURALES FINALES

### 1.1 Signatures unifiées

Tous les atomics adoptent une signature JAX standardisée, permettant une vmappabilité sans wrappers.

| Type     | Signature                                                                 | Rôle de `key`                                      |
|----------|---------------------------------------------------------------------------|----------------------------------------------------|
| Encoding | `create(n_dof: int, params: dict, key: jax.Array) -> jnp.ndarray`        | Génération aléatoire de l'état initial (hors scan) |
| Gamma    | `apply(state: jnp.ndarray, prev_state: jnp.ndarray, params: dict, key: jax.Array) -> jnp.ndarray` | Aléatoire pendant le scan (si stochastique)       |
| Modifier | `apply(state: jnp.ndarray, params: dict, key: jax.Array) -> jnp.ndarray` | Aléatoire avant le scan (si stochastique)         |

- Les gammas markoviens ignorent `prev_state` et `key` (sauf si stochastiques).
- Les gammas non-markoviens utilisent `prev_state` pour la dynamique.
- Les encodings ne sont pas vmappés mais doivent être purs et rapides.

### 1.2 Métadonnées obligatoires

Chaque atomic expose un dictionnaire `METADATA` contenant :

```python
METADATA = {
    'id': str,                     # Identifiant unique
    'family': str,                  # Pour gammas : markovian, non_markovian, stochastic, structural
    'rank_constraint': None|2|'square',  # Contrainte sur le rang du tenseur d'entrée
    'differentiable': bool,         # True si la fonction est différentiable JAX
    'stochastic': bool,             # True si la fonction utilise key
    'non_markovian': bool,          # True pour les gammas ayant besoin de prev_state
    # Pour encodings uniquement :
    'rank': int|None,                # Rang fixe ou None si paramétrique
    'jax_vmappable': bool,           # True si la fonction supporte la vmap (par défaut True)
}
```

Les modifiers ajoutent aussi `rank_constraint=None` et `non_markovian=False`.

### 1.3 Gestion du batch

Les fonctions `apply` des gammas et modifiers sont écrites pour un **seul échantillon** (pas de dimension batch). La vectorisation est appliquée par `jax.vmap` au moment de la compilation du kernel.

### 1.4 Optimisation des encodages

- **ASY-005** : réécrit en version vectorisée (indexation modulo) pour éliminer la boucle Python.
- **RN-004** : réécrit avec génération vectorisée des vecteurs et `einsum` dynamique ; la micro-boucle de décompression (taille ≤5) est conservée car négligeable.

Ces optimisations améliorent les performances lors de la génération des groupes (exécutée en Python).

### 1.5 Suppression des vérifications redondantes

Tous les `if state.ndim != 2` ont été supprimés. La compatibilité est garantie par le plan (`rank_constraint`). Les axes absolus ont été remplacés par des axes négatifs (`axis=-2`, `axis=-1`) pour les opérations spatiales.

### 1.6 Gammas non-markoviens

GAM-005, GAM-006, GAM-008 sont désormais implémentés avec leur forme complète utilisant `prev_state`.

### 1.7 Corrections de batch

- **GAM-011** : utilisation de `jnp.tensordot(W, state, axes=([1],[0]))` (pas de batch implicite).
- **GAM-014** : `U @ state @ U.swapaxes(-1,-2)` pour une matrice unique.
- **GAM-015** : réécrit sans supposer de batch (reshape et SVD sur un tenseur unique).

---

## 2. ENCODINGS

### 2.1 SYM-* — Symétrique rang 2

| ID       | Fichier                          | Forme                                   | Params                          | Stochastique |
|----------|----------------------------------|-----------------------------------------|---------------------------------|--------------|
| SYM-001  | `sym_001_identity.py`            | I_ij = δ_ij                             | –                               | Non          |
| SYM-002  | `sym_002_random_uniform.py`      | (B+Bᵀ)/2, B~U[-1,1]                     | –                               | Oui          |
| SYM-003  | `sym_003_random_gaussian.py`     | (B+Bᵀ)/2, B~N(0,σ)                      | `sigma=0.3`                     | Oui          |
| SYM-004  | `sym_004_correlation_matrix.py`  | C = A·Aᵀ / diag(A·Aᵀ), A~N(0,1)         | –                               | Oui          |
| SYM-005  | `sym_005_banded.py`               | Bande symétrique                         | `bandwidth=3`, `amplitude=0.5`  | Non          |
| SYM-006  | `sym_006_block_hierarchical.py`   | Blocs intra/inter                        | `n_blocks=10`, `intra=0.7`, `inter=0.1` | Oui |
| SYM-007  | `sym_007_uniform_correlation.py`  | C_ij=ρ (i≠j), 1 (diag)                   | `correlation=0.5`                | Non          |
| SYM-008  | `sym_008_random_clipped.py`       | N(mean,std) clippé [-1,1]                | `mean=0.0`, `std=0.3`            | Oui          |

### 2.2 ASY-* — Asymétrique rang 2

| ID       | Fichier                          | Forme                                   | Params                          | Stochastique |
|----------|----------------------------------|-----------------------------------------|---------------------------------|--------------|
| ASY-001  | `asy_001_random_asymmetric.py`   | A_ij ~ U[-1,1]                          | –                               | Oui          |
| ASY-002  | `asy_002_lower_triangular.py`    | Triangulaire inférieure                  | –                               | Oui          |
| ASY-003  | `asy_003_antisymmetric.py`       | (B-Bᵀ)/2                                | –                               | Oui          |
| ASY-004  | `asy_004_directional_gradient.py`| gradient·(i-j) + bruit                   | `gradient=0.1`, `noise=0.2`      | Oui          |
| ASY-005  | `asy_005_circulant.py`           | Circulante : A_ij = exp(-((j-i) mod n))  | –                               | Non          |
| ASY-006  | `asy_006_sparse.py`              | Sparse asymétrique                       | `density=0.2`                    | Oui          |

### 2.3 RN-* — Rang paramétrique (nouveaux)

| ID       | Fichier                          | Forme                                   | Params                          | Stochastique | Remplace |
|----------|----------------------------------|-----------------------------------------|---------------------------------|--------------|----------|
| RN-001   | `rn_001_random_uniform.py`       | T ~ U[-1,1] sur toutes les composantes   | `rank=3`                        | Oui          | R3-001   |
| RN-002   | `rn_002_partial_symmetric.py`    | T symétrique sur axes 0 et 1             | `rank=3`                        | Oui          | R3-002, R3-004 |
| RN-003   | `rn_003_diagonal.py`             | T non nul seulement si tous indices égaux| `rank=3`                        | Non          | R3-005   |
| RN-004   | `rn_004_separable.py`            | Produit tensoriel de vecteurs normalisés | `rank=3`                        | Oui          | R3-006   |

### 2.4 R3-* conservés (rang 3 spécifiques)

| ID       | Fichier                          | Forme                                   | Params                          | Justification |
|----------|----------------------------------|-----------------------------------------|---------------------------------|---------------|
| R3-003   | `r3_003_local_coupling.py`       | Couplage local 3D (rayon)               | `radius=2`                      | Structure spatiale non généralisable |
| R3-007   | `r3_007_block_structure.py`      | Blocs 3D avec valeurs intra/inter        | `n_blocks=4`, `intra=0.8`, `inter=0.05` | Structure bloc 3D spécifique |

---

## 3. GAMMAS

### 3.1 Légende des métadonnées

- **family** : `markovian` | `non_markovian` | `stochastic` | `structural`
- **rank_constraint** : `None` (tout rang), `2` (rang 2), `'square'` (matrice carrée rang 2)
- **differentiable** : `True` → jacfwd disponible (F4)
- **stochastic** : `True` → `key` utilisée dans `apply`
- **non_markovian** : `True` → carry étendu avec `prev_state`

### 3.2 Liste complète des gammas

| ID       | Fichier                          | Forme                                   | Family         | rank_constraint | diff | stoch | non_markov | Rôle / Notes |
|----------|----------------------------------|-----------------------------------------|----------------|-----------------|------|-------|------------|--------------|
| GAM-001  | `gam_001_tanh.py`                | `tanh(β·state)`                         | markovian      | None            | ✓    | ✗     | ✗          | Saturation baseline |
| GAM-002  | `gam_002_diffusion.py`           | `state + α·Δ(state)` (Laplacien)        | markovian      | 2               | ✓    | ✗     | ✗          | Diffusion spatiale |
| GAM-003  | `gam_003_exp_growth.py`          | `state · exp(γ)`                        | markovian      | None            | ✓    | ✗     | ✗          | Explosion contrôlée |
| GAM-004  | `gam_004_exp_decay.py`            | `state · exp(-γ)`                       | markovian      | None            | ✓    | ✗     | ✗          | Décroissance exponentielle |
| GAM-005  | `gam_005_harmonic.py`             | `cos(ω)·state - sin(ω)·prev_state`      | non_markovian  | None            | ✓    | ✗     | ✓          | Oscillateur harmonique |
| GAM-006  | `gam_006_memory_tanh.py`          | `tanh(β·state + α·(state-prev_state))`  | non_markovian  | None            | ✓    | ✗     | ✓          | Saturation avec mémoire |
| GAM-007  | `gam_007_sliding_avg.py`          | Moyenne 8-voisins                        | markovian      | 2               | ✓    | ✗     | ✗          | Lissage spatial |
| GAM-008  | `gam_008_diff_memory.py`          | `tanh((1+β+γ)·state - γ·prev_state)`    | non_markovian  | None            | ✓    | ✗     | ✓          | Mémoire différentielle |
| GAM-009  | `gam_009_stochastic_tanh.py`      | `tanh(β·state) + σ·ε`                    | stochastic     | None            | ✗    | ✓     | ✗          | Saturation bruitée additive |
| GAM-010  | `gam_010_mult_noise.py`            | `tanh(state·(1+σ·ε))`                   | stochastic     | None            | ✗    | ✓     | ✗          | Bruit multiplicatif |
| GAM-011  | `gam_011_linear_tensordot.py`      | `tensordot(W, state, axes=([1],[0]))`   | markovian      | None            | ✓    | ✗     | ✗          | Linéaire rang-agnostique, validation F7 |
| GAM-012  | `gam_012_forced_sym.py`            | `(tanh(β·state) + sym) / 2`              | structural     | 2               | ✓    | ✗     | ✗          | Symétrisation forcée |
| GAM-013  | `gam_013_hebbian.py`               | `state + η·(state @ state)`              | structural     | square          | ✓    | ✗     | ✗          | Renforcement hebbien |
| GAM-014  | `gam_014_orthogonal.py`            | `U @ state @ Uᵀ`                        | markovian      | 2               | ✓    | ✗     | ✗          | Transformation orthogonale, null model A3 |
| GAM-015  | `gam_015_svd_truncation.py`        | `SVD_k(state)/σ₁`                        | structural     | None            | ✗    | ✗     | ✗          | Troncature SVD, attracteur holographique |

### 3.3 Notes importantes

- **GAM-011** : `W` est généré une fois par `prepare_params` et figé. `scale` contrôle le rayon spectral.
- **GAM-014** : `U` est généré par `prepare_params` (orthogonal). La norme de Frobenius est conservée.
- **GAM-015** : La valeur de `k` (rang de troncature) doit être statique (fournie dans `params`). L’implémentation utilise un masque pour éviter les problèmes de concrétisation.

---

## 4. MODIFIERS

| ID   | Fichier                    | Transformation                | Params          | Stochastique |
|------|----------------------------|-------------------------------|-----------------|--------------|
| M0   | `m0_baseline.py`           | `state` inchangé              | –               | Non          |
| M1   | `m1_gaussian_noise.py`     | `state + N(0,σ)`              | `sigma=0.05`    | Oui          |
| M2   | `m2_uniform_noise.py`      | `state + U[-a,+a]`            | `amplitude=0.1` | Oui          |

Les métadonnées des modifiers incluent désormais :
```python
METADATA = {
    'id': str,
    'family': 'modifier',
    'stochastic': bool,
    'differentiable': True,
    'rank_constraint': None,
    'non_markovian': False,
}
```

---

## 5. COMPATIBILITÉ ENCODINGS × GAMMAS

### 5.1 Règles de filtrage dans `plan_v8.py`

- `rank_constraint=None` : compatible avec tout `rank_eff`.
- `rank_constraint=2` : compatible uniquement si `rank_eff == 2`.
- `rank_constraint='square'` : compatible uniquement si `rank_eff == 2` et la matrice est carrée (vérifié à la volée par la forme du tenseur).

### 5.2 Tableau de compatibilité

| rank_eff | Gammas compatibles (ID)                                                                 |
|----------|-----------------------------------------------------------------------------------------|
| 2        | Tous sauf ceux avec `rank_constraint=None` (ils acceptent aussi) → tous les gammas     |
| 3        | GAM-001,003,004,005,006,008,009,010,011,015 (tous ceux avec `rank_constraint=None`)    |
| ≥4       | Idem rang 3                                                                              |

Les gammas avec `rank_constraint=2` ou `'square'` (GAM-002,007,012,013,014) sont exclus pour `rank_eff ≠ 2`.

---

## 6. AUDIT VMAP & PERFORMANCES

### 6.1 Groupes de compilation

Un groupe vmappable est défini par :
```
(gamma_id, encoding_id, n_dof, rank_eff, max_it, non_markovian)
```
où `non_markovian` détermine la version du kernel (carry standard ou étendu).

### 6.2 Nombre de compilations

- Gammas : 15
- Encodings : 20 (tous confondus)
- `n_dof` : valeurs typiques {5,10,20,50} → 4
- `rank_eff` : valeurs {2,3,4,5} → 4
- `max_it` : valeurs {50,100,200,500,1000,2000} → 6
- Versions kernel : 2 (standard / non-markovien)

Maximum théorique : 15 × 20 × 4 × 4 × 6 × 2 = 57 600 groupes.  
En pratique, les contraintes de rang réduisent ce nombre (ex. un gamma `rank_constraint=2` n'apparaîtra qu'avec `rank_eff=2`). Le nombre réel est inférieur à 10 000.

### 6.3 Différentiabilité et Hutchinson

- Les gammas `differentiable=True` supportent `jax.jacfwd` pour le calcul exact de la trace (F4). Utilisé uniquement pour `n_dof ≤ 50` pour des raisons de mémoire.
- La méthode de Hutchinson (F4.1) est disponible pour tous les gammas, y compris non différentiables.

---

## 7. RÉCAPITULATIF NUMÉRIQUE

| Catégorie | Nombre |
|-----------|--------|
| Encodings SYM | 8 |
| Encodings ASY | 6 |
| Encodings RN | 4 |
| Encodings R3 conservés | 2 |
| **Total encodings** | **20** |
| Gammas | 15 |
| Modifiers | 3 |
| **Total atomics** | **38** |

---

**FIN ATOMICS INVENTORY PRC v8**  
```
