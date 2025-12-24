# Catalogue D^(base) - Générateurs et Vérification

## Vue d'ensemble

Ce catalogue implémente **13 générateurs** de tenseurs D^(base) répartis en 3 catégories, conformément à la Charte PRC 5.1 et à la Feuille de Route R0.

### Structure

```
D_encodings/
├── __init__.py                    # Exports et helpers
├── rank2_symmetric.py             # 6 générateurs symétriques
├── rank2_asymmetric.py            # 4 générateurs asymétriques
├── rank3_correlations.py          # 3 générateurs rang 3
└── verification_tests.py          # Tests de vérification complets
```

---

## Catalogue Complet

### Rang 2 Symétrique (6 générateurs)

| ID | Fonction | Description | Propriétés clés |
|----|----------|-------------|-----------------|
| **SYM-001** | `create_identity` | Matrice identité | Diag=1, sparse, SPD |
| **SYM-002** | `create_random_uniform` | Aléatoire uniforme | U[-1,1], générique |
| **SYM-003** | `create_random_gaussian` | Aléatoire gaussienne | N(0, σ), non bornée |
| **SYM-004** | `create_correlation_matrix` | Corrélation SPD | Diag=1, définie positive |
| **SYM-005** | `create_banded` | Structure bande | Localité, sparse |
| **SYM-006** | `create_block_hierarchical` | Hiérarchique | Structure modulaire |

### Rang 2 Asymétrique (4+ générateurs)

| ID | Fonction | Description | Propriétés clés |
|----|----------|-------------|-----------------|
| **ASY-001** | `create_random_asymmetric` | Asymétrique uniforme | Générique |
| **ASY-002** | `create_lower_triangular` | Triangulaire inf. | Directionnel, sparse |
| **ASY-003** | `create_antisymmetric` | Antisymétrique | A = -A^T |
| **ASY-004** | `create_directional_gradient` | Gradient | Brisure symétrie |

### Rang 3 (3+ générateurs)

| ID | Fonction | Description | Propriétés clés |
|----|----------|-------------|-----------------|
| **R3-001** | `create_random_rank3` | Aléatoire N×N×N | Générique |
| **R3-002** | `create_partial_symmetric_rank3` | Symétrie partielle | T[i,j,k] = T[i,k,j] |
| **R3-003** | `create_local_coupling_rank3` | Couplages locaux | Sparse, localité 3-corps |

---

## Utilisation

### Import basique

```python
from D_encodings import (
    create_identity,
    create_random_uniform,
    create_banded,
    run_verification_suite
)
```

### Exemple 1: Génération simple

```python
# Générer une base symétrique
D_base = create_identity(n_dof=50)

# Générer une base avec structure
D_base = create_banded(n_dof=50, bandwidth=3, seed=42)

# Générer base rang 3
D_base = create_random_rank3(n_dof=20, seed=42)
```

### Exemple 2: Vérification des propriétés

```python
from D_encodings import create_correlation_matrix, run_verification_suite

# Générer
D_base = create_correlation_matrix(n_dof=50, seed=42)

# Vérifier
results = run_verification_suite(
    create_correlation_matrix,
    {'n_dof': 50, 'seed': 42},
    ['shape_2d', 'symmetry', 'diagonal_ones', 'positive_definite'],
    name="test_SPD"
)

# Afficher
for result in results.values():
    print(result)
```

### Exemple 3: Utilisation dans un TM

```python
from core.state_preparation import prepare_state
from core.kernel import run_kernel
from modifiers.noise import add_gaussian_noise
from operators.gamma_hyp_001 import PureSaturationGamma

# 1. Choisir base du catalogue
from D_encodings import get_generator_by_id
gen = get_generator_by_id("SYM-002")
D_base = gen(n_dof=50, seed=42)

# 2. Appliquer modifiers
D_final = prepare_state(D_base, [
    add_gaussian_noise(sigma=0.05, seed=123)
])

# 3. Exécuter avec Γ
gamma = PureSaturationGamma(beta=2.0)
for i, state in run_kernel(D_final, gamma, max_iterations=1000):
    if i % 100 == 0:
        print(f"Iteration {i}")
```

### Exemple 4: Test automatique par ID

```python
from D_encodings import (
    get_generator_by_id,
    get_recommended_tests,
    run_verification_suite
)

# Pour n'importe quel ID du catalogue
base_id = "SYM-004"
generator = get_generator_by_id(base_id)
tests = get_recommended_tests(base_id)

# Test automatique
results = run_verification_suite(
    generator,
    {'n_dof': 50, 'seed': 42},
    tests,
    name=base_id
)
```

---

## Tests de Vérification

### Tests disponibles

#### Génériques (tous tenseurs)
- `shape_2d` / `shape_3d`: Vérifier dimensions
- `bounds`: Vérifier bornes [-1, 1]
- `non_constant`: Vérifier diversité

#### Symétrie (rang 2)
- `symmetry`: A = A^T
- `antisymmetry`: A = -A^T
- `diagonal_ones`: diag(A) = 1
- `positive_definite`: A ≻ 0 (via valeurs propres)

#### Structure (rang 2)
- `sparsity_X`: X% d'éléments nuls
- `banded_X`: Structure bande de largeur X
- `triangular_lower`: Triangulaire inférieure

#### Rang 3
- `partial_symmetry_23`: T[i,j,k] = T[i,k,j]
- `full_symmetry_rank3`: Toutes permutations

#### Reproductibilité
- `reproducibility`: Même seed → mêmes résultats

### Lancer la suite complète

```bash
# Tester tout le catalogue
python -m D_encodings.example_usage --mode full

# Tester une famille
python -m D_encodings.example_usage --mode symmetric --n_dof 50

# Voir exemple d'utilisation TM
python -m D_encodings.example_usage --mode example
```

---

## Helpers

### Lister les générateurs

```python
from D_encodings import list_generators
list_generators()
```

### Accès par ID

```python
from D_encodings import get_generator_by_id

# Retourne la fonction
gen = get_generator_by_id("ASY-003")
D = gen(n_dof=50, seed=42)
```

### Tests recommandés

```python
from D_encodings import get_recommended_tests

tests = get_recommended_tests("SYM-005")
# ['shape_2d', 'symmetry', 'diagonal_ones', 'banded_3']
```

---

## Conformité PRC 5.1

### Présuppositions HORS du core

✅ Tous les générateurs documentent explicitement leurs présuppositions:
- Rang du tenseur
- Symétries
- Bornes
- Structures

✅ Le **core** (`kernel.py`, `state_preparation.py`) reste **aveugle**:
- Ne connaît ni rang ni symétrie
- Manipule des `np.ndarray` opaques

✅ Validation **à la création** uniquement:
- Tests de vérification vérifient propriétés initiales
- Aucune validation pendant exécution Γ
- Observations post-exécution dans `tests/utilities/`

### Nomenclature

✅ Terminologie conforme:
- "Tenseur rang 2" (pas "matrice de corrélation")
- "Indice (i,j)" (pas "position")
- "Itération" (pas "temps")

---

## Intégration Feuille de Route R0

### Phase 0 (Calibration)
- ✅ 6 bases SYM implémentées
- ✅ 4 bases ASY implémentées
- ✅ 3 bases R3 implémentées
- ✅ Tests de vérification complets

### Phase 1 (Balayage)
Utiliser avec 4 modificateurs standards:
```python
# M0: Base seule
D = prepare_state(D_base, [])

# M1: Base + bruit gaussien
D = prepare_state(D_base, [add_gaussian_noise(sigma=0.05)])

# M2: Base + bruit uniforme
D = prepare_state(D_base, [add_uniform_noise(amplitude=0.1)])

# M3: Base + sparsification (à implémenter dans modifiers/)
```

### Comptage total
- **13 bases** × **4 modifiers** × **5 seeds** = **260 configurations de base** par Γ

---

## Extensions Futures

### Bases additionnelles (bonus)

Le package inclut aussi:
- `create_circulant_asymmetric`: Asymétrique périodique
- `create_sparse_asymmetric`: Asymétrique sparse contrôlée
- `create_fully_symmetric_rank3`: Symétrie complète rang 3
- `create_diagonal_rank3`: Diagonal rang 3
- `create_separable_rank3`: Factorisé rang 1

### Ajout de nouveaux générateurs

1. Implémenter dans le module approprié
2. Documenter présuppositions explicites
3. Ajouter tests de vérification recommandés
4. Exporter dans `__init__.py`
5. Mettre à jour catalogue avec ID

---

## Notes Importantes

### Mémoire (rang 3)
⚠️ Les tenseurs rang 3 sont coûteux:
- N=20: 8,000 éléments (~64 Ko)
- N=30: 27,000 éléments (~216 Ko)
- N=50: 125,000 éléments (~1 Mo)

Recommandation: N ≤ 30 pour rang 3

### Reproductibilité
✅ Toujours fixer `seed` pour:
- Comparaisons entre Γ
- Debugging
- Validation résultats

### Tests avant production
⚠️ Toujours vérifier propriétés avant utilisation:
```python
# TOUJOURS faire
results = run_verification_suite(generator, params, tests, name)
if not all(r.passed for r in results.values()):
    print("WARNING: Some properties not satisfied")
```

---

## Support

- Documentation: Charte PRC 5.1, Section 3.1
- Feuille de route: Section 2 (Catalogue bases D)
- Issues: Documenter avec ID du générateur

---

**Version**: 2.0.0  
**Statut**: COMPLET pour Phase 0  
**Conforme**: PRC Charter 5.1