# Catalogue Tests Observables R0

## Inventaire Rang 2

### GRA-001 - Propriétés Graphe
- **Catégorie** : `GRAPH`
- **Métriques** : `density`, `clustering_local`, `degree_variance`
- **Applicability** : `requires_rank=2`, `requires_square=True`, `minimum_dimension=5`

### PAT-001 - Diversité Distribution
- **Catégorie** : `PAT`
- **Métriques** : `diversity_simpson`, `concentration_top10`, `uniformity`
- **Applicability** : `requires_rank=None` (tout rang)

### SPA-001 - Rugosité Spatiale
- **Catégorie** : `SPATIAL`
- **Métriques** : `gradient_magnitude`, `laplacian_energy`, `smoothness`
- **Applicability** : `requires_rank=2`, `minimum_dimension=5`

### SPE-001 - Spectre Valeurs Propres
- **Catégorie** : `SPECTRAL`
- **Métriques** : `eigenvalue_max`, `spectral_gap`
- **Applicability** : `requires_rank=2`, `requires_square=True`

### SPE-002 - Rayon Spectral
- **Catégorie** : `SPECTRAL`
- **Métriques** : `spectral_radius`
- **Applicability** : `requires_rank=2`, `requires_square=True`, `minimum_dimension=3`

### SYM-001 - Conservation Symétrie
- **Catégorie** : `SYM`
- **Métriques** : `asymmetry_norm`, `asymmetry_norm_normalized`
- **Applicability** : `requires_rank=2`, `requires_square=True`, `allowed_d_types=['SYM','ASY']`

### TOP-001 - Invariants Topologiques
- **Catégorie** : `TOPOLOGICAL`
- **Métriques** : `connected_components`, `holes_count`, `euler_characteristic`
- **Applicability** : `requires_rank=2`, `minimum_dimension=10`

### UNIV-001 - Norme Frobenius
- **Catégorie** : `UNIV`
- **Métriques** : `frobenius_norm`
- **Applicability** : `requires_rank=None` (tout rang)

### UNIV-002 - Trace Normalisée
- **Catégorie** : `UNIV`
- **Métriques** : `trace_normalized`, `trace_absolute`
- **Applicability** : `requires_rank=2`, `requires_square=True`

## Table Récapitulative

| Test | Rang | Carré | Min Dim | Encodings | N Métriques |
|------|------|-------|---------|-----------|-------------|
| GRA-001 | 2 | ✅ | 5 | ALL | 3 |
| PAT-001 | Tous | ❌ | - | ALL | 3 |
| SPA-001 | 2 | ❌ | 5 | ALL | 3 |
| SPE-001 | 2 | ✅ | - | ALL | 2 |
| SPE-002 | 2 | ✅ | 3 | ALL | 1 |
| SYM-001 | 2 | ✅ | - | SYM, ASY | 2 |
| TOP-001 | 2 | ❌ | 10 | ALL | 3 |
| UNIV-001 | Tous | ❌ | - | ALL | 1 |
| UNIV-002 | 2 | ✅ | - | ALL | 2 |

## Utilisation
```python
import tests.test_SYM_001 as test_module

# Vérifier applicabilité
from tests.utilities.applicability import check
is_applicable, reason = check(test_module, run_metadata)

# Exécuter
from tests.utilities.test_engine import TestEngine
engine = TestEngine()
result = engine.execute_test(test_module, run_metadata, history, params_config_id)
```

**Source** : `tests/test_*.py`