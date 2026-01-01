# Guide Référence Rapide - Architecture Registres 5.4

## Table des Matières
1. [Registres Disponibles](#registres-disponibles)
2. [Créer un Nouveau Test](#créer-un-nouveau-test)
3. [Créer un Nouveau Registre](#créer-un-nouveau-registre)
4. [Post-Processors](#post-processors)
5. [Commandes Utiles](#commandes-utiles)

---

## Registres Disponibles

### 1. Algebra Registry (`algebra`)
**Opérations algébriques fondamentales**

| Fonction | Description | Paramètres Clés |
|----------|-------------|-----------------|
| `matrix_norm` | Norme tenseur | `norm_type='frobenius'` |
| `frobenius_norm` | Norme Frobenius | - |
| `matrix_asymmetry` | Norme partie anti-symétrique | `normalize=True` |
| `trace_value` | Trace matrice | `normalize=False` |
| `determinant_value` | Déterminant | `log_scale=False` |
| `spectral_norm` | Norme spectrale | - |
| `condition_number` | Conditionnement | `norm_type='spectral'` |
| `rank_estimate` | Rang effectif | `tolerance=1e-10` |

### 2. Spectral Registry (`spectral`)
**Analyses spectrales**

| Fonction | Description | Paramètres Clés |
|----------|-------------|-----------------|
| `eigenvalue_max` | Plus grande valeur propre | `absolute=True` |
| `eigenvalue_distribution` | Stats valeurs propres | `stat='mean'` |
| `spectral_gap` | Écart λ₁ - λ₂ | `normalize=False` |
| `fft_power` | Puissance FFT | `normalize=True` |
| `fft_entropy` | Entropie spectrale | - |
| `spectral_radius` | Rayon spectral | - |

### 3. Statistical Registry (`statistical`)
**Analyses statistiques**

| Fonction | Description | Paramètres Clés |
|----------|-------------|-----------------|
| `entropy` | Entropie Shannon | `bins=50, normalize=True` |
| `kurtosis` | Aplatissement | `fisher=True` |
| `skewness` | Asymétrie | - |
| `variance` | Variance | `normalize=False` |
| `std_normalized` | Coefficient variation | - |
| `correlation_mean` | Corrélation moyenne | `axis=0, method='pearson'` |
| `sparsity` | Mesure parcimonie | `threshold=1e-6` |

### 4. Spatial Registry (`spatial`)
**Analyses spatiales**

| Fonction | Description | Paramètres Clés |
|----------|-------------|-----------------|
| `gradient_magnitude` | Magnitude gradient | `normalize=False` |
| `laplacian_energy` | Énergie laplacien | `normalize=True` |
| `local_variance` | Variance locale | `window_size=3` |
| `edge_density` | Densité contours | `threshold=0.1, method='sobel'` |
| `spatial_autocorrelation` | Autocorrélation | `lag=1` |
| `smoothness` | Mesure lissage | - |

### 5. Pattern Registry (`pattern`)
**Détection patterns**

| Fonction | Description | Paramètres Clés |
|----------|-------------|-----------------|
| `periodicity` | Périodicité | `axis=0, method='autocorr'` |
| `symmetry_score` | Score symétrie | `symmetry_type='reflection'` |
| `clustering_coefficient` | Clustering valeurs | `threshold=0.5` |
| `diversity` | Indice Simpson | `bins=50` |
| `uniformity` | Uniformité | `bins=50` |
| `concentration_ratio` | Concentration énergie | `top_percent=0.1` |

### 6. Topological Registry (`topological`)
**Invariants topologiques**

| Fonction | Description | Paramètres Clés |
|----------|-------------|-----------------|
| `connected_components` | Composantes connexes | `threshold=0.0` |
| `euler_characteristic` | Caractéristique Euler | `threshold=0.0` |
| `perimeter_area_ratio` | Ratio P/A | `threshold=0.0` |
| `compactness` | Compacité | `threshold=0.0` |
| `holes_count` | Nombre trous | `min_hole_size=4` |
| `fractal_dimension` | Dimension fractale | `num_scales=10` |

### 7. Graph Registry (`graph`)
**Métriques graphes**

| Fonction | Description | Paramètres Clés |
|----------|-------------|-----------------|
| `density` | Densité graphe | `threshold=0.0` |
| `degree_variance` | Variance degrés | `normalize=False` |
| `clustering_local` | Clustering local | `threshold=0.0` |
| `average_path_length` | Longueur chemin | `sample_fraction=0.1` |
| `centrality_concentration` | Concentration centralité | - |
| `small_world_coefficient` | Coefficient petit-monde | - |

---

## Créer un Nouveau Test

### Template Minimal
#templates/template_test_category_nnn.py

### Checklist Création Test

- [ ] Docstring complète (objectif + métriques + exclusions)
- [ ] `TEST_ID` format "CAT-NNN"
- [ ] `TEST_VERSION = "5.4"`
- [ ] `APPLICABILITY_SPEC` défini
- [ ] `COMPUTATION_SPECS` : 1 à 5 métriques
- [ ] Chaque métrique a `registry_key` + `default_params`
- [ ] `registry_key` existe dans registres
- [ ] `post_process` est clé valide (si utilisé)
- [ ] Pas de `FORMULAS` legacy

---

## Créer un Nouveau Registre

### Template Minimal

# templates/template_mydomain_registry.py

### Checklist Création Registre

- [ ] Nom fichier : `*_registry.py`
- [ ] Hérite de `BaseRegistry`
- [ ] `REGISTRY_KEY` unique défini
- [ ] Fonctions décorées `@register_function`
- [ ] Signature : `func(state, **params) -> float`
- [ ] Tous params ont valeurs par défaut
- [ ] Docstring complète (Args, Returns, Raises, Notes, Examples)
- [ ] Validation entrées explicite
- [ ] Fonction pure (pas d'effets de bord)
- [ ] `return float(...)` systématique

---

## Post-Processors

### Post-Processors Disponibles

```python
# Identité
'identity': lambda x: x

# Arrondi
'round_2': lambda x: round(float(x), 2)
'round_4': lambda x: round(float(x), 4)
'round_6': lambda x: round(float(x), 6)

# Valeur absolue
'abs': lambda x: abs(float(x))

# Logarithmique
'log': lambda x: float(np.log(x + 1e-10))
'log10': lambda x: float(np.log10(x + 1e-10))
'log1p': lambda x: float(np.log1p(x))

# Clipping
'clip_01': lambda x: float(np.clip(x, 0, 1))
'clip_positive': lambda x: float(max(0, x))

# Notation scientifique
'scientific_3': lambda x: float(f"{x:.3e}")
```

### Ajouter Post-Processor Custom

```python
# Dans tests/utilities/registries/post_processors.py
from .post_processors import add_post_processor

add_post_processor('sigmoid', lambda x: 1 / (1 + np.exp(-x)))
add_post_processor('tanh', lambda x: np.tanh(x))
```

---

## Commandes Utiles

### Validation

```bash
# Lister registres chargés
python -c "
from tests.utilities.registries.registry_manager import RegistryManager
rm = RegistryManager()
print(rm.list_available_functions())
"

# Lister tests actifs
python -c "
from tests.utilities.discovery import discover_active_tests
print(list(discover_active_tests().keys()))
"

# Valider test
python -c "
from tests import test_univ_001
from tests.utilities.applicability import check
metadata = {
    'gamma_id': 'GAM-001',
    'd_base_id': 'SYM-001',
    'modifier_id': 'M0',
    'seed': 42,
    'state_shape': (10, 10)
}
print(check(test_univ_001, metadata))
"
```

### Test Registres

```bash
# Validation complète
python tests/test_registries_validation.py

# Verbose
python tests/test_registries_validation.py --verbose
```

### Workflow Batch

```bash
# 1. Initialiser bases
python prc_automation/init_databases.py

# 2. Collecte données brutes
python batch_runner.py --brut --gamma GAM-001

# 3. Appliquer tests
python batch_runner.py --test --gamma GAM-001 --params default_v1

# 4. Calculer verdicts
python batch_runner.py --verdict --gamma GAM-001 \
    --params default_v1 \
    --scoring default_v1 \
    --thresholds strict_v1

# 5. Pipeline complet
python batch_runner.py --all --gamma GAM-001
```

---

## Exemples d'Usage

### Exemple 1 : Test Simple (1 métrique)

```python
TEST_ID = "SIMPLE-001"
COMPUTATION_SPECS = {
    'norm': {
        'registry_key': 'algebra.frobenius_norm',
        'default_params': {},
        'post_process': 'round_4',
    }
}
```

### Exemple 2 : Test Multi-Métriques

```python
TEST_ID = "MULTI-001"
COMPUTATION_SPECS = {
    'asymmetry_raw': {
        'registry_key': 'algebra.matrix_asymmetry',
        'default_params': {'normalize': False},
    },
    'asymmetry_normalized': {
        'registry_key': 'algebra.matrix_asymmetry',
        'default_params': {'normalize': True},
        'post_process': 'round_6',
    },
    'trace': {
        'registry_key': 'algebra.trace_value',
        'default_params': {'normalize': True},
        'post_process': 'round_4',
    }
}
```

### Exemple 3 : Override YAML

```yaml
# tests/config/tests/UNIV-001/params_custom_v1.yaml
version: "1.0"
config_id: "UNIV-001_params_custom_v1"

# Override params globaux
explosion_threshold: 500.0
stability_tolerance: 0.05

# Override métrique spécifique
frobenius_norm:
  post_process: 'log'  # Change post-processing
```

---

## Dépannage

### Erreur : "registry_key not found"
```python
# Vérifier chargement
rm = RegistryManager()
print(rm.list_available_functions())
# Si registre manquant → vérifier nom fichier *_registry.py
```

### Erreur : "Paramètre invalide"
```python
# Vérifier signature fonction
import inspect
func = rm.get_function('algebra.matrix_norm')
print(inspect.signature(func))
```

### Erreur : "État incompatible"
```python
# Vérifier APPLICABILITY_SPEC
# requires_rank, requires_square, etc.
```

### Test pas découvert
```python
# Vérifier :
# 1. Nom fichier test_*.py
# 2. Pas dans *_deprecated.py
# 3. Structure valide (TEST_ID, COMPUTATION_SPECS, etc.)
```

---

## Ressources

- **Charter complet** : `charter 5.4 mini.txt`
- **Templates** : `prc_documentation/templates/`
- **Validation** : `tests/test_registries_validation.py`
- **Exemples** : `tests/test_*_001.py`

---

**Version** : 5.4  
**Date** : 2025-01-01  
**Statut** : Production