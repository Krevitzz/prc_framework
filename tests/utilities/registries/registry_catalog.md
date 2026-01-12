# Catalogue Registries (Opérateurs de Calcul)

## Vue d'ensemble

Registres de fonctions de calcul organisés par domaine. Chaque fonction prend `state: np.ndarray` et retourne `float`.

---

## Algebra Registry

**Clé** : `algebra`

### Fonctions

- `matrix_asymmetry` : Norme partie anti-symétrique
- `matrix_norm` : Norme tenseur (frobenius, spectral, nuclear)
- `frobenius_norm` : Norme Frobenius (alias)
- `trace_value` : Trace matrice
- `determinant_value` : Déterminant
- `spectral_norm` : Norme spectrale
- `condition_number` : Conditionnement matrice
- `rank_estimate` : Estimation rang effectif

---

## Graph Registry

**Clé** : `graph`

### Fonctions

- `density` : Densité graphe
- `degree_variance` : Variance degrés
- `clustering_local` : Clustering local moyen
- `average_path_length` : Longueur chemin moyenne
- `centrality_concentration` : Concentration centralité
- `small_world_coefficient` : Coefficient petit-monde

---

## Pattern Registry

**Clé** : `pattern`

### Fonctions

- `periodicity` : Détection périodicité
- `symmetry_score` : Score symétrie
- `clustering_coefficient` : Coefficient clustering
- `diversity` : Diversité (indice Simpson)
- `uniformity` : Uniformité distribution
- `concentration_ratio` : Ratio concentration

---

## Spatial Registry

**Clé** : `spatial`

### Fonctions

- `gradient_magnitude` : Magnitude gradient moyen
- `laplacian_energy` : Énergie laplacien
- `local_variance` : Variance locale
- `edge_density` : Densité contours
- `spatial_autocorrelation` : Autocorrélation spatiale
- `smoothness` : Mesure lissage

---

## Spectral Registry

**Clé** : `spectral`

### Fonctions

- `eigenvalue_max` : Plus grande valeur propre
- `eigenvalue_distribution` : Statistiques spectre
- `spectral_gap` : Écart spectral
- `fft_power` : Puissance fréquentielle
- `fft_entropy` : Entropie spectrale
- `spectral_radius` : Rayon spectral

---

## Statistical Registry

**Clé** : `statistical`

### Fonctions

- `entropy` : Entropie Shannon
- `kurtosis` : Aplatissement distribution
- `skewness` : Asymétrie distribution
- `variance` : Variance
- `std_normalized` : Coefficient variation
- `correlation_mean` : Corrélation moyenne
- `sparsity` : Mesure parcimonie

---

## Topological Registry

**Clé** : `topological`

### Fonctions

- `connected_components` : Nombre composantes connexes
- `euler_characteristic` : Caractéristique Euler
- `perimeter_area_ratio` : Ratio périmètre/aire
- `compactness` : Compacité formes
- `holes_count` : Estimation nombre trous
- `fractal_dimension` : Dimension fractale

---

## Utilisation

### Appel Direct
```python
from tests.utilities.registries.registry_manager import RegistryManager

manager = RegistryManager()
func = manager.get_function('algebra.matrix_norm')
result = func(state, norm_type='frobenius')
```

### Via Test Engine
```python
COMPUTATION_SPECS = {
    'asymmetry_norm': {
        'registry_key': 'algebra.matrix_asymmetry',
        'default_params': {
            'norm_type': 'frobenius',
            'normalize': False,
        },
        'post_process': 'round_6',
    }
}
```

### Post-Processors Disponibles

- `identity` : Aucune transformation
- `round_2`, `round_4`, `round_6` : Arrondi
- `abs` : Valeur absolue
- `log`, `log10`, `log1p` : Logarithmique
- `clip_01`, `clip_positive` : Écrêtage
- `scientific_3` : Notation scientifique

---

## Table Récapitulative

| Registry | N Fonctions | Rang Requis | Applicabilité |
|----------|-------------|-------------|---------------|
| algebra | 8 | 2 (majorité) | Matrices carrées |
| graph | 6 | 2 | Matrices carrées |
| pattern | 6 | Tous | Universel |
| spatial | 6 | 2-3 | Structures spatiales |
| spectral | 6 | 2 | Matrices carrées |
| statistical | 7 | Tous | Universel |
| topological | 6 | 2 | Structures 2D |

---

**Source** : `tests/utilities/registries/*_registry.py`