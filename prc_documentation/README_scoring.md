# Guide Scoring Pathologies Charter 5.4

## Structure Fichiers
```
tests/config/
├── global/
│   ├── params_default_v1.yaml         # Paramètres tests
│   ├── scoring_pathologies_v1.yaml    # Scoring baseline
│   ├── scoring_conservative_v1.yaml   # Variante permissive
│   └── scoring_strict_v1.yaml         # Variante stricte
│
└── tests/
    ├── UNIV-001/
    │   └── scoring_custom_v1.yaml     # Override UNIV-001
    └── SYM-001/
        └── scoring_custom_v1.yaml     # Override SYM-001
```

## Principe Scoring

**Scores [0,1] détectent pathologies, pas qualités**

- `0.0` = Aucun signal pathologique (sain)
- `1.0` = Pathologie maximale (toxique)

### Types Pathologies

| Type | Description | Exemple |
|------|-------------|---------|
| **S1_COLLAPSE** | Valeur trop faible | `entropy < 0.1` (perte diversité) |
| **S2_EXPLOSION** | Valeur trop élevée | `norm > 1000` (divergence) |
| **S3_PLATEAU** | Intervalle toxique | `uniformity ∈ [0.9, 1.0]` (trop uniforme) |
| **S4_INSTABILITY** | Variation rapide | `Δmetric > threshold` |
| **MAPPING** | Catégoriel | `transition = "explosive" → 1.0` |

## Format Règle Scoring
```yaml
metric_name:
  # OBLIGATOIRE
  source: "statistics.metric_name.final"  # Path extraction
  pathology_type: "S2_EXPLOSION"          # Type pathologie
  
  # Seuils (selon type)
  threshold_high: 1000.0                  # Seuil pathologie
  critical_high: 10000.0                  # Seuil critique
  
  # Metadata
  weight: 2.0                             # Criticité (défaut 1.0)
  mode: "soft"                            # soft | hard (défaut soft)
```

### Source Paths

| Format | Exemple | Description |
|--------|---------|-------------|
| `statistics.{metric}.final` | `statistics.norm.final` | Valeur finale |
| `statistics.{metric}.mean` | `statistics.norm.mean` | Moyenne temporelle |
| `statistics.{metric}.std` | `statistics.norm.std` | Écart-type temporel |
| `evolution.{metric}.transition` | `evolution.norm.transition` | Transition qualitative |
| `evolution.{metric}.trend` | `evolution.norm.trend` | Tendance |
| `evolution.{metric}.slope` | `evolution.norm.slope` | Coefficient tendance |

## Calibrage Seuils

### Stratégie Initiale

1. **Baseline exploratoire** : `scoring_pathologies_v1.yaml`
2. **Exécuter batch** : `--test --scoring pathologies_v1`
3. **Analyser patterns** : `--verdict`
4. **Identifier problèmes** :
   - `non_discriminant` → Augmenter seuils
   - `over_discriminant` → Réduire seuils

### Exemple Calibrage
```yaml
# Avant (non_discriminant détecté)
frobenius_norm:
  threshold_high: 1000.0
  # Résultat: max(scores) = 0.05 (jamais > 0.1)

# Après (calibrage)
frobenius_norm:
  threshold_high: 500.0      # Divisé par 2
  critical_high: 5000.0
  # Résultat: max(scores) = 0.42 (discriminant)
```

## Agrégation Modes

| Mode | Formule | Usage |
|------|---------|-------|
| `max` | `max(scores)` | **R0 default** : Une pathologie suffit |
| `weighted_mean` | `Σ(score×weight) / Σ(weight)` | R1+ : Compensation possible |
| `weighted_max` | `max(score × weight/max_weight)` | R1+ : Max pondéré |

## Création Nouveau Test

1. **Créer test** : `tests/test_xxx_001.py`
2. **Ajouter scoring** dans `scoring_pathologies_v1.yaml` :
```yaml
XXX-001:
  test_weight: 1.0
  aggregation_mode: "max"
  
  scoring_rules:
    metric_name:
      source: "statistics.metric_name.final"
      pathology_type: "S2_EXPLOSION"
      threshold_high: 1000.0
      weight: 2.0
```

3. **Tester** : `python batch_runner.py --test --gamma GAM-001`
4. **Analyser** : Vérifier patterns (discriminant, calibré)

## Override Spécifique

Créer `tests/config/tests/TEST-ID/scoring_custom_v1.yaml` :
```yaml
version: "1.0"
config_id: "TEST-001_scoring_custom_v1"
test_id: "TEST-001"

# Override seuils
metric_name:
  threshold_high: 500.0  # vs 1000.0 global
```

Usage : `--scoring TEST-001_scoring_custom_v1`

## Debugging

### Score toujours 0
```bash
# Vérifier extraction
python -c "
from tests.utilities.scoring import extract_value_from_observation
obs = {...}  # Charger observation
val = extract_value_from_observation(obs, 'metric', 'statistics.metric.final')
print(val)
"

# → Si KeyError : source invalide
# → Si val faible : seuils trop élevés
```

### Score toujours 1
```bash
# Seuils trop stricts
# Augmenter threshold_high ou threshold_low
```

## Exemples Patterns

### Pattern 1 : Non discriminant
```
Rapport verdict:
  Patterns: non_discriminant: ['UNIV-001.stat_min']
  Action: Augmenter threshold_low de 0.01 à 0.1
```

### Pattern 2 : D-correlated
```
Rapport verdict:
  Patterns: d_correlated: UNIV-001.norm
    - SYM-001: mean=0.85 (échec)
    - ASY-001: mean=0.12 (ok)
  Action: Γ spécialisé, investiguer SYM-001
```

### Pattern 3 : Systematic failure
```
Rapport verdict:
  Patterns: systematic_failures: ['UNIV-001.norm']
  Verdict: REJECTED[R0]
  Raison: Pathologie systématique norm > 0.7 sur 87% runs
```

## Références

- **Charter 5.4** : Section 12.8 (Scoring)
- **Types pathologies** : S1-S4 + MAPPING
- **Verdict patterns** : Section 12.9
- **Template test** : `prc_documentation/templates/test_xxx_yyy.py`