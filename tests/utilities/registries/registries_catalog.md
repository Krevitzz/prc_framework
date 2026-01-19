# REGISTRIES CATALOG

> Fonctions pures de calcul (state → float)  
> Zéro dépendance interne PRC  

## ARCHITECTURE REGISTRES

### BaseRegistry (base_registry.py)
**Responsabilité** : Classe abstraite pour tous registres  
**Pattern** : Découverte automatique via décorateur `@register_function`

**Méthodes publiques** :
- `get_function(registry_key)` : Récupère fonction par clé complète
- `list_functions()` : Liste toutes fonctions avec documentation
- `_register_all_functions()` : Découverte automatique (privée)

**Contraintes** :
- REGISTRY_KEY doit être défini et unique
- Fonctions décorées avec `@register_function(name)`
- Signature : `func(self, state, **params) -> float`

---

### RegistryManager (registry_manager.py)
**Responsabilité** : Singleton gérant tous registres  
**Architecture** : Chargement dynamique, validation, cache

**Méthodes publiques** :
- `get_function(registry_key)` : Récupère fonction (avec cache)
- `validate_computation_spec(spec)` : Valide COMPUTATION_SPECS
- `list_available_functions()` : Liste toutes fonctions par registre
- `_load_all_registries()` : Chargement automatique (privée)
- `_validate_params(function, user_params)` : Validation paramètres (privée)

## RÈGLES CONCEPTION FONCTIONS

**R-REG-1** : Signature standard
```python
def compute_metric(
    self,
    state: np.ndarray,
    param1: type = default,
    param2: type = default,
    epsilon: float = 1e-10
) -> float:
    """Docstring avec Args, Returns, Raises, Notes, Examples."""
```

**R-REG-2** : Validation applicabilité
```python
if state.ndim != expected_rank:
    raise ValueError(f"Attendu {expected_rank}D, reçu {state.ndim}D")
```

**R-REG-3** : Robustesse numérique
```python
# Protection division par zéro
result = numerator / (denominator + epsilon)

# Protection log
result = np.log(value + epsilon)

# Gestion valeurs non finies
if not np.isfinite(result):
    return default_value
```

**R-REG-4** : Retour type strict
```python
return float(result)  # Toujours float, jamais np.float64
```

**R-REG-5** : Décorateur obligatoire
```python
@register_function("function_name")
def compute_metric(...):
    pass
```

## PARAMÈTRES COMMUNS

**Paramètres fréquents** :
- `threshold: float` : Seuil binarisation/filtrage
- `normalize: bool` : Normaliser résultat ?
- `epsilon: float = 1e-10` : Protection division/log
- `bins: int` : Nombre bins histogramme
- `axis: int` : Axe analyse (multi-dim)
- `method: str` : Variante algorithme

**Conventions** :
- `epsilon` toujours optionnel avec défaut `1e-10`
- `normalize` par défaut `False` (sauf si justifié)
- Paramètres booléens nommés explicitement (`absolute`, `fisher`, etc.)

## DÉPENDANCES

**Autorisées** :
- `numpy` : Opérations numériques
- `scipy` : Algorithmes spécialisés (stats, ndimage, sparse)
- `base_registry` : Classe parente

**Interdites** :
- Tout module PRC (core/, operators/, tests/, utilities/HUB, utilities/UTIL)
- Imports circulaires entre registres
- Dépendances bibliothèques lourdes (sklearn, torch, etc.)

## EXTENSIONS FUTURES

**Checklist ajout nouvelle fonction** :
- [ ] Identifier registre approprié (ou créer nouveau)
- [ ] Définir signature standard (state, **params) → float
- [ ] Implémenter validation applicabilité (rang, dimension)
- [ ] Ajouter docstring complète (Args, Returns, Raises, Notes, Examples)
- [ ] Décorer avec `@register_function("name")`
- [ ] Tester robustesse numérique (epsilon, np.isfinite)
- [ ] Ajouter à table "AVANT DE CODER" ce catalogue
- [ ] Mettre à jour FUNCTIONS_INDEX.md

**Checklist création nouveau registre** :
- [ ] Hériter de `BaseRegistry`
- [ ] Définir `REGISTRY_KEY` unique
- [ ] Implémenter fonctions avec `@register_function`
- [ ] Créer fichier `{category}_registry.py`
- [ ] RegistryManager chargera automatiquement
- [ ] Ajouter section à ce catalogue
- [ ] Mettre à jour PATTERNS.md avec best practices

## NOTES ARCHITECTURALES

**Découverte automatique** :
- RegistryManager scan `tests/utilities/registries/*_registry.py`
- Instancie classes héritant de `BaseRegistry`
- Enregistre via `REGISTRY_KEY` unique

**Cache performances** :
- `RegistryManager.function_cache` : Évite lookups répétés
- `TestEngine.computation_cache` : Évite validations répétées

**Singleton RegistryManager** :
- Une seule instance par processus
- Chargement registres à l'initialisation
- Partage entre tous tests

**Validation paramètres** :
- Introspection signature via `inspect.signature()`
- Vérification types annotations
- Conversion automatique si possible
- Completion defaults manquants

**FIN REGISTRIES CATALOG**