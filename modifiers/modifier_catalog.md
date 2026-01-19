# MODIFIER CATALOG

> Transformations D appliquées sur tenseur base  
> Composition : D^(base) + modifiers → D^(final) 

**RAPPEL** : Modifiers sont des fonctions `(np.ndarray → np.ndarray)` passthrough

## MODIFIERS DISPONIBLES

### M0 - Baseline
**Fichier** : (implicite, aucun modifier appliqué)  
**Transformation** : Identité (D^(final) = D^(base))  
**Usage** : `prepare_state(base, modifiers=None)` ou `prepare_state(base, [])`  
**Propriétés** : Aucune modification, référence pure

### M1 - Bruit Gaussien
**Fichier** : `noise.py::add_gaussian_noise()`  
**Transformation** : D' = D + N(0, σ)  
**Paramètres** : 
- `sigma=0.05` (défaut R0)
- `seed` (optionnel, reproductibilité)

**Propriétés** :
- Bruit centré, amplitude σ
- Distribution normale
- Préserve structure moyenne

**Usage typique** :
```python
modifier = add_gaussian_noise(sigma=0.05, seed=42)
D_final = prepare_state(D_base, [modifier])
```

### M2 - Bruit Uniforme
**Fichier** : `noise.py::add_uniform_noise()`  
**Transformation** : D' = D + U[-amplitude, +amplitude]  
**Paramètres** : 
- `amplitude=0.1` (défaut R0)
- `seed` (optionnel, reproductibilité)

**Propriétés** :
- Bruit uniforme, bornes [-amplitude, +amplitude]
- Distribution équiprobable
- Perturbation non gaussienne

**Usage typique** :
```python
modifier = add_uniform_noise(amplitude=0.1, seed=42)
D_final = prepare_state(D_base, [modifier])
```

## PATTERN FACTORY

**Tous les modifiers suivent le pattern Factory** :
```python
def modifier_factory(param1, param2, ..., seed=None) -> Callable:
    """Factory retournant fonction modifier."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    
    def modifier(state: np.ndarray) -> np.ndarray:
        """Applique transformation sur state."""
        # Transformation
        return transformed_state
    
    return modifier
```

**Justification** :
- Permet paramétrisation (sigma, amplitude, etc.)
- Encapsulation RNG (reproductibilité via seed)
- Compatible signature `prepare_state(base, [modifier1, modifier2, ...])`

## CONTRAINTES TOPOLOGIQUES (Non implémentées R0)

**Extensions futures anticipées** :
- `modifiers/constraints/periodic.py` : Contraintes périodiques
- `modifiers/constraints/bounded.py` : Clipping bornes
- `modifiers/plugins/` : Transformations domaine-spécifiques
- `modifiers/domains/` : Contraintes par domaine

## COMPOSITION SÉQUENTIELLE

**Ordre d'application** :
```python
D_final = prepare_state(D_base, [modifier1, modifier2, modifier3])
# Équivalent à :
# state = D_base.copy()
# state = modifier1(state)
# state = modifier2(state)
# state = modifier3(state)
# D_final = state
```

**Propriétés** :
- Non commutatif (ordre compte)
- Chaque modifier reçoit résultat du précédent
- Core applique aveuglément (aucune validation)

## VALIDATION

**Modifiers DOIVENT** :
- Retourner `np.ndarray` de même shape que input
- Être déterministes si `seed` fourni
- Ne pas lever d'exception (sauf erreur critique)

**Validation effectuée HORS du core** (responsabilité modifier)

## DÉPENDANCES

**Autorisées** : NumPy uniquement  
**Interdites** : 
- core/ (pas de dépendance circulaire)
- operators/, D_encodings/, tests/, utilities/

## NOMENCLATURE

**IDs modifier** :
- `M0` : Baseline (implicite)
- `M1`, `M2`, ... : Modifiers explicites
- Format : `M{N}` où N est séquentiel

**Conventions nommage fonctions** :
- Factory : `add_{type}_noise()`, `apply_{constraint}()`
- Retour : Fonction anonyme `modifier(state)`

## NOTES ARCHITECTURALES

**Séparation stricte** :
- Modifiers transforment D (D → D')
- Encodings créent D^(base)
- Core compose séquentiellement (prepare_state)
- Tests observent résultat

**Principe** : Modifiers sont **transformations pures** sans connaissance contexte (gamma, test, etc.)

## EXTENSIONS FUTURES

**Checklist ajout nouveau modifier** :
- [ ] Suivre pattern Factory
- [ ] Paramètre `seed` pour reproductibilité
- [ ] Retourner `np.ndarray` shape identique
- [ ] Documenter transformation mathématique
- [ ] Ajouter ID séquentiel (M3, M4, ...)
- [ ] Mettre à jour ce catalogue

**Exemples extensions acceptables** :
- ✅ Bruit corrélé spatialement
- ✅ Clipping bornes [-1, 1]
- ✅ Normalisation (rescaling)
- ✅ Projection sur sous-espace

**Exemples extensions REFUSÉES** :
- ❌ Validation propriétés (symétrie, etc.) — Responsabilité encodings
- ❌ Adaptation selon gamma — Violé aveuglement core
- ❌ Optimisation état — Pas une transformation passive

**FIN MODIFIER CATALOG**