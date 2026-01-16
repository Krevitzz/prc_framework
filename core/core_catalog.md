# core_catalog.md

> Catalogue fonctionnel du cœur d'exécution PRC  
> Responsabilité : Exécution aveugle (composition + itération)  
> Version : 6.0  
> Dernière mise à jour : 2025-01-15

---

## VUE D'ENSEMBLE

Le `core/` est **aveugle** : il ne connaît ni la dimension, ni la structure, ni l'interprétation des tenseurs qu'il manipule.

**Modules** :
- `kernel.py` : Itération aveugle state_{n+1} = γ(state_n)
- `state_preparation.py` : Composition aveugle D^(base) + modifiers

**Règles critiques** :
- ❌ Aucune validation sémantique
- ❌ Aucune classe State/Operator
- ❌ Aucun branchement dépendant de D ou γ

---

## SECTION 1 : kernel.py

### 1.1 run_kernel()

**Signature** :
```python
def run_kernel(
    initial_state: np.ndarray,
    gamma: Callable[[np.ndarray], np.ndarray],
    max_iterations: int = 10000,
    convergence_check: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None,
    record_history: bool = False
) -> Generator[Union[Tuple[int, np.ndarray], Tuple[int, np.ndarray, List[np.ndarray]]], None, None]
```

**Responsabilité** :
- Générer séquence états : state_0, state_1, ..., state_n
- Appliquer itérativement : state_{n+1} = gamma(state_n)
- **AUCUNE** décision d'arrêt autonome

**Paramètres** :
| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `initial_state` | `np.ndarray` | - | État initial (shape quelconque) |
| `gamma` | `Callable` | - | Fonction (np.ndarray → np.ndarray) |
| `max_iterations` | `int` | 10000 | Limite sécurité |
| `convergence_check` | `Callable` \| `None` | None | Fonction (state_n, state_{n+1}) → bool |
| `record_history` | `bool` | False | Si True, stocke tous états |

**Retours (yield)** :
- Si `record_history=False` : `(iteration: int, state: np.ndarray)`
- Si `record_history=True` : `(iteration: int, state: np.ndarray, history: List[np.ndarray])`

**Comportements arrêt** :
1. `max_iterations` atteint → yield dernier état
2. `convergence_check` retourne True → yield état final + return
3. TM break la boucle → arrêt immédiat

**Cas d'usage** :
```python
# Mode standard (sans historique)
for i, state in run_kernel(D_initial, gamma, max_iterations=1000):
    if detect_explosion(state):
        break

# Mode historique complet
for i, state, history in run_kernel(D_initial, gamma, record_history=True):
    pass

# Mode convergence automatique
def check_conv(s_n, s_next):
    return np.linalg.norm(s_next - s_n) < 1e-6

for i, state in run_kernel(D_initial, gamma, convergence_check=check_conv):
    pass
```

**Consommateurs** :
- `prc_automation/batch_runner.py` (mode `--brut`)
- Tests manuels (notebooks, scripts exploratoires)

**Dépendances** :
- `numpy` (standard library)

**Notes techniques** :
- Mémoire : O(1) si `record_history=False`, O(n × size(state)) sinon
- Générateur paresseux (lazy evaluation)
- Aucune copie inutile (sauf si `record_history=True`)

---

## SECTION 2 : state_preparation.py

### 2.1 prepare_state()

**Signature** :
```python
def prepare_state(
    base: np.ndarray,
    modifiers: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None
) -> np.ndarray
```

**Responsabilité** :
- Composer état final : D^(base) → modifier_1 → modifier_2 → ... → D^(final)
- Application séquentielle aveugle
- **AUCUNE** validation contenu

**Paramètres** :
| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `base` | `np.ndarray` | - | Tenseur de base (shape quelconque) |
| `modifiers` | `List[Callable]` \| `None` | None | Liste fonctions (np.ndarray → np.ndarray) |

**Retour** :
- `np.ndarray` : Tenseur composé final (même shape que base si modifiers corrects)

**Comportement** :
1. Copie `base` (évite mutation)
2. Si `modifiers` None ou vide → retourne copie base
3. Sinon, applique séquentiellement chaque modifier
4. Retourne état final

**Cas d'usage** :
```python
# Sans modificateur
D = prepare_state(base_state)

# Avec modificateurs
D = prepare_state(base_state, [
    add_gaussian_noise(sigma=0.05),
    apply_periodic_constraint()
])

# Pipeline typique
D_base = encoding_registry.get("SYM-001")(n=50)
modifiers = [
    noise.add_gaussian(sigma=0.05),
    constraints.apply_periodic()
]
D_final = prepare_state(D_base, modifiers)
```

**Consommateurs** :
- `prc_automation/batch_runner.py` (construction D avant kernel)
- Tests (préparation états initiaux)

**Dépendances** :
- `numpy` (standard library)

**Notes techniques** :
- Modifiers définis HORS core (dans `modifiers/`)
- Core ne valide RIEN sur contenu tenseurs
- Chaque modifier reçoit résultat du précédent
- Ordre modifiers est significatif (non-commutatif en général)

---

## SECTION 3 : GRAPHE DE DÉPENDANCES

### 3.1 Relations inter-modules

```
kernel.py
    ├─ Appelé par : batch_runner.py (mode --brut)
    ├─ Reçoit : initial_state (from prepare_state), gamma (from operators/)
    └─ Retourne : Generator[(int, np.ndarray) | (int, np.ndarray, List)]

state_preparation.py
    ├─ Appelé par : batch_runner.py (avant run_kernel)
    ├─ Reçoit : base (from D_encodings/), modifiers (from modifiers/)
    └─ Retourne : np.ndarray (état composé final)
```

### 3.2 Flux typique exécution

```
1. batch_runner.py
   └─> prepare_state(D_base, [M1, M2])
       └─> D_final

2. batch_runner.py
   └─> run_kernel(D_final, gamma)
       └─> yield (i, state) × max_iterations
           └─> Sauvegarde db_raw (Snapshots, Metrics)
```

---

## SECTION 4 : INVARIANTS CRITIQUES

### 4.1 Règles K1-K5 (rappel Charter)

**K1** : Core ne valide JAMAIS contenu
- ❌ Pas de `assert np.all(D >= 0)`
- ❌ Pas de `if is_symmetric(state)`

**K2** : Core ne connaît RIEN de la structure
- ❌ Pas de `if state.ndim == 2`
- ❌ Pas de `if state.shape[0] == state.shape[1]`

**K3** : Core ne branch JAMAIS sur D ou γ
- ❌ Pas de `if gamma_id == "GAM-001"`
- ❌ Pas de `if modifier_id == "M1"`

**K4** : Aucune classe State/Operator
- ✅ `np.ndarray` bruts uniquement
- ❌ Pas de `State(data=D, metadata={...})`

**K5** : Aucune interprétation sémantique
- ❌ Pas de `compute_symmetry_error(state)`
- ❌ Pas de `detect_conservation(state)`

### 4.2 Séparation stricte responsabilités

| Responsabilité | Où | Pas dans core |
|----------------|-----|---------------|
| Validation dimensionnelle | `D_encodings/` | ❌ |
| Validation sémantique | `tests/` | ❌ |
| Définition γ | `operators/` | ❌ |
| Définition modifiers | `modifiers/` | ❌ |
| Détection explosion | `tests/test_engine.py` | ❌ |
| Classification régimes | `tests/utilities/regime_utils.py` | ❌ |

---

## SECTION 5 : EXTENSIONS FUTURES

### 5.1 Extensions INTERDITES

❌ **Ajout validation dans core** :
```python
# INTERDIT
def run_kernel(...):
    if not is_valid_state(state):  # ❌
        raise ValueError(...)
```

❌ **Ajout logging métier dans core** :
```python
# INTERDIT
def prepare_state(...):
    logger.info(f"Symmetry score: {compute_symmetry(base)}")  # ❌
```

❌ **Ajout optimisations conditionnelles** :
```python
# INTERDIT
def run_kernel(...):
    if gamma_id == "GAM-001":  # ❌
        # Fast path spécifique
```

### 5.2 Extensions AUTORISÉES

✅ **Ajout paramètres techniques** :
```python
# OK (si reste aveugle)
def run_kernel(..., checkpoint_every: int = 100):
    if iteration % checkpoint_every == 0:
        yield iteration, state.copy()  # Pas d'interprétation
```

✅ **Ajout hooks génériques** :
```python
# OK (callback aveugle)
def run_kernel(..., on_iteration: Optional[Callable] = None):
    if on_iteration:
        on_iteration(iteration, state)  # Core ne sait pas ce que fait callback
```

---

## SECTION 6 : TESTS ASSOCIÉS

### 6.1 Tests unitaires core (si existants)

**Emplacement** : `tests/test_core_*.py` (à créer si pertinent)

**Scénarios minimaux** :
- `run_kernel` avec γ trivial (identité) → états stables
- `run_kernel` avec convergence_check → arrêt correct
- `prepare_state` sans modifiers → copie base
- `prepare_state` avec modifiers → application séquentielle

**Note** : Tests core doivent rester **aveugles** aussi (pas de validation sémantique)

---

## SECTION 7 : CHECKLIST AJOUT FONCTION CORE

Avant d'ajouter une fonction dans `core/` :

- [ ] Fonction est-elle vraiment aveugle ? (aucune connaissance D/γ)
- [ ] Paramètres sont-ils génériques ? (pas d'ID, pas de métier)
- [ ] Retour est-il brut ? (np.ndarray, Generator, pas de dict métier)
- [ ] Aucune validation sémantique ? (pas de assert sur contenu)
- [ ] Aucun branchement métier ? (pas de if gamma_id == ...)
- [ ] Documentée dans ce catalogue ?
- [ ] Cas d'usage identifiés ?
- [ ] Consommateurs listés ?

Si **toutes** réponses = OUI → OK ajout  
Si **une seule** = NON → fonction doit aller ailleurs (D_encodings, modifiers, tests, utilities)

---

## ANNEXE : HISTORIQUE MODIFICATIONS

| Date | Version | Changement |
|------|---------|------------|
| 2025-01-15 | 6.0.0 | Création catalogue initial |

---

**FIN core_catalog.md**