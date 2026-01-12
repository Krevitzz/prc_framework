# Catalogue Core R0

## Composants

### kernel.py
- **Fonction** : `run_kernel(initial_state, gamma, max_iterations, convergence_check, record_history)`
- **Responsabilité** : Itération aveugle state_{n+1} = gamma(state_n)
- **Yields** : `(iteration, state)` ou `(iteration, state, history)`

### state_preparation.py
- **Fonction** : `prepare_state(base, modifiers)`
- **Responsabilité** : Composition aveugle modifiers
- **Returns** : `np.ndarray` transformé

## Utilisation
```python
from core.kernel import run_kernel
from core.state_preparation import prepare_state

# Préparation
D = prepare_state(D_base, [modifier])

# Exécution
for i, state in run_kernel(D, gamma, max_iterations=1000):
    pass
```

**Source** : `core/`