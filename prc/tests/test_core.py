"""
tests/test_core.py

Observations pures — core (discovery, prepare_state, run_kernel)

Règles R-TEST-1/R-TEST-2 :
- Aucun verdict (pas de assert sur comportement métier)
- Retour : dict features scalaires uniquement
- Paramètres depuis YAML (zéro hardcodé)

Usage : python -m tests.test_core (depuis prc/)
"""

import numpy as np
from pathlib import Path

from utils.data_loading_lite import (
    discover_gammas,
    discover_encodings,
    discover_modifiers,
    load_yaml,
)
from core.state_preparation import prepare_state
from core.kernel import run_kernel


# =============================================================================
# CONFIG
# =============================================================================

def load_test_config() -> dict:
    """Charge configs/tests/test_core.yaml → dict params."""
    config_path = Path(__file__).parent.parent / 'configs' / 'tests' / 'test_core.yaml'
    return load_yaml(config_path)


# =============================================================================
# TESTS
# =============================================================================

def test_discovery() -> dict:
    """
    Observe ce que discovery trouve dans atomics/.

    Retour : {
        'nb_gammas'   : int,
        'nb_encodings': int,
        'nb_modifiers': int,
        'gamma_ids'   : list[str],
        'encoding_ids': list[str],
        'modifier_ids': list[str],
    }
    """
    gammas    = discover_gammas()
    encodings = discover_encodings()
    modifiers = discover_modifiers()

    return {
        'nb_gammas'   : len(gammas),
        'nb_encodings': len(encodings),
        'nb_modifiers': len(modifiers),
        'gamma_ids'   : [g['id'] for g in gammas],
        'encoding_ids': [e['id'] for e in encodings],
        'modifier_ids': [m['id'] for m in modifiers],
    }


def test_prepare_state(config: dict) -> tuple:
    """
    Observe l'état produit par prepare_state.
    Encoding : config['encoding_id'] (SYM-007 par défaut)
    Modifier  : config['modifier_id'] (M0 par défaut)

    Retour : (state, {
        'shape': tuple,
        'dtype': str,
        'min'  : float,
        'max'  : float,
        'mean' : float,
    })
    """
    # Résoudre encoding
    encodings   = discover_encodings()
    enc_entity  = next(e for e in encodings if e['id'] == config['encoding_id'])
    enc_func    = enc_entity['callable']

    # Résoudre modifier
    modifiers    = discover_modifiers()
    mod_entity   = next(m for m in modifiers if m['id'] == config['modifier_id'])
    mod_func     = mod_entity['callable']

    # Préparer state
    state = prepare_state(
        encoding_func=enc_func,
        encoding_params={'n_dof': config['n_dof']},
        modifiers=[mod_func],
    )

    observations = {
        'shape': state.shape,
        'dtype': str(state.dtype),
        'min'  : float(np.min(state)),
        'max'  : float(np.max(state)),
        'mean' : float(np.mean(state)),
    }

    return state, observations


def test_run_kernel(config: dict, state: np.ndarray) -> dict:
    """
    Observe l'évolution produite par run_kernel.
    Gamma : config['gamma_id'] (GAM-001 par défaut)

    Retour : {
        'n_iterations' : int,
        'shape_finale' : tuple,
        'norm_initiale': float,
        'norm_finale'  : float,
    }
    """
    # Résoudre gamma
    gammas      = discover_gammas()
    gam_entity  = next(g for g in gammas if g['id'] == config['gamma_id'])
    gamma       = gam_entity['callable']()

    # Collecter history complet
    history = []
    for iteration, current_state in run_kernel(
        initial_state=state,
        gamma=gamma,
        max_iterations=config['max_iterations'],
    ):
        history.append(current_state.copy())

    return {
        'n_iterations' : len(history),
        'shape_finale' : history[-1].shape,
        'norm_initiale': float(np.linalg.norm(history[0])),
        'norm_finale'  : float(np.linalg.norm(history[-1])),
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    config = load_test_config()

    print("=== test_discovery ===")
    res = test_discovery()
    print(f"  gammas    ({res['nb_gammas']})   : {res['gamma_ids']}")
    print(f"  encodings ({res['nb_encodings']}) : {res['encoding_ids']}")
    print(f"  modifiers ({res['nb_modifiers']}) : {res['modifier_ids']}")

    print("\n=== test_prepare_state ===")
    state, res = test_prepare_state(config)
    for k, v in res.items():
        print(f"  {k}: {v}")

    print("\n=== test_run_kernel ===")
    res = test_run_kernel(config, state)
    for k, v in res.items():
        print(f"  {k}: {v}")