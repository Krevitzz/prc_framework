"""
prc.running.runner

Responsabilité : Exécution run unique : prepare_state → run_kernel → history np.ndarray

Gestion erreurs inspirée legacy batch_runner.py (lignes 466-502)
"""

import numpy as np
from collections import defaultdict
from typing import Dict
import warnings


from core.state_preparation import prepare_state
from core.kernel import run_kernel
from featuring.hub_featuring import get_active_layers, measure_state, extract_from_signals


class RunnerError(Exception):
    """Erreur durant exécution run."""
    pass


def run_single(
    composition     : Dict,
    featuring_config: Dict,
    verbose         : bool = False,
) -> Dict:
    """
    Exécute un run unique en mode streaming — history jamais matérialisé en RAM.

    Args:
        composition      : Dict depuis generate_compositions()
        featuring_config : Dict depuis load_all_configs()
        verbose          : Si True, affiche warnings explosion

    Returns:
        {
            'features': Dict[str, float],
            'layers'  : List[str],
        }

    Raises:
        RunnerError : Si erreur kernel, explosion non récupérable, ou état initial invalide

    Notes:
        - History jamais matérialisé — signaux 1D accumulés uniquement
        - Gain RAM : n_functions × max_iterations × 8 octets vs (T × *dims) × 8 octets
        - Gamma non-mutant garanti par convention atomics_catalog
        - last_state = state sans .copy() — kernel rebind, pas mutation in-place
    """
    # Extraire métadonnées pour messages erreurs
    gamma_id    = composition['gamma_id']
    encoding_id = composition['encoding_id']
    modifier_id = composition['modifier_id']
    combo_str   = f"{gamma_id} × {encoding_id} × {modifier_id}"

    # 1. Préparer état initial via core
    try:
        D_initial = prepare_state(
            encoding_func=composition['encoding_callable'],
            encoding_params=composition['encoding_params'],
            modifiers=[composition['modifier_callable']],
            modifier_configs={
                composition['modifier_callable']: composition['modifier_params']
            }
        )
    except Exception as e:
        raise RunnerError(
            f"Erreur prepare_state : {combo_str}\n"
            f"  Détails : {e}"
        ) from e

    if not isinstance(D_initial, np.ndarray):
        raise RunnerError(
            f"prepare_state n'a pas retourné np.ndarray : {combo_str}\n"
            f"  Type reçu : {type(D_initial)}"
        )

    if D_initial.size == 0:
        raise RunnerError(
            f"prepare_state a retourné tenseur vide : {combo_str}\n"
            f"  Shape : {D_initial.shape}"
        )

    # 2. Layers applicables depuis shape D_initial (une fois, avant boucle)
    active_layers = get_active_layers(D_initial, featuring_config)

    # 3. Préparer gamma
    gamma_callable = composition['gamma_callable']
    max_iterations = composition['max_iterations']

    if hasattr(gamma_callable, 'reset'):
        gamma_callable.reset()

    # 4. Boucle streaming — history jamais matérialisé
    signals        = defaultdict(list)
    has_nan_inf    = False
    last_state     = None
    last_iteration = -1
    n_states       = 0

    try:
        for iteration, state in run_kernel(D_initial, gamma_callable, max_iterations):
            last_iteration = iteration

            # Détection explosion → flag + arrêt propre
            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                has_nan_inf = True
                if verbose:
                    warnings.warn(
                        f"Explosion détectée iteration {iteration} : {combo_str}\n"
                        f"  NaN : {np.any(np.isnan(state))}, "
                        f"Inf : {np.any(np.isinf(state))}\n"
                        f"  Arrêt anticipé — {n_states} états mesurés"
                    )
                break

            # Mesure état courant → accumulation signaux
            measures = measure_state(state, active_layers, featuring_config)
            for k, v in measures.items():
                signals[k].append(v)

            # Dernier état — pas de .copy() (kernel rebind, gamma non-mutant)
            last_state = state
            n_states  += 1

    except Exception as e:
        raise RunnerError(
            f"Kernel crash iteration {last_iteration} : {combo_str}\n"
            f"  États mesurés : {n_states}\n"
            f"  Erreur : {e}"
        ) from e

    # 5. Validation
    if n_states == 0:
        raise RunnerError(
            f"Aucun état capturé : {combo_str}\n"
            f"  Dernière iteration : {last_iteration}\n"
            f"  Vérifier : encoding génère tenseur valide, gamma compatible"
        )

    # 6. Extract features depuis signaux accumulés
    features = extract_from_signals(
        dict(signals), has_nan_inf, last_state, active_layers, featuring_config
    )

    return {
        'features': features,
        'layers'  : active_layers,
    }