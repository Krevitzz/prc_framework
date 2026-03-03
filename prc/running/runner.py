"""
prc.running.runner

Responsabilité : Exécution run unique : prepare_state → run_kernel → history np.ndarray

Gestion erreurs inspirée legacy batch_runner.py (lignes 466-502)
"""

import numpy as np
from typing import Dict
import warnings


from core.state_preparation import prepare_state
from core.kernel import run_kernel


class RunnerError(Exception):
    """Erreur durant exécution run."""
    pass


def run_single(composition: Dict, verbose: bool = False) -> np.ndarray:
    """
    Exécute un run unique.
    
    Args:
        composition : Dict depuis generate_compositions()
                      Contient : gamma_id, encoding_id, modifier_id,
                                gamma_callable, encoding_callable, modifier_callable,
                                gamma_params, encoding_params, modifier_params,
                                max_iterations, n_dof, phase
    
    Returns:
        np.ndarray history (n_iterations+1, *state_shape)
        Exemple : shape (201, 10, 10) pour matrice 10×10 sur 200 iterations
    
    Raises:
        RunnerError : Si erreur kernel, explosion non récupérable, ou validation échoue
    
    Notes:
        - prepare_state : encodings + modifiers → D_initial
        - run_kernel : itération gamma → history
        - Totalement aveugle au contenu des tenseurs
        - Détection explosion (NaN/Inf) → break propre avec warning
    
    Examples:
        >>> from running.compositions import generate_compositions, load_run_config
        >>> config = load_run_config(Path('configs/phases/poc/poc.yaml'))
        >>> compositions = generate_compositions(config)
        >>> history = run_single(compositions[0])
        >>> history.shape
        (11, 10, 10)  # 10 iterations + état initial
    """
    # Extraire métadonnées pour messages erreurs
    gamma_id = composition['gamma_id']
    encoding_id = composition['encoding_id']
    modifier_id = composition['modifier_id']
    combo_str = f"{gamma_id} × {encoding_id} × {modifier_id}"
    
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
    
    # Validation shape D_initial
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
    
    # 2. Préparer gamma
    gamma_callable = composition['gamma_callable']
    max_iterations = composition['max_iterations']
    
    # Reset mémoire si gamma non-markovien (legacy batch_runner.py:457-458)
    if hasattr(gamma_callable, 'reset'):
        gamma_callable.reset()
    
    # 3. Exécuter kernel avec capture history
    history = []
    last_iteration = -1
    
    try:
        for iteration, state in run_kernel(D_initial, gamma_callable, max_iterations):
            last_iteration = iteration
            
            # Détection explosion (NaN/Inf) — legacy batch_runner.py:477-479
            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                if verbose:
                    warnings.warn(
                        f"Explosion détectée iteration {iteration} : {combo_str}\n"
                        f"  NaN présents : {np.any(np.isnan(state))}\n"
                        f"  Inf présents : {np.any(np.isinf(state))}\n"
                        f"  Arrêt anticipé — {len(history)} états capturés"
                    )
                break  # Arrêt propre
            
            # Capture état (snapshot legacy batch_runner.py:474)
            history.append(state.copy())
    
    except Exception as e:
        raise RunnerError(
            f"Kernel crash iteration {last_iteration} : {combo_str}\n"
            f"  États capturés : {len(history)}\n"
            f"  Erreur : {e}"
        ) from e
    
    # 4. Validation history finale (legacy batch_runner.py:488-501)
    if len(history) == 0:
        raise RunnerError(
            f"History vide (aucun état capturé) : {combo_str}\n"
            f"  Dernière iteration : {last_iteration}\n"
            f"  Vérifier : encoding génère tenseur valide, gamma compatible"
        )
    
    # Validation cohérence shapes
    reference_shape = history[0].shape
    for i, snapshot in enumerate(history):
        if snapshot.shape != reference_shape:
            raise RunnerError(
                f"Shape snapshot {i} incohérente : {combo_str}\n"
                f"  Attendu : {reference_shape}\n"
                f"  Reçu    : {snapshot.shape}"
            )
    
    # 5. Convertir en np.ndarray
    # Note : np.array() fait déjà validation shapes homogènes
    try:
        history_array = np.array(history)
    except Exception as e:
        raise RunnerError(
            f"Conversion history → np.ndarray échouée : {combo_str}\n"
            f"  Erreur : {e}"
        ) from e
    
    return history_array
