# prc_automation/batch_runner.py

"""
Orchestration exhaustive exécutions R0.

Usage:
    python batch_runner.py --phase 0 --tm GAM-001
    python batch_runner.py --phase 1 --all
    python batch_runner.py --phase 2 --gamma-list GAM-006,GAM-009
"""

import argparse
from core.kernel import run_kernel
from core.state_preparation import prepare_state
from database import insert_execution, insert_test_results
from tests.utilities import run_all_applicable_tests

def execute_single_run(gamma_id, gamma_params, d_base_id, 
                       modifier_id, seed, db_conn):
    """Exécute 1 cellule de l'hypercube."""
    
    # Préparer D
    D_base = load_d_base(d_base_id)
    modifier = load_modifier(modifier_id, seed)
    D = prepare_state(D_base, [modifier] if modifier else [])
    
    # Charger Γ
    gamma = load_gamma(gamma_id, gamma_params)
    
    # Exécuter kernel
    history = []
    try:
        for i, state in run_kernel(D, gamma, max_iterations=2000, 
                                    record_history=True):
            if i % 10 == 0:
                history.append(state.copy())
        
        status = "COMPLETED"
    except Exception as e:
        status = "ERROR"
        log_error(e)
    
    # Enregistrer exécution
    exec_id = insert_execution(db_conn, gamma_id, gamma_params,
                               d_base_id, modifier_id, seed, status)
    
    # Appliquer tous les tests applicables
    if status == "COMPLETED":
        test_results = run_all_applicable_tests(
            history, D, d_base_id, gamma_id
        )
        insert_test_results(db_conn, exec_id, test_results)
    
    return exec_id, status

def run_tm_exhaustive(tm_id, db_conn):
    """Exécute tous les runs d'un TM."""
    
    # Charger spécification TM
    tm_spec = load_tm_specification(tm_id)
    
    total_runs = (len(tm_spec.gamma_params_grid) × 
                  len(tm_spec.d_bases) × 
                  len(tm_spec.modifiers) × 
                  len(tm_spec.seeds))
    
    print(f"TM {tm_id}: {total_runs} runs à exécuter")
    
    completed = 0
    for gamma_params in tm_spec.gamma_params_grid:
        for d_base_id in tm_spec.d_bases:
            for modifier_id in tm_spec.modifiers:
                for seed in tm_spec.seeds:
                    
                    # Vérifier si déjà exécuté
                    if exists_in_db(db_conn, tm_spec.gamma_id, 
                                   gamma_params, d_base_id, 
                                   modifier_id, seed):
                        print(f"Skip (déjà exécuté)")
                        completed += 1
                        continue
                    
                    # Exécuter
                    exec_id, status = execute_single_run(
                        tm_spec.gamma_id, gamma_params,
                        d_base_id, modifier_id, seed, db_conn
                    )
                    
                    completed += 1
                    print(f"Progress: {completed}/{total_runs} "
                          f"({100*completed/total_runs:.1f}%)")
    
    print(f"TM {tm_id} terminé: {completed} runs")

# Main avec gestion phases
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[0,1,2])
    parser.add_argument("--tm", type=str, help="TM-GAM-XXX")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    
    db = connect_database()
    
    if args.phase == 0:
        # Phase calibration: 1 TM pilote
        run_tm_exhaustive("TM-GAM-001", db)
    
    elif args.phase == 1:
        # Phase balayage: tous TM avec params nominaux
        for tm_id in get_all_tm_phase1():
            run_tm_exhaustive(tm_id, db)
    
    elif args.phase == 2:
        # Phase exploration: TM prometteurs grille complète
        promising_tms = get_promising_tms(db, score_threshold=10)
        for tm_id in promising_tms:
            run_tm_exhaustive(tm_id, db)