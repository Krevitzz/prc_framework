# prc_automation/batch_runner.py

import argparse
import sys
from tests.utilities.discovery import discover_active_tests
from tests.utilities.applicability import filter_applicable_tests
from tests.utilities.test_engine import run_observation
from tests.utilities.scoring import score_observation
from tests.utilities.verdict_engine import compute_gamma_verdict

class CriticalTestError(Exception):
    """Exception pour erreurs critiques nécessitant arrêt"""
    pass


def main():
    args = parse_args()
    
    if args.mode == 'brut':
        run_batch_brut(args)
    elif args.mode == 'test':
        run_batch_test(args)
    elif args.mode == 'verdict':
        run_batch_verdict(args)
    elif args.mode == 'all':
        run_batch_all(args)


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', required=True,
                       choices=['brut', 'test', 'verdict', 'all'])
    
    parser.add_argument('--gamma', required=True,
                       help="Gamma ID (ex: GAM-001)")
    
    parser.add_argument('--params', default='params_default_v1',
                       help="Global params config ID")
    
    parser.add_argument('--scoring', default='scoring_default_v1',
                       help="Scoring config ID")
    
    parser.add_argument('--thresholds', default='thresholds_strict_v1',
                       help="Thresholds config ID")
    
    return parser.parse_args()

def run_batch_brut(args):
    """
    Exécute kernel pour toutes configs.
    Stocke dans db_raw uniquement.
    """
    gamma_id = args.gamma
    
    # Charger toutes configs pour cette gamma
    configs = generate_all_configs(gamma_id)
    
    log.info(f"Running {len(configs)} configs for {gamma_id}")
    
    for i, config in enumerate(configs):
        try:
            log.info(f"Config {i+1}/{len(configs)}: {config}")
            
            # Exécuter kernel
            result = run_kernel_for_config(config)
            
            # Stocker dans db_raw
            exec_id = store_execution_raw(result)
            
            log.info(f"  ✓ Stored as exec_id={exec_id}")
            
        except CriticalTestError as e:
            log.error("=" * 80)
            log.error("CRITICAL ERROR DETECTED")
            log.error(f"Error: {e}")
            log.error(f"Config: {config}")
            log.error(f"Progress: {i}/{len(configs)} configs completed")
            log.error("=" * 80)
            log.error("Batch execution stopped.")
            log.error("Action required:")
            log.error("  1. Fix the code causing the error")
            log.error("  2. Delete prc_r0_raw.db and prc_r0_results.db")
            log.error("  3. Rerun: batch_runner --brut --gamma {gamma_id}")
            log.error("=" * 80)
            
            # Arrêt total
            sys.exit(1)
    
    log.info(f"✓ Batch brut completed: {len(configs)} configs")


def generate_all_configs(gamma_id: str) -> List[dict]:
    """
    Génère toutes les configs pour une gamma.
    
    Returns:
        list de dicts {gamma_id, d_base_id, modifier_id, seed, ...}
    """
    # Récupérer params gamma
    gamma_params = load_gamma_definition(gamma_id)
    
    # Générer toutes combinaisons
    configs = []
    for d_base_id in gamma_params['d_bases']:
        for modifier_id in gamma_params['modifiers']:
            for seed in gamma_params['seeds']:
                configs.append({
                    'gamma_id': gamma_id,
                    'd_base_id': d_base_id,
                    'modifier_id': modifier_id,
                    'seed': seed,
                    'gamma_params': gamma_params['operator_params'],
                })
    
    return configs


def run_kernel_for_config(config: dict) -> dict:
    """
    Exécute kernel pour une config.
    
    Returns:
        {
            'config': dict,
            'history': list[np.ndarray],
            'metrics': list[dict],
            'converged': bool,
            'final_iteration': int,
        }
    """
    # Préparer D
    d_state = prepare_state(
        base=create_d_base(config['d_base_id']),
        modifiers=load_modifiers(config['modifier_id'])
    )
    
    # Créer gamma
    gamma = create_gamma_operator(config['gamma_id'], config['gamma_params'])
    
    # Exécuter kernel
    history = []
    metrics = []
    
    for iteration, state in run_kernel(
        d_state, gamma, 
        max_iterations=10000,
        record_history=True
    ):
        history.append(state.copy())
        metrics.append({
            'iteration': iteration,
            'norm': np.linalg.norm(state),
            # autres métriques kernel
        })
    
    return {
        'config': config,
        'history': history,
        'metrics': metrics,
        'converged': len(history) < 10000,
        'final_iteration': len(history),
    }


def store_execution_raw(result: dict) -> int:
    """
    Stocke résultat dans db_raw.
    
    Returns:
        exec_id: ID de l'exécution
    """
    conn = sqlite3.connect('prc_database/prc_r0_raw.db')
    cursor = conn.cursor()
    
    # Insérer Execution
    cursor.execute("""
        INSERT INTO Executions (
            gamma_id, d_base_id, modifier_id, seed,
            gamma_params, converged, final_iteration,
            executed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result['config']['gamma_id'],
        result['config']['d_base_id'],
        result['config']['modifier_id'],
        result['config']['seed'],
        json.dumps(result['config']['gamma_params']),
        result['converged'],
        result['final_iteration'],
        datetime.now().isoformat(),
    ))
    
    exec_id = cursor.lastrowid
    
    # Insérer Snapshots (échantillonnage)
    snapshot_indices = np.linspace(0, len(result['history'])-1, 100, dtype=int)
    for idx in snapshot_indices:
        state = result['history'][idx]
        cursor.execute("""
            INSERT INTO Snapshots (exec_id, iteration, state_blob)
            VALUES (?, ?, ?)
        """, (
            exec_id,
            int(idx),
            compress_state(state),
        ))
    
    # Insérer Metrics
    for metric in result['metrics']:
        cursor.execute("""
            INSERT INTO Metrics (
                exec_id, iteration, norm, ...
            ) VALUES (?, ?, ?, ...)
        """, (exec_id, metric['iteration'], metric['norm'], ...))
    
    conn.commit()
    conn.close()
    
    return exec_id
	
def run_batch_test(args):
    """
    Applique tests sur runs existants.
    Calcule observations et scores.
    Stocke dans db_results.
    """
    gamma_id = args.gamma
    params_config_id = args.params
    scoring_config_id = args.scoring
    
    # Vérifier que runs existent
    exec_ids = get_exec_ids_for_gamma(gamma_id)
    if not exec_ids:
        log.warning(f"No runs found for {gamma_id} in db_raw")
        log.info("Running --brut first...")
        run_batch_brut(args)
        exec_ids = get_exec_ids_for_gamma(gamma_id)
    
    # Découvrir tests actifs
    all_tests = discover_active_tests()
    log.info(f"Discovered {len(all_tests)} active tests")
    
    # Pour chaque run
    for exec_id in exec_ids:
        log.info(f"Processing exec_id={exec_id}")
        
        # Charger contexte et history
        context = load_execution_context(exec_id)
        history = load_execution_history(exec_id)
        
        # Filtrer tests applicables
        applicable_tests = filter_applicable_tests(all_tests, context)
        log.info(f"  {len(applicable_tests)} applicable tests")
        
        # Appliquer chaque test
        for test_id, test_module in applicable_tests.items():
            try:
                # Phase 1: Observation
                observation = run_observation(
                    test_module, history, context, params_config_id
                )
                
                # Stocker observation
                store_test_observation(exec_id, observation)
                
                # Si NOT_APPLICABLE ou ERROR, skip scoring
                if observation['status'] not in ['SUCCESS', 'SKIPPED']:
                    continue
                
                # Phase 2: Scoring
                scores = score_observation(
                    observation, test_module, context, scoring_config_id
                )
                
                # Stocker scores
                store_test_scores(exec_id, scores)
                
                log.info(f"    ✓ {test_id}: {observation['status']}")
                
            except Exception as e:
                log.error(f"    ✗ {test_id}: {e}")
                # Ne pas lever - continuer avec autres tests
    
    log.info(f"✓ Batch test completed for {gamma_id}")


def load_execution_context(exec_id: int) -> dict:
    """Charge contexte depuis db_raw"""
    conn = sqlite3.connect('prc_database/prc_r0_raw.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT gamma_id, d_base_id, modifier_id, seed
        FROM Executions WHERE id = ?
    """, (exec_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    # Récupérer shape depuis premier snapshot
    history = load_execution_history(exec_id)
    state_shape = history[0].shape
    
    return {
        'gamma_id': row[0],
        'd_base_id': row[1],
        'modifier_id': row[2],
        'seed': row[3],
        'state_shape': state_shape,
    }


def load_execution_history(exec_id: int) -> List[np.ndarray]:
    """Charge history depuis db_raw"""
    conn = sqlite3.connect('prc_database/prc_r0_raw.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT iteration, state_blob
        FROM Snapshots
        WHERE exec_id = ?
        ORDER BY iteration
    """, (exec_id,))
    
    history = []
    for row in cursor.fetchall():
        state = decompress_state(row[1])
        history.append(state)
    
    conn.close()
    return history


def store_test_observation(exec_id: int, observation: dict):
    """Stocke observation dans db_results"""
    conn = sqlite3.connect('prc_database/prc_r0_results.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO TestObservations (
            exec_id, test_name, test_category,
            params_config_id,
            applicable, status, message,
            stat_initial, stat_final, stat_min, stat_max, stat_mean, stat_std,
            evolution_transition, evolution_trend, evolution_trend_coefficient,
            observation_data,
            computed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        exec_id,
        observation['test_name'],
        observation['test_name'].split('-')[0],
        observation['config_params_id'],
        observation['status'] not in ['NOT_APPLICABLE', 'SKIPPED'],
        observation['status'],
        observation['message'],
        observation['statistics'].get('initial', 0.0),
        observation['statistics'].get('final', 0.0),
        observation['statistics'].get('min', 0.0),
        observation['statistics'].get('max', 0.0),
        observation['statistics'].get('mean', 0.0),
        observation['statistics'].get('std', 0.0),
        observation['evolution'].get('transition', 'unknown'),
        observation['evolution'].get('trend', 'unknown'),
        observation['evolution'].get('trend_coefficient', 0.0),
        json.dumps(observation),
        datetime.now().isoformat(),
    ))
    
    conn.commit()
    conn.close()


def store_test_scores(exec_id: int, scores: dict):
    """Stocke scores dans db_results"""
    conn = sqlite3.connect('prc_database/prc_r0_results.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO TestScores (
            exec_id, test_name,
            params_config_id, scoring_config_id,
            test_weight, metric_scores, weighted_average,
            computed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        exec_id,
        scores['test_name'],
        scores['config_params_id'],
        scores['config_scoring_id'],
        scores['test_weight'],
        json.dumps(scores['metric_scores']),
        scores['weighted_average'],
        datetime.now().isoformat(),
    ))
    
    conn.commit()
    conn.close()
	
def run_batch_verdict(args):
    """
    Calcule verdict global pour gamma.
    Agrège tous tests sur tous runs.
    Stocke dans db_results.
    """
    gamma_id = args.gamma
    params_config_id = args.params
    scoring_config_id = args.scoring
    thresholds_config_id = args.thresholds
    
    # Vérifier que scores existent
    scores_exist = check_scores_exist(
        gamma_id, params_config_id, scoring_config_id
    )
    
    if not scores_exist:
        log.warning("Scores not found, running --test first...")
        run_batch_test(args)
    
    # Calculer verdict
    log.info(f"Computing verdict for {gamma_id}")
    verdict = compute_gamma_verdict(
        gamma_id,
        params_config_id,
        scoring_config_id,
        thresholds_config_id
    )
    
    # Stocker verdict
    store_gamma_verdict(verdict)
    
    # Générer rapport
    generate_verdict_report(verdict)
    
    log.info(f"✓ Verdict: {verdict['verdict']}")
    log.info(f"  Score: {verdict['global_score']:.3f}")
    log.info(f"  Robustness: {verdict['robustness_pct']:.1f}%")
    log.info(f"  Majority: {verdict['majority_pct']:.1f}%")


def check_scores_exist(gamma_id, params_config_id, scoring_config_id) -> bool:
    """Vérifie si scores existent pour cette config"""
    conn = sqlite3.connect('prc_database/prc_r0_results.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) FROM TestScores ts
        JOIN Executions e ON ts.exec_id = e.id
        WHERE e.gamma_id = ?
          AND ts.params_config_id = ?
          AND ts.scoring_config_id = ?
    """, (gamma_id, params_config_id, scoring_config_id))
    
    count = cursor.fetchone()[0]
    conn.close()
    
    return count > 0


def store_gamma_verdict(verdict: dict):
    """Stocke verdict dans db_results"""
    conn = sqlite3.connect('prc_database/prc_r0_results.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO GammaVerdicts (
            gamma_id,
            params_config_id, scoring_config_id, thresholds_config_id,
            majority_pct, robustness_pct, global_score,
            verdict, verdict_reason,
            computed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        verdict['gamma_id'],
        verdict['params_config_id'],
        verdict['scoring_config_id'],
        verdict['thresholds_config_id'],
        verdict['majority_pct'],
        verdict['robustness_pct'],
        verdict['global_score'],
        verdict['verdict'],
        verdict['verdict_reason'],
        datetime.now().isoformat(),
    ))
    
    conn.commit()
    conn.close()


def generate_verdict_report(verdict: dict):
    """Génère rapport texte lisible"""
    report_path = f"reports/verdict_{verdict['gamma_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"VERDICT REPORT: {verdict['gamma_id']}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write(f"  Params:     {verdict['params_config_id']}\n")
        f.write(f"  Scoring:    {verdict['scoring_config_id']}\n")
        f.write(f"  Thresholds: {verdict['thresholds_config_id']}\n\n")
        
        f.write("CRITERIA:\n")
        f.write(f"  Global Score:  {verdict['global_score']:.3f} / 1.0\n")
        f.write(f"  Robustness:    {verdict['robustness_pct']:.1f}%\n")
        f.write(f"  Majority:      {verdict['majority_pct']:.1f}%\n\n")
        
        f.write("VERDICT:\n")
        f.write(f"  {verdict['verdict']}\n\n")
        
        f.write("REASON:\n")
        f.write(f"  {verdict['verdict_reason']}\n\n")
        
        f.write("=" * 80 + "\n")
    
    log.info(f"Report saved: {report_path}")
	
def run_batch_all(args):
    """Exécute pipeline complet en une commande"""
    log.info("Running full pipeline: brut → test → verdict")
    
    run_batch_brut(args)
    run_batch_test(args)
    run_batch_verdict(args)
    
    log.info("✓ Full pipeline completed")