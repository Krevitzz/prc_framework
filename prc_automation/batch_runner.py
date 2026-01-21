# prc_automation/batch_runner.py
"""
Batch Runner Charter 5.5 - Pipeline exécution complet.

Modes:
- --brut: Collecte données (db_raw)
- --test: Application tests (db_results)
- --verdict: Génération verdicts exploratoires (rapports)
- --all: Pipeline complet
"""

import argparse
import sys
import sqlite3
import json
from datetime import datetime
from pathlib import Path

from tests.utilities.utils.discovery import discover_active_tests
from tests.utilities.utils.applicability import check as check_applicability
from tests.utilities.HUB.test_engine import TestEngine


class CriticalTestError(Exception):
    """Exception pour erreurs critiques nécessitant arrêt."""
    pass


# =============================================================================
# MODE BRUT (collecte données)
# =============================================================================

def run_batch_brut(args):
    """
    Exécute kernel pour toutes configs.
    Stocke dans db_raw uniquement.
    """
    print(f"\n{'='*70}")
    print("MODE BRUT - Collecte données")
    print(f"{'='*70}\n")
    
    gamma_id = args.gamma
    
    # TODO: Implémenter génération configs + exécution kernel
    # Pour l'instant, assume que db_raw existe déjà
    
    print(f"⚠️ Mode --brut assume db_raw existante")
    print(f"  Vérifier executions pour {gamma_id}...")
    
    conn = sqlite3.connect('./prc_automation/prc_database/prc_r0_raw.db')
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM Executions WHERE gamma_id = ?", (gamma_id,))
    count = cursor.fetchone()[0]
    conn.close()
    
    if count == 0:
        print(f"\n❌ Aucune exécution trouvée pour {gamma_id} dans db_raw")
        print(f"   Action: Exécuter kernel manuellement ou implémenter génération configs")
        sys.exit(1)
    
    print(f"✓ {count} exécutions trouvées pour {gamma_id}")


# =============================================================================
# MODE TEST (observations)
# =============================================================================

def run_batch_test(args):
    """
    Applique tests sur runs existants.
    Calcule observations.
    Stocke dans db_results.
    """
    print(f"\n{'='*70}")
    print("MODE TEST - Application tests")
    print(f"{'='*70}\n")
    
    gamma_id = args.gamma
    params_config_id = args.params
    
    # Vérifier db_raw
    exec_ids = get_exec_ids_for_gamma(gamma_id)
    if not exec_ids:
        print(f"❌ Aucune exécution trouvée pour {gamma_id} dans db_raw")
        print(f"   Action: Exécuter --brut d'abord")
        sys.exit(1)
    
    print(f"✓ {len(exec_ids)} exécutions trouvées")
    
    # Découvrir tests actifs
    print("\nDécouverte tests...")
    all_tests = discover_active_tests()
    print(f"✓ {len(all_tests)} tests actifs découverts")
    
    # Initialiser engine
    engine = TestEngine()
    
    # Compteurs
    total_observations = 0
    errors = []
    
    # Pour chaque run
    for i, exec_id in enumerate(exec_ids, 1):
        print(f"\n[{i}/{len(exec_ids)}] Processing exec_id={exec_id}")
        
        try:
            # Charger contexte
            context = load_execution_context(exec_id)
            
            # Charger premier snapshot pour state_shape
            first_snapshot = load_first_snapshot(exec_id)
            context['state_shape'] = first_snapshot.shape
            context['exec_id'] = exec_id  # ⚠️ TRAÇABILITÉ
            
            # Filtrer tests applicables
            applicable_tests = {}
            for test_id, test_module in all_tests.items():
                applicable, reason = check_applicability(test_module, context)
                if applicable:
                    applicable_tests[test_id] = test_module
            
            print(f"  {len(applicable_tests)}/{len(all_tests)} tests applicables")
            
            if not applicable_tests:
                continue
            
            # Charger history complète
            history = load_execution_history(exec_id)
            
            # Appliquer chaque test
            for test_id, test_module in applicable_tests.items():
                try:
                    # Phase : Observation
                    observation = engine.execute_test(
                        test_module, context, history, params_config_id
                    )
                    
                    # Stocker observation
                    store_test_observation(exec_id, observation)
                    total_observations += 1
                    
                    status = observation['status']
                    print(f"    ✓ {test_id}: {status}")
                
                except Exception as e:
                    error_msg = f"exec_id={exec_id}, test={test_id}: {str(e)}"
                    errors.append(error_msg)
                    print(f"    ✗ {test_id}: {str(e)}")
        
        except Exception as e:
            error_msg = f"exec_id={exec_id}: {str(e)}"
            errors.append(error_msg)
            print(f"  ✗ Erreur run: {str(e)}")
    
    # Résumé
    print(f"\n{'='*70}")
    print("RÉSUMÉ MODE TEST")
    print(f"{'='*70}")
    print(f"Observations générées: {total_observations}")
    print(f"Erreurs:               {len(errors)}")
    
    if errors:
        print("\nErreurs détaillées:")
        for err in errors[:10]:  # Limiter affichage
            print(f"  - {err}")


# =============================================================================
# MODE VERDICT (analyse exploratoire)
# =============================================================================

def run_batch_verdict(args):
    """
    Génère verdicts exploratoires sur observations existantes.
    
    Architecture 5.5 (NOUVEAU):
    - verdict_reporter orchestre verdict_engine + gamma_profiling
    - Génération rapports structurés selon Charter R0
    """
    print(f"\n{'='*70}")
    print("MODE VERDICT - Analyse exploratoire R0")
    print(f"{'='*70}\n")
    
    params_config_id = args.params
    verdict_config_id = args.verdict
    
    print(f"Params config:  {params_config_id}")
    print(f"Verdict config: {verdict_config_id}\n")
    
    # Vérifier que observations existent
    n_observations = count_observations(params_config_id)
    if n_observations == 0:
        print(f"❌ Aucune observation trouvée pour params={params_config_id}")
        print(f"   Action: Exécuter --mode test d'abord")
        sys.exit(1)
    
    print(f"✓ {n_observations} observations trouvées\n")
    
    # Import verdict_reporter (NOUVEAU)
    try:
        from tests.utilities.HUB.verdict_reporter import generate_verdict_report
    except ImportError as e:
        print(f"❌ Erreur import verdict_reporter: {e}")
        sys.exit(1)
    
    # Exécution pipeline (SIMPLIFIÉ)
    try:
        results = generate_verdict_report(
            params_config_id=params_config_id,
            verdict_config_id=verdict_config_id
        )
        
        # Résumé rapide
        print("="*70)
        print("VERDICT GÉNÉRÉ")
        print("="*70)
        print(f"Répertoire      : {Path(results['report_paths']['summary_global']).parent}")
        print(f"Rapport global  : summary_global.txt")
        print(f"Rapports axes   : summary_test.txt, summary_gamma.txt, summary_modifier.txt, summary_encoding.txt")
        print(f"Fichiers totaux : {len(results['report_paths'])}")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Erreur génération verdict: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# =============================================================================
# MODE ALL (pipeline complet)
# =============================================================================

def run_batch_all(args):
    """Exécute pipeline complet en une commande."""
    print(f"\n{'#'*70}")
    print("# PIPELINE COMPLET: brut → test → verdict")
    print(f"{'#'*70}\n")
    
    # run_batch_brut(args)  # Si gamma spécifié
    # run_batch_test(args)  # Si gamma spécifié
    run_batch_verdict(args)
    
    print(f"\n{'#'*70}")
    print("# PIPELINE TERMINÉ")
    print(f"{'#'*70}\n")


# =============================================================================
# UTILITAIRES DATABASE
# =============================================================================

def get_exec_ids_for_gamma(gamma_id: str) -> list:
    """Récupère tous exec_ids pour une gamma."""
    conn = sqlite3.connect('./prc_automation/prc_database/prc_r0_raw.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM Executions WHERE gamma_id = ?", (gamma_id,))
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]


def count_observations(params_config_id: str) -> int:
    """Compte observations SUCCESS pour une config."""
    conn = sqlite3.connect('./prc_automation/prc_database/prc_r0_results.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM TestObservations 
        WHERE params_config_id = ? AND status = 'SUCCESS'
    """, (params_config_id,))
    count = cursor.fetchone()[0]
    conn.close()
    return count


def load_execution_context(exec_id: int) -> dict:
    """Charge contexte depuis db_raw."""
    conn = sqlite3.connect('./prc_automation/prc_database/prc_r0_raw.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT gamma_id, d_encoding_id, modifier_id, seed, run_id
        FROM Executions WHERE id = ?
    """, (exec_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise ValueError(f"exec_id={exec_id} non trouvé dans db_raw")
    
    return {
        'gamma_id': row[0],
        'd_encoding_id': row[1],
        'modifier_id': row[2],
        'seed': row[3],
        'run_id': row[4]
    }


def load_first_snapshot(exec_id: int):
    """Charge premier snapshot pour déduire state_shape."""
    conn = sqlite3.connect('./prc_automation/prc_database/prc_r0_raw.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT state_blob
        FROM Snapshots
        WHERE exec_id = ?
        ORDER BY iteration
        LIMIT 1
    """, (exec_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise ValueError(f"Aucun snapshot pour exec_id={exec_id}")
    
    # Décompresser
    import gzip
    import pickle
    state = pickle.loads(gzip.decompress(row[0]))
    return state


def load_execution_history(exec_id: int) -> list:
    """Charge history complète depuis db_raw."""
    import gzip
    import pickle
    
    conn = sqlite3.connect('./prc_automation/prc_database/prc_r0_raw.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT iteration, state_blob
        FROM Snapshots
        WHERE exec_id = ?
        ORDER BY iteration
    """, (exec_id,))
    
    history = []
    for row in cursor.fetchall():
        state = pickle.loads(gzip.decompress(row[1]))
        history.append(state)
    
    conn.close()
    return history


def store_test_observation(exec_id: int, observation: dict):
    """Stocke observation dans db_results."""
    conn = sqlite3.connect('./prc_automation/prc_database/prc_r0_results.db')
    cursor = conn.cursor()
    
    # Extraire stats pour colonnes rapides
    stats = observation.get('statistics', {})
    first_metric = list(stats.keys())[0] if stats else None
    
    if first_metric:
        stat_data = stats[first_metric]
        stat_initial = stat_data.get('initial')
        stat_final = stat_data.get('final')
        stat_min = stat_data.get('min')
        stat_max = stat_data.get('max')
        stat_mean = stat_data.get('mean')
        stat_std = stat_data.get('std')
    else:
        stat_initial = stat_final = stat_min = None
        stat_max = stat_mean = stat_std = None
    
    # Extraire evolution
    evol = observation.get('evolution', {})
    first_evol = list(evol.keys())[0] if evol else None
    
    if first_evol:
        evol_data = evol[first_evol]
        evolution_transition = evol_data.get('transition')
        evolution_trend = evol_data.get('trend')
        evolution_trend_coefficient = evol_data.get('slope')
    else:
        evolution_transition = evolution_trend = None
        evolution_trend_coefficient = None
    
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
        observation['test_category'],
        observation['config_params_id'],
        observation['status'] not in ['NOT_APPLICABLE', 'SKIPPED'],
        observation['status'],
        observation['message'],
        stat_initial, stat_final, stat_min, stat_max, stat_mean, stat_std,
        evolution_transition, evolution_trend, evolution_trend_coefficient,
        json.dumps(observation),
        datetime.now().isoformat()
    ))
    
    conn.commit()
    conn.close()


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch Runner Charter 5.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Mode brut (collecte données)
  python -m prc_automation.batch_runner --mode brut --gamma GAM-001
  
  # Mode test (observations)
  python -m prc_automation.batch_runner --mode test --gamma GAM-001 --params params_default_v1
  
  # Mode verdict (analyse exploratoire, configs par défaut)
  python -m prc_automation.batch_runner --mode verdict
  
  # Mode verdict (configs spécifiques)
  python -m prc_automation.batch_runner --mode verdict --params params_default_v1 --verdict verdict_strict_v1
  
  # Pipeline complet
  python -m prc_automation.batch_runner --mode all --gamma GAM-001
        """
    )
    
    parser.add_argument('--mode', required=True,
                       choices=['brut', 'test', 'verdict', 'all'],
                       help="Mode exécution")
    
    parser.add_argument('--gamma', default=None,
                       help="Gamma ID (ex: GAM-001) - Requis pour modes brut/test")
    
    parser.add_argument('--params', default='params_default_v1',
                       help="Params config ID (défaut: params_default_v1)")
    
    parser.add_argument('--verdict', default='verdict_default_v1',
                       help="Verdict config ID (défaut: verdict_default_v1)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validation arguments
    if args.mode in ['brut', 'test'] and not args.gamma:
        print("❌ Erreur: --gamma requis pour modes 'brut' et 'test'")
        sys.exit(1)
    
    # Exécution
    if args.mode == 'brut':
        run_batch_brut(args)
    elif args.mode == 'test':
        run_batch_test(args)
    elif args.mode == 'verdict':
        run_batch_verdict(args)
    elif args.mode == 'all':
        run_batch_all(args)


if __name__ == "__main__":
    main()