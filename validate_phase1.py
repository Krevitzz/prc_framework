#!/usr/bin/env python3
"""
validate_phase1.py

Script de validation infrastructure Phase 1.

Vérifie:
1. Opérateurs Γ implémentés (≥ 2)
2. Base de données créable
3. batch_runner1 fonctionnel
4. report_generator fonctionnel
5. Connexion DB → opérateurs → tests

Usage:
    python validate_phase1.py
    python validate_phase1.py --quick
"""

import sys
import sqlite3
import json
from pathlib import Path


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(title):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*70}")
    print(f"{title}")
    print(f"{'='*70}{Colors.RESET}\n")


def print_success(msg):
    print(f"{Colors.GREEN}✓{Colors.RESET} {msg}")


def print_error(msg):
    print(f"{Colors.RED}✗{Colors.RESET} {msg}")


def print_warning(msg):
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {msg}")


def print_info(msg):
    print(f"  {msg}")


# ============================================================================
# VALIDATION 1 : FICHIERS PHASE 1
# ============================================================================

def validate_phase1_files(verbose=False):
    """Vérifie que les nouveaux fichiers Phase 1 existent."""
    print_header("VALIDATION 1 : Fichiers Phase 1")
    
    required_files = {
        'operators_new': [
            'operators/gamma_hyp_002.py','operators/gamma_hyp_003.py','operators/gamma_hyp_004.py','operators/gamma_hyp_005.py',
            'operators/gamma_hyp_006.py','operators/gamma_hyp_007.py','operators/gamma_hyp_008.py','operators/gamma_hyp_009.py','operators/gamma_hyp_010.py','operators/gamma_hyp_012.py','operators/gamma_hyp_013.py',
        ],
        'database': [
            'prc_database/schema.sql',
        ],
        'automation': [
            'prc_automation/batch_runner1.py',
            'prc_automation/report_generator.py',
        ],
        'docs': [
            'README_Phase1.md',
        ]
    }
    
    all_ok = True
    for category, files in required_files.items():
        if verbose:
            print(f"\n{category.upper()}:")
        
        category_ok = True
        for filepath in files:
            path = Path(filepath)
            if path.exists():
                if verbose:
                    print_success(f"{filepath}")
            else:
                print_error(f"{filepath} MANQUANT")
                category_ok = False
                all_ok = False
        
        if not verbose:
            if category_ok:
                print_success(f"{category}: {len(files)} fichiers OK")
            else:
                print_error(f"{category}: Fichiers manquants")
    
    return all_ok


# ============================================================================
# VALIDATION 2 : OPÉRATEURS IMPLÉMENTÉS
# ============================================================================

def validate_operators_count(verbose=False):
    """Vérifie qu'au moins 2 Γ sont implémentés."""
    print_header("VALIDATION 2 : Opérateurs Γ Implémentés")
    
    try:
        from operators import OPERATOR_REGISTRY
        
        implemented = [
            (gamma_id, info['name']) 
            for gamma_id, info in OPERATOR_REGISTRY.items()
            if info['implemented']
        ]
        
        n_impl = len(implemented)
        n_total = len(OPERATOR_REGISTRY)
        
        if verbose:
            print("Opérateurs implémentés:")
            for gamma_id, name in implemented:
                print_success(f"  {gamma_id}: {name}")
            
            print("\nOpérateurs manquants:")
            for gamma_id, info in OPERATOR_REGISTRY.items():
                if not info['implemented']:
                    print_warning(f"  {gamma_id}: {info['name']}")
        
        if n_impl >= 2:
            print_success(f"{n_impl}/{n_total} opérateurs implémentés (minimum atteint)")
            return True
        else:
            print_error(f"{n_impl}/{n_total} opérateurs implémentés (minimum: 2)")
            return False
    
    except Exception as e:
        print_error(f"Impossible de charger OPERATOR_REGISTRY: {str(e)}")
        return False


# ============================================================================
# VALIDATION 3 : BASE DE DONNÉES
# ============================================================================

def validate_database(verbose=False):
    """Vérifie que la DB peut être créée et le schéma chargé."""
    print_header("VALIDATION 3 : Base de Données")
    
    db_path = Path("prc_database/prc_r0_test.db")
    schema_path = Path("prc_database/schema.sql")
    
    try:
        # Nettoyer DB test existante
        if db_path.exists():
            db_path.unlink()
        
        # Créer DB
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        
        if verbose:
            print_success("Connexion DB créée")
        
        # Charger schéma
        if not schema_path.exists():
            print_error("schema.sql non trouvé")
            return False
        
        with open(schema_path, 'r') as f:
            schema = f.read()
        
        conn.executescript(schema)
        
        if verbose:
            print_success("Schéma SQL chargé")
        
        # Vérifier tables créées
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        required_tables = ['Executions', 'TestResults', 'Histories', 'Scores']
        
        for table in required_tables:
            if table in tables:
                if verbose:
                    print_success(f"  Table {table} créée")
            else:
                print_error(f"  Table {table} manquante")
                return False
        
        # Vérifier vues
        cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
        views = [row[0] for row in cursor.fetchall()]
        
        if 'GammaSummary' in views or 'GammaSum' in views:
            if verbose:
                print_success("  Vues créées")
        
        conn.close()
        
        # Nettoyer
        db_path.unlink()
        
        print_success("Base de données : OK")
        return True
    
    except Exception as e:
        print_error(f"Erreur DB: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


# ============================================================================
# VALIDATION 4 : BATCH RUNNER
# ============================================================================

def validate_batch_runner(verbose=False, quick=False):
    """Vérifie que batch_runner peut s'exécuter."""
    print_header("VALIDATION 4 : Batch Runner")
    
    if quick:
        print_warning("Mode quick: vérification import uniquement")
        try:
            import prc_automation.batch_runner1 as batch_runner
            print_success("Import batch_runner: OK")
            return True
        except Exception as e:
            print_error(f"Import échoué: {str(e)}")
            return False
    
    try:
        # Import
        import prc_automation.batch_runner1 as batch_runner
        
        if verbose:
            print_success("Import batch_runner")
        
        # Vérifier fonctions principales
        required_functions = [
            'init_database',
            'execute_single_run',
            'run_phase1_gamma',
        ]
        
        for func_name in required_functions:
            if hasattr(batch_runner, func_name):
                if verbose:
                    print_success(f"  Fonction {func_name} disponible")
            else:
                print_error(f"  Fonction {func_name} manquante")
                return False
        
        # Vérifier catalogues
        if hasattr(batch_runner, 'D_BASE_CATALOG'):
            n_d = len(batch_runner.D_BASE_CATALOG)
            if verbose:
                print_success(f"  D_BASE_CATALOG: {n_d} bases")
        
        if hasattr(batch_runner, 'MODIFIERS'):
            n_m = len(batch_runner.MODIFIERS)
            if verbose:
                print_success(f"  MODIFIERS: {n_m} modifiers")
        
        print_success("Batch runner: OK")
        return True
    
    except Exception as e:
        print_error(f"Batch runner: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


# ============================================================================
# VALIDATION 5 : REPORT GENERATOR
# ============================================================================

def validate_report_generator(verbose=False):
    """Vérifie que report_generator peut s'exécuter."""
    print_header("VALIDATION 5 : Report Generator")
    
    try:
        import prc_automation.report_generator as report_gen
        
        if verbose:
            print_success("Import report_generator")
        
        # Vérifier fonctions
        required_functions = [
            'get_gamma_summary',
            'get_gamma_x_d_matrix',
            'print_summary_report',
        ]
        
        for func_name in required_functions:
            if hasattr(report_gen, func_name):
                if verbose:
                    print_success(f"  Fonction {func_name} disponible")
            else:
                print_error(f"  Fonction {func_name} manquante")
                return False
        
        print_success("Report generator: OK")
        return True
    
    except Exception as e:
        print_error(f"Report generator: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


# ============================================================================
# VALIDATION 6 : INTÉGRATION COMPLÈTE
# ============================================================================

def validate_integration(verbose=False):
    """Test d'intégration complet (optionnel, coûteux)."""
    print_header("VALIDATION 6 : Intégration Complète")
    
    print_warning("Test intégration complet (peut prendre 1-2 minutes)")
    
    try:
        import numpy as np
        from operators import get_operator_by_id
        from D_encodings import create_identity
        from core.kernel import run_kernel
        from core.state_preparation import prepare_state
        from tests.utilities import run_all_applicable_tests
        
        # 1. Créer D
        D_base = create_identity(n_dof=20)
        D = prepare_state(D_base, [])
        
        if verbose:
            print_success("  D^(base) créé")
        
        # 2. Créer Γ (utiliser GAM-001 qui existe)
        gamma = get_operator_by_id("GAM-001", beta=2.0)
        
        if verbose:
            print_success("  Γ créé")
        
        # 3. Exécuter kernel court
        history = []
        for i, state in run_kernel(D, gamma, max_iterations=50):
            if i % 5 == 0:
                history.append(state.copy())
        
        if verbose:
            print_success(f"  Kernel exécuté: {len(history)} snapshots")
        
        # 4. Appliquer tests
        results = run_all_applicable_tests(history, D_base, "SYM-001", "GAM-001")
        
        if verbose:
            print_success(f"  Tests appliqués: {len(results)} résultats")
        
        # 5. Simuler insertion DB (sans réellement insérer)
        # Vérifier que les données sont sérialisables
        test_data = {
            'gamma_id': 'GAM-001',
            'gamma_params': json.dumps({'beta': 2.0}),
            'd_base_id': 'SYM-001',
            'modifier_id': 'M0',
            'seed': 1,
            'test_results': {
                test_name: {
                    'status': result.status,
                    'message': result.message
                }
                for test_name, result in results.items()
            }
        }
        
        # Vérifier sérialisation JSON
        json.dumps(test_data)
        
        if verbose:
            print_success("  Données sérialisables")
        
        print_success("Intégration complète: OK")
        return True
    
    except Exception as e:
        print_error(f"Intégration échouée: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validation Phase 1")
    parser.add_argument('--verbose', action='store_true', help='Affichage détaillé')
    parser.add_argument('--quick', action='store_true', help='Validation rapide (sans intégration)')
    parser.add_argument('--skip-integration', action='store_true', help='Sauter test intégration')
    
    args = parser.parse_args()
    
    print(f"\n{Colors.BOLD}{'#'*70}")
    print("#" + " "*68 + "#")
    print("#" + " "*20 + "VALIDATION PHASE 1" + " "*30 + "#")
    print("#" + " "*68 + "#")
    print(f"{'#'*70}{Colors.RESET}\n")
    
    # Exécuter validations
    results = {}
    
    results['files'] = validate_phase1_files(verbose=args.verbose)
    results['operators'] = validate_operators_count(verbose=args.verbose)
    results['database'] = validate_database(verbose=args.verbose)
    results['batch_runner'] = validate_batch_runner(verbose=args.verbose, quick=args.quick)
    results['report_gen'] = validate_report_generator(verbose=args.verbose)
    
    if not args.skip_integration and not args.quick:
        results['integration'] = validate_integration(verbose=args.verbose)
    
    # Rapport final
    print_header("RAPPORT FINAL")
    
    for category, success in results.items():
        if success:
            print_success(f"{category.upper()}: OK")
        else:
            print_error(f"{category.upper()}: ÉCHEC")
    
    n_success = sum(1 for v in results.values() if v)
    n_total = len(results)
    
    print(f"\n{Colors.BOLD}{'─'*70}")
    print(f"Résultat: {n_success}/{n_total} validations réussies")
    print(f"{'─'*70}{Colors.RESET}\n")
    
    if n_success == n_total:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ PHASE 1 INFRASTRUCTURE PRÊTE{Colors.RESET}")
        print("\nCommandes suivantes:")
        print(f"  {Colors.BLUE}# 1. Initialiser DB{Colors.RESET}")
        print(f"  python prc_automation/batch_runner1.py --init-db")
        print(f"\n  {Colors.BLUE}# 2. Exécuter premier Γ{Colors.RESET}")
        print(f"  python prc_automation/batch_runner1.py --phase 1 --gamma GAM-001")
        print(f"\n  {Colors.BLUE}# 3. Générer rapport{Colors.RESET}")
        print(f"  python prc_automation/report_generator.py --summary")
        print()
        
        # Info sur Γ restants
        from operators import OPERATOR_REGISTRY
        n_impl = sum(1 for info in OPERATOR_REGISTRY.values() if info['implemented'])
        n_total_gamma = len(OPERATOR_REGISTRY)
        
        if n_impl < n_total_gamma:
            print(f"{Colors.YELLOW}⚠ Opérateurs restants: {n_total_gamma - n_impl}/14{Colors.RESET}")
            print(f"  Voir README_Phase1.md section 'À CRÉER'")
        
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ VALIDATION ÉCHOUÉE{Colors.RESET}")
        print("\nCorrections nécessaires avant utilisation Phase 1")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())