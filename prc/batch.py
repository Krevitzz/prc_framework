"""
prc.batch

Responsabilité : Point d'entrée CLI — 3 modes orchestration

CHANGEMENT : Génère rapport TXT + JSON
                                 
                          
  
                         
                                    
  
                         
                                
"""

import argparse
from pathlib import Path

from running.hub_running import run_batch
from analysing.verdict import (
    run_verdict_intra,
    run_verdict_from_parquet,
    run_verdict_cross_phases,
    write_verdict_report,
    write_verdict_report_txt,
)


def main():
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description='PRC Batch Runner — Run + Verdict',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Mode 1 : Run + verdict auto
  python -m batch poc
  
  # Mode 2 : Verdict single-phase
  python -m batch --verdict poc
  
  # Mode 3 : Verdict cross-phases
  python -m batch --verdict
        """
    )
    
    parser.add_argument(
        'phase',
        nargs='?',
        help='Phase (ex: poc, r0, r1). Obligatoire sauf --verdict sans phase.'
    )
    
    parser.add_argument(
        '--verdict',
        action='store_true',
        help='Mode verdict uniquement (skip run)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Afficher détails explosions et compositions skippées'
    )
    
    parser.add_argument(
        '--auto-confirm',
        action='store_true',
        help='Skip confirmation dry-run (pour tests)'
    )
    
    parser.add_argument(
        '--regime-profile',
        choices=['default', 'laxe', 'strict'],
        default='default',
        help='Profil seuils régimes (default: default)'
    )
    
    args = parser.parse_args()
    
    # ==========================================================================
    # MODE 3 : Verdict cross-phases
    # ==========================================================================
    if args.verdict and args.phase is None:
        print("=== MODE 3 : Verdict cross-phases ===\n")
        
        verdict = run_verdict_cross_phases(regime_profile=args.regime_profile)
        
        # Write rapports JSON + TXT
        output_json = Path('reports/verdict_cross_phases.json')
        output_txt = Path('reports/verdict_cross_phases.txt')
        
        write_verdict_report(verdict, output_json)
        write_verdict_report_txt(verdict, output_txt)
        
        return
    
    # ==========================================================================
    # Validation phase
    # ==========================================================================
    if args.phase is None:
        parser.error("Phase obligatoire (sauf --verdict sans phase)")
    
    phase = args.phase
    yaml_path = Path(f'configs/phases/poc/{phase}.yaml')
    
    if not yaml_path.exists():
        print(f"❌ Config introuvable: {yaml_path}")
        return
    
    # ==========================================================================
    # MODE 2 : Verdict single-phase
    # ==========================================================================
    if args.verdict:
        print(f"=== MODE 2 : Verdict single-phase ({phase}) ===\n")
        
        parquet_path = Path(f'data/results/{phase}.parquet')
        
        if not parquet_path.exists():
            print(f"❌ Parquet introuvable: {parquet_path}")
            print(f"   Lancer d'abord: python -m batch {phase}")
            return
        
        verdict = run_verdict_from_parquet(parquet_path, regime_profile=args.regime_profile)
        
        # Write rapports JSON + TXT
        output_json = Path(f'reports/verdict_{phase}.json')
        output_txt = Path(f'reports/verdict_{phase}.txt')
        
        write_verdict_report(verdict, output_json)
        write_verdict_report_txt(verdict, output_txt)
        
        return
    
    # ==========================================================================
    # MODE 1 : Run + verdict auto
    # ==========================================================================
    print(f"=== MODE 1 : Run + Verdict auto ({phase}) ===\n")
    
    # Run batch
    results = run_batch(
        yaml_path=yaml_path,
        auto_confirm=args.auto_confirm,
        verbose=args.verbose,
    )
    
    # Verdict auto si success
    if results['n_success'] > 0:
        print("\n" + "="*60)
        print("VERDICT AUTO (RAM)")
        print("="*60)
        
        verdict = run_verdict_intra(
            results['rows'],
            regime_profile=args.regime_profile,
            n_skipped_batch=results['n_skipped'],
        )
        
        # Write rapports JSON + TXT
        output_json = Path(f'reports/verdict_{phase}.json')
        output_txt = Path(f'reports/verdict_{phase}.txt')
        
        write_verdict_report(verdict, output_json)
        write_verdict_report_txt(verdict, output_txt)
    else:
        print("\n⚠️  Aucune observation valide — verdict skipped")


if __name__ == '__main__':
    main()
