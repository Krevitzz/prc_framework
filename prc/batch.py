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
        '--debug',
        action='store_true',
        help='Sauvegarder fichiers intermédiaires debug (JSON, labels, trace). Implique --verdict.'
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
        
        out_dir = Path('reports') / 'cross'
        out_dir.mkdir(parents=True, exist_ok=True)
        write_verdict_report_txt(verdict, out_dir / 'verdict.txt')
        if args.debug:
            write_verdict_report(verdict, out_dir / 'verdict.json')
        
        return
    
    # ==========================================================================
    # Validation phase
    # ==========================================================================
    # --debug implique --verdict (relance verdict uniquement, pas le run)
    if args.debug and not args.verdict:
        args.verdict = True

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
        
        out_dir = Path('reports') / phase
        out_dir.mkdir(parents=True, exist_ok=True)
        verdict = run_verdict_from_parquet(
            parquet_path,
            regime_profile=args.regime_profile,
            output_dir=str(out_dir),
            label=phase,
            debug=args.debug,
        )
        write_verdict_report_txt(verdict, out_dir / 'verdict.txt')
        if args.debug:
            write_verdict_report(verdict, out_dir / 'verdict.json')
        
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
        
        out_dir = Path('reports') / phase
        out_dir.mkdir(parents=True, exist_ok=True)
        verdict = run_verdict_intra(
            results['rows'],
            regime_profile=args.regime_profile,
            n_skipped_batch=results['n_skipped'],
            output_dir=str(out_dir),
            label=phase,
            debug=args.debug,
        )
        write_verdict_report_txt(verdict, out_dir / 'verdict.txt')
        if args.debug:
            write_verdict_report(verdict, out_dir / 'verdict.json')
    else:
        print("\n⚠️  Aucune observation valide — verdict skipped")


if __name__ == '__main__':
    main()