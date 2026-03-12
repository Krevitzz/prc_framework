"""
batch.py

Point d'entrée CLI — PRC v7 JAX.

Modes :
  Mode 1 : Run batch               python -m batch poc_v7
  Mode 2 : Verdict single-phase    python -m batch --verdict poc_v7
  Mode 3 : Verdict cross-phases    python -m batch --verdict
"""

# Filtres warnings — AVANT tout import (y compris JAX/numpy)
import warnings
warnings.filterwarnings('ignore', message='.*SLASCLS.*')

# Env XLA — AVANT tout import JAX
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import argparse
from pathlib import Path

from running.hub_running_new import run_phase
from analysing.hub_analysing_v2 import (
    run_verdict_from_parquet,
    run_verdict_cross_phases,
)


def main():
    parser = argparse.ArgumentParser(
        description='PRC v7 JAX Batch Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python -m batch poc_v7
  python -m batch poc_v7 --auto-confirm
  python -m batch poc_v7 --verbose
        """
    )

    parser.add_argument(
        'phase',
        nargs='?',
        help='Nom de phase (ex: poc_v7). Obligatoire sauf --verdict sans phase.'
    )
    parser.add_argument('--verdict',      action='store_true',
                        help='Mode verdict uniquement (skip run)')
    parser.add_argument('--verbose',      action='store_true',
                        help='Afficher détails FAIL')
    parser.add_argument('--auto-confirm', action='store_true',
                        help='Skip confirmation dry-run (tests / CI)')
    parser.add_argument('--cfg',          type=str, default=None,
                        help='Config analysing YAML (défaut: analysing/configs/analysing_default.yaml)')
    parser.add_argument('--plot',         action='store_true',
                        help='Générer PNG t-SNE (verdict uniquement)')
    parser.add_argument('--debug',        action='store_true',
                        help='Sauvegarder labels + trace peeling (verdict uniquement)')

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # MODE 2 / 3 : Verdict
    # -------------------------------------------------------------------------
    if args.verdict:
        output_dir = Path('data/results')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Résolution cfg — chemin absolu depuis CWD
        _default_cfg = Path('analysing/configs/analysing_default.yaml').resolve()
        cfg_path     = Path(args.cfg).resolve() if args.cfg else _default_cfg

        if args.phase:
            # Mode 2 — verdict single-phase depuis parquet
            parquet_path = Path('data/results') / f'{args.phase}.parquet'
            if not parquet_path.exists():
                print(f"\n❌  Parquet introuvable : {parquet_path}\n")
                return
            run_verdict_from_parquet(
                parquet_path = parquet_path,
                cfg_path     = cfg_path,
                output_dir   = output_dir / 'reports' / args.phase,
                plot         = args.plot,
                save_debug   = args.debug,
            )
        else:
            # Mode 3 — verdict cross-phases
            run_verdict_cross_phases(
                results_dir = Path('data/results'),
                cfg_path    = cfg_path,
                output_dir  = output_dir / 'reports',
                plot        = args.plot,
            )
        return

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    if args.phase is None:
        parser.error("Phase obligatoire (ex: python -m batch poc_v7)")

    yaml_path = Path(f'configs/phases/poc/{args.phase}.yaml')
    if not yaml_path.exists():
        print(f"\n❌  Config introuvable : {yaml_path}")
        print(f"    Créer configs/phases/poc/{args.phase}.yaml\n")
        return

    # -------------------------------------------------------------------------
    # MODE 1 : Run batch
    # -------------------------------------------------------------------------
    run_phase(
        yaml_path    = yaml_path,
        output_dir   = Path('data/results'),
        auto_confirm = args.auto_confirm,
        verbose      = args.verbose,
    )


if __name__ == '__main__':
    main()
