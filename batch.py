"""
batch.py — Point d'entrée CLI PRC.

Modes :
  Mode 1 : Run batch               python -m batch phase_name
  Mode 2 : Verdict single-phase    python -m batch --verdict phase_name
  Mode 3 : Verdict cross-phases    python -m batch --verdict
  Mode 4 : Verdict multi-parquets  python -m batch --verdict p1 p2 p3

@ROLE    Point d'entrée CLI — route vers running (mode 1) ou analysing (modes 2-4)
@LAYER   root
"""

import warnings
warnings.filterwarnings('ignore', message='.*SLASCLS.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
warnings.filterwarnings('ignore', category=FutureWarning, module='jax')
warnings.filterwarnings('ignore', message='All-NaN slice encountered')
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice')
warnings.filterwarnings('ignore', message='invalid value encountered in subtract')

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_command_buffer='
os.environ["JAX_PLATFORMS"] = "cuda,cpu"

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='PRC Pipeline — Running & Analysing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python -m batch test_baseline                         # run phase
  python -m batch test_baseline --auto-confirm          # run sans confirmation
  python -m batch --verdict test_baseline               # verdict single-phase
  python -m batch --verdict test_baseline --plot         # verdict + PNG
  python -m batch --verdict                             # verdict cross-phases (tous parquets)
  python -m batch --verdict p1 p2 p3                    # verdict multi-parquets
  python -m batch --verdict p1 --pool pool.yaml         # verdict avec filtre pool
        """
    )

    parser.add_argument(
        'phase',
        nargs='*',
        help='Nom(s) de phase. Mode run : 1 phase. Mode verdict : 0-N phases.'
    )
    parser.add_argument('--verdict',      action='store_true',
                        help='Mode analysing (skip run)')
    parser.add_argument('--auto-confirm', action='store_true',
                        help='Skip confirmation dry-run')
    parser.add_argument('--verbose',      action='store_true',
                        help='Détails peeling / debug')
    parser.add_argument('--cfg',          type=str, default=None,
                        help='Config analysing YAML (défaut: analysing/configs/analysing.yaml)')
    parser.add_argument('--pool',         type=str, default=None,
                        help='YAML pool atomics (filtrage multi-parquets)')
    parser.add_argument('--plot',         action='store_true',
                        help='Générer PNG (verdict uniquement)')

    args = parser.parse_args()

    # =====================================================================
    # MODE VERDICT (analysing)
    # =====================================================================
    if args.verdict:
        from analysing.hub import run_analysing

        results_dir = Path('data/results')
        output_dir = results_dir / 'reports'
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg_path = Path(args.cfg) if args.cfg else None
        pool_yaml = Path(args.pool) if args.pool else None

        if args.phase and len(args.phase) == 1:
            # Mode 2 — verdict single-phase
            phase_name = args.phase[0]
            parquet_path = results_dir / f'{phase_name}.parquet'
            if not parquet_path.exists():
                print(f"\n  Parquet introuvable : {parquet_path}\n")
                return

            run_analysing(
                source=parquet_path,
                cfg_path=cfg_path,
                output_dir=output_dir / phase_name,
                pool_yaml=pool_yaml,
                label=phase_name,
                plot=args.plot,
                verbose=args.verbose,
            )

        elif args.phase and len(args.phase) > 1:
            # Mode 4 — verdict multi-parquets explicites
            parquet_paths = []
            for name in args.phase:
                p = results_dir / f'{name}.parquet'
                if not p.exists():
                    print(f"  Parquet introuvable : {p}")
                    return
                parquet_paths.append(p)

            multi_label = '_'.join(args.phase)

            run_analysing(
                source=parquet_paths,
                cfg_path=cfg_path,
                output_dir=output_dir / multi_label,
                pool_yaml=pool_yaml,
                label=multi_label,
                plot=args.plot,
                verbose=args.verbose,
            )

        else:
            # Mode 3 — verdict cross-phases (tous les parquets du répertoire)
            run_analysing(
                source=results_dir,
                cfg_path=cfg_path,
                output_dir=output_dir / 'all',
                pool_yaml=pool_yaml,
                label='all',
                plot=args.plot,
                verbose=args.verbose,
            )

        return

    # =====================================================================
    # MODE RUN (running)
    # =====================================================================
    if not args.phase or len(args.phase) != 1:
        parser.error("Phase obligatoire en mode run (ex: python -m batch phase_name)")

    phase_name = args.phase[0]
    yaml_path = Path(f'configs/phases/{phase_name}.yaml')
    if not yaml_path.exists():
        print(f"\n  Config introuvable : {yaml_path}")
        print(f"  Créer configs/phases/{phase_name}.yaml\n")
        return

    from running.hub import run_phase

    run_phase(
        yaml_path=yaml_path,
        output_dir=Path('data/results'),
        auto_confirm=args.auto_confirm,
    )


if __name__ == '__main__':
    main()
