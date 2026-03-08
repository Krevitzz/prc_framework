"""
batch_jax.py

Point d'entrée CLI — PRC v7 JAX.

Modes :
  Mode 1 : Run batch               python -m batch_jax poc_v7
  Mode 2 : Verdict single-phase    python -m batch_jax --verdict poc_v7   [PARKÉ]
  Mode 3 : Verdict cross-phases    python -m batch_jax --verdict           [PARKÉ]

Modes 2 et 3 parkés — verdict/analysing non portés en v7.
TAG: CLEANUP_PHASE2 (débrancher stub quand verdict_lite disponible)
"""

# Cache XLA disque — doit être configuré AVANT tout import JAX
# Survit aux relances process, invalidé automatiquement si le bytecode change.
import os
os.environ['JAX_COMPILATION_CACHE_DIR'] = str(
    __import__('pathlib').Path(__file__).parent / 'jax_cache'
)
# Supprime les warnings/errors C++ XLA (SIGILL prefer-no-gather, version compat...)
# 0=tout, 1=info+, 2=warning+, 3=error+, 4=fatal uniquement
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '4')

import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', message='.*SLASCLS.*')

from running.hub_running import run_batch_jax

def main():
    parser = argparse.ArgumentParser(
        description='PRC v7 JAX Batch Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python -m batch_jax poc_v7
  python -m batch_jax poc_v7 --auto-confirm
  python -m batch_jax poc_v7 --verbose
        """
    )

    parser.add_argument(
        'phase',
        nargs='?',
        help='Nom de phase (ex: poc_v7). Obligatoire sauf --verdict sans phase.'
    )

    parser.add_argument(
        '--verdict',
        action='store_true',
        help='[PARKÉ v7] Mode verdict uniquement (skip run)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Afficher détails chunks skippés'
    )

    parser.add_argument(
        '--auto-confirm',
        action='store_true',
        help='Skip confirmation dry-run (tests / CI)'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=256,
        help='Taille max d\'un chunk vmap (défaut: 256)'
    )

    parser.add_argument(
        '--flush-every',
        type=int,
        default=1000,
        help='Flush parquet tous les N rows (défaut: 1000)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/results',
        help='Dossier output parquet (défaut: data/results)'
    )

    args = parser.parse_args()

    # =========================================================================
    # MODE 2 / 3 : Verdict — PARKÉ
    # =========================================================================
    if args.verdict:
        print()
        print("⚠  Modes verdict non disponibles en v7.")
        print("   verdict/analysing sera porté en phase 2.")
        print("   TAG: CLEANUP_PHASE2")
        print()
        return

    # =========================================================================
    # Validation phase
    # =========================================================================
    if args.phase is None:
        parser.error("Phase obligatoire (ex: python -m batch_jax poc_v7)")

    phase     = args.phase
    yaml_path = Path(f'configs/phases/poc/{phase}.yaml')

    if not yaml_path.exists():
        print(f"\n❌  Config introuvable : {yaml_path}")
        print(f"    Créer configs/phases/poc/{phase}.yaml")
        return

    # =========================================================================
    # MODE 1 : Run batch
    # =========================================================================
    print(f"\n=== PRC v7 — Run batch : {phase} ===\n")

    result = run_batch_jax(
        yaml_path    = yaml_path,
        output_dir   = Path(args.output_dir),
        auto_confirm = args.auto_confirm,
        chunk_size   = args.chunk_size,
        flush_every  = args.flush_every,
        verbose      = args.verbose,
    )

    print()
    print("=" * 50)
    print(f"  Phase      : {result['phase']}")
    print(f"  OK         : {result.get('n_ok', 0)}")
    print(f"  EXPLOSION  : {result.get('n_explosion', 0)}")
    print(f"  INVALID    : {result.get('n_invalid', 0)}")
    print(f"  FAIL       : {result.get('n_fail', 0)}")
    print(f"  Groupes    : {result.get('n_groups', 0)}")
    if result.get('parquet'):
        print(f"  Parquet    : {result['parquet']}")
    print("=" * 50)
    print()


if __name__ == '__main__':
    main()