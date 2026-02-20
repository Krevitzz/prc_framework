"""
prc.batch

Responsabilité : Façade racine — point d'entrée utilisateur → running/hub.py

Usage : python -m prc.batch <yaml_path>
"""

import sys
from pathlib import Path

from running.hub import run_batch


def main():
    """Point d'entrée CLI."""
    if len(sys.argv) < 2:
        print("Usage: python -m batch <phase>")
        print("Exemple: python -m batch poc")
        sys.exit(1)
    
    phase = sys.argv[1]
    yaml_path = Path(f'configs/phases/{phase}/{phase}.yaml')
    
    if not yaml_path.exists():
        print(f"Erreur: config phase introuvable {yaml_path}")
        sys.exit(1)
    
    # Execute batch
    result = run_batch(yaml_path)
    
    # Exit code selon résultats
    if result['n_skipped'] > 0:
        sys.exit(2)  # Warnings (certains skipped)
    
    sys.exit(0)  # Success


if __name__ == '__main__':
    main()
