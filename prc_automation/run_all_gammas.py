#!/usr/bin/env python3
"""
Script pour exécuter --test et --verdict sur tous les gammas disponibles.
"""

import subprocess
import sys
from pathlib import Path

# Liste de tous les gammas disponibles dans le registre
ALL_GAMMAS = [
    "GAM-001", "GAM-002", "GAM-003", "GAM-004", "GAM-005",
    "GAM-006", "GAM-007", "GAM-008", "GAM-009", "GAM-010",
     "GAM-012", "GAM-013"
]

def run_for_all_gammas(config_id="weights_default", thresholds_id="thresholds_default"):
    """Exécute --test puis --verdict pour tous les gammas."""
    
    for gamma_id in ALL_GAMMAS:
        print(f"\n{'='*80}")
        print(f"Traitement de {gamma_id}")
        print(f"{'='*80}")
        
        # 1. Mode --test
        print(f"\n[1/2] Exécution --test pour {gamma_id}")
        cmd_test = [
            sys.executable, "batch_runner.py",
            "--test",
            "--gamma", gamma_id,
            "--config", config_id,
            "--verbose"
        ]
        
        result_test = subprocess.run(cmd_test, cwd=Path(__file__).parent)
        if result_test.returncode != 0:
            print(f"⚠ Attention: --test pour {gamma_id} a retourné code {result_test.returncode}")
        
        # 2. Mode --verdict
        print(f"\n[2/2] Exécution --verdict pour {gamma_id}")
        cmd_verdict = [
            sys.executable, "batch_runner.py",
            "--verdict",
            "--gamma", gamma_id,
            "--config", config_id,
            "--thresholds", thresholds_id,
            "--verbose"
        ]
        
        result_verdict = subprocess.run(cmd_verdict, cwd=Path(__file__).parent)
        if result_verdict.returncode != 0:
            print(f"⚠ Attention: --verdict pour {gamma_id} a retourné code {result_verdict.returncode}")
    
    print(f"\n{'='*80}")
    print("Tous les gammas ont été traités!")
    print(f"{'='*80}")

if __name__ == "__main__":
    # Vous pouvez passer des arguments en ligne de commande
    config = sys.argv[1] if len(sys.argv) > 1 else "weights_default"
    thresholds = sys.argv[2] if len(sys.argv) > 2 else "thresholds_default"
    
    run_for_all_gammas(config, thresholds)