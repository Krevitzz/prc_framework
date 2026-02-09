
# Script: scripts/r1_2_compensation/generate_r1_2_report.py

"""
Génère rapport Phase R1.2 (compensation).

CONFORMITÉ:
- Pattern similaire generate_r1_1_report.py
- Formulations compatibilistes strictes
"""

import json
from pathlib import Path
from datetime import datetime

INPUT_ANALYSIS = Path("outputs/r1_2_compensation/compensation_analysis.json")
INPUT_LOG = Path("outputs/r1_2_compensation/execution_compensated_log.json")

OUTPUT_DIR = Path("reports/r1_2")


def generate_report():
    """Génère rapport R1.2 complet."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = OUTPUT_DIR / f"{timestamp}_report_r1_2"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Charger données
    with open(INPUT_ANALYSIS) as f:
        analysis = json.load(f)
    
    with open(INPUT_LOG) as f:
        log = json.load(f)
    
    # =========================================================================
    # RAPPORT TXT
    # =========================================================================
    
    with open(report_dir / "summary_r1_2.txt", 'w') as f:
        f.write("="*70 + "\n")
        f.write("RAPPORT PHASE R1.2 - COMPENSATION INSTABILITÉS\n")
        f.write(f"{timestamp}\n")
        f.write("="*70 + "\n\n")
        
        # 1. Métadonnées
        f.write("1. MÉTADONNÉES\n")
        f.write("-"*70 + "\n")
        f.write(f"Phase:           R1.2\n")
        f.write(f"Type:            Séquences compensées\n")
        f.write(f"Timestamp:       {timestamp}\n")
        f.write(f"\nVolumes exécutés:\n")
        f.write(f"  Séquences:     {log['n_sequences_executed']}\n")
        f.write(f"  Runs:          {log['n_runs_completed']}\n")
        f.write("\n")
        
        # 2. Résultats compensations
        f.write("2. RÉSULTATS COMPENSATIONS\n")
        f.write("-"*70 + "\n")
        f.write(f"Total séquences analysées: {analysis['n_sequences_total']}\n\n")
        
        f.write("Classification:\n")
        f.write(f"  SUCCESS: {analysis['n_compensations_success']:3d} ({analysis['compensation_rate']*100:5.1f}%)\n")
        f.write(f"  FAILED:  {analysis['n_compensations_failed']:3d} ({(1-analysis['compensation_rate'])*100:5.1f}%)\n")
        f.write("\n")
        
        # 3. Patterns identifiés
        f.write("3. PATTERNS IDENTIFIÉS\n")
        f.write("-"*70 + "\n")
        
        f.write(f"Gammas explosifs compensables ({len(analysis['compensable_explosive_gammas'])}):\n")
        for gid in analysis['compensable_explosive_gammas']:
            f.write(f"  - {gid}\n")
        f.write("\n")
        
        f.write(f"Gammas encadrants efficaces ({len(analysis['effective_framing_gammas'])}):\n")
        for gid in analysis['effective_framing_gammas']:
            f.write(f"  - {gid}\n")
        f.write("\n")
        
        # 4. Décision hypothèse explosions
        f.write("4. DÉCISION HYPOTHÈSE EXPLOSIONS\n")
        f.write("-"*70 + "\n")
        
        f.write("Hypothèses testées:\n")
        f.write("  H_LOCAL:  Explosions = instabilités locales compensables\n")
        f.write("  H_GLOBAL: Explosions = exclusions nécessaires (non compensables)\n")
        f.write("\n")
        
        # Critère décision
        if analysis['compensation_rate'] >= 0.10:
            hypothesis_verdict = "H_LOCAL compatible"
            hypothesis_rationale = (
                f"Taux compensation {analysis['compensation_rate']*100:.1f}% ≥ 10% seuil.\n"
                f"    Les explosions R0 peuvent être partiellement compensées par composition.\n"
                f"    Les {len(analysis['compensable_explosive_gammas'])} gammas explosifs identifiés comme compensables\n"
                f"    suggèrent que ces configurations ne sont pas définitivement inadmissibles."
            )
        else:
            hypothesis_verdict = "H_GLOBAL compatible"
            hypothesis_rationale = (
                f"Taux compensation {analysis['compensation_rate']*100:.1f}% < 10% seuil.\n"
                f"    La majorité des explosions R0 ne sont pas compensables par composition.\n"
                f"    Les 26 combinaisons explosives R0 restent définitivement inadmissibles."
            )
        
        f.write(f"VERDICT: {hypothesis_verdict}\n")
        f.write(f"Rationale:\n    {hypothesis_rationale}\n")
        f.write("\n")
        
        f.write("IMPORTANT (Checklist Glissement 1):\n")
        f.write("  Ce verdict est une COMPATIBILITÉ, pas une VALIDATION.\n")
        f.write("  Les données R1.2 sont compatibles avec l'hypothèse, mais ne la prouvent pas.\n")
        f.write("  Validation finale nécessiterait tests cross-domaine (R2+).\n")
        f.write("\n")
        
        f.write("="*70 + "\n")
    
    # =========================================================================
    # RAPPORT JSON
    # =========================================================================
    
    report_json = {
        'timestamp': timestamp,
        'phase': 'R1.2',
        'type': 'compensation_analysis',
        'volumes': {
            'n_sequences_executed': log['n_sequences_executed'],
            'n_runs_completed': log['n_runs_completed']
        },
        'results': analysis,
        'hypothesis_decision': {
            'verdict': hypothesis_verdict,
            'rationale': hypothesis_rationale,
            'compensation_rate': analysis['compensation_rate'],
            'threshold': 0.10,
            'criterion_met': analysis['compensation_rate'] >= 0.10
        }
    }
    
    with open(report_dir / "report_r1_2.json", 'w') as f:
        json.dump(report_json, f, indent=2)
    
    print(f"\n✓ Rapport R1.2 généré: {report_dir}")
    print(f"\nHYPOTHÈSE: {hypothesis_verdict}")
    print(f"Taux compensation: {analysis['compensation_rate']*100:.1f}%")
    
    return hypothesis_verdict, report_dir


if __name__ == "__main__":
    generate_report()