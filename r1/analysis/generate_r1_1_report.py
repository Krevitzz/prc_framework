
# Script: scripts/r1_analysis/generate_r1_1_report.py

"""
Génère rapport synthétique Phase R1.1.

CONFORMITÉ:
- Délégation report_writers.py (si formatage complexe)
- Format output similaire verdict_reporter.py
"""

import json
from pathlib import Path
from datetime import datetime

INPUT_BASELINES = Path("outputs/r1_analysis/baselines_r0.json")
INPUT_CLASSIFICATION = Path("outputs/r1_analysis/classification_sequences_r1.json")
INPUT_LOG_N2 = Path("outputs/r1_execution/execution_n2_log.json")
INPUT_LOG_N3 = Path("outputs/r1_execution/execution_n3_log.json")

OUTPUT_DIR = Path("reports/r1_1")


def generate_report():
    """
    Génère rapport R1.1 complet.
    
    SECTIONS:
    1. Métadonnées (timestamp, configs, volumes)
    2. Baselines R0 (référence)
    3. Résultats R1.1 (classification séquences)
    4. Comparaisons R0 vs R1
    5. Décision Go/No-Go R1.2
    """
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = OUTPUT_DIR / f"{timestamp}_report_r1_1"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Charger données
    with open(INPUT_BASELINES) as f:
        baselines_r0 = json.load(f)
    
    with open(INPUT_CLASSIFICATION) as f:
        classification = json.load(f)
    
    with open(INPUT_LOG_N2) as f:
        log_n2 = json.load(f)
    
    with open(INPUT_LOG_N3) as f:
        log_n3 = json.load(f)
    
    # =========================================================================
    # RAPPORT TXT (humain)
    # =========================================================================
    
    with open(report_dir / "summary_r1_1.txt", 'w') as f:
        f.write("="*70 + "\n")
        f.write("RAPPORT PHASE R1.1 - COMPOSITION ROBUSTE\n")
        f.write(f"{timestamp}\n")
        f.write("="*70 + "\n\n")
        
        # 1. Métadonnées
        f.write("1. MÉTADONNÉES\n")
        f.write("-"*70 + "\n")
        f.write(f"Phase:           R1.1\n")
        f.write(f"Params config:   params_default_v1\n")
        f.write(f"Timestamp:       {timestamp}\n")
        f.write(f"\nVolumes exécutés:\n")
        f.write(f"  Séquences n=2: {log_n2['n_sequences_executed']}\n")
        f.write(f"  Séquences n=3: {log_n3['n_sequences_executed']}\n")
        f.write(f"  Total:         {log_n2['n_sequences_executed'] + log_n3['n_sequences_executed']}\n")
        f.write(f"\n  Runs n=2:      {log_n2['n_runs_completed']}\n")
        f.write(f"  Runs n=3:      {log_n3['n_runs_completed']}\n")
        f.write(f"  Total:         {log_n2['n_runs_completed'] + log_n3['n_runs_completed']}\n")
        f.write("\n")
        
        # 2. Baselines R0
        f.write("2. BASELINES R0 (RÉFÉRENCE)\n")
        f.write("-"*70 + "\n")
        f.write(f"Observations:       {baselines_r0['n_observations']}\n")
        f.write(f"Rejection rate:     {baselines_r0['rejection_rate']:.4%}\n")
        f.write(f"Explosion rate:     {baselines_r0['explosion_rate']:.4%}\n")
        f.write(f"Regime concordance: {baselines_r0['regime_concordance']:.2%}\n")
        f.write("\n")
        
        # 3. Résultats R1.1
        f.write("3. RÉSULTATS R1.1\n")
        f.write("-"*70 + "\n")
        f.write(f"Total séquences analysées: {classification['n_sequences_total']}\n\n")
        
        f.write("Classification:\n")
        f.write(f"  ROBUSTES:    {classification['n_sequences_robustes']:3d} ({classification['n_sequences_robustes']/classification['n_sequences_total']*100:5.1f}%)\n")
        f.write(f"  DÉGRADANTES: {classification['n_sequences_degradantes']:3d} ({classification['n_sequences_degradantes']/classification['n_sequences_total']*100:5.1f}%)\n")
        f.write(f"  INSTABLES:   {classification['n_sequences_instables']:3d} ({classification['n_sequences_instables']/classification['n_sequences_total']*100:5.1f}%)\n")
        f.write("\n")
        
        f.write("Métriques globales R1:\n")
        f.write(f"  Explosion rate: {classification['metrics']['explosion_rate_r1']:.4%}\n")
        f.write(f"  Concordance:    {classification['metrics']['concordance_r1']:.2%}\n")
        f.write("\n")
        
        # 4. Comparaisons R0 vs R1
        f.write("4. COMPARAISONS R0 vs R1\n")
        f.write("-"*70 + "\n")
        
        explosion_ratio = classification['metrics']['explosion_rate_r1'] / baselines_r0['explosion_rate']
        f.write(f"Explosion rate:\n")
        f.write(f"  R0:    {baselines_r0['explosion_rate']:.4%}\n")
        f.write(f"  R1:    {classification['metrics']['explosion_rate_r1']:.4%}\n")
        f.write(f"  Ratio: {explosion_ratio:.2f}x\n")
        
        if explosion_ratio <= 2.0:
            f.write(f"  ✓ Acceptable (≤ 2× baseline)\n")
        else:
            f.write(f"  ⚠ Élevé (> 2× baseline)\n")
        f.write("\n")
        
        f.write(f"Concordance régimes:\n")
        f.write(f"  R0: {baselines_r0['regime_concordance']:.2%}\n")
        f.write(f"  R1: {classification['metrics']['concordance_r1']:.2%}\n")
        
        if classification['metrics']['concordance_r1'] >= 0.95:
            f.write(f"  ✓ Acceptable (≥ 95%)\n")
        else:
            f.write(f"  ⚠ Dégradé (< 95%)\n")
        f.write("\n")
        
        # 5. Décision Go/No-Go
        f.write("5. DÉCISION GO/NO-GO R1.2\n")
        f.write("-"*70 + "\n")
        
        # Critères succès (feuille_de_route Section 3.1)
        criterion_1 = (classification['n_sequences_robustes'] / classification['n_sequences_total']) >= 0.30
        criterion_2 = explosion_ratio <= 2.0
        criterion_3 = classification['metrics']['concordance_r1'] >= 0.95
        
        f.write("Critères succès Phase R1.1:\n")
        f.write(f"  [{'✓' if criterion_1 else '✗'}] ≥30% séquences ROBUSTES: {classification['n_sequences_robustes']/classification['n_sequences_total']*100:.1f}%\n")
        f.write(f"  [{'✓' if criterion_2 else '✗'}] Explosion rate ≤ 2× R0:  {explosion_ratio:.2f}x\n")
        f.write(f"  [{'✓' if criterion_3 else '✗'}] Concordance ≥ 95%:       {classification['metrics']['concordance_r1']:.2%}\n")
        f.write("\n")
        
        all_criteria_met = criterion_1 and criterion_2 and criterion_3
        
        if all_criteria_met:
            decision = "GO"
            rationale = "Tous critères succès Phase R1.1 satisfaits. Procéder Phase R1.2 (Compensation)."
        else:
            decision = "NO-GO"
            failed = []
            if not criterion_1:
                failed.append("Taux séquences robustes < 30%")
            if not criterion_2:
                failed.append("Explosion rate > 2× baseline")
            if not criterion_3:
                failed.append("Concordance régimes < 95%")
            
            rationale = f"Critères non satisfaits: {', '.join(failed)}. Skip Phase R1.2, procéder directement Phase R1.3."
        
        f.write(f"DÉCISION: {decision}\n")
        f.write(f"Rationale: {rationale}\n")
        f.write("\n")
        
        f.write("="*70 + "\n")
    
    # =========================================================================
    # RAPPORT JSON (machine-readable)
    # =========================================================================
    
    report_json = {
        'timestamp': timestamp,
        'phase': 'R1.1',
        'params_config_id': 'params_default_v1',
        'volumes': {
            'n_sequences_n2': log_n2['n_sequences_executed'],
            'n_sequences_n3': log_n3['n_sequences_executed'],
            'n_runs_n2': log_n2['n_runs_completed'],
            'n_runs_n3': log_n3['n_runs_completed']
        },
        'baselines_r0': baselines_r0,
        'results_r1': classification,
        'comparisons': {
            'explosion_rate_ratio': explosion_ratio,
            'concordance_delta': classification['metrics']['concordance_r1'] - baselines_r0['regime_concordance']
        },
        'criteria': {
            'robustness_rate': {
                'value': classification['n_sequences_robustes'] / classification['n_sequences_total'],
                'threshold': 0.30,
                'met': criterion_1
            },
            'explosion_rate_ratio': {
                'value': explosion_ratio,
                'threshold': 2.0,
                'met': criterion_2
            },
            'concordance_r1': {
                'value': classification['metrics']['concordance_r1'],
                'threshold': 0.95,
                'met': criterion_3
            }
        },
        'decision': {
            'go_no_go': decision,
            'rationale': rationale,
            'all_criteria_met': all_criteria_met
        }
    }
    
    with open(report_dir / "report_r1_1.json", 'w') as f:
        json.dump(report_json, f, indent=2)
    
    print(f"\n✓ Rapport R1.1 généré: {report_dir}")
    print(f"\nDÉCISION: {decision}")
    print(f"Rationale: {rationale}")
    
    return decision, report_dir


if __name__ == "__main__":
    decision, report_dir = generate_report()