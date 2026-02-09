
# Script: scripts/r1_3_convergence/generate_r1_3_report.py

"""
Génère rapport final Phase R1.3 (convergence).

CONFORMITÉ:
- Pattern similaire generate_r1_1_report.py, generate_r1_2_report.py
- Classification issue (A/B/C) factuelle
- Formulations conformes Checklist Glissement 3
"""

import json
from pathlib import Path
from datetime import datetime

# Inputs
INPUT_CONSTRAINTS = Path("outputs/r1_3_convergence/constraints_compiled.json")
INPUT_SCORES = Path("outputs/r1_3_convergence/gamma_scores.json")
INPUT_SURVIVORS = Path("outputs/r1_3_convergence/survivors.json")
INPUT_STRUCTURE = Path("outputs/r1_3_convergence/convergent_structure.json")

OUTPUT_DIR = Path("reports/r1_3")


def classify_issue(structure_type: str, n_survivors: int) -> Tuple[str, str]:
    """
    Classifie issue R1 (A/B/C).
    
    CRITÈRES (feuille_de_route Section 2.4):
    - Issue A (Réduction réussie):
      * A1 (fort): structure_type in ['UNIQUE', 'TIGHT_CLASS']
      * A2 (partiel): structure_type in ['PARAMETRIC_FAMILY', 'TOPOLOGICAL_FAMILY']
      * A3 (partiel): structure_type == 'CORE_EXTENSIONS'
    
    - Issue B (Dégradation): structure_type == 'MULTIPLE_UNCORRELATED'
    
    - Issue C (Incompatibilité): structure_type == 'NONE'
    
    Args:
        structure_type: Type structure détectée
        n_survivors: Nombre survivants
    
    Returns:
        (issue, rationale)
    """
    if structure_type == 'NONE':
        return 'C', 'Incompatibilité contraintes (aucun gamma survivant)'
    
    if structure_type == 'UNIQUE':
        return 'A1', f'Réduction réussie (forte): Gamma unique identifié'
    
    if structure_type == 'TIGHT_CLASS':
        return 'A1', f'Réduction réussie (forte): Classe étroite ({n_survivors} gammas, même famille)'
    
    if structure_type == 'PARAMETRIC_FAMILY':
        return 'A2', f'Réduction réussie (partielle): Famille paramétrée ({n_survivors} gammas)'
    
    if structure_type == 'TOPOLOGICAL_FAMILY':
        return 'A2', f'Réduction réussie (partielle): Famille topologique ({n_survivors} gammas, dépendance D)'
    
    if structure_type == 'CORE_EXTENSIONS':
        return 'A3', f'Réduction réussie (partielle): Noyau + extensions ({n_survivors} gammas)'
    
    if structure_type == 'MULTIPLE_UNCORRELATED':
        return 'B', f'Dégradation: Multiple gammas non corrélés ({n_survivors}), pas de structure simple'
    
    # Fallback
    return 'UNKNOWN', f'Structure non classifiable: {structure_type}'


def generate_report():
    """Génère rapport R1.3 complet."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = OUTPUT_DIR / f"{timestamp}_report_r1_3"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Charger données
    with open(INPUT_CONSTRAINTS) as f:
        constraints = json.load(f)
    
    with open(INPUT_SCORES) as f:
        scores = json.load(f)
    
    with open(INPUT_SURVIVORS) as f:
        survivors_data = json.load(f)
    
    with open(INPUT_STRUCTURE) as f:
        structure = json.load(f)
    
    # Classifier issue
    issue, rationale = classify_issue(
        structure['structure_type'],
        structure['n_survivors']
    )
    
    # =========================================================================
    # RAPPORT TXT
    # =========================================================================
    
    with open(report_dir / "summary_r1_3.txt", 'w') as f:
        f.write("="*70 + "\n")
        f.write("RAPPORT PHASE R1.3 - CONVERGENCE CANDIDATS\n")
        f.write(f"{timestamp}\n")
        f.write("="*70 + "\n\n")
        
        # 1. Métadonnées
        f.write("1. MÉTADONNÉES\n")
        f.write("-"*70 + "\n")
        f.write(f"Phase:           R1.3\n")
        f.write(f"Timestamp:       {timestamp}\n")
        f.write(f"\nContraintes compilées:\n")
        f.write(f"  Gammas analysés: {len(constraints)}\n")
        f.write(f"  Phases incluses: R0, R1.1")
        
        # Vérifier R1.2
        has_r1_2 = any('r1_2' in c for c in constraints.values())
        if has_r1_2:
            f.write(", R1.2")
        f.write("\n\n")
        
        # 2. Scoring
        f.write("2. SCORING ADMISSIBILITÉ\n")
        f.write("-"*70 + "\n")
        f.write(f"Gammas scorés:   {len(scores)}\n")
        f.write(f"Threshold:       {survivors_data['threshold']}\n")
        f.write(f"\nTop 10 gammas:\n")
        
        ranked = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
        for rank, (gamma_id, data) in enumerate(ranked[:10], 1):
            f.write(f"  {rank:2d}. {gamma_id}: {data['score']:.3f}\n")
        f.write("\n")
        
        # 3. Réduction
        f.write("3. RÉDUCTION ESPACE CANDIDAT\n")
        f.write("-"*70 + "\n")
        stats = survivors_data['stats']
        f.write(f"Candidats initiaux: {stats['n_candidates_initial']}\n")
        f.write(f"Survivants:         {stats['n_survivors']}\n")
        f.write(f"Facteur réduction:  {stats['reduction_factor']:.3f} ({stats['reduction_factor']*100:.1f}%)\n")
        f.write(f"\nSurvivants:\n")
        
        for gamma_id in structure['survivors']:
            score = scores[gamma_id]['score']
            rank = scores[gamma_id]['rank']
            f.write(f"  {rank:2d}. {gamma_id}: {score:.3f}\n")
        f.write("\n")
        
        # 4. Structure convergente
        f.write("4. STRUCTURE CONVERGENTE\n")
        f.write("-"*70 + "\n")
        f.write(f"Type:        {structure['structure_type']}\n")
        f.write(f"Description: {structure['description']}\n")
        f.write(f"\nPropriétés communes:\n")
        
        common = structure['common_properties']
        f.write(f"  Familles ({common['n_families']}): {', '.join(common['families'])}\n")
        
        if common['d_applicability_common']:
            f.write(f"  d_applicability commune: {', '.join(common['d_applicability_common'])}\n")
        else:
            f.write(f"  d_applicability commune: Aucune\n")
        f.write("\n")
        
        # 5. Classification issue
        f.write("5. CLASSIFICATION ISSUE R1\n")
        f.write("-"*70 + "\n")
        f.write(f"ISSUE: {issue}\n")
        f.write(f"Rationale: {rationale}\n")
        f.write("\n")
        
        # Interprétation par issue
        f.write("INTERPRÉTATION:\n")
        
        if issue.startswith('A'):
            f.write("  ✓ Réduction effective vers structure simple.\n")
            f.write("  ✓ Contraintes R0 + R1 compatibles avec existence Γ restreint.\n")
            
            if issue == 'A1':
                f.write("  → Prochaine étape: Tests prédictifs cross-domaine (R2+).\n")
            elif issue == 'A2':
                f.write("  → Prochaine étape: Caractériser dépendances paramètres/topologie.\n")
            elif issue == 'A3':
                f.write("  → Prochaine étape: Isoler noyau Γ_core vs corrections.\n")
        
        elif issue == 'B':
            f.write("  ⚠ Dégradation: Réduction limitée, pas de structure simple identifiable.\n")
            f.write("  → Limites composition révélées (échelle dégradation caractérisée).\n")
            f.write("  → Prochaine étape: Contraintes raffinées, ajuster protocole R2.\n")
        
        elif issue == 'C':
            f.write("  ✗ Incompatibilité: Aucune réduction possible.\n")
            f.write("  → Tensions structurelles identifiées entre contraintes.\n")
            f.write("  → Prochaine étape: Réviser hypothèses framework ou identifier sources tensions.\n")
        
        f.write("\n")
        
        # IMPORTANT (Checklist Glissement 3)
        f.write("IMPORTANT (Checklist Anti-Glissement):\n")
        f.write("  Cette classification est basée sur RÉDUCTION ESPACE, pas UNICITÉ présupposée.\n")
        f.write("  Les trois issues (A/B/C) sont scientifiquement valides.\n")
        f.write("  Issue A n'est PAS \"succès\" absolu, ni B/C \"échecs\".\n")
        f.write("  Chaque issue produit connaissance scientifique exploitable.\n")
        f.write("\n")
        
        f.write("="*70 + "\n")
    
    # =========================================================================
    # RAPPORT JSON
    # =========================================================================
    
    report_json = {
        'timestamp': timestamp,
        'phase': 'R1.3',
        'type': 'convergence_analysis',
        'constraints': {
            'n_gammas_analyzed': len(constraints),
            'phases_included': ['R0', 'R1.1'] + (['R1.2'] if has_r1_2 else [])
        },
        'scoring': {
            'n_gammas_scored': len(scores),
            'threshold': survivors_data['threshold'],
            'top_10': [
                {'rank': rank, 'gamma_id': gamma_id, 'score': data['score']}
                for rank, (gamma_id, data) in enumerate(ranked[:10], 1)
            ]
        },
        'reduction': survivors_data['stats'],
        'structure': structure,
        'issue_classification': {
            'issue': issue,
            'rationale': rationale,
            'scientifically_valid': True,  # Toutes issues valides
            'next_steps': {
                'A1': 'Tests prédictifs cross-domaine (R2+)',
                'A2': 'Caractériser dépendances',
                'A3': 'Isoler noyau',
                'B': 'Contraintes raffinées',
                'C': 'Réviser hypothèses framework'
            }.get(issue, 'Unknown')
        }
    }
    
    with open(report_dir / "report_r1_3.json", 'w') as f:
        json.dump(report_json, f, indent=2)
    
    print(f"\n✓ Rapport R1.3 généré: {report_dir}")
    print(f"\nISSUE CLASSIFIÉE: {issue}")
    print(f"Rationale: {rationale}")
    
    return issue, report_dir


if __name__ == "__main__":
    generate_report()