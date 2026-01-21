# tests/utilities/report_writers.py
"""
Report Writers - Formatage et écriture rapports structurés.

RESPONSABILITÉS :
- Écriture JSON structurés
- Formatage TXT lisibles humains
- Génération CSVs analyse
- Helpers formatage (sections, tableaux)

ARCHITECTURE :
- write_json() : JSON avec indent
- write_summary_section() : Formatage sections TXT
- write_regime_synthesis() : Synthèse régimes
- write_timeline_signatures() : Signatures dynamiques

PRINCIPE R0 :
- Séparation calcul/formatting (modules analytiques ≠ writers)
- Réutilisabilité formatters (verdict, modifier, test profiling)
- Structure rapports standardisée

UTILISATEURS :
- verdict_reporter.py (rapports complets)
- Futurs : modifier_profiling.py, test_profiling.py
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


# =============================================================================
# ÉCRITURE JSON
# =============================================================================

def write_json(
    data: Dict,
    filepath: Path,
    indent: int = 2
) -> None:
    """
    Écrit dict → JSON formaté.
    
    Args:
        data: Données à sérialiser
        filepath: Chemin fichier sortie
        indent: Indentation (défaut 2)
    
    Examples:
        >>> write_json({'key': 'value'}, Path('output.json'))
    """
    # Conversion tuples → strings pour sérialisation
    data_serializable = _make_json_serializable(data)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data_serializable, f, indent=indent)


def _make_json_serializable(obj):
    """
    Convertit récursivement tuples en strings pour JSON.
    
    Problème : Clés tuple (test, metric, proj) non JSON-serializable.
    Solution : tuple → "test|metric|proj" string.
    """
    if isinstance(obj, dict):
        return {
            (str(k) if isinstance(k, tuple) else k): _make_json_serializable(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return "|".join(str(x) for x in obj)
    else:
        return obj


# =============================================================================
# FORMATAGE SECTIONS TXT
# =============================================================================

def write_header(
    f,
    title: str,
    width: int = 80,
    char: str = '='
) -> None:
    """
    Écrit header section.
    
    Args:
        f: File handle
        title: Titre section
        width: Largeur ligne
        char: Caractère bordure
    
    Examples:
        >>> with open('report.txt', 'w') as f:
        ...     write_header(f, 'METADATA')
    """
    f.write(char * width + "\n")
    f.write(f"{title}\n")
    f.write(char * width + "\n\n")


def write_subheader(
    f,
    title: str,
    width: int = 80,
    char: str = '-'
) -> None:
    """
    Écrit sous-header section.
    
    Args:
        f: File handle
        title: Titre sous-section
        width: Largeur ligne
        char: Caractère bordure
    """
    f.write(f"{title}\n")
    f.write(char * width + "\n")


def write_key_value(
    f,
    key: str,
    value: Any,
    indent: int = 0
) -> None:
    """
    Écrit paire clé-valeur.
    
    Args:
        f: File handle
        key: Clé
        value: Valeur
        indent: Niveau indentation (espaces)
    
    Examples:
        >>> write_key_value(f, 'Total observations', 4320)
        Total observations    : 4320
    """
    prefix = " " * indent
    f.write(f"{prefix}{key:25s} : {value}\n")


# =============================================================================
# SYNTHÈSE RÉGIMES (TRANSVERSALE)
# =============================================================================

def write_regime_synthesis(
    f,
    gamma_profiles: Dict,
    width: int = 80
) -> None:
    """
    Écrit synthèse régimes transversale.
    
    Agrège régimes sur tous gammas × tests, groupés par famille.
    
    Args:
        f: File handle
        gamma_profiles: Profils gammas formatés
        width: Largeur section
    
    Structure :
        CONSERVATION (régimes sains):
          CONSERVES_SYMMETRY : 120 (12.5%)
          ...
        PATHOLOGIES:
          NUMERIC_INSTABILITY : 45 (4.7%)
          ...
    """
    write_header(f, "SYNTHÈSE RÉGIMES (vue transversale)", width)
    
    # Compter régimes globalement
    regime_counter = defaultdict(int)
    for gamma_data in gamma_profiles.values():
        for test_data in gamma_data.get('tests', {}).values():
            regime = test_data.get('regime', '')
            regime_counter[regime] += 1
    
    total = sum(regime_counter.values())
    
    f.write(f"Total profils : {total}\n\n")
    
    # Grouper par famille
    conservation = {k: v for k, v in regime_counter.items() if 'CONSERVES_' in k}
    pathologies = {k: v for k, v in regime_counter.items() if k in ['NUMERIC_INSTABILITY', 'OSCILLATORY_UNSTABLE', 'TRIVIAL', 'DEGRADING']}
    mixed = {k: v for k, v in regime_counter.items() if k.startswith('MIXED::')}
    other = {k: v for k, v in regime_counter.items() if k not in conservation and k not in pathologies and k not in mixed}
    
    if conservation:
        f.write("CONSERVATION (régimes sains):\n")
        for regime, count in sorted(conservation.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            f.write(f"  {regime:30s} : {count:3d} ({pct:5.1f}%)\n")
        f.write(f"  Total conservation : {sum(conservation.values())} ({sum(conservation.values())/total*100:.1f}%)\n\n")
    
    if pathologies:
        f.write("PATHOLOGIES:\n")
        for regime, count in sorted(pathologies.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            f.write(f"  {regime:30s} : {count:3d} ({pct:5.1f}%)\n")
        f.write(f"  Total pathologies : {sum(pathologies.values())} ({sum(pathologies.values())/total*100:.1f}%)\n\n")
    
    if mixed:
        f.write("MULTIMODALITÉ (MIXED::X):\n")
        for regime, count in sorted(mixed.items(), key=lambda x: -x[1])[:5]:
            pct = count / total * 100
            f.write(f"  {regime:30s} : {count:3d} ({pct:5.1f}%)\n")
        f.write(f"  Total multimodal : {sum(mixed.values())} ({sum(mixed.values())/total*100:.1f}%)\n\n")


# =============================================================================
# SIGNATURES DYNAMIQUES
# =============================================================================

def write_dynamic_signatures(
    f,
    gamma_profiles: Dict,
    width: int = 80
) -> None:
    """
    Écrit signatures dynamiques par gamma.
    
    Args:
        f: File handle
        gamma_profiles: Profils gammas formatés
        width: Largeur section
    
    Structure :
        GAM-001:
          Timeline dominante : early_deviation_then_saturation (75% tests)
          Variantes (3 timelines distinctes):
            - mid_instability_then_collapse (8 tests)
    """
    write_header(f, "SIGNATURES DYNAMIQUES PAR GAMMA", width)
    
    for gamma_id in sorted(gamma_profiles.keys()):
        gamma_data = gamma_profiles[gamma_id]
        
        # Compter timelines dominantes
        timeline_counter = defaultdict(int)
        for test_data in gamma_data.get('tests', {}).values():
            timeline = test_data.get('timeline', 'unknown')
            timeline_counter[timeline] += 1
        
        # Timeline dominante
        if timeline_counter:
            dominant_timeline, count = max(timeline_counter.items(), key=lambda x: x[1])
            confidence = count / len(gamma_data.get('tests', {}))
            
            f.write(f"\n{gamma_id}:\n")
            f.write(f"  Timeline dominante : {dominant_timeline} ({confidence*100:.0f}% tests)\n")
            
            # Diversité timelines
            if len(timeline_counter) > 1:
                f.write(f"  Variantes ({len(timeline_counter)} timelines distinctes):\n")
                for tl, cnt in sorted(timeline_counter.items(), key=lambda x: -x[1])[:3]:
                    if tl != dominant_timeline:
                        f.write(f"    - {tl} ({cnt} tests)\n")
    
    f.write("\n")


# =============================================================================
# COMPARAISONS ENRICHIES (PAR PROPRIÉTÉ)
# =============================================================================

def write_comparisons_enriched(
    f,
    comparisons: Dict,
    gamma_profiles: Dict,
    width: int = 80
) -> None:
    """
    Écrit comparaisons enrichies avec contexte propriétés.
    
    Args:
        f: File handle
        comparisons: Retour compare_gammas_summary()
        gamma_profiles: Profils gammas (pour extraction propriétés)
        width: Largeur section
    
    Structure :
        SYMÉTRIE:
          SYM-001:
            Meilleur : GAM-001
            Pire     : GAM-013
            Classement : GAM-001, GAM-004, GAM-007...
    """
    write_header(f, "COMPARISONS INTER-GAMMAS (par propriété)", width)
    
    # Grouper tests par propriété
    tests_by_property = {
        'Symétrie': ['SYM-001'],
        'Norme': ['SPE-001', 'SPE-002', 'UNIV-001', 'UNIV-002'],
        'Pattern': ['PAT-001'],
        'Topologie': ['TOP-001'],
        'Gradient': ['GRA-001'],
        'Spectre': ['SPA-001']
    }
    
    by_test = comparisons.get('by_test', {})
    
    for property_name, test_list in tests_by_property.items():
        tests_in_data = [t for t in test_list if t in by_test]
        
        if not tests_in_data:
            continue
        
        f.write(f"\n{property_name.upper()}:\n")
        
        for test_name in tests_in_data:
            comp = by_test[test_name]
            f.write(f"\n  {test_name}:\n")
            f.write(f"    Meilleur : {comp['best_conservation']}\n")
            f.write(f"    Pire     : {comp['worst_conservation']}\n")
            f.write(f"    Classement : {', '.join(comp['ranking'][:5])}...\n")
    
    f.write("\n")


# =============================================================================
# FOOTER CONSULTATION
# =============================================================================

def write_consultation_footer(
    f,
    width: int = 80,
    char: str = '='
) -> None:
    """
    Écrit footer avec fichiers consultation.
    
    Args:
        f: File handle
        width: Largeur section
        char: Caractère bordure
    """
    f.write(char * width + "\n")
    f.write("CONSULTATION DÉTAILLÉE\n")
    f.write(char * width + "\n")
    f.write("gamma_profiles.json       : Profils complets tous gammas × tests\n")
    f.write("gamma_profiles.csv        : Vue tabulaire pour analyse\n")
    f.write("comparisons.json          : Classements inter-gammas\n")
    f.write("structural_patterns.json  : Analyses globales (variance, interactions)\n")
    f.write("diagnostics.json          : Diagnostics numériques détaillés\n")
    f.write("marginal_variance_*.csv   : Données brutes analyses (3 strates)\n")
    f.write(char * width + "\n")