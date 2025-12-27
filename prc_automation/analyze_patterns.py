#!/usr/bin/env python3
"""
prc_automation/analyze_patterns.py

Analyse automatique de patterns dans les résultats R0.

Détecte :
  - Corrélations entre tests
  - Familles de comportement
  - Tests discriminants
  - D problématiques/triviaux
  - Suggestions de contraintes CON-GAM-XXX

Usage:
    python analyze_patterns.py --all
    python analyze_patterns.py --correlations
    python analyze_patterns.py --families
    python analyze_patterns.py --suggest-constraints
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

DB_RAW_PATH = Path("prc_database/prc_r0_raw.db")
DB_RESULTS_PATH = Path("prc_database/prc_r0_results.db")


@dataclass
class Pattern:
    """Pattern détecté."""
    type: str  # "correlation" | "family" | "discriminant" | "constraint"
    description: str
    evidence: Dict
    confidence: float  # 0-1
    
    def __repr__(self):
        conf_str = f"[{self.confidence:.0%}]"
        return f"{conf_str} {self.type.upper()}: {self.description}"


# =============================================================================
# 1. CORRÉLATIONS ENTRE TESTS
# =============================================================================

def analyze_test_correlations() -> List[Pattern]:
    """Détecte corrélations entre tests."""
    print("\n" + "="*80)
    print("ANALYSE DES CORRÉLATIONS ENTRE TESTS")
    print("="*80 + "\n")
    
    patterns = []
    
    conn_results = sqlite3.connect(DB_RESULTS_PATH)
    cursor = conn_results.cursor()
    cursor.execute(f"ATTACH DATABASE '{DB_RAW_PATH}' AS db_raw")
    
    # Construire matrice Γ × Test
    cursor.execute("""
        SELECT 
            e.gamma_id,
            ts.test_name,
            AVG(ts.score) as avg_score
        FROM TestScores ts
        JOIN db_raw.Executions e ON ts.exec_id = e.id
        WHERE ts.config_id = 'weights_default'
        GROUP BY e.gamma_id, ts.test_name
    """)
    
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=['gamma_id', 'test_name', 'score'])
    
    # Pivoter
    pivot = df.pivot(index='gamma_id', columns='test_name', values='score')
    pivot = pivot.fillna(0)  # Tests non applicables
    
    if pivot.shape[0] < 3 or pivot.shape[1] < 2:
        print("⚠ Données insuffisantes pour corrélations")
        cursor.execute("DETACH DATABASE db_raw")
        conn_results.close()
        return patterns
    
    # Calculer corrélations
    tests = pivot.columns.tolist()
    n_tests = len(tests)
    
    print(f"Analysant corrélations entre {n_tests} tests sur {len(pivot)} Γ...\n")
    
    for i in range(n_tests):
        for j in range(i+1, n_tests):
            test_a = tests[i]
            test_b = tests[j]
            
            scores_a = pivot[test_a].values
            scores_b = pivot[test_b].values
            
            # Corrélation de Spearman (robuste aux non-linéarités)
            corr, p_value = spearmanr(scores_a, scores_b)
            
            # Seuil significatif
            if abs(corr) > 0.7 and p_value < 0.05:
                pattern = Pattern(
                    type="correlation",
                    description=f"{test_a} et {test_b} sont {'positivement' if corr > 0 else 'négativement'} corrélés",
                    evidence={
                        'correlation': corr,
                        'p_value': p_value,
                        'test_a': test_a,
                        'test_b': test_b,
                    },
                    confidence=min(abs(corr), 1 - p_value)
                )
                patterns.append(pattern)
                
                print(f"✓ {pattern}")
    
    if not patterns:
        print("ℹ Aucune corrélation forte détectée")
    
    cursor.execute("DETACH DATABASE db_raw")
    conn_results.close()
    
    return patterns


# =============================================================================
# 2. FAMILLES DE COMPORTEMENT
# =============================================================================

def analyze_behavior_families() -> List[Pattern]:
    """Détecte familles de Γ avec comportement similaire."""
    print("\n" + "="*80)
    print("ANALYSE DES FAMILLES DE COMPORTEMENT")
    print("="*80 + "\n")
    
    patterns = []
    
    conn_results = sqlite3.connect(DB_RESULTS_PATH)
    cursor = conn_results.cursor()
    cursor.execute(f"ATTACH DATABASE '{DB_RAW_PATH}' AS db_raw")
    
    # Construire matrice Γ × Test
    cursor.execute("""
        SELECT 
            e.gamma_id,
            ts.test_name,
            AVG(ts.score) as avg_score
        FROM TestScores ts
        JOIN db_raw.Executions e ON ts.exec_id = e.id
        WHERE ts.config_id = 'weights_default'
        GROUP BY e.gamma_id, ts.test_name
    """)
    
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=['gamma_id', 'test_name', 'score'])
    
    pivot = df.pivot(index='gamma_id', columns='test_name', values='score')
    pivot = pivot.fillna(0)
    
    if pivot.shape[0] < 3:
        print("⚠ Données insuffisantes pour clustering")
        cursor.execute("DETACH DATABASE db_raw")
        conn_results.close()
        return patterns
    
    print(f"Clustering de {len(pivot)} Γ...\n")
    
    # Clustering hiérarchique
    distances = pdist(pivot.values, metric='euclidean')
    linkage_matrix = linkage(distances, method='ward')
    
    # Découper en clusters (k=3 par défaut)
    n_clusters = min(3, len(pivot) // 2)
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # Analyser chaque cluster
    for cluster_id in range(1, n_clusters + 1):
        gamma_ids = pivot.index[clusters == cluster_id].tolist()
        
        if len(gamma_ids) < 2:
            continue
        
        # Caractériser cluster
        cluster_scores = pivot.loc[gamma_ids].mean(axis=0)
        
        # Trouver tests caractéristiques (scores élevés ou bas)
        high_tests = cluster_scores[cluster_scores > 0.7].index.tolist()
        low_tests = cluster_scores[cluster_scores < 0.3].index.tolist()
        
        description_parts = []
        if high_tests:
            description_parts.append(f"réussissent {', '.join(high_tests[:3])}")
        if low_tests:
            description_parts.append(f"échouent {', '.join(low_tests[:3])}")
        
        if description_parts:
            pattern = Pattern(
                type="family",
                description=f"Famille {cluster_id}: {', '.join(gamma_ids)} - {' et '.join(description_parts)}",
                evidence={
                    'cluster_id': cluster_id,
                    'members': gamma_ids,
                    'high_tests': high_tests,
                    'low_tests': low_tests,
                    'avg_scores': cluster_scores.to_dict(),
                },
                confidence=len(gamma_ids) / len(pivot)
            )
            patterns.append(pattern)
            
            print(f"✓ {pattern}")
    
    cursor.execute("DETACH DATABASE db_raw")
    conn_results.close()
    
    return patterns


# =============================================================================
# 3. TESTS DISCRIMINANTS
# =============================================================================

def analyze_discriminant_tests() -> List[Pattern]:
    """Identifie tests les plus discriminants (forte variance)."""
    print("\n" + "="*80)
    print("ANALYSE DES TESTS DISCRIMINANTS")
    print("="*80 + "\n")
    
    patterns = []
    
    conn_results = sqlite3.connect(DB_RESULTS_PATH)
    cursor = conn_results.cursor()
    cursor.execute(f"ATTACH DATABASE '{DB_RAW_PATH}' AS db_raw")
    
    # Variance par test
    cursor.execute("""
        SELECT 
            ts.test_name,
            AVG(ts.score) as mean_score,
            -- Approximation variance
            AVG(ts.score * ts.score) - AVG(ts.score) * AVG(ts.score) as variance,
            COUNT(DISTINCT e.gamma_id) as n_gammas
        FROM TestScores ts
        JOIN db_raw.Executions e ON ts.exec_id = e.id
        WHERE ts.config_id = 'weights_default'
        GROUP BY ts.test_name
        HAVING n_gammas >= 3
        ORDER BY variance DESC
    """)
    
    results = cursor.fetchall()
    
    if not results:
        print("⚠ Pas de données de tests")
        cursor.execute("DETACH DATABASE db_raw")
        conn_results.close()
        return patterns
    
    print(f"Analysant variance de {len(results)} tests...\n")
    
    # Top 3 tests discriminants
    for test_name, mean_score, variance, n_gammas in results[:3]:
        std = np.sqrt(variance)
        
        pattern = Pattern(
            type="discriminant",
            description=f"{test_name} est très discriminant (std={std:.3f})",
            evidence={
                'test_name': test_name,
                'mean': mean_score,
                'std': std,
                'variance': variance,
                'n_gammas': n_gammas,
            },
            confidence=min(std * 2, 1.0)  # Plus la variance est haute, plus c'est significatif
        )
        patterns.append(pattern)
        
        print(f"✓ {pattern}")
    
    # Tests peu discriminants (à considérer retirer ?)
    print("\nTests peu discriminants (faible variance):")
    for test_name, mean_score, variance, n_gammas in results[-3:]:
        std = np.sqrt(variance)
        print(f"  ℹ {test_name}: std={std:.3f} (peu informatif)")
    
    cursor.execute("DETACH DATABASE db_raw")
    conn_results.close()
    
    return patterns


# =============================================================================
# 4. D PROBLÉMATIQUES OU TRIVIAUX
# =============================================================================

def analyze_d_bases() -> List[Pattern]:
    """Détecte D où tous Γ échouent ou réussissent."""
    print("\n" + "="*80)
    print("ANALYSE DES BASES D")
    print("="*80 + "\n")
    
    patterns = []
    
    conn_results = sqlite3.connect(DB_RESULTS_PATH)
    cursor = conn_results.cursor()
    cursor.execute(f"ATTACH DATABASE '{DB_RAW_PATH}' AS db_raw")
    
    # Score moyen par D
    cursor.execute("""
        SELECT 
            e.d_base_id,
            AVG(ts.score) as avg_score,
            COUNT(DISTINCT e.gamma_id) as n_gammas
        FROM db_raw.Executions e
        JOIN TestScores ts ON ts.exec_id = e.id
        WHERE ts.config_id = 'weights_default'
        GROUP BY e.d_base_id
        HAVING n_gammas >= 2
        ORDER BY avg_score
    """)
    
    results = cursor.fetchall()
    
    print(f"Analysant {len(results)} bases D...\n")
    
    for d_base_id, avg_score, n_gammas in results:
        # D où tous échouent
        if avg_score < 0.3:
            pattern = Pattern(
                type="constraint",
                description=f"{d_base_id} est problématique : tous les Γ échouent (avg={avg_score:.3f})",
                evidence={
                    'd_base_id': d_base_id,
                    'avg_score': avg_score,
                    'n_gammas': n_gammas,
                },
                confidence=1.0 - avg_score
            )
            patterns.append(pattern)
            print(f"⚠ {pattern}")
        
        # D où tous réussissent (possiblement trivial)
        elif avg_score > 0.8:
            pattern = Pattern(
                type="constraint",
                description=f"{d_base_id} est trivial : tous les Γ réussissent (avg={avg_score:.3f})",
                evidence={
                    'd_base_id': d_base_id,
                    'avg_score': avg_score,
                    'n_gammas': n_gammas,
                },
                confidence=avg_score
            )
            patterns.append(pattern)
            print(f"ℹ {pattern}")
    
    cursor.execute("DETACH DATABASE db_raw")
    conn_results.close()
    
    return patterns


# =============================================================================
# 5. SUGGESTIONS DE CONTRAINTES
# =============================================================================

def suggest_constraints() -> List[Pattern]:
    """Suggère contraintes CON-GAM-XXX basées sur patterns d'échec."""
    print("\n" + "="*80)
    print("SUGGESTIONS DE CONTRAINTES CON-GAM-XXX")
    print("="*80 + "\n")
    
    patterns = []
    
    conn_results = sqlite3.connect(DB_RESULTS_PATH)
    cursor = conn_results.cursor()
    cursor.execute(f"ATTACH DATABASE '{DB_RAW_PATH}' AS db_raw")
    
    # Détecter échecs systématiques par famille
    cursor.execute("""
        SELECT 
            SUBSTR(e.gamma_id, 1, 7) as gamma_family,
            COUNT(DISTINCT e.gamma_id) as n_gammas,
            AVG(ts.score) as avg_score,
            COUNT(CASE WHEN v.verdict = 'SURVIVES[R0]' THEN 1 END) as n_survives,
            COUNT(CASE WHEN v.verdict = 'FLAGGED_FOR_REVIEW' THEN 1 END) as n_flagged
        FROM db_raw.Executions e
        JOIN TestScores ts ON ts.exec_id = e.id
        LEFT JOIN GammaVerdicts v ON v.gamma_id = e.gamma_id AND v.config_id = 'weights_default'
        WHERE ts.config_id = 'weights_default'
        GROUP BY gamma_family
        HAVING n_gammas >= 2
    """)
    
    results = cursor.fetchall()
    
    print(f"Analysant {len(results)} familles de Γ...\n")
    
    for family, n_gammas, avg_score, n_survives, n_flagged in results:
        # Famille entière échoue
        if avg_score < 0.4 and n_flagged >= n_gammas * 0.5:
            pattern = Pattern(
                type="constraint",
                description=f"SUGGESTION CON-GAM-XXX: Famille {family} échoue systématiquement (avg={avg_score:.3f}, {n_flagged}/{n_gammas} flagged)",
                evidence={
                    'family': family,
                    'n_gammas': n_gammas,
                    'avg_score': avg_score,
                    'n_survives': n_survives,
                    'n_flagged': n_flagged,
                    'elimination_rate': n_flagged / n_gammas,
                },
                confidence=n_flagged / n_gammas
            )
            patterns.append(pattern)
            print(f"💡 {pattern}")
    
    # Détecter tests échoués systématiquement
    cursor.execute("""
        SELECT 
            ts.test_name,
            AVG(ts.score) as avg_score,
            COUNT(DISTINCT e.gamma_id) as n_gammas,
            COUNT(CASE WHEN ts.score < 0.3 THEN 1 END) as n_failures
        FROM TestScores ts
        JOIN db_raw.Executions e ON ts.exec_id = e.id
        WHERE ts.config_id = 'weights_default'
        GROUP BY ts.test_name
        HAVING n_gammas >= 3 AND avg_score < 0.3
    """)
    
    for test_name, avg_score, n_gammas, n_failures in cursor.fetchall():
        pattern = Pattern(
            type="constraint",
            description=f"SUGGESTION CON-GAM-XXX: Test {test_name} échoue pour tous les Γ (avg={avg_score:.3f})",
            evidence={
                'test_name': test_name,
                'avg_score': avg_score,
                'n_gammas': n_gammas,
                'n_failures': n_failures,
            },
            confidence=1.0 - avg_score
        )
        patterns.append(pattern)
        print(f"💡 {pattern}")
    
    cursor.execute("DETACH DATABASE db_raw")
    conn_results.close()
    
    return patterns


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyse automatique de patterns")
    
    parser.add_argument('--all', action='store_true',
                       help='Toutes les analyses')
    parser.add_argument('--correlations', action='store_true',
                       help='Corrélations entre tests')
    parser.add_argument('--families', action='store_true',
                       help='Familles de comportement')
    parser.add_argument('--discriminant', action='store_true',
                       help='Tests discriminants')
    parser.add_argument('--d-bases', action='store_true',
                       help='Analyse des D')
    parser.add_argument('--suggest-constraints', action='store_true',
                       help='Suggérer contraintes CON-GAM-XXX')
    
    args = parser.parse_args()
    
    if not DB_RAW_PATH.exists() or not DB_RESULTS_PATH.exists():
        print("❌ Bases de données non trouvées")
        return
    
    all_patterns = []
    
    if args.all or args.correlations:
        all_patterns.extend(analyze_test_correlations())
    
    if args.all or args.families:
        all_patterns.extend(analyze_behavior_families())
    
    if args.all or args.discriminant:
        all_patterns.extend(analyze_discriminant_tests())
    
    if args.all or args.d_bases:
        all_patterns.extend(analyze_d_bases())
    
    if args.all or args.suggest_constraints:
        all_patterns.extend(suggest_constraints())
    
    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ DES PATTERNS DÉTECTÉS")
    print("="*80 + "\n")
    
    by_type = {}
    for pattern in all_patterns:
        if pattern.type not in by_type:
            by_type[pattern.type] = []
        by_type[pattern.type].append(pattern)
    
    for ptype, patterns in sorted(by_type.items()):
        print(f"\n{ptype.upper()} ({len(patterns)}):")
        for pattern in sorted(patterns, key=lambda p: p.confidence, reverse=True):
            print(f"  {pattern}")
    
    print(f"\n{'='*80}")
    print(f"Total: {len(all_patterns)} patterns détectés")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()