#!/usr/bin/env python3
"""
prc_automation/analyze_and_export.py

Analyse exhaustive multi-configs/thresholds/tests avec export format Claude.

Scanne automatiquement tous les triplets (config, threshold, test) disponibles
dans db_results et génère un rapport comparatif complet.

Usage:
    python analyze_and_export.py --output analysis_complete.txt
    python analyze_and_export.py --output report.txt --min-gammas 5
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

DB_RAW_PATH = Path("prc_database/prc_r0_raw.db")
DB_RESULTS_PATH = Path("prc_database/prc_r0_results.db")


@dataclass
class Triplet:
    """Triplet (config, threshold, test)."""
    config_id: str
    threshold_id: str
    test_name: str


@dataclass
class ConfigThreshold:
    """Paire (config, threshold)."""
    config_id: str
    threshold_id: str
    
    def __hash__(self):
        return hash((self.config_id, self.threshold_id))
    
    def __eq__(self, other):
        return (self.config_id == other.config_id and 
                self.threshold_id == other.threshold_id)


# =============================================================================
# 1. SCAN DE L'ESPACE DISPONIBLE
# =============================================================================

def scan_available_space():
    """Scanne tous les configs, thresholds, tests disponibles."""
    print("Scanning available space...")
    
    conn = sqlite3.connect(DB_RESULTS_PATH)
    cursor = conn.cursor()
    
    # Configs
    cursor.execute("SELECT DISTINCT config_id FROM TestScores ORDER BY config_id")
    configs = [row[0] for row in cursor.fetchall()]
    
    # Thresholds
    cursor.execute("SELECT DISTINCT threshold_id FROM GammaVerdicts ORDER BY threshold_id")
    thresholds = [row[0] for row in cursor.fetchall()]
    
    # Tests
    cursor.execute("SELECT DISTINCT test_name FROM TestScores ORDER BY test_name")
    tests = [row[0] for row in cursor.fetchall()]
    
    # Gammas
    cursor.execute(f"ATTACH DATABASE '{DB_RAW_PATH}' AS db_raw")
    cursor.execute("SELECT DISTINCT gamma_id FROM db_raw.Executions ORDER BY gamma_id")
    gammas = [row[0] for row in cursor.fetchall()]
    cursor.execute("DETACH DATABASE db_raw")
    
    conn.close()
    
    print(f"  Found: {len(configs)} configs, {len(thresholds)} thresholds, "
          f"{len(tests)} tests, {len(gammas)} Γ")
    
    return configs, thresholds, tests, gammas


# =============================================================================
# 2. ANALYSES PAR (CONFIG, THRESHOLD)
# =============================================================================

def analyze_config_threshold(config_id: str, threshold_id: str, gammas: List[str]):
    """Analyse complète pour une paire (config, threshold)."""
    conn = sqlite3.connect(DB_RESULTS_PATH)
    cursor = conn.cursor()
    cursor.execute(f"ATTACH DATABASE '{DB_RAW_PATH}' AS db_raw")
    
    results = {
        'verdicts': {},
        'test_scores': {},
        'test_scores_detailed': {},  # NOUVEAU: scores par (gamma, test)
        'correlations': {},
        'clusters': {},
        'discriminance': {},
    }
    
    # --- Verdicts ---
    cursor.execute("""
        SELECT gamma_id, verdict, score_global, majority_pct, robustness_pct
        FROM GammaVerdicts
        WHERE config_id = ? AND threshold_id = ?
    """, (config_id, threshold_id))
    
    for row in cursor.fetchall():
        results['verdicts'][row[0]] = {
            'verdict': row[1],
            'score_global': row[2],
            'majority_pct': row[3],
            'robustness_pct': row[4],
        }
    
    # --- Scores moyens par test ---
    cursor.execute("""
        SELECT 
            ts.test_name,
            AVG(ts.score) as avg_score,
            MIN(ts.score) as min_score,
            MAX(ts.score) as max_score,
            AVG(ts.score * ts.score) - AVG(ts.score) * AVG(ts.score) as variance
        FROM TestScores ts
        WHERE ts.config_id = ?
        GROUP BY ts.test_name
    """, (config_id,))
    
    for row in cursor.fetchall():
        results['test_scores'][row[0]] = {
            'avg': row[1],
            'min': row[2],
            'max': row[3],
            'variance': row[4],
            'std': np.sqrt(max(0, row[4])),
        }
    
    # --- Scores détaillés par (gamma, test) ---
    cursor.execute("""
        SELECT 
            e.gamma_id,
            ts.test_name,
            AVG(ts.score) as avg_score
        FROM TestScores ts
        JOIN db_raw.Executions e ON ts.exec_id = e.id
        WHERE ts.config_id = ?
        GROUP BY e.gamma_id, ts.test_name
        ORDER BY ts.test_name, avg_score DESC
    """, (config_id,))
    
    for row in cursor.fetchall():
        test_name = row[1]
        if test_name not in results['test_scores_detailed']:
            results['test_scores_detailed'][test_name] = []
        results['test_scores_detailed'][test_name].append({
            'gamma_id': row[0],
            'score': row[2],
        })
    
    # --- Corrélations entre tests ---
    cursor.execute("""
        SELECT 
            e.gamma_id,
            ts.test_name,
            AVG(ts.score) as avg_score
        FROM TestScores ts
        JOIN db_raw.Executions e ON ts.exec_id = e.id
        WHERE ts.config_id = ?
        GROUP BY e.gamma_id, ts.test_name
    """, (config_id,))
    
    data = cursor.fetchall()
    if data:
        df = pd.DataFrame(data, columns=['gamma_id', 'test_name', 'score'])
        pivot = df.pivot(index='gamma_id', columns='test_name', values='score')
        pivot = pivot.fillna(0)
        
        if pivot.shape[0] >= 3 and pivot.shape[1] >= 2:
            # Filtrer colonnes constantes (variance = 0)
            non_constant_tests = []
            for col in pivot.columns:
                if pivot[col].std() > 1e-10:  # Non constant
                    non_constant_tests.append(col)
            
            pivot = pivot[non_constant_tests]
            tests = pivot.columns.tolist()
            
            for i in range(len(tests)):
                for j in range(i+1, len(tests)):
                    try:
                        corr, p_value = spearmanr(pivot[tests[i]], pivot[tests[j]])
                        if abs(corr) > 0.6 and p_value < 0.05 and not np.isnan(corr):
                            results['correlations'][(tests[i], tests[j])] = {
                                'corr': corr,
                                'p_value': p_value,
                            }
                    except:
                        pass
            
            # --- Clustering ---
            try:
                distances = pdist(pivot.values, metric='euclidean')
                linkage_matrix = linkage(distances, method='ward')
                n_clusters = min(3, len(pivot) // 2)
                if n_clusters >= 2:
                    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                    
                    for cluster_id in range(1, n_clusters + 1):
                        members = pivot.index[clusters == cluster_id].tolist()
                        if len(members) >= 2:
                            cluster_scores = pivot.loc[members].mean(axis=0)
                            high_tests = cluster_scores[cluster_scores > 0.7].index.tolist()
                            low_tests = cluster_scores[cluster_scores < 0.3].index.tolist()
                            
                            results['clusters'][cluster_id] = {
                                'members': members,
                                'high_tests': high_tests,
                                'low_tests': low_tests,
                            }
            except:
                pass
    
    # --- Discriminance (variance) ---
    sorted_tests = sorted(results['test_scores'].items(), 
                         key=lambda x: x[1]['std'], reverse=True)
    
    # Séparer most/least sans overlap
    n_tests = len(sorted_tests)
    if n_tests >= 6:
        most_discriminant = sorted_tests[:3]
        least_discriminant = sorted_tests[-3:]
    elif n_tests >= 3:
        # Split en deux groupes
        split = n_tests // 2
        most_discriminant = sorted_tests[:split]
        least_discriminant = sorted_tests[split:]
    else:
        # Trop peu de tests
        most_discriminant = sorted_tests
        least_discriminant = []
    
    results['discriminance'] = {
        'most_discriminant': most_discriminant,
        'least_discriminant': least_discriminant,
    }
    
    cursor.execute("DETACH DATABASE db_raw")
    conn.close()
    
    return results


# =============================================================================
# 3. COMPARAISONS MULTI-CONFIGS
# =============================================================================

def compare_configs(configs: List[str], threshold_id: str, gammas: List[str]):
    """Compare verdicts et tests entre configs (même threshold)."""
    conn = sqlite3.connect(DB_RESULTS_PATH)
    cursor = conn.cursor()
    
    comparisons = {
        'verdict_changes': defaultdict(list),
        'test_discriminance_changes': {},
        'correlation_stability': {},
    }
    
    # --- Verdicts qui changent ---
    for gamma_id in gammas:
        verdicts = {}
        for config_id in configs:
            cursor.execute("""
                SELECT verdict FROM GammaVerdicts
                WHERE gamma_id = ? AND config_id = ? AND threshold_id = ?
            """, (gamma_id, config_id, threshold_id))
            row = cursor.fetchone()
            if row:
                verdicts[config_id] = row[0]
        
        # Si verdicts diffèrent entre configs
        if len(set(verdicts.values())) > 1:
            comparisons['verdict_changes'][gamma_id] = verdicts
    
    # --- Tests qui changent de discriminance ---
    for config_id in configs:
        cursor.execute("""
            SELECT 
                test_name,
                AVG(score * score) - AVG(score) * AVG(score) as variance
            FROM TestScores
            WHERE config_id = ?
            GROUP BY test_name
        """, (config_id,))
        
        test_vars = {row[0]: row[1] for row in cursor.fetchall()}
        comparisons['test_discriminance_changes'][config_id] = test_vars
    
    # --- Corrélations stables/instables ---
    for config_id in configs:
        cursor.execute(f"ATTACH DATABASE '{DB_RAW_PATH}' AS db_raw")
        cursor.execute("""
            SELECT 
                e.gamma_id,
                ts.test_name,
                AVG(ts.score) as avg_score
            FROM TestScores ts
            JOIN db_raw.Executions e ON ts.exec_id = e.id
            WHERE ts.config_id = ?
            GROUP BY e.gamma_id, ts.test_name
        """, (config_id,))
        
        data = cursor.fetchall()
        if data:
            df = pd.DataFrame(data, columns=['gamma_id', 'test_name', 'score'])
            pivot = df.pivot(index='gamma_id', columns='test_name', values='score')
            pivot = pivot.fillna(0)
            
            if pivot.shape[0] >= 3 and pivot.shape[1] >= 2:
                corr_matrix = pivot.corr(method='spearman')
                comparisons['correlation_stability'][config_id] = corr_matrix
        
        cursor.execute("DETACH DATABASE db_raw")
    
    conn.close()
    return comparisons


# =============================================================================
# 4. COMPARAISONS MULTI-THRESHOLDS
# =============================================================================

def compare_thresholds(config_id: str, thresholds: List[str], gammas: List[str]):
    """Compare verdicts entre thresholds (même config)."""
    conn = sqlite3.connect(DB_RESULTS_PATH)
    cursor = conn.cursor()
    
    comparisons = {
        'verdict_changes': defaultdict(list),
        'threshold_sensitivity': {},
    }
    
    # --- Verdicts qui changent ---
    for gamma_id in gammas:
        verdicts = {}
        for threshold_id in thresholds:
            cursor.execute("""
                SELECT verdict FROM GammaVerdicts
                WHERE gamma_id = ? AND config_id = ? AND threshold_id = ?
            """, (gamma_id, config_id, threshold_id))
            row = cursor.fetchone()
            if row:
                verdicts[threshold_id] = row[0]
        
        if len(set(verdicts.values())) > 1:
            comparisons['verdict_changes'][gamma_id] = verdicts
    
    # --- Sensibilité globale aux thresholds ---
    for threshold_id in thresholds:
        cursor.execute("""
            SELECT verdict, COUNT(*) 
            FROM GammaVerdicts
            WHERE config_id = ? AND threshold_id = ?
            GROUP BY verdict
        """, (config_id, threshold_id))
        
        distribution = {row[0]: row[1] for row in cursor.fetchall()}
        comparisons['threshold_sensitivity'][threshold_id] = distribution
    
    conn.close()
    return comparisons


# =============================================================================
# 5. SUGGESTIONS CONTRAINTES
# =============================================================================

def suggest_constraints(config_id: str, threshold_id: str):
    """Suggère contraintes CON-GAM-XXX."""
    conn = sqlite3.connect(DB_RESULTS_PATH)
    cursor = conn.cursor()
    cursor.execute(f"ATTACH DATABASE '{DB_RAW_PATH}' AS db_raw")
    
    suggestions = []
    
    # --- Familles qui échouent systématiquement ---
    cursor.execute("""
        SELECT 
            SUBSTR(v.gamma_id, 1, 7) as family,
            COUNT(DISTINCT v.gamma_id) as n_gammas,
            COUNT(CASE WHEN v.verdict = 'FLAGGED_FOR_REVIEW' THEN 1 END) as n_flagged,
            AVG(v.score_global) as avg_score
        FROM GammaVerdicts v
        WHERE v.config_id = ? AND v.threshold_id = ?
        GROUP BY family
        HAVING n_gammas >= 2 AND n_flagged >= n_gammas * 0.5
    """, (config_id, threshold_id))
    
    for family, n_gammas, n_flagged, avg_score in cursor.fetchall():
        suggestions.append({
            'type': 'family_failure',
            'family': family,
            'n_gammas': n_gammas,
            'n_flagged': n_flagged,
            'avg_score': avg_score,
            'description': f"Famille {family} échoue systématiquement ({n_flagged}/{n_gammas} flagged)",
        })
    
    # --- Tests échoués par tous les Γ ---
    cursor.execute("""
        SELECT 
            ts.test_name,
            AVG(ts.score) as avg_score,
            COUNT(DISTINCT e.gamma_id) as n_gammas
        FROM TestScores ts
        JOIN db_raw.Executions e ON ts.exec_id = e.id
        WHERE ts.config_id = ?
        GROUP BY ts.test_name
        HAVING n_gammas >= 3 AND avg_score < 0.3
    """, (config_id,))
    
    for test_name, avg_score, n_gammas in cursor.fetchall():
        suggestions.append({
            'type': 'test_universal_failure',
            'test_name': test_name,
            'avg_score': avg_score,
            'n_gammas': n_gammas,
            'description': f"Test {test_name} échoue pour tous les Γ (avg={avg_score:.3f})",
        })
    
    cursor.execute("DETACH DATABASE db_raw")
    conn.close()
    
    return suggestions


# =============================================================================
# 6. GÉNÉRATION RAPPORT FORMAT CLAUDE
# =============================================================================

def generate_claude_report(configs, thresholds, tests, gammas, output_path):
    """Génère rapport complet format Claude."""
    lines = []
    
    # === HEADER ===
    lines.append("="*80)
    lines.append("ANALYSE EXHAUSTIVE R0 - MULTI-CONFIGS/THRESHOLDS/TESTS")
    lines.append("="*80)
    lines.append(f"Export: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("VUE D'ENSEMBLE")
    lines.append("-" * 80)
    lines.append(f"Configs: {len(configs)} ({', '.join(configs)})")
    lines.append(f"Thresholds: {len(thresholds)} ({', '.join(thresholds)})")
    lines.append(f"Tests: {len(tests)}")
    lines.append(f"Γ testés: {len(gammas)}")
    lines.append("")
    
    # === ANALYSES PAR (CONFIG, THRESHOLD) ===
    for config_id in configs:
        for threshold_id in thresholds:
            lines.append("="*80)
            lines.append(f"ANALYSE: config={config_id}, threshold={threshold_id}")
            lines.append("="*80)
            lines.append("")
            
            results = analyze_config_threshold(config_id, threshold_id, gammas)
            
            # Verdicts
            lines.append("DISTRIBUTION VERDICTS")
            lines.append("-" * 80)
            verdict_counts = defaultdict(int)
            for v in results['verdicts'].values():
                verdict_counts[v['verdict']] += 1
            
            for verdict, count in sorted(verdict_counts.items()):
                lines.append(f"{verdict}: {count}")
            lines.append("")
            
            # Verdicts par Γ
            lines.append("VERDICTS PAR Γ")
            lines.append("-" * 80)
            lines.append(f"{'Γ':<12} {'Verdict':<20} {'Score':<8} {'Maj%':<8} {'Rob%':<8}")
            lines.append("-" * 80)
            
            sorted_verdicts = sorted(results['verdicts'].items(),
                                    key=lambda x: x[1]['score_global'],
                                    reverse=True)
            
            for gamma_id, v in sorted_verdicts:
                lines.append(f"{gamma_id:<12} {v['verdict']:<20} "
                           f"{v['score_global']:>7.2f}  "
                           f"{v['majority_pct']:>7.1f}  "
                           f"{v['robustness_pct']:>7.1f}")
            lines.append("")
            
            # Tests discriminants
            lines.append("TESTS DISCRIMINANTS")
            lines.append("-" * 80)
            
            if results['discriminance']['most_discriminant']:
                lines.append("Plus discriminants:")
                for test_name, stats in results['discriminance']['most_discriminant']:
                    lines.append(f"  {test_name}: std={stats['std']:.3f}, "
                               f"avg={stats['avg']:.3f}")
            
            if results['discriminance']['least_discriminant']:
                lines.append("\nMoins discriminants:")
                for test_name, stats in results['discriminance']['least_discriminant']:
                    lines.append(f"  {test_name}: std={stats['std']:.3f}, "
                               f"avg={stats['avg']:.3f}")
            
            if not results['discriminance']['least_discriminant']:
                lines.append("\n(Pas assez de tests pour identifier les moins discriminants)")
            
            lines.append("")
            
            # Scores détaillés par test
            lines.append("SCORES DÉTAILLÉS PAR TEST")
            lines.append("-" * 80)
            
            if results['test_scores_detailed']:
                # Afficher par test
                for test_name in sorted(results['test_scores_detailed'].keys()):
                    test_scores = results['test_scores_detailed'][test_name]
                    
                    lines.append(f"\n{test_name}:")
                    lines.append(f"  {'Γ':<12} {'Score':>8}")
                    lines.append(f"  {'-'*21}")
                    
                    for entry in test_scores:
                        lines.append(f"  {entry['gamma_id']:<12} {entry['score']:>8.3f}")
            else:
                lines.append("(Aucune donnée de scores détaillés)")
            
            lines.append("")
            
            # Corrélations fortes
            if results['correlations']:
                lines.append("CORRÉLATIONS FORTES (|ρ| > 0.6, p < 0.05)")
                lines.append("-" * 80)
                for (test_a, test_b), stats in sorted(results['correlations'].items(),
                                                       key=lambda x: abs(x[1]['corr']),
                                                       reverse=True)[:5]:
                    lines.append(f"{test_a} ↔ {test_b}: ρ={stats['corr']:.3f}, "
                               f"p={stats['p_value']:.4f}")
                lines.append("")
            
            # Clusters
            if results['clusters']:
                lines.append("FAMILLES DE COMPORTEMENT")
                lines.append("-" * 80)
                for cluster_id, info in results['clusters'].items():
                    lines.append(f"\nFamille {cluster_id}: {', '.join(info['members'])}")
                    if info['high_tests']:
                        lines.append(f"  Réussissent: {', '.join(info['high_tests'][:3])}")
                    if info['low_tests']:
                        lines.append(f"  Échouent: {', '.join(info['low_tests'][:3])}")
                lines.append("")
    
    # === COMPARAISONS MULTI-CONFIGS ===
    if len(configs) > 1:
        for threshold_id in thresholds:
            lines.append("="*80)
            lines.append(f"COMPARAISON CONFIGS (threshold={threshold_id})")
            lines.append("="*80)
            lines.append("")
            
            comp = compare_configs(configs, threshold_id, gammas)
            
            # Verdicts qui changent
            if comp['verdict_changes']:
                lines.append("VERDICTS QUI CHANGENT ENTRE CONFIGS")
                lines.append("-" * 80)
                for gamma_id, verdicts in sorted(comp['verdict_changes'].items()):
                    verdict_str = " → ".join([f"{c}: {v}" for c, v in verdicts.items()])
                    lines.append(f"{gamma_id}: {verdict_str}")
                lines.append("")
            
            # Tests qui changent de discriminance
            lines.append("DISCRIMINANCE DES TESTS PAR CONFIG")
            lines.append("-" * 80)
            for config_id, test_vars in comp['test_discriminance_changes'].items():
                sorted_tests = sorted(test_vars.items(), key=lambda x: x[1], reverse=True)[:3]
                lines.append(f"\n{config_id} (top 3 discriminants):")
                for test_name, var in sorted_tests:
                    lines.append(f"  {test_name}: std={np.sqrt(max(0, var)):.3f}")
            lines.append("")
    
    # === COMPARAISONS MULTI-THRESHOLDS ===
    if len(thresholds) > 1:
        for config_id in configs:
            lines.append("="*80)
            lines.append(f"COMPARAISON THRESHOLDS (config={config_id})")
            lines.append("="*80)
            lines.append("")
            
            comp = compare_thresholds(config_id, thresholds, gammas)
            
            # Verdicts qui changent
            if comp['verdict_changes']:
                lines.append("VERDICTS QUI CHANGENT ENTRE THRESHOLDS")
                lines.append("-" * 80)
                for gamma_id, verdicts in sorted(comp['verdict_changes'].items()):
                    verdict_str = " → ".join([f"{t}: {v}" for t, v in verdicts.items()])
                    lines.append(f"{gamma_id}: {verdict_str}")
                lines.append("")
            
            # Distribution par threshold
            lines.append("DISTRIBUTION VERDICTS PAR THRESHOLD")
            lines.append("-" * 80)
            for threshold_id, dist in comp['threshold_sensitivity'].items():
                lines.append(f"\n{threshold_id}:")
                for verdict, count in sorted(dist.items()):
                    lines.append(f"  {verdict}: {count}")
            lines.append("")
    
    # === SUGGESTIONS CONTRAINTES ===
    lines.append("="*80)
    lines.append("SUGGESTIONS CONTRAINTES CON-GAM-XXX")
    lines.append("="*80)
    lines.append("")
    
    for config_id in configs:
        for threshold_id in thresholds:
            suggestions = suggest_constraints(config_id, threshold_id)
            
            if suggestions:
                lines.append(f"Config={config_id}, Threshold={threshold_id}:")
                lines.append("-" * 80)
                
                for sugg in suggestions:
                    lines.append(f"\n💡 {sugg['description']}")
                    if sugg['type'] == 'family_failure':
                        lines.append(f"   Evidence: {sugg['n_flagged']}/{sugg['n_gammas']} "
                                   f"flagged, score={sugg['avg_score']:.2f}")
                    elif sugg['type'] == 'test_universal_failure':
                        lines.append(f"   Evidence: {sugg['n_gammas']} Γ testés, "
                                   f"avg={sugg['avg_score']:.3f}")
                lines.append("")
    
    # === MÉTA-ANALYSE ===
    lines.append("="*80)
    lines.append("MÉTA-ANALYSE: QU'EST-CE QUI EST STABLE/INSTABLE?")
    lines.append("="*80)
    lines.append("")
    
    # Stabilité des verdicts
    if len(configs) > 1 or len(thresholds) > 1:
        stable_verdicts = []
        unstable_verdicts = []
        
        for gamma_id in gammas:
            all_verdicts = set()
            for config_id in configs:
                for threshold_id in thresholds:
                    conn = sqlite3.connect(DB_RESULTS_PATH)
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT verdict FROM GammaVerdicts
                        WHERE gamma_id = ? AND config_id = ? AND threshold_id = ?
                    """, (gamma_id, config_id, threshold_id))
                    row = cursor.fetchone()
                    if row:
                        all_verdicts.add(row[0])
                    conn.close()
            
            if len(all_verdicts) == 1:
                stable_verdicts.append(gamma_id)
            elif len(all_verdicts) > 1:
                unstable_verdicts.append(gamma_id)
        
        lines.append(f"Γ avec verdicts STABLES (tous configs/thresholds): {len(stable_verdicts)}")
        if stable_verdicts[:5]:
            lines.append(f"  Exemples: {', '.join(stable_verdicts[:5])}")
        
        lines.append(f"\nΓ avec verdicts INSTABLES (changent selon config/threshold): {len(unstable_verdicts)}")
        if unstable_verdicts[:5]:
            lines.append(f"  Exemples: {', '.join(unstable_verdicts[:5])}")
        lines.append("")
    
    # === QUESTIONS POUR ANALYSE ===
    lines.append("="*80)
    lines.append("QUESTIONS POUR ANALYSE APPROFONDIE")
    lines.append("="*80)
    lines.append("")
    lines.append("1. Quels Γ sont robustes aux changements de config/threshold?")
    lines.append("   → Candidats fiables pour R1")
    lines.append("")
    lines.append("2. Quels tests sont les plus stables entre configs?")
    lines.append("   → Tests à prioriser dans scoring")
    lines.append("")
    lines.append("3. Impact des nouveaux tests (UNIV-002b, DIV-HETERO)?")
    lines.append("   → Comparaison weights_default vs weights_post_audit")
    lines.append("")
    lines.append("4. Sensibilité aux seuils de thresholds?")
    lines.append("   → Ajuster thresholds ou accepter zone floue WIP?")
    lines.append("")
    lines.append("5. Familles de Γ persistent-elles entre analyses?")
    lines.append("   → Valider clustering comme vraie structure")
    lines.append("")
    lines.append("="*80)
    
    # Écrire fichier
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"\n✓ Rapport exporté: {output_path}")
    print(f"  {len(lines)} lignes générées")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyse exhaustive multi-configs/thresholds"
    )
    
    parser.add_argument('--output', type=str, required=True,
                       help='Fichier de sortie')
    parser.add_argument('--min-gammas', type=int, default=2,
                       help='Nombre minimum de Γ pour analyses (défaut: 2)')
    
    args = parser.parse_args()
    
    if not DB_RAW_PATH.exists() or not DB_RESULTS_PATH.exists():
        print("❌ Bases de données non trouvées")
        return
    
    # Scan espace disponible
    configs, thresholds, tests, gammas = scan_available_space()
    
    if len(gammas) < args.min_gammas:
        print(f"❌ Pas assez de Γ testés (min: {args.min_gammas})")
        return
    
    # Génération rapport
    print("\nGénération rapport complet...")
    generate_claude_report(configs, thresholds, tests, gammas, args.output)
    
    print("\n✓ Analyse terminée")
    print(f"\nPour analyse par Claude:")
    print(f"  cat {args.output}")


if __name__ == "__main__":
    main()