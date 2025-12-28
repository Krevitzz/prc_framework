#!/usr/bin/env python3
"""
prc_automation/analyze_and_export_all_configs.py

Analyse automatique de patterns et export pour Claude, 
pour TOUTES les configurations présentes dans la base.

Fonctionnalités:
1. Analyse les patterns pour chaque config individuellement
2. Compare les résultats entre configs
3. Exporte un résumé structuré pour Claude
4. Détecte les Γ sensibles aux pondérations

Usage:
    python analyze_and_export_all_configs.py --output analysis_all_configs.txt
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Set
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

DB_RAW_PATH = Path("prc_database/prc_r0_raw.db")
DB_RESULTS_PATH = Path("prc_database/prc_r0_results.db")


class PatternAnalyzer:
    """Analyseur de patterns pour une configuration donnée."""
    
    def __init__(self, config_id: str):
        self.config_id = config_id
        self.conn_results = sqlite3.connect(DB_RESULTS_PATH)
        self.cursor = self.conn_results.cursor()
        self.cursor.execute(f"ATTACH DATABASE '{DB_RAW_PATH}' AS db_raw")
    
    def close(self):
        """Ferme proprement les connexions."""
        try:
            self.cursor.execute("DETACH DATABASE db_raw")
            self.conn_results.close()
        except:
            pass
    
    def get_all_configs(self) -> List[str]:
        """Liste toutes les configurations disponibles."""
        self.cursor.execute("SELECT DISTINCT config_id FROM TestScores ORDER BY config_id")
        return [row[0] for row in self.cursor.fetchall()]
    
    def get_gamma_verdicts(self) -> List[Tuple]:
        """Récupère les verdicts pour la config courante."""
        self.cursor.execute("""
            SELECT gamma_id, verdict, score_global, majority_pct, robustness_pct
            FROM GammaVerdicts
            WHERE config_id = ? AND threshold_id = 'thresholds_default'
            ORDER BY score_global DESC
        """, (self.config_id,))
        return self.cursor.fetchall()
    
    def get_test_scores_summary(self) -> List[Tuple]:
        """Récupère les scores moyens par test."""
        self.cursor.execute("""
            SELECT 
                ts.test_name,
                AVG(ts.score) as avg_score,
                MIN(ts.score) as min_score,
                MAX(ts.score) as max_score,
                COUNT(DISTINCT e.gamma_id) as n_gammas
            FROM TestScores ts
            JOIN db_raw.Executions e ON ts.exec_id = e.id
            WHERE ts.config_id = ?
            GROUP BY ts.test_name
            ORDER BY ts.test_name
        """, (self.config_id,))
        return self.cursor.fetchall()
    
    def get_gamma_test_matrix(self) -> Tuple[List[str], List[str], Dict]:
        """Construit la matrice Γ × Test."""
        # Récupérer tous les Γ
        self.cursor.execute("""
            SELECT DISTINCT e.gamma_id 
            FROM db_raw.Executions e
            JOIN TestScores ts ON ts.exec_id = e.id
            WHERE ts.config_id = ?
            ORDER BY e.gamma_id
        """, (self.config_id,))
        gammas = [row[0] for row in self.cursor.fetchall()]
        
        # Récupérer tous les tests
        self.cursor.execute("""
            SELECT DISTINCT test_name 
            FROM TestScores 
            WHERE config_id = ? 
            ORDER BY test_name
        """, (self.config_id,))
        tests = [row[0] for row in self.cursor.fetchall()]
        
        # Récupérer les scores
        self.cursor.execute("""
            SELECT e.gamma_id, ts.test_name, AVG(ts.score)
            FROM TestScores ts
            JOIN db_raw.Executions e ON ts.exec_id = e.id
            WHERE ts.config_id = ?
            GROUP BY e.gamma_id, ts.test_name
        """, (self.config_id,))
        
        scores = {}
        for gamma_id, test_name, score in self.cursor.fetchall():
            if gamma_id not in scores:
                scores[gamma_id] = {}
            scores[gamma_id][test_name] = score
        
        return gammas, tests, scores
    
    def analyze_correlations(self, min_corr: float = 0.7) -> List[Dict]:
        """Analyse les corrélations entre tests."""
        gammas, tests, scores = self.get_gamma_test_matrix()
        
        if len(gammas) < 3 or len(tests) < 2:
            return []
        
        # Construire DataFrame pour corrélations
        data = []
        for gamma_id in gammas:
            row = []
            for test in tests:
                row.append(scores.get(gamma_id, {}).get(test, 0.0))
            data.append(row)
        
        df = pd.DataFrame(data, columns=tests, index=gammas)
        
        # Calculer corrélations avec suppression d'avertissement
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            correlations = []
            for i in range(len(tests)):
                for j in range(i + 1, len(tests)):
                    test_a = tests[i]
                    test_b = tests[j]
                    
                    try:
                        corr, p_value = spearmanr(df[test_a], df[test_b])
                        
                        if abs(corr) > min_corr and p_value < 0.05:
                            correlations.append({
                                'test_a': test_a,
                                'test_b': test_b,
                                'correlation': corr,
                                'p_value': p_value,
                                'direction': 'positive' if corr > 0 else 'negative'
                            })
                    except:
                        # Ignorer les erreurs de corrélation
                        continue
        
        return correlations
    
    def analyze_behavior_clusters(self, n_clusters: int = 3) -> List[Dict]:
        """Analyse les clusters de comportement."""
        gammas, tests, scores = self.get_gamma_test_matrix()
        
        if len(gammas) < 3:
            return []
        
        # Préparer données pour clustering
        data = []
        for gamma_id in gammas:
            row = []
            for test in tests:
                row.append(scores.get(gamma_id, {}).get(test, 0.0))
            data.append(row)
        
        # Clustering hiérarchique
        try:
            distances = pdist(data, metric='euclidean')
            linkage_matrix = linkage(distances, method='ward')
            
            n_clusters = min(n_clusters, len(gammas) // 2)
            if n_clusters < 2:
                return []
            
            clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Analyser clusters
            clusters_analysis = []
            for cluster_id in range(1, n_clusters + 1):
                cluster_gammas = [gammas[i] for i in range(len(gammas)) if clusters[i] == cluster_id]
                
                if len(cluster_gammas) < 2:
                    continue
                
                # Caractériser le cluster
                cluster_scores = []
                for test in tests:
                    test_scores = [scores[g].get(test, 0.0) for g in cluster_gammas if g in scores]
                    cluster_scores.append(np.mean(test_scores) if test_scores else 0.0)
                
                # Tests où le cluster performe bien (>0.7) ou mal (<0.3)
                good_tests = [tests[i] for i, s in enumerate(cluster_scores) if s > 0.7]
                bad_tests = [tests[i] for i, s in enumerate(cluster_scores) if s < 0.3]
                
                clusters_analysis.append({
                    'cluster_id': cluster_id,
                    'members': cluster_gammas,
                    'size': len(cluster_gammas),
                    'good_tests': good_tests[:3],  # Limiter à 3 pour lisibilité
                    'bad_tests': bad_tests[:3],
                    'avg_score': np.mean(cluster_scores)
                })
            
            return clusters_analysis
        except Exception as e:
            # Si clustering échoue, retourner liste vide
            return []
    
    def analyze_problematic_d_bases(self, threshold_low: float = 0.3, threshold_high: float = 0.8) -> List[Dict]:
        """Analyse les bases D problématiques ou triviales."""
        self.cursor.execute("""
            SELECT 
                e.d_base_id,
                AVG(ts.score) as avg_score,
                COUNT(DISTINCT e.gamma_id) as n_gammas
            FROM db_raw.Executions e
            JOIN TestScores ts ON ts.exec_id = e.id
            WHERE ts.config_id = ?
            GROUP BY e.d_base_id
            HAVING n_gammas >= 2
        """, (self.config_id,))
        
        results = self.cursor.fetchall()
        
        problematic = []
        for d_base_id, avg_score, n_gammas in results:
            if avg_score < threshold_low:
                problematic.append({
                    'd_base_id': d_base_id,
                    'type': 'PROBLEMATIC',
                    'avg_score': avg_score,
                    'n_gammas': n_gammas,
                    'reason': f'Tous les Γ échouent (moyenne={avg_score:.3f})'
                })
            elif avg_score > threshold_high:
                problematic.append({
                    'd_base_id': d_base_id,
                    'type': 'TRIVIAL',
                    'avg_score': avg_score,
                    'n_gammas': n_gammas,
                    'reason': f'Tous les Γ réussissent (moyenne={avg_score:.3f})'
                })
        
        return problematic
    
    def get_total_tests_count(self) -> int:
        """Retourne le nombre total de tests distincts."""
        self.cursor.execute("SELECT COUNT(DISTINCT test_name) FROM TestScores")
        return self.cursor.fetchone()[0]


def analyze_all_configs_and_export(output_path: str):
    """Analyse toutes les configs et exporte pour Claude."""
    
    print(f"\n{'='*80}")
    print("ANALYSE COMPLÈTE POUR TOUTES LES CONFIGURATIONS")
    print(f"{'='*80}\n")
    
    # Initialiser pour récupérer la liste des configs
    temp_analyzer = PatternAnalyzer("weights_default")
    all_configs = temp_analyzer.get_all_configs()
    temp_analyzer.close()
    
    print(f"Configurations détectées: {', '.join(all_configs)}")
    
    # Compter tests totaux
    total_tests = temp_analyzer.get_total_tests_count()
    
    lines = []
    
    # Header
    lines.append("="*80)
    lines.append("RAPPORT D'ANALYSE COMPLET - TOUTES CONFIGURATIONS")
    lines.append("="*80)
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Configurations analysées: {len(all_configs)}")
    lines.append("")
    
    # Statistiques globales
    lines.append("STATISTIQUES GLOBALES")
    lines.append("-"*80)
    
    # Connexion temporaire pour stats globales
    conn = sqlite3.connect(DB_RAW_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM Executions")
    n_runs = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT gamma_id) FROM Executions")
    n_gammas = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT d_base_id) FROM Executions")
    n_d_bases = cursor.fetchone()[0]
    
    conn.close()
    
    lines.append(f"Runs totaux: {n_runs}")
    lines.append(f"Γ uniques: {n_gammas}")
    lines.append(f"Bases D uniques: {n_d_bases}")
    lines.append(f"Tests distincts: {total_tests}")
    lines.append("")
    
    # Analyser chaque configuration
    all_verdicts_by_config = {}
    all_correlations_by_config = {}
    all_clusters_by_config = {}
    all_problematic_d_by_config = {}
    
    for config_id in all_configs:
        lines.append(f"\n{'='*80}")
        lines.append(f"CONFIGURATION: {config_id}")
        lines.append(f"{'='*80}\n")
        
        # Utiliser un nouvel analyseur pour cette config
        analyzer = PatternAnalyzer(config_id)
        
        # 1. Verdicts
        verdicts = analyzer.get_gamma_verdicts()
        all_verdicts_by_config[config_id] = verdicts
        
        lines.append("DISTRIBUTION DES VERDICTS:")
        lines.append("-"*40)
        
        verdict_counts = {}
        for _, verdict, _, _, _ in verdicts:
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        
        for verdict, count in sorted(verdict_counts.items()):
            pct = (count / len(verdicts)) * 100
            lines.append(f"  {verdict:<25} {count:>3} ({pct:>5.1f}%)")
        lines.append("")
        
        # 2. Scores par test
        test_scores = analyzer.get_test_scores_summary()
        
        lines.append("PERFORMANCE DES TESTS (moyenne sur tous les Γ):")
        lines.append("-"*40)
        lines.append(f"{'Test':<20} {'Moyenne':<8} {'Min':<8} {'Max':<8} {'N Γ':<6}")
        lines.append("-"*40)
        
        for test_name, avg, min_s, max_s, n_g in test_scores:
            lines.append(f"  {test_name:<20} {avg:>7.3f}  {min_s:>7.3f}  {max_s:>7.3f}  {n_g:>6}")
        lines.append("")
        
        # 3. Corrélations
        correlations = analyzer.analyze_correlations()
        all_correlations_by_config[config_id] = correlations
        
        if correlations:
            lines.append("CORRÉLATIONS SIGNIFICATIVES ENTRE TESTS (|ρ| > 0.7):")
            lines.append("-"*40)
            for corr in correlations[:5]:  # Limiter à 5 pour lisibilité
                lines.append(f"  {corr['test_a']} ↔ {corr['test_b']}: ρ = {corr['correlation']:.3f} ({corr['direction']})")
        else:
            lines.append("Aucune corrélation forte détectée")
        lines.append("")
        
        # 4. Clusters de comportement
        clusters = analyzer.analyze_behavior_clusters()
        all_clusters_by_config[config_id] = clusters
        
        if clusters:
            lines.append("CLUSTERS DE COMPORTEMENT DÉTECTÉS:")
            lines.append("-"*40)
            for cluster in clusters:
                desc = f"Cluster {cluster['cluster_id']} ({cluster['size']} Γ): "
                if cluster['good_tests']:
                    desc += f"Réussit {', '.join(cluster['good_tests'])}"
                if cluster['bad_tests']:
                    if cluster['good_tests']:
                        desc += ", "
                    desc += f"Échoue {', '.join(cluster['bad_tests'])}"
                lines.append(f"  {desc}")
        lines.append("")
        
        # 5. Bases D problématiques
        problematic_d = analyzer.analyze_problematic_d_bases()
        all_problematic_d_by_config[config_id] = problematic_d
        
        if problematic_d:
            lines.append("BASES D PROBLÉMATIQUES:")
            lines.append("-"*40)
            for d_info in problematic_d:
                lines.append(f"  {d_info['d_base_id']}: {d_info['type']} - {d_info['reason']} ({d_info['n_gammas']} Γ testés)")
        else:
            lines.append("Aucune base D particulièrement problématique détectée")
        
        analyzer.close()
    
    # =========================================================================
    # ANALYSE COMPARATIVE ENTRE CONFIGS
    # =========================================================================
    
    lines.append(f"\n{'='*80}")
    lines.append("ANALYSE COMPARATIVE ENTRE CONFIGURATIONS")
    lines.append(f"{'='*80}\n")
    
    if len(all_configs) > 1:
        # Comparer les verdicts
        lines.append("SENSIBILITÉ DES Γ AUX PONDÉRATIONS:")
        lines.append("-"*40)
        
        # Pour chaque Γ, voir si le verdict change entre configs
        gamma_verdicts_all = {}
        for config_id in all_configs:
            for gamma_id, verdict, _, _, _ in all_verdicts_by_config[config_id]:
                if gamma_id not in gamma_verdicts_all:
                    gamma_verdicts_all[gamma_id] = {}
                gamma_verdicts_all[gamma_id][config_id] = verdict
        
        # Compter Γ sensibles (verdict change)
        sensitive_gammas = []
        stable_gammas = []
        
        for gamma_id, verdicts_by_config in gamma_verdicts_all.items():
            all_verdicts = list(verdicts_by_config.values())
            if len(set(all_verdicts)) > 1:
                sensitive_gammas.append(gamma_id)
            else:
                stable_gammas.append(gamma_id)
        
        lines.append(f"Γ stables (même verdict toutes configs): {len(stable_gammas)}")
        lines.append(f"Γ sensibles (verdict change): {len(sensitive_gammas)}")
        
        if sensitive_gammas:
            lines.append("\nΓ les plus sensibles (verdicts par config):")
            for gamma_id in sensitive_gammas[:5]:  # 5 premiers
                verdict_str = ", ".join([f"{c}: {gamma_verdicts_all[gamma_id][c]}" 
                                       for c in all_configs if c in gamma_verdicts_all[gamma_id]])
                lines.append(f"  {gamma_id}: {verdict_str}")
        
        # Comparer les patterns de corrélation
        lines.append("\nSTABILITÉ DES CORRÉLATIONS ENTRE CONFIGS:")
        lines.append("-"*40)
        
        # Identifier corrélations présentes dans toutes/most configs
        all_corr_pairs = set()
        for config_id, correlations in all_correlations_by_config.items():
            for corr in correlations:
                pair = tuple(sorted([corr['test_a'], corr['test_b']]))
                all_corr_pairs.add(pair)
        
        if all_corr_pairs:
            lines.append("Corrélations détectées (au moins dans une config):")
            for test_a, test_b in sorted(list(all_corr_pairs))[:10]:  # 10 premières
                # Compter dans combien de configs cette corrélation apparaît
                count_configs = 0
                for config_id, correlations in all_correlations_by_config.items():
                    for corr in correlations:
                        if (corr['test_a'] == test_a and corr['test_b'] == test_b) or \
                           (corr['test_a'] == test_b and corr['test_b'] == test_a):
                            count_configs += 1
                            break
                
                lines.append(f"  {test_a} ↔ {test_b}: présente dans {count_configs}/{len(all_configs)} configs")
        else:
            lines.append("Aucune corrélation forte détectée dans aucune config")
    else:
        lines.append("Une seule configuration disponible - pas d'analyse comparative possible")
    
    # =========================================================================
    # RECOMMANDATIONS
    # =========================================================================
    
    lines.append(f"\n{'='*80}")
    lines.append("RECOMMANDATIONS ET QUESTIONS POUR ANALYSE")
    lines.append(f"{'='*80}\n")
    
    lines.append("QUESTIONS D'ANALYSE:")
    lines.append("1. Pourquoi certains Γ sont-ils sensibles aux pondérations ?")
    lines.append("   → Analyser quels tests font basculer leur verdict")
    lines.append("")
    lines.append("2. Les corrélations entre tests sont-elles robustes ?")
    lines.append("   → Celles présentes dans toutes configs sont-elles plus significatives ?")
    lines.append("")
    lines.append("3. Y a-t-il des clusters de comportement cohérents ?")
    lines.append("   → Les mêmes familles de Γ émergent-elles dans toutes configs ?")
    lines.append("")
    lines.append("4. Que faire des bases D problématiques/triviales ?")
    lines.append("   → Faut-il les exclure ou les étudier spécifiquement ?")
    lines.append("")
    lines.append("5. Quels tests sont les plus discriminants ?")
    lines.append("   → Ceux avec la plus grande variance entre Γ")
    lines.append("")
    lines.append("SUGGESTIONS DE CONTRAINTES CON-GAM-XXX:")
    lines.append("- Si une famille de Γ échoue systématiquement sur certaines bases D")
    lines.append("- Si certains tests sont toujours échoués par les mêmes Γ")
    lines.append("- Si des patterns de corrélation suggèrent des redondances")
    lines.append("")
    
    lines.append("="*80)
    lines.append("FIN DU RAPPORT")
    lines.append("="*80)
    
    # Écrire le fichier
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ Rapport exporté: {output_path}")
    print(f"✓ Configurations analysées: {len(all_configs)}")
    print(f"✓ Γ analysés: {n_gammas}")
    print(f"✓ Tests analysés: {total_tests}")
    
    # Afficher un aperçu
    print(f"\n{'='*80}")
    print("APERÇU DES CONFIGURATIONS:")
    for config_id in all_configs:
        analyzer = PatternAnalyzer(config_id)
        verdicts = analyzer.get_gamma_verdicts()
        
        # Compter verdicts
        verdict_counts = {}
        for _, verdict, _, _, _ in verdicts:
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        
        print(f"\n{config_id}:")
        for verdict, count in sorted(verdict_counts.items()):
            pct = (count / len(verdicts)) * 100
            print(f"  {verdict}: {count} ({pct:.1f}%)")
        
        analyzer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyse complète pour toutes les configurations et export pour Claude"
    )
    
    parser.add_argument('--output', type=str, default='analysis_all_configs.txt',
                       help='Fichier de sortie pour le rapport')
    
    args = parser.parse_args()
    
    if not DB_RAW_PATH.exists() or not DB_RESULTS_PATH.exists():
        print(f"❌ Bases de données non trouvées:")
        print(f"   - {DB_RAW_PATH}")
        print(f"   - {DB_RESULTS_PATH}")
        return
    
    # Supprimer l'avertissement ConstantInputWarning
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    analyze_all_configs_and_export(args.output)
    
    print(f"\n{'='*80}")
    print("INSTRUCTIONS:")
    print("1. Copiez-collez le contenu du fichier dans Claude")
    print("2. Posez des questions spécifiques basées sur les patterns détectés")
    print("3. Demandez des visualisations ou analyses supplémentaires")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()