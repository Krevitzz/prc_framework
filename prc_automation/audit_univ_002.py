#!/usr/bin/env python3
"""
audit_univ_002.py

Audit exhaustif post-hoc du test UNIV-002 (diversité).
Conforme Charte PRC 5.1 Section 14 - AUCUN rerun.

Usage:
    python audit_univ_002.py

Output:
    audit_univ_002_report.md (rapport structuré)
"""

import sys
import sqlite3
import json
import gzip
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from scipy.signal import find_peaks


# =============================================================================
# CONFIGURATION
# =============================================================================

DB_RAW_PATH = Path("prc_database/prc_r0_raw.db")
DB_RESULTS_PATH = Path("prc_database/prc_r0_results.db")
REPORT_PATH = Path("audit_univ_002_report.md")
FIGURES_DIR = Path("figures_audit_univ_002")

# Créer dossier figures
FIGURES_DIR.mkdir(exist_ok=True)


# =============================================================================
# STRUCTURES DE DONNÉES
# =============================================================================

@dataclass
class ExecutionContext:
    """Contexte d'une exécution."""
    exec_id: int
    gamma_id: str
    gamma_params: Dict
    d_base_id: str
    modifier_id: str
    seed: int
    final_iteration: int


@dataclass
class UNIV002Data:
    """Données brutes UNIV-002 pour une exécution."""
    exec_id: int
    context: ExecutionContext
    
    # Observations brutes
    initial_diversity: float
    final_diversity: float
    ratio: float
    evolution: str  # "increasing" | "decreasing" | "stable"
    
    # Score normalisé
    score: float
    
    # Série temporelle (si disponible)
    diversity_series: List[float] = None  # std_value par itération
    iterations: List[int] = None


@dataclass
class AuditResult:
    """Résultat d'un audit spécifique."""
    section: str
    observations: List[str]
    signals_strong: List[str]
    signals_weak: List[str]
    figures: List[Path]


# =============================================================================
# EXTRACTION DONNÉES
# =============================================================================

def connect_databases() -> Tuple[sqlite3.Connection, sqlite3.Connection]:
    """Connecte aux deux bases de données."""
    if not DB_RAW_PATH.exists():
        raise FileNotFoundError(f"db_raw not found: {DB_RAW_PATH}")
    if not DB_RESULTS_PATH.exists():
        raise FileNotFoundError(f"db_results not found: {DB_RESULTS_PATH}")
    
    conn_raw = sqlite3.connect(DB_RAW_PATH)
    conn_results = sqlite3.connect(DB_RESULTS_PATH)
    
    # Attacher db_raw depuis db_results pour requêtes cross-DB
    cursor_results = conn_results.cursor()
    cursor_results.execute(f"ATTACH DATABASE '{DB_RAW_PATH}' AS db_raw")
    
    return conn_raw, conn_results


def extract_univ002_data(conn_raw: sqlite3.Connection, 
                         conn_results: sqlite3.Connection) -> List[UNIV002Data]:
    """
    Extrait toutes les données UNIV-002 depuis les bases.
    
    Returns:
        Liste de UNIV002Data pour chaque exécution
    """
    cursor_results = conn_results.cursor()
    
    # Récupérer toutes les observations UNIV-002
    cursor_results.execute("""
        SELECT 
            ts.exec_id,
            e.gamma_id,
            e.d_base_id,
            e.modifier_id,
            e.seed,
            e.final_iteration,
            obs.observation_data,
            ts.score
        FROM TestScores ts
        JOIN db_raw.Executions e ON ts.exec_id = e.id
        JOIN TestObservations obs ON obs.exec_id = ts.exec_id AND obs.test_name = ts.test_name
        WHERE ts.test_name = 'UNIV-002'
          AND ts.config_id = 'weights_default'
          AND e.status = 'COMPLETED'
        ORDER BY e.gamma_id, e.d_base_id, e.modifier_id, e.seed
    """)
    
    data_list = []
    
    for row in cursor_results.fetchall():
        exec_id, gamma_id, d_base_id, modifier_id, seed, final_iter, obs_json, score = row
        
        # Parser observation JSON
        obs_data = json.loads(obs_json)
        
        # Extraire paramètres Γ depuis Executions
        cursor_raw = conn_raw.cursor()
        cursor_raw.execute("""
            SELECT alpha, beta, gamma_param, omega, memory_weight, 
                   epsilon, sigma, lambda_param, eta
            FROM Executions WHERE id = ?
        """, (exec_id,))
        params_row = cursor_raw.fetchone()
        gamma_params = {
            k: v for k, v in zip(
                ['alpha', 'beta', 'gamma', 'omega', 'memory_weight', 
                 'epsilon', 'sigma', 'lambda', 'eta'],
                params_row
            ) if v is not None
        }
        
        context = ExecutionContext(
            exec_id=exec_id,
            gamma_id=gamma_id,
            gamma_params=gamma_params,
            d_base_id=d_base_id,
            modifier_id=modifier_id,
            seed=seed,
            final_iteration=final_iter
        )
        
        # Extraire série temporelle depuis Metrics
        cursor_raw.execute("""
            SELECT iteration, std_value
            FROM Metrics
            WHERE exec_id = ?
            ORDER BY iteration
        """, (exec_id,))
        
        metrics_rows = cursor_raw.fetchall()
        diversity_series = [row[1] for row in metrics_rows if row[1] is not None]
        iterations = [row[0] for row in metrics_rows if row[1] is not None]
        
        data = UNIV002Data(
            exec_id=exec_id,
            context=context,
            initial_diversity=obs_data.get('initial_value', 0.0),
            final_diversity=obs_data.get('final_value', 0.0),
            ratio=obs_data.get('final_value', 0.0) / obs_data.get('initial_value', 1.0) if obs_data.get('initial_value', 1.0) > 0 else 0.0,
            evolution=obs_data.get('transition', 'unknown'),
            score=score,
            diversity_series=diversity_series if diversity_series else None,
            iterations=iterations if iterations else None
        )
        
        data_list.append(data)
    
    return data_list


# =============================================================================
# AUDIT 1 : DISTRIBUTIONS BRUTES
# =============================================================================

def audit_1_distributions(data_list: List[UNIV002Data]) -> AuditResult:
    """
    Audit 1 : Distributions des valeurs brutes UNIV-002.
    
    Questions:
    - Distribution des ratios final/initial
    - Histogrammes par Γ
    - Outliers
    """
    observations = []
    signals_strong = []
    signals_weak = []
    figures = []
    
    # Extraire ratios
    ratios = [d.ratio for d in data_list]
    scores = [d.score for d in data_list]
    
    # Statistiques globales
    observations.append(f"Nombre d'exécutions: {len(data_list)}")
    observations.append(f"Ratio moyen: {np.mean(ratios):.3f}")
    observations.append(f"Ratio médian: {np.median(ratios):.3f}")
    observations.append(f"Ratio min/max: [{np.min(ratios):.3f}, {np.max(ratios):.3f}]")
    observations.append(f"Écart-type ratios: {np.std(ratios):.3f}")
    
    observations.append(f"\nScore moyen: {np.mean(scores):.3f}")
    observations.append(f"Score médian: {np.median(scores):.3f}")
    observations.append(f"Score min/max: [{np.min(scores):.3f}, {np.max(scores):.3f}]")
    
    # Distribution ratios < 1 vs > 1
    n_decrease = sum(1 for r in ratios if r < 0.9)
    n_stable = sum(1 for r in ratios if 0.9 <= r <= 1.1)
    n_increase = sum(1 for r in ratios if r > 1.1)
    
    observations.append(f"\nRépartition dynamique:")
    observations.append(f"  Décroissance (ratio < 0.9): {n_decrease} ({100*n_decrease/len(ratios):.1f}%)")
    observations.append(f"  Stable (0.9 ≤ ratio ≤ 1.1): {n_stable} ({100*n_stable/len(ratios):.1f}%)")
    observations.append(f"  Croissance (ratio > 1.1): {n_increase} ({100*n_increase/len(ratios):.1f}%)")
    
    # Signaux
    if n_decrease / len(ratios) > 0.8:
        signals_strong.append(f"DÉCROISSANCE MASSIVE: {100*n_decrease/len(ratios):.0f}% des exécutions perdent diversité")
    
    if np.std(ratios) < 0.2:
        signals_weak.append("Faible variance des ratios (comportement uniforme)")
    
    # Histogramme global
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Ratios
    axes[0, 0].hist(ratios, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(1.0, color='red', linestyle='--', label='Ratio=1 (stable)')
    axes[0, 0].set_xlabel('Ratio diversity_final / diversity_initial')
    axes[0, 0].set_ylabel('Fréquence')
    axes[0, 0].set_title('Distribution des ratios UNIV-002')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Scores
    axes[0, 1].hist(scores, bins=30, alpha=0.7, edgecolor='black', color='orange')
    axes[0, 1].set_xlabel('Score UNIV-002 (0-1)')
    axes[0, 1].set_ylabel('Fréquence')
    axes[0, 1].set_title('Distribution des scores UNIV-002')
    axes[0, 1].grid(alpha=0.3)
    
    # Scatter ratio vs score
    axes[1, 0].scatter(ratios, scores, alpha=0.5)
    axes[1, 0].set_xlabel('Ratio diversity')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Relation ratio → score')
    axes[1, 0].grid(alpha=0.3)
    
    # Distribution par Γ
    gamma_ids = sorted(set(d.context.gamma_id for d in data_list))
    ratios_by_gamma = {g: [d.ratio for d in data_list if d.context.gamma_id == g] 
                       for g in gamma_ids}
    
    axes[1, 1].boxplot([ratios_by_gamma[g] for g in gamma_ids], 
                       labels=gamma_ids, vert=True)
    axes[1, 1].axhline(1.0, color='red', linestyle='--')
    axes[1, 1].set_ylabel('Ratio diversity')
    axes[1, 1].set_title('Distribution par Γ')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "audit1_distributions.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    figures.append(fig_path)
    
    return AuditResult(
        section="1. Distributions brutes",
        observations=observations,
        signals_strong=signals_strong,
        signals_weak=signals_weak,
        figures=figures
    )


# =============================================================================
# AUDIT 2 : ANALYSE TEMPORELLE
# =============================================================================

def audit_2_temporal(data_list: List[UNIV002Data]) -> AuditResult:
    """
    Audit 2 : Évolution temporelle de la diversité.
    
    Questions:
    - Collapse rapide vs décroissance lente ?
    - Temps caractéristique (demi-vie)
    - Patterns communs (plateau, oscillations)
    """
    observations = []
    signals_strong = []
    signals_weak = []
    figures = []
    
    # Filtrer données avec séries temporelles
    data_with_series = [d for d in data_list if d.diversity_series and len(d.diversity_series) > 10]
    
    observations.append(f"Exécutions avec séries temporelles: {len(data_with_series)}/{len(data_list)}")
    
    if len(data_with_series) == 0:
        signals_strong.append("AUCUNE SÉRIE TEMPORELLE disponible - audit impossible")
        return AuditResult(
            section="2. Analyse temporelle",
            observations=observations,
            signals_strong=signals_strong,
            signals_weak=signals_weak,
            figures=figures
        )
    
    # Analyser patterns de décroissance
    half_lives = []
    collapse_types = {'fast': 0, 'slow': 0, 'plateau': 0, 'oscillating': 0}
    
    for d in data_with_series:
        series = np.array(d.diversity_series)
        initial = series[0]
        final = series[-1]
        
        if initial == 0:
            continue
        
        # Calcul demi-vie (itération où diversité < 0.5 * initial)
        half_threshold = 0.5 * initial
        half_idx = np.where(series < half_threshold)[0]
        
        if len(half_idx) > 0:
            half_life = half_idx[0]
            half_lives.append(half_life)
            
            # Classifier type de collapse
            if half_life < len(series) * 0.1:
                collapse_types['fast'] += 1
            elif half_life < len(series) * 0.5:
                collapse_types['slow'] += 1
            else:
                collapse_types['plateau'] += 1
        else:
            collapse_types['plateau'] += 1
        
        # Détecter oscillations
        peaks, _ = find_peaks(series)
        if len(peaks) > 3:
            collapse_types['oscillating'] += 1
    
    # Statistiques
    if half_lives:
        observations.append(f"\nDemi-vie moyenne: {np.mean(half_lives):.1f} itérations")
        observations.append(f"Demi-vie médiane: {np.median(half_lives):.1f} itérations")
        observations.append(f"Demi-vie min/max: [{np.min(half_lives):.0f}, {np.max(half_lives):.0f}]")
    
    observations.append(f"\nTypes de collapse:")
    for ctype, count in collapse_types.items():
        pct = 100 * count / len(data_with_series)
        observations.append(f"  {ctype}: {count} ({pct:.1f}%)")
    
    # Signaux
    if collapse_types['fast'] / len(data_with_series) > 0.5:
        signals_strong.append(f"COLLAPSE RAPIDE dominant: {collapse_types['fast']}/{len(data_with_series)} exécutions")
    
    if collapse_types['oscillating'] > len(data_with_series) * 0.1:
        signals_weak.append(f"Oscillations détectées: {collapse_types['oscillating']} cas")
    
    # Visualisations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Trajectoires échantillon (10 aléatoires par Γ)
    gamma_ids = sorted(set(d.context.gamma_id for d in data_with_series))
    colors = plt.cm.tab10(np.linspace(0, 1, len(gamma_ids)))
    
    for i, gamma_id in enumerate(gamma_ids[:5]):  # Limiter à 5 Γ
        gamma_data = [d for d in data_with_series if d.context.gamma_id == gamma_id]
        sample = np.random.choice(gamma_data, min(3, len(gamma_data)), replace=False)
        
        for d in sample:
            norm_series = np.array(d.diversity_series) / d.diversity_series[0] if d.diversity_series[0] > 0 else d.diversity_series
            axes[0, 0].plot(d.iterations, norm_series, alpha=0.3, color=colors[i])
    
    axes[0, 0].axhline(0.5, color='red', linestyle='--', label='50% initial')
    axes[0, 0].set_xlabel('Itération')
    axes[0, 0].set_ylabel('Diversity normalisée')
    axes[0, 0].set_title('Trajectoires échantillon (normalisées à t=0)')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Histogramme demi-vies
    if half_lives:
        axes[0, 1].hist(half_lives, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Demi-vie (itérations)')
        axes[0, 1].set_ylabel('Fréquence')
        axes[0, 1].set_title('Distribution des demi-vies')
        axes[0, 1].grid(alpha=0.3)
    
    # Types de collapse
    axes[1, 0].bar(collapse_types.keys(), collapse_types.values())
    axes[1, 0].set_ylabel('Nombre d\'exécutions')
    axes[1, 0].set_title('Types de collapse')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(alpha=0.3)
    
    # Moyenne par Γ
    mean_trajectories = {}
    for gamma_id in gamma_ids:
        gamma_data = [d for d in data_with_series if d.context.gamma_id == gamma_id]
        if gamma_data:
            # Interpoler sur grille commune
            max_len = max(len(d.diversity_series) for d in gamma_data)
            interpolated = []
            for d in gamma_data:
                if d.diversity_series[0] > 0:
                    norm = np.array(d.diversity_series) / d.diversity_series[0]
                    interpolated.append(np.interp(np.linspace(0, 1, max_len), 
                                                  np.linspace(0, 1, len(norm)), 
                                                  norm))
            
            if interpolated:
                mean_trajectories[gamma_id] = np.mean(interpolated, axis=0)
    
    for i, (gamma_id, traj) in enumerate(mean_trajectories.items()):
        axes[1, 1].plot(np.linspace(0, 1, len(traj)), traj, 
                       label=gamma_id, color=colors[i % len(colors)])
    
    axes[1, 1].axhline(0.5, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Progression normalisée (0=début, 1=fin)')
    axes[1, 1].set_ylabel('Diversity normalisée')
    axes[1, 1].set_title('Trajectoires moyennes par Γ')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "audit2_temporal.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    figures.append(fig_path)
    
    return AuditResult(
        section="2. Analyse temporelle",
        observations=observations,
        signals_strong=signals_strong,
        signals_weak=signals_weak,
        figures=figures
    )


# =============================================================================
# AUDIT 3 : VALIDATION SCORING
# =============================================================================

def audit_3_scoring(data_list: List[UNIV002Data]) -> AuditResult:
    """
    Audit 3 : Validation du mapping mesure → score.
    
    Questions:
    - Fonction de scoring actuelle est-elle appropriée ?
    - Effet des seuils
    - Scores alternatifs
    """
    observations = []
    signals_strong = []
    signals_weak = []
    figures = []
    
    ratios = np.array([d.ratio for d in data_list])
    scores_actual = np.array([d.score for d in data_list])
    
    # Reconstruire fonction de scoring (depuis tests/utilities/scoring.py implicite)
    # Hypothèse : scoring basé sur ratio avec seuils [0.3, 3.0]
    
    def score_linear(ratio):
        """Score linéaire simple: 1.0 si ratio=1.0, décroît linéairement."""
        if ratio >= 1.0:
            return 1.0
        else:
            return max(0.0, ratio)
    
    def score_sigmoid(ratio):
        """Score sigmoïde centré sur 1.0."""
        return 1.0 / (1.0 + np.exp(-5 * (ratio - 1.0)))
    
    scores_linear = np.array([score_linear(r) for r in ratios])
    scores_sigmoid = np.array([score_sigmoid(r) for r in ratios])
    
    # Comparaison
    corr_linear, _ = pearsonr(scores_actual, scores_linear)
    corr_sigmoid, _ = pearsonr(scores_actual, scores_sigmoid)
    
    observations.append("Fonction de scoring actuelle:")
    observations.append(f"  Corrélation avec score linéaire: {corr_linear:.3f}")
    observations.append(f"  Corrélation avec score sigmoïde: {corr_sigmoid:.3f}")
    
    # Analyser effet des seuils
    # Seuils implicites : ratio < 0.3 → score proche 0, ratio > 3.0 → score proche 1
    n_below_03 = sum(1 for r in ratios if r < 0.3)
    n_above_30 = sum(1 for r in ratios if r > 3.0)
    
    observations.append(f"\nDistribution vs seuils:")
    observations.append(f"  Ratio < 0.3 (score ≈ 0): {n_below_03} ({100*n_below_03/len(ratios):.1f}%)")
    observations.append(f"  Ratio > 3.0 (score ≈ 1): {n_above_30} ({100*n_above_30/len(ratios):.1f}%)")
    observations.append(f"  Ratio dans [0.3, 3.0]: {len(ratios)-n_below_03-n_above_30} ({100*(len(ratios)-n_below_03-n_above_30)/len(ratios):.1f}%)")
    
    # Signaux
    if n_below_03 / len(ratios) > 0.5:
        signals_strong.append(f"SEUIL INFÉRIEUR SATURÉ: {100*n_below_03/len(ratios):.0f}% des ratios < 0.3")
    
    if n_above_30 == 0:
        signals_weak.append("AUCUN ratio > 3.0 (seuil supérieur jamais atteint)")
    
    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Scatter ratio vs score actuel
    axes[0].scatter(ratios, scores_actual, alpha=0.5, label='Score actuel')
    axes[0].plot([0, 5], [0, 1], 'r--', label='Linéaire idéal', alpha=0.5)
    axes[0].axvline(0.3, color='orange', linestyle='--', alpha=0.5, label='Seuil inf')
    axes[0].axvline(3.0, color='green', linestyle='--', alpha=0.5, label='Seuil sup')
    axes[0].set_xlabel('Ratio diversity')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Mapping ratio → score actuel')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Comparaison fonctions de scoring
    axes[1].scatter(ratios, scores_actual, alpha=0.3, label='Actuel', s=10)
    axes[1].scatter(ratios, scores_linear, alpha=0.3, label='Linéaire', s=10)
    axes[1].scatter(ratios, scores_sigmoid, alpha=0.3, label='Sigmoïde', s=10)
    axes[1].set_xlabel('Ratio diversity')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Comparaison fonctions de scoring')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Histogramme différences
    diff_linear = scores_actual - scores_linear
    diff_sigmoid = scores_actual - scores_sigmoid
    
    axes[2].hist(diff_linear, bins=30, alpha=0.5, label='Actuel - Linéaire', edgecolor='black')
    axes[2].hist(diff_sigmoid, bins=30, alpha=0.5, label='Actuel - Sigmoïde', edgecolor='black')
    axes[2].axvline(0, color='red', linestyle='--')
    axes[2].set_xlabel('Différence de score')
    axes[2].set_ylabel('Fréquence')
    axes[2].set_title('Écarts vs fonctions alternatives')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "audit3_scoring.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    figures.append(fig_path)
    
    return AuditResult(
        section="3. Validation scoring",
        observations=observations,
        signals_strong=signals_strong,
        signals_weak=signals_weak,
        figures=figures
    )


# =============================================================================
# AUDIT 4 : SENSIBILITÉ PARAMÈTRES
# =============================================================================

def audit_4_sensitivity(data_list: List[UNIV002Data]) -> AuditResult:
    """
    Audit 4 : Sensibilité aux paramètres Γ.
    
    Questions:
    - Effet des paramètres internes (β, α, etc.)
    - Robustesse inter-graines
    """
    observations = []
    signals_strong = []
    signals_weak = []
    figures = []
    
    # Grouper par Γ et paramètres
    gamma_groups = defaultdict(lambda: defaultdict(list))
    
    for d in data_list:
        gamma_id = d.context.gamma_id
        # Créer clé paramètres (hors seed)
        param_key = tuple(sorted(d.context.gamma_params.items()))
        gamma_groups[gamma_id][param_key].append(d)
    
    # Analyser variance intra-paramètres (robustesse seeds)
    observations.append("Robustesse inter-graines (variance intra-paramètres):\n")
    
    for gamma_id, param_dict in sorted(gamma_groups.items()):
        observations.append(f"\n{gamma_id}:")
        
        for param_key, runs in param_dict.items():
            if len(runs) < 2:
                continue
            
            scores = [r.score for r in runs]
            ratios = [r.ratio for r in runs]
            
            std_score = np.std(scores)
            std_ratio = np.std(ratios)
            
            observations.append(f"  Params {dict(param_key)}: "
                              f"std_score={std_score:.3f}, std_ratio={std_ratio:.3f} "
                              f"(n={len(runs)} seeds)")
            
            if std_score > 0.2:
                signals_weak.append(f"{gamma_id} avec params {dict(param_key)} : variance élevée inter-seeds (std={std_score:.3f})")
    
    # Analyser effet des paramètres (si plusieurs valeurs testées)
    observations.append("\n\nEffet des paramètres (si grille disponible):\n")
    
    for gamma_id, param_dict in sorted(gamma_groups.items()):
        if len(param_dict) < 2:
            observations.append(f"\n{gamma_id}: Un seul jeu de paramètres (pas d'analyse possible)")
            continue
        
        observations.append(f"\n{gamma_id}: {len(param_dict)} jeux de paramètres")
        
        # Calculer moyenne par jeu de paramètres
        param_means = {}
        for param_key, runs in param_dict.items():
            param_means[param_key] = np.mean([r.score for r in runs])
        
        # Variance inter-paramètres
        mean_scores = list(param_means.values())
        std_params = np.std(mean_scores)
        
        observations.append(f"  Variance inter-paramètres: std={std_params:.3f}")
        
        if std_params > 0.1:
            signals_weak.append(f"{gamma_id}: Sensibilité significative aux paramètres (std={std_params:.3f})")
    
    # Visualisation (limiter à quelques Γ)
    gamma_ids_sample = sorted(gamma_groups.keys())[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, gamma_id in enumerate(gamma_ids_sample):
        param_dict = gamma_groups[gamma_id]
        
        # Préparer données pour boxplot
        param_labels = []
        param_scores = []
        
        for param_key, runs in sorted(param_dict.items()):
            param_labels.append(f"P{len(param_labels)+1}")
            param_scores.append([r.score for r in runs])
        
        if param_scores:
            axes[i].boxplot(param_scores, labels=param_labels)
            axes[i].set_ylabel('Score UNIV-002')
            axes[i].set_title(f'{gamma_id} - Robustesse par paramètres')
            axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "audit4_sensitivity.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    figures.append(fig_path)
    
    return AuditResult(
        section="4. Sensibilité paramètres",
        observations=observations,
        signals_strong=signals_strong,
        signals_weak=signals_weak,
        figures=figures
    )


# =============================================================================
# AUDIT 5 : CORRÉLATIONS INTER-TESTS
# =============================================================================

def audit_5_correlations(conn_results: sqlite3.Connection, 
                         data_list: List[UNIV002Data]) -> AuditResult:
    """
    Audit 5 : Corrélations UNIV-002 avec autres tests.
    
    Questions:
    - Corrélation avec CONV-LYAPUNOV (attendue négative)
    - Corrélation avec BND-001, SYM-*
    """
    observations = []
    signals_strong = []
    signals_weak = []
    figures = []
    
    # Récupérer scores de tous les tests pour les mêmes exec_id
    exec_ids = [d.exec_id for d in data_list]
    
    cursor = conn_results.cursor()
    cursor.execute(f"""
        SELECT exec_id, test_name, score
        FROM TestScores
        WHERE exec_id IN ({','.join('?' for _ in exec_ids)})
          AND config_id = 'weights_default'
        ORDER BY exec_id, test_name
    """, exec_ids)
    
    # Construire matrice scores
    scores_matrix = defaultdict(dict)
    for exec_id, test_name, score in cursor.fetchall():
        scores_matrix[exec_id][test_name] = score
    
    # Tests à corréler
    target_tests = ['CONV-LYAPUNOV', 'BND-001', 'SYM-001', 'SYM-002', 'UNIV-001', 'UNIV-003']
    
    # Calculer corrélations
    observations.append("Corrélations UNIV-002 avec autres tests:\n")
    
    correlations = {}
    
    for test_name in target_tests:
        # Extraire paires (UNIV-002, test_name) pour exec_id communs
        pairs = []
        for d in data_list:
            if test_name in scores_matrix[d.exec_id]:
                pairs.append((d.score, scores_matrix[d.exec_id][test_name]))
        
        if len(pairs) < 3:
            continue
        
        scores_univ002, scores_test = zip(*pairs)
        
        # Pearson et Spearman
        corr_pearson, p_pearson = pearsonr(scores_univ002, scores_test)
        corr_spearman, p_spearman = spearmanr(scores_univ002, scores_test)
        
        correlations[test_name] = {
            'pearson': corr_pearson,
            'p_pearson': p_pearson,
            'spearman': corr_spearman,
            'p_spearman': p_spearman,
            'n': len(pairs)
        }
        
        observations.append(f"\n{test_name}:")
        observations.append(f"  Pearson: r={corr_pearson:.3f}, p={p_pearson:.4f}")
        observations.append(f"  Spearman: ρ={corr_spearman:.3f}, p={p_spearman:.4f}")
        observations.append(f"  n={len(pairs)} paires")
        
        # Signaux
        if abs(corr_pearson) > 0.5 and p_pearson < 0.001:
            if corr_pearson < 0:
                signals_strong.append(f"CORRÉLATION NÉGATIVE FORTE: UNIV-002 ↔ {test_name} (r={corr_pearson:.2f})")
            else:
                signals_strong.append(f"CORRÉLATION POSITIVE FORTE: UNIV-002 ↔ {test_name} (r={corr_pearson:.2f})")
    
    # Visualisation
    n_tests = len(correlations)
    if n_tests == 0:
        observations.append("\n⚠️ Aucune corrélation calculable")
        return AuditResult(
            section="5. Corrélations inter-tests",
            observations=observations,
            signals_strong=signals_strong,
            signals_weak=signals_weak,
            figures=figures
        )
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (test_name, corr_data) in enumerate(sorted(correlations.items())[:6]):
        # Récupérer données pour scatter
        pairs = []
        for d in data_list:
            if test_name in scores_matrix[d.exec_id]:
                pairs.append((d.score, scores_matrix[d.exec_id][test_name]))
        
        if pairs:
            scores_univ002, scores_test = zip(*pairs)
            
            axes[i].scatter(scores_univ002, scores_test, alpha=0.5)
            axes[i].set_xlabel('UNIV-002 score')
            axes[i].set_ylabel(f'{test_name} score')
            axes[i].set_title(f'{test_name}\nr={corr_data["pearson"]:.2f}, p={corr_data["p_pearson"]:.3f}')
            axes[i].grid(alpha=0.3)
            
            # Ligne de tendance
            z = np.polyfit(scores_univ002, scores_test, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(scores_univ002), max(scores_univ002), 100)
            axes[i].plot(x_line, p(x_line), "r--", alpha=0.5)
    
    # Cacher axes vides
    for i in range(len(correlations), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "audit5_correlations.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    figures.append(fig_path)
    
    return AuditResult(
        section="5. Corrélations inter-tests",
        observations=observations,
        signals_strong=signals_strong,
        signals_weak=signals_weak,
        figures=figures
    )


# =============================================================================
# AUDIT 6 : LOCAL VS GLOBAL
# =============================================================================

def audit_6_local_vs_global(conn_raw: sqlite3.Connection,
                            data_list: List[UNIV002Data]) -> AuditResult:
    """
    Audit 6 : Diversité locale vs globale.
    
    Questions:
    - UNIV-002 mesure-t-il variance globale ou diversité structurelle ?
    - Comparaison avec métriques locales (patches)
    
    Note: Nécessite snapshots, peut être coûteux.
    """
    observations = []
    signals_strong = []
    signals_weak = []
    figures = []
    
    observations.append("Audit local vs global nécessite snapshots complets.")
    observations.append("Échantillonnage aléatoire de 20 exécutions pour analyse.\n")
    
    # Échantillonner
    sample = np.random.choice(data_list, min(20, len(data_list)), replace=False)
    
    local_global_ratios = []
    
    for d in sample:
        # Charger snapshot final depuis db_raw
        cursor = conn_raw.cursor()
        cursor.execute("""
            SELECT state_blob
            FROM Snapshots
            WHERE exec_id = ?
            ORDER BY iteration DESC
            LIMIT 1
        """, (d.exec_id,))
        
        row = cursor.fetchone()
        if not row:
            continue
        
        # Décompresser état
        state_bytes = gzip.decompress(row[0])
        state = pickle.loads(state_bytes)
        
        if state.ndim != 2:
            continue  # Skip si pas rang 2
        
        # Calculer diversité globale
        diversity_global = np.std(state)
        
        # Calculer diversité locale (patches 5x5)
        patch_size = 5
        n_patches = min(10, (state.shape[0] // patch_size) * (state.shape[1] // patch_size))
        
        local_stds = []
        for _ in range(n_patches):
            i = np.random.randint(0, state.shape[0] - patch_size)
            j = np.random.randint(0, state.shape[1] - patch_size)
            patch = state[i:i+patch_size, j:j+patch_size]
            local_stds.append(np.std(patch))
        
        diversity_local = np.mean(local_stds)
        
        if diversity_global > 0:
            local_global_ratios.append(diversity_local / diversity_global)
    
    if local_global_ratios:
        observations.append(f"Nombre d'échantillons analysés: {len(local_global_ratios)}")
        observations.append(f"Ratio moyen local/global: {np.mean(local_global_ratios):.3f}")
        observations.append(f"Ratio médian local/global: {np.median(local_global_ratios):.3f}")
        observations.append(f"Écart-type ratios: {np.std(local_global_ratios):.3f}")
        
        if np.mean(local_global_ratios) < 0.5:
            signals_weak.append("Diversité locale significativement plus faible que globale")
        elif np.mean(local_global_ratios) > 1.5:
            signals_weak.append("Diversité locale plus élevée que globale (structure hétérogène)")
    else:
        observations.append("⚠️ Aucun échantillon analysable")
    
    # Visualisation
    if local_global_ratios:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        ax.hist(local_global_ratios, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(1.0, color='red', linestyle='--', label='Égalité local/global')
        ax.set_xlabel('Ratio diversité locale / globale')
        ax.set_ylabel('Fréquence')
        ax.set_title('Distribution diversité locale vs globale')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        fig_path = FIGURES_DIR / "audit6_local_global.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        figures.append(fig_path)
    
    return AuditResult(
        section="6. Local vs global",
        observations=observations,
        signals_strong=signals_strong,
        signals_weak=signals_weak,
        figures=figures
    )


# =============================================================================
# AUDIT 7 : INVARIANCE D'ÉCHELLE
# =============================================================================

def audit_7_scale_invariance(data_list: List[UNIV002Data]) -> AuditResult:
    """
    Audit 7 : Invariance d'échelle.
    
    Questions:
    - Effet taille D (nombre de DOF)
    - Effet type de base (SYM vs ASY vs R3)
    """
    observations = []
    signals_strong = []
    signals_weak = []
    figures = []
    
    # Grouper par type de base
    by_d_type = defaultdict(list)
    for d in data_list:
        if d.context.d_base_id.startswith('SYM'):
            by_d_type['SYM'].append(d)
        elif d.context.d_base_id.startswith('ASY'):
            by_d_type['ASY'].append(d)
        elif d.context.d_base_id.startswith('R3'):
            by_d_type['R3'].append(d)
    
    observations.append("Distribution par type de base:\n")
    
    for d_type, items in sorted(by_d_type.items()):
        scores = [item.score for item in items]
        ratios = [item.ratio for item in items]
        
        observations.append(f"\n{d_type} (n={len(items)}):")
        observations.append(f"  Score moyen: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        observations.append(f"  Ratio moyen: {np.mean(ratios):.3f} ± {np.std(ratios):.3f}")
    
    # Test ANOVA si ≥3 groupes
    if len(by_d_type) >= 2:
        from scipy.stats import f_oneway
        score_groups = [np.array([item.score for item in items]) 
                       for items in by_d_type.values()]
        
        if all(len(g) > 1 for g in score_groups):
            f_stat, p_value = f_oneway(*score_groups)
            observations.append(f"\nANOVA entre types de base:")
            observations.append(f"  F={f_stat:.3f}, p={p_value:.4f}")
            
            if p_value < 0.05:
                signals_strong.append(f"EFFET TYPE DE BASE significatif (ANOVA p={p_value:.4f})")
    
    # Analyser effet taille (si diversité dans DOF)
    # Note: Dans notre cas, la plupart sont n_dof=50 ou 20 (R3)
    # Grouper par n_dof estimé
    by_size = defaultdict(list)
    for d in data_list:
        # Heuristique: R3 → 20, autres → 50
        size = 20 if d.context.d_base_id.startswith('R3') else 50
        by_size[size].append(d)
    
    observations.append("\n\nDistribution par taille estimée:\n")
    
    for size, items in sorted(by_size.items()):
        scores = [item.score for item in items]
        observations.append(f"\nTaille ≈{size} DOF (n={len(items)}):")
        observations.append(f"  Score moyen: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Par type de base
    d_types = sorted(by_d_type.keys())
    scores_by_type = [[item.score for item in by_d_type[dt]] for dt in d_types]
    
    axes[0].boxplot(scores_by_type, labels=d_types)
    axes[0].set_ylabel('Score UNIV-002')
    axes[0].set_title('Distribution par type de base')
    axes[0].grid(alpha=0.3)
    
    # Par taille
    sizes = sorted(by_size.keys())
    scores_by_size = [[item.score for item in by_size[s]] for s in sizes]
    
    axes[1].boxplot(scores_by_size, labels=[f'{s} DOF' for s in sizes])
    axes[1].set_ylabel('Score UNIV-002')
    axes[1].set_title('Distribution par taille')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "audit7_scale.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    figures.append(fig_path)
    
    return AuditResult(
        section="7. Invariance d'échelle",
        observations=observations,
        signals_strong=signals_strong,
        signals_weak=signals_weak,
        figures=figures
    )


# =============================================================================
# GÉNÉRATION RAPPORT
# =============================================================================

def generate_report(audit_results: List[AuditResult], 
                   data_list: List[UNIV002Data]) -> Path:
    """
    Génère le rapport Markdown final.
    
    Returns:
        Path du rapport généré
    """
    report_lines = []
    
    # En-tête
    report_lines.append("# AUDIT UNIV-002 - Rapport Exhaustif")
    report_lines.append(f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Conformité**: Charte PRC 5.1 Section 14")
    report_lines.append(f"**Méthode**: Audit post-hoc (aucun rerun)")
    report_lines.append(f"\n**Données**: {len(data_list)} exécutions analysées")
    
    report_lines.append("\n---\n")
    
    # Résumé exécutif
    report_lines.append("## RÉSUMÉ EXÉCUTIF\n")
    report_lines.append("### Objectif\n")
    report_lines.append("Déterminer si l'échec systématique de UNIV-002 est:\n")
    report_lines.append("1. Un artefact de scoring/normalisation")
    report_lines.append("2. Un biais de métrique (variance globale vs diversité structurelle)")
    report_lines.append("3. Un effet de taille/base D")
    report_lines.append("4. Un pattern dynamique réel\n")
    
    # Collecter tous les signaux forts
    all_strong_signals = []
    for result in audit_results:
        all_strong_signals.extend(result.signals_strong)
    
    if all_strong_signals:
        report_lines.append("### Signaux forts détectés\n")
        for i, signal in enumerate(all_strong_signals, 1):
            report_lines.append(f"{i}. **{signal}**")
        report_lines.append("\n")
    
    # Sections d'audit
    report_lines.append("\n---\n")
    
    for result in audit_results:
        report_lines.append(f"## {result.section.upper()}\n")
        
        if result.observations:
            report_lines.append("### Observations\n")
            for obs in result.observations:
                report_lines.append(obs)
            report_lines.append("\n")
        
        if result.signals_strong:
            report_lines.append("### ⚠️ Signaux forts\n")
            for signal in result.signals_strong:
                report_lines.append(f"- **{signal}**")
            report_lines.append("\n")
        
        if result.signals_weak:
            report_lines.append("### ℹ️ Signaux faibles\n")
            for signal in result.signals_weak:
                report_lines.append(f"- {signal}")
            report_lines.append("\n")
        
        if result.figures:
            report_lines.append("### Figures\n")
            for fig_path in result.figures:
                report_lines.append(f"![{fig_path.stem}]({fig_path})\n")
        
        report_lines.append("\n---\n")
    
    # Tableau de décision
    report_lines.append("## TABLEAU DE DÉCISION\n")
    report_lines.append("| Critère | Statut | Justification |")
    report_lines.append("|---------|--------|---------------|")
    
    # Analyse des signaux pour remplir le tableau
    decisions = {
        'valide_tel_quel': False,
        'valide_mal_score': False,
        'mesure_autre_chose': False,
        'non_concluant': False
    }
    
    justifications = {
        'valide_tel_quel': [],
        'valide_mal_score': [],
        'mesure_autre_chose': [],
        'non_concluant': []
    }
    
    # Analyser signaux forts pour décisions
    for result in audit_results:
        if "SEUIL INFÉRIEUR SATURÉ" in str(result.signals_strong):
            decisions['valide_mal_score'] = True
            justifications['valide_mal_score'].append("Seuils de scoring inadaptés")
        
        if "CORRÉLATION NÉGATIVE FORTE" in str(result.signals_strong):
            decisions['mesure_autre_chose'] = True
            justifications['mesure_autre_chose'].append("Anti-corrélation avec stabilité Lyapunov")
        
        if "COLLAPSE RAPIDE dominant" in str(result.signals_strong):
            decisions['mesure_autre_chose'] = True
            justifications['mesure_autre_chose'].append("Mesure collapse, pas diversité structurelle")
        
        if "DÉCROISSANCE MASSIVE" in str(result.signals_strong):
            decisions['mesure_autre_chose'] = True
            justifications['mesure_autre_chose'].append("Pattern systématique de perte")
    
    # Remplir tableau
    if decisions['valide_tel_quel']:
        report_lines.append("| UNIV-002 valide tel quel | ✅ | " + "; ".join(justifications['valide_tel_quel']) + " |")
    else:
        report_lines.append("| UNIV-002 valide tel quel | ❌ | Signaux contradictoires détectés |")
    
    if decisions['valide_mal_score']:
        report_lines.append("| UNIV-002 valide mais mal scoré | ⚠️ | " + "; ".join(justifications['valide_mal_score']) + " |")
    else:
        report_lines.append("| UNIV-002 valide mais mal scoré | ➖ | Scoring semble cohérent |")
    
    if decisions['mesure_autre_chose']:
        report_lines.append("| UNIV-002 mesure autre chose | ⚠️ | " + "; ".join(justifications['mesure_autre_chose']) + " |")
    else:
        report_lines.append("| UNIV-002 mesure autre chose | ➖ | Cohérence avec attentes |")
    
    if decisions['non_concluant']:
        report_lines.append("| UNIV-002 non concluant à R0 | ⚠️ | " + "; ".join(justifications['non_concluant']) + " |")
    else:
        report_lines.append("| UNIV-002 non concluant à R0 | ➖ | Données suffisamment informatives |")
    
    report_lines.append("\n")
    
    # Recommandations méthodologiques
    report_lines.append("## RECOMMANDATIONS MÉTHODOLOGIQUES\n")
    report_lines.append("### Actions immédiates\n")
    
    if decisions['valide_mal_score']:
        report_lines.append("1. **Réviser fonction de scoring UNIV-002**")
        report_lines.append("   - Ajuster seuils [0.3, 3.0] selon distributions observées")
        report_lines.append("   - Tester scoring linéaire vs sigmoïde")
        report_lines.append("   - Relancer `--verdict` avec nouveau scoring\n")
    
    if decisions['mesure_autre_chose']:
        report_lines.append("2. **Clarifier définition UNIV-002**")
        report_lines.append("   - Documenter explicitement: diversité globale vs structurelle")
        report_lines.append("   - Envisager test complémentaire pour diversité locale")
        report_lines.append("   - Dissocier 'variance' de 'diversité structurelle'\n")
    
    report_lines.append("3. **Ne PAS ériger en contrainte L2**")
    report_lines.append("   - Pattern observé dépend de l'instrumentation actuelle")
    report_lines.append("   - Journaliser comme OBS-GAM-001 (L4), pas CON-GAM-001 (L2)")
    report_lines.append("   - Validation nécessite tests R1 pour départager hypothèses\n")
    
    report_lines.append("### Exploration R1\n")
    report_lines.append("- Tester compositions Γ pour vérifier si UNIV-002 reste pertinent")
    report_lines.append("- Comparer UNIV-002 avec métriques alternatives (entropie, structure locale)")
    report_lines.append("- Analyser si compositions R1 dépassent limites observées R0\n")
    
    # Interdictions explicites
    report_lines.append("## INTERDICTIONS MÉTHODOLOGIQUES\n")
    report_lines.append("❌ **NE PAS conclure** que \"les mécanismes isolés échouent par nature\"")
    report_lines.append("   - Pattern observé dépend de l'opérateur UNIV-002 actuel")
    report_lines.append("   - Généralisation hors R0 non validée\n")
    
    report_lines.append("❌ **NE PAS proposer** de clôture R0")
    report_lines.append("   - Instrumentation instable (tests plats, corrélations suspectes)")
    report_lines.append("   - R0 'cohérent mais partiel', pas 'exhaustif'\n")
    
    report_lines.append("❌ **NE PAS ériger** observation en contrainte")
    report_lines.append("   - Passage L3→L2 nécessite validation instrumentale")
    report_lines.append("   - Discussion JOUR obligatoire avant CON-GAM-XXX\n")
    
    # Métadonnées
    report_lines.append("\n---\n")
    report_lines.append("## MÉTADONNÉES\n")
    report_lines.append(f"- **Script**: audit_univ_002.py")
    report_lines.append(f"- **Base db_raw**: {DB_RAW_PATH}")
    report_lines.append(f"- **Base db_results**: {DB_RESULTS_PATH}")
    report_lines.append(f"- **Figures**: {FIGURES_DIR}/")
    report_lines.append(f"- **Conformité Charte**: Section 14 (rejouabilité sans reruns)")
    
    # Écrire rapport
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    return REPORT_PATH


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Point d'entrée principal."""
    print("="*70)
    print("AUDIT UNIV-002 - Analyse exhaustive post-hoc")
    print("="*70)
    print()
    
    # 1. Connexion bases de données
    print("1. Connexion aux bases de données...")
    try:
        conn_raw, conn_results = connect_databases()
        print("   ✓ Connexions établies\n")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        sys.exit(1)
    
    # 2. Extraction données
    print("2. Extraction données UNIV-002...")
    try:
        data_list = extract_univ002_data(conn_raw, conn_results)
        print(f"   ✓ {len(data_list)} exécutions extraites\n")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        conn_raw.close()
        conn_results.close()
        sys.exit(1)
    
    if len(data_list) == 0:
        print("   ⚠️ Aucune donnée UNIV-002 trouvée")
        conn_raw.close()
        conn_results.close()
        sys.exit(1)
    
    # 3. Exécuter audits
    audit_results = []
    
    print("3. Exécution des audits...\n")
    
    print("   [1/7] Distributions brutes...")
    try:
        result1 = audit_1_distributions(data_list)
        audit_results.append(result1)
        print(f"        ✓ {len(result1.signals_strong)} signaux forts")
    except Exception as e:
        print(f"        ❌ Erreur: {e}")
    
    print("   [2/7] Analyse temporelle...")
    try:
        result2 = audit_2_temporal(data_list)
        audit_results.append(result2)
        print(f"        ✓ {len(result2.signals_strong)} signaux forts")
    except Exception as e:
        print(f"        ❌ Erreur: {e}")
    
    print("   [3/7] Validation scoring...")
    try:
        result3 = audit_3_scoring(data_list)
        audit_results.append(result3)
        print(f"        ✓ {len(result3.signals_strong)} signaux forts")
    except Exception as e:
        print(f"        ❌ Erreur: {e}")
    
    print("   [4/7] Sensibilité paramètres...")
    try:
        result4 = audit_4_sensitivity(data_list)
        audit_results.append(result4)
        print(f"        ✓ {len(result4.signals_strong)} signaux forts")
    except Exception as e:
        print(f"        ❌ Erreur: {e}")
    
    print("   [5/7] Corrélations inter-tests...")
    try:
        result5 = audit_5_correlations(conn_results, data_list)
        audit_results.append(result5)
        print(f"        ✓ {len(result5.signals_strong)} signaux forts")
    except Exception as e:
        print(f"        ❌ Erreur: {e}")
    
    print("   [6/7] Local vs global...")
    try:
        result6 = audit_6_local_vs_global(conn_raw, data_list)
        audit_results.append(result6)
        print(f"        ✓ {len(result6.signals_strong)} signaux forts")
    except Exception as e:
        print(f"        ❌ Erreur: {e}")
    
    print("   [7/7] Invariance d'échelle...")
    try:
        result7 = audit_7_scale_invariance(data_list)
        audit_results.append(result7)
        print(f"        ✓ {len(result7.signals_strong)} signaux forts")
    except Exception as e:
        print(f"        ❌ Erreur: {e}")
    
    print()
    
    # 4. Génération rapport
    print("4. Génération rapport final...")
    try:
        report_path = generate_report(audit_results, data_list)
        print(f"   ✓ Rapport généré: {report_path}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    # 5. Fermeture connexions
    conn_raw.close()
    cursor_results = conn_results.cursor()
    cursor_results.execute("DETACH DATABASE db_raw")
    conn_results.close()
    
    print()
    print("="*70)
    print("AUDIT TERMINÉ")
    print("="*70)
    print(f"\nRapport: {report_path}")
    print(f"Figures: {FIGURES_DIR}/")
    print()


if __name__ == "__main__":
    main()
