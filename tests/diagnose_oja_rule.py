"""
diagnose_oja_rule.py

Diagnostic approfondi de OjaRule selon le cadre PRC.

Questions à répondre:
1. Le cycle période-2 est-il robuste et structuré?
2. Le mode dominant est-il stable, non-uniforme, structurant?
3. La dynamique dépend-elle de C₀?
4. La compression rang-1 encode-t-elle de la structure?

ÉVITE STRICTEMENT:
- Jacobien naïf sur espace plat
- Interprétation "rang faible = dégénérescence"
- Confusion uniformité/trivialité
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core import InformationSpace, PRCKernel
from operators.hebbian import OjaRuleOperator


# ============================================================================
# CONFIGURATION
# ============================================================================

N_DOF = 500
EPSILON = 0.05
N_ITERATIONS = 1000
BETA_OJA = 0.01
SEED = 42


# ============================================================================
# ANALYSE 1: CYCLE PÉRIODE-2 (SIGNAL MAJEUR)
# ============================================================================

def analyze_cycle_structure(C_history: list, period: int = 2):
    """
    Analyse détaillée du cycle période-2.
    
    Questions:
    - Le cycle est-il parfaitement stable?
    - Les deux états sont-ils structurellement distincts?
    - La transition encode-t-elle une information?
    """
    print("\n" + "="*70)
    print("ANALYSE 1: CYCLE PÉRIODE-2")
    print("="*70)
    
    if len(C_history) < 100:
        print("⚠️  Historique trop court")
        return
    
    # Prend derniers 50 états (stabilisés)
    stable_window = C_history[-50:]
    
    # Vérifie périodicité stricte
    diffs_period = []
    for i in range(len(stable_window) - period):
        diff = np.linalg.norm(stable_window[i] - stable_window[i + period])
        diffs_period.append(diff)
    
    mean_period_error = np.mean(diffs_period)
    
    print(f"\n📊 Stabilité du cycle:")
    print(f"  Erreur période-{period}: {mean_period_error:.3e}")
    
    if mean_period_error < 1e-6:
        print(f"  ✅ Cycle parfaitement stable")
    elif mean_period_error < 1e-3:
        print(f"  ✅ Cycle robuste")
    else:
        print(f"  ⚠️  Cycle imparfait")
    
    # Analyse les deux états du cycle
    C_even = stable_window[-2]  # État pair
    C_odd = stable_window[-1]   # État impair
    
    # Différence structurelle
    diff_structure = np.linalg.norm(C_even - C_odd)
    print(f"\n📊 Structure du cycle:")
    print(f"  Distance entre états: {diff_structure:.6f}")
    
    # Analyse spectrale des deux états
    eig_even = np.linalg.eigvalsh(C_even)
    eig_odd = np.linalg.eigvalsh(C_odd)
    
    print(f"  Spectre état pair:  λ₁={eig_even[-1]:.3f}, λₙ={eig_even[0]:.3f}")
    print(f"  Spectre état impair: λ₁={eig_odd[-1]:.3f}, λₙ={eig_odd[0]:.3f}")
    
    # Différence spectrale
    spectral_diff = np.linalg.norm(eig_even - eig_odd)
    print(f"  Différence spectrale: {spectral_diff:.6f}")
    
    if spectral_diff > 1e-3:
        print(f"  ✅ Deux états spectralement distincts")
        return "structured_cycle"
    else:
        print(f"  ⚠️  États spectralement similaires")
        return "degenerate_cycle"


# ============================================================================
# ANALYSE 2: MODE DOMINANT (VECTEUR PROPRE PRINCIPAL)
# ============================================================================

def analyze_dominant_mode(C_history: list):
    """
    Analyse le vecteur propre dominant au fil du temps.
    
    Questions:
    - Est-il stable?
    - Est-il uniforme (tous composantes égales) ou structuré?
    - Comment évolue-t-il?
    """
    print("\n" + "="*70)
    print("ANALYSE 2: MODE DOMINANT")
    print("="*70)
    
    # Extrait vecteur propre dominant à différents moments
    times = [0, len(C_history)//4, len(C_history)//2, 3*len(C_history)//4, -1]
    
    dominant_vectors = []
    dominant_values = []
    
    for t in times:
        C = C_history[t]
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        
        # Vecteur propre de la plus grande valeur propre
        idx_max = np.argmax(eigenvalues)
        v_dom = eigenvectors[:, idx_max]
        lambda_dom = eigenvalues[idx_max]
        
        # Normalise signe pour comparaison
        if v_dom[0] < 0:
            v_dom = -v_dom
        
        dominant_vectors.append(v_dom)
        dominant_values.append(lambda_dom)
    
    print(f"\n📊 Évolution valeur propre dominante:")
    for i, (t, val) in enumerate(zip(times, dominant_values)):
        iter_label = "initial" if t == 0 else f"iter {C_history[t] if t > 0 else len(C_history)+t}"
        print(f"  {iter_label:15}: λ₁ = {val:.6f}")
    
    # Stabilité du vecteur propre
    v_final = dominant_vectors[-1]
    v_prev = dominant_vectors[-2]
    
    alignment = np.abs(np.dot(v_final, v_prev))
    print(f"\n📊 Stabilité du vecteur propre:")
    print(f"  Alignement t-1→t: {alignment:.6f}")
    
    if alignment > 0.9999:
        print(f"  ✅ Vecteur propre stable")
    
    # Structure du vecteur propre final
    print(f"\n📊 Structure du vecteur propre dominant:")
    print(f"  Composantes (5 premières): {v_final[:5]}")
    print(f"  Écart-type: {np.std(v_final):.6f}")
    print(f"  Min/Max: {v_final.min():.6f} / {v_final.max():.6f}")
    
    # Test d'uniformité
    uniform_vector = np.ones(len(v_final)) / np.sqrt(len(v_final))
    uniformity = np.abs(np.dot(v_final, uniform_vector))
    
    print(f"  Proximité au vecteur uniforme: {uniformity:.6f}")
    
    if uniformity > 0.99:
        print(f"  ⚠️  Vecteur quasi-uniforme")
        is_structured = False
    else:
        print(f"  ✅ Vecteur structuré (non-uniforme)")
        is_structured = True
    
    return is_structured, v_final


# ============================================================================
# ANALYSE 3: DÉPENDANCE À C₀
# ============================================================================

def test_initial_dependence():
    """
    Teste si la dynamique dépend réellement de C₀.
    
    Simule avec plusieurs C₀ différents et compare les états finaux.
    """
    print("\n" + "="*70)
    print("ANALYSE 3: DÉPENDANCE À C₀")
    print("="*70)
    
    gamma = OjaRuleOperator(beta=BETA_OJA)
    
    # Teste 5 conditions initiales différentes
    initial_states = []
    final_states = []
    
    for seed in [42, 123, 456, 789, 1011]:
        # Crée C₀ différent
        np.random.seed(seed)
        R = np.random.randn(N_DOF, N_DOF)
        R = (R + R.T) / 2
        np.fill_diagonal(R, 0)
        R = R / np.max(np.abs(R))
        C0 = np.eye(N_DOF) + EPSILON * R
        C0 = np.clip(C0, -1, 1)
        np.fill_diagonal(C0, 1)
        
        D0 = InformationSpace(C0, {"seed": seed})
        initial_states.append(C0)
        
        # Simule
        kernel = PRCKernel(D0, gamma)
        kernel.step(N_ITERATIONS)
        final_states.append(kernel.C_current)
    
    # Compare états finaux
    print(f"\n📊 Variabilité des états finaux:")
    
    diversities = []
    for i in range(len(final_states)):
        for j in range(i+1, len(final_states)):
            diff = np.linalg.norm(final_states[i] - final_states[j])
            diversities.append(diff)
    
    mean_diversity = np.mean(diversities)
    
    print(f"  Distance moyenne entre états finaux: {mean_diversity:.6f}")
    
    if mean_diversity < 1e-3:
        print(f"  ⚠️  États finaux quasi-identiques (attracteur unique)")
        dependence = "weak"
    elif mean_diversity < 0.1:
        print(f"  ✅ Variabilité modérée (dépendance à C₀)")
        dependence = "moderate"
    else:
        print(f"  ✅ Forte variabilité (forte dépendance à C₀)")
        dependence = "strong"
    
    # Compare spectres
    print(f"\n📊 Variabilité spectrale:")
    spectra = [np.linalg.eigvalsh(C) for C in final_states]
    
    # Valeur propre dominante
    lambda_max = [s[-1] for s in spectra]
    print(f"  λ₁ range: [{min(lambda_max):.4f}, {max(lambda_max):.4f}]")
    print(f"  λ₁ std: {np.std(lambda_max):.6f}")
    
    return dependence


# ============================================================================
# ANALYSE 4: COMPRESSION INFORMATIONNELLE
# ============================================================================

def analyze_information_compression(C_history: list):
    """
    Analyse comment l'information est compressée au fil du temps.
    
    Métriques:
    - Rang effectif (participation ratio)
    - Entropie spectrale
    - Distribution des valeurs propres
    """
    print("\n" + "="*70)
    print("ANALYSE 4: COMPRESSION INFORMATIONNELLE")
    print("="*70)
    
    # Analyse à différents moments
    times = [0, len(C_history)//4, len(C_history)//2, -1]
    
    print(f"\n📊 Évolution de la compression:")
    
    for t in times:
        C = C_history[t]
        eigenvalues = np.linalg.eigvalsh(C)
        
        # Normalise (positif)
        eig_pos = eigenvalues - eigenvalues.min() + 1e-10
        eig_norm = eig_pos / eig_pos.sum()
        
        # Participation ratio (rang effectif)
        participation = 1 / np.sum(eig_norm**2)
        
        # Entropie spectrale
        entropy = -np.sum(eig_norm * np.log(eig_norm + 1e-10))
        
        # Spread spectral
        spectral_spread = eigenvalues[-1] - eigenvalues[0]
        
        iter_label = "initial" if t == 0 else f"final" if t == -1 else f"t={t}"
        print(f"  {iter_label:15}:")
        print(f"    Rang effectif:     {participation:.2f} / {N_DOF}")
        print(f"    Entropie spectrale: {entropy:.4f}")
        print(f"    Spread spectral:    {spectral_spread:.4f}")
    
    # Analyse finale détaillée
    C_final = C_history[-1]
    eig_final = np.linalg.eigvalsh(C_final)
    
    print(f"\n📊 Distribution spectrale finale:")
    print(f"  5 plus grandes: {eig_final[-5:]}")
    print(f"  5 plus petites: {eig_final[:5]}")
    
    # Gap spectral
    gap = eig_final[-1] - eig_final[-2]
    print(f"  Gap λ₁-λ₂: {gap:.6f}")
    
    if gap > 0.1 * eig_final[-1]:
        print(f"  ✅ Gap significatif (mode dominant isolé)")


# ============================================================================
# ANALYSE 5: VISUALISATION
# ============================================================================

def visualize_dynamics(C_history: list):
    """
    Crée visualisations de l'évolution de C.
    """
    print("\n" + "="*70)
    print("ANALYSE 5: VISUALISATIONS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Matrices à différents temps
    times = [0, len(C_history)//4, len(C_history)//2, -10, -5, -1]
    
    for idx, t in enumerate(times[:6]):
        ax = axes[idx // 3, idx % 3]
        C = C_history[t]
        
        im = ax.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f"Iteration {t if t >= 0 else len(C_history)+t}")
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('oja_matrices.png', dpi=150, bbox_inches='tight')
    print(f"  ✅ Matrices sauvegardées: oja_matrices.png")
    
    # 2. Évolution spectrale
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Valeur propre dominante
    ax = axes[0, 0]
    lambda_max = [np.max(np.linalg.eigvalsh(C)) for C in C_history]
    ax.plot(lambda_max)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('λ_max')
    ax.set_title('Valeur propre dominante')
    ax.grid(True, alpha=0.3)
    
    # Rang effectif
    ax = axes[0, 1]
    ranks = []
    for C in C_history:
        eig = np.linalg.eigvalsh(C)
        eig_pos = eig - eig.min() + 1e-10
        eig_norm = eig_pos / eig_pos.sum()
        rank = 1 / np.sum(eig_norm**2)
        ranks.append(rank)
    ax.plot(ranks)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Rang effectif')
    ax.set_title('Compression informationnelle')
    ax.grid(True, alpha=0.3)
    
    # Corrélation moyenne
    ax = axes[1, 0]
    mean_corrs = []
    for C in C_history:
        mask = ~np.eye(C.shape[0], dtype=bool)
        mean_corrs.append(np.mean(np.abs(C[mask])))
    ax.plot(mean_corrs)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Corrélation moyenne')
    ax.set_title('Intensité corrélations')
    ax.grid(True, alpha=0.3)
    
    # Différences successives (cycle?)
    ax = axes[1, 1]
    if len(C_history) > 100:
        recent = C_history[-100:]
        diffs = []
        for i in range(len(recent)-1):
            diff = np.linalg.norm(recent[i+1] - recent[i])
            diffs.append(diff)
        ax.plot(diffs)
        ax.set_xlabel('Iteration (dernières 100)')
        ax.set_ylabel('||C_n+1 - C_n||')
        ax.set_title('Changements successifs')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('oja_dynamics.png', dpi=150, bbox_inches='tight')
    print(f"  ✅ Dynamiques sauvegardées: oja_dynamics.png")
    
    plt.close('all')


# ============================================================================
# DIAGNOSTIC COMPLET
# ============================================================================

def run_full_diagnosis():
    """
    Exécute toutes les analyses sur OjaRule.
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC APPROFONDI: OjaRule")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  N_DOF: {N_DOF}")
    print(f"  ε: {EPSILON}")
    print(f"  β: {BETA_OJA}")
    print(f"  Iterations: {N_ITERATIONS}")
    
    # Simule
    np.random.seed(SEED)
    R = np.random.randn(N_DOF, N_DOF)
    R = (R + R.T) / 2
    np.fill_diagonal(R, 0)
    R = R / np.max(np.abs(R))
    C0 = np.eye(N_DOF) + EPSILON * R
    C0 = np.clip(C0, -1, 1)
    np.fill_diagonal(C0, 1)
    
    D0 = InformationSpace(C0, {"test": "oja_diagnosis"})
    gamma = OjaRuleOperator(beta=BETA_OJA)
    
    kernel = PRCKernel(D0, gamma)
    kernel.start_recording()
    kernel.step(N_ITERATIONS)
    
    history = kernel.get_history()
    C_history = [state.C for state in history]
    
    # Analyses
    cycle_type = analyze_cycle_structure(C_history, period=2)
    is_structured, v_dom = analyze_dominant_mode(C_history)
    dependence = test_initial_dependence()
    analyze_information_compression(C_history)
    visualize_dynamics(C_history)
    
    # Verdict final
    print("\n" + "="*70)
    print("VERDICT FINAL")
    print("="*70)
    
    scores = {
        "cycle_structure": cycle_type == "structured_cycle",
        "mode_structure": is_structured,
        "c0_dependence": dependence in ["moderate", "strong"],
    }
    
    total = sum(scores.values())
    
    print(f"\nScore: {total}/3")
    print(f"\n  Cycle structuré:    {'✅' if scores['cycle_structure'] else '⚠️ '}")
    print(f"  Mode structuré:     {'✅' if scores['mode_structure'] else '⚠️ '}")
    print(f"  Dépend de C₀:       {'✅' if scores['c0_dependence'] else '⚠️ '}")
    
    if total == 3:
        print(f"\n🎯 EXCELLENT: OjaRule encode structure riche")
    elif total == 2:
        print(f"\n✅ BON: OjaRule montre propriétés intéressantes")
    else:
        print(f"\n⚠️  MITIGÉ: Compression possiblement excessive")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    run_full_diagnosis()