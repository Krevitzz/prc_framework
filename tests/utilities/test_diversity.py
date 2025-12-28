"""
tests/utilities/test_diversity.py

Tests modulaires observant l'évolution de la diversité structurale.

PRINCIPE: OBSERVATION PURE.

Ces tests détectent:
- TEST-UNIV-002: Évolution diversité (écart-type)
- Homogénéisation (perte de diversité)
- Différenciation (augmentation diversité)
"""

import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class DiversityResult:
    """Résultat d'un test de diversité."""
    test_name: str
    initial_diversity: float
    final_diversity: float
    max_diversity: float 
    min_diversity: float 
    ratio_final_initial: float
    evolution: str  # "increasing", "decreasing", "stable", "collapsed"
    status: str
    message: str
    blocking: bool = False
    
    def __repr__(self):
        symbol = "✓" if self.status == "PASS" else "✗" if self.status == "FAIL" else "○"
        return f"{symbol} {self.test_name}: {self.message}"


# ============================================================================
# TEST-UNIV-002: Évolution diversité (std)
# ============================================================================

def test_diversity_preservation(history: List[np.ndarray],
                               collapse_threshold: float = 0.1,
                               explosion_threshold: float = 10.0,
                               name: str = "TEST-UNIV-002") -> DiversityResult:
    """
    TEST-UNIV-002: Observe l'évolution de la diversité.
    
    MESURE: σ(D_flat) à chaque itération (écart-type de tous les éléments)
    
    OUTPUT:
    - trend: évolution de la diversité
    - ratio_final/initial
    
    VERDICT:
    - PASS si ratio ∈ [0.3, 3.0] (diversité préservée)
    - FAIL si ratio < collapse_threshold (homogénéisation) OU > explosion_threshold (explosion)
    - NEUTRAL sinon
    
    Args:
        history: Liste d'états
        collapse_threshold: Ratio minimal acceptable (0.1 = 90% perte)
        explosion_threshold: Ratio maximal acceptable (10.0 = 10× augmentation)
        name: Nom du test
    
    Returns:
        DiversityResult avec status
    """
    if not history:
        return DiversityResult(
            test_name=name,
            initial_diversity=0.0,
            final_diversity=0.0,
            max_diversity=0.0,
            min_diversity=0.0,
            ratio_final_initial=0.0,
            evolution="unknown",
            status="FAIL",
            message="Historique vide",
            blocking=True
        )
    
    # Calcule diversité (écart-type) pour chaque état
    diversities = []
    for state in history:
        flat = state.flatten()
        diversity = np.std(flat)
        diversities.append(diversity)
    
    # Statistiques
    initial_div = diversities[0]
    final_div = diversities[-1]
    max_div = max(diversities)
    min_div = min(diversities)
    
    # Ratio (avec protection division par zéro)
    if initial_div < 1e-10:
        if final_div < 1e-10:
            ratio = 1.0
        else:
            ratio = np.inf
    else:
        ratio = final_div / initial_div
    
    # Détecte tendance
    if len(diversities) > 1:
        trend_coef = np.polyfit(range(len(diversities)), diversities, 1)[0]
        relative_change = abs(final_div - initial_div) / (initial_div + 1e-10)
        
        if relative_change < 0.1:
            evolution = "stable"
        elif trend_coef > 0:
            evolution = "increasing"
        elif trend_coef < 0:
            evolution = "decreasing"
        else:
            evolution = "stable"
    else:
        evolution = "stable"
        trend_coef = 0.0
    
    # Détecte effondrement
    if ratio < collapse_threshold:
        evolution = "collapsed"
    
    # Verdict
    if 0.3 <= ratio <= 3.0:
        status = "PASS"
        message = f"Diversité préservée: ratio={ratio:.2f}"
        blocking = False
    elif ratio < collapse_threshold:
        status = "FAIL"
        message = f"Homogénéisation: {initial_div:.3e} → {final_div:.3e} (ratio={ratio:.3f})"
        blocking = True
    elif ratio > explosion_threshold:
        status = "FAIL"
        message = f"Explosion diversité: {initial_div:.3e} → {final_div:.3e} (ratio={ratio:.1f})"
        blocking = True
    else:
        status = "NEUTRAL"
        message = f"Évolution diversité: {evolution}, ratio={ratio:.2f}"
        blocking = False
    
    return DiversityResult(
        test_name=name,
        initial_diversity=initial_div,
        final_diversity=final_div,
        max_diversity=max_div,
        min_diversity=min_div,
        ratio_final_initial=ratio,
        evolution=evolution,
        status=status,
        message=message,
        blocking=blocking
    )


# ============================================================================
# TESTS COMPLÉMENTAIRES
# ============================================================================

def test_entropy_evolution(history: List[np.ndarray],
                          name: str = "TEST-DIV-ENTROPY") -> DiversityResult:
    """
    Mesure l'entropie de la distribution des valeurs.
    
    Complément à la diversité (std) : détecte structures émergentes.
    
    MESURE: Entropie de Shannon de l'histogramme des valeurs
    """
    if not history:
        return DiversityResult(
            test_name=name,
            initial_diversity=0.0,
            final_diversity=0.0,
            max_diversity=0.0,
            min_diversity=0.0,
            ratio_final_initial=0.0,
            evolution="unknown",
            status="FAIL",
            message="Historique vide",
            blocking=False
        )
    
    def compute_entropy(state, n_bins=50):
        """Calcule l'entropie de Shannon."""
        flat = state.flatten()
        # Histogramme normalisé
        hist, _ = np.histogram(flat, bins=n_bins, density=True)
        # Normaliser pour avoir une distribution de probabilité
        hist = hist / (np.sum(hist) + 1e-10)
        # Entropie (éviter log(0))
        hist_nonzero = hist[hist > 0]
        entropy = -np.sum(hist_nonzero * np.log(hist_nonzero))
        return entropy
    
    entropies = [compute_entropy(state) for state in history]
    
    initial_ent = entropies[0]
    final_ent = entropies[-1]
    max_ent = max(entropies)
    min_ent = min(entropies)
    
    ratio = final_ent / (initial_ent + 1e-10)
    
    if ratio < 0.5:
        evolution = "decreasing"
        status = "NEUTRAL"
        message = f"Entropie réduite: {initial_ent:.2f} → {final_ent:.2f}"
    elif ratio > 2.0:
        evolution = "increasing"
        status = "NEUTRAL"
        message = f"Entropie augmentée: {initial_ent:.2f} → {final_ent:.2f}"
    else:
        evolution = "stable"
        status = "PASS"
        message = f"Entropie stable: {final_ent:.2f}"
    
    return DiversityResult(
        test_name=name,
        initial_diversity=initial_ent,
        final_diversity=final_ent,
        max_diversity=max_ent,
        min_diversity=min_ent,
        ratio_final_initial=ratio,
        evolution=evolution,
        status=status,
        message=message,
        blocking=False
    )


def test_range_evolution(history: List[np.ndarray],
                        name: str = "TEST-DIV-RANGE") -> DiversityResult:
    """
    Mesure l'évolution de la plage dynamique (max - min).
    
    Complément à std : détecte compression/expansion de la distribution.
    """
    if not history:
        return DiversityResult(
            test_name=name,
            initial_diversity=0.0,
            final_diversity=0.0,
            max_diversity=0.0,
            min_diversity=0.0,
            ratio_final_initial=0.0,
            evolution="unknown",
            status="FAIL",
            message="Historique vide",
            blocking=False
        )
    
    ranges = []
    for state in history:
        range_val = np.max(state) - np.min(state)
        ranges.append(range_val)
    
    initial_range = ranges[0]
    final_range = ranges[-1]
    max_range = max(ranges)
    min_range = min(ranges)
    
    ratio = final_range / (initial_range + 1e-10)
    
    if ratio < 0.3:
        evolution = "compressed"
        status = "NEUTRAL"
        message = f"Plage compressée: {initial_range:.3f} → {final_range:.3f}"
    elif ratio > 3.0:
        evolution = "expanded"
        status = "NEUTRAL"
        message = f"Plage élargie: {initial_range:.3f} → {final_range:.3f}"
    else:
        evolution = "stable"
        status = "PASS"
        message = f"Plage stable: {final_range:.3f}"
    
    return DiversityResult(
        test_name=name,
        initial_diversity=initial_range,
        final_diversity=final_range,
        max_diversity=max_range,
        min_diversity=min_range,
        ratio_final_initial=ratio,
        evolution=evolution,
        status=status,
        message=message,
        blocking=False
    )


def test_uniformity(history: List[np.ndarray],
                   name: str = "TEST-DIV-UNIFORM") -> DiversityResult:
    """
    Détecte convergence vers valeur uniforme.
    
    MESURE: Coefficient de variation CV = std / |mean|
    
    CV → 0 indique uniformisation
    """
    if not history:
        return DiversityResult(
            test_name=name,
            initial_diversity=0.0,
            final_diversity=0.0,
            max_diversity=0.0,
            min_diversity=0.0,
            ratio_final_initial=0.0,
            evolution="unknown",
            status="FAIL",
            message="Historique vide",
            blocking=False
        )
    
    def compute_cv(state):
        """Coefficient de variation."""
        flat = state.flatten()
        mean = np.mean(flat)
        std = np.std(flat)
        return std / (abs(mean) + 1e-10)
    
    cvs = [compute_cv(state) for state in history]
    
    initial_cv = cvs[0]
    final_cv = cvs[-1]
    max_cv = max(cvs)
    min_cv = min(cvs)
    
    ratio = final_cv / (initial_cv + 1e-10)
    
    # Détecte uniformisation
    if final_cv < 0.01:
        evolution = "uniform"
        status = "FAIL"
        message = f"Uniformisation détectée: CV={final_cv:.3e}"
        blocking = True
    elif final_cv < 0.1:
        evolution = "nearly_uniform"
        status = "NEUTRAL"
        message = f"Quasi-uniforme: CV={final_cv:.3f}"
        blocking = False
    else:
        if ratio < 0.5:
            evolution = "decreasing"
        elif ratio > 2.0:
            evolution = "increasing"
        else:
            evolution = "stable"
        
        status = "PASS"
        message = f"Diversité maintenue: CV={final_cv:.3f}"
        blocking = False
    
    return DiversityResult(
        test_name=name,
        initial_diversity=initial_cv,
        final_diversity=final_cv,
        max_diversity=max_cv,
        min_diversity=min_cv,
        ratio_final_initial=ratio,
        evolution=evolution,
        status=status,
        message=message,
        blocking=blocking
    )


def test_distinct_values(history: List[np.ndarray],
                        tolerance: float = 1e-6,
                        name: str = "TEST-DIV-DISTINCT") -> DiversityResult:
    """
    Compte le nombre de valeurs distinctes (avec tolérance).
    
    Détecte collapsus vers ensemble fini de valeurs.
    """
    if not history:
        return DiversityResult(
            test_name=name,
            initial_diversity=0.0,
            final_diversity=0.0,
            max_diversity=0.0,
            min_diversity=0.0,
            ratio_final_initial=0.0,
            evolution="unknown",
            status="FAIL",
            message="Historique vide",
            blocking=False
        )
    
    def count_distinct(state, tol):
        """Compte valeurs distinctes avec tolérance."""
        flat = np.sort(state.flatten())
        diffs = np.diff(flat)
        n_distinct = 1 + np.sum(diffs > tol)
        return n_distinct
    
    distinct_counts = [count_distinct(state, tolerance) for state in history]
    
    initial_distinct = distinct_counts[0]
    final_distinct = distinct_counts[-1]
    max_distinct = max(distinct_counts)
    min_distinct = min(distinct_counts)
    
    ratio = final_distinct / (initial_distinct + 1e-10)
    
    # Verdict
    n_elements = history[-1].size
    final_ratio = final_distinct / n_elements
    
    if final_ratio < 0.01:
        evolution = "collapsed"
        status = "FAIL"
        message = f"Collapsus: {initial_distinct} → {final_distinct} valeurs distinctes"
        blocking = True
    elif ratio < 0.5:
        evolution = "decreasing"
        status = "NEUTRAL"
        message = f"Diversité réduite: {initial_distinct} → {final_distinct} valeurs"
        blocking = False
    else:
        evolution = "preserved"
        status = "PASS"
        message = f"Diversité préservée: {final_distinct} valeurs distinctes"
        blocking = False
    
    return DiversityResult(
        test_name=name,
        initial_diversity=float(initial_distinct),
        final_diversity=float(final_distinct),
        max_diversity=float(max_distinct),
        min_diversity=float(min_distinct),
        ratio_final_initial=ratio,
        evolution=evolution,
        status=status,
        message=message,
        blocking=blocking
    )
	# Ajouter à la fin du fichier test_diversity.py

def test_local_diversity_preservation(
    history: List[np.ndarray],
    patch_size: int = 5,
    n_patches: int = 20,
    name: str = "UNIV-002b") -> DiversityResult:
    """
    Test UNIV-002b : Diversité locale/structurelle.
    
    DIFFÉRENCE avec UNIV-002 (global) :
    - UNIV-002  : mesure std(état_complet) → variance GLOBALE
    - UNIV-002b : mesure std(patches_locaux) → variance LOCALE
    
    Objectif :
    Détecter si structure locale persiste même quand variance globale chute.
    
    Méthode :
    - Échantillonne n_patches patches de taille patch_size×patch_size
    - Calcule std dans chaque patch
    - Moyenne des std locaux = diversité locale
    
    Cas d'usage :
    - État homogène globalement mais structuré localement : UNIV-002b > UNIV-002
    - État hétérogène globalement mais lisse localement : UNIV-002 > UNIV-002b
    
    Args:
        history: Liste des états [D_0, D_1, ..., D_T]
        patch_size: Taille des patches (défaut 5×5)
        n_patches: Nombre de patches à échantillonner (défaut 20)
        name: Nom du test
    
    Returns:
        DiversityResult avec diversité locale
    
    Applicabilité :
    - Uniquement rang 2 (matrices)
    - Nécessite shape[0] ≥ patch_size et shape[1] ≥ patch_size
    
    Références:
    - Audit UNIV-002 (2025-12-28) : Section 6 "Local vs global"
    - OBS-GAM-001 : Pattern de collapse global observé
    """
    if not history:
        return DiversityResult(
            test_name=name,
            initial_diversity=0.0,
            max_diversity=0.0,
            min_diversity=0.0,
            final_diversity=0.0,
            ratio_final_initial=0.0,
            evolution="unknown",
            status="ERROR",
            message="Historique vide",
            blocking=False
        )
    
    initial_state = history[0]
    final_state = history[-1]
    
    # Vérifier applicabilité
    if initial_state.ndim != 2:
        return DiversityResult(
            test_name=name,
            initial_diversity=0.0,
            max_diversity=0.0,
            min_diversity=0.0,
            final_diversity=0.0,
            ratio_final_initial=0.0,
            evolution="not_applicable",
            status="NEUTRAL",
            message="Test applicable uniquement rang 2",
            blocking=False
        )
    
    if (initial_state.shape[0] < patch_size or 
        initial_state.shape[1] < patch_size):
        return DiversityResult(
            test_name=name,
            initial_diversity=0.0,
            final_diversity=0.0,
            ratio_final_initial=0.0,
            evolution="not_applicable",
            status="NEUTRAL",
            message=f"État trop petit pour patches {patch_size}×{patch_size}",
            blocking=False
        )
    
    # Fonction helper : calcul diversité locale
    def compute_local_diversity(state: np.ndarray) -> float:
        """Calcule diversité locale via échantillonnage de patches."""
        local_stds = []
        
        # Fixer seed pour reproductibilité (basé sur shape et somme)
        seed = int(np.sum(state) * 1000) % 2**31
        rng = np.random.RandomState(seed)
        
        for _ in range(n_patches):
            # Échantillonner position aléatoire
            i = rng.randint(0, state.shape[0] - patch_size + 1)
            j = rng.randint(0, state.shape[1] - patch_size + 1)
            
            # Extraire patch
            patch = state[i:i+patch_size, j:j+patch_size]
            
            # Calculer std du patch
            patch_std = np.std(patch)
            local_stds.append(patch_std)
        
        # Moyenne des std locaux
        return np.mean(local_stds)
    
    # Calculer diversité locale initiale et finale
    initial_local = compute_local_diversity(initial_state)
    final_local = compute_local_diversity(final_state)
    
    # Calculer ratio
    if initial_local > 1e-10:
        ratio = final_local / initial_local
    else:
        ratio = 0.0
    
    # Déterminer évolution
    if ratio < 0.9:
        evolution = "decreasing"
    elif ratio > 1.1:
        evolution = "increasing"
    else:
        evolution = "stable"
    
    # Status (pas de verdict strict, observationnel)
    if ratio < 0.3:
        status = "FAIL"
        message = f"Collapse diversité locale: {ratio:.3f}x initial"
    elif ratio > 0.7:
        status = "PASS"
        message = f"Maintien diversité locale: {ratio:.3f}x initial"
    else:
        status = "NEUTRAL"
        message = f"Diversité locale partielle: {ratio:.3f}x initial"
    
    return DiversityResult(
        test_name=name,
        initial_diversity=initial_local,
        max_diversity=0.0,
        min_diversity=0.0,
        final_diversity=final_local,
        ratio_final_initial=ratio,
        evolution=evolution,
        status=status,
        message=message,
        blocking=False
    )


def test_spatial_heterogeneity(
    history: List[np.ndarray],
    grid_size: int = 10,
    name: str = "DIV-HETERO"
) -> DiversityResult:
    """
    Test DIV-HETERO : Hétérogénéité spatiale.
    
    Mesure la variance de la variance locale (méta-diversité).
    
    Différence avec UNIV-002b :
    - UNIV-002b : moyenne(std_patches) → diversité locale moyenne
    - DIV-HETERO : std(std_patches) → hétérogénéité spatiale
    
    Interprétation :
    - Score élevé : État hétérogène (zones denses + zones lisses)
    - Score faible : État homogène (variance locale uniforme)
    
    Cas d'usage :
    - Détecter patterns à plusieurs échelles
    - Distinguer homogénéisation globale vs structuration modulaire
    
    Args:
        history: Liste des états
        grid_size: Taille de la grille de découpage
        name: Nom du test
    
    Returns:
        DiversityResult avec hétérogénéité spatiale
    """
    if not history:
        return DiversityResult(
            test_name=name,
            initial_diversity=0.0,
            max_diversity=0.0,
            min_diversity=0.0,
            final_diversity=0.0,
            ratio_final_initial=0.0,
            evolution="unknown",
            status="ERROR",
            message="Historique vide",
            blocking=False
        )
    
    initial_state = history[0]
    final_state = history[-1]
    
    # Vérifier applicabilité
    if initial_state.ndim != 2:
        return DiversityResult(
            test_name=name,
            initial_diversity=0.0,
            max_diversity=0.0,
            min_diversity=0.0,
            final_diversity=0.0,
            ratio_final_initial=0.0,
            evolution="not_applicable",
            status="NEUTRAL",
            message="Test applicable uniquement rang 2",
            blocking=False
        )
    
    def compute_spatial_heterogeneity(state: np.ndarray) -> float:
        """Calcule hétérogénéité via grille régulière."""
        h, w = state.shape
        cell_h = max(1, h // grid_size)
        cell_w = max(1, w // grid_size)
        
        local_stds = []
        
        for i in range(0, h, cell_h):
            for j in range(0, w, cell_w):
                cell = state[i:min(i+cell_h, h), j:min(j+cell_w, w)]
                if cell.size > 1:
                    local_stds.append(np.std(cell))
        
        if len(local_stds) > 1:
            # Hétérogénéité = std des std locaux
            return np.std(local_stds)
        else:
            return 0.0
    
    initial_hetero = compute_spatial_heterogeneity(initial_state)
    final_hetero = compute_spatial_heterogeneity(final_state)
    
    if initial_hetero > 1e-10:
        ratio = final_hetero / initial_hetero
    else:
        ratio = 0.0
    
    # Évolution
    if ratio < 0.9:
        evolution = "decreasing"
    elif ratio > 1.1:
        evolution = "increasing"
    else:
        evolution = "stable"
    
    # Status
    if ratio < 0.5:
        status = "FAIL"
        message = f"Perte hétérogénéité: {ratio:.3f}x initial"
    elif ratio > 0.8:
        status = "PASS"
        message = f"Maintien hétérogénéité: {ratio:.3f}x initial"
    else:
        status = "NEUTRAL"
        message = f"Hétérogénéité partielle: {ratio:.3f}x initial"
    
    return DiversityResult(
        test_name=name,
        initial_diversity=initial_hetero,
        max_diversity=0.0,
        min_diversity=0.0,
        final_diversity=final_hetero,
        ratio_final_initial=ratio,
        evolution=evolution,
        status=status,
        message=message,
        blocking=False
    )