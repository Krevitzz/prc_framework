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
# UTILITAIRES
# ============================================================================

def _safe_std(arr: np.ndarray) -> float:
    """Calcul sécurisé de l'écart-type avec gestion des NaN et overflow."""
    flat = arr.flatten()
    
    # Vérifier les valeurs infinies
    if not np.all(np.isfinite(flat)):
        # Remplacer inf par de grandes valeurs finies
        flat = np.where(np.isinf(flat), np.finfo(np.float64).max * np.sign(flat), flat)
        # Remplacer NaN par la médiane
        mask = np.isnan(flat)
        if np.any(mask):
            median_val = np.nanmedian(flat)
            flat[mask] = median_val
    
    # Vérifier si le tableau est constant
    if np.allclose(flat, flat[0]):
        return 0.0
    
    # Calculer l'écart-type avec gestion de l'overflow
    try:
        # Utiliser une méthode robuste pour éviter l'overflow
        mean = np.mean(flat)
        # Centrer et réduire pour éviter l'overflow
        centered = flat - mean
        # Normaliser par la valeur absolue max pour éviter l'overflow au carré
        max_abs = np.max(np.abs(centered))
        if max_abs > 0:
            centered_normalized = centered / max_abs
            std_normalized = np.std(centered_normalized)
            return std_normalized * max_abs
        else:
            return 0.0
    except (OverflowError, ValueError):
        # Méthode alternative: calcul par batches
        batch_size = 1000
        n_batches = len(flat) // batch_size + 1
        variances = []
        
        for i in range(n_batches):
            batch = flat[i*batch_size:(i+1)*batch_size]
            if len(batch) > 1:
                batch_mean = np.mean(batch)
                batch_var = np.mean((batch - batch_mean)**2)
                variances.append(batch_var)
        
        if variances:
            return np.sqrt(np.mean(variances))
        else:
            return 0.0


def _safe_entropy(state: np.ndarray, n_bins: int = 50) -> float:
    """Calcul sécurisé de l'entropie."""
    flat = state.flatten()
    
    # Nettoyer les données
    flat_clean = flat[np.isfinite(flat)]
    if len(flat_clean) == 0:
        return 0.0
    
    # Éviter les valeurs extrêmes qui créent des bins vides
    q1, q3 = np.percentile(flat_clean, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Filtrer les outliers pour un histogramme stable
    filtered = flat_clean[(flat_clean >= lower_bound) & (flat_clean <= upper_bound)]
    
    if len(filtered) < 10:  # Pas assez de données
        return 0.0
    
    try:
        hist, _ = np.histogram(filtered, bins=min(n_bins, len(filtered)//2), density=True)
        hist = hist / (np.sum(hist) + 1e-10)
        hist_nonzero = hist[hist > 0]
        entropy = -np.sum(hist_nonzero * np.log(hist_nonzero + 1e-10))
        return entropy
    except:
        return 0.0


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
        diversity = _safe_std(state)
        diversities.append(diversity)
    
    # Vérifier les valeurs NaN dans diversities
    diversities = np.array(diversities)
    if np.any(np.isnan(diversities)):
        # Remplacer NaN par la dernière valeur valide
        for i in range(1, len(diversities)):
            if np.isnan(diversities[i]):
                diversities[i] = diversities[i-1] if i > 0 else 0.0
    
    # Statistiques
    initial_div = diversities[0] if len(diversities) > 0 else 0.0
    final_div = diversities[-1] if len(diversities) > 0 else 0.0
    max_div = np.nanmax(diversities) if len(diversities) > 0 else 0.0
    min_div = np.nanmin(diversities) if len(diversities) > 0 else 0.0
    
    # Ratio (avec protection division par zéro)
    if initial_div < 1e-10:
        if final_div < 1e-10:
            ratio = 1.0
        else:
            ratio = np.inf if final_div > 0 else 0.0
    else:
        ratio = final_div / (initial_div + 1e-10)
    
    # Détecte tendance
    if len(diversities) > 1:
        # Filtrer les NaN pour le calcul de tendance
        valid_idx = np.where(np.isfinite(diversities))[0]
        if len(valid_idx) > 1:
            valid_diversities = diversities[valid_idx]
            try:
                trend_coef = np.polyfit(valid_idx, valid_diversities, 1)[0]
            except:
                trend_coef = 0.0
        else:
            trend_coef = 0.0
        
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
        initial_diversity=float(initial_div),
        final_diversity=float(final_div),
        max_diversity=float(max_div),
        min_diversity=float(min_div),
        ratio_final_initial=float(ratio),
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
    
    entropies = [_safe_entropy(state) for state in history]
    
    initial_ent = entropies[0] if len(entropies) > 0 else 0.0
    final_ent = entropies[-1] if len(entropies) > 0 else 0.0
    max_ent = max(entropies) if entropies else 0.0
    min_ent = min(entropies) if entropies else 0.0
    
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
        # Nettoyer les données pour le calcul de range
        flat = state.flatten()
        clean_flat = flat[np.isfinite(flat)]
        if len(clean_flat) == 0:
            range_val = 0.0
        else:
            range_val = np.max(clean_flat) - np.min(clean_flat)
        ranges.append(range_val)
    
    initial_range = ranges[0] if len(ranges) > 0 else 0.0
    final_range = ranges[-1] if len(ranges) > 0 else 0.0
    max_range = max(ranges) if ranges else 0.0
    min_range = min(ranges) if ranges else 0.0
    
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
        clean_flat = flat[np.isfinite(flat)]
        if len(clean_flat) == 0:
            return 0.0
        
        mean = np.mean(clean_flat)
        std = np.std(clean_flat)
        if abs(mean) < 1e-10:
            return 0.0 if std < 1e-10 else np.inf
        return std / abs(mean)
    
    cvs = [compute_cv(state) for state in history]
    
    initial_cv = cvs[0] if len(cvs) > 0 else 0.0
    final_cv = cvs[-1] if len(cvs) > 0 else 0.0
    max_cv = max(cvs) if cvs else 0.0
    min_cv = min(cvs) if cvs else 0.0
    
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
        flat = state.flatten()
        clean_flat = flat[np.isfinite(flat)]
        if len(clean_flat) == 0:
            return 0
        
        sorted_flat = np.sort(clean_flat)
        diffs = np.diff(sorted_flat)
        n_distinct = 1 + np.sum(diffs > tol)
        return n_distinct
    
    distinct_counts = [count_distinct(state, tolerance) for state in history]
    
    initial_distinct = distinct_counts[0] if len(distinct_counts) > 0 else 0
    final_distinct = distinct_counts[-1] if len(distinct_counts) > 0 else 0
    max_distinct = max(distinct_counts) if distinct_counts else 0
    min_distinct = min(distinct_counts) if distinct_counts else 0
    
    ratio = final_distinct / (initial_distinct + 1e-10)
    
    # Verdict
    n_elements = history[-1].size if len(history) > 0 else 1
    final_ratio = final_distinct / n_elements if n_elements > 0 else 0
    
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
        
        # Fixer seed pour reproductibilité
        try:
            state_sum = np.nansum(state)  # Utiliser nansum pour éviter NaN
            if not np.isfinite(state_sum):
                state_sum = 0.0
            seed = int(abs(state_sum) * 1000) % 2**31
            rng = np.random.RandomState(seed)
        except:
            rng = np.random.RandomState(42)  # Seed par défaut
        
        for _ in range(n_patches):
            # Échantillonner position aléatoire
            i = rng.randint(0, max(1, state.shape[0] - patch_size + 1))
            j = rng.randint(0, max(1, state.shape[1] - patch_size + 1))
            
            # Extraire patch
            patch = state[i:i+patch_size, j:j+patch_size]
            
            # Calculer std du patch avec gestion des NaN
            patch_clean = patch[np.isfinite(patch)]
            if len(patch_clean) > 1:
                patch_std = np.std(patch_clean)
            else:
                patch_std = 0.0
            local_stds.append(patch_std)
        
        # Moyenne des std locaux
        if local_stds:
            return float(np.nanmean(local_stds))
        else:
            return 0.0
    
    # Calculer diversité locale initiale et finale
    try:
        initial_local = compute_local_diversity(initial_state)
        final_local = compute_local_diversity(final_state)
    except Exception as e:
        return DiversityResult(
            test_name=name,
            initial_diversity=0.0,
            max_diversity=0.0,
            min_diversity=0.0,
            final_diversity=0.0,
            ratio_final_initial=0.0,
            evolution="error",
            status="ERROR",
            message=f"Erreur calcul: {str(e)}",
            blocking=False
        )
    
    # Calculer ratio
    if initial_local > 1e-10:
        ratio = final_local / initial_local
    else:
        ratio = 0.0 if final_local < 1e-10 else np.inf
    
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
                cell_clean = cell[np.isfinite(cell)]
                if len(cell_clean) > 1:
                    local_stds.append(np.std(cell_clean))
        
        if len(local_stds) > 1:
            # Hétérogénéité = std des std locaux
            return float(np.std(local_stds))
        else:
            return 0.0
    
    try:
        initial_hetero = compute_spatial_heterogeneity(initial_state)
        final_hetero = compute_spatial_heterogeneity(final_state)
    except Exception as e:
        return DiversityResult(
            test_name=name,
            initial_diversity=0.0,
            max_diversity=0.0,
            min_diversity=0.0,
            final_diversity=0.0,
            ratio_final_initial=0.0,
            evolution="error",
            status="ERROR",
            message=f"Erreur calcul: {str(e)}",
            blocking=False
        )
    
    if initial_hetero > 1e-10:
        ratio = final_hetero / initial_hetero
    else:
        ratio = 0.0 if final_hetero < 1e-10 else np.inf
    
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