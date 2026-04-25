"""
Cohérence clusters HDBSCAN ↔ classification P1 du running.

Le running classifie chaque run en P1 (FLAT, OSCILLATING, TRANSITIONAL,
EXPLOSIVE, MIXED) sur les signaux dynamiques O(n²). L'analysing cluster
sur toutes les features (incluant spectrales). Les deux doivent être cohérents.

Ce module ne rejette rien — il documente les incohérences.

@ROLE    Validation : cohérence clusters ↔ P1 + statuts
@LAYER   analysing

@EXPORTS
  ClusterCoherence                                → dataclass | cohérence d'un cluster
  compute_cluster_coherence(data, labels, indices) → List      | cohérence par cluster

@LIFECYCLE
  CREATES  List[ClusterCoherence]  objets légers
  RECEIVES labels                  depuis clustering
  PASSES   coherence_results       vers hub → outputs

@CONFORMITY
  OK   Pas de rejet — documentation des incohérences
  OK   Zéro matérialisation lourde (comptages sur metadata)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


# =========================================================================
# CLUSTER COHERENCE
# =========================================================================

@dataclass
class ClusterCoherence:
    """Cohérence entre un cluster HDBSCAN et la classification P1.

    regime_entropy basse = cluster pur (dominé par un seul régime P1).
    regime_entropy haute = cluster de mélange (à investiguer).
    """
    cluster_id:          int
    n_runs:              int
    regime_distribution: Dict[str, float]   # {FLAT: 0.6, OSCILLATING: 0.3, ...}
    regime_entropy:      float              # Shannon entropy normalisée [0, 1]
    dominant_regime:     str
    dominant_fraction:   float
    truncated_fraction:  float
    status_distribution: Dict[str, float]   # {OK: 0.85, OK_TRUNCATED: 0.15}

    @property
    def is_pure(self) -> bool:
        """Un cluster est 'pur' si dominé à >70% par un seul régime."""
        return self.dominant_fraction > 0.70

    @property
    def is_mixed(self) -> bool:
        """Un cluster est 'mixte' si aucun régime ne dépasse 50%."""
        return self.dominant_fraction < 0.50


# =========================================================================
# SHANNON ENTROPY NORMALISÉE
# =========================================================================

def _normalized_entropy(distribution: Dict[str, float]) -> float:
    """Entropie de Shannon normalisée sur [0, 1].

    0 = un seul régime (cluster pur).
    1 = distribution uniforme (cluster de mélange maximal).
    """
    values = [v for v in distribution.values() if v > 0]
    if len(values) <= 1:
        return 0.0

    total = sum(values)
    probs = [v / total for v in values]

    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    max_entropy = np.log2(len(probs))

    if max_entropy == 0:
        return 0.0

    return float(entropy / max_entropy)


# =========================================================================
# CALCUL COHÉRENCE
# =========================================================================

def _coherence_for_group(data,  # AnalysingData
                          indices: np.ndarray,
                          cluster_id: int) -> ClusterCoherence:
    """Calcule la cohérence pour un groupe de runs (cluster ou résidu)."""
    n = len(indices)
    if n == 0:
        return ClusterCoherence(
            cluster_id=cluster_id, n_runs=0,
            regime_distribution={}, regime_entropy=0.0,
            dominant_regime='', dominant_fraction=0.0,
            truncated_fraction=0.0, status_distribution={},
        )

    # Distribution P1 regime
    regimes = data.p1_regime_class[indices]
    unique_reg, counts_reg = np.unique(regimes, return_counts=True)
    regime_dist = {str(r): int(c) / n for r, c in zip(unique_reg, counts_reg)}

    if len(unique_reg) > 0:
        dominant_idx = np.argmax(counts_reg)
        dominant = str(unique_reg[dominant_idx])
        dominant_frac = float(counts_reg[dominant_idx] / n)
    else:
        dominant = ''
        dominant_frac = 0.0

    # Entropie
    entropy = _normalized_entropy(regime_dist)

    # Statuts
    statuses = data.run_statuses[indices]
    unique_st, counts_st = np.unique(statuses, return_counts=True)
    status_dist = {str(s): int(c) / n for s, c in zip(unique_st, counts_st)}

    truncated_frac = float(np.sum(statuses == 'OK_TRUNCATED') / n)

    return ClusterCoherence(
        cluster_id=cluster_id,
        n_runs=n,
        regime_distribution=regime_dist,
        regime_entropy=entropy,
        dominant_regime=dominant,
        dominant_fraction=dominant_frac,
        truncated_fraction=truncated_frac,
        status_distribution=status_dist,
    )


def compute_cluster_coherence(data,  # AnalysingData
                                labels: np.ndarray,
                                run_indices: np.ndarray,
                                verbose: bool = True) -> List[ClusterCoherence]:
    """Cohérence pour chaque cluster + le résidu.

    Args:
        data : AnalysingData complet.
        labels : labels du peeling (n_strate,). -1 = résidu.
        run_indices : indices de la strate dans data.
        verbose : afficher le résumé.

    Returns:
        Liste de ClusterCoherence (un par cluster + un pour le résidu).
    """
    results = []

    # Clusters
    unique_labels = sorted(set(labels.tolist()) - {-1})
    for cid in unique_labels:
        cluster_local = np.where(labels == cid)[0]
        cluster_data_idx = run_indices[cluster_local]
        coh = _coherence_for_group(data, cluster_data_idx, cid)
        results.append(coh)

    # Résidu
    residual_local = np.where(labels == -1)[0]
    if len(residual_local) > 0:
        residual_data_idx = run_indices[residual_local]
        coh_res = _coherence_for_group(data, residual_data_idx, -1)
        results.append(coh_res)

    if verbose:
        print(f"  [validate] Cohérence P1 ↔ clusters :")
        for coh in results:
            tag = 'RÉSIDU' if coh.cluster_id == -1 else f'C{coh.cluster_id}'
            purity = 'PUR' if coh.is_pure else ('MIXTE' if coh.is_mixed else 'MOYEN')
            trunc = f' trunc={coh.truncated_fraction:.0%}' if coh.truncated_fraction > 0 else ''
            print(f"    {tag:>8} ({coh.n_runs:>4} runs) "
                  f"H={coh.regime_entropy:.2f} {purity:5s} "
                  f"dom={coh.dominant_regime}{trunc}")

    return results
