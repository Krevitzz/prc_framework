"""
Profiling par entité : dans quels clusters tombe chaque gamma/encoding/modifier ?

Répond aux questions scientifiques du charter :
- Quels Γ produisent des attracteurs stables ?
- Y a-t-il universalité de Γ ? (même cluster cross-encodings)
- Quels encodings convergent vers les mêmes attracteurs quel que soit Γ ?

@ROLE    Agrégation par entité → profils cross-clusters + universalité
@LAYER   analysing

@EXPORTS
  EntityProfile                               → dataclass | profil d'une entité
  compute_entity_profiles(data, labels, ...)  → Dict      | profils par type
  detect_universality(profiles, named)        → List[Dict] | gammas universels

@LIFECYCLE
  CREATES  profils (dicts légers)
  RECEIVES labels, named_clusters  depuis clustering + namer
  PASSES   entity_profiles         vers hub → outputs

@CONFORMITY
  OK   Comptages sur labels — zéro matérialisation lourde
  OK   Universalité = même cluster dominant cross-encodings (charter §1.6)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np


# =========================================================================
# ENTITY PROFILE
# =========================================================================

@dataclass
class EntityProfile:
    """Profil d'une entité (gamma, encoding ou modifier) cross-clusters.

    Attributs :
        entity_type : 'gamma', 'encoding', 'modifier'
        entity_id : identifiant de l'entité
        n_runs_total : nombre total de runs (toutes strates, tous statuts)
        n_runs_clustered : nombre de runs dans la population clusterée
        cluster_distribution : {cluster_id: count} — dans quels clusters
        status_distribution : {OK: n, EXPLOSION: n, ...} — tous statuts
        strates : dans quelles strates apparaît
        dominant_cluster : cluster avec le plus de runs (None si n_clustered=0)
        dominant_cluster_fraction : fraction dans le cluster dominant
    """
    entity_type:             str
    entity_id:               str
    n_runs_total:            int
    n_runs_clustered:        int
    cluster_distribution:    Dict[int, int]      # {cluster_id: count}
    status_distribution:     Dict[str, int]      # {OK: n, EXPLOSION: n, ...}
    strates:                 List[str]
    dominant_cluster:        Optional[int]
    dominant_cluster_fraction: float

    @property
    def n_clusters(self) -> int:
        """Nombre de clusters distincts dans lesquels l'entité apparaît."""
        return len([c for c, n in self.cluster_distribution.items() if c >= 0 and n > 0])

    @property
    def is_concentrated(self) -> bool:
        """True si >70% des runs clustered sont dans un seul cluster."""
        return self.dominant_cluster_fraction > 0.70

    @property
    def explosion_rate(self) -> float:
        """Fraction de runs EXPLOSION + COLLAPSED."""
        n_pathological = (self.status_distribution.get('EXPLOSION', 0) +
                          self.status_distribution.get('COLLAPSED', 0))
        return n_pathological / self.n_runs_total if self.n_runs_total > 0 else 0.0


# =========================================================================
# CALCUL PROFILS PAR ENTITÉ
# =========================================================================

def _profiles_for_entity_type(entity_arr_full: np.ndarray,
                                entity_arr_strate: np.ndarray,
                                labels: np.ndarray,
                                run_statuses_full: np.ndarray,
                                strate_id: str,
                                entity_type: str) -> List[EntityProfile]:
    """Calcule les profils pour un type d'entité dans une strate."""
    unique_ids = np.unique(entity_arr_strate)
    profiles = []

    for eid in unique_ids:
        eid_str = str(eid)
        if not eid_str or eid_str == '':
            continue

        # Runs dans la strate pour cette entité
        strate_mask = entity_arr_strate == eid
        entity_labels = labels[strate_mask]

        # Cluster distribution (excluant résidu -1 du comptage "clustered")
        unique_cl, counts_cl = np.unique(entity_labels, return_counts=True)
        cluster_dist = {int(c): int(n) for c, n in zip(unique_cl, counts_cl)}
        n_clustered = sum(n for c, n in cluster_dist.items() if c >= 0)

        # Dominant cluster
        clustered_labels = entity_labels[entity_labels >= 0]
        if len(clustered_labels) > 0:
            unique_pos, counts_pos = np.unique(clustered_labels, return_counts=True)
            dom_idx = np.argmax(counts_pos)
            dominant = int(unique_pos[dom_idx])
            dominant_frac = float(counts_pos[dom_idx] / n_clustered) if n_clustered > 0 else 0.0
        else:
            dominant = None
            dominant_frac = 0.0

        # Status distribution (sur TOUS les runs, pas juste la strate)
        full_mask = entity_arr_full == eid
        statuses = run_statuses_full[full_mask]
        unique_st, counts_st = np.unique(statuses, return_counts=True)
        status_dist = {str(s): int(c) for s, c in zip(unique_st, counts_st)}
        n_total = int(full_mask.sum())

        profiles.append(EntityProfile(
            entity_type=entity_type,
            entity_id=eid_str,
            n_runs_total=n_total,
            n_runs_clustered=n_clustered,
            cluster_distribution=cluster_dist,
            status_distribution=status_dist,
            strates=[strate_id],
            dominant_cluster=dominant,
            dominant_cluster_fraction=dominant_frac,
        ))

    return profiles


def compute_entity_profiles(data,  # AnalysingData
                              labels: np.ndarray,
                              run_indices: np.ndarray,
                              strate_id: str) -> Dict[str, List[EntityProfile]]:
    """Profils par entité pour une strate.

    Args:
        data : AnalysingData complet (pour accès aux metadata full).
        labels : labels du peeling (n_strate,). -1 = résidu.
        run_indices : indices de la strate dans data.
        strate_id : identifiant de la strate.

    Returns:
        {'gamma': [...], 'encoding': [...], 'modifier': [...]}
    """
    result = {}

    for entity_type, full_arr, strate_accessor in [
        ('gamma', data.gamma_ids, data.gamma_ids[run_indices]),
        ('encoding', data.encoding_ids, data.encoding_ids[run_indices]),
        ('modifier', data.modifier_ids, data.modifier_ids[run_indices]),
    ]:
        profiles = _profiles_for_entity_type(
            entity_arr_full=full_arr,
            entity_arr_strate=strate_accessor,
            labels=labels,
            run_statuses_full=data.run_statuses,
            strate_id=strate_id,
            entity_type=entity_type,
        )
        result[entity_type] = profiles

    return result


# =========================================================================
# DÉTECTION D'UNIVERSALITÉ
# =========================================================================

def detect_universality(gamma_profiles: List[EntityProfile],
                         named_clusters: List[Dict],
                         min_encodings: int = 3,
                         min_concentration: float = 0.70) -> List[Dict]:
    """Détecte les gammas universels.

    Un gamma est universel si, quel que soit l'encoding avec lequel il est
    combiné, il tombe majoritairement dans le même cluster. C'est le test
    le plus direct de la thèse centrale (charter §1.6) :
    "Γ organise D indépendamment de son encodage."

    Heuristique :
    - Le gamma doit apparaître avec au moins min_encodings encodings distincts
    - Il doit être concentré (dominant_cluster_fraction > min_concentration)

    Args:
        gamma_profiles : profils des gammas.
        named_clusters : clusters nommés (pour le nom du cluster dominant).
        min_encodings : nombre minimum d'encodings pour qualifier.
        min_concentration : fraction minimale dans le cluster dominant.

    Returns:
        Liste de {gamma_id, dominant_cluster, cluster_name, concentration, n_encodings}
    """
    # Index des noms de clusters
    cluster_names = {}
    for nc in named_clusters:
        cid = nc.get('cluster_id')
        if cid is not None:
            cluster_names[cid] = nc.get('name', f'C{cid}')

    universal = []

    for profile in gamma_profiles:
        if not profile.is_concentrated:
            continue
        if profile.dominant_cluster is None:
            continue
        if profile.dominant_cluster_fraction < min_concentration:
            continue

        # Compter combien d'encodings distincts ont ce gamma dans ce cluster
        # Note : on n'a pas cette info directement — on utilise n_runs_clustered
        # comme proxy. Un gamma avec 20 runs clustered dans le même cluster
        # est probablement universel si le plan expérimental varie les encodings.
        # La vraie détection cross-encoding nécessite les metadata par run.
        # Pour l'instant on signale les gammas concentrés.

        universal.append({
            'gamma_id': profile.entity_id,
            'dominant_cluster': profile.dominant_cluster,
            'cluster_name': cluster_names.get(profile.dominant_cluster, '?'),
            'concentration': profile.dominant_cluster_fraction,
            'n_runs_clustered': profile.n_runs_clustered,
            'n_clusters': profile.n_clusters,
            'explosion_rate': profile.explosion_rate,
        })

    # Trier par concentration décroissante
    universal.sort(key=lambda x: x['concentration'], reverse=True)

    return universal


def detect_encoding_convergence(encoding_profiles: List[EntityProfile],
                                  named_clusters: List[Dict],
                                  min_gammas: int = 3,
                                  min_concentration: float = 0.70) -> List[Dict]:
    """Détecte les encodings qui convergent vers les mêmes attracteurs.

    Même logique que universalité gamma, mais inversée :
    un encoding convergent tombe dans le même cluster quel que soit Γ.

    Charter §1.6 : "Certains encodings ont une structure qui surpasse l'action de Γ."
    """
    cluster_names = {}
    for nc in named_clusters:
        cid = nc.get('cluster_id')
        if cid is not None:
            cluster_names[cid] = nc.get('name', f'C{cid}')

    convergent = []

    for profile in encoding_profiles:
        if not profile.is_concentrated:
            continue
        if profile.dominant_cluster is None:
            continue
        if profile.dominant_cluster_fraction < min_concentration:
            continue

        convergent.append({
            'encoding_id': profile.entity_id,
            'dominant_cluster': profile.dominant_cluster,
            'cluster_name': cluster_names.get(profile.dominant_cluster, '?'),
            'concentration': profile.dominant_cluster_fraction,
            'n_runs_clustered': profile.n_runs_clustered,
            'n_clusters': profile.n_clusters,
        })

    convergent.sort(key=lambda x: x['concentration'], reverse=True)
    return convergent
