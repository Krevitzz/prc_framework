"""
Stratification des runs par propriétés structurelles des atomics.

Utilise les METADATA AST des gammas (differentiable, rank_constraint) croisées
avec rank_eff du parquet pour déterminer les features applicables par strate.
Chaque strate a un espace de features homogène (zéro NaN structurel).

@ROLE    Stratification : (rank_eff × differentiable) → strates avec features applicables
@LAYER   analysing

@EXPORTS
  Strate                                     → dataclass | définition d'une strate
  build_gamma_properties(gamma_ids)           → Dict      | propriétés depuis METADATA AST
  compute_strates(data, run_indices)          → List[Strate] | strates avec features

@LIFECYCLE
  CREATES  List[Strate]    objets légers (indices + listes de noms)
  RECEIVES AnalysingData    depuis hub
  PASSES   List[Strate]     vers hub → prepare + clustering

@CONFORMITY
  OK   Stratification via METADATA AST, pas heuristique (point 3 validé)
  OK   FEATURES_STRUCTURAL_NAN du registre = source de vérité (P7)
  OK   Fallback heuristique si gamma absent du discovery, documenté
  OK   Strates = indices + noms, zéro copie de données (SD-2)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import warnings

import numpy as np

from running.features_registry import (
    FEATURE_NAMES,
    FEATURES_STRUCTURAL_NAN,
)


# =========================================================================
# STRATE — définition
# =========================================================================

@dataclass
class Strate:
    """Définition d'une strate de clustering.

    Chaque strate regroupe les runs ayant les mêmes propriétés structurelles
    (rank_eff, differentiable), donc le même espace de features applicables.

    Attributs :
        strate_id : identifiant lisible (ex: "rank2_diff", "rank3_nodiff")
        rank_eff : rang effectif des runs de la strate
        differentiable : True si les gammas de la strate sont différentiables
        run_indices : indices dans AnalysingData (population principale)
        features_applicable : FEATURE_NAMES - NaN structurels pour cette strate
        features_excluded : {feature: raison} — traçabilité des exclusions
    """
    strate_id:           str
    rank_eff:            int
    differentiable:      bool
    run_indices:         np.ndarray         # indices dans la population principale
    features_applicable: List[str]
    features_excluded:   Dict[str, str]     # {feature_name: raison}

    @property
    def n_runs(self) -> int:
        return len(self.run_indices)

    @property
    def n_features(self) -> int:
        return len(self.features_applicable)


# =========================================================================
# PROPRIÉTÉS GAMMA DEPUIS METADATA AST
# =========================================================================

def build_gamma_properties(gamma_ids: np.ndarray) -> Dict[str, Dict]:
    """Pour chaque gamma_id unique, récupère differentiable et rank_constraint
    depuis discover_gammas_metadata() (AST, zéro JAX).

    Fallback si gamma absent du discovery :
        - Log warning
        - Retourne {'differentiable': None, 'rank_constraint': None}
        Le caller devra déduire depuis les données.

    Returns:
        {gamma_id: {'differentiable': bool|None, 'rank_constraint': ...}}
    """
    from utils.io_v8 import discover_gammas_metadata

    unique_gammas = np.unique(gamma_ids)
    unique_gammas = [g for g in unique_gammas if g and g != '']

    # Discovery AST — une seule lecture pour tous les gammas
    try:
        gamma_registry = discover_gammas_metadata()
    except FileNotFoundError as e:
        warnings.warn(f"[stratify] Discovery gammas impossible : {e}. "
                      f"Fallback heuristique pour tous les gammas.")
        return {str(gid): {'differentiable': None, 'rank_constraint': None}
                for gid in unique_gammas}

    properties = {}
    n_missing = 0

    for gid in unique_gammas:
        gid_str = str(gid)
        entry = gamma_registry.get(gid_str)

        if entry is not None and entry.get('metadata') is not None:
            meta = entry['metadata']
            properties[gid_str] = {
                'differentiable': meta.get('differentiable'),
                'rank_constraint': meta.get('rank_constraint'),
            }
        else:
            n_missing += 1
            properties[gid_str] = {
                'differentiable': None,
                'rank_constraint': None,
            }

    if n_missing > 0:
        warnings.warn(
            f"[stratify] {n_missing}/{len(unique_gammas)} gammas absents du discovery. "
            f"Fallback heuristique pour ceux-ci."
        )

    return properties


# =========================================================================
# DÉDUCTION HEURISTIQUE (fallback)
# =========================================================================

def _infer_differentiable_from_data(gamma_id: str,
                                      gamma_mask: np.ndarray,
                                      features_matrix_f4: Optional[np.ndarray]
                                      ) -> bool:
    """Déduit differentiable depuis les données quand le METADATA est absent.

    Logique : si TOUS les runs de ce gamma ont f4_trace_J_mean = NaN,
    le gamma est non-différentiable. C'est déterministe car les NaN structurels
    sont injectés à la compilation par le running.

    Args:
        gamma_id : pour le logging
        gamma_mask : boolean mask des runs de ce gamma
        features_matrix_f4 : colonne f4_trace_J_mean (peut être None)

    Returns:
        True si différentiable, False sinon.
    """
    if features_matrix_f4 is None:
        # Pas de colonne f4 → on suppose différentiable (conservateur)
        warnings.warn(f"[stratify] Pas de colonne f4_trace_J_mean, "
                      f"gamma '{gamma_id}' supposé différentiable")
        return True

    f4_vals = features_matrix_f4[gamma_mask]
    all_nan = np.all(np.isnan(f4_vals))

    if all_nan:
        return False
    return True


# =========================================================================
# CALCUL DES FEATURES APPLICABLES PAR STRATE
# =========================================================================

def _features_for_strate(rank_eff: int, differentiable: bool) -> tuple:
    """Détermine les features applicables et exclues pour une strate.

    Utilise FEATURES_STRUCTURAL_NAN du registre comme source de vérité.

    Returns:
        (features_applicable: List[str], features_excluded: Dict[str, str])
    """
    excluded = {}

    for feature_name, condition_str in FEATURES_STRUCTURAL_NAN.items():
        # Évaluer la condition
        if 'rank_eff == 2' in condition_str and rank_eff == 2:
            excluded[feature_name] = condition_str
        elif 'differentiable == False' in condition_str and not differentiable:
            excluded[feature_name] = condition_str

    excluded_set = set(excluded.keys())
    applicable = [f for f in FEATURE_NAMES if f not in excluded_set]

    return applicable, excluded


# =========================================================================
# POINT D'ENTRÉE — CALCUL DES STRATES
# =========================================================================

def compute_strates(data,  # AnalysingData — pas typé pour éviter import circulaire
                     run_indices: np.ndarray,
                     verbose: bool = True) -> List[Strate]:
    """Découpe la population en strates (rank_eff × differentiable).

    Utilise les METADATA AST des gammas pour déterminer differentiable.
    Fallback heuristique si un gamma est absent du discovery.

    Args:
        data : AnalysingData (population principale, post-triage).
        run_indices : indices des runs dans data (population principale).
        verbose : afficher les stats.

    Returns:
        Liste de Strate. Chaque strate contient les indices des runs
        et la liste des features applicables (FEATURE_NAMES - NaN structurels).
    """
    # --- Propriétés gamma depuis METADATA AST ---
    gamma_ids = data.gamma_ids[run_indices]
    rank_effs = data.rank_effs[run_indices]

    gamma_props = build_gamma_properties(gamma_ids)

    # --- Résoudre differentiable par run ---
    n = len(run_indices)
    differentiable_per_run = np.empty(n, dtype=bool)

    # Identifier les gammas qui nécessitent un fallback heuristique
    needs_fallback = set()
    for i in range(n):
        gid = str(gamma_ids[i])
        props = gamma_props.get(gid, {})
        diff = props.get('differentiable')
        if diff is None:
            needs_fallback.add(gid)
            differentiable_per_run[i] = True  # placeholder
        else:
            differentiable_per_run[i] = diff

    # Fallback heuristique pour les gammas sans METADATA
    if needs_fallback:
        # Matérialiser f4_trace_J_mean pour le fallback
        if 'f4_trace_J_mean' in data.feature_names:
            f4_col = data.materialize_features(
                columns=['f4_trace_J_mean'], rows=run_indices
            ).ravel()
        else:
            f4_col = None

        for gid in needs_fallback:
            gamma_mask = np.array([str(g) == gid for g in gamma_ids])
            is_diff = _infer_differentiable_from_data(gid, gamma_mask, f4_col)
            differentiable_per_run[gamma_mask] = is_diff
            gamma_props[gid]['differentiable'] = is_diff

        del f4_col  # Libérer

    # --- Grouper par (rank_eff, differentiable) ---
    strate_groups = {}  # {(rank_eff, diff): list of local indices}

    for i in range(n):
        key = (int(rank_effs[i]), bool(differentiable_per_run[i]))
        if key not in strate_groups:
            strate_groups[key] = []
        strate_groups[key].append(i)

    # --- Construire les Strate ---
    strates = []

    for (rank, diff), local_indices in sorted(strate_groups.items()):
        local_idx = np.array(local_indices)
        global_idx = run_indices[local_idx]

        diff_str = 'diff' if diff else 'nodiff'
        strate_id = f"rank{rank}_{diff_str}"

        applicable, excluded = _features_for_strate(rank, diff)

        strate = Strate(
            strate_id=strate_id,
            rank_eff=rank,
            differentiable=diff,
            run_indices=global_idx,
            features_applicable=applicable,
            features_excluded=excluded,
        )
        strates.append(strate)

        if verbose:
            print(f"  [stratify] {strate_id} : {strate.n_runs} runs, "
                  f"{strate.n_features} features "
                  f"({len(excluded)} exclues)")

    if verbose:
        total = sum(s.n_runs for s in strates)
        print(f"  [stratify] Total : {len(strates)} strates, {total} runs")

    return strates
