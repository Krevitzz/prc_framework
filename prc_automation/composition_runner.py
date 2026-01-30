# File: prc_automation/composition_runner.py

"""
Composition Runner R1 - Orchestration séquences Γ₁→Γ₂→...→Γₙ

Architecture:
1. Génération séquences (permutations contrôlées)
2. Exécution séquentielle gammas
3. Stockage états intermédiaires + final
4. Traçabilité complète (sequence_exec_id)

CONFORMITÉ:
- Charter 6.1 Section 5.3 (HUB délégation stricte)
- Réutilise patterns batch_runner.py
- Délègue core/ (prepare_state, run_kernel)

Usage:
    from prc_automation.composition_runner import run_batch_composition
    
    sequences = [['GAM-001', 'GAM-002'], ['GAM-002', 'GAM-001']]
    exec_ids = run_batch_composition(
        sequences, 
        ['SYM-001'], 
        ['M0'], 
        [42],
        phase='R1'
    )
"""

import numpy as np
import sqlite3
import json
import gzip
import pickle
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from itertools import product

# Imports Core (aveugles)
from core.kernel import run_kernel
from core.state_preparation import prepare_state

# Imports Discovery
from tests.utilities.utils.data_loading import discover_entities


# =============================================================================
# CONFIGURATION
# =============================================================================

MAX_ITERATIONS_PER_GAMMA = 2000
SNAPSHOT_INTERVAL = 10
DB_DIR = Path("./prc_automation/prc_database")


def get_db_path(phase: str = 'R1') -> Path:
    """Retourne chemin db_raw pour phase."""
    return DB_DIR / f"prc_{phase.lower()}_raw.db"


# =============================================================================
# GÉNÉRATION SÉQUENCES
# =============================================================================

def generate_sequences(
    gamma_ids: List[str],
    n: int,
    allow_repetition: bool = True
) -> List[List[str]]:
    """
    Génère séquences longueur n (permutations avec répétition).
    
    Args:
        gamma_ids: Liste gammas disponibles
        n: Longueur séquence (2-5)
        allow_repetition: Autoriser Γᵢ→Γᵢ (défaut: True)
    
    Returns:
        Liste séquences [['GAM-001', 'GAM-002'], ...]
    
    Raises:
        ValueError: Si n hors range [2, 5]
    
    Examples:
        >>> generate_sequences(['GAM-001', 'GAM-002'], n=2)
        [
            ['GAM-001', 'GAM-001'],
            ['GAM-001', 'GAM-002'],
            ['GAM-002', 'GAM-001'],
            ['GAM-002', 'GAM-002']
        ]
    """
    if not (2 <= n <= 5):
        raise ValueError(f"n doit être dans [2, 5], reçu {n}")
    
    # Permutations avec répétition
    sequences = list(product(gamma_ids, repeat=n))
    
    # Filtrer répétitions si nécessaire
    if not allow_repetition:
        sequences = [seq for seq in sequences if len(set(seq)) == n]
    
    # Convertir tuples → listes
    sequences = [list(seq) for seq in sequences]
    
    print(f"✓ Généré {len(sequences)} séquences longueur n={n}")
    
    return sequences


# =============================================================================
# EXÉCUTION SÉQUENCE
# =============================================================================

def run_composition_sequence(
    sequence: List[str],
    d_encoding_id: str,
    modifier_id: str,
    seed: int,
    active_entities: Dict[str, List[Dict]],
    phase: str = 'R1'
) -> str:
    """
    Exécute séquence composition gamma.
    
    WORKFLOW:
    1. Préparer D_final (encoding + modifiers)
    2. Boucle sur chaque gamma de la séquence:
       a. Appliquer gamma_i sur state
       b. Sauvegarder snapshots intermédiaires
       c. state devient input gamma_{i+1}
    3. Insérer metadata + snapshots dans DB
    
    Args:
        sequence: Liste gamma_ids ['GAM-001', 'GAM-002', ...]
        d_encoding_id: ID encoding
        modifier_id: ID modifier
        seed: Graine reproductibilité
        active_entities: Résultat discover_entities() (pour résolution factories)
        phase: Phase cible
    
    Returns:
        sequence_exec_id (UUID)
    
    Raises:
        ValueError: Si gamma non trouvé ou non applicable
    """
    # TODO: Implémenter
        
    print(f"[SEQ] {' → '.join(sequence)} | {d_encoding_id} | {modifier_id} | seed={seed}")
    
    # Index entités
    gammas_by_id = {g['id']: g for g in active_entities['gammas']}
    encodings_by_id = {e['id']: e for e in active_entities['encodings']}
    modifiers_by_id = {m['id']: m for m in active_entities['modifiers']}
    
    # Vérifier tous gammas existent
    for gamma_id in sequence:
        if gamma_id not in gammas_by_id:
            raise ValueError(f"Gamma {gamma_id} non trouvé dans active_entities")
    
    # 1. Préparer D_final (identique batch_runner.py)
    encoding_info = encodings_by_id[d_encoding_id]
    modifier_info = modifiers_by_id.get(modifier_id)
    
    encoding_module = encoding_info['module']
    encoding_func = getattr(encoding_module, encoding_info['function_name'])
    
    modifier_module = modifier_info['module']
    modifier_func = getattr(modifier_module, modifier_info['function_name'])
    
    encoding_params = {'n_dof': 100}
    modifiers_list = [modifier_func]
    modifier_configs = {modifier_func: {}}
    
    D_final = prepare_state(
        encoding_func=encoding_func,
        encoding_params=encoding_params,
        modifiers=modifiers_list,
        modifier_configs=modifier_configs,
        seed=seed
    )
    
    # 2. Boucle séquence gammas
    state = D_final.copy()
    snapshots_per_gamma = []
    n_iterations_per_gamma = []
    
    for gamma_step, gamma_id in enumerate(sequence):
        print(f"  [STEP {gamma_step}] Applying {gamma_id}...")
        
        # Créer gamma
        gamma_info = gammas_by_id[gamma_id]
        gamma_module = gamma_info['module']
        factory = getattr(gamma_module, gamma_info['function_name'])
        gamma = factory(seed=seed)
        
        # Reset si non-markovien
        if hasattr(gamma, 'reset'):
            gamma.reset()
        
        # Exécuter gamma
        snapshots_gamma = []
        
        for iteration, state_iter in run_kernel(
            state, gamma,
            max_iterations=MAX_ITERATIONS_PER_GAMMA,
            record_history=False
        ):
            # Sauvegarder snapshot
            if iteration % SNAPSHOT_INTERVAL == 0:
                snap = {
                    'iteration': iteration,
                    'state': state_iter.copy(),
                    'norm_frobenius': float(np.linalg.norm(state_iter.flatten())),
                    'min_value': float(np.min(state_iter)),
                    'max_value': float(np.max(state_iter)),
                    'mean_value': float(np.mean(state_iter)),
                    'std_value': float(np.std(state_iter)),
                }
                
                # Norme spectrale si rang 2 carré
                if state_iter.ndim == 2 and state_iter.shape[0] == state_iter.shape[1]:
                    try:
                        eigs = np.linalg.eigvalsh(state_iter)
                        snap['norm_spectral'] = float(np.max(np.abs(eigs)))
                    except:
                        snap['norm_spectral'] = None
                else:
                    snap['norm_spectral'] = None
                
                snapshots_gamma.append(snap)
            
            # Détection explosion
            if np.any(np.isnan(state_iter)) or np.any(np.isinf(state_iter)):
                print(f"    ⚠ Explosion détectée: iter={iteration}")
                break
        
        final_iteration = iteration
        n_iterations_per_gamma.append(final_iteration)
        snapshots_per_gamma.append(snapshots_gamma)
        
        # État devient input prochain gamma
        state = state_iter.copy()
        
        print(f"    ✓ {final_iteration} iterations, {len(snapshots_gamma)} snapshots")
    
    # 3. Insérer dans DB
    db_path = get_db_path(phase)
    conn = sqlite3.connect(db_path)
    
    try:
        sequence_exec_id = insert_execution_sequence(
            conn, phase,
            sequence, d_encoding_id, modifier_id, seed,
            D_final.shape, n_iterations_per_gamma,
            'SUCCESS', snapshots_per_gamma
        )
        
        print(f"  ✓ sequence_exec_id={sequence_exec_id}")
        
        return sequence_exec_id
    
    except Exception as e:
        print(f"  ✗ Erreur insertion: {e}")
        raise
    
    finally:
        conn.close()
    pass


def run_batch_composition(
    sequences: List[List[str]],
    encodings: List[str],
    modifiers: List[str],
    seeds: List[int],
    phase: str = 'R1'
) -> List[str]:
    """
    Exécute batch séquences (orchestration).
    
    DÉLÉGATION:
    - run_composition_sequence() × N
    
    Args:
        sequences: Liste séquences
        encodings: Liste d_encoding_ids
        modifiers: Liste modifier_ids
        seeds: Liste seeds
        phase: Phase cible
    
    Returns:
        Liste sequence_exec_ids SUCCESS uniquement
    """
    # TODO: Implémenter
        
    print(f"\n{'='*70}")
    print(f"BATCH COMPOSITION - {len(sequences)} séquences")
    print(f"{'='*70}\n")
    
    # Découvrir entités
    print("1. Découverte entités...")
    active_entities = {
        'gammas': discover_entities('gamma', phase),
        'encodings': discover_entities('encoding', phase),
        'modifiers': discover_entities('modifier', phase),
    }
    print(f"   ✓ Gammas:    {len(active_entities['gammas'])}")
    print(f"   ✓ Encodings: {len(active_entities['encodings'])}")
    print(f"   ✓ Modifiers: {len(active_entities['modifiers'])}")
    
    # Exécuter séquences
    print("\n2. Exécution séquences...")
    sequence_exec_ids = []
    completed = 0
    errors = 0
    
    total_runs = len(sequences) * len(encodings) * len(modifiers) * len(seeds)
    current = 0
    
    for sequence in sequences:
        for encoding_id in encodings:
            for modifier_id in modifiers:
                for seed in seeds:
                    current += 1
                    print(f"\n[{current}/{total_runs}] SEQ={' → '.join(sequence[:2])}{'...' if len(sequence) > 2 else ''} | {encoding_id} | {modifier_id} | s{seed}")
                    
                    try:
                        seq_exec_id = run_composition_sequence(
                            sequence, encoding_id, modifier_id, seed,
                            active_entities, phase
                        )
                        sequence_exec_ids.append(seq_exec_id)
                        completed += 1
                    
                    except Exception as e:
                        print(f"  ✗ Erreur: {e}")
                        errors += 1
    
    print(f"\n{'='*70}")
    print(f"RÉSUMÉ BATCH COMPOSITION")
    print(f"{'='*70}")
    print(f"Complétés: {completed}")
    print(f"Erreurs:   {errors}")
    print(f"{'='*70}\n")
    
    return sequence_exec_ids
    pass


# =============================================================================
# INSERTION DB
# =============================================================================

def insert_execution_sequence(
    conn: sqlite3.Connection,
    phase: str,
    sequence: List[str],
    d_encoding_id: str,
    modifier_id: str,
    seed: int,
    state_shape: tuple,
    n_iterations_per_gamma: List[int],
    status: str,
    snapshots_per_gamma: List[List[Dict]],
    error_message: str = None
) -> str:
    """
    Insère séquence dans db_raw.
    
    SCHEMA:
    - Table sequences (metadata)
    - Table snapshots_sequences (états intermédiaires)
    
    Args:
        snapshots_per_gamma: [
            [snap_gamma1_iter0, snap_gamma1_iter10, ...],
            [snap_gamma2_iter0, snap_gamma2_iter10, ...],
            ...
        ]
    
    Returns:
        sequence_exec_id (UUID)
    """
    # TODO: Implémenter
        
    sequence_exec_id = str(uuid.uuid4())
    cursor = conn.cursor()
    
    try:
        # 1. Insertion table sequences
        cursor.execute("""
            INSERT INTO sequences (
                sequence_exec_id, sequence_gammas, sequence_length,
                d_encoding_id, modifier_id, seed, phase,
                timestamp, state_shape, n_iterations_per_gamma, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sequence_exec_id,
            json.dumps(sequence),
            len(sequence),
            d_encoding_id,
            modifier_id,
            seed,
            phase,
            datetime.now().isoformat(),
            json.dumps(list(state_shape)),
            json.dumps(n_iterations_per_gamma),
            status
        ))
        
        # 2. Insertion snapshots_sequences
        snapshots_inserted = 0
        
        for gamma_step, snapshots_gamma in enumerate(snapshots_per_gamma):
            gamma_id = sequence[gamma_step]
            
            for snap in snapshots_gamma:
                try:
                    state_compressed = gzip.compress(pickle.dumps(snap['state']))
                    
                    cursor.execute("""
                        INSERT INTO snapshots_sequences (
                            sequence_exec_id, gamma_step, gamma_id, iteration,
                            state_blob, norm_frobenius, norm_spectral,
                            min_value, max_value, mean_value, std_value
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        sequence_exec_id,
                        gamma_step,
                        gamma_id,
                        snap['iteration'],
                        state_compressed,
                        snap['norm_frobenius'],
                        snap.get('norm_spectral'),
                        snap['min_value'],
                        snap['max_value'],
                        snap['mean_value'],
                        snap['std_value']
                    ))
                    snapshots_inserted += 1
                
                except Exception as e:
                    print(f"      ⚠ Erreur snapshot gamma_step={gamma_step}, iter={snap['iteration']}: {e}")
                    continue
        
        # Validation
        if snapshots_inserted == 0 and len(snapshots_per_gamma) > 0:
            conn.rollback()
            raise ValueError("Aucun snapshot inséré")
        
        # Commit
        conn.commit()
        
        print(f"    ✓ {snapshots_inserted} snapshots insérés")
        
        return sequence_exec_id
    
    except Exception as e:
        conn.rollback()
        print(f"    ✗ Erreur insertion: {e}")
        raise
    pass


# =============================================================================
# CLI (optionnel, pour tests manuels)
# =============================================================================

def main():
    """Point d'entrée CLI (test manuel)."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Composition Runner R1")
    parser.add_argument('--n', type=int, default=2, help="Longueur séquences")
    parser.add_argument('--phase', default='R1', help="Phase cible")
    
    args = parser.parse_args()
    
    # Charger gammas sélectionnés
    with open("outputs/phase0/gammas_selected_r1_1.json") as f:
        selected = json.load(f)
    
    gamma_ids = selected['gamma_ids']
    
    # Générer séquences
    sequences = generate_sequences(gamma_ids, n=args.n)
    
    print(f"✓ {len(sequences)} séquences générées")
    print(f"  Exemple: {sequences[0]}")


if __name__ == "__main__":
    main()