
# File: prc_automation/sequence_test_utils.py

"""
Utilitaires application tests sur séquences R1.

CONFORMITÉ:
- Réutilise patterns batch_runner.py
- Délégation test_engine (pas duplication)
"""

import sqlite3
import gzip
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict

DB_DIR = Path("prc_automation/prc_database")


def load_sequence_context(sequence_exec_id: str, phase: str = 'R1') -> Dict:
    """
    Charge contexte séquence depuis db_raw.
    
    SIMILAIRE: batch_runner.load_execution_context()
    DIFFÉRENCE: Table sequences vs executions
    
    Args:
        sequence_exec_id: UUID séquence
        phase: Phase cible
    
    Returns:
        {
            'sequence_exec_id': str,
            'sequence_gammas': List[str],
            'sequence_length': int,
            'd_encoding_id': str,
            'modifier_id': str,
            'seed': int
        }
    """
    db_path = DB_DIR / f"prc_{phase.lower()}_raw.db"
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT sequence_gammas, sequence_length, 
               d_encoding_id, modifier_id, seed
        FROM sequences
        WHERE sequence_exec_id = ?
    """, (sequence_exec_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise ValueError(f"sequence_exec_id={sequence_exec_id} non trouvé")
    
    import json
    
    return {
        'sequence_exec_id': sequence_exec_id,
        'sequence_gammas': json.loads(row['sequence_gammas']),
        'sequence_length': row['sequence_length'],
        'd_encoding_id': row['d_encoding_id'],
        'modifier_id': row['modifier_id'],
        'seed': row['seed']
    }


def load_sequence_history(sequence_exec_id: str, phase: str = 'R1') -> List[np.ndarray]:
    """
    Charge history complète séquence (états finaux chaque gamma).
    
    DIFFÉRENCE batch_runner.load_execution_history():
    - Charge snapshots_sequences (pas snapshots)
    - Filtre gamma_step + iteration
    
    STRATÉGIE:
    - Pour chaque gamma_step: charger dernier snapshot (iteration max)
    - Ordre chronologique (gamma_step croissant)
    
    Args:
        sequence_exec_id: UUID séquence
        phase: Phase cible
    
    Returns:
        Liste états [state_after_gamma1, state_after_gamma2, ...]
    """
    db_path = DB_DIR / f"prc_{phase.lower()}_raw.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Charger contexte pour sequence_length
    context = load_sequence_context(sequence_exec_id, phase)
    sequence_length = context['sequence_length']
    
    history = []
    
    for gamma_step in range(sequence_length):
        # Charger dernier snapshot de ce gamma_step
        cursor.execute("""
            SELECT state_blob
            FROM snapshots_sequences
            WHERE sequence_exec_id = ? AND gamma_step = ?
            ORDER BY iteration DESC
            LIMIT 1
        """, (sequence_exec_id, gamma_step))
        
        row = cursor.fetchone()
        
        if not row:
            raise ValueError(f"Aucun snapshot pour gamma_step={gamma_step}")
        
        state = pickle.loads(gzip.decompress(row[0]))
        history.append(state)
    
    conn.close()
    
    return history


def load_first_sequence_snapshot(sequence_exec_id: str, phase: str = 'R1') -> np.ndarray:
    """
    Charge premier snapshot séquence (état initial).
    
    SIMILAIRE: batch_runner.load_first_snapshot()
    
    Args:
        sequence_exec_id: UUID séquence
        phase: Phase cible
    
    Returns:
        État initial (avant application gammas)
    """
    db_path = DB_DIR / f"prc_{phase.lower()}_raw.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT state_blob
        FROM snapshots_sequences
        WHERE sequence_exec_id = ? AND gamma_step = 0
        ORDER BY iteration
        LIMIT 1
    """, (sequence_exec_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise ValueError(f"Aucun snapshot initial pour {sequence_exec_id}")
    
    state = pickle.loads(gzip.decompress(row[0]))
    return state


def store_sequence_test_observation(
    db_path: Path,
    phase: str,
    sequence_exec_id: str,
    sequence_gammas: List[str],
    sequence_length: int,
    d_encoding_id: str,
    modifier_id: str,
    seed: int,
    observation: Dict
) -> None:
    """
    Stocke observation test séquence dans db_results.
    
    SIMILAIRE: batch_runner.store_test_observation()
    DIFFÉRENCE: Colonnes sequence_* (vs gamma_id)
    
    Args:
        observation: Retour TestEngine.execute_test()
    """
    import json
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Extraire projections rapides (identique R0)
    stats = observation.get('statistics', {})
    first_metric = list(stats.keys())[0] if stats else None
    
    if first_metric:
        stat_data = stats[first_metric]
        stat_initial = stat_data.get('initial')
        stat_final = stat_data.get('final')
        stat_mean = stat_data.get('mean')
        stat_std = stat_data.get('std')
    else:
        stat_initial = stat_final = stat_mean = stat_std = None
    
    evol = observation.get('evolution', {})
    first_evol = list(evol.keys())[0] if evol else None
    
    if first_evol:
        evol_data = evol[first_evol]
        evolution_slope = evol_data.get('slope')
        evolution_relative_change = evol_data.get('relative_change')
    else:
        evolution_slope = evolution_relative_change = None
    
    cursor.execute("""
        INSERT OR REPLACE INTO observations (
            test_name, 
            sequence_exec_id, sequence_gammas, sequence_length,
            gamma_id,
            d_encoding_id, modifier_id, seed, phase,
            exec_id, timestamp, test_category, params_config_id,
            status, message, observation_data,
            stat_initial, stat_final, stat_mean, stat_std,
            evolution_slope, evolution_relative_change
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        observation['test_name'],
        sequence_exec_id,
        json.dumps(sequence_gammas),
        sequence_length,
        None,  # gamma_id NULL pour séquences
        d_encoding_id,
        modifier_id,
        seed,
        phase,
        sequence_exec_id,  # exec_id = sequence_exec_id pour traçabilité
        observation.get('timestamp', ''),
        observation['test_category'],
        observation['config_params_id'],
        observation['status'],
        observation.get('message', ''),
        json.dumps(observation),
        stat_initial, stat_final, stat_mean, stat_std,
        evolution_slope, evolution_relative_change
    ))
    
    conn.commit()
    conn.close()