# tests/utilities/data_loading.py
"""
Data Loading Utilities - I/O Observations depuis DBs.

RESPONSABILITÉS :
- Connexions DB (prc_r0_results.db + prc_r0_raw.db)
- Fusion observations + métadonnées runs
- Conversion observations → DataFrame normalisé
- Cache observations (futur)

ARCHITECTURE :
- load_all_observations() : Double connexion (TestObservations + Executions)
- observations_to_dataframe() : Normalisation pour analyses stats

UTILISATEURS :
- verdict_engine.py (compute_verdict)
- verdict_reporter.py (generate_verdict_report)
- Futurs : modifier_profiling.py, test_profiling.py
"""

import sqlite3
import json
import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path


def load_all_observations(
    params_config_id: str,
    db_results_path: str = './prc_automation/prc_database/prc_r0_results.db',
    db_raw_path: str = './prc_automation/prc_database/prc_r0_raw.db'
) -> List[Dict]:
    """
    Charge observations SUCCESS avec métadonnées runs.
    
    DOUBLE CONNEXION :
    - db_results : TestObservations (observation_data, status)
    - db_raw : Executions (gamma_id, d_encoding_id, modifier_id, seed)
    
    Args:
        params_config_id: ID config params (ex: 'params_default_v1')
        db_results_path: Chemin DB résultats tests
        db_raw_path: Chemin DB exécutions brutes
    
    Returns:
        List[dict]: Observations avec métadonnées fusionnées
        Format :
        {
            'observation_id': int,
            'exec_id': int,
            'run_id': str,
            'gamma_id': str,
            'd_encoding_id': str,
            'modifier_id': str,
            'seed': int,
            'test_name': str,
            'params_config_id': str,
            'observation_data': dict,
            'computed_at': str
        }
    
    Raises:
        ValueError: Si aucune observation SUCCESS trouvée
    
    Examples:
        >>> obs = load_all_observations('params_default_v1')
        >>> len(obs)
        4320
        >>> obs[0]['gamma_id']
        'GAM-001'
    """
    # 1. Charger observations depuis db_results
    conn_results = sqlite3.connect(db_results_path)
    conn_results.row_factory = sqlite3.Row
    cursor_results = conn_results.cursor()
    
    cursor_results.execute("""
        SELECT 
            observation_id,
            exec_id,
            test_name,
            params_config_id,
            status,
            observation_data,
            computed_at
        FROM TestObservations
        WHERE params_config_id = ?
          AND status = 'SUCCESS'
    """, (params_config_id,))
    
    obs_rows = cursor_results.fetchall()
    conn_results.close()
    
    if not obs_rows:
        raise ValueError(
            f"Aucune observation SUCCESS pour params={params_config_id}"
        )
    
    # 2. Extraire exec_ids uniques
    exec_ids = list(set(row['exec_id'] for row in obs_rows))
    
    # 3. Charger métadonnées Executions depuis db_raw
    conn_raw = sqlite3.connect(db_raw_path)
    conn_raw.row_factory = sqlite3.Row
    cursor_raw = conn_raw.cursor()
    
    placeholders = ','.join('?' * len(exec_ids))
    cursor_raw.execute(f"""
        SELECT 
            id,
            run_id,
            gamma_id,
            d_encoding_id,
            modifier_id,
            seed
        FROM Executions
        WHERE id IN ({placeholders})
    """, exec_ids)
    
    exec_rows = cursor_raw.fetchall()
    conn_raw.close()
    
    # 4. Index executions par id
    executions_by_id = {
        row['id']: {
            'run_id': row['run_id'],
            'gamma_id': row['gamma_id'],
            'd_encoding_id': row['d_encoding_id'],
            'modifier_id': row['modifier_id'],
            'seed': row['seed']
        }
        for row in exec_rows
    }
    
    # 5. Fusionner observations + métadonnées
    observations = []
    for row in obs_rows:
        exec_id = row['exec_id']
        
        if exec_id not in executions_by_id:
            print(f"⚠️ Skip observation {row['observation_id']}: "
                  f"exec_id={exec_id} introuvable dans db_raw")
            continue
        
        exec_meta = executions_by_id[exec_id]
        
        try:
            obs_data = json.loads(row['observation_data'])
            
            observations.append({
                'observation_id': row['observation_id'],
                'exec_id': exec_id,
                'run_id': exec_meta['run_id'],
                'gamma_id': exec_meta['gamma_id'],
                'd_encoding_id': exec_meta['d_encoding_id'],
                'modifier_id': exec_meta['modifier_id'],
                'seed': exec_meta['seed'],
                'test_name': row['test_name'],
                'params_config_id': row['params_config_id'],
                'observation_data': obs_data,
                'computed_at': row['computed_at']
            })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Skip observation {row['observation_id']}: {e}")
            continue
    
    return observations


def observations_to_dataframe(observations: List[Dict]) -> pd.DataFrame:
    """
    Convertit observations → DataFrame normalisé pour analyses stats.
    
    PROJECTIONS EXTRAITES :
    - value_final, value_initial, value_mean, value_std, value_min, value_max
    - slope, volatility, relative_change
    - transition, trend (catégorielles)
    
    Args:
        observations: Liste observations (retour load_all_observations)
    
    Returns:
        DataFrame avec colonnes :
        - Identifiants : gamma_id, d_encoding_id, modifier_id, seed, 
                        test_name, params_config_id, metric_name
        - Projections numériques : value_*, slope, volatility, relative_change
        - Catégorielles : transition, trend
    
    Notes:
        - Filtre lignes avec NaN dans TOUTES projections numériques
        - Une ligne par (observation, metric)
    
    Examples:
        >>> df = observations_to_dataframe(obs)
        >>> df.columns
        ['gamma_id', 'test_name', 'value_final', 'slope', ...]
        >>> df.shape
        (8640, 17)  # 4320 obs × 2 métriques moyennes
    """
    rows = []
    
    for obs in observations:
        gamma_id = obs['gamma_id']
        d_encoding_id = obs['d_encoding_id']
        modifier_id = obs['modifier_id']
        seed = obs['seed']
        test_name = obs['test_name']
        params_config_id = obs['params_config_id']
        
        obs_data = obs['observation_data']
        
        if 'statistics' not in obs_data or 'evolution' not in obs_data:
            continue
        
        stats = obs_data['statistics']
        evolution = obs_data['evolution']
        
        for metric_name in stats.keys():
            if metric_name not in evolution:
                continue
            
            metric_stats = stats[metric_name]
            metric_evol = evolution[metric_name]
            
            rows.append({
                # Identifiants
                'gamma_id': gamma_id,
                'd_encoding_id': d_encoding_id,
                'modifier_id': modifier_id,
                'seed': seed,
                'test_name': test_name,
                'params_config_id': params_config_id,
                'metric_name': metric_name,
                
                # Projections numériques
                'value_final': metric_stats.get('final', np.nan),
                'value_initial': metric_stats.get('initial', np.nan),
                'value_mean': metric_stats.get('mean', np.nan),
                'value_std': metric_stats.get('std', np.nan),
                'value_min': metric_stats.get('min', np.nan),
                'value_max': metric_stats.get('max', np.nan),
                
                'slope': metric_evol.get('slope', np.nan),
                'volatility': metric_evol.get('volatility', np.nan),
                'relative_change': metric_evol.get('relative_change', np.nan),
                
                # Catégorielles
                'transition': metric_evol.get('transition', 'unknown'),
                'trend': metric_evol.get('trend', 'unknown'),
            })
    
    df = pd.DataFrame(rows)
    
    # Nettoyer NaN (lignes sans aucune projection valide)
    numeric_cols = [
        'value_final', 'value_initial', 'value_mean', 'value_std',
        'slope', 'volatility', 'relative_change'
    ]
    df = df.dropna(subset=numeric_cols, how='all')
    
    return df


def cache_observations(
    observations: List[Dict],
    cache_path: str = './cache/observations.pkl'
) -> None:
    """
    Cache observations sur disque (pickle).
    
    FUTUR : Optimisation chargement répété.
    
    Args:
        observations: Liste observations
        cache_path: Chemin cache
    
    Examples:
        >>> cache_observations(obs, './cache/obs_params_v1.pkl')
    """
    import pickle
    
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(cache_file, 'wb') as f:
        pickle.dump(observations, f)
    
    print(f"✓ Cache observations : {cache_path}")


def load_cached_observations(cache_path: str) -> List[Dict]:
    """
    Charge observations depuis cache.
    
    Args:
        cache_path: Chemin cache
    
    Returns:
        Liste observations
    
    Raises:
        FileNotFoundError: Si cache absent
    
    Examples:
        >>> obs = load_cached_observations('./cache/obs_params_v1.pkl')
    """
    import pickle
    
    cache_file = Path(cache_path)
    
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache non trouvé : {cache_path}")
    
    with open(cache_file, 'rb') as f:
        observations = pickle.load(f)
    
    print(f"✓ Chargé cache : {cache_path} ({len(observations)} observations)")
    return observations