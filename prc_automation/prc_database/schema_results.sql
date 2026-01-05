-- prc_database/schema_results.sql
-- NOUVEAU CONTENU COMPLET

-- =============================================================================
-- SCHEMA DB_RESULTS - CHARTER 5.5
-- =============================================================================

-- Table unique : Observations tests
CREATE TABLE IF NOT EXISTS TestObservations (
  observation_id INTEGER PRIMARY KEY,  -- ✅ PK explicite
  exec_id INTEGER NOT NULL,
  test_name TEXT NOT NULL,
  
  -- Config params utilisée
  params_config_id TEXT NOT NULL,
  
  -- Applicabilité
  applicable BOOLEAN NOT NULL,
  status TEXT NOT NULL,         -- SUCCESS | ERROR | NOT_APPLICABLE
  message TEXT,
  
  -- Statistiques (extraction rapide)
  stat_initial REAL,
  stat_final REAL,
  stat_min REAL,
  stat_max REAL,
  stat_mean REAL,
  stat_std REAL,
  
  -- Évolution (extraction rapide)
  evolution_transition TEXT,
  evolution_trend TEXT,
  evolution_trend_coefficient REAL,
  
  -- Observation complète (backup JSON)
  observation_data TEXT NOT NULL,
  
  computed_at TEXT NOT NULL,
  
  UNIQUE(exec_id, test_name, params_config_id),
  FOREIGN KEY(exec_id) REFERENCES Executions(id)
);

-- Index
CREATE INDEX IF NOT EXISTS idx_obs_exec 
ON TestObservations(exec_id);

CREATE INDEX IF NOT EXISTS idx_obs_test 
ON TestObservations(test_name);

CREATE INDEX IF NOT EXISTS idx_obs_params 
ON TestObservations(params_config_id);

-- =============================================================================
-- FIN SCHEMA
-- =============================================================================