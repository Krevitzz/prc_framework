-- File: prc_automation/prc_database/schema_results_r1.sql

-- =============================================================================
-- SCHEMA DB_RESULTS R1 - Extension observations séquences
-- =============================================================================

-- Table observations identique R0, colonnes additionnelles
CREATE TABLE IF NOT EXISTS observations (
    -- Identité test
    test_name TEXT NOT NULL,
    
    -- NOUVEAU R1: Séquence ou gamma simple
    sequence_exec_id TEXT,              -- UUID séquence (NULL si R0)
    sequence_gammas TEXT,               -- JSON ['GAM-001', 'GAM-002'] (NULL si R0)
    sequence_length INTEGER,            -- 2, 3, 4, 5 (NULL si R0)
    
    -- Contexte (compatible R0)
    gamma_id TEXT,                      -- Single gamma si R0, NULL si R1
    d_encoding_id TEXT NOT NULL,
    modifier_id TEXT NOT NULL,
    seed INTEGER NOT NULL,
    phase TEXT NOT NULL DEFAULT 'R1',
    
    -- Métadonnées test
    exec_id TEXT,                       -- Traçabilité (deprecated R1, kept for compatibility)
    timestamp TEXT NOT NULL,
    test_category TEXT NOT NULL,
    params_config_id TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT,
    
    -- Résultats test (JSON)
    observation_data TEXT NOT NULL,
    
    -- Projections rapides (identique R0)
    stat_initial REAL,
    stat_final REAL,
    stat_mean REAL,
    stat_std REAL,
    evolution_slope REAL,
    evolution_relative_change REAL,
    
    -- Contraintes
    CHECK (status IN ('SUCCESS', 'ERROR', 'NOT_APPLICABLE')),
    CHECK ((sequence_exec_id IS NOT NULL AND gamma_id IS NULL) OR 
           (sequence_exec_id IS NULL AND gamma_id IS NOT NULL)),
    
    -- Unicité selon contexte (R0 vs R1)
    UNIQUE (test_name, sequence_exec_id, d_encoding_id, modifier_id, seed, phase),
    UNIQUE (test_name, gamma_id, d_encoding_id, modifier_id, seed, phase)
);

-- Index performance
CREATE INDEX IF NOT EXISTS idx_obs_r1_sequence ON observations(sequence_exec_id, phase);
CREATE INDEX IF NOT EXISTS idx_obs_r1_seq_length ON observations(sequence_length, phase);
CREATE INDEX IF NOT EXISTS idx_obs_r1_test ON observations(test_name, phase);
CREATE INDEX IF NOT EXISTS idx_obs_r1_status ON observations(status, phase);

-- =============================================================================
-- FIN SCHEMA
-- =============================================================================