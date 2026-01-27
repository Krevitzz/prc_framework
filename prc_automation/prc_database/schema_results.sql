-- =============================================================================
-- SCHEMA DB_RESULTS V2 - CHARTER PHASE 10
-- Identité complète sans indirection
-- =============================================================================

-- Table unique : Observations tests
CREATE TABLE IF NOT EXISTS observations (
    -- Identité complète (clé primaire composite)
    test_name TEXT NOT NULL,
    gamma_id TEXT NOT NULL,
    d_encoding_id TEXT NOT NULL,
    modifier_id TEXT NOT NULL,
    seed INTEGER NOT NULL,
    phase TEXT NOT NULL DEFAULT 'R0',
    
    -- Métadonnées test
    exec_id TEXT NOT NULL,         -- Traçabilité (pas FK)
    timestamp TEXT NOT NULL,
    test_category TEXT NOT NULL,
    params_config_id TEXT NOT NULL,
    status TEXT NOT NULL,          -- 'SUCCESS' | 'ERROR' | 'NOT_APPLICABLE'
    message TEXT,
    
    -- Résultats test (JSON complet)
    observation_data TEXT NOT NULL,  -- JSON structure complète
    
    -- Projections rapides (extraction première métrique - OPTIONNEL)
    stat_initial REAL,
    stat_final REAL,
    stat_mean REAL,
    stat_std REAL,
    evolution_slope REAL,
    evolution_relative_change REAL,
    
    -- Contraintes
    PRIMARY KEY (test_name, gamma_id, d_encoding_id, modifier_id, seed, phase),
    CHECK (status IN ('SUCCESS', 'ERROR', 'NOT_APPLICABLE'))
);

-- Index performance
CREATE INDEX idx_obs_test ON observations(test_name, phase);
CREATE INDEX idx_obs_gamma ON observations(gamma_id, phase);
CREATE INDEX idx_obs_encoding ON observations(d_encoding_id, phase);
CREATE INDEX idx_obs_modifier ON observations(modifier_id, phase);
CREATE INDEX idx_obs_status ON observations(status, phase);
CREATE INDEX idx_obs_params ON observations(params_config_id, phase);
CREATE INDEX idx_obs_uuid ON observations(exec_id);

-- =============================================================================
-- FIN SCHEMA
-- =============================================================================