-- =============================================================================
-- SCHEMA DB_RAW V2 - CHARTER PHASE 10
-- Identité complète sans indirection
-- =============================================================================

-- Table principale : Exécutions kernel
CREATE TABLE IF NOT EXISTS executions (
    -- Identité complète (clé primaire composite)
    gamma_id TEXT NOT NULL,
    d_encoding_id TEXT NOT NULL,
    modifier_id TEXT NOT NULL,
    seed INTEGER NOT NULL,
    phase TEXT NOT NULL DEFAULT 'R0',
    
    -- Métadonnées exécution
    exec_id TEXT NOT NULL UNIQUE,  -- UUID traçabilité
    timestamp TEXT NOT NULL,
    state_shape TEXT NOT NULL,     -- JSON "[10, 10]"
    n_iterations INTEGER NOT NULL,
    status TEXT NOT NULL,          -- 'SUCCESS' | 'ERROR' | 'TIMEOUT'
    error_message TEXT,            -- Message erreur si status='ERROR'
    
    -- Paramètres gamma (JSON blob)
    gamma_params TEXT,             -- JSON {"alpha": 0.1, "beta": 2.0, ...}
    
    -- Contraintes
    PRIMARY KEY (gamma_id, d_encoding_id, modifier_id, seed, phase),
    CHECK (status IN ('SUCCESS', 'ERROR', 'TIMEOUT'))
);

-- Index performance
CREATE INDEX idx_exec_gamma ON executions(gamma_id, phase);
CREATE INDEX idx_exec_encoding ON executions(d_encoding_id, phase);
CREATE INDEX idx_exec_modifier ON executions(modifier_id, phase);
CREATE INDEX idx_exec_status ON executions(status, phase);
CREATE INDEX idx_exec_uuid ON executions(exec_id);

-- =============================================================================
-- Table snapshots : États sauvegardés
-- =============================================================================

CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exec_id TEXT NOT NULL,  -- Référence executions.exec_id (UUID)
    iteration INTEGER NOT NULL,
    
    -- État compressé (gzip + pickle)
    state_blob BLOB,
    
    -- Métriques rapides (queries sans décompresser)
    norm_frobenius REAL,
    norm_spectral REAL,
    min_value REAL,
    max_value REAL,
    mean_value REAL,
    std_value REAL,
    
    UNIQUE(exec_id, iteration)
);

CREATE INDEX idx_snapshot_exec ON snapshots(exec_id);

-- =============================================================================
-- FIN SCHEMA
-- =============================================================================