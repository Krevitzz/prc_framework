-- File: prc_automation/prc_database/schema_sequences_r1.sql

-- =============================================================================
-- SCHEMA DB_RAW R1 - Extension séquences composition
-- =============================================================================

CREATE TABLE IF NOT EXISTS sequences (
    -- Identité séquence
    sequence_exec_id TEXT NOT NULL UNIQUE,  -- UUID
    sequence_gammas TEXT NOT NULL,          -- JSON ['GAM-001', 'GAM-002']
    sequence_length INTEGER NOT NULL,
    
    -- Contexte exécution
    d_encoding_id TEXT NOT NULL,
    modifier_id TEXT NOT NULL,
    seed INTEGER NOT NULL,
    phase TEXT NOT NULL DEFAULT 'R1',
    
    -- Métadonnées
    timestamp TEXT NOT NULL,
    state_shape TEXT NOT NULL,              -- JSON "[10, 10]"
    n_iterations_per_gamma TEXT NOT NULL,   -- JSON [200, 200, ...]
    status TEXT NOT NULL,                   -- 'SUCCESS' | 'ERROR'
    error_message TEXT,
    
    -- Contraintes
    PRIMARY KEY (sequence_gammas, d_encoding_id, modifier_id, seed, phase),
    CHECK (status IN ('SUCCESS', 'ERROR')),
    CHECK (sequence_length >= 2 AND sequence_length <= 5)
);

-- Index performance
CREATE INDEX idx_seq_length ON sequences(sequence_length, phase);
CREATE INDEX idx_seq_encoding ON sequences(d_encoding_id, phase);
CREATE INDEX idx_seq_status ON sequences(status, phase);
CREATE INDEX idx_seq_uuid ON sequences(sequence_exec_id);

-- =============================================================================
-- Table snapshots_sequences : États intermédiaires + final
-- =============================================================================

CREATE TABLE IF NOT EXISTS snapshots_sequences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence_exec_id TEXT NOT NULL,
    gamma_step INTEGER NOT NULL,           -- Position gamma dans séquence (0-indexed)
    gamma_id TEXT NOT NULL,                -- Gamma appliqué à ce step
    iteration INTEGER NOT NULL,            -- Itération dans gamma_step
    
    -- État compressé
    state_blob BLOB,
    
    -- Métriques rapides
    norm_frobenius REAL,
    norm_spectral REAL,
    min_value REAL,
    max_value REAL,
    mean_value REAL,
    std_value REAL,
    
    UNIQUE(sequence_exec_id, gamma_step, iteration),
    FOREIGN KEY (sequence_exec_id) REFERENCES sequences(sequence_exec_id)
);

CREATE INDEX idx_snapseq_exec ON snapshots_sequences(sequence_exec_id);
CREATE INDEX idx_snapseq_step ON snapshots_sequences(gamma_step);