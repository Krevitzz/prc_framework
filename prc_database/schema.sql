-- prc_database/schema.sql
-- 
-- Schéma de base de données pour stockage résultats exploration R0
-- Conforme à la Feuille de Route Section 1.3

-- ============================================================================
-- TABLE PRINCIPALE: Exécutions atomiques
-- ============================================================================

CREATE TABLE IF NOT EXISTS Executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Identifiants
    gamma_id TEXT NOT NULL,                    -- "GAM-001", "GAM-002", etc.
    gamma_params TEXT NOT NULL,                -- JSON: {"beta": 2.0, "alpha": 0.3}
    d_base_id TEXT NOT NULL,                   -- "SYM-001", "ASY-002", "R3-001"
    modifier_id TEXT NOT NULL,                 -- "M0", "M1", "M2", "M3"
    seed INTEGER NOT NULL,
    
    -- Configuration exécution
    max_iterations INTEGER NOT NULL,
    convergence_threshold REAL,
    
    -- Métadonnées temporelles
    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    execution_time_seconds REAL,
    
    -- Status exécution
    status TEXT NOT NULL CHECK(status IN ('COMPLETED', 'ERROR', 'TIMEOUT')),
    error_message TEXT,
    
    -- Résultats agrégés
    final_iteration INTEGER,
    converged BOOLEAN,
    global_verdict TEXT CHECK(global_verdict IN ('PASS', 'POOR', 'REJECTED', 'NEUTRAL')),
    
    -- Contrainte unicité
    UNIQUE(gamma_id, gamma_params, d_base_id, modifier_id, seed)
);

-- Index pour requêtes fréquentes
CREATE INDEX IF NOT EXISTS idx_exec_gamma 
    ON Executions(gamma_id, gamma_params);

CREATE INDEX IF NOT EXISTS idx_exec_d 
    ON Executions(d_base_id);

CREATE INDEX IF NOT EXISTS idx_exec_status 
    ON Executions(status);

CREATE INDEX IF NOT EXISTS idx_exec_verdict 
    ON Executions(global_verdict);

-- ============================================================================
-- TABLE RÉSULTATS TESTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS TestResults (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exec_id INTEGER NOT NULL,
    
    -- Identification test
    test_name TEXT NOT NULL,                   -- "TEST-UNIV-001", "TEST-SYM-001", etc.
    test_category TEXT NOT NULL,               -- "UNIV", "SYM", "STR", "BND", etc.
    
    -- Résultat
    status TEXT NOT NULL CHECK(status IN ('PASS', 'FAIL', 'NEUTRAL')),
    blocking BOOLEAN DEFAULT 0,
    
    -- Valeurs numériques
    value REAL,                                -- Valeur principale du test
    metadata TEXT,                             -- JSON: détails additionnels
    
    -- Message
    message TEXT,
    
    FOREIGN KEY (exec_id) REFERENCES Executions(id) ON DELETE CASCADE
);

-- Index pour analyses
CREATE INDEX IF NOT EXISTS idx_test_exec 
    ON TestResults(exec_id);

CREATE INDEX IF NOT EXISTS idx_test_name 
    ON TestResults(test_name);

CREATE INDEX IF NOT EXISTS idx_test_status 
    ON TestResults(status);

-- ============================================================================
-- TABLE HISTORIQUES (optionnel, compression)
-- ============================================================================

CREATE TABLE IF NOT EXISTS Histories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exec_id INTEGER NOT NULL,
    
    -- Snapshot
    iteration INTEGER NOT NULL,
    state_blob BLOB,                           -- np.ndarray compressé (pickle/gzip)
    
    -- Statistiques rapides (évite de décompresser)
    norm_frobenius REAL,
    diversity_std REAL,
    min_value REAL,
    max_value REAL,
    
    FOREIGN KEY (exec_id) REFERENCES Executions(id) ON DELETE CASCADE,
    UNIQUE(exec_id, iteration)
);

CREATE INDEX IF NOT EXISTS idx_history_exec 
    ON Histories(exec_id);

-- ============================================================================
-- TABLE SCORES AGRÉGÉS (pour analyses rapides)
-- ============================================================================

CREATE TABLE IF NOT EXISTS Scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Configuration
    gamma_id TEXT NOT NULL,
    gamma_params TEXT NOT NULL,
    d_base_id TEXT NOT NULL,
    modifier_id TEXT NOT NULL,
    
    -- Scores agrégés sur seeds
    n_runs INTEGER NOT NULL,                   -- Nombre de seeds
    n_completed INTEGER NOT NULL,
    n_errors INTEGER NOT NULL,
    
    -- Statistiques verdict
    n_pass INTEGER DEFAULT 0,
    n_poor INTEGER DEFAULT 0,
    n_rejected INTEGER DEFAULT 0,
    n_neutral INTEGER DEFAULT 0,
    
    -- Scores tests (moyennés sur seeds)
    score_univ_001 REAL,                       -- UNIV-001: Norme
    score_univ_002 REAL,                       -- UNIV-002: Diversité
    score_univ_003 REAL,                       -- UNIV-003: Convergence
    score_sym_001 REAL,                        -- SYM-001: Préservation
    score_str_002 REAL,                        -- STR-002: Spectre
    score_bnd_001 REAL,                        -- BND-001: Bornes
    
    -- Score global (0-20)
    global_score REAL,
    
    -- Métadonnées
    last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(gamma_id, gamma_params, d_base_id, modifier_id)
);

CREATE INDEX IF NOT EXISTS idx_scores_gamma 
    ON Scores(gamma_id);

CREATE INDEX IF NOT EXISTS idx_scores_global 
    ON Scores(global_score DESC);

-- ============================================================================
-- VUES UTILES
-- ============================================================================

-- Vue: Résumé par Γ
CREATE VIEW IF NOT EXISTS GammaSum AS
SELECT 
    gamma_id,
    gamma_params,
    COUNT(*) as total_runs,
    SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) as completed,
    SUM(CASE WHEN status = 'ERROR' THEN 1 ELSE 0 END) as errors,
    SUM(CASE WHEN global_verdict = 'PASS' THEN 1 ELSE 0 END) as pass_count,
    SUM(CASE WHEN global_verdict = 'REJECTED' THEN 1 ELSE 0 END) as rejected_count,
    AVG(execution_time_seconds) as avg_time
FROM Executions
GROUP BY gamma_id, gamma_params;

-- Vue: Résumé par D_base
CREATE VIEW IF NOT EXISTS DBaseSum AS
SELECT 
    d_base_id,
    COUNT(*) as total_runs,
    SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) as completed,
    SUM(CASE WHEN global_verdict = 'PASS' THEN 1 ELSE 0 END) as pass_count,
    AVG(execution_time_seconds) as avg_time
FROM Executions
GROUP BY d_base_id;

-- Vue: Tests échouant le plus
CREATE VIEW IF NOT EXISTS FailingTests AS
SELECT 
    test_name,
    COUNT(*) as total_runs,
    SUM(CASE WHEN status = 'FAIL' THEN 1 ELSE 0 END) as fail_count,
    CAST(SUM(CASE WHEN status = 'FAIL' THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as fail_rate
FROM TestResults
GROUP BY test_name
ORDER BY fail_rate DESC;

-- Vue: Matrice Γ × D (scores moyens)
CREATE VIEW IF NOT EXISTS GammaDMatrix AS
SELECT 
    s.gamma_id,
    s.gamma_params,
    s.d_base_id,
    s.global_score,
    s.n_pass,
    s.n_runs
FROM Scores s
ORDER BY s.gamma_id, s.d_base_id;

-- ============================================================================
-- TRIGGERS pour maintenir Scores
-- ============================================================================

-- Trigger: Après insertion Execution, mettre à jour Scores
CREATE TRIGGER IF NOT EXISTS update_scores_after_exec
AFTER INSERT ON Executions
FOR EACH ROW
BEGIN
    -- Insérer ou mettre à jour Scores
    INSERT INTO Scores (
        gamma_id, gamma_params, d_base_id, modifier_id,
        n_runs, n_completed, n_errors,
        n_pass, n_poor, n_rejected, n_neutral
    )
    VALUES (
        NEW.gamma_id, NEW.gamma_params, NEW.d_base_id, NEW.modifier_id,
        1,
        CASE WHEN NEW.status = 'COMPLETED' THEN 1 ELSE 0 END,
        CASE WHEN NEW.status = 'ERROR' THEN 1 ELSE 0 END,
        CASE WHEN NEW.global_verdict = 'PASS' THEN 1 ELSE 0 END,
        CASE WHEN NEW.global_verdict = 'POOR' THEN 1 ELSE 0 END,
        CASE WHEN NEW.global_verdict = 'REJECTED' THEN 1 ELSE 0 END,
        CASE WHEN NEW.global_verdict = 'NEUTRAL' THEN 1 ELSE 0 END
    )
    ON CONFLICT(gamma_id, gamma_params, d_base_id, modifier_id) DO UPDATE SET
        n_runs = n_runs + 1,
        n_completed = n_completed + CASE WHEN NEW.status = 'COMPLETED' THEN 1 ELSE 0 END,
        n_errors = n_errors + CASE WHEN NEW.status = 'ERROR' THEN 1 ELSE 0 END,
        n_pass = n_pass + CASE WHEN NEW.global_verdict = 'PASS' THEN 1 ELSE 0 END,
        n_poor = n_poor + CASE WHEN NEW.global_verdict = 'POOR' THEN 1 ELSE 0 END,
        n_rejected = n_rejected + CASE WHEN NEW.global_verdict = 'REJECTED' THEN 1 ELSE 0 END,
        n_neutral = n_neutral + CASE WHEN NEW.global_verdict = 'NEUTRAL' THEN 1 ELSE 0 END,
        last_updated = CURRENT_TIMESTAMP;
END;

-- ============================================================================
-- REQUÊTES UTILES (commentées)
-- ============================================================================

-- Trouver meilleurs Γ (score global):
-- SELECT gamma_id, AVG(global_score) as avg_score
-- FROM Scores
-- GROUP BY gamma_id
-- ORDER BY avg_score DESC;

-- Matrice robustesse Γ × D:
-- SELECT gamma_id, d_base_id, 
--        CAST(n_pass AS REAL) / n_runs as success_rate
-- FROM Scores
-- WHERE n_runs >= 3
-- ORDER BY gamma_id, success_rate DESC;

-- Tests bloquants les plus fréquents:
-- SELECT test_name, COUNT(*) as blocker_count
-- FROM TestResults
-- WHERE status = 'FAIL' AND blocking = 1
-- GROUP BY test_name
-- ORDER BY blocker_count DESC;