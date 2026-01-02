-- prc_automation/prc_database/schema_results_migration.sql
-- Migration schema db_results pour scoring/verdicts R0
-- Architecture Charter 5.4 - Section 12.8-12.9

-- =============================================================================
-- MIGRATION TestScores
-- =============================================================================

-- Si table existe, créer backup
CREATE TABLE IF NOT EXISTS TestScores_backup AS
SELECT * FROM TestScores;

-- Drop ancienne table
DROP TABLE IF EXISTS TestScores;

-- Recréer avec nouveau schema
CREATE TABLE TestScores (
  id INTEGER PRIMARY KEY,
  exec_id INTEGER NOT NULL,
  test_name TEXT NOT NULL,
  
  -- Configs utilisées (traçabilité)
  params_config_id TEXT NOT NULL,
  scoring_config_id TEXT NOT NULL,
  
  -- Score test agrégé [0,1]
  test_score REAL NOT NULL,             -- NOUVEAU NOM (était weighted_average)
  aggregation_mode TEXT NOT NULL,       -- NOUVEAU
  
  -- Détails métriques (JSON complet)
  metric_scores TEXT NOT NULL,          -- JSON {metric_key: {value, score, flag, type, weight, evidence}}
  pathology_flags TEXT,                 -- NOUVEAU: JSON liste métriques flaggées
  critical_metrics TEXT,                -- NOUVEAU: JSON liste métriques score >= 0.8
  
  -- Metadata
  test_weight REAL NOT NULL,
  computed_at TEXT NOT NULL,
  
  UNIQUE(exec_id, test_name, params_config_id, scoring_config_id),
  FOREIGN KEY(exec_id) REFERENCES Executions(id),
  FOREIGN KEY(params_config_id) REFERENCES ConfigRegistry(config_id),
  FOREIGN KEY(scoring_config_id) REFERENCES ConfigRegistry(config_id)
);

-- Index
CREATE INDEX idx_test_scores_lookup 
ON TestScores(exec_id, params_config_id, scoring_config_id);

CREATE INDEX idx_test_scores_test 
ON TestScores(test_name);

CREATE INDEX idx_test_scores_score 
ON TestScores(test_score);

-- =============================================================================
-- MIGRATION GammaVerdicts
-- =============================================================================

-- Si table existe, créer backup
CREATE TABLE IF NOT EXISTS GammaVerdicts_backup AS
SELECT * FROM GammaVerdicts;

-- Drop ancienne table
DROP TABLE IF EXISTS GammaVerdicts;

-- Recréer avec nouveau schema
CREATE TABLE GammaVerdicts (
  id INTEGER PRIMARY KEY,
  gamma_id TEXT NOT NULL,
  
  -- Configs utilisées (traçabilité complète)
  params_config_id TEXT NOT NULL,
  scoring_config_id TEXT NOT NULL,
  thresholds_config_id TEXT NOT NULL,
  
  -- Critères
  majority_pct REAL NOT NULL,
  robustness_pct REAL NOT NULL,
  global_score REAL NOT NULL,  -- [0,1] pathology score moyen
  
  -- Verdict
  verdict TEXT NOT NULL,  -- SURVIVES[R0] | WIP[R0-open] | REJECTED[R0]
  verdict_reason TEXT NOT NULL,
  
  -- Détails (JSON complet pour traçabilité)
  details TEXT NOT NULL,  -- NOUVEAU: JSON {n_configs, critical_tests, d_breakdown, ...}
  
  -- Metadata
  computed_at TEXT NOT NULL,
  
  UNIQUE(gamma_id, params_config_id, scoring_config_id, thresholds_config_id),
  FOREIGN KEY(params_config_id) REFERENCES ConfigRegistry(config_id),
  FOREIGN KEY(scoring_config_id) REFERENCES ConfigRegistry(config_id),
  FOREIGN KEY(thresholds_config_id) REFERENCES ConfigRegistry(config_id)
);

-- Index
CREATE INDEX idx_gamma_verdicts_lookup 
ON GammaVerdicts(gamma_id, params_config_id, scoring_config_id, thresholds_config_id);

CREATE INDEX idx_gamma_verdicts_gamma 
ON GammaVerdicts(gamma_id);

CREATE INDEX idx_gamma_verdicts_verdict 
ON GammaVerdicts(verdict);

CREATE INDEX idx_gamma_verdicts_score 
ON GammaVerdicts(global_score);

-- =============================================================================
-- Vérifications
-- =============================================================================

-- Compter enregistrements backup
SELECT 'TestScores_backup:' as table_name, COUNT(*) as count FROM TestScores_backup;
SELECT 'GammaVerdicts_backup:' as table_name, COUNT(*) as count FROM GammaVerdicts_backup;

-- Note: Pour restaurer backup si nécessaire:
-- DROP TABLE TestScores;
-- ALTER TABLE TestScores_backup RENAME TO TestScores;