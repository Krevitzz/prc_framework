-- prc_database/schema_results.sql
-- Schema résultats Charter 5.4 (pathologies + patterns)

-- =============================================================================
-- TABLE 1 : REGISTRY CONFIGS
-- =============================================================================

CREATE TABLE IF NOT EXISTS ConfigRegistry (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  config_type TEXT NOT NULL,     -- 'params' | 'scoring'
  config_scope TEXT NOT NULL,    -- 'global' | test_id
  config_id TEXT NOT NULL,       -- 'params_default_v1' | 'UNIV-001_params_custom_v1'
  config_data TEXT NOT NULL,     -- JSON du YAML complet
  created_at TEXT NOT NULL,
  
  UNIQUE(config_type, config_scope, config_id)
);

CREATE INDEX IF NOT EXISTS idx_config_registry_lookup 
ON ConfigRegistry(config_type, config_scope, config_id);

-- =============================================================================
-- TABLE 2 : OBSERVATIONS TESTS (inchangé)
-- =============================================================================

CREATE TABLE IF NOT EXISTS TestObservations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  
  -- Traçabilité run
  exec_id INTEGER NOT NULL,
  test_name TEXT NOT NULL,
  test_category TEXT NOT NULL,
  
  -- Config params utilisée
  params_config_id TEXT NOT NULL,
  
  -- Applicabilité
  applicable BOOLEAN NOT NULL,
  status TEXT NOT NULL,         -- SUCCESS | NOT_APPLICABLE | SKIPPED | ERROR
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
  FOREIGN KEY(exec_id) REFERENCES Executions(id),
  FOREIGN KEY(params_config_id) REFERENCES ConfigRegistry(config_id)
);

CREATE INDEX IF NOT EXISTS idx_test_obs_lookup 
ON TestObservations(exec_id, test_name, params_config_id);

CREATE INDEX IF NOT EXISTS idx_test_obs_applicable 
ON TestObservations(applicable, status);

-- =============================================================================
-- TABLE 3 : SCORES TESTS (pathologies)
-- =============================================================================

CREATE TABLE IF NOT EXISTS TestScores (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  
  -- Traçabilité run
  exec_id INTEGER NOT NULL,
  test_name TEXT NOT NULL,
  
  -- Configs utilisées
  params_config_id TEXT NOT NULL,
  scoring_config_id TEXT NOT NULL,
  
  -- Score pathologie [0,1]
  test_score REAL NOT NULL,
  aggregation_mode TEXT NOT NULL,    -- max | weighted_mean | weighted_max
  
  -- Détails métriques (JSON complet)
  metric_scores TEXT NOT NULL,
  -- Structure: {metric_key: {value, score, flag, pathology_type, weight, source}}
  
  -- Flags
  pathology_flags TEXT NOT NULL,     -- JSON liste métriques flaggées
  critical_metrics TEXT NOT NULL,    -- JSON liste métriques score >= 0.8
  
  -- Metadata
  test_weight REAL NOT NULL,
  computed_at TEXT NOT NULL,
  
  UNIQUE(exec_id, test_name, params_config_id, scoring_config_id),
  FOREIGN KEY(exec_id) REFERENCES Executions(id),
  FOREIGN KEY(params_config_id) REFERENCES ConfigRegistry(config_id),
  FOREIGN KEY(scoring_config_id) REFERENCES ConfigRegistry(config_id)
);

CREATE INDEX IF NOT EXISTS idx_test_scores_lookup 
ON TestScores(exec_id, test_name, params_config_id, scoring_config_id);

CREATE INDEX IF NOT EXISTS idx_test_scores_gamma
ON TestScores(exec_id);

-- =============================================================================
-- TABLE 4 : VERDICTS GAMMA (patterns)
-- =============================================================================

CREATE TABLE IF NOT EXISTS GammaVerdicts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  gamma_id TEXT NOT NULL,
  
  -- Configs utilisées
  params_config_id TEXT NOT NULL,
  scoring_config_id TEXT NOT NULL,
  
  -- Verdict patterns-based
  verdict TEXT NOT NULL,            -- SURVIVES[R0] | WIP[R0-open] | REJECTED[R0]
  verdict_reason TEXT NOT NULL,
  
  -- Patterns détectés (JSON complet)
  patterns_summary TEXT NOT NULL,
  -- Structure: {
  --   patterns_detected: {...},
  --   metric_quality: {...},
  --   actionable_insights: [...],
  --   all_exec_ids: [...],
  --   failing_d: [...],
  --   failing_modifiers: [...],
  --   critical_metrics: [...]
  -- }
  
  -- Metadata
  num_runs_analyzed INTEGER NOT NULL,
  num_tests_analyzed INTEGER NOT NULL,
  computed_at TEXT NOT NULL,
  
  UNIQUE(gamma_id, params_config_id, scoring_config_id),
  FOREIGN KEY(params_config_id) REFERENCES ConfigRegistry(config_id),
  FOREIGN KEY(scoring_config_id) REFERENCES ConfigRegistry(config_id)
);

CREATE INDEX IF NOT EXISTS idx_gamma_verdicts_lookup 
ON GammaVerdicts(gamma_id, params_config_id, scoring_config_id);

CREATE INDEX IF NOT EXISTS idx_gamma_verdicts_gamma
ON GammaVerdicts(gamma_id);

-- =============================================================================
-- VUES UTILITAIRES
-- =============================================================================

-- Vue : Scores avec contexte complet
CREATE VIEW IF NOT EXISTS v_scores_with_context AS
SELECT 
    ts.id,
    ts.exec_id,
    e.run_id,
    e.gamma_id,
    e.d_base_id,
    e.modifier_id,
    e.seed,
    ts.test_name,
    ts.test_score,
    ts.pathology_flags,
    ts.critical_metrics,
    ts.params_config_id,
    ts.scoring_config_id,
    ts.computed_at
FROM TestScores ts
JOIN Executions e ON ts.exec_id = e.id;

-- Vue : Résumé verdict par gamma
CREATE VIEW IF NOT EXISTS v_verdict_summary AS
SELECT 
    gv.gamma_id,
    gv.verdict,
    gv.num_runs_analyzed,
    gv.num_tests_analyzed,
    gv.params_config_id,
    gv.scoring_config_id,
    gv.computed_at
FROM GammaVerdicts gv;