-- Table registry configs
CREATE TABLE ConfigRegistry (
  id INTEGER PRIMARY KEY,
  config_type TEXT NOT NULL,  -- 'params' | 'scoring' | 'thresholds'
  config_scope TEXT NOT NULL, -- 'global' | test_id
  config_id TEXT NOT NULL,    -- 'params_default_v1' | 'UNIV-001_params_custom_v1'
  config_data TEXT NOT NULL,  -- JSON du YAML complet
  created_at TEXT NOT NULL,
  UNIQUE(config_type, config_scope, config_id)
);

-- Index
CREATE INDEX idx_config_registry_lookup 
ON ConfigRegistry(config_type, config_scope, config_id);


-- Observations tests
CREATE TABLE TestObservations (
  id INTEGER PRIMARY KEY,
  exec_id INTEGER NOT NULL,
  test_name TEXT NOT NULL,
  test_category TEXT NOT NULL,  -- UNIV, SYM, STR, etc.
  
  -- Config params utilisée
  params_config_id TEXT NOT NULL,
  
  -- Applicabilité
  applicable BOOLEAN NOT NULL,
  status TEXT NOT NULL,  -- SUCCESS | NOT_APPLICABLE | SKIPPED
  message TEXT,
  
  -- Statistiques (dict v2)
  stat_initial REAL,
  stat_final REAL,
  stat_min REAL,
  stat_max REAL,
  stat_mean REAL,
  stat_std REAL,
  
  -- Évolution (dict v2)
  evolution_transition TEXT,
  evolution_trend TEXT,
  evolution_trend_coefficient REAL,
  
  -- Backup JSON
  observation_data TEXT NOT NULL,  -- JSON complet dict v2
  
  computed_at TEXT NOT NULL,
  
  UNIQUE(exec_id, test_name, params_config_id),
  FOREIGN KEY(exec_id) REFERENCES Executions(id),
  FOREIGN KEY(params_config_id) REFERENCES ConfigRegistry(config_id)
);

-- Index
CREATE INDEX idx_test_obs_lookup 
ON TestObservations(exec_id, test_name, params_config_id);

CREATE INDEX idx_test_obs_applicable 
ON TestObservations(applicable, status);


-- Scores tests
CREATE TABLE TestScores (
  id INTEGER PRIMARY KEY,
  exec_id INTEGER NOT NULL,
  test_name TEXT NOT NULL,
  
  -- Configs utilisées
  params_config_id TEXT NOT NULL,
  scoring_config_id TEXT NOT NULL,
  
  -- Poids et scores
  test_weight REAL NOT NULL,
  metric_scores TEXT NOT NULL,  -- JSON: {metric_key: {score, weight, skipped}}
  weighted_average REAL NOT NULL,  -- 0-1
  
  computed_at TEXT NOT NULL,
  
  UNIQUE(exec_id, test_name, params_config_id, scoring_config_id),
  FOREIGN KEY(exec_id) REFERENCES Executions(id),
  FOREIGN KEY(params_config_id) REFERENCES ConfigRegistry(config_id),
  FOREIGN KEY(scoring_config_id) REFERENCES ConfigRegistry(config_id)
);

-- Index
CREATE INDEX idx_test_scores_lookup 
ON TestScores(exec_id, params_config_id, scoring_config_id);


-- Verdicts gamma
CREATE TABLE GammaVerdicts (
  id INTEGER PRIMARY KEY,
  gamma_id TEXT NOT NULL,
  
  -- Configs utilisées
  params_config_id TEXT NOT NULL,
  scoring_config_id TEXT NOT NULL,
  thresholds_config_id TEXT NOT NULL,
  
  -- Critères agrégés
  majority_pct REAL NOT NULL,
  robustness_pct REAL NOT NULL,
  global_score REAL NOT NULL,  -- 0-1
  
  -- Verdict
  verdict TEXT NOT NULL,  -- SURVIVES[R0] | WIP[R0-closed] | FLAGGED_FOR_REVIEW
  verdict_reason TEXT,
  
  computed_at TEXT NOT NULL,
  
  UNIQUE(gamma_id, params_config_id, scoring_config_id, thresholds_config_id),
  FOREIGN KEY(params_config_id) REFERENCES ConfigRegistry(config_id),
  FOREIGN KEY(scoring_config_id) REFERENCES ConfigRegistry(config_id),
  FOREIGN KEY(thresholds_config_id) REFERENCES ConfigRegistry(config_id)
);

-- Index
CREATE INDEX idx_gamma_verdicts_lookup 
ON GammaVerdicts(gamma_id, params_config_id, scoring_config_id, thresholds_config_id);