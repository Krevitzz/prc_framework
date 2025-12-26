-- prc_database/schema_results.sql
-- Schéma pour analyses, scores et verdicts (Section 14.3)

-- ===========================================================================
-- RÈGLE CRITIQUE (Section 14.3) :
-- Cette base est REJOUABLE et DÉRIVÉE de db_raw.
-- Elle peut être supprimée et reconstruite sans perte d'information.
-- ===========================================================================

-- ===========================================================================
-- TABLE 1 : TestObservations - Observations brutes des tests
-- ===========================================================================

CREATE TABLE IF NOT EXISTS TestObservations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  
  -- Référence au run dans db_raw (OBLIGATOIRE)
  exec_id INTEGER NOT NULL,
  
  -- Identification du test
  test_name TEXT NOT NULL,              -- "TEST-UNIV-001", "TEST-SYM-001", etc.
  test_category TEXT NOT NULL,          -- "UNIV", "SYM", "STR", "BND", etc.
  
  -- Applicabilité technique
  applicable BOOLEAN NOT NULL,          -- False si test non applicable à ce contexte
  
  -- Observations brutes (JSON)
  observation_data TEXT NOT NULL,       -- JSON sérialisé de l'observation
  
  -- Métriques clés extraites (pour requêtes rapides)
  initial_value REAL,
  final_value REAL,
  transition TEXT,                      -- "preserved" | "created" | "destroyed" | "absent"
  
  -- Métadonnées
  computed_at TEXT NOT NULL,
  
  -- Contraintes
  FOREIGN KEY (exec_id) REFERENCES Executions(id),
  UNIQUE(exec_id, test_name)
);

-- ===========================================================================
-- TABLE 2 : TestScores - Scores calculés avec config spécifique
-- ===========================================================================

CREATE TABLE IF NOT EXISTS TestScores (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  
  -- Référence au run dans db_raw
  exec_id INTEGER NOT NULL,
  
  -- Référence au test
  test_name TEXT NOT NULL,
  
  -- Configuration de scoring (OBLIGATOIRE Section 14.5)
  config_id TEXT NOT NULL,              -- "weights_default", "weights_conservative", etc.
  
  -- Score normalisé 0-1
  score REAL NOT NULL,                  -- Échelle universelle 0-1
  
  -- Pondération appliquée
  weight REAL NOT NULL,                 -- Depuis config YAML
  
  -- Score pondéré
  weighted_score REAL NOT NULL,         -- score × weight
  
  -- Métadonnées
  computed_at TEXT NOT NULL,
  
  -- Contraintes
  FOREIGN KEY (exec_id) REFERENCES Executions(id),
  UNIQUE(exec_id, test_name, config_id),
  CHECK(score >= 0.0 AND score <= 1.0)
);

-- ===========================================================================
-- TABLE 3 : GammaVerdicts - Verdicts agrégés (Section 14.6)
-- ===========================================================================

CREATE TABLE IF NOT EXISTS GammaVerdicts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  
  -- Identification Γ
  gamma_id TEXT NOT NULL,
  
  -- Configuration scoring/seuils (OBLIGATOIRE Section 14.5)
  config_id TEXT NOT NULL,
  threshold_id TEXT NOT NULL,
  
  -- Les 3 critères (Section 7 de la Feuille de Route)
  majority_pct REAL NOT NULL,           -- % configs PASS
  robustness_pct REAL NOT NULL,         -- % D avec ≥1 config viable
  score_global REAL NOT NULL,           -- Moyenne pondérée /20
  
  -- Verdict final
  verdict TEXT NOT NULL,                -- "SURVIVES[R0]" | "WIP[R0-closed]" | "FLAGGED_FOR_REVIEW"
  verdict_reason TEXT,
  
  -- Statistiques complémentaires
  n_total_configs INTEGER NOT NULL,     -- Nombre total de configurations testées
  n_pass_configs INTEGER NOT NULL,      -- Nombre de configs avec verdict PASS
  n_viable_d_bases INTEGER NOT NULL,    -- Nombre de D avec ≥1 config viable
  n_total_d_bases INTEGER NOT NULL,     -- Nombre total de D testés
  
  -- Métadonnées
  computed_at TEXT NOT NULL,
  
  -- Contraintes
  UNIQUE(gamma_id, config_id, threshold_id),
  CHECK(verdict IN ('SURVIVES[R0]', 'WIP[R0-closed]', 'FLAGGED_FOR_REVIEW')),
  CHECK(majority_pct >= 0.0 AND majority_pct <= 100.0),
  CHECK(robustness_pct >= 0.0 AND robustness_pct <= 100.0),
  CHECK(score_global >= 0.0 AND score_global <= 20.0)
);

-- ===========================================================================
-- INDEXES POUR PERFORMANCE
-- ===========================================================================

-- TestObservations
CREATE INDEX IF NOT EXISTS idx_obs_exec ON TestObservations(exec_id);
CREATE INDEX IF NOT EXISTS idx_obs_test ON TestObservations(test_name);
CREATE INDEX IF NOT EXISTS idx_obs_category ON TestObservations(test_category);
CREATE INDEX IF NOT EXISTS idx_obs_applicable ON TestObservations(applicable);

-- TestScores
CREATE INDEX IF NOT EXISTS idx_score_exec ON TestScores(exec_id);
CREATE INDEX IF NOT EXISTS idx_score_test ON TestScores(test_name);
CREATE INDEX IF NOT EXISTS idx_score_config ON TestScores(config_id);
CREATE INDEX IF NOT EXISTS idx_score_exec_config ON TestScores(exec_id, config_id);

-- GammaVerdicts
CREATE INDEX IF NOT EXISTS idx_verdict_gamma ON GammaVerdicts(gamma_id);
CREATE INDEX IF NOT EXISTS idx_verdict_config ON GammaVerdicts(config_id, threshold_id);
CREATE INDEX IF NOT EXISTS idx_verdict_status ON GammaVerdicts(verdict);

-- ===========================================================================
-- VUES UTILES (OPTIONNELLES)
-- ===========================================================================

-- Vue : Résumé des tests par run
CREATE VIEW IF NOT EXISTS v_test_summary AS
SELECT 
  exec_id,
  COUNT(*) as n_tests,
  SUM(CASE WHEN applicable = 1 THEN 1 ELSE 0 END) as n_applicable,
  AVG(CASE WHEN applicable = 1 THEN (SELECT score FROM TestScores WHERE TestScores.exec_id = TestObservations.exec_id AND TestScores.test_name = TestObservations.test_name AND config_id = 'weights_default') END) as avg_score_default
FROM TestObservations
GROUP BY exec_id;

-- Vue : Verdicts actifs (dernière config)
CREATE VIEW IF NOT EXISTS v_current_verdicts AS
SELECT 
  gamma_id,
  verdict,
  config_id,
  threshold_id,
  score_global,
  majority_pct,
  robustness_pct,
  computed_at
FROM GammaVerdicts
WHERE config_id = 'weights_default' AND threshold_id = 'thresholds_default';

-- ===========================================================================
-- MÉTADONNÉES SCHÉMA
-- ===========================================================================

CREATE TABLE IF NOT EXISTS schema_metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

INSERT OR REPLACE INTO schema_metadata (key, value) VALUES
  ('version', '1.0'),
  ('created_date', datetime('now')),
  ('charter_section', '14.3'),
  ('description', 'Base de données des analyses et verdicts (dérivée de db_raw)');

-- ===========================================================================
-- NOTES D'IMPLÉMENTATION
-- ===========================================================================

-- 1. Séparation stricte avec db_raw :
--    - exec_id référence db_raw.Executions.id
--    - Pas de duplication des données de runs
--    - Cette base peut être supprimée et reconstruite

-- 2. Rejouabilité (Section 14.1) :
--    - Chaque config_id génère un nouveau jeu de scores
--    - Chaque threshold_id génère un nouveau verdict
--    - Plusieurs verdicts peuvent coexister

-- 3. Append-only :
--    - INSERT uniquement (sauf bugs critiques)
--    - Pas de UPDATE sur scores/verdicts existants
--    - Nouveau verdict = nouvelle ligne

-- 4. Validation externe :
--    - tests/validation/validate_db_results.py vérifie cohérence
--    - Pas de contraintes SQL complexes (performances)