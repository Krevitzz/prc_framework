-- prc_database/schema_raw.sql
-- Schéma pour données brutes uniquement (pas de verdicts)

-- Table principale : exécutions avec TOUS les paramètres possibles
CREATE TABLE IF NOT EXISTS Executions (
  -- ID et métadonnées
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT UNIQUE NOT NULL,           -- "GAM-001_beta2.0_SYM-001_M0_s1"
  timestamp TEXT NOT NULL,
  
  -- Configuration Γ
  gamma_id TEXT NOT NULL,                -- "GAM-001"
  
  -- TOUS les paramètres possibles (NULL si non applicable)
  -- Markoviens
  alpha REAL,                            -- Diffusion, mémoire
  beta REAL,                             -- Saturation
  gamma_param REAL,                      -- Croissance/décroissance (évite conflit avec gamma_id)
  omega REAL,                            -- Oscillateur
  
  -- Non-markoviens
  memory_weight REAL,                    -- Poids mémoire
  window_size INTEGER,                   -- Taille fenêtre moyenne glissante
  epsilon REAL,                          -- Régulation
  
  -- Stochastiques
  sigma REAL,                            -- Bruit additif/multiplicatif
  lambda_param REAL,                     -- Branchement
  
  -- Structurels
  eta REAL,                              -- Hebbien
  subspace_dim INTEGER,                  -- Projection
  
  -- Configuration D
  d_base_id TEXT NOT NULL,              -- "SYM-001"
  modifier_id TEXT NOT NULL,            -- "M0", "M1", etc.
  seed INTEGER NOT NULL,
  
  -- Configuration exécution
  max_iterations INTEGER NOT NULL,
  snapshot_interval INTEGER DEFAULT 10,
  
  -- Résultats exécution
  status TEXT NOT NULL,                 -- "COMPLETED" | "ERROR" | "TIMEOUT"
  error_message TEXT,
  final_iteration INTEGER,
  execution_time_seconds REAL,
  
  -- Métriques finales (observables brutes)
  converged BOOLEAN,
  convergence_iteration INTEGER,
  
  -- Indexes pour requêtes rapides
  UNIQUE(gamma_id, alpha, beta, gamma_param, omega, memory_weight, window_size, 
         epsilon, sigma, lambda_param, eta, subspace_dim, 
         d_base_id, modifier_id, seed)
);

-- Table snapshots : états sauvegardés à intervalles
CREATE TABLE IF NOT EXISTS Snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  exec_id INTEGER NOT NULL,
  iteration INTEGER NOT NULL,
  
  -- État compressé (pickle + gzip)
  state_blob BLOB,
  
  -- Métriques rapides (pour queries sans décompresser)
  norm_frobenius REAL,
  norm_spectral REAL,
  min_value REAL,
  max_value REAL,
  mean_value REAL,
  std_value REAL,
  
  FOREIGN KEY (exec_id) REFERENCES Executions(id),
  UNIQUE(exec_id, iteration)
);

-- Table métriques : évolution par itération (sans sauver l'état)
CREATE TABLE IF NOT EXISTS Metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  exec_id INTEGER NOT NULL,
  iteration INTEGER NOT NULL,
  
  -- Normes
  norm_frobenius REAL,
  norm_spectral REAL,
  norm_max REAL,
  
  -- Statistiques éléments
  min_value REAL,
  max_value REAL,
  mean_value REAL,
  std_value REAL,
  
  -- Distances (convergence)
  distance_to_previous REAL,
  
  -- Symétrie (si applicable)
  asymmetry_norm REAL,
  
  FOREIGN KEY (exec_id) REFERENCES Executions(id),
  UNIQUE(exec_id, iteration)
);

-- Indexes pour performance
CREATE INDEX IF NOT EXISTS idx_exec_gamma ON Executions(gamma_id);
CREATE INDEX IF NOT EXISTS idx_exec_dbase ON Executions(d_base_id);
CREATE INDEX IF NOT EXISTS idx_exec_status ON Executions(status);
CREATE INDEX IF NOT EXISTS idx_exec_params ON Executions(gamma_id, beta, alpha, sigma);

CREATE INDEX IF NOT EXISTS idx_snapshot_exec ON Snapshots(exec_id);
CREATE INDEX IF NOT EXISTS idx_metrics_exec ON Metrics(exec_id);