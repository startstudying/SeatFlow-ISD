CREATE TABLE IF NOT EXISTS sf_experiment (
  experiment_id BIGINT PRIMARY KEY AUTO_INCREMENT,
  experiment_key VARCHAR(128) NOT NULL,
  name VARCHAR(255) NOT NULL,
  scope VARCHAR(64) NOT NULL DEFAULT 'local_internal_research',
  owner VARCHAR(128) NULL,
  description TEXT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_sf_experiment_key (experiment_key)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS sf_experiment_run (
  run_id VARCHAR(128) PRIMARY KEY,
  experiment_id BIGINT NULL,
  upstream_run_id VARCHAR(128) NOT NULL,
  status VARCHAR(32) NOT NULL,
  model_id VARCHAR(128) NULL,
  data_scope VARCHAR(64) NOT NULL DEFAULT 'unknown',
  model_validity_status VARCHAR(64) NOT NULL DEFAULT 'unavailable',
  lineage_hash VARCHAR(128) NULL,
  risk_flags_json JSON NOT NULL,
  upstream_payload_json JSON NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  CONSTRAINT fk_sf_run_experiment FOREIGN KEY (experiment_id) REFERENCES sf_experiment(experiment_id),
  INDEX idx_sf_run_status (status, created_at),
  INDEX idx_sf_run_lineage (lineage_hash)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS sf_prediction_reference (
  prediction_ref_id BIGINT PRIMARY KEY AUTO_INCREMENT,
  provider VARCHAR(64) NOT NULL,
  model_id VARCHAR(128) NOT NULL,
  trade_date DATE NOT NULL,
  target_stock_id VARCHAR(64) NOT NULL,
  upstream_run_id VARCHAR(128) NULL,
  data_scope VARCHAR(64) NOT NULL DEFAULT 'unknown',
  model_validity_status VARCHAR(64) NOT NULL DEFAULT 'unavailable',
  lineage_hash VARCHAR(128) NULL,
  risk_flags_json JSON NOT NULL,
  upstream_artifact_uri VARCHAR(512) NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_sf_prediction_ref (provider, model_id, trade_date, target_stock_id),
  INDEX idx_sf_prediction_lineage (lineage_hash)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS sf_backtest_reference (
  backtest_ref_id BIGINT PRIMARY KEY AUTO_INCREMENT,
  run_id VARCHAR(128) NOT NULL,
  upstream_run_id VARCHAR(128) NOT NULL,
  data_scope VARCHAR(64) NOT NULL DEFAULT 'unknown',
  model_validity_status VARCHAR(64) NOT NULL DEFAULT 'unavailable',
  backtest_valid BOOLEAN NOT NULL DEFAULT FALSE,
  lineage_hash VARCHAR(128) NULL,
  risk_flags_json JSON NOT NULL,
  upstream_artifact_uri VARCHAR(512) NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_sf_backtest_run (run_id),
  INDEX idx_sf_backtest_lineage (lineage_hash)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS sf_audit_status (
  audit_id BIGINT PRIMARY KEY AUTO_INCREMENT,
  scope VARCHAR(64) NOT NULL DEFAULT 'local_internal_research',
  data_ready BOOLEAN NOT NULL DEFAULT FALSE,
  model_valid BOOLEAN NOT NULL DEFAULT FALSE,
  backtest_valid BOOLEAN NOT NULL DEFAULT FALSE,
  competition_ready BOOLEAN NOT NULL DEFAULT FALSE,
  seat_detail_scope VARCHAR(64) NOT NULL DEFAULT 'none',
  evidence_hashes_json JSON NOT NULL,
  blocked_reasons_json JSON NOT NULL,
  last_verified_at DATETIME NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_sf_audit_created (created_at)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS sf_api_audit_log (
  audit_log_id BIGINT PRIMARY KEY AUTO_INCREMENT,
  trace_id VARCHAR(64) NOT NULL,
  actor VARCHAR(128) NOT NULL DEFAULT 'local-internal',
  action VARCHAR(128) NOT NULL,
  resource_type VARCHAR(64) NOT NULL,
  resource_id VARCHAR(128) NULL,
  outcome VARCHAR(32) NOT NULL,
  risk_flags_json JSON NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_sf_api_audit_trace (trace_id),
  INDEX idx_sf_api_audit_resource (resource_type, resource_id)
) ENGINE=InnoDB;
