CREATE TABLE IF NOT EXISTS sf_data_asset (
  asset_id VARCHAR(64) PRIMARY KEY,
  provider VARCHAR(64) NOT NULL,
  dataset VARCHAR(128) NOT NULL,
  store VARCHAR(32) NOT NULL,
  data_scope VARCHAR(32) NOT NULL DEFAULT 'unknown',
  seat_detail_scope VARCHAR(32) NOT NULL DEFAULT 'unknown',
  lineage_hash VARCHAR(128) NULL,
  row_count BIGINT NOT NULL DEFAULT 0,
  version BIGINT NOT NULL DEFAULT 0,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_sf_data_asset (provider, dataset, store),
  INDEX idx_sf_data_asset_scope (data_scope, seat_detail_scope)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS sf_data_quality_report (
  report_id VARCHAR(64) PRIMARY KEY,
  provider VARCHAR(64) NOT NULL,
  dataset VARCHAR(128) NOT NULL,
  trade_date DATE NOT NULL,
  passed BOOLEAN NOT NULL DEFAULT FALSE,
  completeness_rate DECIMAL(8,6) NOT NULL DEFAULT 0,
  duplicate_rate DECIMAL(8,6) NOT NULL DEFAULT 0,
  risk_flags_json JSON NOT NULL,
  version BIGINT NOT NULL DEFAULT 0,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_sf_quality_report (provider, dataset, trade_date),
  INDEX idx_sf_quality_passed (passed, created_at)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS sf_graph_snapshot (
  snapshot_id VARCHAR(64) PRIMARY KEY,
  provider VARCHAR(64) NOT NULL,
  trade_date DATE NOT NULL,
  node_count BIGINT NOT NULL DEFAULT 0,
  edge_count BIGINT NOT NULL DEFAULT 0,
  data_scope VARCHAR(32) NOT NULL DEFAULT 'unknown',
  seat_detail_scope VARCHAR(32) NOT NULL DEFAULT 'unknown',
  lineage_hash VARCHAR(128) NULL,
  version BIGINT NOT NULL DEFAULT 0,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_sf_graph_snapshot (provider, trade_date),
  INDEX idx_sf_graph_lineage (lineage_hash)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS sf_dataset_version (
  version_id VARCHAR(64) PRIMARY KEY,
  provider VARCHAR(64) NOT NULL,
  dataset VARCHAR(128) NOT NULL,
  version VARCHAR(128) NOT NULL,
  artifact_uri VARCHAR(512) NOT NULL,
  content_sha256 VARCHAR(128) NULL,
  active BOOLEAN NOT NULL DEFAULT FALSE,
  version_lock BIGINT NOT NULL DEFAULT 0,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_sf_dataset_version (provider, dataset, version),
  INDEX idx_sf_dataset_version_active (provider, dataset, active)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS sf_sync_task (
  task_id VARCHAR(64) PRIMARY KEY,
  idempotency_key VARCHAR(256) NOT NULL,
  provider VARCHAR(64) NOT NULL,
  dataset VARCHAR(128) NOT NULL,
  mode VARCHAR(64) NOT NULL,
  status VARCHAR(32) NOT NULL,
  progress INT NOT NULL DEFAULT 0,
  message VARCHAR(512) NOT NULL DEFAULT 'accepted',
  risk_flags_json JSON NOT NULL,
  version BIGINT NOT NULL DEFAULT 0,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_sf_sync_task_idempotency (idempotency_key),
  INDEX idx_sf_sync_task_status (status, created_at)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS sf_sync_audit_event (
  event_id VARCHAR(64) PRIMARY KEY,
  trace_id VARCHAR(128) NOT NULL,
  task_id VARCHAR(64) NULL,
  actor VARCHAR(128) NOT NULL,
  action VARCHAR(128) NOT NULL,
  outcome VARCHAR(64) NOT NULL,
  risk_flags_json JSON NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_sf_sync_audit_trace (trace_id),
  INDEX idx_sf_sync_audit_task (task_id, created_at)
) ENGINE=InnoDB;
