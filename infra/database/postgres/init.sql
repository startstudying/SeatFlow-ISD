-- SeatFlow V3 control_center database initial schema.
--
-- This script runs automatically on first PostgreSQL container startup
-- (mounted via docker-entrypoint-initdb.d). All tables use UUID primary keys
-- and include standard audit timestamps (created_at, updated_at).

-- Ensure the pgcrypto extension is available for gen_random_uuid().
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- Core task / experiment tracking
-- ============================================================================

CREATE TABLE IF NOT EXISTS tasks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    description     TEXT,
    task_type       TEXT NOT NULL,              -- e.g. research, training, evaluation
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending / running / completed / failed
    config          JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS experiments (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id         UUID REFERENCES tasks(id) ON DELETE SET NULL,
    name            TEXT NOT NULL,
    description     TEXT,
    experiment_type TEXT NOT NULL,              -- e.g. ablation, hyperparam_search, baseline
    status          TEXT NOT NULL DEFAULT 'active',
    tags            JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id   UUID NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    name            TEXT,
    status          TEXT NOT NULL DEFAULT 'scheduled',  -- scheduled / running / finished / failed / killed
    start_time      TIMESTAMPTZ,
    end_time        TIMESTAMPTZ,
    metrics         JSONB DEFAULT '{}',
    params          JSONB DEFAULT '{}',
    tags            JSONB DEFAULT '{}',
    artifact_uri    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================================
-- Model & dataset registry
-- ============================================================================

CREATE TABLE IF NOT EXISTS models (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    version         TEXT NOT NULL,
    description     TEXT,
    framework       TEXT,                        -- e.g. pytorch, lightgbm, catboost
    model_type      TEXT,                        -- e.g. gnn, gradient_boosting, ensemble
    artifact_path   TEXT,                        -- S3 / MinIO path to serialized model
    metrics         JSONB DEFAULT '{}',
    hyperparams     JSONB DEFAULT '{}',
    status          TEXT NOT NULL DEFAULT 'development',  -- development / staging / production / archived
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(name, version)
);

CREATE TABLE IF NOT EXISTS datasets (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    version         TEXT NOT NULL,
    description     TEXT,
    source_path     TEXT,                        -- S3 / MinIO path to dataset
    format          TEXT,                        -- e.g. parquet, csv
    schema_info     JSONB DEFAULT '{}',
    row_count       BIGINT,
    size_bytes      BIGINT,
    tags            JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(name, version)
);

-- ============================================================================
-- Quality & prediction tracking
-- ============================================================================

CREATE TABLE IF NOT EXISTS quality_gates (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id        UUID REFERENCES models(id) ON DELETE CASCADE,
    run_id          UUID REFERENCES runs(id) ON DELETE SET NULL,
    name            TEXT NOT NULL,
    gate_type       TEXT NOT NULL,               -- e.g. accuracy_threshold, data_drift, latency
    threshold       JSONB NOT NULL,
    actual_value    JSONB,
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending / passed / failed / warning
    details         JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS predictions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id        UUID NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    dataset_id      UUID REFERENCES datasets(id) ON DELETE SET NULL,
    prediction_type TEXT NOT NULL,               -- e.g. batch, online, backtest
    input_path      TEXT,                        -- path to input data
    output_path     TEXT,                        -- path to prediction output
    parameters      JSONB DEFAULT '{}',
    summary_metrics JSONB DEFAULT '{}',
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending / running / completed / failed
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS backtests (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id        UUID NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    dataset_id      UUID REFERENCES datasets(id) ON DELETE SET NULL,
    name            TEXT NOT NULL,
    strategy        TEXT,                        -- e.g. walk_forward, expanding_window
    start_date      DATE NOT NULL,
    end_date        DATE NOT NULL,
    parameters      JSONB DEFAULT '{}',
    results         JSONB DEFAULT '{}',
    performance_metrics JSONB DEFAULT '{}',
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending / running / completed / failed
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================================
-- Research & audit
-- ============================================================================

CREATE TABLE IF NOT EXISTS paper_sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title           TEXT NOT NULL,
    authors         TEXT,
    abstract        TEXT,
    venue           TEXT,
    year             INT,
    notes           JSONB DEFAULT '{}',
    tags            JSONB DEFAULT '{}',
    status          TEXT NOT NULL DEFAULT 'draft',  -- draft / in_review / published / archived
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    actor           TEXT NOT NULL,                -- user / service identifier
    action          TEXT NOT NULL,                -- e.g. create, update, delete, deploy
    resource_type   TEXT NOT NULL,                -- e.g. model, dataset, experiment
    resource_id     UUID,
    details         JSONB DEFAULT '{}',
    ip_address      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================================
-- Indexes for common query patterns
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_tasks_status       ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_experiments_task    ON experiments(task_id);
CREATE INDEX IF NOT EXISTS idx_runs_experiment     ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_runs_status         ON runs(status);
CREATE INDEX IF NOT EXISTS idx_models_status       ON models(status);
CREATE INDEX IF NOT EXISTS idx_models_name         ON models(name);
CREATE INDEX IF NOT EXISTS idx_datasets_name       ON datasets(name);
CREATE INDEX IF NOT EXISTS idx_quality_gates_model ON quality_gates(model_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model   ON predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_backtests_model     ON backtests(model_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_resource  ON audit_logs(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created   ON audit_logs(created_at);
