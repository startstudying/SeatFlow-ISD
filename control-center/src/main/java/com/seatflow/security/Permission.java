package com.seatflow.security;

/**
 * Fine-grained permissions used by RBAC access control decisions.
 *
 * <p>Permissions are derived from roles:
 * <ul>
 *   <li>ADMIN: all permissions</li>
 *   <li>RESEARCHER: PROJECT_READ, PROJECT_WRITE, EXPERIMENT_READ, EXPERIMENT_WRITE,
 *       RUN_READ, RUN_WRITE, MODEL_READ, MODEL_WRITE, PREDICTION_READ, PREDICTION_WRITE,
 *       BACKTEST_READ, BACKTEST_WRITE, AUDIT_READ</li>
 *   <li>VIEWER: PROJECT_READ, EXPERIMENT_READ, RUN_READ, MODEL_READ,
 *       PREDICTION_READ, BACKTEST_READ, AUDIT_READ, SYSTEM_READ</li>
 * </ul>
 */
public enum Permission {
    // Project
    PROJECT_READ,
    PROJECT_WRITE,
    PROJECT_DELETE,

    // Experiment
    EXPERIMENT_READ,
    EXPERIMENT_WRITE,

    // Run
    RUN_READ,
    RUN_WRITE,
    RUN_CANCEL,

    // Model
    MODEL_READ,
    MODEL_WRITE,
    MODEL_PROMOTE,
    MODEL_RETIRE,

    // Prediction
    PREDICTION_READ,
    PREDICTION_WRITE,

    // Backtest
    BACKTEST_READ,
    BACKTEST_WRITE,

    // Audit
    AUDIT_READ,

    // System
    SYSTEM_READ,
    SYSTEM_ADMIN
}
