package com.seatflow.model;

/**
 * Lifecycle states for a run, mirroring the OpenAPI {@code RunStatus} enum.
 */
public enum RunStatus {
    PENDING,
    RUNNING,
    SUCCEEDED,
    FAILED,
    CANCELLED
}
