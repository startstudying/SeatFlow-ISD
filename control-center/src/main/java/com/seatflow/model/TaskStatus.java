package com.seatflow.model;

/**
 * Lifecycle states for long-running tasks.
 *
 * <p>Allowed transitions:
 * <ul>
 *   <li>REQUESTED -> QUEUED -> RUNNING -> COMPLETED</li>
 *   <li>REQUESTED | QUEUED | RUNNING -> CANCELLED</li>
 *   <li>RUNNING -> FAILED</li>
 *   <li>REQUESTED | QUEUED | RUNNING -> BLOCKED</li>
 * </ul>
 * Terminal states (COMPLETED, CANCELLED, FAILED) must not transition further.
 */
public enum TaskStatus {
    REQUESTED,
    QUEUED,
    RUNNING,
    COMPLETED,
    CANCELLED,
    FAILED,
    BLOCKED
}
