package com.seatflow.sse;

import com.seatflow.model.TaskStatus;

/**
 * Event type enumeration for task lifecycle SSE events, mirroring the
 * {@code TaskEvent.eventType} enum in the OpenAPI contract.
 */
public enum TaskEventType {
    TASK_CREATED,
    TASK_QUEUED,
    TASK_STARTED,
    TASK_PROGRESS,
    TASK_COMPLETED,
    TASK_FAILED,
    TASK_CANCELLED,
    TASK_BLOCKED
}
