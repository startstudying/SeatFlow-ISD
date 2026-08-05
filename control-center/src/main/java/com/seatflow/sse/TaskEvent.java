package com.seatflow.sse;

import java.time.Instant;
import java.util.Map;

/**
 * SSE event payload for task lifecycle transitions, conforming to the
 * {@code TaskEvent} schema in the OpenAPI contract.
 *
 * @param eventType type of task lifecycle event
 * @param taskId    unique identifier of the task
 * @param timestamp ISO 8601 timestamp of when the event was emitted
 * @param payload   event-specific data (varies by eventType)
 * @param traceId   distributed trace identifier
 */
public record TaskEvent(
        TaskEventType eventType,
        String taskId,
        Instant timestamp,
        Map<String, Object> payload,
        String traceId) {
}
