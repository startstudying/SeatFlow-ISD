package com.seatflow.model;

import java.time.Instant;
import java.util.Map;

/**
 * An audit event recording a state-changing operation.
 *
 * @param id        unique event identifier
 * @param actor     user or system that performed the action
 * @param action    stable action code (e.g. {@code project.create})
 * @param timestamp time the event occurred (UTC)
 * @param entity    optional structured entity reference
 */
public record AuditEvent(
        String id,
        String actor,
        String action,
        Instant timestamp,
        Map<String, Object> entity) {
}
