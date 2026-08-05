package com.seatflow.model;

import java.time.Instant;

/**
 * A SeatFlow project: the top-level container for datasets, experiments, runs and models.
 *
 * @param id          unique project identifier
 * @param name        human-readable project name
 * @param description optional free-form description
 * @param createdAt   creation timestamp (UTC)
 * @param updatedAt   last update timestamp (UTC)
 */
public record Project(
        String id,
        String name,
        String description,
        Instant createdAt,
        Instant updatedAt) {
}
