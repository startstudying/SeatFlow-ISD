package com.seatflow.model;

import java.time.Instant;

/**
 * An experiment within a project, grouping related runs for comparison.
 *
 * @param id          unique experiment identifier
 * @param name        human-readable experiment name
 * @param description optional free-form description
 * @param projectId   parent project identifier
 * @param createdAt   creation timestamp (UTC)
 */
public record Experiment(
        String id,
        String name,
        String description,
        String projectId,
        Instant createdAt) {
}
