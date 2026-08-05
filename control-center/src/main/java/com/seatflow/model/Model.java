package com.seatflow.model;

import java.time.Instant;

/**
 * A registered model artifact tracked by the model registry.
 *
 * @param id          unique model identifier
 * @param name        model family name
 * @param version     semantic version string
 * @param stage       current promotion stage
 * @param artifactUri URI of the model artifact
 * @param createdAt   registration timestamp (UTC)
 */
public record Model(
        String id,
        String name,
        String version,
        ModelStage stage,
        String artifactUri,
        Instant createdAt) {
}
