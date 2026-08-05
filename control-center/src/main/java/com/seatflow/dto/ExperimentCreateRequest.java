package com.seatflow.dto;

import jakarta.validation.constraints.NotBlank;

/**
 * Request body for creating an experiment.
 *
 * @param name        experiment name (required)
 * @param projectId   parent project identifier (required)
 * @param description optional description
 */
public record ExperimentCreateRequest(
        @NotBlank String name,
        @NotBlank String projectId,
        String description) {
}
