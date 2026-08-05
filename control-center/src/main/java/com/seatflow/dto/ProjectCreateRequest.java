package com.seatflow.dto;

import jakarta.validation.constraints.NotBlank;

/**
 * Request body for creating a project.
 *
 * @param name        project name (required)
 * @param description optional description
 */
public record ProjectCreateRequest(
        @NotBlank String name,
        String description) {
}
