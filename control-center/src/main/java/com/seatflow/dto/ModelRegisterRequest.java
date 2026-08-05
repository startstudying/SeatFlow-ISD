package com.seatflow.dto;

import jakarta.validation.constraints.NotBlank;

/**
 * Request body for registering a model artifact.
 *
 * @param name        model family name (required)
 * @param version     semantic version (required)
 * @param artifactUri URI of the model artifact (required)
 */
public record ModelRegisterRequest(
        @NotBlank String name,
        @NotBlank String version,
        @NotBlank String artifactUri) {
}
