package com.seatflow.dto;

import jakarta.validation.constraints.NotBlank;
import java.util.Map;

/**
 * Request body for submitting a run.
 *
 * @param experimentId parent experiment identifier (required)
 * @param config       run configuration key-value pairs (required)
 */
public record RunCreateRequest(
        @NotBlank String experimentId,
        Map<String, Object> config) {
}
