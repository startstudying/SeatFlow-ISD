package com.seatflow.backend.settings.dto;

import com.fasterxml.jackson.databind.JsonNode;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public final class ExperimentDtos {
    private ExperimentDtos() {
    }

    public record UpsertExperimentRequest(
            @NotBlank @Size(max = 128) String name,
            @NotBlank @Size(max = 64) String status,
            JsonNode config,
            Long expectedVersion
    ) implements VersionedWriteRequest {
    }

    public record ExperimentResponse(String id, String name, String status, JsonNode config, long version) {
    }
}
