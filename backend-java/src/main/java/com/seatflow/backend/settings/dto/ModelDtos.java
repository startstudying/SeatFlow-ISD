package com.seatflow.backend.settings.dto;

import com.fasterxml.jackson.databind.JsonNode;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public final class ModelDtos {
    private ModelDtos() {
    }

    public record UpsertModelRequest(
            @NotBlank @Size(max = 128) String displayName,
            @NotBlank @Size(max = 64) String provider,
            @NotBlank @Size(max = 64) String status,
            JsonNode parameters,
            Long expectedVersion
    ) implements VersionedWriteRequest {
    }

    public record ModelResponse(String id, String displayName, String provider, String status,
                                JsonNode parameters, long version) {
    }
}
