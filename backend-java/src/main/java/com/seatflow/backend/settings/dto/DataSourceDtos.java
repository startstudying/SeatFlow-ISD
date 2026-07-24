package com.seatflow.backend.settings.dto;

import com.fasterxml.jackson.databind.JsonNode;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public final class DataSourceDtos {
    private DataSourceDtos() {
    }

    public record UpsertDataSourceRequest(
            @NotBlank @Size(max = 128) String displayName,
            @NotBlank @Size(max = 64) String provider,
            @NotBlank @Size(max = 512) String endpoint,
            JsonNode metadata,
            @Size(max = 4096) String apiKey,
            Long expectedVersion
    ) implements VersionedWriteRequest {
    }

    public record DataSourceResponse(
            String id,
            String displayName,
            String provider,
            String endpoint,
            JsonNode metadata,
            String apiKeyMasked,
            String apiKeyKeyId,
            long version
    ) {
    }
}
