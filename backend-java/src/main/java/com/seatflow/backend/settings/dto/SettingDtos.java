package com.seatflow.backend.settings.dto;

import com.fasterxml.jackson.databind.JsonNode;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public final class SettingDtos {
    private SettingDtos() {
    }

    public record UpsertSettingRequest(
            @NotBlank @Size(max = 64) String category,
            JsonNode value,
            Long expectedVersion
    ) implements VersionedWriteRequest {
    }

    public record SettingResponse(
            String key,
            String category,
            JsonNode value,
            long version
    ) {
    }
}
