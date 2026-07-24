package com.seatflow.backend.monitoring.dto;

import com.fasterxml.jackson.databind.JsonNode;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import java.time.OffsetDateTime;

public final class ServiceMonitorDtos {
    private ServiceMonitorDtos() {
    }

    public record UpsertServiceMonitorRequest(
            @NotBlank @Size(max = 32) String status,
            JsonNode metrics,
            Long expectedVersion
    ) {
    }

    public record ServiceMonitorResponse(
            String serviceId,
            String status,
            OffsetDateTime checkedAt,
            JsonNode metrics,
            long version
    ) {
    }
}
