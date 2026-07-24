package com.seatflow.backend.settings.dto;

import com.fasterxml.jackson.databind.JsonNode;
import java.time.OffsetDateTime;

public record AuditLogResponse(
        Long id,
        String actor,
        String action,
        String resourceType,
        String resourceId,
        String requestId,
        String idempotencyKey,
        JsonNode detail,
        OffsetDateTime createdAt
) {
}
