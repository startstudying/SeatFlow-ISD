package com.seatflow.backend.common.audit;

import java.util.List;

public record AuditEvent(
        String action,
        String resourceType,
        String resourceId,
        AuditOutcome outcome,
        List<String> riskFlags
) {
    public AuditEvent {
        if (riskFlags == null) {
            riskFlags = List.of();
        } else {
            riskFlags = List.copyOf(riskFlags);
        }
    }
}
