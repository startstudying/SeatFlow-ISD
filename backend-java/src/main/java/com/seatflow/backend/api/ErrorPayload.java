package com.seatflow.backend.api;

import java.util.List;

public record ErrorPayload(
        String error,
        String modelValidityStatus,
        String dataScope,
        String lineageHash,
        List<String> riskFlags
) {
    public static ErrorPayload failClosed(String error, List<String> riskFlags) {
        return new ErrorPayload(error, "unavailable", "unknown", null, riskFlags);
    }
}
