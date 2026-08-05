package com.seatflow.model;

import java.time.Instant;
import java.util.Map;

/**
 * Error envelope conforming to the {@code SeatFlowError} OpenAPI schema.
 *
 * @param code          stable machine-readable error code
 * @param message       human-readable error description
 * @param details       optional structured details
 * @param traceId       distributed trace id for correlation
 * @param timestamp     time the error occurred (UTC)
 * @param retryable     whether the caller may retry the request
 * @param remediation   suggested next steps
 */
public record ErrorResponse(
        String code,
        String message,
        Map<String, Object> details,
        String traceId,
        Instant timestamp,
        boolean retryable,
        String remediation) {
}
