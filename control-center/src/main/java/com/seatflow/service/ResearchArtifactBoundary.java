package com.seatflow.service;

import com.seatflow.model.ErrorResponse;
import java.time.Instant;
import java.util.Map;
import java.util.UUID;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

/**
 * Fail-closed boundary used until the Control Center has a verified connection
 * to the canonical research-engine artifact gateway.
 */
@Service
public class ResearchArtifactBoundary {

    /** Stable error code exposed to API clients. */
    public static final String ERROR_CODE = "RESEARCH_ARTIFACT_UNAVAILABLE";

    /** Stable reason for the current blocked state. */
    public static final String BLOCK_REASON = "VERIFIED_RESEARCH_GATEWAY_NOT_CONFIGURED";

    /**
     * Build a machine-readable blocked response without manufacturing research
     * results or task handles.
     *
     * @param operation control-plane operation that cannot be completed
     * @return a 503 SeatFlow error envelope
     */
    public ResponseEntity<ErrorResponse> unavailable(String operation) {
        Map<String, Object> details = Map.of(
                "status", "BLOCKED",
                "operation", operation,
                "reason", BLOCK_REASON,
                "gateway", "research-engine:/research/v1");
        ErrorResponse error = new ErrorResponse(
                ERROR_CODE,
                "Verified research artifacts are unavailable for this operation.",
                details,
                UUID.randomUUID().toString(),
                Instant.now(),
                true,
                "Configure the canonical research-engine gateway and retry after its artifact lineage is verified.");
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(error);
    }
}
