package com.seatflow.backend.web;

import com.fasterxml.jackson.databind.JsonNode;
import com.seatflow.backend.api.ApiResponse;
import com.seatflow.backend.api.TraceIds;
import com.seatflow.backend.service.OrchestrationService;
import java.time.LocalDate;
import jakarta.validation.constraints.Pattern;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@Validated
@RequestMapping("/api/v1/providers/{provider}")
public class SnapshotController {
    private final OrchestrationService service;

    public SnapshotController(OrchestrationService service) {
        this.service = service;
    }

    @GetMapping("/snapshot-summary")
    public ApiResponse<JsonNode> snapshotSummary(
            @PathVariable @Pattern(regexp = "^[a-z][a-z0-9_-]{0,31}$") String provider
    ) {
        return ApiResponse.ok(service.snapshotSummary(provider), TraceIds.next());
    }

    @GetMapping("/graph/{tradeDate}")
    public ApiResponse<JsonNode> graphSnapshot(
            @PathVariable @Pattern(regexp = "^[a-z][a-z0-9_-]{0,31}$") String provider,
            @PathVariable @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate tradeDate
    ) {
        return ApiResponse.ok(service.graphSnapshot(provider, tradeDate), TraceIds.next());
    }
}
