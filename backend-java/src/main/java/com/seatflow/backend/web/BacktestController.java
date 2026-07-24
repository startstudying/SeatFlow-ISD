package com.seatflow.backend.web;

import com.fasterxml.jackson.databind.JsonNode;
import com.seatflow.backend.api.ApiResponse;
import com.seatflow.backend.api.TraceIds;
import com.seatflow.backend.service.OrchestrationService;
import jakarta.validation.constraints.Pattern;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Validated
@RestController
@RequestMapping("/api/v1/backtests")
public class BacktestController {
    private static final String SAFE_ID = "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$";
    private final OrchestrationService service;

    public BacktestController(OrchestrationService service) {
        this.service = service;
    }

    @GetMapping("/{runId}")
    public ApiResponse<JsonNode> backtestStatus(
            @PathVariable @Pattern(regexp = SAFE_ID) String runId
    ) {
        return ApiResponse.ok(service.backtestStatus(runId), TraceIds.next());
    }
}
