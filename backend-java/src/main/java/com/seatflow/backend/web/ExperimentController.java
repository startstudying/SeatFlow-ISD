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
@RequestMapping("/api/v1")
public class ExperimentController {
    private static final String SAFE_ID = "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$";
    private final OrchestrationService service;

    public ExperimentController(OrchestrationService service) {
        this.service = service;
    }

    @GetMapping("/models/{modelId}")
    public ApiResponse<JsonNode> modelStatus(
            @PathVariable @Pattern(regexp = SAFE_ID) String modelId
    ) {
        return ApiResponse.ok(service.modelStatus(modelId), TraceIds.next());
    }

    @GetMapping("/experiments/runs/{runId}")
    public ApiResponse<JsonNode> experimentRun(
            @PathVariable @Pattern(regexp = SAFE_ID) String runId
    ) {
        return runStatus(runId);
    }

    @GetMapping("/runs/{runId}")
    public ApiResponse<JsonNode> runStatus(
            @PathVariable @Pattern(regexp = SAFE_ID) String runId
    ) {
        return ApiResponse.ok(service.runStatus(runId), TraceIds.next());
    }
}
