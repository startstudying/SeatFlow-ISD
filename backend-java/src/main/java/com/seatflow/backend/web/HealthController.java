package com.seatflow.backend.web;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.seatflow.backend.api.ApiResponse;
import com.seatflow.backend.api.TraceIds;
import com.seatflow.backend.service.OrchestrationService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class HealthController {
    private final OrchestrationService service;

    public HealthController(OrchestrationService service) {
        this.service = service;
    }

    @GetMapping({"/health", "/api/v1/health"})
    public ApiResponse<ObjectNode> health() {
        return ApiResponse.ok(service.readiness(), TraceIds.next());
    }

    @GetMapping({"/ready", "/api/v1/readiness"})
    public ApiResponse<ObjectNode> readiness() {
        return ApiResponse.ok(service.readiness(), TraceIds.next());
    }

    @GetMapping("/api/v1/upstream/fastapi/health")
    public ApiResponse<JsonNode> upstreamHealth() {
        return ApiResponse.ok(service.upstreamHealth(), TraceIds.next());
    }
}
