package com.seatflow.backend.web;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.seatflow.backend.api.ApiResponse;
import com.seatflow.backend.api.TraceIds;
import com.seatflow.backend.service.OrchestrationService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/audit")
public class AuditController {
    private final OrchestrationService service;

    public AuditController(OrchestrationService service) {
        this.service = service;
    }

    @GetMapping("/status")
    public ApiResponse<ObjectNode> status() {
        return ApiResponse.ok(service.auditStatus(), TraceIds.next());
    }
}
