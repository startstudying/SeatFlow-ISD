package com.seatflow.backend.settings.web;

import com.seatflow.backend.api.ApiResponse;
import com.seatflow.backend.api.TraceIds;
import com.seatflow.backend.settings.dto.AuditLogResponse;
import com.seatflow.backend.settings.service.AuditLogService;
import java.util.List;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping({"/api/v1/admin/audit-logs", "/api/v1/audit-logs"})
@PreAuthorize("hasAnyRole('ADMIN','AUDITOR')")
public class AuditLogController {
    private final AuditLogService service;

    public AuditLogController(AuditLogService service) {
        this.service = service;
    }

    @GetMapping
    public ApiResponse<List<AuditLogResponse>> latest(@RequestParam(defaultValue = "50") int limit) {
        return ApiResponse.ok(service.latest(limit), TraceIds.next());
    }
}
