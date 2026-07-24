package com.seatflow.backend.monitoring.web;

import com.seatflow.backend.api.ApiResponse;
import com.seatflow.backend.api.TraceIds;
import com.seatflow.backend.monitoring.dto.ServiceMonitorDtos;
import com.seatflow.backend.monitoring.service.ServiceMonitoringService;
import jakarta.validation.Valid;
import jakarta.validation.constraints.Pattern;
import java.util.List;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Validated
@RestController
@RequestMapping("/api/v1/monitoring/services")
public class ServiceMonitoringController {
    private static final String SAFE_ID = "^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$";
    private final ServiceMonitoringService service;

    public ServiceMonitoringController(ServiceMonitoringService service) {
        this.service = service;
    }

    @GetMapping
    @PreAuthorize("hasAnyRole('ADMIN','AUDITOR','OPERATOR')")
    public ApiResponse<List<ServiceMonitorDtos.ServiceMonitorResponse>> list() {
        return ApiResponse.ok(service.list(), TraceIds.next());
    }

    @PutMapping("/{serviceId}")
    @PreAuthorize("hasRole('ADMIN')")
    public ApiResponse<ServiceMonitorDtos.ServiceMonitorResponse> upsert(
            @PathVariable @Pattern(regexp = SAFE_ID) String serviceId,
            @RequestHeader(value = "Idempotency-Key", required = false) String idempotencyKey,
            @Valid @RequestBody ServiceMonitorDtos.UpsertServiceMonitorRequest request
    ) {
        return ApiResponse.ok(service.upsert(serviceId, request, idempotencyKey), TraceIds.next());
    }
}
