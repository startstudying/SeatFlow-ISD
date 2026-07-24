package com.seatflow.backend.monitoring.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.seatflow.backend.api.ErrorPayload;
import com.seatflow.backend.monitoring.dto.ServiceMonitorDtos;
import com.seatflow.backend.monitoring.entity.ServiceMonitorSnapshot;
import com.seatflow.backend.monitoring.repository.ServiceMonitorSnapshotRepository;
import com.seatflow.backend.service.ApiException;
import com.seatflow.backend.settings.service.AuditLogService;
import com.seatflow.backend.settings.service.IdempotencyService;
import java.util.List;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class ServiceMonitoringService {
    private final ServiceMonitorSnapshotRepository repository;
    private final ObjectMapper objectMapper;
    private final IdempotencyService idempotencyService;
    private final AuditLogService auditLogService;

    public ServiceMonitoringService(ServiceMonitorSnapshotRepository repository, ObjectMapper objectMapper,
                                    IdempotencyService idempotencyService, AuditLogService auditLogService) {
        this.repository = repository;
        this.objectMapper = objectMapper;
        this.idempotencyService = idempotencyService;
        this.auditLogService = auditLogService;
    }

    @Transactional(readOnly = true)
    public List<ServiceMonitorDtos.ServiceMonitorResponse> list() {
        return repository.findAll().stream().map(this::toResponse).toList();
    }

    @Transactional
    public ServiceMonitorDtos.ServiceMonitorResponse upsert(
            String serviceId,
            ServiceMonitorDtos.UpsertServiceMonitorRequest request,
            String idempotencyKey
    ) {
        if (!idempotencyService.claim(idempotencyKey, "monitoring.upsert", serviceId)) {
            return toResponse(repository.findById(serviceId).orElseThrow(() -> notFound(serviceId)));
        }
        ServiceMonitorSnapshot existing = repository.findById(serviceId).orElse(null);
        long actual = existing == null ? 0 : existing.getVersion();
        if (request.expectedVersion() == null) {
            throw invalid("expectedVersion is required for write requests");
        }
        if (actual != request.expectedVersion()) {
            throw new ApiException(HttpStatus.CONFLICT, "VERSION_CONFLICT",
                    "expected version " + request.expectedVersion() + " but current version is " + actual,
                    ErrorPayload.failClosed("version conflict", List.of("version_conflict")));
        }
        ServiceMonitorSnapshot saved = repository.save(new ServiceMonitorSnapshot(serviceId, request.status(), json(request.metrics())));
        auditLogService.record("UPSERT", "SERVICE_MONITOR", serviceId, idempotencyKey,
                objectMapper.createObjectNode().put("status", request.status()));
        return toResponse(saved);
    }

    private ServiceMonitorDtos.ServiceMonitorResponse toResponse(ServiceMonitorSnapshot snapshot) {
        return new ServiceMonitorDtos.ServiceMonitorResponse(
                snapshot.getServiceId(),
                snapshot.getStatus(),
                snapshot.getCheckedAt(),
                readJson(snapshot.getMetricsJson()),
                snapshot.getVersion()
        );
    }

    private ApiException notFound(String serviceId) {
        return new ApiException(HttpStatus.NOT_FOUND, "MONITORING_NOT_FOUND", "service monitor not found: " + serviceId,
                ErrorPayload.failClosed("service monitor not found", List.of("not_found")));
    }

    private ApiException invalid(String message) {
        return new ApiException(HttpStatus.BAD_REQUEST, "INVALID_MONITORING_REQUEST", message,
                ErrorPayload.failClosed(message, List.of("invalid_monitoring_request")));
    }

    private String json(JsonNode node) {
        return node == null ? "{}" : node.toString();
    }

    private JsonNode readJson(String value) {
        try {
            return objectMapper.readTree(value == null ? "{}" : value);
        } catch (Exception error) {
            return objectMapper.createObjectNode();
        }
    }
}
