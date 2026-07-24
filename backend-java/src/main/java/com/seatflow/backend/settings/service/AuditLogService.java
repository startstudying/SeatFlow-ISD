package com.seatflow.backend.settings.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.seatflow.backend.common.web.RequestContext;
import com.seatflow.backend.settings.dto.AuditLogResponse;
import com.seatflow.backend.settings.entity.AuditLogEntry;
import com.seatflow.backend.settings.repository.AuditLogEntryRepository;
import java.util.List;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class AuditLogService {
    private final AuditLogEntryRepository repository;
    private final ObjectMapper objectMapper;

    public AuditLogService(AuditLogEntryRepository repository, ObjectMapper objectMapper) {
        this.repository = repository;
        this.objectMapper = objectMapper;
    }

    @Transactional
    public void record(String action, String resourceType, String resourceId, String idempotencyKey, JsonNode detail) {
        String detailJson = detail == null ? "{}" : detail.toString();
        repository.save(new AuditLogEntry(
                RequestActor.current(),
                action,
                resourceType,
                resourceId,
                RequestContext.requestId(),
                idempotencyKey,
                detailJson
        ));
    }

    @Transactional(readOnly = true)
    public List<AuditLogResponse> latest(int limit) {
        int safeLimit = Math.max(1, Math.min(limit, 200));
        return repository.findAll(PageRequest.of(0, safeLimit, Sort.by(Sort.Direction.DESC, "createdAt")))
                .stream()
                .map(this::toResponse)
                .toList();
    }

    private AuditLogResponse toResponse(AuditLogEntry entry) {
        return new AuditLogResponse(
                entry.getId(),
                entry.getActor(),
                entry.getAction(),
                entry.getResourceType(),
                entry.getResourceId(),
                entry.getRequestId(),
                entry.getIdempotencyKey(),
                readJson(entry.getDetailJson()),
                entry.getCreatedAt()
        );
    }

    private JsonNode readJson(String value) {
        try {
            return objectMapper.readTree(value == null ? "{}" : value);
        } catch (Exception error) {
            return objectMapper.createObjectNode();
        }
    }
}
