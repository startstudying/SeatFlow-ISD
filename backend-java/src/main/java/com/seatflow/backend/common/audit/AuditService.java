package com.seatflow.backend.common.audit;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.seatflow.backend.common.web.RequestContext;
import com.seatflow.backend.config.SeatFlowProperties;
import java.util.List;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class AuditService {
    private final ApiAuditLogRepository repository;
    private final ObjectMapper objectMapper;
    private final SeatFlowProperties properties;

    public AuditService(ApiAuditLogRepository repository, ObjectMapper objectMapper, SeatFlowProperties properties) {
        this.repository = repository;
        this.objectMapper = objectMapper;
        this.properties = properties;
    }

    @Transactional
    public ApiAuditLog record(AuditEvent event) {
        ApiAuditLog log = new ApiAuditLog(
                traceId(),
                actor(),
                event.action(),
                event.resourceType(),
                event.resourceId(),
                event.outcome(),
                toJson(event.riskFlags())
        );
        return repository.save(log);
    }

    private String traceId() {
        String requestId = RequestContext.requestId();
        return requestId == null || requestId.isBlank() ? "system" : requestId;
    }

    private String actor() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        if (authentication != null && authentication.isAuthenticated() && authentication.getName() != null) {
            return authentication.getName();
        }
        return properties.security().defaultActor();
    }

    private String toJson(List<String> riskFlags) {
        try {
            return objectMapper.writeValueAsString(riskFlags);
        } catch (JsonProcessingException error) {
            return "[\"audit_serialization_failed\"]";
        }
    }
}
