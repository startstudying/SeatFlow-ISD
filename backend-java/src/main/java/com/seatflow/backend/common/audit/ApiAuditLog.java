package com.seatflow.backend.common.audit;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.PrePersist;
import jakarta.persistence.Table;
import java.time.OffsetDateTime;

@Entity
@Table(name = "sf_api_audit_log")
public class ApiAuditLog {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "audit_log_id")
    private Long id;

    @Column(name = "trace_id", nullable = false, length = 64)
    private String traceId;

    @Column(name = "actor", nullable = false, length = 128)
    private String actor;

    @Column(name = "action", nullable = false, length = 128)
    private String action;

    @Column(name = "resource_type", nullable = false, length = 64)
    private String resourceType;

    @Column(name = "resource_id", length = 128)
    private String resourceId;

    @Enumerated(EnumType.STRING)
    @Column(name = "outcome", nullable = false, length = 32)
    private AuditOutcome outcome;

    @Column(name = "risk_flags_json", columnDefinition = "json")
    private String riskFlagsJson;

    @Column(name = "created_at", nullable = false)
    private OffsetDateTime createdAt;

    protected ApiAuditLog() {
    }

    public ApiAuditLog(
            String traceId,
            String actor,
            String action,
            String resourceType,
            String resourceId,
            AuditOutcome outcome,
            String riskFlagsJson
    ) {
        this.traceId = traceId;
        this.actor = actor;
        this.action = action;
        this.resourceType = resourceType;
        this.resourceId = resourceId;
        this.outcome = outcome;
        this.riskFlagsJson = riskFlagsJson;
    }

    @PrePersist
    void prePersist() {
        if (createdAt == null) {
            createdAt = OffsetDateTime.now();
        }
    }

    public Long id() {
        return id;
    }

    public String traceId() {
        return traceId;
    }

    public AuditOutcome outcome() {
        return outcome;
    }
}
