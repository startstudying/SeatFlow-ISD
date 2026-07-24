package com.seatflow.backend.settings.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Lob;
import jakarta.persistence.Table;
import java.time.OffsetDateTime;

@Entity
@Table(name = "sf_admin_audit_log")
public class AuditLogEntry {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, length = 128)
    private String actor;

    @Column(nullable = false, length = 64)
    private String action;

    @Column(nullable = false, length = 64)
    private String resourceType;

    @Column(nullable = false, length = 128)
    private String resourceId;

    @Column(length = 128)
    private String requestId;

    @Column(length = 128)
    private String idempotencyKey;

    @Lob
    private String detailJson;

    @Column(nullable = false)
    private OffsetDateTime createdAt;

    protected AuditLogEntry() {
    }

    public AuditLogEntry(String actor, String action, String resourceType, String resourceId,
                         String requestId, String idempotencyKey, String detailJson) {
        this.actor = actor;
        this.action = action;
        this.resourceType = resourceType;
        this.resourceId = resourceId;
        this.requestId = requestId;
        this.idempotencyKey = idempotencyKey;
        this.detailJson = detailJson;
        this.createdAt = OffsetDateTime.now();
    }

    public Long getId() { return id; }
    public String getActor() { return actor; }
    public String getAction() { return action; }
    public String getResourceType() { return resourceType; }
    public String getResourceId() { return resourceId; }
    public String getRequestId() { return requestId; }
    public String getIdempotencyKey() { return idempotencyKey; }
    public String getDetailJson() { return detailJson; }
    public OffsetDateTime getCreatedAt() { return createdAt; }
}
