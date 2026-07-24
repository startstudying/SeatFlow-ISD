package com.seatflow.backend.settings.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import java.time.OffsetDateTime;

@Entity
@Table(name = "sf_idempotency_record")
public class IdempotencyRecord {
    @Id
    @Column(name = "idempotency_key", length = 160)
    private String key;

    @Column(nullable = false, length = 64)
    private String operation;

    @Column(nullable = false, length = 128)
    private String resourceId;

    @Column(nullable = false)
    private OffsetDateTime createdAt;

    protected IdempotencyRecord() {
    }

    public IdempotencyRecord(String key, String operation, String resourceId) {
        this.key = key;
        this.operation = operation;
        this.resourceId = resourceId;
        this.createdAt = OffsetDateTime.now();
    }

    public String getKey() { return key; }
    public String getOperation() { return operation; }
    public String getResourceId() { return resourceId; }
}
