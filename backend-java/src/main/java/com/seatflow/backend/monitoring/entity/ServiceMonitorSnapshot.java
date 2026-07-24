package com.seatflow.backend.monitoring.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import com.seatflow.backend.settings.entity.BaseVersionedEntity;
import java.time.OffsetDateTime;

@Entity
@Table(name = "sf_service_monitor")
public class ServiceMonitorSnapshot extends BaseVersionedEntity {
    @Id
    @Column(name = "service_id", length = 128)
    private String serviceId;

    @Column(nullable = false, length = 32)
    private String status;

    @Column(nullable = false)
    private OffsetDateTime checkedAt;

    @Column(columnDefinition = "TEXT")
    private String metricsJson;

    protected ServiceMonitorSnapshot() {
    }

    public ServiceMonitorSnapshot(String serviceId, String status, String metricsJson) {
        this.serviceId = serviceId;
        this.status = status;
        this.metricsJson = metricsJson;
        this.checkedAt = OffsetDateTime.now();
    }

    public String getServiceId() { return serviceId; }
    public String getStatus() { return status; }
    public OffsetDateTime getCheckedAt() { return checkedAt; }
    public String getMetricsJson() { return metricsJson; }
}
