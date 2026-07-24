package com.seatflow.backend.sync.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.Version;
import java.time.OffsetDateTime;
import java.util.List;

@Entity
@Table(name = "sf_sync_task")
public class SyncTask {
    @Id
    private String taskId;
    private String idempotencyKey;
    private String provider;
    private String dataset;
    private String mode;
    private String status;
    private int progress;
    private String message;
    private String riskFlagsJson;
    @Version
    private long version;
    private OffsetDateTime createdAt;
    private OffsetDateTime updatedAt;

    protected SyncTask() {
    }

    public SyncTask(String taskId, String idempotencyKey, String provider, String dataset, String mode) {
        this.taskId = taskId;
        this.idempotencyKey = idempotencyKey;
        this.provider = provider;
        this.dataset = dataset;
        this.mode = mode;
        this.status = "QUEUED";
        this.progress = 0;
        this.message = "任务已受理";
        this.riskFlagsJson = "[]";
    }

    public void running(String message) { this.status = "RUNNING"; this.progress = 15; this.message = message; }
    public void succeeded(String message) { this.status = "SUCCEEDED"; this.progress = 100; this.message = message; this.riskFlagsJson = "[]"; }
    public void failed(String message) { this.status = "FAILED"; this.progress = 100; this.message = message; this.riskFlagsJson = "[\"upstream_task_failed\"]"; }
    public void cancel() { if (!isTerminal()) { this.status = "CANCELLED"; this.progress = 100; this.message = "任务已取消"; } }
    public boolean isTerminal() { return List.of("SUCCEEDED", "FAILED", "CANCELLED").contains(status); }

    public String getTaskId() { return taskId; }
    public String getIdempotencyKey() { return idempotencyKey; }
    public String getProvider() { return provider; }
    public String getDataset() { return dataset; }
    public String getMode() { return mode; }
    public String getStatus() { return status; }
    public int getProgress() { return progress; }
    public String getMessage() { return message; }
    public String getRiskFlagsJson() { return riskFlagsJson; }
    public long getVersion() { return version; }
    public OffsetDateTime getCreatedAt() { return createdAt; }
    public OffsetDateTime getUpdatedAt() { return updatedAt; }
}
