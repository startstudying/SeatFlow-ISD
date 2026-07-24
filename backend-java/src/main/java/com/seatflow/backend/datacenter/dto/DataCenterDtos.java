package com.seatflow.backend.datacenter.dto;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.util.List;

public final class DataCenterDtos {
    private DataCenterDtos() {
    }

    public record PageResponse<T>(List<T> items, int page, int size, long total) {
    }

    public record OverviewResponse(long assets, long qualityReports, long graphSnapshots, long syncTasks) {
    }

    public record AssetResponse(
            String id, String provider, String dataset, String store, String dataScope,
            String seatDetailScope, String lineageHash, long rowCount, long version
    ) {
    }

    public record QualityResponse(
            String id, String provider, String dataset, LocalDate tradeDate, boolean passed,
            BigDecimal completenessRate, BigDecimal duplicateRate, List<String> riskFlags, long version
    ) {
    }

    public record GraphResponse(
            String id, String provider, LocalDate tradeDate, long nodeCount, long edgeCount,
            String dataScope, String seatDetailScope, String lineageHash, long version
    ) {
    }

    public record DatasetVersionResponse(
            String id, String provider, String dataset, String versionTag, String artifactUri,
            String contentSha256, boolean active, long version
    ) {
    }

    public record CreateSyncTaskRequest(
            @NotBlank String provider,
            @NotBlank String dataset,
            @NotBlank String mode,
            LocalDate tradeDate,
            @Min(1) @Max(10_000) Integer requestedLimit
    ) {
    }

    public record SyncTaskResponse(
            String id, String provider, String dataset, String mode, String status, int progress,
            String message, List<String> riskFlags, long version, OffsetDateTime createdAt, OffsetDateTime updatedAt
    ) {
    }
}
