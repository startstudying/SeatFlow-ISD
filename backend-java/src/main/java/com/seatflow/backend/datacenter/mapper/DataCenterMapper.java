package com.seatflow.backend.datacenter.mapper;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.seatflow.backend.datacenter.dto.DataCenterDtos;
import com.seatflow.backend.datacenter.entity.DataAsset;
import com.seatflow.backend.datacenter.entity.DataQualityReport;
import com.seatflow.backend.datacenter.entity.DatasetVersion;
import com.seatflow.backend.datacenter.entity.GraphSnapshot;
import com.seatflow.backend.sync.entity.SyncTask;
import java.util.List;

public final class DataCenterMapper {
    private static final TypeReference<List<String>> STRING_LIST = new TypeReference<>() { };

    private DataCenterMapper() {
    }

    public static DataCenterDtos.AssetResponse asset(DataAsset value) {
        return new DataCenterDtos.AssetResponse(value.getAssetId(), value.getProvider(), value.getDataset(),
                value.getStore(), value.getDataScope(), value.getSeatDetailScope(), value.getLineageHash(),
                value.getRowCount(), value.getVersion());
    }

    public static DataCenterDtos.QualityResponse quality(DataQualityReport value, ObjectMapper mapper) {
        return new DataCenterDtos.QualityResponse(value.getReportId(), value.getProvider(), value.getDataset(),
                value.getTradeDate(), value.isPassed(), value.getCompletenessRate(), value.getDuplicateRate(),
                list(value.getRiskFlagsJson(), mapper), value.getVersion());
    }

    public static DataCenterDtos.GraphResponse graph(GraphSnapshot value) {
        return new DataCenterDtos.GraphResponse(value.getSnapshotId(), value.getProvider(), value.getTradeDate(),
                value.getNodeCount(), value.getEdgeCount(), value.getDataScope(), value.getSeatDetailScope(),
                value.getLineageHash(), value.getVersion());
    }

    public static DataCenterDtos.DatasetVersionResponse version(DatasetVersion value) {
        return new DataCenterDtos.DatasetVersionResponse(value.getVersionId(), value.getProvider(), value.getDataset(),
                value.getVersion(), value.getArtifactUri(), value.getContentSha256(), value.isActive(), value.getVersionLock());
    }

    public static DataCenterDtos.SyncTaskResponse task(SyncTask value, ObjectMapper mapper) {
        return new DataCenterDtos.SyncTaskResponse(value.getTaskId(), value.getProvider(), value.getDataset(),
                value.getMode(), value.getStatus(), value.getProgress(), value.getMessage(),
                list(value.getRiskFlagsJson(), mapper), value.getVersion(), value.getCreatedAt(), value.getUpdatedAt());
    }

    private static List<String> list(String raw, ObjectMapper mapper) {
        try {
            return mapper.readValue(raw == null ? "[]" : raw, STRING_LIST);
        } catch (Exception ignored) {
            return List.of("metadata_unreadable");
        }
    }
}
