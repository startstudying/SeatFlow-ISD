package com.seatflow.backend.datacenter.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.seatflow.backend.datacenter.dto.DataCenterDtos;
import com.seatflow.backend.datacenter.entity.DataAsset;
import com.seatflow.backend.datacenter.entity.DataQualityReport;
import com.seatflow.backend.datacenter.entity.DatasetVersion;
import com.seatflow.backend.datacenter.entity.GraphSnapshot;
import com.seatflow.backend.datacenter.mapper.DataCenterMapper;
import com.seatflow.backend.datacenter.repository.DataAssetRepository;
import com.seatflow.backend.datacenter.repository.DataQualityReportRepository;
import com.seatflow.backend.datacenter.repository.DatasetVersionRepository;
import com.seatflow.backend.datacenter.repository.GraphSnapshotRepository;
import com.seatflow.backend.graph.GraphProjectionPublisher;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.time.LocalDate;
import java.util.UUID;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;

@Service
public class DataCenterService {
    private final DataAssetRepository assetRepository;
    private final DataQualityReportRepository qualityRepository;
    private final GraphSnapshotRepository graphRepository;
    private final DatasetVersionRepository versionRepository;
    private final ObjectMapper objectMapper;
    private final GraphProjectionPublisher graphProjectionPublisher;

    public DataCenterService(
            DataAssetRepository assetRepository,
            DataQualityReportRepository qualityRepository,
            GraphSnapshotRepository graphRepository,
            DatasetVersionRepository versionRepository,
            ObjectMapper objectMapper,
            GraphProjectionPublisher graphProjectionPublisher
    ) {
        this.assetRepository = assetRepository;
        this.qualityRepository = qualityRepository;
        this.graphRepository = graphRepository;
        this.versionRepository = versionRepository;
        this.objectMapper = objectMapper;
        this.graphProjectionPublisher = graphProjectionPublisher;
    }

    @Transactional(readOnly = true)
    public DataCenterDtos.OverviewResponse overview() {
        return new DataCenterDtos.OverviewResponse(
                assetRepository.count(), qualityRepository.count(), graphRepository.count(), 0
        );
    }

    @Transactional(readOnly = true)
    public DataCenterDtos.PageResponse<DataCenterDtos.AssetResponse> assets(
            String keyword, int page, int size, String sort
    ) {
        Pageable pageable = pageable(page, size, sort, "dataset");
        Page<DataAsset> result = StringUtils.hasText(keyword)
                ? assetRepository.findByDatasetContainingIgnoreCaseOrProviderContainingIgnoreCase(keyword, keyword, pageable)
                : assetRepository.findAll(pageable);
        return page(result, DataCenterMapper::asset);
    }

    @Transactional(readOnly = true)
    public DataCenterDtos.PageResponse<DataCenterDtos.QualityResponse> quality(
            String dataset, int page, int size, String sort
    ) {
        Pageable pageable = pageable(page, size, sort, "tradeDate");
        Page<DataQualityReport> result = StringUtils.hasText(dataset)
                ? qualityRepository.findByDatasetContainingIgnoreCase(dataset, pageable)
                : qualityRepository.findAll(pageable);
        return page(result, value -> DataCenterMapper.quality(value, objectMapper));
    }

    @Transactional(readOnly = true)
    public DataCenterDtos.PageResponse<DataCenterDtos.GraphResponse> graphs(
            String provider, int page, int size, String sort
    ) {
        Pageable pageable = pageable(page, size, sort, "tradeDate");
        Page<GraphSnapshot> result = StringUtils.hasText(provider)
                ? graphRepository.findByProviderContainingIgnoreCase(provider, pageable)
                : graphRepository.findAll(pageable);
        return page(result, DataCenterMapper::graph);
    }

    @Transactional(readOnly = true)
    public DataCenterDtos.PageResponse<DataCenterDtos.DatasetVersionResponse> versions(
            String dataset, int page, int size, String sort
    ) {
        Pageable pageable = pageable(page, size, sort, "createdAt");
        Page<DatasetVersion> result = StringUtils.hasText(dataset)
                ? versionRepository.findByDatasetContainingIgnoreCase(dataset, pageable)
                : versionRepository.findAll(pageable);
        return page(result, DataCenterMapper::version);
    }

    @Transactional
    public void persistSummary(String provider, JsonNode payload) {
        JsonNode datasets = payload.path("datasets");
        if (!datasets.isArray()) {
            return;
        }
        for (JsonNode dataset : datasets) {
            String store = text(dataset, "store", "unknown");
            String name = text(dataset, "dataset", "unknown");
            String id = stableId("asset", provider, store, name);
            DataAsset asset = assetRepository.findById(id)
                    .orElseGet(() -> new DataAsset(id, provider, name, store, 0));
            asset.refresh(
                    dataset.path("row_count").asLong(0),
                    text(payload, "data_scope", "unknown"),
                    text(payload, "seat_detail_scope", "unknown"),
                    nullableText(payload, "lineage_hash")
            );
            assetRepository.save(asset);
        }
    }

    @Transactional
    public void persistGraph(String provider, LocalDate tradeDate, JsonNode payload) {
        JsonNode graph = payload.path("graph");
        String id = stableId("graph", provider, tradeDate.toString());
        GraphSnapshot snapshot = graphRepository.findById(id)
                .orElseGet(() -> new GraphSnapshot(id, provider, tradeDate));
        snapshot.refresh(
                graph.path("nodes").isArray() ? graph.path("nodes").size() : 0,
                graph.path("edges").isArray() ? graph.path("edges").size() : 0,
                text(payload, "data_scope", text(payload.path("manifest").path("metadata"), "data_scope", "unknown")),
                text(payload, "seat_detail_scope", text(payload.path("manifest").path("metadata"), "seat_detail_scope", "unknown")),
                nullableText(payload, "lineage_hash")
        );
        GraphSnapshot savedSnapshot = graphRepository.save(snapshot);
        graphProjectionPublisher.publish(savedSnapshot, graph);
        String qualityId = stableId("quality", provider, "hetero_graph", tradeDate.toString());
        DataQualityReport report = new DataQualityReport(
                qualityId, provider, "hetero_graph", tradeDate,
                graph.path("nodes").isArray() && graph.path("edges").isArray(),
                BigDecimal.ONE, BigDecimal.ZERO, "[\"data_audit_not_verified\"]"
        );
        qualityRepository.save(report);
    }

    private Pageable pageable(int page, int size, String sort, String fallback) {
        int safePage = Math.max(0, page);
        int safeSize = Math.max(1, Math.min(size, 100));
        String property = StringUtils.hasText(sort) && sort.matches("[A-Za-z][A-Za-z0-9]*") ? sort : fallback;
        return PageRequest.of(safePage, safeSize, Sort.by(Sort.Direction.DESC, property));
    }

    private <T, R> DataCenterDtos.PageResponse<R> page(Page<T> source, java.util.function.Function<T, R> mapper) {
        return new DataCenterDtos.PageResponse<>(source.getContent().stream().map(mapper).toList(),
                source.getNumber(), source.getSize(), source.getTotalElements());
    }

    private static String stableId(String... parts) {
        return UUID.nameUUIDFromBytes(String.join("|", parts).getBytes(StandardCharsets.UTF_8)).toString();
    }

    private static String text(JsonNode node, String key, String fallback) {
        JsonNode value = node.path(key);
        return value.isTextual() && !value.asText().isBlank() ? value.asText() : fallback;
    }

    private static String nullableText(JsonNode node, String key) {
        JsonNode value = node.path(key);
        return value.isTextual() && !value.asText().isBlank() ? value.asText() : null;
    }
}
