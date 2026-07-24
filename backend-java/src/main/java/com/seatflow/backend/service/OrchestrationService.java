package com.seatflow.backend.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.seatflow.backend.config.SeatFlowProperties;
import com.seatflow.backend.web.PredictionRequest;
import java.time.LocalDate;
import java.time.OffsetDateTime;
import org.springframework.stereotype.Service;

@Service
public class OrchestrationService {
    private final FastApiClient fastApiClient;
    private final SeatFlowProperties properties;
    private final ObjectMapper objectMapper;

    public OrchestrationService(
            FastApiClient fastApiClient,
            SeatFlowProperties properties,
            ObjectMapper objectMapper
    ) {
        this.fastApiClient = fastApiClient;
        this.properties = properties;
        this.objectMapper = objectMapper;
    }

    public ObjectNode readiness() {
        ObjectNode readiness = objectMapper.createObjectNode();
        readiness.put("status", "UP");
        readiness.put("service", "seatflow-backend-java");
        readiness.put("scope", properties.scope());
        readiness.put("auth_mode", properties.authMode());
        readiness.put("checked_at", OffsetDateTime.now().toString());
        return readiness;
    }

    public ObjectNode auditStatus() {
        try {
            JsonNode upstream = fastApiClient.auditStatus();
            if (upstream != null && upstream.isObject()) {
                ObjectNode status = ((ObjectNode) upstream).deepCopy();
                status.put("java_scope", properties.scope());
                status.put("auth_mode", properties.authMode());
                status.put("checked_at", OffsetDateTime.now().toString());
                return status;
            }
            return unavailableAuditStatus("FastAPI audit status response is empty or invalid");
        } catch (ApiException error) {
            return unavailableAuditStatus("FastAPI audit status is unavailable: " + error.code());
        }
    }

    private ObjectNode unavailableAuditStatus(String reason) {
        ObjectNode status = objectMapper.createObjectNode();
        status.put("scope", "none");
        status.put("java_scope", properties.scope());
        status.put("auth_mode", properties.authMode());
        status.put("engineering_complete", false);
        status.put("data_ready", false);
        status.put("model_valid", false);
        status.put("backtest_valid", false);
        status.put("competition_ready", false);
        status.putArray("blocked_reasons")
                .add(reason)
                .add("local/internal research scope only");
        status.put("checked_at", OffsetDateTime.now().toString());
        return status;
    }

    public JsonNode upstreamHealth() {
        return fastApiClient.health();
    }

    public JsonNode snapshotSummary(String provider) {
        return fastApiClient.snapshotSummary(provider);
    }

    public JsonNode graphSnapshot(String provider, LocalDate tradeDate) {
        return fastApiClient.graphSnapshot(provider, tradeDate);
    }

    public JsonNode modelStatus(String modelId) {
        return fastApiClient.modelStatus(modelId);
    }

    public JsonNode runStatus(String runId) {
        return fastApiClient.runStatus(runId);
    }

    public JsonNode backtestStatus(String runId) {
        return fastApiClient.backtestStatus(runId);
    }

    public JsonNode prediction(String provider, String modelId, PredictionRequest request) {
        return fastApiClient.prediction(provider, modelId, request);
    }
}
