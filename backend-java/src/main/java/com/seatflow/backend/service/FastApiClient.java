package com.seatflow.backend.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.seatflow.backend.web.PredictionRequest;
import java.time.LocalDate;

public interface FastApiClient {
    JsonNode health();

    JsonNode auditStatus();

    JsonNode snapshotSummary(String provider);

    JsonNode graphSnapshot(String provider, LocalDate tradeDate);

    JsonNode modelStatus(String modelId);

    JsonNode runStatus(String runId);

    JsonNode backtestStatus(String runId);

    JsonNode prediction(String provider, String modelId, PredictionRequest request);
}
