package com.seatflow.backend.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.seatflow.backend.api.ErrorPayload;
import com.seatflow.backend.web.PredictionRequest;
import java.time.LocalDate;
import java.util.List;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;

@Component
public class HttpFastApiClient implements FastApiClient {
    private final RestClient restClient;
    private final ObjectMapper objectMapper;

    public HttpFastApiClient(RestClient fastApiRestClient, ObjectMapper objectMapper) {
        this.restClient = fastApiRestClient;
        this.objectMapper = objectMapper;
    }

    @Override
    public JsonNode health() {
        return get("/health");
    }

    @Override
    public JsonNode auditStatus() {
        return get("/api/v1/audit/status");
    }

    @Override
    public JsonNode snapshotSummary(String provider) {
        return get("/api/v1/providers/{provider}/snapshot-summary", provider);
    }

    @Override
    public JsonNode graphSnapshot(String provider, LocalDate tradeDate) {
        return get("/api/v1/providers/{provider}/graph/{tradeDate}", provider, tradeDate);
    }

    @Override
    public JsonNode modelStatus(String modelId) {
        return get("/api/v1/models/{modelId}", modelId);
    }

    @Override
    public JsonNode runStatus(String runId) {
        return get("/api/v1/runs/{runId}", runId);
    }

    @Override
    public JsonNode backtestStatus(String runId) {
        return get("/api/v1/backtests/{runId}", runId);
    }

    @Override
    public JsonNode prediction(String provider, String modelId, PredictionRequest request) {
        return exchange(() -> restClient.post()
                .uri("/api/v1/providers/{provider}/models/{modelId}/predict", provider, modelId)
                .body(request)
                .retrieve()
                .body(JsonNode.class));
    }

    private JsonNode get(String uri, Object... uriVariables) {
        return exchange(() -> restClient.get()
                .uri(uri, uriVariables)
                .retrieve()
                .body(JsonNode.class));
    }

    private JsonNode exchange(RestCall call) {
        try {
            JsonNode response = call.execute();
            if (response == null) {
                throw failClosed(HttpStatus.BAD_GATEWAY, "UPSTREAM_EMPTY_RESPONSE",
                        "FastAPI returned an empty response", List.of("fastapi_empty_response"));
            }
            return response;
        } catch (RestClientResponseException error) {
            throw upstreamStatus(error);
        } catch (RestClientException error) {
            throw failClosed(HttpStatus.SERVICE_UNAVAILABLE, "FASTAPI_UNAVAILABLE",
                    "FastAPI service is unavailable", List.of("fastapi_unavailable"));
        }
    }

    private ApiException upstreamStatus(RestClientResponseException error) {
        ErrorPayload payload = parseErrorPayload(error);
        return new ApiException(error.getStatusCode(), "FASTAPI_FAIL_CLOSED", payload.error(), payload);
    }

    private ErrorPayload parseErrorPayload(RestClientResponseException error) {
        try {
            JsonNode node = objectMapper.readTree(error.getResponseBodyAsString());
            String message = text(node, "error", "detail", error.getMessage());
            return new ErrorPayload(
                    message,
                    text(node, "model_validity_status", "modelValidityStatus", "unavailable"),
                    text(node, "data_scope", "dataScope", "unknown"),
                    nullableText(node, "lineage_hash", "lineageHash"),
                    riskFlags(node)
            );
        } catch (Exception parseError) {
            return ErrorPayload.failClosed(error.getMessage(), List.of("fastapi_error_response_unparseable"));
        }
    }

    private ApiException failClosed(HttpStatusCode status, String code, String message, List<String> riskFlags) {
        return new ApiException(status, code, message, ErrorPayload.failClosed(message, riskFlags));
    }

    private static String text(JsonNode node, String snakeKey, String camelKey, String fallback) {
        String value = nullableText(node, snakeKey, camelKey);
        return value == null || value.isBlank() ? fallback : value;
    }

    private static String nullableText(JsonNode node, String snakeKey, String camelKey) {
        JsonNode value = node.get(snakeKey);
        if (value == null) {
            value = node.get(camelKey);
        }
        return value != null && value.isTextual() ? value.asText() : null;
    }

    private static List<String> riskFlags(JsonNode node) {
        JsonNode flags = node.get("risk_flags");
        if (flags == null) {
            flags = node.get("riskFlags");
        }
        if (flags == null || !flags.isArray()) {
            return List.of("fastapi_fail_closed");
        }
        java.util.ArrayList<String> parsed = new java.util.ArrayList<>();
        flags.forEach(flag -> {
            if (flag.isTextual() && !flag.asText().isBlank()) {
                parsed.add(flag.asText());
            }
        });
        return parsed.isEmpty() ? List.of("fastapi_fail_closed") : List.copyOf(parsed);
    }

    @FunctionalInterface
    private interface RestCall {
        JsonNode execute();
    }
}
