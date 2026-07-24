package com.seatflow.backend.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.seatflow.backend.config.HttpClientConfig;
import com.seatflow.backend.config.SeatFlowProperties;
import com.seatflow.backend.web.PredictionRequest;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpStatus;
import org.springframework.web.client.RestClient;

class HttpFastApiClientTest {
    private HttpServer server;

    @AfterEach
    void stopServer() {
        if (server != null) {
            server.stop(0);
        }
    }

    @Test
    void proxiesPersistedPredictionContractWithoutGeneratingPrediction() throws Exception {
        server = startServer(200, """
                {
                  "model_validity_status": "valid",
                  "data_scope": "real",
                  "lineage_hash": "abc123",
                  "risk_flags": [],
                  "prediction": {"rise_probability": 0.61}
                }
                """);
        HttpFastApiClient client = client("http://localhost:" + server.getAddress().getPort());

        JsonNode response = client.prediction(
                "sample",
                "seat-hgt-v1",
                new PredictionRequest("2026-01-15", "600000.SH")
        );

        assertThat(response.get("prediction").get("rise_probability").decimalValue())
                .isEqualByComparingTo("0.61");
        assertThat(response.get("lineage_hash").asText()).isEqualTo("abc123");
    }

    @Test
    void propagatesFastApiFailClosedPayload() throws Exception {
        server = startServer(409, """
                {
                  "error": "model is not marked valid",
                  "model_validity_status": "unavailable",
                  "data_scope": "unknown",
                  "lineage_hash": null,
                  "risk_flags": ["model_not_valid"]
                }
                """);
        HttpFastApiClient client = client("http://localhost:" + server.getAddress().getPort());

        assertThatThrownBy(() -> client.backtestStatus("run-1"))
                .isInstanceOf(ApiException.class)
                .satisfies(error -> {
                    ApiException apiError = (ApiException) error;
                    assertThat(apiError.statusCode()).isEqualTo(HttpStatus.CONFLICT);
                    assertThat(apiError.payload().riskFlags()).contains("model_not_valid");
                });
    }

    @Test
    void fastApiUnavailableFailsClosed() {
        HttpFastApiClient client = client("http://localhost:1");

        assertThatThrownBy(client::health)
                .isInstanceOf(ApiException.class)
                .satisfies(error -> {
                    ApiException apiError = (ApiException) error;
                    assertThat(apiError.statusCode()).isEqualTo(HttpStatus.SERVICE_UNAVAILABLE);
                    assertThat(apiError.payload().riskFlags()).contains("fastapi_unavailable");
                });
    }

    private static HttpFastApiClient client(String baseUrl) {
        SeatFlowProperties properties = new SeatFlowProperties(
                "local_internal_research",
                "local-internal",
                new SeatFlowProperties.FastApi(baseUrl, Duration.ofSeconds(1), Duration.ofSeconds(1))
        );
        RestClient restClient = new HttpClientConfig().fastApiRestClient(properties, RestClient.builder());
        return new HttpFastApiClient(restClient, new ObjectMapper());
    }

    private static HttpServer startServer(int status, String body) throws IOException {
        HttpServer httpServer = HttpServer.create(new InetSocketAddress("localhost", 0), 0);
        httpServer.createContext("/", exchange -> {
            byte[] payload = body.getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.sendResponseHeaders(status, payload.length);
            exchange.getResponseBody().write(payload);
            exchange.close();
        });
        httpServer.start();
        return httpServer;
    }
}
