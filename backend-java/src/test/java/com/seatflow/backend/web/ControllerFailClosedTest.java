package com.seatflow.backend.web;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.seatflow.backend.service.ApiException;
import com.seatflow.backend.service.FastApiClient;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.http.HttpStatus;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.web.servlet.MockMvc;

@WebMvcTest({
        HealthController.class,
        AuditController.class,
        PredictionController.class,
        BacktestController.class,
        ExperimentController.class,
        GlobalExceptionHandler.class
})
@AutoConfigureMockMvc(addFilters = false)
class ControllerFailClosedTest {
    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private com.seatflow.backend.service.OrchestrationService orchestrationService;

    @MockBean
    private FastApiClient fastApiClient;

    @Test
    void healthDeclaresLocalInternalScope() throws Exception {
        org.mockito.Mockito.when(orchestrationService.readiness()).thenReturn(new com.fasterxml.jackson.databind.ObjectMapper()
                .createObjectNode()
                .put("status", "UP")
                .put("scope", "local_internal_research")
                .put("auth_mode", "local-internal"));

        mockMvc.perform(get("/api/v1/health"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code", is("OK")))
                .andExpect(jsonPath("$.data.scope", is("local_internal_research")));
    }

    @Test
    void predictionDoesNotSynthesizeWhenFastApiFailsClosed() throws Exception {
        org.mockito.Mockito.when(orchestrationService.prediction(
                        org.mockito.Mockito.eq("sample"),
                        org.mockito.Mockito.eq("missing-model"),
                        org.mockito.Mockito.any(PredictionRequest.class)
                ))
                .thenThrow(new ApiException(
                        HttpStatus.CONFLICT,
                        "FASTAPI_FAIL_CLOSED",
                        "checkpoint metadata not found",
                        com.seatflow.backend.api.ErrorPayload.failClosed(
                                "checkpoint metadata not found",
                                java.util.List.of("checkpoint_missing")
                        )
                ));

        mockMvc.perform(post("/api/v1/providers/sample/models/missing-model/predict")
                        .contentType("application/json")
                        .content("""
                                {"trade_date":"2026-01-15","target_stock_id":"600000.SH"}
                                """))
                .andExpect(status().isConflict())
                .andExpect(jsonPath("$.code", is("FASTAPI_FAIL_CLOSED")))
                .andExpect(jsonPath("$.data.riskFlags[0]", is("checkpoint_missing")))
                .andExpect(jsonPath("$.data.error", containsString("checkpoint metadata not found")));
    }
}
