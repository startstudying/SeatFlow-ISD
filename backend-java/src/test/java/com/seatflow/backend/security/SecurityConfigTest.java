package com.seatflow.backend.security;

import static org.hamcrest.Matchers.is;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.seatflow.backend.config.SeatFlowProperties;
import com.seatflow.backend.service.OrchestrationService;
import com.seatflow.backend.web.GlobalExceptionHandler;
import com.seatflow.backend.web.HealthController;
import com.seatflow.backend.web.PredictionController;
import com.seatflow.backend.web.PredictionRequest;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.context.annotation.Import;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.TestPropertySource;
import org.springframework.test.web.servlet.MockMvc;

@WebMvcTest({
        HealthController.class,
        PredictionController.class,
        GlobalExceptionHandler.class
})
@Import(SecurityConfig.class)
@EnableConfigurationProperties(SeatFlowProperties.class)
@ActiveProfiles("security-test")
@TestPropertySource(properties = {
        "seatflow.security.enabled=true",
        "seatflow.security.api-key=test-key",
        "seatflow.security.default-actor=test-actor",
        "seatflow.security.default-roles=ADMIN"
})
class SecurityConfigTest {
    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private OrchestrationService orchestrationService;

    @Test
    void healthIsPublicButApiIsProtected() throws Exception {
        Mockito.when(orchestrationService.readiness()).thenReturn(new ObjectMapper()
                .createObjectNode()
                .put("status", "UP"));

        mockMvc.perform(get("/api/v1/health"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code", is("OK")));

        mockMvc.perform(post("/api/v1/providers/sample/models/model-1/predict")
                        .contentType("application/json")
                        .content("{\"trade_date\":\"2026-01-15\",\"target_stock_id\":\"600000.SH\"}"))
                .andExpect(status().isForbidden());
    }

    @Test
    void validApiKeyAllowsRbacProtectedApi() throws Exception {
        Mockito.when(orchestrationService.prediction(eq("sample"), eq("model-1"), any(PredictionRequest.class)))
                .thenReturn(new ObjectMapper().createObjectNode().put("accepted", true));

        mockMvc.perform(post("/api/v1/providers/sample/models/model-1/predict")
                        .header("X-SeatFlow-Api-Key", "test-key")
                        .contentType("application/json")
                        .content("{\"trade_date\":\"2026-01-15\",\"target_stock_id\":\"600000.SH\"}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.accepted", is(true)));
    }
}
