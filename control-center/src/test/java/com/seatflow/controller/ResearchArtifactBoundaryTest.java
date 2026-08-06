package com.seatflow.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.seatflow.dto.ExperimentCreateRequest;
import com.seatflow.dto.ModelRegisterRequest;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;
import org.springframework.test.web.servlet.ResultActions;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class ResearchArtifactBoundaryTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    void datasetEndpointsDoNotInventCatalogEntriesOrTaskHandles() throws Exception {
        expectResearchArtifactUnavailable(mockMvc.perform(get("/api/v1/datasets")), "datasets.list");
        expectResearchArtifactUnavailable(mockMvc.perform(get("/api/v1/datasets/ds-missing")), "datasets.get");
        expectResearchArtifactUnavailable(
                mockMvc.perform(post("/api/v1/datasets/ds-missing/build")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{}")),
                "datasets.build");
        expectResearchArtifactUnavailable(
                mockMvc.perform(post("/api/v1/datasets/ds-missing/validate")),
                "datasets.validate");
    }

    @Test
    void predictionEndpointsDoNotInventNumericOutputsOrTaskHandles() throws Exception {
        expectResearchArtifactUnavailable(mockMvc.perform(get("/api/v1/predictions")), "predictions.list");
        expectResearchArtifactUnavailable(
                mockMvc.perform(get("/api/v1/predictions/prediction-missing")),
                "predictions.get");
        expectResearchArtifactUnavailable(
                mockMvc.perform(post("/api/v1/predictions")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"modelId\":\"model-missing\",\"inputs\":{}}")),
                "predictions.create");
        expectResearchArtifactUnavailable(
                mockMvc.perform(post("/api/v1/predictions/batch")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"modelId\":\"model-missing\",\"datasetId\":\"dataset-missing\"}")),
                "predictions.createBatch");
    }

    @Test
    void backtestEndpointsDoNotInventCompletedReportsOrTaskHandles() throws Exception {
        expectResearchArtifactUnavailable(mockMvc.perform(get("/api/v1/backtests")), "backtests.list");
        expectResearchArtifactUnavailable(
                mockMvc.perform(post("/api/v1/backtests")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{}")),
                "backtests.create");
        expectResearchArtifactUnavailable(mockMvc.perform(get("/api/v1/backtests/bt-missing")), "backtests.get");
        expectResearchArtifactUnavailable(
                mockMvc.perform(get("/api/v1/backtests/bt-missing/report")),
                "backtests.getReport");
    }

    @Test
    void runExecutionEndpointsDoNotIssueUntrackedTaskHandlesOrArtifactLists() throws Exception {
        ExperimentCreateRequest request = new ExperimentCreateRequest("Blocked Run", "proj-1", null);
        MvcResult createResult = mockMvc.perform(post("/api/v1/experiments")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isCreated())
                .andReturn();
        String experimentId = objectMapper.readTree(createResult.getResponse().getContentAsString()).get("id").asText();

        expectResearchArtifactUnavailable(
                mockMvc.perform(post("/api/v1/runs")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"experimentId\":\"" + experimentId + "\",\"config\":{}}")),
                "runs.create");
        expectResearchArtifactUnavailable(
                mockMvc.perform(post("/api/v1/runs/run-missing/cancel")),
                "runs.cancel");
        expectResearchArtifactUnavailable(
                mockMvc.perform(get("/api/v1/runs/run-missing/artifacts")),
                "runs.listArtifacts");
    }

    @Test
    void modelLifecycleDoesNotClaimUnverifiedPromotionOrIssueRetirementTasks() throws Exception {
        ModelRegisterRequest request = new ModelRegisterRequest(
                "unverified-model", "1.0", "s3://artifacts/unverified-model");
        MvcResult createResult = mockMvc.perform(post("/api/v1/models")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isCreated())
                .andReturn();
        String modelId = objectMapper.readTree(createResult.getResponse().getContentAsString()).get("id").asText();

        expectResearchArtifactUnavailable(
                mockMvc.perform(post("/api/v1/models/" + modelId + "/promote")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"stage\":\"STAGING\"}")),
                "models.promote");
        expectResearchArtifactUnavailable(
                mockMvc.perform(post("/api/v1/models/" + modelId + "/retire")),
                "models.retire");
    }

    private static void expectResearchArtifactUnavailable(ResultActions result, String operation) throws Exception {
        result.andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.code").value("RESEARCH_ARTIFACT_UNAVAILABLE"))
                .andExpect(jsonPath("$.message").isNotEmpty())
                .andExpect(jsonPath("$.details.status").value("BLOCKED"))
                .andExpect(jsonPath("$.details.operation").value(operation))
                .andExpect(jsonPath("$.details.reason").value("VERIFIED_RESEARCH_GATEWAY_NOT_CONFIGURED"))
                .andExpect(jsonPath("$.traceId").isNotEmpty())
                .andExpect(jsonPath("$.timestamp").isNotEmpty())
                .andExpect(jsonPath("$.retryable").value(true))
                .andExpect(jsonPath("$.remediation").isNotEmpty())
                .andExpect(jsonPath("$.taskId").doesNotExist())
                .andExpect(jsonPath("$.outputs").doesNotExist())
                .andExpect(jsonPath("$.metrics").doesNotExist());
    }
}
