package com.seatflow.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.seatflow.dto.ModelPromoteRequest;
import com.seatflow.dto.ModelRegisterRequest;
import com.seatflow.model.ModelStage;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class ModelControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    void list_returns200WithPage() throws Exception {
        mockMvc.perform(get("/api/v1/models"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.items").isArray());
    }

    @Test
    void register_returns201WithModel() throws Exception {
        ModelRegisterRequest request = new ModelRegisterRequest("lstm-v1", "1.0.0", "s3://artifacts/lstm-v1-1.0.0");
        MvcResult result = mockMvc.perform(post("/api/v1/models")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.id").isNotEmpty())
                .andExpect(jsonPath("$.name").value("lstm-v1"))
                .andExpect(jsonPath("$.version").value("1.0.0"))
                .andExpect(jsonPath("$.stage").value("NONE"))
                .andExpect(jsonPath("$.artifactUri").value("s3://artifacts/lstm-v1-1.0.0"))
                .andReturn();
    }

    @Test
    void get_existingModel_returns200() throws Exception {
        ModelRegisterRequest request = new ModelRegisterRequest("xgb", "0.1", "s3://artifacts/xgb");
        MvcResult createResult = mockMvc.perform(post("/api/v1/models")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andReturn();
        String id = objectMapper.readTree(createResult.getResponse().getContentAsString()).get("id").asText();

        mockMvc.perform(get("/api/v1/models/" + id))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value(id));
    }

    @Test
    void get_unknownModel_returns404() throws Exception {
        mockMvc.perform(get("/api/v1/models/does-not-exist"))
                .andExpect(status().isNotFound());
    }

    @Test
    void promote_withoutVerifiedResearchArtifact_returnsUnavailableEnvelope() throws Exception {
        ModelRegisterRequest register = new ModelRegisterRequest("promote-test", "1.0", "s3://artifacts/promote-test");
        MvcResult createResult = mockMvc.perform(post("/api/v1/models")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(register)))
                .andReturn();
        String id = objectMapper.readTree(createResult.getResponse().getContentAsString()).get("id").asText();

        ModelPromoteRequest promote = new ModelPromoteRequest(ModelStage.STAGING);
        mockMvc.perform(post("/api/v1/models/" + id + "/promote")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(promote)))
                .andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.code").value("RESEARCH_ARTIFACT_UNAVAILABLE"))
                .andExpect(jsonPath("$.details.operation").value("models.promote"))
                .andExpect(jsonPath("$.remediation").isNotEmpty());
    }

    @Test
    void promote_invalidTransition_isNotEvaluatedWithoutVerifiedResearchArtifact() throws Exception {
        ModelRegisterRequest register = new ModelRegisterRequest("bad-transition", "1.0", "s3://artifacts/bad");
        MvcResult createResult = mockMvc.perform(post("/api/v1/models")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(register)))
                .andReturn();
        String id = objectMapper.readTree(createResult.getResponse().getContentAsString()).get("id").asText();

        // NONE -> PRODUCTION is invalid (must go through STAGING first)
        ModelPromoteRequest promote = new ModelPromoteRequest(ModelStage.PRODUCTION);
        mockMvc.perform(post("/api/v1/models/" + id + "/promote")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(promote)))
                .andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.code").value("RESEARCH_ARTIFACT_UNAVAILABLE"));
    }

    @Test
    void promote_unknownModel_stillFailsClosedAtResearchBoundary() throws Exception {
        ModelPromoteRequest promote = new ModelPromoteRequest(ModelStage.STAGING);
        mockMvc.perform(post("/api/v1/models/does-not-exist/promote")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(promote)))
                .andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.code").value("RESEARCH_ARTIFACT_UNAVAILABLE"));
    }

    @Test
    void retire_withoutTaskGateway_returnsUnavailableInsteadOfUntrackedHandle() throws Exception {
        ModelRegisterRequest register = new ModelRegisterRequest("retire-test", "1.0", "s3://artifacts/retire");
        MvcResult createResult = mockMvc.perform(post("/api/v1/models")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(register)))
                .andReturn();
        String id = objectMapper.readTree(createResult.getResponse().getContentAsString()).get("id").asText();

        mockMvc.perform(post("/api/v1/models/" + id + "/retire"))
                .andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.code").value("RESEARCH_ARTIFACT_UNAVAILABLE"))
                .andExpect(jsonPath("$.details.operation").value("models.retire"))
                .andExpect(jsonPath("$.taskId").doesNotExist());
    }
}
