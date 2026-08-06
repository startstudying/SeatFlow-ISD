package com.seatflow.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.seatflow.dto.ExperimentCreateRequest;
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
class ExperimentControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    void list_returns200WithPage() throws Exception {
        mockMvc.perform(get("/api/v1/experiments"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.items").isArray());
    }

    @Test
    void create_returns201WithExperiment() throws Exception {
        ExperimentCreateRequest request = new ExperimentCreateRequest("Test Experiment", "proj-1", "desc");
        mockMvc.perform(post("/api/v1/experiments")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.id").isNotEmpty())
                .andExpect(jsonPath("$.name").value("Test Experiment"))
                .andExpect(jsonPath("$.projectId").value("proj-1"))
                .andExpect(jsonPath("$.createdAt").isNotEmpty());
    }

    @Test
    void get_existingExperiment_returns200() throws Exception {
        ExperimentCreateRequest request = new ExperimentCreateRequest("Find Me", "proj-1", null);
        MvcResult createResult = mockMvc.perform(post("/api/v1/experiments")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andReturn();
        String id = objectMapper.readTree(createResult.getResponse().getContentAsString()).get("id").asText();

        mockMvc.perform(get("/api/v1/experiments/" + id))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value(id))
                .andExpect(jsonPath("$.name").value("Find Me"));
    }

    @Test
    void get_unknownExperiment_returns404() throws Exception {
        mockMvc.perform(get("/api/v1/experiments/does-not-exist"))
                .andExpect(status().isNotFound());
    }

    @Test
    void compare_withoutVerifiedResearchArtifact_returnsUnavailableEnvelope() throws Exception {
        ExperimentCreateRequest request = new ExperimentCreateRequest("Compare Exp", "proj-1", null);
        MvcResult createResult = mockMvc.perform(post("/api/v1/experiments")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andReturn();
        String id = objectMapper.readTree(createResult.getResponse().getContentAsString()).get("id").asText();

        String body = "{\"runIds\":[\"run-1\",\"run-2\"]}";
        mockMvc.perform(post("/api/v1/experiments/" + id + "/compare")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body))
                .andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.code").value("RESEARCH_ARTIFACT_UNAVAILABLE"))
                .andExpect(jsonPath("$.details.status").value("BLOCKED"))
                .andExpect(jsonPath("$.details.operation").value("experiments.compare"))
                .andExpect(jsonPath("$.retryable").value(true))
                .andExpect(jsonPath("$.remediation").isNotEmpty())
                .andExpect(jsonPath("$.metrics").doesNotExist());
    }
}
