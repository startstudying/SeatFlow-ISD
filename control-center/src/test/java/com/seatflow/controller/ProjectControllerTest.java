package com.seatflow.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.seatflow.dto.ProjectCreateRequest;
import com.seatflow.dto.ProjectUpdateRequest;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.patch;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class ProjectControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    void list_returns200WithPage() throws Exception {
        mockMvc.perform(get("/api/v1/projects"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.items").isArray());
    }

    @Test
    void create_returns201WithProject() throws Exception {
        ProjectCreateRequest request = new ProjectCreateRequest("Test Project", "A test description");
        MvcResult result = mockMvc.perform(post("/api/v1/projects")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.id").isNotEmpty())
                .andExpect(jsonPath("$.name").value("Test Project"))
                .andExpect(jsonPath("$.description").value("A test description"))
                .andExpect(jsonPath("$.createdAt").isNotEmpty())
                .andReturn();
    }

    @Test
    void get_existingProject_returns200() throws Exception {
        ProjectCreateRequest request = new ProjectCreateRequest("Get Me", "desc");
        MvcResult createResult = mockMvc.perform(post("/api/v1/projects")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andReturn();
        String id = objectMapper.readTree(createResult.getResponse().getContentAsString()).get("id").asText();

        mockMvc.perform(get("/api/v1/projects/" + id))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value(id))
                .andExpect(jsonPath("$.name").value("Get Me"));
    }

    @Test
    void get_unknownProject_returns404() throws Exception {
        mockMvc.perform(get("/api/v1/projects/does-not-exist"))
                .andExpect(status().isNotFound());
    }

    @Test
    void update_existingProject_returns200() throws Exception {
        ProjectCreateRequest create = new ProjectCreateRequest("Original", "original desc");
        MvcResult createResult = mockMvc.perform(post("/api/v1/projects")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(create)))
                .andReturn();
        String id = objectMapper.readTree(createResult.getResponse().getContentAsString()).get("id").asText();

        ProjectUpdateRequest update = new ProjectUpdateRequest("Updated Name", null);
        mockMvc.perform(patch("/api/v1/projects/" + id)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(update)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value(id))
                .andExpect(jsonPath("$.name").value("Updated Name"))
                .andExpect(jsonPath("$.description").value("original desc"));
    }

    @Test
    void update_unknownProject_returns404() throws Exception {
        ProjectUpdateRequest update = new ProjectUpdateRequest("Updated Name", null);
        mockMvc.perform(patch("/api/v1/projects/does-not-exist")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(update)))
                .andExpect(status().isNotFound());
    }

    @Test
    void delete_existingProject_returns202() throws Exception {
        ProjectCreateRequest create = new ProjectCreateRequest("Delete Me", null);
        MvcResult createResult = mockMvc.perform(post("/api/v1/projects")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(create)))
                .andReturn();
        String id = objectMapper.readTree(createResult.getResponse().getContentAsString()).get("id").asText();

        mockMvc.perform(delete("/api/v1/projects/" + id))
                .andExpect(status().isAccepted());
    }

    @Test
    void delete_unknownProject_returns404() throws Exception {
        mockMvc.perform(delete("/api/v1/projects/does-not-exist"))
                .andExpect(status().isNotFound());
    }
}
