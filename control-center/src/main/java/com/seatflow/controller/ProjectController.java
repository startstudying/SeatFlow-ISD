package com.seatflow.controller;

import com.seatflow.dto.Page;
import com.seatflow.dto.ProjectCreateRequest;
import com.seatflow.dto.ProjectUpdateRequest;
import com.seatflow.model.Project;
import com.seatflow.security.UserPrincipal;
import com.seatflow.service.ProjectService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import java.net.URI;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Project lifecycle and configuration endpoints.
 */
@RestController
@RequestMapping("/api/v1/projects")
@Tag(name = "projects", description = "Project lifecycle and configuration")
public class ProjectController {

    private final ProjectService projectService;

    /**
     * Construct the controller.
     *
     * @param projectService project service
     */
    public ProjectController(ProjectService projectService) {
        this.projectService = projectService;
    }

    /**
     * List all projects.
     *
     * @return paginated project list
     */
    @GetMapping
    @Operation(operationId = "projectsList", summary = "List all projects")
    public ResponseEntity<Page<Project>> list() {
        return ResponseEntity.ok(Page.of(projectService.list()));
    }

    /**
     * Create a new project.
     *
     * @param request   creation request
     * @param principal authenticated principal
     * @return created project (201)
     */
    @PostMapping
    @Operation(operationId = "projectsCreate", summary = "Create a new project")
    public ResponseEntity<Project> create(
            @RequestBody ProjectCreateRequest request,
            @AuthenticationPrincipal UserPrincipal principal) {
        Project created = projectService.create(request, principal);
        return ResponseEntity
                .created(URI.create("/api/v1/projects/" + created.id()))
                .body(created);
    }

    /**
     * Get project by id.
     *
     * @param projectId project identifier
     * @return project, or 404
     */
    @GetMapping("/{projectId}")
    @Operation(operationId = "projectsGet", summary = "Get project by id")
    public ResponseEntity<Project> get(@PathVariable String projectId) {
        return projectService.get(projectId)
                .map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.status(HttpStatus.NOT_FOUND).build());
    }

    /**
     * Update project metadata.
     *
     * @param projectId project identifier
     * @param request   update request
     * @param principal authenticated principal
     * @return updated project, or 404
     */
    @PatchMapping("/{projectId}")
    @Operation(operationId = "projectsUpdate", summary = "Update project metadata")
    public ResponseEntity<Project> update(
            @PathVariable String projectId,
            @RequestBody ProjectUpdateRequest request,
            @AuthenticationPrincipal UserPrincipal principal) {
        return projectService.update(projectId, request, principal)
                .map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.status(HttpStatus.NOT_FOUND).build());
    }

    /**
     * Delete a project (soft).
     *
     * @param projectId project identifier
     * @param principal authenticated principal
     * @return 202 if deletion was accepted, or 404
     */
    @DeleteMapping("/{projectId}")
    @Operation(operationId = "projectsDelete", summary = "Delete a project (soft)")
    public ResponseEntity<Void> delete(
            @PathVariable String projectId,
            @AuthenticationPrincipal UserPrincipal principal) {
        boolean deleted = projectService.delete(projectId, principal);
        return deleted
                ? ResponseEntity.accepted().build()
                : ResponseEntity.status(HttpStatus.NOT_FOUND).build();
    }
}
