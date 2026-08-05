package com.seatflow.service;

import com.seatflow.dto.ProjectCreateRequest;
import com.seatflow.dto.ProjectUpdateRequest;
import com.seatflow.model.Project;
import com.seatflow.security.UserPrincipal;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import org.springframework.stereotype.Service;

/**
 * In-memory project store. All CRUD operations are thread-safe.
 *
 * <p>Production deployments should replace this with a JPA-backed
 * implementation backed by PostgreSQL.</p>
 */
@Service
public class ProjectService {

    private final ConcurrentHashMap<String, Project> store = new ConcurrentHashMap<>();
    private final AuditService auditService;

    /**
     * Construct the service.
     *
     * @param auditService audit service for recording project operations
     */
    public ProjectService(AuditService auditService) {
        this.auditService = auditService;
    }

    /**
     * List all projects.
     *
     * @return immutable list of all projects
     */
    public List<Project> list() {
        return List.copyOf(store.values());
    }

    /**
     * Get a project by id.
     *
     * @param id project identifier
     * @return the project, or empty if not found
     */
    public Optional<Project> get(String id) {
        return Optional.ofNullable(store.get(id));
    }

    /**
     * Create a new project.
     *
     * @param request  creation request
     * @param actor    acting principal
     * @return the created project
     */
    public Project create(ProjectCreateRequest request, UserPrincipal actor) {
        Instant now = Instant.now();
        Project project = new Project(
                UUID.randomUUID().toString(),
                request.name(),
                request.description(),
                now,
                now);
        store.put(project.id(), project);
        auditService.record(actor, "project.create", Map.of("projectId", project.id(), "name", project.name()));
        return project;
    }

    /**
     * Update an existing project.
     *
     * @param id      project identifier
     * @param request update request
     * @param actor   acting principal
     * @return the updated project, or empty if not found
     */
    public Optional<Project> update(String id, ProjectUpdateRequest request, UserPrincipal actor) {
        return get(id).map(existing -> {
            String newName = request.name() != null ? request.name() : existing.name();
            String newDesc = request.description() != null ? request.description() : existing.description();
            Project updated = new Project(existing.id(), newName, newDesc, existing.createdAt(), Instant.now());
            store.put(id, updated);
            auditService.record(actor, "project.update", Map.of("projectId", id));
            return updated;
        });
    }

    /**
     * Delete (soft) a project.
     *
     * @param id    project identifier
     * @param actor acting principal
     * @return true if the project existed and was removed
     */
    public boolean delete(String id, UserPrincipal actor) {
        Project removed = store.remove(id);
        if (removed != null) {
            auditService.record(actor, "project.delete", Map.of("projectId", id));
            return true;
        }
        return false;
    }
}
