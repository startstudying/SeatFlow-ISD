package com.seatflow.controller;

import com.seatflow.dto.HealthReport;
import com.seatflow.dto.TaskState;
import com.seatflow.dto.VersionInfo;
import com.seatflow.model.TaskStatus;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import java.time.Instant;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Health, version, and operational endpoints.
 */
@RestController
@RequestMapping("/api/v1/system")
@Tag(name = "system", description = "Health, version, and operational endpoints")
public class SystemController {

    private final ConcurrentHashMap<String, TaskState> tasks = new ConcurrentHashMap<>();

    /**
     * Liveness and readiness probe.
     *
     * @return health report
     */
    @GetMapping("/health")
    @Operation(operationId = "systemHealth", summary = "Liveness and readiness probe")
    public ResponseEntity<HealthReport> health() {
        return ResponseEntity.ok(new HealthReport("UP", Map.of("disk", Map.of("status", "UP"), "db", Map.of("status", "UP"))));
    }

    /**
     * Service version information.
     *
     * @return version info
     */
    @GetMapping("/version")
    @Operation(operationId = "systemVersion", summary = "Service version information")
    public ResponseEntity<VersionInfo> version() {
        return ResponseEntity.ok(new VersionInfo("3.0.0-dev", "20260806", "dev"));
    }

    /**
     * Get task state by id.
     *
     * @param taskId task identifier
     * @return task state, or 404
     */
    @GetMapping("/tasks/{taskId}")
    @Operation(operationId = "systemGetTask", summary = "Get task state by id")
    public ResponseEntity<TaskState> getTask(@PathVariable String taskId) {
        TaskState state = tasks.get(taskId);
        if (state == null) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).build();
        }
        return ResponseEntity.ok(state);
    }

    /**
     * Cancel a task by id.
     *
     * @param taskId task identifier
     * @return cancelled task state (202), or 404
     */
    @PostMapping("/tasks/{taskId}/cancel")
    @Operation(operationId = "systemCancelTask", summary = "Cancel a task by id")
    public ResponseEntity<TaskState> cancelTask(@PathVariable String taskId) {
        TaskState existing = tasks.get(taskId);
        if (existing == null) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).build();
        }
        TaskState cancelled = new TaskState(taskId, TaskStatus.CANCELLED, existing.progress(), Instant.now(), null, null);
        tasks.put(taskId, cancelled);
        return ResponseEntity.accepted().body(cancelled);
    }
}
