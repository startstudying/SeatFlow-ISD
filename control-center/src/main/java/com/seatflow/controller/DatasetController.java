package com.seatflow.controller;

import com.seatflow.dto.Page;
import com.seatflow.model.TaskStatus;
import com.seatflow.dto.TaskHandle;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Dataset listing endpoints. Dataset build and validation are represented as
 * long-running tasks that return a {@link TaskHandle}.
 */
@RestController
@RequestMapping("/api/v1/datasets")
@Tag(name = "datasets", description = "Dataset versioning, build, and validation")
public class DatasetController {

    /**
     * List datasets.
     *
     * @return paginated dataset list
     */
    @GetMapping
    @Operation(operationId = "datasetsList", summary = "List datasets")
    public ResponseEntity<Page<Map<String, Object>>> list() {
        // In-memory placeholder dataset list.
        List<Map<String, Object>> items = List.of(
                Map.of("id", "ds-1", "name", "default", "version", "v1"),
                Map.of("id", "ds-2", "name", "enriched", "version", "v2"));
        return ResponseEntity.ok(Page.of(items));
    }

    /**
     * Get a dataset definition.
     *
     * @param datasetId dataset identifier
     * @return dataset, or 404
     */
    @GetMapping("/{datasetId}")
    @Operation(operationId = "datasetsGet", summary = "Get a dataset definition")
    public ResponseEntity<Map<String, Object>> get(@PathVariable String datasetId) {
        return ResponseEntity.ok(Map.of("id", datasetId, "name", "dataset-" + datasetId, "version", "v1"));
    }

    /**
     * Build a dataset version (long-running).
     *
     * @param datasetId dataset identifier
     * @param request   build request body
     * @return task handle (202)
     */
    @PostMapping("/{datasetId}/build")
    @Operation(operationId = "datasetsBuild", summary = "Build a dataset version (long-running)")
    public ResponseEntity<TaskHandle> build(
            @PathVariable String datasetId,
            @RequestBody Map<String, Object> request) {
        String taskId = UUID.randomUUID().toString();
        TaskHandle handle = new TaskHandle(taskId, TaskStatus.QUEUED, Instant.now(),
                "/api/v1/system/tasks/" + taskId,
                "/api/v1/system/tasks/" + taskId + "/events");
        return ResponseEntity.accepted().body(handle);
    }

    /**
     * Validate a dataset against quality gates.
     *
     * @param datasetId dataset identifier
     * @return task handle (202)
     */
    @PostMapping("/{datasetId}/validate")
    @Operation(operationId = "datasetsValidate", summary = "Validate a dataset against quality gates")
    public ResponseEntity<TaskHandle> validate(@PathVariable String datasetId) {
        String taskId = UUID.randomUUID().toString();
        TaskHandle handle = new TaskHandle(taskId, TaskStatus.QUEUED, Instant.now(),
                "/api/v1/system/tasks/" + taskId,
                "/api/v1/system/tasks/" + taskId + "/events");
        return ResponseEntity.accepted().body(handle);
    }
}
