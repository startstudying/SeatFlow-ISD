package com.seatflow.controller;

import com.seatflow.dto.TaskHandle;
import com.seatflow.model.TaskStatus;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import java.time.Instant;
import java.util.Map;
import java.util.UUID;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Online and batch prediction endpoints.
 */
@RestController
@RequestMapping("/api/v1/predictions")
@Tag(name = "predictions", description = "Online and batch prediction endpoints")
public class PredictionController {

    /**
     * Generate predictions for a single inference request.
     *
     * @param request prediction request (modelId, inputs)
     * @return prediction response
     */
    @PostMapping
    @Operation(operationId = "predictionsCreate", summary = "Generate predictions for a single inference request")
    public ResponseEntity<Map<String, Object>> predict(@RequestBody Map<String, Object> request) {
        String modelId = (String) request.getOrDefault("modelId", "unknown");
        Map<String, Object> response = Map.of(
                "modelVersion", "1.0.0",
                "outputs", Map.of("prediction", 0.0));
        return ResponseEntity.ok(response);
    }

    /**
     * Submit a batch prediction job (long-running).
     *
     * @param request batch request (modelId, datasetId)
     * @return task handle (202)
     */
    @PostMapping("/batch")
    @Operation(operationId = "predictionsCreateBatch", summary = "Submit a batch prediction job (long-running)")
    public ResponseEntity<TaskHandle> predictBatch(@RequestBody Map<String, Object> request) {
        String taskId = UUID.randomUUID().toString();
        TaskHandle handle = new TaskHandle(taskId, TaskStatus.QUEUED, Instant.now(),
                "/api/v1/system/tasks/" + taskId,
                "/api/v1/system/tasks/" + taskId + "/events");
        return ResponseEntity.accepted().body(handle);
    }
}
