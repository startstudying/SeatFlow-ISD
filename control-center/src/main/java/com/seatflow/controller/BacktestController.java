package com.seatflow.controller;

import com.seatflow.dto.Page;
import com.seatflow.dto.TaskHandle;
import com.seatflow.model.TaskStatus;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Backtest orchestration and replay endpoints.
 */
@RestController
@RequestMapping("/api/v1/backtests")
@Tag(name = "backtests", description = "Backtest orchestration and replay")
public class BacktestController {

    /**
     * List backtests.
     *
     * @return paginated backtest list
     */
    @GetMapping
    @Operation(operationId = "backtestsList", summary = "List backtests")
    public ResponseEntity<Page<Map<String, Object>>> list() {
        return ResponseEntity.ok(Page.of(List.of()));
    }

    /**
     * Create a backtest (long-running).
     *
     * @param request backtest creation request
     * @return task handle (202)
     */
    @PostMapping
    @Operation(operationId = "backtestsCreate", summary = "Create a backtest (long-running)")
    public ResponseEntity<TaskHandle> create(@RequestBody Map<String, Object> request) {
        String taskId = UUID.randomUUID().toString();
        TaskHandle handle = new TaskHandle(taskId, TaskStatus.QUEUED, Instant.now(),
                "/api/v1/system/tasks/" + taskId,
                "/api/v1/system/tasks/" + taskId + "/events");
        return ResponseEntity.accepted().body(handle);
    }

    /**
     * Get backtest detail.
     *
     * @param backtestId backtest identifier
     * @return backtest, or 404
     */
    @GetMapping("/{backtestId}")
    @Operation(operationId = "backtestsGet", summary = "Get backtest detail")
    public ResponseEntity<Map<String, Object>> get(@PathVariable String backtestId) {
        return ResponseEntity.ok(Map.of("id", backtestId, "modelId", "model-1", "status", "COMPLETED"));
    }

    /**
     * Get backtest report.
     *
     * @param backtestId backtest identifier
     * @return report, or 404
     */
    @GetMapping("/{backtestId}/report")
    @Operation(operationId = "backtestsGetReport", summary = "Get backtest report")
    public ResponseEntity<Map<String, Object>> getReport(@PathVariable String backtestId) {
        Map<String, Object> report = Map.of(
                "metrics", Map.of("sharpe", 1.5, "maxDrawdown", -0.12),
                "equityCurve", List.of());
        return ResponseEntity.ok(report);
    }
}
