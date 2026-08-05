package com.seatflow.controller;

import com.seatflow.dto.Page;
import com.seatflow.dto.RunCreateRequest;
import com.seatflow.dto.TaskHandle;
import com.seatflow.model.Run;
import com.seatflow.model.RunStatus;
import com.seatflow.model.TaskStatus;
import com.seatflow.security.UserPrincipal;
import com.seatflow.service.ExperimentService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import java.time.Instant;
import java.util.UUID;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * Run orchestration and lifecycle endpoints.
 */
@RestController
@RequestMapping("/api/v1/runs")
@Tag(name = "runs", description = "Run orchestration and lifecycle")
public class RunController {

    private final ExperimentService experimentService;

    /**
     * Construct the controller.
     *
     * @param experimentService experiment service
     */
    public RunController(ExperimentService experimentService) {
        this.experimentService = experimentService;
    }

    /**
     * List runs, optionally filtered by experimentId and status.
     *
     * @param experimentId optional experiment filter
     * @param status        optional status filter
     * @return paginated run list
     */
    @GetMapping
    @Operation(operationId = "runsList", summary = "List runs")
    public ResponseEntity<Page<Run>> list(
            @RequestParam(required = false) String experimentId,
            @RequestParam(required = false) RunStatus status) {
        return ResponseEntity.ok(Page.of(experimentService.listRuns(experimentId, status)));
    }

    /**
     * Submit a new run (long-running).
     *
     * @param request   run creation request
     * @param principal authenticated principal
     * @return task handle (202)
     */
    @PostMapping
    @Operation(operationId = "runsCreate", summary = "Submit a new run (long-running)")
    public ResponseEntity<TaskHandle> create(
            @RequestBody RunCreateRequest request,
            @AuthenticationPrincipal UserPrincipal principal) {
        Run run = experimentService.createRun(request, principal);
        TaskHandle handle = new TaskHandle(run.id(), TaskStatus.QUEUED, Instant.now(),
                "/api/v1/system/tasks/" + run.id(),
                "/api/v1/system/tasks/" + run.id() + "/events");
        return ResponseEntity.accepted().body(handle);
    }

    /**
     * Get run detail.
     *
     * @param runId run identifier
     * @return run, or 404
     */
    @GetMapping("/{runId}")
    @Operation(operationId = "runsGet", summary = "Get run detail")
    public ResponseEntity<Run> get(@PathVariable String runId) {
        return experimentService.getRun(runId)
                .map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.status(HttpStatus.NOT_FOUND).build());
    }

    /**
     * Cancel a running run.
     *
     * @param runId     run identifier
     * @param principal authenticated principal
     * @return task handle (202), or 404
     */
    @PostMapping("/{runId}/cancel")
    @Operation(operationId = "runsCancel", summary = "Cancel a running run")
    public ResponseEntity<TaskHandle> cancel(
            @PathVariable String runId,
            @AuthenticationPrincipal UserPrincipal principal) {
        return experimentService.cancelRun(runId, principal)
                .map(run -> ResponseEntity.accepted().body(new TaskHandle(
                        run.id(), TaskStatus.CANCELLED, Instant.now(),
                        "/api/v1/system/tasks/" + run.id(),
                        "/api/v1/system/tasks/" + run.id() + "/events")))
                .orElseGet(() -> ResponseEntity.status(HttpStatus.NOT_FOUND).build());
    }

    /**
     * List artifacts produced by a run.
     *
     * @param runId run identifier
     * @return paginated artifact list
     */
    @GetMapping("/{runId}/artifacts")
    @Operation(operationId = "runsListArtifacts", summary = "List artifacts produced by a run")
    public ResponseEntity<Page<java.util.Map<String, Object>>> listArtifacts(@PathVariable String runId) {
        return ResponseEntity.ok(Page.of(java.util.List.of()));
    }
}
