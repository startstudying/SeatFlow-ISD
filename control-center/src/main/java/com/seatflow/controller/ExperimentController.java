package com.seatflow.controller;

import com.seatflow.dto.ExperimentCreateRequest;
import com.seatflow.dto.Page;
import com.seatflow.model.Experiment;
import com.seatflow.security.UserPrincipal;
import com.seatflow.service.ExperimentService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import java.net.URI;
import java.util.List;
import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Experiment design, comparison, and reporting endpoints.
 */
@RestController
@RequestMapping("/api/v1/experiments")
@Tag(name = "experiments", description = "Experiment design, comparison, and reporting")
public class ExperimentController {

    private final ExperimentService experimentService;

    /**
     * Construct the controller.
     *
     * @param experimentService experiment service
     */
    public ExperimentController(ExperimentService experimentService) {
        this.experimentService = experimentService;
    }

    /**
     * List experiments.
     *
     * @return paginated experiment list
     */
    @GetMapping
    @Operation(operationId = "experimentsList", summary = "List experiments")
    public ResponseEntity<Page<Experiment>> list() {
        return ResponseEntity.ok(Page.of(experimentService.listExperiments()));
    }

    /**
     * Create a new experiment.
     *
     * @param request   creation request
     * @param principal authenticated principal
     * @return created experiment (201)
     */
    @PostMapping
    @Operation(operationId = "experimentsCreate", summary = "Create a new experiment")
    public ResponseEntity<Experiment> create(
            @RequestBody ExperimentCreateRequest request,
            @AuthenticationPrincipal UserPrincipal principal) {
        Experiment created = experimentService.createExperiment(request, principal);
        return ResponseEntity
                .created(URI.create("/api/v1/experiments/" + created.id()))
                .body(created);
    }

    /**
     * Get experiment detail.
     *
     * @param experimentId experiment identifier
     * @return experiment, or 404
     */
    @GetMapping("/{experimentId}")
    @Operation(operationId = "experimentsGet", summary = "Get experiment detail")
    public ResponseEntity<Experiment> get(@PathVariable String experimentId) {
        return experimentService.getExperiment(experimentId)
                .map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.status(HttpStatus.NOT_FOUND).build());
    }

    /**
     * Compare runs within an experiment.
     *
     * @param experimentId experiment identifier
     * @param request      compare request body (runIds)
     * @return comparison report
     */
    @PostMapping("/{experimentId}/compare")
    @Operation(operationId = "experimentsCompare", summary = "Compare runs within an experiment")
    public ResponseEntity<Map<String, Object>> compare(
            @PathVariable String experimentId,
            @RequestBody Map<String, Object> request) {
        List<String> runIds = request.get("runIds") instanceof List<?> l
                ? l.stream().map(Object::toString).toList()
                : List.of();
        Map<String, Object> report = Map.of(
                "experimentId", experimentId,
                "runIds", runIds,
                "metrics", Map.of());
        return ResponseEntity.ok(report);
    }
}
