package com.seatflow.controller;

import com.seatflow.model.ErrorResponse;
import com.seatflow.service.ResearchArtifactBoundary;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import java.util.Map;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Dataset endpoints delegated to the canonical research artifact boundary.
 */
@RestController
@RequestMapping("/api/v1/datasets")
@Tag(name = "datasets", description = "Dataset versioning, build, and validation")
public class DatasetController {

    private final ResearchArtifactBoundary researchBoundary;

    /**
     * @param researchBoundary fail-closed research artifact boundary
     */
    public DatasetController(ResearchArtifactBoundary researchBoundary) {
        this.researchBoundary = researchBoundary;
    }

    /**
     * List datasets.
     *
     * @return blocked response until a verified artifact gateway is available
     */
    @GetMapping
    @Operation(operationId = "datasetsList", summary = "List datasets")
    public ResponseEntity<ErrorResponse> list() {
        return researchBoundary.unavailable("datasets.list");
    }

    /**
     * Get a dataset definition.
     *
     * @param datasetId dataset identifier
     * @return blocked response until a verified dataset artifact is available
     */
    @GetMapping("/{datasetId}")
    @Operation(operationId = "datasetsGet", summary = "Get a dataset definition")
    public ResponseEntity<ErrorResponse> get(@PathVariable String datasetId) {
        return researchBoundary.unavailable("datasets.get");
    }

    /**
     * Build a dataset version (long-running).
     *
     * @param datasetId dataset identifier
     * @param request   build request body
     * @return blocked response instead of an untracked task handle
     */
    @PostMapping("/{datasetId}/build")
    @Operation(operationId = "datasetsBuild", summary = "Build a dataset version (long-running)")
    public ResponseEntity<ErrorResponse> build(
            @PathVariable String datasetId,
            @RequestBody Map<String, Object> request) {
        return researchBoundary.unavailable("datasets.build");
    }

    /**
     * Validate a dataset against quality gates.
     *
     * @param datasetId dataset identifier
     * @return blocked response instead of an untracked task handle
     */
    @PostMapping("/{datasetId}/validate")
    @Operation(operationId = "datasetsValidate", summary = "Validate a dataset against quality gates")
    public ResponseEntity<ErrorResponse> validate(@PathVariable String datasetId) {
        return researchBoundary.unavailable("datasets.validate");
    }
}
