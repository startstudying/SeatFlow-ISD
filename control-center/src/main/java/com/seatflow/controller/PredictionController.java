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
 * Online and batch prediction endpoints.
 */
@RestController
@RequestMapping("/api/v1/predictions")
@Tag(name = "predictions", description = "Online and batch prediction endpoints")
public class PredictionController {

    private final ResearchArtifactBoundary researchBoundary;

    /**
     * @param researchBoundary fail-closed research artifact boundary
     */
    public PredictionController(ResearchArtifactBoundary researchBoundary) {
        this.researchBoundary = researchBoundary;
    }

    /**
     * List verified prediction artifacts.
     *
     * @return blocked response until verified prediction artifacts are available
     */
    @GetMapping
    @Operation(operationId = "predictionsList", summary = "List prediction artifacts")
    public ResponseEntity<ErrorResponse> list() {
        return researchBoundary.unavailable("predictions.list");
    }

    /**
     * Get a verified prediction artifact.
     *
     * @param predictionId prediction artifact identifier
     * @return blocked response until the requested artifact is available
     */
    @GetMapping("/{predictionId}")
    @Operation(operationId = "predictionsGet", summary = "Get a prediction artifact")
    public ResponseEntity<ErrorResponse> get(@PathVariable String predictionId) {
        return researchBoundary.unavailable("predictions.get");
    }

    /**
     * Generate predictions for a single inference request.
     *
     * @param request prediction request (modelId, inputs)
     * @return blocked response until a verified model artifact is available
     */
    @PostMapping
    @Operation(operationId = "predictionsCreate", summary = "Generate predictions for a single inference request")
    public ResponseEntity<ErrorResponse> predict(@RequestBody Map<String, Object> request) {
        return researchBoundary.unavailable("predictions.create");
    }

    /**
     * Submit a batch prediction job (long-running).
     *
     * @param request batch request (modelId, datasetId)
     * @return blocked response instead of an untracked task handle
     */
    @PostMapping("/batch")
    @Operation(operationId = "predictionsCreateBatch", summary = "Submit a batch prediction job (long-running)")
    public ResponseEntity<ErrorResponse> predictBatch(@RequestBody Map<String, Object> request) {
        return researchBoundary.unavailable("predictions.createBatch");
    }
}
