package com.seatflow.controller;

import com.seatflow.dto.ModelPromoteRequest;
import com.seatflow.dto.ModelRegisterRequest;
import com.seatflow.dto.Page;
import com.seatflow.model.ErrorResponse;
import com.seatflow.model.Model;
import com.seatflow.security.UserPrincipal;
import com.seatflow.service.ModelService;
import com.seatflow.service.ResearchArtifactBoundary;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import java.net.URI;
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
 * Model registry, promotion, and retirement endpoints.
 */
@RestController
@RequestMapping("/api/v1/models")
@Tag(name = "models", description = "Model registry, promotion, and retirement")
public class ModelController {

    private final ModelService modelService;
    private final ResearchArtifactBoundary researchBoundary;

    /**
     * Construct the controller.
     *
     * @param modelService model service
     * @param researchBoundary fail-closed research artifact boundary
     */
    public ModelController(ModelService modelService, ResearchArtifactBoundary researchBoundary) {
        this.modelService = modelService;
        this.researchBoundary = researchBoundary;
    }

    /**
     * List registered models.
     *
     * @return paginated model list
     */
    @GetMapping
    @Operation(operationId = "modelsList", summary = "List registered models")
    public ResponseEntity<Page<Model>> list() {
        return ResponseEntity.ok(Page.of(modelService.list()));
    }

    /**
     * Register a model artifact.
     *
     * @param request   registration request
     * @param principal authenticated principal
     * @return registered model (201)
     */
    @PostMapping
    @Operation(operationId = "modelsRegister", summary = "Register a model artifact")
    public ResponseEntity<Model> register(
            @RequestBody ModelRegisterRequest request,
            @AuthenticationPrincipal UserPrincipal principal) {
        Model registered = modelService.register(request, principal);
        return ResponseEntity
                .created(URI.create("/api/v1/models/" + registered.id()))
                .body(registered);
    }

    /**
     * Get model detail.
     *
     * @param modelId model identifier
     * @return model, or 404
     */
    @GetMapping("/{modelId}")
    @Operation(operationId = "modelsGet", summary = "Get model detail")
    public ResponseEntity<Model> get(@PathVariable String modelId) {
        return modelService.get(modelId)
                .map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.status(HttpStatus.NOT_FOUND).build());
    }

    /**
     * Promote a model version to a stage.
     *
     * @param modelId   model identifier
     * @param request   promotion request
     * @param principal authenticated principal
     * @return blocked response until the model artifact can be verified
     */
    @PostMapping("/{modelId}/promote")
    @Operation(operationId = "modelsPromote", summary = "Promote a model version to a stage")
    public ResponseEntity<ErrorResponse> promote(
            @PathVariable String modelId,
            @RequestBody ModelPromoteRequest request,
            @AuthenticationPrincipal UserPrincipal principal) {
        return researchBoundary.unavailable("models.promote");
    }

    /**
     * Retire a model version.
     *
     * @param modelId model identifier
     * @return blocked response instead of an untracked task handle
     */
    @PostMapping("/{modelId}/retire")
    @Operation(operationId = "modelsRetire", summary = "Retire a model version")
    public ResponseEntity<ErrorResponse> retire(@PathVariable String modelId) {
        return researchBoundary.unavailable("models.retire");
    }
}
