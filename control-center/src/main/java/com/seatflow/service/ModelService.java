package com.seatflow.service;

import com.seatflow.dto.ModelPromoteRequest;
import com.seatflow.dto.ModelRegisterRequest;
import com.seatflow.model.Model;
import com.seatflow.model.ModelStage;
import com.seatflow.security.UserPrincipal;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import org.springframework.stereotype.Service;

/**
 * In-memory model registry with promotion logic.
 *
 * <p>Promotion enforces a linear stage progression:
 * {@code NONE -> STAGING -> PRODUCTION -> ARCHIVED}. Demotion (e.g. from
 * PRODUCTION back to STAGING) is rejected with an IllegalStateException. At most
 * one model per (name) may occupy the PRODUCTION stage; promoting a second model
 * to PRODUCTION archives the currently-promoted one.</p>
 */
@Service
public class ModelService {

    private final ConcurrentHashMap<String, Model> models = new ConcurrentHashMap<>();
    private final AuditService auditService;

    /**
     * Construct the service.
     *
     * @param auditService audit service for recording model operations
     */
    public ModelService(AuditService auditService) {
        this.auditService = auditService;
    }

    /**
     * List all registered models.
     *
     * @return immutable list of models
     */
    public List<Model> list() {
        return List.copyOf(models.values());
    }

    /**
     * Get a model by id.
     *
     * @param id model identifier
     * @return the model, or empty if not found
     */
    public Optional<Model> get(String id) {
        return Optional.ofNullable(models.get(id));
    }

    /**
     * Register a new model artifact.
     *
     * @param request registration request
     * @param actor   acting principal
     * @return the registered model
     */
    public Model register(ModelRegisterRequest request, UserPrincipal actor) {
        Model model = new Model(
                UUID.randomUUID().toString(),
                request.name(),
                request.version(),
                ModelStage.NONE,
                request.artifactUri(),
                Instant.now());
        models.put(model.id(), model);
        auditService.record(actor, "model.register", Map.of("modelId", model.id(), "name", model.name()));
        return model;
    }

    /**
     * Promote a model version to the requested stage.
     *
     * @param id      model identifier
     * @param request promotion request
     * @param actor   acting principal
     * @return the promoted model
     * @throws IllegalStateException if the transition is invalid
     */
    public Model promote(String id, ModelPromoteRequest request, UserPrincipal actor) {
        Model model = models.get(id);
        if (model == null) {
            throw new IllegalArgumentException("Model not found: " + id);
        }
        ModelStage target = request.stage();
        validateTransition(model.stage(), target);
        // Enforce single production model per name.
        if (target == ModelStage.PRODUCTION) {
            models.values().stream()
                    .filter(m -> m.name().equals(model.name()) && m.stage() == ModelStage.PRODUCTION)
                    .forEach(m -> models.put(m.id(), new Model(m.id(), m.name(), m.version(),
                            ModelStage.ARCHIVED, m.artifactUri(), m.createdAt())));
        }
        Model promoted = new Model(model.id(), model.name(), model.version(), target, model.artifactUri(), model.createdAt());
        models.put(id, promoted);
        auditService.record(actor, "model.promote",
                Map.of("modelId", id, "from", model.stage().name(), "to", target.name()));
        return promoted;
    }

    private void validateTransition(ModelStage from, ModelStage to) {
        if (from == to) {
            return;
        }
        boolean valid = switch (from) {
            case NONE -> to == ModelStage.STAGING || to == ModelStage.ARCHIVED;
            case STAGING -> to == ModelStage.PRODUCTION || to == ModelStage.ARCHIVED;
            case PRODUCTION -> to == ModelStage.ARCHIVED;
            case ARCHIVED -> false;
        };
        if (!valid) {
            throw new IllegalStateException(
                    "Invalid model stage transition: " + from + " -> " + to);
        }
    }
}
