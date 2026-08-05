package com.seatflow.service;

import com.seatflow.dto.ExperimentCreateRequest;
import com.seatflow.dto.RunCreateRequest;
import com.seatflow.model.Experiment;
import com.seatflow.model.Run;
import com.seatflow.model.RunStatus;
import com.seatflow.security.UserPrincipal;
import java.time.Instant;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import org.springframework.stereotype.Service;

/**
 * In-memory experiment and run store.
 *
 * <p>Experiments are grouped under projects and contain runs. Runs are created
 * with a PENDING status. Production deployments should replace this with a
 * JPA-backed implementation backed by PostgreSQL.</p>
 */
@Service
public class ExperimentService {

    private final ConcurrentHashMap<String, Experiment> experiments = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Run> runs = new ConcurrentHashMap<>();
    private final AuditService auditService;

    /**
     * Construct the service.
     *
     * @param auditService audit service for recording experiment/run operations
     */
    public ExperimentService(AuditService auditService) {
        this.auditService = auditService;
    }

    /**
     * List all experiments.
     *
     * @return immutable list of experiments
     */
    public List<Experiment> listExperiments() {
        return List.copyOf(experiments.values());
    }

    /**
     * Get an experiment by id.
     *
     * @param id experiment identifier
     * @return the experiment, or empty if not found
     */
    public Optional<Experiment> getExperiment(String id) {
        return Optional.ofNullable(experiments.get(id));
    }

    /**
     * Create a new experiment.
     *
     * @param request creation request
     * @param actor  acting principal
     * @return the created experiment
     */
    public Experiment createExperiment(ExperimentCreateRequest request, UserPrincipal actor) {
        Experiment experiment = new Experiment(
                UUID.randomUUID().toString(),
                request.name(),
                request.description(),
                request.projectId(),
                Instant.now());
        experiments.put(experiment.id(), experiment);
        auditService.record(actor, "experiment.create", Map.of("experimentId", experiment.id()));
        return experiment;
    }

    /**
     * Submit a new run.
     *
     * @param request run creation request
     * @param actor   acting principal
     * @return the created run
     */
    public Run createRun(RunCreateRequest request, UserPrincipal actor) {
        Run run = new Run(
                UUID.randomUUID().toString(),
                request.experimentId(),
                RunStatus.PENDING,
                request.config() == null ? Collections.emptyMap() : Map.copyOf(request.config()),
                Instant.now(),
                null);
        runs.put(run.id(), run);
        auditService.record(actor, "run.create", Map.of("runId", run.id(), "experimentId", run.experimentId()));
        return run;
    }

    /**
     * List runs, optionally filtered by experimentId and/or status.
     *
     * @param experimentId optional experiment filter
     * @param status        optional status filter
     * @return immutable list of matching runs
     */
    public List<Run> listRuns(String experimentId, RunStatus status) {
        return runs.values().stream()
                .filter(r -> experimentId == null || experimentId.isBlank() || experimentId.equals(r.experimentId()))
                .filter(r -> status == null || status == r.status())
                .toList();
    }

    /**
     * Get a run by id.
     *
     * @param id run identifier
     * @return the run, or empty if not found
     */
    public Optional<Run> getRun(String id) {
        return Optional.ofNullable(runs.get(id));
    }

    /**
     * Cancel a running run.
     *
     * @param id    run identifier
     * @param actor acting principal
     * @return the updated run, or empty if not found
     */
    public Optional<Run> cancelRun(String id, UserPrincipal actor) {
        return getRun(id).map(existing -> {
            Run cancelled = new Run(existing.id(), existing.experimentId(), RunStatus.CANCELLED,
                    existing.config(), existing.startedAt(), Instant.now());
            runs.put(id, cancelled);
            auditService.record(actor, "run.cancel", Map.of("runId", id));
            return cancelled;
        });
    }
}
