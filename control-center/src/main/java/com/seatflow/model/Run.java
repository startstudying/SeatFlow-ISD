package com.seatflow.model;

import java.time.Instant;
import java.util.Map;

/**
 * A single training or evaluation run within an experiment.
 *
 * @param id          unique run identifier
 * @param experimentId parent experiment identifier
 * @param status      current lifecycle state
 * @param config      run configuration key-value pairs
 * @param startedAt   time the run started executing
 * @param finishedAt  time the run reached a terminal state
 */
public record Run(
        String id,
        String experimentId,
        RunStatus status,
        Map<String, Object> config,
        Instant startedAt,
        Instant finishedAt) {
}
