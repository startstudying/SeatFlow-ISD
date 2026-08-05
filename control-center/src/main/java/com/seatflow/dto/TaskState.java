package com.seatflow.dto;

import com.seatflow.model.TaskStatus;
import java.time.Instant;
import java.util.Map;

/**
 * Current state of a long-running task.
 *
 * @param taskId     task identifier
 * @param status     current task status
 * @param progress   progress fraction in [0,1]
 * @param updatedAt  time of last update
 * @param result     optional result payload
 * @param error      optional error detail
 */
public record TaskState(
        String taskId,
        TaskStatus status,
        Double progress,
        Instant updatedAt,
        Map<String, Object> result,
        Map<String, Object> error) {
}
