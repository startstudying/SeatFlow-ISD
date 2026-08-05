package com.seatflow.dto;

import com.seatflow.model.TaskStatus;
import java.time.Instant;

/**
 * Handle returned for long-running asynchronous operations.
 *
 * @param taskId     task identifier
 * @param status     current task status
 * @param acceptedAt time the task was accepted
 * @param pollUrl    URL to poll task state
 * @param streamUrl  URL to stream task events via SSE
 */
public record TaskHandle(
        String taskId,
        TaskStatus status,
        Instant acceptedAt,
        String pollUrl,
        String streamUrl) {
}
