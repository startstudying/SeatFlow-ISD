package com.seatflow.sse;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

/**
 * SSE endpoint for streaming task lifecycle events.
 */
@RestController
@RequestMapping("/api/v1/system/tasks")
@Tag(name = "system", description = "Health, version, and operational endpoints")
public class TaskEventController {

    private final TaskEventEmitter emitter;

    /**
     * Construct the controller.
     *
     * @param emitter task event emitter
     */
    public TaskEventController(TaskEventEmitter emitter) {
        this.emitter = emitter;
    }

    /**
     * Stream task events via Server-Sent Events.
     *
     * @param taskId task identifier
     * @return SSE emitter that will push task events
     */
    @GetMapping(value = "/{taskId}/events", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    @Operation(operationId = "systemStreamTaskEvents", summary = "Stream task events via Server-Sent Events")
    public SseEmitter streamTaskEvents(@PathVariable String taskId) {
        return emitter.subscribe(taskId);
    }
}
