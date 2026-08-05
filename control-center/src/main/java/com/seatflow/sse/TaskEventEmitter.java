package com.seatflow.sse;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import org.springframework.stereotype.Component;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

/**
 * Manages SSE subscriptions per task id. When a task event is emitted it is
 * broadcast to every subscriber of the corresponding task.
 *
 * <p>Subscribers that error or complete are automatically removed.</p>
 */
@Component
public class TaskEventEmitter {

    private static final long TIMEOUT_MS = 300_000L;

    private final Map<String, List<SseEmitter>> subscriptions = new ConcurrentHashMap<>();

    /**
     * Create and register a new SSE subscription for a task.
     *
     * @param taskId task identifier to subscribe to
     * @return a configured SseEmitter
     */
    public SseEmitter subscribe(String taskId) {
        SseEmitter emitter = new SseEmitter(TIMEOUT_MS);
        subscriptions.computeIfAbsent(taskId, k -> new CopyOnWriteArrayList<>()).add(emitter);
        emitter.onCompletion(() -> remove(taskId, emitter));
        emitter.onTimeout(() -> remove(taskId, emitter));
        emitter.onError(e -> remove(taskId, emitter));
        return emitter;
    }

    /**
     * Emit a task event to all subscribers of the event's task.
     *
     * @param event task event
     */
    public void emit(TaskEvent event) {
        List<SseEmitter> emitters = subscriptions.get(event.taskId());
        if (emitters == null || emitters.isEmpty()) {
            return;
        }
        SseEmitter.SseEventBuilder builder = SseEmitter.event()
                .id(UUID.randomUUID().toString())
                .name(event.eventType().name())
                .data(event);
        for (SseEmitter emitter : emitters) {
            try {
                emitter.send(builder);
            } catch (IOException | IllegalStateException ex) {
                remove(event.taskId(), emitter);
            }
        }
    }

    private void remove(String taskId, SseEmitter emitter) {
        List<SseEmitter> emitters = subscriptions.get(taskId);
        if (emitters != null) {
            emitters.remove(emitter);
            emitter.complete();
            if (emitters.isEmpty()) {
                subscriptions.remove(taskId);
            }
        }
    }
}
