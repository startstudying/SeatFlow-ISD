package com.seatflow.service;

import com.seatflow.model.AuditEvent;
import com.seatflow.security.UserPrincipal;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import org.springframework.stereotype.Service;

/**
 * In-memory audit event log. Records every state-changing operation with the
 * acting principal, action code, timestamp and affected entity.
 *
 * <p>The log is append-only and thread-safe. Querying supports optional filters
 * on actor, action and time range.</p>
 */
@Service
public class AuditService {

    private final List<AuditEvent> events = new CopyOnWriteArrayList<>();

    /**
     * Record an audit event.
     *
     * @param actor  acting user principal (may be null for system events)
     * @param action stable action code (e.g. {@code project.create})
     * @param entity optional structured entity reference
     * @return the recorded event
     */
    public AuditEvent record(UserPrincipal actor, String action, Map<String, Object> entity) {
        AuditEvent event = new AuditEvent(
                UUID.randomUUID().toString(),
                actor == null ? "system" : actor.email(),
                action,
                Instant.now(),
                entity);
        events.add(event);
        return event;
    }

    /**
     * Query the audit log.
     *
     * @param actor  optional actor filter
     * @param action optional action filter
     * @param since  optional inclusive lower timestamp bound
     * @param until  optional exclusive upper timestamp bound
     * @return matching events, newest first
     */
    public List<AuditEvent> query(String actor, String action, Instant since, Instant until) {
        List<Predicate<AuditEvent>> filters = new ArrayList<>();
        if (actor != null && !actor.isBlank()) {
            filters.add(e -> actor.equals(e.actor()));
        }
        if (action != null && !action.isBlank()) {
            filters.add(e -> action.equals(e.action()));
        }
        if (since != null) {
            filters.add(e -> !e.timestamp().isBefore(since));
        }
        if (until != null) {
            filters.add(e -> e.timestamp().isBefore(until));
        }
        return events.stream()
                .filter(e -> filters.stream().allMatch(f -> f.test(e)))
                .collect(Collectors.toList());
    }

    /**
     * @return all recorded audit events (defensive copy)
     */
    public List<AuditEvent> all() {
        return new ArrayList<>(events);
    }
}
