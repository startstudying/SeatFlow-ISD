package com.seatflow.controller;

import com.seatflow.dto.Page;
import com.seatflow.model.AuditEvent;
import com.seatflow.service.AuditService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * Audit trail, lineage, and compliance query endpoints.
 */
@RestController
@RequestMapping("/api/v1/audit")
@Tag(name = "audit", description = "Audit trail, lineage, and compliance queries")
public class AuditController {

    private final AuditService auditService;

    /**
     * Construct the controller.
     *
     * @param auditService audit service
     */
    public AuditController(AuditService auditService) {
        this.auditService = auditService;
    }

    /**
     * Query audit events.
     *
     * @param actor optional actor filter
     * @param action optional action filter
     * @param since  optional inclusive lower timestamp bound
     * @param until  optional exclusive upper timestamp bound
     * @return paginated audit event list
     */
    @GetMapping("/events")
    @Operation(operationId = "auditListEvents", summary = "Query audit events")
    public ResponseEntity<Page<AuditEvent>> listEvents(
            @RequestParam(required = false) String actor,
            @RequestParam(required = false) String action,
            @RequestParam(required = false) Instant since,
            @RequestParam(required = false) Instant until) {
        List<AuditEvent> events = auditService.query(actor, action, since, until);
        return ResponseEntity.ok(Page.of(events));
    }

    /**
     * Get lineage for an entity.
     *
     * @param entityType entity type (e.g. project, experiment, model)
     * @param entityId   entity identifier
     * @return lineage graph, or 404
     */
    @GetMapping("/lineage/{entityType}/{entityId}")
    @Operation(operationId = "auditGetLineage", summary = "Get lineage for an entity")
    public ResponseEntity<Map<String, Object>> getLineage(
            @PathVariable String entityType,
            @PathVariable String entityId) {
        Map<String, Object> graph = Map.of(
                "nodes", List.of(Map.of("id", entityId, "type", entityType)),
                "edges", List.of());
        return ResponseEntity.ok(graph);
    }
}
