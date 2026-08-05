package com.seatflow.dto;

import java.util.Map;

/**
 * Service health report.
 *
 * @param status     overall status (UP, DOWN, DEGRADED)
 * @param components per-component health details
 */
public record HealthReport(
        String status,
        Map<String, Object> components) {
}
