package com.seatflow.backend.settings.service;

import com.seatflow.backend.api.ErrorPayload;
import com.seatflow.backend.service.ApiException;
import java.util.List;
import org.springframework.http.HttpStatus;

final class SettingsErrors {
    private SettingsErrors() {
    }

    static ApiException notFound(String resource, String id) {
        return new ApiException(HttpStatus.NOT_FOUND, "SETTINGS_NOT_FOUND",
                resource + " not found: " + id, ErrorPayload.failClosed(resource + " not found", List.of("not_found")));
    }

    static ApiException versionConflict(long expected, long actual) {
        return new ApiException(HttpStatus.CONFLICT, "VERSION_CONFLICT",
                "expected version " + expected + " but current version is " + actual,
                ErrorPayload.failClosed("version conflict", List.of("version_conflict")));
    }

    static ApiException idempotencyConflict() {
        return new ApiException(HttpStatus.CONFLICT, "IDEMPOTENCY_CONFLICT",
                "idempotency key was already used for a different operation",
                ErrorPayload.failClosed("idempotency conflict", List.of("idempotency_conflict")));
    }

    static ApiException invalid(String message) {
        return new ApiException(HttpStatus.BAD_REQUEST, "INVALID_SETTINGS_REQUEST",
                message, ErrorPayload.failClosed(message, List.of("invalid_settings_request")));
    }
}
