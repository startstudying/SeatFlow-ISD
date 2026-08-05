package com.seatflow.model;

import java.time.Instant;

/**
 * Session response body mapping to the {@code Session} OpenAPI schema.
 *
 * @param token     JWT session token
 * @param expiresAt token expiry timestamp (UTC)
 * @param userId    authenticated user identifier
 * @param email     authenticated user email
 * @param displayName human-readable display name
 * @param roles     assigned role names
 */
public record AuthResponse(
        String token,
        Instant expiresAt,
        String userId,
        String email,
        String displayName,
        java.util.List<String> roles) {
}
