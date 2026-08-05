package com.seatflow.model;

/**
 * Login request body mapping to the {@code LoginRequest} OpenAPI schema.
 *
 * @param email    user email address
 * @param password user password
 */
public record AuthRequest(
        String email,
        String password) {
}
