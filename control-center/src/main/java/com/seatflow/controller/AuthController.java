package com.seatflow.controller;

import com.seatflow.model.AuthRequest;
import com.seatflow.model.AuthResponse;
import com.seatflow.security.UserPrincipal;
import com.seatflow.service.AuthService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Authentication and session management endpoints.
 */
@RestController
@RequestMapping("/api/v1/auth")
@Tag(name = "auth", description = "Authentication and session management")
public class AuthController {

    private final AuthService authService;

    /**
     * Construct the controller.
     *
     * @param authService authentication service
     */
    public AuthController(AuthService authService) {
        this.authService = authService;
    }

    /**
     * Exchange credentials for a session token.
     *
     * @param request login credentials
     * @return session response, or 401 on invalid credentials
     */
    @PostMapping("/login")
    @Operation(operationId = "authLogin", summary = "Exchange credentials for a session token")
    public ResponseEntity<AuthResponse> login(@RequestBody AuthRequest request) {
        AuthResponse response = authService.login(request);
        if (response == null) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).build();
        }
        return ResponseEntity.ok(response);
    }

    /**
     * Refresh an expiring session token.
     *
     * @param principal authenticated principal
     * @return refreshed session response, or 401 if the session is invalid
     */
    @PostMapping("/refresh")
    @Operation(operationId = "authRefresh", summary = "Refresh an expiring session token")
    public ResponseEntity<AuthResponse> refresh(@AuthenticationPrincipal UserPrincipal principal) {
        if (principal == null) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).build();
        }
        AuthResponse response = authService.refresh(principal);
        if (response == null) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).build();
        }
        return ResponseEntity.ok(response);
    }

    /**
     * Terminate the current session.
     *
     * @param principal authenticated principal
     * @return 204 on success
     */
    @PostMapping("/logout")
    @Operation(operationId = "authLogout", summary = "Terminate the current session")
    public ResponseEntity<Void> logout(@AuthenticationPrincipal UserPrincipal principal) {
        if (principal != null) {
            authService.logout(principal);
        }
        return ResponseEntity.noContent().build();
    }

    /**
     * Get the current authenticated user.
     *
     * @param principal authenticated principal
     * @return user profile
     */
    @GetMapping("/me")
    @Operation(operationId = "authGetCurrentUser", summary = "Get the current authenticated user")
    public ResponseEntity<UserPrincipal> me(@AuthenticationPrincipal UserPrincipal principal) {
        if (principal == null) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).build();
        }
        return ResponseEntity.ok(principal);
    }
}
