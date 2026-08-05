package com.seatflow.service;

import com.seatflow.model.AuthRequest;
import com.seatflow.model.AuthResponse;
import com.seatflow.security.JwtTokenProvider;
import com.seatflow.security.Role;
import com.seatflow.security.UserPrincipal;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

/**
 * Authentication service: credential verification and JWT session issuance.
 *
 * <p>A small in-memory user directory is seeded with demo accounts so that the
 * control center is usable out of the box. Production deployments should
 * replace this with a persistent user store.</p>
 */
@Service
public class AuthService {

    private final JwtTokenProvider tokenProvider;
    private final PasswordEncoder passwordEncoder;
    private final AuditService auditService;
    private final Map<String, StoredUser> users = new ConcurrentHashMap<>();

    /**
     * Construct the service and seed demo users.
     *
     * @param tokenProvider  JWT token provider
     * @param passwordEncoder password encoder
     * @param auditService   audit service
     */
    public AuthService(JwtTokenProvider tokenProvider, PasswordEncoder passwordEncoder, AuditService auditService) {
        this.tokenProvider = tokenProvider;
        this.passwordEncoder = passwordEncoder;
        this.auditService = auditService;
        seedUsers();
    }

    private void seedUsers() {
        register("admin@seatflow.com", "admin", "SeatFlow Admin", List.of(Role.ADMIN));
        register("researcher@seatflow.com", "researcher", "SeatFlow Researcher", List.of(Role.RESEARCHER));
        register("viewer@seatflow.com", "viewer", "SeatFlow Viewer", List.of(Role.VIEWER));
    }

    /**
     * Register a new user.
     *
     * @param email       user email
     * @param password    plaintext password
     * @param displayName display name
     * @param roles       assigned roles
     */
    public void register(String email, String password, String displayName, List<Role> roles) {
        users.put(email.toLowerCase(), new StoredUser(
                UUID.randomUUID().toString(),
                email.toLowerCase(),
                passwordEncoder.encode(password),
                displayName,
                roles));
    }

    /**
     * Authenticate and issue a session token.
     *
     * @param request login credentials
     * @return session response, or null if credentials are invalid
     */
    public AuthResponse login(AuthRequest request) {
        StoredUser user = users.get(request.email().toLowerCase());
        if (user == null || !passwordEncoder.matches(request.password(), user.passwordHash())) {
            return null;
        }
        return issueSession(user);
    }

    /**
     * Refresh the current session, issuing a new token.
     *
     * @param principal authenticated principal
     * @return refreshed session response
     */
    public AuthResponse refresh(UserPrincipal principal) {
        StoredUser user = users.get(principal.email().toLowerCase());
        if (user == null) {
            return null;
        }
        return issueSession(user);
    }

    /**
     * Terminate the current session. With stateless JWTs this is primarily an
     * audit-recording operation; a production deployment should also revoke the
     * token via a denylist.
     *
     * @param principal authenticated principal
     */
    public void logout(UserPrincipal principal) {
        auditService.record(principal, "auth.logout", Map.of("userId", principal.userId()));
    }

    private AuthResponse issueSession(StoredUser user) {
        UserPrincipal principal = new UserPrincipal(user.id(), user.email(), user.displayName(), user.roles());
        String token = tokenProvider.generateToken(principal);
        Instant expiresAt = tokenProvider.expiryFromNow();
        auditService.record(principal, "auth.login", Map.of("userId", user.id()));
        return new AuthResponse(token, expiresAt, user.id(), user.email(), user.displayName(),
                user.roles().stream().map(Enum::name).toList());
    }

    private record StoredUser(
            String id,
            String email,
            String passwordHash,
            String displayName,
            List<Role> roles) {
    }
}
