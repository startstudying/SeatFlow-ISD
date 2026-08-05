package com.seatflow.security;

import io.jsonwebtoken.Claims;
import java.time.Duration;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for {@link JwtTokenProvider} that do not require a Spring context.
 */
class JwtTokenProviderTest {

    private static final String SECRET = "seatflow-test-secret-key-for-testing-only-0123456789abcdef";

    private JwtTokenProvider provider;

    @BeforeEach
    void setUp() {
        provider = new JwtTokenProvider(SECRET, Duration.ofHours(1));
    }

    @Test
    void generateToken_returnsNonEmptyToken() {
        UserPrincipal principal = new UserPrincipal("u-1", "admin@seatflow.com", "Admin", List.of(Role.ADMIN));
        String token = provider.generateToken(principal);
        assertNotNull(token);
        assertFalse(token.isBlank());
    }

    @Test
    void validate_validToken_returnsTrue() {
        UserPrincipal principal = new UserPrincipal("u-1", "admin@seatflow.com", "Admin", List.of(Role.ADMIN));
        String token = provider.generateToken(principal);
        assertTrue(provider.validate(token));
    }

    @Test
    void validate_invalidToken_returnsFalse() {
        assertFalse(provider.validate("not-a-jwt"));
    }

    @Test
    void validate_nullToken_returnsFalse() {
        assertFalse(provider.validate(null));
    }

    @Test
    void validate_tamperedToken_returnsFalse() {
        UserPrincipal principal = new UserPrincipal("u-1", "admin@seatflow.com", "Admin", List.of(Role.ADMIN));
        String token = provider.generateToken(principal);
        String tampered = token.substring(0, token.length() - 5) + "XXXXX";
        assertFalse(provider.validate(tampered));
    }

    @Test
    void parse_validToken_returnsClaims() {
        UserPrincipal principal = new UserPrincipal("u-1", "admin@seatflow.com", "Admin", List.of(Role.ADMIN, Role.RESEARCHER));
        String token = provider.generateToken(principal);

        Claims claims = provider.parse(token);
        assertNotNull(claims);
        assertEquals("u-1", claims.getSubject());
        assertEquals("admin@seatflow.com", claims.get("email"));
        assertEquals("Admin", claims.get("name"));
        @SuppressWarnings("unchecked")
        List<String> roles = claims.get("roles", List.class);
        assertTrue(roles.contains("ADMIN"));
        assertTrue(roles.contains("RESEARCHER"));
    }

    @Test
    void parse_invalidToken_returnsNull() {
        assertNull(provider.parse("not-a-jwt"));
    }

    @Test
    void toPrincipal_reconstructsAllFields() {
        UserPrincipal original = new UserPrincipal("u-1", "admin@seatflow.com", "Admin", List.of(Role.ADMIN));
        String token = provider.generateToken(original);
        Claims claims = provider.parse(token);
        assertNotNull(claims);

        UserPrincipal reconstructed = provider.toPrincipal(claims);
        assertEquals(original.userId(), reconstructed.userId());
        assertEquals(original.email(), reconstructed.email());
        assertEquals(original.displayName(), reconstructed.displayName());
        assertEquals(original.roles(), reconstructed.roles());
    }

    @Test
    void expiryFromNow_isInFuture() {
        UserPrincipal principal = new UserPrincipal("u-1", "admin@seatflow.com", "Admin", List.of(Role.ADMIN));
        provider.generateToken(principal);
        assertTrue(provider.expiryFromNow().isAfter(java.time.Instant.now()));
    }

    @Test
    void ttl_returnsConfiguredDuration() {
        assertEquals(Duration.ofHours(1), provider.ttl());
    }

    @Test
    void generateToken_withDifferentSecret_isInvalid() {
        JwtTokenProvider other = new JwtTokenProvider("a-different-secret-key-for-testing-only-9876543210fedcba", Duration.ofHours(1));
        UserPrincipal principal = new UserPrincipal("u-1", "admin@seatflow.com", "Admin", List.of(Role.ADMIN));
        String token = other.generateToken(principal);

        // Token signed with different key should fail validation
        assertFalse(provider.validate(token));
        assertNull(provider.parse(token));
    }
}
