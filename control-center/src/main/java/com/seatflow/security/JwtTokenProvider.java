package com.seatflow.security;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.JwtException;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.security.Keys;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.Date;
import java.util.List;
import javax.crypto.SecretKey;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

/**
 * Creates and validates SeatFlow session JWTs.
 *
 * <p>Tokens carry the user id, email, display name and roles as claims. The
 * signing key and validity window are configured via application properties
 * ({@code seatflow.security.jwt.secret} and {@code seatflow.security.jwt.ttl}).</p>
 */
@Component
public class JwtTokenProvider {

    /** Default signing secret used when none is configured (development only). */
    static final String DEFAULT_SECRET = "seatflow-dev-secret-key-please-override-in-production-0123456789";

    private final SecretKey signingKey;
    private final Duration ttl;

    /**
     * Construct the provider.
     *
     * @param secret signing secret (must be at least 256 bits)
     * @param ttl   token time-to-live in ISO-8601 duration form (e.g. {@code PT1H})
     */
    public JwtTokenProvider(
            @Value("${seatflow.security.jwt.secret:" + DEFAULT_SECRET + "}") String secret,
            @Value("${seatflow.security.jwt.ttl:PT1H}") Duration ttl) {
        byte[] keyBytes = secret.getBytes(StandardCharsets.UTF_8);
        this.signingKey = Keys.hmacShaKeyFor(keyBytes);
        this.ttl = ttl;
    }

    /**
     * Generate a signed JWT for the given principal.
     *
     * @param principal authenticated user
     * @return signed compact JWT
     */
    public String generateToken(UserPrincipal principal) {
        Instant now = Instant.now();
        Instant exp = now.plus(ttl);
        return Jwts.builder()
                .subject(principal.userId())
                .claim("email", principal.email())
                .claim("name", principal.displayName())
                .claim("roles", principal.roles().stream().map(Enum::name).toList())
                .issuedAt(Date.from(now))
                .expiration(Date.from(exp))
                .signWith(signingKey)
                .compact();
    }

    /**
     * Resolve and validate a compact JWT, returning the parsed claims.
     *
     * @param token compact JWT
     * @return parsed claims, or null if the token is invalid or expired
     */
    public Claims parse(String token) {
        try {
            return Jwts.parser()
                    .verifyWith(signingKey)
                    .build()
                    .parseSignedClaims(token)
                    .getPayload();
        } catch (JwtException | IllegalArgumentException e) {
            return null;
        }
    }

    /**
     * Validate a compact JWT.
     *
     * @param token compact JWT
     * @return true if the token is well-formed, correctly signed and not expired
     */
    public boolean validate(String token) {
        return parse(token) != null;
    }

    /**
     * Build a {@link UserPrincipal} from parsed claims.
     *
     * @param claims parsed JWT claims
     * @return reconstructed principal
     */
    @SuppressWarnings("unchecked")
    public UserPrincipal toPrincipal(Claims claims) {
        List<String> roleNames = claims.get("roles", List.class);
        List<Role> roles = roleNames == null
                ? List.of()
                : roleNames.stream().map(Role::valueOf).toList();
        return new UserPrincipal(
                claims.getSubject(),
                claims.get("email", String.class),
                claims.get("name", String.class),
                roles);
    }

    /**
     * Compute the expiry instant for tokens issued now.
     *
     * @return expiry instant
     */
    public Instant expiryFromNow() {
        return Instant.now().plus(ttl);
    }

    /**
     * @return configured token time-to-live
     */
    public Duration ttl() {
        return ttl;
    }
}
