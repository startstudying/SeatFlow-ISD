package com.seatflow.backend.config;

import java.time.Duration;
import org.springframework.boot.context.properties.bind.ConstructorBinding;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "seatflow")
public record SeatFlowProperties(
        String scope,
        String authMode,
        Security security,
        FastApi fastapi
) {
    public SeatFlowProperties(String scope, String authMode, FastApi fastapi) {
        this(scope, authMode, null, fastapi);
    }

    @ConstructorBinding
    public SeatFlowProperties {
        scope = scope == null || scope.isBlank() ? "local_internal_research" : scope;
        authMode = authMode == null || authMode.isBlank() ? "local-internal" : authMode;
        security = security == null
                ? new Security(true, "X-SeatFlow-Api-Key", null, "local-internal", java.util.List.of("ADMIN"))
                : security;
        fastapi = fastapi == null
                ? new FastApi("http://localhost:8000", Duration.ofSeconds(2), Duration.ofSeconds(5))
                : fastapi;
    }

    public record Security(
            boolean enabled,
            String apiKeyHeader,
            String apiKey,
            String defaultActor,
            java.util.List<String> defaultRoles
    ) {
        public Security {
            apiKeyHeader = apiKeyHeader == null || apiKeyHeader.isBlank() ? "X-SeatFlow-Api-Key" : apiKeyHeader;
            defaultActor = defaultActor == null || defaultActor.isBlank() ? "local-internal" : defaultActor;
            defaultRoles = defaultRoles == null || defaultRoles.isEmpty()
                    ? java.util.List.of("ADMIN")
                    : java.util.List.copyOf(defaultRoles);
        }
    }

    public record FastApi(
            String baseUrl,
            Duration connectTimeout,
            Duration readTimeout
    ) {
        public FastApi {
            baseUrl = baseUrl == null || baseUrl.isBlank() ? "http://localhost:8000" : baseUrl;
            connectTimeout = connectTimeout == null ? Duration.ofSeconds(2) : connectTimeout;
            readTimeout = readTimeout == null ? Duration.ofSeconds(5) : readTimeout;
        }
    }
}
