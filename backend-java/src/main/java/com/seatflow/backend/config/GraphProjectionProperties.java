package com.seatflow.backend.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "seatflow.neo4j")
public record GraphProjectionProperties(
        boolean enabled,
        String baseUrl,
        String username,
        String password,
        String database
) {
    public GraphProjectionProperties {
        baseUrl = baseUrl == null || baseUrl.isBlank() ? "http://localhost:7474" : baseUrl.replaceAll("/$", "");
        username = username == null || username.isBlank() ? "neo4j" : username;
        database = database == null || database.isBlank() ? "neo4j" : database;
    }
}
