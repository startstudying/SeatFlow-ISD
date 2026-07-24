package com.seatflow.backend.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "seatflow.encryption")
public record EncryptionProperties(
        String key,
        String keyId
) {
    public EncryptionProperties {
        keyId = keyId == null || keyId.isBlank() ? "local" : keyId;
    }
}
