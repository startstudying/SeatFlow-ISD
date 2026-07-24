package com.seatflow.backend.settings.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "sf_data_source_config")
public class DataSourceConfig extends BaseVersionedEntity {
    @Id
    @Column(name = "source_id", length = 128)
    private String id;

    @Column(nullable = false, length = 128)
    private String displayName;

    @Column(nullable = false, length = 64)
    private String provider;

    @Column(nullable = false, length = 512)
    private String endpoint;

    @Column(columnDefinition = "TEXT")
    private String metadataJson;

    @Column(length = 64)
    private String apiKeyKeyId;

    @Column(columnDefinition = "TEXT")
    private String encryptedApiKey;

    protected DataSourceConfig() {
    }

    public DataSourceConfig(String id) {
        this.id = id;
    }

    public String getId() { return id; }
    public String getDisplayName() { return displayName; }
    public String getProvider() { return provider; }
    public String getEndpoint() { return endpoint; }
    public String getMetadataJson() { return metadataJson; }
    public String getApiKeyKeyId() { return apiKeyKeyId; }
    public boolean hasApiKey() { return encryptedApiKey != null && !encryptedApiKey.isBlank(); }

    public void update(String displayName, String provider, String endpoint, String metadataJson) {
        this.displayName = displayName;
        this.provider = provider;
        this.endpoint = endpoint;
        this.metadataJson = metadataJson;
    }

    public void storeSecret(String apiKeyKeyId, String encryptedApiKey) {
        this.apiKeyKeyId = apiKeyKeyId;
        this.encryptedApiKey = encryptedApiKey;
    }
}
