package com.seatflow.backend.settings.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "sf_model_config")
public class ModelConfig extends BaseVersionedEntity {
    @Id
    @Column(name = "model_id", length = 128)
    private String id;

    @Column(nullable = false, length = 128)
    private String displayName;

    @Column(nullable = false, length = 64)
    private String provider;

    @Column(nullable = false, length = 64)
    private String status;

    @Column(columnDefinition = "TEXT")
    private String parametersJson;

    protected ModelConfig() {
    }

    public ModelConfig(String id) {
        this.id = id;
    }

    public String getId() { return id; }
    public String getDisplayName() { return displayName; }
    public String getProvider() { return provider; }
    public String getStatus() { return status; }
    public String getParametersJson() { return parametersJson; }

    public void update(String displayName, String provider, String status, String parametersJson) {
        this.displayName = displayName;
        this.provider = provider;
        this.status = status;
        this.parametersJson = parametersJson;
    }
}
