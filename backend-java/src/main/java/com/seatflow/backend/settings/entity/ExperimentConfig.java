package com.seatflow.backend.settings.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "sf_experiment_config")
public class ExperimentConfig extends BaseVersionedEntity {
    @Id
    @Column(name = "experiment_id", length = 128)
    private String id;

    @Column(nullable = false, length = 128)
    private String name;

    @Column(nullable = false, length = 64)
    private String status;

    @Column(columnDefinition = "TEXT")
    private String configJson;

    protected ExperimentConfig() {
    }

    public ExperimentConfig(String id) {
        this.id = id;
    }

    public String getId() { return id; }
    public String getName() { return name; }
    public String getStatus() { return status; }
    public String getConfigJson() { return configJson; }

    public void update(String name, String status, String configJson) {
        this.name = name;
        this.status = status;
        this.configJson = configJson;
    }
}
