package com.seatflow.backend.settings.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "sf_permission")
public class PermissionDefinition extends BaseVersionedEntity {
    @Id
    @Column(name = "permission_code", length = 96)
    private String code;

    @Column(nullable = false, length = 128)
    private String name;

    @Column(nullable = false, length = 64)
    private String resource;

    protected PermissionDefinition() {
    }

    public PermissionDefinition(String code) {
        this.code = code;
    }

    public String getCode() { return code; }
    public String getName() { return name; }
    public String getResource() { return resource; }

    public void update(String name, String resource) {
        this.name = name;
        this.resource = resource;
    }
}
