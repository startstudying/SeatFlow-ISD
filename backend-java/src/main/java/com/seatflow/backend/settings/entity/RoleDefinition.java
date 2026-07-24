package com.seatflow.backend.settings.entity;

import jakarta.persistence.CollectionTable;
import jakarta.persistence.Column;
import jakarta.persistence.ElementCollection;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.Table;
import java.util.LinkedHashSet;
import java.util.Set;

@Entity
@Table(name = "sf_role")
public class RoleDefinition extends BaseVersionedEntity {
    @Id
    @Column(name = "role_code", length = 64)
    private String code;

    @Column(nullable = false, length = 128)
    private String name;

    @ElementCollection(fetch = FetchType.EAGER)
    @CollectionTable(name = "sf_role_permission", joinColumns = @JoinColumn(name = "role_code"))
    @Column(name = "permission_code", length = 96)
    private Set<String> permissions = new LinkedHashSet<>();

    protected RoleDefinition() {
    }

    public RoleDefinition(String code) {
        this.code = code;
    }

    public String getCode() { return code; }
    public String getName() { return name; }
    public Set<String> getPermissions() { return permissions; }

    public void update(String name, Set<String> permissions) {
        this.name = name;
        this.permissions = permissions == null ? new LinkedHashSet<>() : new LinkedHashSet<>(permissions);
    }
}
