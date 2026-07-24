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
@Table(name = "sf_user_account")
public class UserAccount extends BaseVersionedEntity {
    @Id
    @Column(name = "user_id", length = 128)
    private String id;

    @Column(nullable = false, length = 128)
    private String displayName;

    @Column(nullable = false)
    private boolean active;

    @ElementCollection(fetch = FetchType.EAGER)
    @CollectionTable(name = "sf_user_role", joinColumns = @JoinColumn(name = "user_id"))
    @Column(name = "role_code", length = 64)
    private Set<String> roles = new LinkedHashSet<>();

    protected UserAccount() {
    }

    public UserAccount(String id) {
        this.id = id;
    }

    public String getId() { return id; }
    public String getDisplayName() { return displayName; }
    public boolean isActive() { return active; }
    public Set<String> getRoles() { return roles; }

    public void update(String displayName, boolean active, Set<String> roles) {
        this.displayName = displayName;
        this.active = active;
        this.roles = roles == null ? new LinkedHashSet<>() : new LinkedHashSet<>(roles);
    }
}
