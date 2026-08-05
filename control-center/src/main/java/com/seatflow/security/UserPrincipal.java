package com.seatflow.security;

import java.util.Collection;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

/**
 * Authenticated SeatFlow user principal, wrapping a JWT-derived identity with
 * RBAC roles and derived permissions.
 *
 * @param userId      stable user identifier
 * @param email       user email address
 * @param displayName human-readable display name
 * @param roles       assigned roles
 */
public record UserPrincipal(
        String userId,
        String email,
        String displayName,
        List<Role> roles) implements UserDetails {

    @Override
    public Collection<? extends GrantedAuthority> getAuthorities() {
        if (roles == null || roles.isEmpty()) {
            return Collections.emptyList();
        }
        return roles.stream()
                .map(role -> new SimpleGrantedAuthority("ROLE_" + role.name()))
                .toList();
    }

    @Override
    public String getPassword() {
        return "";
    }

    @Override
    public String getUsername() {
        return email;
    }

    @Override
    public boolean isAccountNonExpired() {
        return true;
    }

    @Override
    public boolean isAccountNonLocked() {
        return true;
    }

    @Override
    public boolean isCredentialsNonExpired() {
        return true;
    }

    @Override
    public boolean isEnabled() {
        return true;
    }

    /**
     * Permissions derived from the assigned roles.
     *
     * @return immutable set of permissions
     */
    public Set<Permission> permissions() {
        Set<Permission> perms = EnumSet.noneOf(Permission.class);
        if (roles != null) {
            for (Role role : roles) {
                perms.addAll(permissionsFor(role));
            }
        }
        return Collections.unmodifiableSet(perms);
    }

    /**
     * Check whether the user holds a specific permission.
     *
     * @param permission permission to check
     * @return true if the user has the permission
     */
    public boolean hasPermission(Permission permission) {
        return permissions().contains(permission);
    }

    /**
     * Derive the permission set for a single role.
     *
     * @param role role to evaluate
     * @return set of permissions granted to the role
     */
    public static Set<Permission> permissionsFor(Role role) {
        Set<Permission> perms = EnumSet.noneOf(Permission.class);
        switch (role) {
            case ADMIN -> perms.addAll(EnumSet.allOf(Permission.class));
            case RESEARCHER -> perms.addAll(EnumSet.of(
                    Permission.PROJECT_READ, Permission.PROJECT_WRITE,
                    Permission.EXPERIMENT_READ, Permission.EXPERIMENT_WRITE,
                    Permission.RUN_READ, Permission.RUN_WRITE, Permission.RUN_CANCEL,
                    Permission.MODEL_READ, Permission.MODEL_WRITE,
                    Permission.PREDICTION_READ, Permission.PREDICTION_WRITE,
                    Permission.BACKTEST_READ, Permission.BACKTEST_WRITE,
                    Permission.AUDIT_READ, Permission.SYSTEM_READ));
            case VIEWER -> perms.addAll(EnumSet.of(
                    Permission.PROJECT_READ,
                    Permission.EXPERIMENT_READ,
                    Permission.RUN_READ,
                    Permission.MODEL_READ,
                    Permission.PREDICTION_READ,
                    Permission.BACKTEST_READ,
                    Permission.AUDIT_READ,
                    Permission.SYSTEM_READ));
        }
        return perms;
    }
}
