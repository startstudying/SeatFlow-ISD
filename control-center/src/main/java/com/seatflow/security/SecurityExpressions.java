package com.seatflow.security;

import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;

/**
 * Helper bean for expression-based and programmatic permission checks.
 */
@Component
public class SecurityExpressions {

    /**
     * @return the current authenticated principal, or null if anonymous
     */
    public UserPrincipal currentPrincipal() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null || !auth.isAuthenticated() || !(auth.getPrincipal() instanceof UserPrincipal)) {
            return null;
        }
        return (UserPrincipal) auth.getPrincipal();
    }

    /**
     * @param permission permission to check
     * @return true if the current user has the given permission
     */
    public boolean hasPermission(Permission permission) {
        UserPrincipal principal = currentPrincipal();
        return principal != null && principal.hasPermission(permission);
    }

    /**
     * @param role role to check
     * @return true if the current user has the given role
     */
    public boolean hasRole(Role role) {
        UserPrincipal principal = currentPrincipal();
        return principal != null && principal.roles() != null && principal.roles().contains(role);
    }
}
