package com.seatflow.backend.settings.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import java.util.Set;

public final class RbacDtos {
    private RbacDtos() {
    }

    public record UpsertUserRequest(
            @NotBlank @Size(max = 128) String displayName,
            boolean active,
            Set<String> roles,
            Long expectedVersion
    ) implements VersionedWriteRequest {
    }

    public record UserResponse(String id, String displayName, boolean active, Set<String> roles, long version) {
    }

    public record UpsertRoleRequest(
            @NotBlank @Size(max = 128) String name,
            Set<String> permissions,
            Long expectedVersion
    ) implements VersionedWriteRequest {
    }

    public record RoleResponse(String code, String name, Set<String> permissions, long version) {
    }

    public record UpsertPermissionRequest(
            @NotBlank @Size(max = 128) String name,
            @NotBlank @Size(max = 64) String resource,
            Long expectedVersion
    ) implements VersionedWriteRequest {
    }

    public record PermissionResponse(String code, String name, String resource, long version) {
    }
}
