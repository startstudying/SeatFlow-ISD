package com.seatflow.backend.settings.web;

import com.seatflow.backend.api.ApiResponse;
import com.seatflow.backend.api.TraceIds;
import com.seatflow.backend.settings.dto.RbacDtos;
import com.seatflow.backend.settings.service.SettingsManagementService;
import jakarta.validation.Valid;
import jakarta.validation.constraints.Pattern;
import java.util.List;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Validated
@RestController
@RequestMapping({"/api/v1/admin", "/api/v1"})
@PreAuthorize("hasRole('ADMIN')")
public class RbacAdminController {
    private static final String SAFE_CODE = "^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$";
    private final SettingsManagementService service;

    public RbacAdminController(SettingsManagementService service) {
        this.service = service;
    }

    @GetMapping("/users")
    public ApiResponse<List<RbacDtos.UserResponse>> users() {
        return ApiResponse.ok(service.users(), TraceIds.next());
    }

    @PutMapping("/users/{id}")
    public ApiResponse<RbacDtos.UserResponse> upsertUser(
            @PathVariable @Pattern(regexp = SAFE_CODE) String id,
            @RequestHeader(value = "Idempotency-Key", required = false) String idempotencyKey,
            @Valid @RequestBody RbacDtos.UpsertUserRequest request
    ) {
        return ApiResponse.ok(service.upsertUser(id, request, idempotencyKey), TraceIds.next());
    }

    @GetMapping("/roles")
    public ApiResponse<List<RbacDtos.RoleResponse>> roles() {
        return ApiResponse.ok(service.roles(), TraceIds.next());
    }

    @PutMapping("/roles/{code}")
    public ApiResponse<RbacDtos.RoleResponse> upsertRole(
            @PathVariable @Pattern(regexp = SAFE_CODE) String code,
            @RequestHeader(value = "Idempotency-Key", required = false) String idempotencyKey,
            @Valid @RequestBody RbacDtos.UpsertRoleRequest request
    ) {
        return ApiResponse.ok(service.upsertRole(code, request, idempotencyKey), TraceIds.next());
    }

    @GetMapping("/permissions")
    public ApiResponse<List<RbacDtos.PermissionResponse>> permissions() {
        return ApiResponse.ok(service.permissions(), TraceIds.next());
    }

    @PutMapping("/permissions/{code}")
    public ApiResponse<RbacDtos.PermissionResponse> upsertPermission(
            @PathVariable @Pattern(regexp = SAFE_CODE) String code,
            @RequestHeader(value = "Idempotency-Key", required = false) String idempotencyKey,
            @Valid @RequestBody RbacDtos.UpsertPermissionRequest request
    ) {
        return ApiResponse.ok(service.upsertPermission(code, request, idempotencyKey), TraceIds.next());
    }
}
