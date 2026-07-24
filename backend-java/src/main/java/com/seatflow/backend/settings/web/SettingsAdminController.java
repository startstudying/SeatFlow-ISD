package com.seatflow.backend.settings.web;

import com.seatflow.backend.api.ApiResponse;
import com.seatflow.backend.api.TraceIds;
import com.seatflow.backend.settings.dto.DataSourceDtos;
import com.seatflow.backend.settings.dto.ExperimentDtos;
import com.seatflow.backend.settings.dto.ModelDtos;
import com.seatflow.backend.settings.dto.SettingDtos;
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
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@Validated
@RestController
@RequestMapping({"/api/v1/admin", "/api/v1"})
public class SettingsAdminController {
    private static final String SAFE_ID = "^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$";
    private final SettingsManagementService service;

    public SettingsAdminController(SettingsManagementService service) {
        this.service = service;
    }

    @GetMapping("/settings")
    @PreAuthorize("hasAnyRole('ADMIN','AUDITOR')")
    public ApiResponse<List<SettingDtos.SettingResponse>> settings(@RequestParam(required = false) String category) {
        return ApiResponse.ok(service.settings(category), TraceIds.next());
    }

    @PutMapping("/settings/{key}")
    @PreAuthorize("hasRole('ADMIN')")
    public ApiResponse<SettingDtos.SettingResponse> upsertSetting(
            @PathVariable @Pattern(regexp = SAFE_ID) String key,
            @RequestHeader(value = "Idempotency-Key", required = false) String idempotencyKey,
            @Valid @RequestBody SettingDtos.UpsertSettingRequest request
    ) {
        return ApiResponse.ok(service.upsertSetting(key, request, idempotencyKey), TraceIds.next());
    }

    @GetMapping("/data-sources")
    @PreAuthorize("hasAnyRole('ADMIN','AUDITOR')")
    public ApiResponse<List<DataSourceDtos.DataSourceResponse>> dataSources() {
        return ApiResponse.ok(service.dataSources(), TraceIds.next());
    }

    @PutMapping("/data-sources/{id}")
    @PreAuthorize("hasRole('ADMIN')")
    public ApiResponse<DataSourceDtos.DataSourceResponse> upsertDataSource(
            @PathVariable @Pattern(regexp = SAFE_ID) String id,
            @RequestHeader(value = "Idempotency-Key", required = false) String idempotencyKey,
            @Valid @RequestBody DataSourceDtos.UpsertDataSourceRequest request
    ) {
        return ApiResponse.ok(service.upsertDataSource(id, request, idempotencyKey), TraceIds.next());
    }

    @GetMapping("/models")
    @PreAuthorize("hasAnyRole('ADMIN','AUDITOR')")
    public ApiResponse<List<ModelDtos.ModelResponse>> models() {
        return ApiResponse.ok(service.models(), TraceIds.next());
    }

    @PutMapping("/models/{id}")
    @PreAuthorize("hasRole('ADMIN')")
    public ApiResponse<ModelDtos.ModelResponse> upsertModel(
            @PathVariable @Pattern(regexp = SAFE_ID) String id,
            @RequestHeader(value = "Idempotency-Key", required = false) String idempotencyKey,
            @Valid @RequestBody ModelDtos.UpsertModelRequest request
    ) {
        return ApiResponse.ok(service.upsertModel(id, request, idempotencyKey), TraceIds.next());
    }

    @GetMapping("/experiments")
    @PreAuthorize("hasAnyRole('ADMIN','AUDITOR')")
    public ApiResponse<List<ExperimentDtos.ExperimentResponse>> experiments() {
        return ApiResponse.ok(service.experiments(), TraceIds.next());
    }

    @PutMapping("/experiments/{id}")
    @PreAuthorize("hasRole('ADMIN')")
    public ApiResponse<ExperimentDtos.ExperimentResponse> upsertExperiment(
            @PathVariable @Pattern(regexp = SAFE_ID) String id,
            @RequestHeader(value = "Idempotency-Key", required = false) String idempotencyKey,
            @Valid @RequestBody ExperimentDtos.UpsertExperimentRequest request
    ) {
        return ApiResponse.ok(service.upsertExperiment(id, request, idempotencyKey), TraceIds.next());
    }
}
