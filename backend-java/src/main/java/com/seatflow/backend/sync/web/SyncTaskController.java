package com.seatflow.backend.sync.web;

import com.seatflow.backend.api.ApiResponse;
import com.seatflow.backend.api.TraceIds;
import com.seatflow.backend.datacenter.dto.DataCenterDtos;
import com.seatflow.backend.sync.service.SyncTaskService;
import jakarta.validation.Valid;
import jakarta.validation.constraints.Min;
import org.springframework.http.HttpStatus;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

@Validated
@RestController
@RequestMapping("/api/v1/sync/tasks")
public class SyncTaskController {
    private final SyncTaskService service;
    public SyncTaskController(SyncTaskService service) { this.service = service; }

    @GetMapping
    public ApiResponse<DataCenterDtos.PageResponse<DataCenterDtos.SyncTaskResponse>> list(
            @RequestParam(required = false) String status,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size
    ) {
        return ApiResponse.ok(service.list(status, page, size), TraceIds.next());
    }

    @PostMapping
    @ResponseStatus(HttpStatus.ACCEPTED)
    @PreAuthorize("hasAnyRole('ADMIN','OPERATOR')")
    public ApiResponse<DataCenterDtos.SyncTaskResponse> create(
            @RequestHeader("Idempotency-Key") String idempotencyKey,
            @Valid @RequestBody DataCenterDtos.CreateSyncTaskRequest request
    ) {
        return ApiResponse.ok(service.create(request, idempotencyKey), TraceIds.next());
    }

    @GetMapping("/{id}")
    public ApiResponse<DataCenterDtos.SyncTaskResponse> get(@PathVariable String id) {
        return ApiResponse.ok(service.get(id), TraceIds.next());
    }

    @PostMapping("/{id}/cancel")
    @PreAuthorize("hasAnyRole('ADMIN','OPERATOR')")
    public ApiResponse<DataCenterDtos.SyncTaskResponse> cancel(
            @PathVariable String id,
            @RequestHeader("If-Match") @Min(0) long expectedVersion
    ) {
        return ApiResponse.ok(service.cancel(id, expectedVersion), TraceIds.next());
    }
}
