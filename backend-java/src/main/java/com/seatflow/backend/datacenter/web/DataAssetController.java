package com.seatflow.backend.datacenter.web;

import com.seatflow.backend.api.ApiResponse;
import com.seatflow.backend.api.TraceIds;
import com.seatflow.backend.datacenter.dto.DataCenterDtos;
import com.seatflow.backend.datacenter.service.DataCenterService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/data-assets")
public class DataAssetController {
    private final DataCenterService service;

    public DataAssetController(DataCenterService service) { this.service = service; }

    @GetMapping
    public ApiResponse<DataCenterDtos.PageResponse<DataCenterDtos.AssetResponse>> list(
            @RequestParam(required = false) String keyword,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size,
            @RequestParam(required = false) String sort
    ) {
        return ApiResponse.ok(service.assets(keyword, page, size, sort), TraceIds.next());
    }
}
