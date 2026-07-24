package com.seatflow.backend.datacenter.web;

import com.seatflow.backend.api.ApiResponse;
import com.seatflow.backend.api.TraceIds;
import com.seatflow.backend.datacenter.dto.DataCenterDtos;
import com.seatflow.backend.datacenter.service.DataCenterService;
import com.seatflow.backend.sync.service.SyncTaskService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/data-center")
public class DataCenterController {
    private final DataCenterService service;
    private final SyncTaskService syncTaskService;

    public DataCenterController(DataCenterService service, SyncTaskService syncTaskService) {
        this.service = service;
        this.syncTaskService = syncTaskService;
    }

    @GetMapping("/overview")
    public ApiResponse<DataCenterDtos.OverviewResponse> overview() {
        DataCenterDtos.OverviewResponse base = service.overview();
        DataCenterDtos.OverviewResponse response = new DataCenterDtos.OverviewResponse(
                base.assets(), base.qualityReports(), base.graphSnapshots(),
                syncTaskService.list(null, 0, 1).total()
        );
        return ApiResponse.ok(response, TraceIds.next());
    }
}
