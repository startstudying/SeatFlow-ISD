package com.seatflow.backend.sync.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.seatflow.backend.common.audit.AuditEvent;
import com.seatflow.backend.common.audit.AuditOutcome;
import com.seatflow.backend.common.audit.AuditService;
import com.seatflow.backend.datacenter.service.DataCenterService;
import com.seatflow.backend.service.FastApiClient;
import com.seatflow.backend.sync.entity.SyncTask;
import com.seatflow.backend.sync.repository.SyncTaskRepository;
import java.time.LocalDate;
import java.util.List;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class SyncTaskRunner {
    private final SyncTaskRepository repository;
    private final FastApiClient fastApiClient;
    private final DataCenterService dataCenterService;
    private final AuditService auditService;
    private final TaskProgressCache taskProgressCache;

    public SyncTaskRunner(
            SyncTaskRepository repository,
            FastApiClient fastApiClient,
            DataCenterService dataCenterService,
            AuditService auditService,
            TaskProgressCache taskProgressCache
    ) {
        this.repository = repository;
        this.fastApiClient = fastApiClient;
        this.dataCenterService = dataCenterService;
        this.auditService = auditService;
        this.taskProgressCache = taskProgressCache;
    }

    @Async
    @Transactional
    public void run(String taskId, LocalDate tradeDate) {
        SyncTask task = repository.findById(taskId).orElse(null);
        if (task == null || task.isTerminal()) {
            return;
        }
        task.running("正在调用内部数据服务");
        repository.save(task);
        taskProgressCache.store(task);
        try {
            if ("graph_refresh".equals(task.getMode())) {
                JsonNode graph = fastApiClient.graphSnapshot(task.getProvider(), tradeDate);
                dataCenterService.persistGraph(task.getProvider(), tradeDate, graph);
            } else {
                JsonNode summary = fastApiClient.snapshotSummary(task.getProvider());
                dataCenterService.persistSummary(task.getProvider(), summary);
            }
            task.succeeded("内部数据服务响应已持久化为业务元数据");
            repository.save(task);
            taskProgressCache.store(task);
            auditService.record(new AuditEvent("SYNC_SUCCEEDED", "SYNC_TASK", taskId, AuditOutcome.SUCCESS, List.of()));
        } catch (RuntimeException error) {
            task.failed("内部数据服务未完成任务");
            repository.save(task);
            taskProgressCache.store(task);
            auditService.record(new AuditEvent(
                    "SYNC_FAILED", "SYNC_TASK", taskId, AuditOutcome.FAILURE, List.of("upstream_task_failed")
            ));
        }
    }
}
