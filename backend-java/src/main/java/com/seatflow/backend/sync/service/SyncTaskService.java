package com.seatflow.backend.sync.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.seatflow.backend.common.audit.AuditEvent;
import com.seatflow.backend.common.audit.AuditOutcome;
import com.seatflow.backend.common.audit.AuditService;
import com.seatflow.backend.datacenter.dto.DataCenterDtos;
import com.seatflow.backend.datacenter.mapper.DataCenterMapper;
import com.seatflow.backend.sync.entity.SyncTask;
import com.seatflow.backend.sync.repository.SyncTaskRepository;
import java.time.LocalDate;
import java.util.List;
import java.util.UUID;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;

@Service
public class SyncTaskService {
    private final SyncTaskRepository repository;
    private final SyncTaskRunner runner;
    private final AuditService auditService;
    private final ObjectMapper objectMapper;
    private final TaskProgressCache taskProgressCache;

    public SyncTaskService(
            SyncTaskRepository repository,
            SyncTaskRunner runner,
            AuditService auditService,
            ObjectMapper objectMapper,
            TaskProgressCache taskProgressCache
    ) {
        this.repository = repository;
        this.runner = runner;
        this.auditService = auditService;
        this.objectMapper = objectMapper;
        this.taskProgressCache = taskProgressCache;
    }

    @Transactional
    public DataCenterDtos.SyncTaskResponse create(DataCenterDtos.CreateSyncTaskRequest request, String idempotencyKey) {
        if (!StringUtils.hasText(idempotencyKey)) {
            throw new IllegalArgumentException("Idempotency-Key header is required");
        }
        SyncTask existing = repository.findByIdempotencyKey(idempotencyKey).orElse(null);
        if (existing != null) {
            return DataCenterMapper.task(existing, objectMapper);
        }
        if (!"snapshot_refresh".equals(request.mode()) && !"graph_refresh".equals(request.mode())) {
            throw new IllegalArgumentException("mode must be snapshot_refresh or graph_refresh");
        }
        SyncTask task = new SyncTask(UUID.randomUUID().toString(), idempotencyKey, request.provider(), request.dataset(), request.mode());
        SyncTask saved = repository.save(task);
        taskProgressCache.store(saved);
        auditService.record(new AuditEvent("SYNC_CREATED", "SYNC_TASK", saved.getTaskId(), AuditOutcome.SUCCESS, List.of()));
        runner.run(saved.getTaskId(), request.tradeDate() == null ? LocalDate.now() : request.tradeDate());
        return DataCenterMapper.task(saved, objectMapper);
    }

    @Transactional(readOnly = true)
    public DataCenterDtos.PageResponse<DataCenterDtos.SyncTaskResponse> list(String status, int page, int size) {
        Pageable pageable = PageRequest.of(Math.max(0, page), Math.max(1, Math.min(size, 100)), Sort.by(Sort.Direction.DESC, "createdAt"));
        Page<SyncTask> values = StringUtils.hasText(status) ? repository.findByStatus(status, pageable) : repository.findAll(pageable);
        return new DataCenterDtos.PageResponse<>(values.getContent().stream().map(value -> DataCenterMapper.task(value, objectMapper)).toList(),
                values.getNumber(), values.getSize(), values.getTotalElements());
    }

    @Transactional(readOnly = true)
    public DataCenterDtos.SyncTaskResponse get(String id) {
        return DataCenterMapper.task(repository.findById(id).orElseThrow(() -> new IllegalArgumentException("sync task not found")), objectMapper);
    }

    @Transactional
    public DataCenterDtos.SyncTaskResponse cancel(String id, long expectedVersion) {
        SyncTask task = repository.findById(id).orElseThrow(() -> new IllegalArgumentException("sync task not found"));
        if (task.getVersion() != expectedVersion) {
            throw new IllegalStateException("sync task version conflict");
        }
        task.cancel();
        SyncTask saved = repository.save(task);
        taskProgressCache.store(saved);
        auditService.record(new AuditEvent("SYNC_CANCELLED", "SYNC_TASK", id, AuditOutcome.SUCCESS, List.of()));
        return DataCenterMapper.task(saved, objectMapper);
    }
}
