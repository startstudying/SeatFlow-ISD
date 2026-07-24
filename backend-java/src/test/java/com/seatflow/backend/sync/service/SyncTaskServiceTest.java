package com.seatflow.backend.sync.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.seatflow.backend.common.audit.AuditService;
import com.seatflow.backend.datacenter.dto.DataCenterDtos;
import com.seatflow.backend.sync.entity.SyncTask;
import com.seatflow.backend.sync.repository.SyncTaskRepository;
import java.time.LocalDate;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class SyncTaskServiceTest {
    private final SyncTaskRepository repository = mock(SyncTaskRepository.class);
    private final SyncTaskRunner runner = mock(SyncTaskRunner.class);
    private final AuditService auditService = mock(AuditService.class);
    private final TaskProgressCache taskProgressCache = mock(TaskProgressCache.class);
    private final SyncTaskService service = new SyncTaskService(
            repository, runner, auditService, new ObjectMapper(), taskProgressCache
    );

    @Test
    void createsDurableTaskCachesProgressAndDispatchesAsyncRunner() {
        when(repository.findByIdempotencyKey("sync-001")).thenReturn(Optional.empty());
        when(repository.save(any(SyncTask.class))).thenAnswer(invocation -> invocation.getArgument(0));

        DataCenterDtos.SyncTaskResponse response = service.create(
                new DataCenterDtos.CreateSyncTaskRequest("akshare", "daily_bar", "snapshot_refresh", LocalDate.of(2026, 7, 19), 100),
                "sync-001"
        );

        assertThat(response.status()).isEqualTo("QUEUED");
        assertThat(response.progress()).isZero();
        verify(taskProgressCache).store(any(SyncTask.class));
        verify(runner).run(any(String.class), org.mockito.Mockito.eq(LocalDate.of(2026, 7, 19)));
    }

    @Test
    void idempotentCreateReturnsExistingTaskWithoutDispatchingAgain() {
        SyncTask existing = new SyncTask("existing-task", "sync-002", "tushare", "daily_bar", "snapshot_refresh");
        when(repository.findByIdempotencyKey("sync-002")).thenReturn(Optional.of(existing));

        DataCenterDtos.SyncTaskResponse response = service.create(
                new DataCenterDtos.CreateSyncTaskRequest("tushare", "daily_bar", "snapshot_refresh", LocalDate.now(), 100),
                "sync-002"
        );

        assertThat(response.id()).isEqualTo("existing-task");
        verify(repository, never()).save(any());
        verify(runner, never()).run(any(), any());
        verify(taskProgressCache, never()).store(any());
    }
}
