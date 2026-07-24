package com.seatflow.backend.sync.repository;

import com.seatflow.backend.sync.entity.SyncTask;
import java.util.Optional;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

public interface SyncTaskRepository extends JpaRepository<SyncTask, String> {
    Optional<SyncTask> findByIdempotencyKey(String idempotencyKey);
    Page<SyncTask> findByStatus(String status, Pageable pageable);
}
