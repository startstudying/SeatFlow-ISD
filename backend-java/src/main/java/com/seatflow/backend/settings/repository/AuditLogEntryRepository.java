package com.seatflow.backend.settings.repository;

import com.seatflow.backend.settings.entity.AuditLogEntry;
import org.springframework.data.jpa.repository.JpaRepository;

public interface AuditLogEntryRepository extends JpaRepository<AuditLogEntry, Long> {
}
