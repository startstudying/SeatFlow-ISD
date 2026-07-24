package com.seatflow.backend.monitoring.repository;

import com.seatflow.backend.monitoring.entity.ServiceMonitorSnapshot;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ServiceMonitorSnapshotRepository extends JpaRepository<ServiceMonitorSnapshot, String> {
}
