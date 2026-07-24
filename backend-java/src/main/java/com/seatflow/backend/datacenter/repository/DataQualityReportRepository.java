package com.seatflow.backend.datacenter.repository;

import com.seatflow.backend.datacenter.entity.DataQualityReport;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

public interface DataQualityReportRepository extends JpaRepository<DataQualityReport, String> {
    Page<DataQualityReport> findByDatasetContainingIgnoreCase(String dataset, Pageable pageable);
}
