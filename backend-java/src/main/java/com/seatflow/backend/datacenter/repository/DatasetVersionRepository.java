package com.seatflow.backend.datacenter.repository;

import com.seatflow.backend.datacenter.entity.DatasetVersion;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

public interface DatasetVersionRepository extends JpaRepository<DatasetVersion, String> {
    Page<DatasetVersion> findByDatasetContainingIgnoreCase(String dataset, Pageable pageable);
}
