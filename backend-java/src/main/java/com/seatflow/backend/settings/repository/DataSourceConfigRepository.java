package com.seatflow.backend.settings.repository;

import com.seatflow.backend.settings.entity.DataSourceConfig;
import org.springframework.data.jpa.repository.JpaRepository;

public interface DataSourceConfigRepository extends JpaRepository<DataSourceConfig, String> {
}
