package com.seatflow.backend.settings.repository;

import com.seatflow.backend.settings.entity.ExperimentConfig;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ExperimentConfigRepository extends JpaRepository<ExperimentConfig, String> {
}
