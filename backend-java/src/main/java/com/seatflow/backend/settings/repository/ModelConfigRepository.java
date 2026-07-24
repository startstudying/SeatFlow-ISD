package com.seatflow.backend.settings.repository;

import com.seatflow.backend.settings.entity.ModelConfig;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ModelConfigRepository extends JpaRepository<ModelConfig, String> {
}
