package com.seatflow.backend.settings.repository;

import com.seatflow.backend.settings.entity.PermissionDefinition;
import org.springframework.data.jpa.repository.JpaRepository;

public interface PermissionDefinitionRepository extends JpaRepository<PermissionDefinition, String> {
}
