package com.seatflow.backend.settings.repository;

import com.seatflow.backend.settings.entity.RoleDefinition;
import org.springframework.data.jpa.repository.JpaRepository;

public interface RoleDefinitionRepository extends JpaRepository<RoleDefinition, String> {
}
