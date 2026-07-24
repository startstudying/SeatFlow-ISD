package com.seatflow.backend.settings.repository;

import com.seatflow.backend.settings.entity.SettingEntry;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;

public interface SettingEntryRepository extends JpaRepository<SettingEntry, String> {
    List<SettingEntry> findByCategory(String category);
}
