package com.seatflow.backend.settings.service;

import com.seatflow.backend.settings.entity.IdempotencyRecord;
import com.seatflow.backend.settings.repository.IdempotencyRecordRepository;
import java.util.Optional;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;

@Service
public class IdempotencyService {
    private final IdempotencyRecordRepository repository;

    public IdempotencyService(IdempotencyRecordRepository repository) {
        this.repository = repository;
    }

    @Transactional
    public boolean claim(String key, String operation, String resourceId) {
        if (!StringUtils.hasText(key)) {
            throw SettingsErrors.invalid("Idempotency-Key header is required for write requests");
        }
        Optional<IdempotencyRecord> existing = repository.findById(key);
        if (existing.isPresent()) {
            IdempotencyRecord record = existing.get();
            if (record.getOperation().equals(operation) && record.getResourceId().equals(resourceId)) {
                return false;
            }
            throw SettingsErrors.idempotencyConflict();
        }
        repository.save(new IdempotencyRecord(key, operation, resourceId));
        return true;
    }
}
