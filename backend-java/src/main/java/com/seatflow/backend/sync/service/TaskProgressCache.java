package com.seatflow.backend.sync.service;

import com.seatflow.backend.sync.entity.SyncTask;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

@Service
public class TaskProgressCache {
    private static final Duration TTL = Duration.ofHours(24);
    private final RedisTemplate<String, Object> redisTemplate;

    public TaskProgressCache(RedisTemplate<String, Object> redisTemplate) {
        this.redisTemplate = redisTemplate;
    }

    /** Redis accelerates progress reads; MySQL remains authoritative. */
    public void store(SyncTask task) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("taskId", task.getTaskId());
        payload.put("status", task.getStatus());
        payload.put("progress", task.getProgress());
        payload.put("message", task.getMessage());
        payload.put("updatedAt", task.getUpdatedAt() == null ? null : task.getUpdatedAt().toString());
        try {
            redisTemplate.opsForValue().set("seatflow:sync-task:" + task.getTaskId(), payload, TTL);
        } catch (RuntimeException ignored) {
            // Cache failures must not alter a durable task result.
        }
    }
}
