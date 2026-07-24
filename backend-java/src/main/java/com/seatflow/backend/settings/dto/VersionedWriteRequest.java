package com.seatflow.backend.settings.dto;

import jakarta.validation.constraints.NotNull;

public interface VersionedWriteRequest {
    @NotNull
    Long expectedVersion();
}
