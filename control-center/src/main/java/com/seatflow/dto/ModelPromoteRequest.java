package com.seatflow.dto;

import com.seatflow.model.ModelStage;
import jakarta.validation.constraints.NotNull;

/**
 * Request body for promoting a model version to a stage.
 *
 * @param stage target stage (required)
 */
public record ModelPromoteRequest(
        @NotNull ModelStage stage) {
}
