package com.seatflow.backend.web;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;

public record PredictionRequest(
        @JsonProperty("trade_date")
        @NotBlank
        @Pattern(regexp = "\\d{4}-\\d{2}-\\d{2}", message = "trade_date must use YYYY-MM-DD")
        String tradeDate,

        @JsonProperty("target_stock_id")
        @NotBlank
        String targetStockId
) {
}
