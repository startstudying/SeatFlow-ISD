package com.seatflow.backend.web;

import com.fasterxml.jackson.databind.JsonNode;
import com.seatflow.backend.api.ApiResponse;
import com.seatflow.backend.api.TraceIds;
import com.seatflow.backend.service.OrchestrationService;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Validated
@RestController
@RequestMapping("/api/v1")
public class PredictionController {
    private static final String SAFE_ID = "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$";
    private final OrchestrationService service;

    public PredictionController(OrchestrationService service) {
        this.service = service;
    }

    @PostMapping("/providers/{provider}/models/{modelId}/predict")
    public ApiResponse<JsonNode> prediction(
            @PathVariable @Pattern(regexp = SAFE_ID) String provider,
            @PathVariable @Pattern(regexp = SAFE_ID) String modelId,
            @Valid @RequestBody PredictionRequest request
    ) {
        return ApiResponse.ok(service.prediction(provider, modelId, request), TraceIds.next());
    }

    @PostMapping("/predictions/query")
    public ApiResponse<JsonNode> predictionQuery(@Valid @RequestBody PredictionProxyQuery request) {
        return ApiResponse.ok(
                service.prediction(request.provider(), request.modelId(), request.toPredictionRequest()),
                TraceIds.next()
        );
    }

    public record PredictionProxyQuery(
            @NotBlank
            @Pattern(regexp = SAFE_ID) String provider,
            @NotBlank
            @Pattern(regexp = SAFE_ID) String modelId,
            @NotNull
            @Valid PredictionRequest prediction
    ) {
        PredictionRequest toPredictionRequest() {
            return prediction;
        }
    }
}
