package com.seatflow.controller;

import com.seatflow.model.ErrorResponse;
import com.seatflow.service.ResearchArtifactBoundary;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import java.util.Map;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Backtest orchestration and replay endpoints.
 */
@RestController
@RequestMapping("/api/v1/backtests")
@Tag(name = "backtests", description = "Backtest orchestration and replay")
public class BacktestController {

    private final ResearchArtifactBoundary researchBoundary;

    /**
     * @param researchBoundary fail-closed research artifact boundary
     */
    public BacktestController(ResearchArtifactBoundary researchBoundary) {
        this.researchBoundary = researchBoundary;
    }

    /**
     * List backtests.
     *
     * @return blocked response until a verified artifact gateway is available
     */
    @GetMapping
    @Operation(operationId = "backtestsList", summary = "List backtests")
    public ResponseEntity<ErrorResponse> list() {
        return researchBoundary.unavailable("backtests.list");
    }

    /**
     * Create a backtest (long-running).
     *
     * @param request backtest creation request
     * @return blocked response instead of an untracked task handle
     */
    @PostMapping
    @Operation(operationId = "backtestsCreate", summary = "Create a backtest (long-running)")
    public ResponseEntity<ErrorResponse> create(@RequestBody Map<String, Object> request) {
        return researchBoundary.unavailable("backtests.create");
    }

    /**
     * Get backtest detail.
     *
     * @param backtestId backtest identifier
     * @return blocked response until a verified artifact is available
     */
    @GetMapping("/{backtestId}")
    @Operation(operationId = "backtestsGet", summary = "Get backtest detail")
    public ResponseEntity<ErrorResponse> get(@PathVariable String backtestId) {
        return researchBoundary.unavailable("backtests.get");
    }

    /**
     * Get backtest report.
     *
     * @param backtestId backtest identifier
     * @return blocked response until a verified report artifact is available
     */
    @GetMapping("/{backtestId}/report")
    @Operation(operationId = "backtestsGetReport", summary = "Get backtest report")
    public ResponseEntity<ErrorResponse> getReport(@PathVariable String backtestId) {
        return researchBoundary.unavailable("backtests.getReport");
    }
}
