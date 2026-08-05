package com.seatflow;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * SeatFlow V3 Control Center — task governance, approvals, and audit.
 *
 * <p>This application is the authoritative source for task state, model
 * promotion decisions, quality gate results and audit records. It must never
 * duplicate financial research semantics that belong in the Research Engine.</p>
 */
@SpringBootApplication
public class SeatFlowApplication {

    public static void main(String[] args) {
        SpringApplication.run(SeatFlowApplication.class, args);
    }
}
