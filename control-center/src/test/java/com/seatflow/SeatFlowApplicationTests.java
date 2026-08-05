package com.seatflow;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;

/**
 * Smoke test that verifies the Spring application context loads successfully.
 */
@SpringBootTest
@ActiveProfiles("test")
class SeatFlowApplicationTests {

    @Test
    void contextLoads() {
        // Verifies the application context starts without errors.
    }
}
