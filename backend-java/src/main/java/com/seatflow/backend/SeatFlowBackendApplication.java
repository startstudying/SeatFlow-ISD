package com.seatflow.backend;

import com.seatflow.backend.config.EncryptionProperties;
import com.seatflow.backend.config.GraphProjectionProperties;
import com.seatflow.backend.config.SeatFlowProperties;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.scheduling.annotation.EnableAsync;

@SpringBootApplication
@EnableAsync
@EnableConfigurationProperties({SeatFlowProperties.class, EncryptionProperties.class, GraphProjectionProperties.class})
public class SeatFlowBackendApplication {
    public static void main(String[] args) {
        SpringApplication.run(SeatFlowBackendApplication.class, args);
    }
}
