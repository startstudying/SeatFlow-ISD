package com.seatflow.config;

import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Contact;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.info.License;
import io.swagger.v3.oas.models.security.SecurityRequirement;
import io.swagger.v3.oas.models.security.SecurityScheme;
import io.swagger.v3.oas.models.Components;
import io.swagger.v3.oas.models.servers.Server;
import java.util.List;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * OpenAPI / Swagger configuration aligned with the SeatFlow Control Center API contract.
 */
@Configuration
public class OpenApiConfig {

    /** Name of the bearer JWT security scheme. */
    public static final String BEARER_AUTH = "bearerAuth";

    /**
     * @return OpenAPI descriptor for the Control Center API
     */
    @Bean
    public OpenAPI seatFlowOpenApi() {
        return new OpenAPI()
                .servers(List.of(new Server().url("/api/v1").description("SeatFlow Control Center API")))
                .info(new Info()
                        .title("SeatFlow Control Center API")
                        .version("3.0.0-dev")
                        .description("Control Center API for SeatFlow V3. Provides a single point of coordination "
                                + "across projects, datasets, experiments, runs, models, predictions, backtests, "
                                + "audit, and system endpoints.")
                        .contact(new Contact().name("SeatFlow API Team"))
                        .license(new License().name("Proprietary")))
                .addSecurityItem(new SecurityRequirement().addList(BEARER_AUTH))
                .components(new Components()
                        .addSecuritySchemes(BEARER_AUTH, new SecurityScheme()
                                .type(SecurityScheme.Type.HTTP)
                                .scheme("bearer")
                                .bearerFormat("JWT")));
    }
}
