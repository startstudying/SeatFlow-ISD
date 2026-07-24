package com.seatflow.backend.config;

import io.swagger.v3.oas.models.Components;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.security.SecurityRequirement;
import io.swagger.v3.oas.models.security.SecurityScheme;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class OpenApiConfig {
    @Bean
    public OpenAPI seatFlowOpenApi() {
        return new OpenAPI()
                .info(new Info()
                        .title("SeatFlow 业务 API")
                        .version("v1")
                        .description("数据中心与系统设置统一业务入口。内部 Python 服务不对浏览器暴露。"))
                .components(new Components().addSecuritySchemes("apiKey",
                        new SecurityScheme().type(SecurityScheme.Type.APIKEY).in(SecurityScheme.In.HEADER)
                                .name("X-SeatFlow-Api-Key")))
                .addSecurityItem(new SecurityRequirement().addList("apiKey"));
    }
}
