package com.seatflow.backend.config;

import org.springframework.boot.web.client.ClientHttpRequestFactories;
import org.springframework.boot.web.client.ClientHttpRequestFactorySettings;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestClient;

@Configuration
public class HttpClientConfig {
    @Bean
    public RestClient fastApiRestClient(SeatFlowProperties properties, RestClient.Builder builder) {
        ClientHttpRequestFactorySettings settings = ClientHttpRequestFactorySettings.DEFAULTS
                .withConnectTimeout(properties.fastapi().connectTimeout())
                .withReadTimeout(properties.fastapi().readTimeout());
        return builder
                .baseUrl(properties.fastapi().baseUrl())
                .requestFactory(ClientHttpRequestFactories.get(settings))
                .build();
    }
}
