package com.seatflow.backend.service;

import com.seatflow.backend.api.ErrorPayload;
import org.springframework.http.HttpStatusCode;

public class ApiException extends RuntimeException {
    private final HttpStatusCode statusCode;
    private final String code;
    private final ErrorPayload payload;

    public ApiException(HttpStatusCode statusCode, String code, String message, ErrorPayload payload) {
        super(message);
        this.statusCode = statusCode;
        this.code = code;
        this.payload = payload;
    }

    public HttpStatusCode statusCode() {
        return statusCode;
    }

    public String code() {
        return code;
    }

    public ErrorPayload payload() {
        return payload;
    }
}
