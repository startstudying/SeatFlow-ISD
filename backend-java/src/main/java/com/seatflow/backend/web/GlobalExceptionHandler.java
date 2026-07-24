package com.seatflow.backend.web;

import com.seatflow.backend.api.ApiResponse;
import com.seatflow.backend.api.ErrorPayload;
import com.seatflow.backend.api.TraceIds;
import com.seatflow.backend.service.ApiException;
import jakarta.validation.ConstraintViolationException;
import java.util.List;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
public class GlobalExceptionHandler {
    @ExceptionHandler(ApiException.class)
    public ResponseEntity<ApiResponse<ErrorPayload>> apiException(ApiException error) {
        ApiResponse<ErrorPayload> response = ApiResponse.error(
                error.code(),
                error.getMessage(),
                error.payload(),
                TraceIds.next()
        );
        return ResponseEntity.status(error.statusCode()).body(response);
    }

    @ExceptionHandler({ConstraintViolationException.class, MethodArgumentNotValidException.class})
    public ResponseEntity<ApiResponse<ErrorPayload>> validation(Exception error) {
        ErrorPayload payload = ErrorPayload.failClosed(
                "request validation failed",
                List.of("invalid_request")
        );
        return ResponseEntity.badRequest().body(
                ApiResponse.error("INVALID_REQUEST", error.getMessage(), payload, TraceIds.next())
        );
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ApiResponse<ErrorPayload>> unexpected(Exception error) {
        ErrorPayload payload = ErrorPayload.failClosed(
                "Java orchestration layer failed closed",
                List.of("java_orchestration_error")
        );
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(
                ApiResponse.error("JAVA_FAIL_CLOSED", error.getMessage(), payload, TraceIds.next())
        );
    }
}
