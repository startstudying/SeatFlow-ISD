package com.seatflow.backend.common.web;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.UUID;
import org.slf4j.MDC;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

@Component
public class RequestIdFilter extends OncePerRequestFilter {
    private static final int MAX_REQUEST_ID_LENGTH = 128;
    private static final String SAFE_REQUEST_ID = "^[A-Za-z0-9_.:-]{1,128}$";

    @Override
    protected void doFilterInternal(
            HttpServletRequest request,
            HttpServletResponse response,
            FilterChain filterChain
    ) throws ServletException, IOException {
        String requestId = sanitize(request.getHeader(RequestContext.REQUEST_ID_HEADER));
        RequestContext.setRequestId(requestId);
        MDC.put("requestId", requestId);
        response.setHeader(RequestContext.REQUEST_ID_HEADER, requestId);
        try {
            filterChain.doFilter(request, response);
        } finally {
            MDC.remove("requestId");
            RequestContext.clear();
        }
    }

    private static String sanitize(String candidate) {
        if (candidate == null || candidate.length() > MAX_REQUEST_ID_LENGTH || !candidate.matches(SAFE_REQUEST_ID)) {
            return UUID.randomUUID().toString();
        }
        return candidate;
    }
}
