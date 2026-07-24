package com.seatflow.backend.common.web;

import static org.assertj.core.api.Assertions.assertThat;

import com.seatflow.backend.api.TraceIds;
import jakarta.servlet.FilterChain;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;

class RequestIdFilterTest {
    @Test
    void echoesSafeRequestIdAndExposesItAsTraceId() throws Exception {
        RequestIdFilter filter = new RequestIdFilter();
        MockHttpServletRequest request = new MockHttpServletRequest("GET", "/api/v1/health");
        MockHttpServletResponse response = new MockHttpServletResponse();
        request.addHeader(RequestContext.REQUEST_ID_HEADER, "req-123");

        filter.doFilter(request, response, (FilterChain) (servletRequest, servletResponse) ->
                assertThat(TraceIds.next()).isEqualTo("req-123"));

        assertThat(response.getHeader(RequestContext.REQUEST_ID_HEADER)).isEqualTo("req-123");
        assertThat(RequestContext.requestId()).isNull();
    }
}
