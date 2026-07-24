package com.seatflow.backend.api;

import com.seatflow.backend.common.web.RequestContext;
import java.util.UUID;

public final class TraceIds {
    private TraceIds() {
    }

    public static String next() {
        String requestId = RequestContext.requestId();
        return requestId == null || requestId.isBlank() ? UUID.randomUUID().toString() : requestId;
    }
}
