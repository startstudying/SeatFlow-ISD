package com.seatflow.backend.common.web;

public final class RequestContext {
    public static final String REQUEST_ID_HEADER = "X-Request-Id";
    private static final ThreadLocal<String> REQUEST_ID = new ThreadLocal<>();

    private RequestContext() {
    }

    public static String requestId() {
        return REQUEST_ID.get();
    }

    static void setRequestId(String requestId) {
        REQUEST_ID.set(requestId);
    }

    static void clear() {
        REQUEST_ID.remove();
    }
}
