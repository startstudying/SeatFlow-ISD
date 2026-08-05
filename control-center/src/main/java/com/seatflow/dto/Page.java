package com.seatflow.dto;

import java.util.List;

/**
 * Generic paginated response envelope with cursor-based pagination.
 *
 * @param items       page items
 * @param nextCursor  opaque cursor for the next page, or null if this is the last page
 * @param <T>         item type
 */
public record Page<T>(
        List<T> items,
        String nextCursor) {

    /**
     * Convenience factory for a page with no next cursor.
     *
     * @param items page items
     * @return a page with an empty next cursor
     */
    public static <T> Page<T> of(List<T> items) {
        return new Page<>(items, null);
    }
}
