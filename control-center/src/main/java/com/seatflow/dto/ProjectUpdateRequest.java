package com.seatflow.dto;

/**
 * Request body for partially updating a project. All fields are optional.
 *
 * @param name        new project name
 * @param description new description
 */
public record ProjectUpdateRequest(
        String name,
        String description) {
}
