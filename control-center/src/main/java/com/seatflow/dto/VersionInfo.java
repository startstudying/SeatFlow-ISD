package com.seatflow.dto;

/**
 * Service version information.
 *
 * @param version semantic version string
 * @param build   build identifier
 * @param commit  source control commit hash
 */
public record VersionInfo(
        String version,
        String build,
        String commit) {
}
