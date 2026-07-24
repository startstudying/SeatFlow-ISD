package com.seatflow.backend.settings.database;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class SettingsMigrationTest {
    @Test
    void v003ContainsSettingsRbacAuditAndMonitoringTables() throws Exception {
        String migration = Files.readString(
                Path.of("src", "main", "resources", "db", "migration", "V003__settings_rbac_audit.sql"),
                StandardCharsets.UTF_8
        );

        assertThat(migration).contains(
                "CREATE TABLE IF NOT EXISTS sf_setting",
                "CREATE TABLE IF NOT EXISTS sf_data_source_config",
                "encrypted_api_key TEXT NULL",
                "CREATE TABLE IF NOT EXISTS sf_model_config",
                "CREATE TABLE IF NOT EXISTS sf_experiment_config",
                "CREATE TABLE IF NOT EXISTS sf_user_account",
                "CREATE TABLE IF NOT EXISTS sf_role",
                "CREATE TABLE IF NOT EXISTS sf_permission",
                "CREATE TABLE IF NOT EXISTS sf_admin_audit_log",
                "CREATE TABLE IF NOT EXISTS sf_idempotency_record",
                "CREATE TABLE IF NOT EXISTS sf_service_monitor"
        );
        assertThat(migration).doesNotContain("plain_api_key", "api_key VARCHAR");
    }
}
