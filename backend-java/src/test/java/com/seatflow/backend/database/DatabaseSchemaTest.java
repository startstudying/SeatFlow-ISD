package com.seatflow.backend.database;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import org.junit.jupiter.api.Test;

class DatabaseSchemaTest {
    @Test
    void mysqlSchemaContainsOnlyMetadataAndReferences() throws Exception {
        Path schemaPath = Path.of("..", "database", "mysql_schema.sql").normalize();
        Path migrationPath = Path.of("src", "main", "resources", "db", "migration",
                "V001__baseline_existing_metadata.sql").normalize();
        String schema = Files.readString(schemaPath, StandardCharsets.UTF_8);
        String migration = Files.readString(migrationPath, StandardCharsets.UTF_8);

        assertThat(schema).contains(
                "CREATE TABLE IF NOT EXISTS sf_experiment",
                "CREATE TABLE IF NOT EXISTS sf_experiment_run",
                "CREATE TABLE IF NOT EXISTS sf_prediction_reference",
                "CREATE TABLE IF NOT EXISTS sf_backtest_reference",
                "CREATE TABLE IF NOT EXISTS sf_audit_status",
                "CREATE TABLE IF NOT EXISTS sf_api_audit_log"
        );
        assertThat(migration).contains(
                "CREATE TABLE IF NOT EXISTS sf_experiment",
                "CREATE TABLE IF NOT EXISTS sf_api_audit_log"
        );
        assertThat(schema).doesNotContain("checkpoint_blob", "prediction_value_generated");
        assertThat(migration).doesNotContain("checkpoint_blob", "prediction_value_generated");
        assertThat(sha256(schema)).hasSize(64);
    }

    private static String sha256(String value) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        return HexFormat.of().formatHex(digest.digest(value.getBytes(StandardCharsets.UTF_8)));
    }
}
