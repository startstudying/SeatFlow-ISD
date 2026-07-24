package com.seatflow.backend.common.crypto;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.seatflow.backend.config.EncryptionProperties;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import org.junit.jupiter.api.Test;

class AesGcmConfigEncryptionServiceTest {
    private static final String KEY = Base64.getEncoder()
            .encodeToString("12345678901234567890123456789012".getBytes(StandardCharsets.UTF_8));

    @Test
    void encryptsAndDecryptsConfigValueWithEnvelope() {
        AesGcmConfigEncryptionService service = new AesGcmConfigEncryptionService(
                new EncryptionProperties(KEY, "test")
        );

        String encrypted = service.encrypt("secret-token");

        assertThat(encrypted).startsWith("seatflow:v1:");
        assertThat(encrypted).doesNotContain("secret-token");
        assertThat(service.decrypt(encrypted)).isEqualTo("secret-token");
    }

    @Test
    void rejectsMissingKeyFailClosed() {
        assertThatThrownBy(() -> new AesGcmConfigEncryptionService(new EncryptionProperties("", "test")))
                .isInstanceOf(ConfigEncryptionException.class)
                .hasMessageContaining("required");
    }
}
