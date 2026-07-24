package com.seatflow.backend.settings.service;

import com.seatflow.backend.config.EncryptionProperties;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.util.Arrays;
import java.util.Base64;
import javax.crypto.Cipher;
import javax.crypto.spec.GCMParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;

@Component
public class AesGcmSecretCrypto implements SecretCrypto {
    private static final int IV_LENGTH = 12;
    private static final int TAG_LENGTH_BITS = 128;
    private final EncryptionProperties properties;
    private final SecureRandom secureRandom = new SecureRandom();

    public AesGcmSecretCrypto(EncryptionProperties properties) {
        this.properties = properties;
    }

    @Override
    public StoredSecret encrypt(String plainText) {
        if (!StringUtils.hasText(plainText)) {
            throw SettingsErrors.invalid("apiKey must not be blank when supplied");
        }
        try {
            byte[] iv = new byte[IV_LENGTH];
            secureRandom.nextBytes(iv);
            Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
            cipher.init(Cipher.ENCRYPT_MODE, new SecretKeySpec(keyBytes(), "AES"), new GCMParameterSpec(TAG_LENGTH_BITS, iv));
            byte[] encrypted = cipher.doFinal(plainText.getBytes(StandardCharsets.UTF_8));
            byte[] combined = new byte[iv.length + encrypted.length];
            System.arraycopy(iv, 0, combined, 0, iv.length);
            System.arraycopy(encrypted, 0, combined, iv.length, encrypted.length);
            return new StoredSecret(properties.keyId(), Base64.getEncoder().encodeToString(combined));
        } catch (Exception error) {
            throw SettingsErrors.invalid("failed to encrypt data source api key");
        }
    }

    @Override
    public String mask(boolean present) {
        return present ? "********" : null;
    }

    private byte[] keyBytes() throws Exception {
        String key = properties.key();
        if (!StringUtils.hasText(key)) {
            key = "seatflow-local-development-key";
        }
        return Arrays.copyOf(MessageDigest.getInstance("SHA-256").digest(key.getBytes(StandardCharsets.UTF_8)), 32);
    }
}
