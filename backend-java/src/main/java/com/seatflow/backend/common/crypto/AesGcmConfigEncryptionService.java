package com.seatflow.backend.common.crypto;

import com.seatflow.backend.config.EncryptionProperties;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.security.SecureRandom;
import java.util.Base64;
import javax.crypto.Cipher;
import javax.crypto.spec.GCMParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

@Service
@ConditionalOnProperty(name = "seatflow.encryption.enabled", havingValue = "true")
public class AesGcmConfigEncryptionService implements ConfigEncryptionService {
    private static final String PREFIX = "seatflow:v1:";
    private static final String ALGORITHM = "AES/GCM/NoPadding";
    private static final int IV_BYTES = 12;
    private static final int TAG_BITS = 128;
    private static final int AES_256_BYTES = 32;

    private final SecretKeySpec key;
    private final SecureRandom secureRandom = new SecureRandom();

    public AesGcmConfigEncryptionService(EncryptionProperties properties) {
        if (!StringUtils.hasText(properties.key())) {
            throw new ConfigEncryptionException("seatflow.encryption.key is required");
        }
        byte[] keyBytes = Base64.getDecoder().decode(properties.key());
        if (keyBytes.length != AES_256_BYTES) {
            throw new ConfigEncryptionException("seatflow.encryption.key must be a base64 encoded 256-bit key");
        }
        this.key = new SecretKeySpec(keyBytes, "AES");
    }

    @Override
    public String encrypt(String plaintext) {
        if (plaintext == null) {
            throw new ConfigEncryptionException("plaintext must not be null");
        }
        try {
            byte[] iv = new byte[IV_BYTES];
            secureRandom.nextBytes(iv);
            Cipher cipher = Cipher.getInstance(ALGORITHM);
            cipher.init(Cipher.ENCRYPT_MODE, key, new GCMParameterSpec(TAG_BITS, iv));
            byte[] ciphertext = cipher.doFinal(plaintext.getBytes(StandardCharsets.UTF_8));
            ByteBuffer payload = ByteBuffer.allocate(iv.length + ciphertext.length);
            payload.put(iv);
            payload.put(ciphertext);
            return PREFIX + Base64.getEncoder().encodeToString(payload.array());
        } catch (GeneralSecurityException error) {
            throw new ConfigEncryptionException("config encryption failed", error);
        }
    }

    @Override
    public String decrypt(String ciphertext) {
        if (!StringUtils.hasText(ciphertext) || !ciphertext.startsWith(PREFIX)) {
            throw new ConfigEncryptionException("ciphertext must use seatflow:v1 envelope");
        }
        try {
            byte[] payload = Base64.getDecoder().decode(ciphertext.substring(PREFIX.length()));
            if (payload.length <= IV_BYTES) {
                throw new ConfigEncryptionException("ciphertext payload is too short");
            }
            ByteBuffer buffer = ByteBuffer.wrap(payload);
            byte[] iv = new byte[IV_BYTES];
            buffer.get(iv);
            byte[] encrypted = new byte[buffer.remaining()];
            buffer.get(encrypted);
            Cipher cipher = Cipher.getInstance(ALGORITHM);
            cipher.init(Cipher.DECRYPT_MODE, key, new GCMParameterSpec(TAG_BITS, iv));
            return new String(cipher.doFinal(encrypted), StandardCharsets.UTF_8);
        } catch (GeneralSecurityException | IllegalArgumentException error) {
            throw new ConfigEncryptionException("config decryption failed", error);
        }
    }
}
