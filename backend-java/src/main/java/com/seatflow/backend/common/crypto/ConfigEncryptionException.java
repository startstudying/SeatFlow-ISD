package com.seatflow.backend.common.crypto;

public class ConfigEncryptionException extends RuntimeException {
    public ConfigEncryptionException(String message) {
        super(message);
    }

    public ConfigEncryptionException(String message, Throwable cause) {
        super(message, cause);
    }
}
