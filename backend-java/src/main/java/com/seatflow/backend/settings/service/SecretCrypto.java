package com.seatflow.backend.settings.service;

public interface SecretCrypto {
    StoredSecret encrypt(String plainText);

    String mask(boolean present);

    record StoredSecret(String keyId, String ciphertext) {
    }
}
