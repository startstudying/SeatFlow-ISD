package com.seatflow.backend.common.crypto;

public interface ConfigEncryptionService {
    String encrypt(String plaintext);

    String decrypt(String ciphertext);
}
