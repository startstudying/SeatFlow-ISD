package com.seatflow.backend.settings.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.seatflow.backend.service.ApiException;
import com.seatflow.backend.settings.dto.DataSourceDtos;
import com.seatflow.backend.settings.entity.DataSourceConfig;
import com.seatflow.backend.settings.repository.DataSourceConfigRepository;
import com.seatflow.backend.settings.repository.ExperimentConfigRepository;
import com.seatflow.backend.settings.repository.ModelConfigRepository;
import com.seatflow.backend.settings.repository.PermissionDefinitionRepository;
import com.seatflow.backend.settings.repository.RoleDefinitionRepository;
import com.seatflow.backend.settings.repository.SettingEntryRepository;
import com.seatflow.backend.settings.repository.UserAccountRepository;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class SettingsManagementServiceTest {
    private final SettingEntryRepository settingRepository = mock(SettingEntryRepository.class);
    private final DataSourceConfigRepository dataSourceRepository = mock(DataSourceConfigRepository.class);
    private final ModelConfigRepository modelRepository = mock(ModelConfigRepository.class);
    private final ExperimentConfigRepository experimentRepository = mock(ExperimentConfigRepository.class);
    private final UserAccountRepository userRepository = mock(UserAccountRepository.class);
    private final RoleDefinitionRepository roleRepository = mock(RoleDefinitionRepository.class);
    private final PermissionDefinitionRepository permissionRepository = mock(PermissionDefinitionRepository.class);
    private final IdempotencyService idempotencyService = mock(IdempotencyService.class);
    private final AuditLogService auditLogService = mock(AuditLogService.class);
    private final SecretCrypto secretCrypto = mock(SecretCrypto.class);
    private final ObjectMapper objectMapper = new ObjectMapper();
    private final SettingsManagementService service = new SettingsManagementService(
            settingRepository,
            dataSourceRepository,
            modelRepository,
            experimentRepository,
            userRepository,
            roleRepository,
            permissionRepository,
            idempotencyService,
            auditLogService,
            secretCrypto,
            objectMapper
    );

    @Test
    void dataSourceApiKeyIsEncryptedAndOnlyMaskedInResponse() {
        when(idempotencyService.claim("idem-1", "dataSource.upsert", "akshare")).thenReturn(true);
        when(dataSourceRepository.findById("akshare")).thenReturn(Optional.empty());
        when(secretCrypto.encrypt("secret-api-key")).thenReturn(new SecretCrypto.StoredSecret("local", "ciphertext"));
        when(secretCrypto.mask(true)).thenReturn("********");
        when(dataSourceRepository.save(any(DataSourceConfig.class))).thenAnswer(invocation -> invocation.getArgument(0));

        DataSourceDtos.DataSourceResponse response = service.upsertDataSource(
                "akshare",
                new DataSourceDtos.UpsertDataSourceRequest(
                        "AkShare",
                        "akshare",
                        "https://example.invalid",
                        objectMapper.createObjectNode().put("timeout", 5),
                        "secret-api-key",
                        0L
                ),
                "idem-1"
        );

        assertThat(response.apiKeyMasked()).isEqualTo("********");
        assertThat(response.toString()).doesNotContain("secret-api-key", "ciphertext");
        verify(secretCrypto).encrypt("secret-api-key");
        verify(auditLogService).record(
                org.mockito.Mockito.eq("UPSERT"),
                org.mockito.Mockito.eq("DATA_SOURCE"),
                org.mockito.Mockito.eq("akshare"),
                org.mockito.Mockito.eq("idem-1"),
                org.mockito.Mockito.any()
        );
    }

    @Test
    void writeRejectsStaleVersionBeforeSaving() {
        DataSourceConfig existing = new DataSourceConfig("akshare");
        existing.update("AkShare", "akshare", "https://example.invalid", "{}");
        when(idempotencyService.claim("idem-2", "dataSource.upsert", "akshare")).thenReturn(true);
        when(dataSourceRepository.findById("akshare")).thenReturn(Optional.of(existing));

        assertThatThrownBy(() -> service.upsertDataSource(
                "akshare",
                new DataSourceDtos.UpsertDataSourceRequest(
                        "AkShare",
                        "akshare",
                        "https://example.invalid",
                        objectMapper.createObjectNode(),
                        null,
                        3L
                ),
                "idem-2"
        ))
                .isInstanceOf(ApiException.class)
                .hasMessageContaining("expected version 3");
    }
}
