package com.seatflow.backend.settings.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.seatflow.backend.settings.dto.DataSourceDtos;
import com.seatflow.backend.settings.dto.ExperimentDtos;
import com.seatflow.backend.settings.dto.ModelDtos;
import com.seatflow.backend.settings.dto.RbacDtos;
import com.seatflow.backend.settings.dto.SettingDtos;
import com.seatflow.backend.settings.entity.DataSourceConfig;
import com.seatflow.backend.settings.entity.ExperimentConfig;
import com.seatflow.backend.settings.entity.ModelConfig;
import com.seatflow.backend.settings.entity.PermissionDefinition;
import com.seatflow.backend.settings.entity.RoleDefinition;
import com.seatflow.backend.settings.entity.SettingEntry;
import com.seatflow.backend.settings.entity.UserAccount;
import com.seatflow.backend.settings.repository.DataSourceConfigRepository;
import com.seatflow.backend.settings.repository.ExperimentConfigRepository;
import com.seatflow.backend.settings.repository.ModelConfigRepository;
import com.seatflow.backend.settings.repository.PermissionDefinitionRepository;
import com.seatflow.backend.settings.repository.RoleDefinitionRepository;
import com.seatflow.backend.settings.repository.SettingEntryRepository;
import com.seatflow.backend.settings.repository.UserAccountRepository;
import java.util.List;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;

@Service
public class SettingsManagementService {
    private final SettingEntryRepository settingRepository;
    private final DataSourceConfigRepository dataSourceRepository;
    private final ModelConfigRepository modelRepository;
    private final ExperimentConfigRepository experimentRepository;
    private final UserAccountRepository userRepository;
    private final RoleDefinitionRepository roleRepository;
    private final PermissionDefinitionRepository permissionRepository;
    private final IdempotencyService idempotencyService;
    private final AuditLogService auditLogService;
    private final SecretCrypto secretCrypto;
    private final ObjectMapper objectMapper;

    public SettingsManagementService(
            SettingEntryRepository settingRepository,
            DataSourceConfigRepository dataSourceRepository,
            ModelConfigRepository modelRepository,
            ExperimentConfigRepository experimentRepository,
            UserAccountRepository userRepository,
            RoleDefinitionRepository roleRepository,
            PermissionDefinitionRepository permissionRepository,
            IdempotencyService idempotencyService,
            AuditLogService auditLogService,
            SecretCrypto secretCrypto,
            ObjectMapper objectMapper
    ) {
        this.settingRepository = settingRepository;
        this.dataSourceRepository = dataSourceRepository;
        this.modelRepository = modelRepository;
        this.experimentRepository = experimentRepository;
        this.userRepository = userRepository;
        this.roleRepository = roleRepository;
        this.permissionRepository = permissionRepository;
        this.idempotencyService = idempotencyService;
        this.auditLogService = auditLogService;
        this.secretCrypto = secretCrypto;
        this.objectMapper = objectMapper;
    }

    @Transactional(readOnly = true)
    public List<SettingDtos.SettingResponse> settings(String category) {
        List<SettingEntry> entries = StringUtils.hasText(category)
                ? settingRepository.findByCategory(category)
                : settingRepository.findAll();
        return entries.stream().map(this::settingResponse).toList();
    }

    @Transactional
    public SettingDtos.SettingResponse upsertSetting(String key, SettingDtos.UpsertSettingRequest request, String idemKey) {
        if (!idempotencyService.claim(idemKey, "setting.upsert", key)) {
            return settingResponse(settingRepository.findById(key).orElseThrow(() -> SettingsErrors.notFound("setting", key)));
        }
        SettingEntry entry = settingRepository.findById(key)
                .orElseGet(() -> new SettingEntry(key, request.category(), "{}"));
        checkVersion(entry.getVersion(), request.expectedVersion());
        entry.update(request.category(), json(request.value()));
        SettingEntry saved = settingRepository.save(entry);
        auditLogService.record("UPSERT", "SETTING", key, idemKey, objectMapper.createObjectNode().put("category", request.category()));
        return settingResponse(saved);
    }

    @Transactional(readOnly = true)
    public List<DataSourceDtos.DataSourceResponse> dataSources() {
        return dataSourceRepository.findAll().stream().map(this::dataSourceResponse).toList();
    }

    @Transactional
    public DataSourceDtos.DataSourceResponse upsertDataSource(String id, DataSourceDtos.UpsertDataSourceRequest request, String idemKey) {
        if (!idempotencyService.claim(idemKey, "dataSource.upsert", id)) {
            return dataSourceResponse(dataSourceRepository.findById(id).orElseThrow(() -> SettingsErrors.notFound("dataSource", id)));
        }
        DataSourceConfig config = dataSourceRepository.findById(id).orElseGet(() -> new DataSourceConfig(id));
        checkVersion(config.getVersion(), request.expectedVersion());
        config.update(request.displayName(), request.provider(), request.endpoint(), json(request.metadata()));
        if (request.apiKey() != null) {
            SecretCrypto.StoredSecret secret = secretCrypto.encrypt(request.apiKey());
            config.storeSecret(secret.keyId(), secret.ciphertext());
        }
        DataSourceConfig saved = dataSourceRepository.save(config);
        auditLogService.record("UPSERT", "DATA_SOURCE", id, idemKey, objectMapper.createObjectNode().put("provider", request.provider()));
        return dataSourceResponse(saved);
    }

    @Transactional(readOnly = true)
    public List<ModelDtos.ModelResponse> models() {
        return modelRepository.findAll().stream().map(this::modelResponse).toList();
    }

    @Transactional
    public ModelDtos.ModelResponse upsertModel(String id, ModelDtos.UpsertModelRequest request, String idemKey) {
        if (!idempotencyService.claim(idemKey, "model.upsert", id)) {
            return modelResponse(modelRepository.findById(id).orElseThrow(() -> SettingsErrors.notFound("model", id)));
        }
        ModelConfig config = modelRepository.findById(id).orElseGet(() -> new ModelConfig(id));
        checkVersion(config.getVersion(), request.expectedVersion());
        config.update(request.displayName(), request.provider(), request.status(), json(request.parameters()));
        ModelConfig saved = modelRepository.save(config);
        auditLogService.record("UPSERT", "MODEL", id, idemKey, objectMapper.createObjectNode().put("status", request.status()));
        return modelResponse(saved);
    }

    @Transactional(readOnly = true)
    public List<ExperimentDtos.ExperimentResponse> experiments() {
        return experimentRepository.findAll().stream().map(this::experimentResponse).toList();
    }

    @Transactional
    public ExperimentDtos.ExperimentResponse upsertExperiment(String id, ExperimentDtos.UpsertExperimentRequest request, String idemKey) {
        if (!idempotencyService.claim(idemKey, "experiment.upsert", id)) {
            return experimentResponse(experimentRepository.findById(id).orElseThrow(() -> SettingsErrors.notFound("experiment", id)));
        }
        ExperimentConfig config = experimentRepository.findById(id).orElseGet(() -> new ExperimentConfig(id));
        checkVersion(config.getVersion(), request.expectedVersion());
        config.update(request.name(), request.status(), json(request.config()));
        ExperimentConfig saved = experimentRepository.save(config);
        auditLogService.record("UPSERT", "EXPERIMENT", id, idemKey, objectMapper.createObjectNode().put("status", request.status()));
        return experimentResponse(saved);
    }

    @Transactional(readOnly = true)
    public List<RbacDtos.UserResponse> users() {
        return userRepository.findAll().stream().map(this::userResponse).toList();
    }

    @Transactional
    public RbacDtos.UserResponse upsertUser(String id, RbacDtos.UpsertUserRequest request, String idemKey) {
        if (!idempotencyService.claim(idemKey, "user.upsert", id)) {
            return userResponse(userRepository.findById(id).orElseThrow(() -> SettingsErrors.notFound("user", id)));
        }
        UserAccount user = userRepository.findById(id).orElseGet(() -> new UserAccount(id));
        checkVersion(user.getVersion(), request.expectedVersion());
        user.update(request.displayName(), request.active(), request.roles());
        UserAccount saved = userRepository.save(user);
        auditLogService.record("UPSERT", "USER", id, idemKey, objectMapper.createObjectNode().put("active", request.active()));
        return userResponse(saved);
    }

    @Transactional(readOnly = true)
    public List<RbacDtos.RoleResponse> roles() {
        return roleRepository.findAll().stream().map(this::roleResponse).toList();
    }

    @Transactional
    public RbacDtos.RoleResponse upsertRole(String code, RbacDtos.UpsertRoleRequest request, String idemKey) {
        if (!idempotencyService.claim(idemKey, "role.upsert", code)) {
            return roleResponse(roleRepository.findById(code).orElseThrow(() -> SettingsErrors.notFound("role", code)));
        }
        RoleDefinition role = roleRepository.findById(code).orElseGet(() -> new RoleDefinition(code));
        checkVersion(role.getVersion(), request.expectedVersion());
        role.update(request.name(), request.permissions());
        RoleDefinition saved = roleRepository.save(role);
        auditLogService.record("UPSERT", "ROLE", code, idemKey, objectMapper.createObjectNode().put("name", request.name()));
        return roleResponse(saved);
    }

    @Transactional(readOnly = true)
    public List<RbacDtos.PermissionResponse> permissions() {
        return permissionRepository.findAll().stream().map(this::permissionResponse).toList();
    }

    @Transactional
    public RbacDtos.PermissionResponse upsertPermission(String code, RbacDtos.UpsertPermissionRequest request, String idemKey) {
        if (!idempotencyService.claim(idemKey, "permission.upsert", code)) {
            return permissionResponse(permissionRepository.findById(code).orElseThrow(() -> SettingsErrors.notFound("permission", code)));
        }
        PermissionDefinition permission = permissionRepository.findById(code).orElseGet(() -> new PermissionDefinition(code));
        checkVersion(permission.getVersion(), request.expectedVersion());
        permission.update(request.name(), request.resource());
        PermissionDefinition saved = permissionRepository.save(permission);
        auditLogService.record("UPSERT", "PERMISSION", code, idemKey, objectMapper.createObjectNode().put("resource", request.resource()));
        return permissionResponse(saved);
    }

    private void checkVersion(long actual, Long expected) {
        if (expected == null) {
            throw SettingsErrors.invalid("expectedVersion is required for write requests");
        }
        if (actual != expected) {
            throw SettingsErrors.versionConflict(expected, actual);
        }
    }

    private String json(JsonNode node) {
        return node == null ? "{}" : node.toString();
    }

    private JsonNode readJson(String value) {
        try {
            return objectMapper.readTree(value == null ? "{}" : value);
        } catch (Exception error) {
            return objectMapper.createObjectNode();
        }
    }

    private SettingDtos.SettingResponse settingResponse(SettingEntry entry) {
        return new SettingDtos.SettingResponse(entry.getKey(), entry.getCategory(), readJson(entry.getValueJson()), entry.getVersion());
    }

    private DataSourceDtos.DataSourceResponse dataSourceResponse(DataSourceConfig config) {
        return new DataSourceDtos.DataSourceResponse(
                config.getId(), config.getDisplayName(), config.getProvider(), config.getEndpoint(),
                readJson(config.getMetadataJson()), secretCrypto.mask(config.hasApiKey()), config.getApiKeyKeyId(), config.getVersion()
        );
    }

    private ModelDtos.ModelResponse modelResponse(ModelConfig config) {
        return new ModelDtos.ModelResponse(config.getId(), config.getDisplayName(), config.getProvider(),
                config.getStatus(), readJson(config.getParametersJson()), config.getVersion());
    }

    private ExperimentDtos.ExperimentResponse experimentResponse(ExperimentConfig config) {
        return new ExperimentDtos.ExperimentResponse(config.getId(), config.getName(), config.getStatus(),
                readJson(config.getConfigJson()), config.getVersion());
    }

    private RbacDtos.UserResponse userResponse(UserAccount user) {
        return new RbacDtos.UserResponse(user.getId(), user.getDisplayName(), user.isActive(), user.getRoles(), user.getVersion());
    }

    private RbacDtos.RoleResponse roleResponse(RoleDefinition role) {
        return new RbacDtos.RoleResponse(role.getCode(), role.getName(), role.getPermissions(), role.getVersion());
    }

    private RbacDtos.PermissionResponse permissionResponse(PermissionDefinition permission) {
        return new RbacDtos.PermissionResponse(permission.getCode(), permission.getName(), permission.getResource(), permission.getVersion());
    }
}
