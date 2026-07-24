package com.seatflow.backend.common.audit;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.seatflow.backend.config.SeatFlowProperties;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mockito;

class AuditServiceTest {
    @Test
    void persistsNormalizedAuditLog() {
        ApiAuditLogRepository repository = Mockito.mock(ApiAuditLogRepository.class);
        when(repository.save(any(ApiAuditLog.class))).thenAnswer(invocation -> invocation.getArgument(0));
        SeatFlowProperties properties = new SeatFlowProperties(
                "local_internal_research",
                "local-internal",
                new SeatFlowProperties.FastApi(null, null, null)
        );
        AuditService service = new AuditService(repository, new ObjectMapper(), properties);

        ApiAuditLog saved = service.record(new AuditEvent(
                "config.update",
                "config",
                "redis",
                AuditOutcome.FAIL_CLOSED,
                List.of("missing_approval")
        ));

        ArgumentCaptor<ApiAuditLog> captor = ArgumentCaptor.forClass(ApiAuditLog.class);
        verify(repository).save(captor.capture());
        assertThat(saved.outcome()).isEqualTo(AuditOutcome.FAIL_CLOSED);
        assertThat(captor.getValue().traceId()).isEqualTo("system");
    }
}
