package com.seatflow.backend.datacenter.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.Version;
import java.math.BigDecimal;
import java.time.LocalDate;

@Entity
@Table(name = "sf_data_quality_report")
public class DataQualityReport {
    @Id
    private String reportId;
    private String provider;
    private String dataset;
    private LocalDate tradeDate;
    private boolean passed;
    private BigDecimal completenessRate;
    private BigDecimal duplicateRate;
    private String riskFlagsJson;
    @Version
    private long version;

    protected DataQualityReport() {
    }

    public DataQualityReport(
            String reportId, String provider, String dataset, LocalDate tradeDate,
            boolean passed, BigDecimal completenessRate, BigDecimal duplicateRate, String riskFlagsJson
    ) {
        this.reportId = reportId;
        this.provider = provider;
        this.dataset = dataset;
        this.tradeDate = tradeDate;
        this.passed = passed;
        this.completenessRate = completenessRate;
        this.duplicateRate = duplicateRate;
        this.riskFlagsJson = riskFlagsJson;
    }

    public String getReportId() { return reportId; }
    public String getProvider() { return provider; }
    public String getDataset() { return dataset; }
    public LocalDate getTradeDate() { return tradeDate; }
    public boolean isPassed() { return passed; }
    public BigDecimal getCompletenessRate() { return completenessRate; }
    public BigDecimal getDuplicateRate() { return duplicateRate; }
    public String getRiskFlagsJson() { return riskFlagsJson; }
    public long getVersion() { return version; }
}
