package com.seatflow.backend.settings.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "sf_setting")
public class SettingEntry extends BaseVersionedEntity {
    @Id
    @Column(name = "setting_key", length = 128)
    private String key;

    @Column(nullable = false, length = 64)
    private String category;

    @Column(columnDefinition = "TEXT")
    private String valueJson;

    protected SettingEntry() {
    }

    public SettingEntry(String key, String category, String valueJson) {
        this.key = key;
        this.category = category;
        this.valueJson = valueJson;
    }

    public String getKey() {
        return key;
    }

    public String getCategory() {
        return category;
    }

    public String getValueJson() {
        return valueJson;
    }

    public void update(String category, String valueJson) {
        this.category = category;
        this.valueJson = valueJson;
    }
}
