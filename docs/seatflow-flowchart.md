# SeatFlow 研究与发布流程图

这张图展示 SeatFlow 从市场数据采集、时间一致性处理、动态图构建、DT-SeatHGT 训练，到分类评估、交易回测和发布闸门的完整流程。

```mermaid
flowchart TD
    A["Tushare Pro（主）/ AkShare（回退）<br/>日线、龙虎榜、席位事件、停牌证据"] --> B{"数据质量、权限与时间可用性通过？"}
    B -- "否" --> B0["记录错误/告警<br/>停止该批次（fail-closed）"]
    B -- "是" --> C["raw / processed verified snapshots<br/>manifest + SHA-256"]
    C --> D["as-of 特征构建<br/>OHLCV + 席位流 + 行业/概念<br/>仅使用 trade_date ≤ as_of_date"]
    D --> E["DynamicGraphBuilder<br/>关系边 + 时间衰减 + cutoff"]
    E --> F["冻结研究窗口与标签<br/>1/3/5 日标签、purge、embargo、future-blind"]
    F --> G["PyG HeteroData 张量化<br/>冻结 graph schema + metadata 审计"]
    G --> H["DT-SeatHGT 多轮训练<br/>方向头 + 收益头<br/>梯度裁剪 / 非有限值检查"]
    H --> I["生成预测<br/>方向概率 与 交易选择分数分离"]
    I --> J["分类评估<br/>AUC、Rank IC、多 horizon"]
    I --> K["交易回测<br/>阈值弃权 / Top-K<br/>as-of 可交易性 + 佣金税费滑点"]
    J --> L{"模型、回测与 release contract 闸门全部通过？"}
    K --> L
    L -- "是" --> M["允许发布 / API<br/>model_valid=true<br/>backtest_valid=true<br/>competition_ready=true"]
    L -- "否" --> N["拒绝晋级<br/>model_valid=false<br/>backtest_valid=false<br/>competition_ready=false<br/>保留负结果与阻塞原因"]

    classDef source fill:#e8f1fb,stroke:#2c6eaa,color:#102a43;
    classDef gate fill:#fff4d6,stroke:#b7791f,color:#5f370e;
    classDef process fill:#eef6ee,stroke:#3f7d4a,color:#19351f;
    classDef model fill:#f1ecfb,stroke:#7452a8,color:#30204b;
    classDef fail fill:#fde8e8,stroke:#b83232,color:#5c1a1a;
    classDef release fill:#e6fffa,stroke:#23827a,color:#164e63;

    class A source;
    class B,L gate;
    class B0,N fail;
    class C,D,E,F,G process;
    class H,I,J,K model;
    class M release;
```

## 关键设计决策

1. **先审计、后建模**：provider 权限、字段质量、可用时间和快照哈希不通过时，流程在数据入口停止，不生成可被误读的模型结论。
2. **所有研究对象按 `as_of_date` 冻结**：特征、图边、标签和可交易性检查都遵守时间截点，并用 purge、embargo、future-blind 降低时间泄漏风险。
3. **分类概率与交易分数分离**：AUC/Rank IC 衡量预测能力，阈值弃权、Top-K、可交易性和成本回测衡量能否形成交易选择，避免一个分数承担两种语义。
4. **发布采用 fail-closed**：只有模型、回测和 release contract 同时通过，才允许 `competition_ready=true`；否则保留负结果和具体阻塞原因。

