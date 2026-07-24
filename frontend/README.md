# SeatFlow Frontend

Next.js UI for the four SeatFlow research pages:

- 市场总览
- 资金传播图
- 股票预测分析
- 实验中心

Production/default behavior is fail-closed: the app calls `NEXT_PUBLIC_SEATFLOW_API_BASE_URL` and does not display local mock values as real metrics. Local UI state stubs are only enabled with `NEXT_PUBLIC_SEATFLOW_ENABLE_STUBS=1` and `?ui_state=loading|empty|error|degraded|real`.

Required evidence fields are displayed on every page:

- `data_scope`
- `seat_detail_scope`
- `lineage_hash`
- `risk_flags`
- `model_validity_status`

Useful commands:

```powershell
npm run lint
npm run typecheck
npm run test
npm run build
```
