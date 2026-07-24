# SeatFlow analysis bundle

This directory contains the local research data exported for reproducible,
private analysis.

## Data archive

- File: `seatflow-data-20260724.tar.gz`
- Original size: approximately 6.6 GB
- Compressed size: approximately 594 MB
- Archive entries: 18,487
- SHA-256: `987fae2585353083e3fcc658f362c4ec340d1f688ac6b857ba02f96a3367c254`

Included source directories:

- `data/historical_backfill`
- `data/real`
- `data/real_tushare_probe`
- `data/real_tushare_probe_v2`

Extract from the repository root:

```powershell
tar -xzf shared-analysis/seatflow-data-20260724.tar.gz
```

Runtime logs are committed separately under `logs/` so they can be inspected
without extracting the data archive.

The archive and logs were scanned for common access-token, API-key, and private
key patterns before publication.
