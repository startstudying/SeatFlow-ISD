# SeatFlow-ISD V2 Spillover Design (Stage H)

> Independent design document. The V2 build is **not** part of the
> V1 deliverable — V1.0/V1.1 remain DIRECT-only and LHB-strict.

## 1. Scope boundary

| Aspect | V1.0 / V1.1 (this build) | V2 (this document) |
|---|---|---|
| Universe | LHB-ever stocks only | LHB-ever + propagation candidates |
| Pool | DIRECT | DIRECT + SPILLOVER |
| Benchmarks | CSI500, adj_factor, tradeability, SW2021 | All V1 sources + concept + industry propagation |
| Identity | Named seat (named only after E0) | Named + proxy + co-seat + co-industry + co-concept |
| Cycle | One event drives the window | A primary event drives a propagation fan-out |
| Maturity | 5-day raw + excess | Same horizon, plus per-candidate S-score |

V1 is **never** modified to add SPILLOVER. V2 lives in
``seatflow-isd-v2/`` and has its own build, audit, and seal.

## 2. Propagation candidate construction

Given a primary event ``e = (stock_s, trade_date, reason)`` in V1,
V2 derives a candidate set ``C(e)`` as the union of:

1. **Co-seat candidates**: every other stock ``s' != s`` that shares
   at least one seat id with the primary event within a rolling
   ±3 trading-day window. The seat set comes from V1
   ``participations``; the window keeps only the most recent
   flow.
2. **Co-industry candidates**: every stock in the same SW2021 L1
   industry as ``s`` that has a non-empty in_date and no out_date
   (still active). The industry source is the V1 industry
   membership.
3. **Co-concept candidates**: every stock sharing a non-empty
   concept with ``s``. The concept source is a new V2-only source
   (out of scope for this build).

A SPILLOVER sample is then a (cycle, day, candidate_stock) tuple
with the same horizon as the V1 DIRECT sample. ``cand_event_id``
is a deterministic hash of
``(cand_stock_id, primary_event_id)`` so the same V1 + V2 inputs
always yield the same V2 dataset.

## 3. SPILLOVER raw label

For each (candidate, horizon_exit_date) the same
``compute_return_path`` rule applies as V1:

```
R1_spill = C1 / O1 - 1
R3_spill = C3 / O1 - 1
R5_spill = C5 / O1 - 1
```

Tail samples beyond the calendar end stay UNMATURE with null
returns (zero-filling remains forbidden).

## 4. SPILLOVER benchmark excess

Each SPILLOVER sample uses the same V1 benchmark source
(``CSI500``) for its 1/3/5-day excess. This makes V1 and V2
excess returns directly comparable across the two pool types.

## 5. S-score (spillover signal)

A V2-only ``S-score`` is computed per SPILLOVER sample as:

```
S = max(R5_spill - R5_benchmark, 0)
```

or a more general form (peak within the 5-day window) once V2
decides on the exact formulation. The score is **never** used to
mark a sample as Mature/UNMATURE; it is an additional field on
the V2 sample.

## 6. Hard prohibitions

V2 inherits every V1 prohibition:

- Zero-fill on missing bars: SPILLOVER UNMATURE rows have null returns.
- Replacing institutional_flow_proxy with named_seat in the graph
  edges: the two scopes remain distinct.
- Inheriting the V1 ``sample_id`` namespace: V2 sample ids use
  the ``smpv2-`` prefix so V1 backfills are unambiguous.

V2 also adds:

- A SPILLOVER row's underlying candidate must be in
  ``lhb_stock_universe`` (the V1 universe). Propagation to never-
  listed stocks is forbidden.
- Co-industry candidate count per event is capped (e.g. <= 50) so
  the SPILLOVER pool doesn't explode in industry-heavy sectors.

## 7. Audit, seal, and ablation

V2 inherits V1's formal-validity gates with two additions:

- ``FORMAL_NO_OUT_OF_UNIVERSE_SPILLOVER`` — every SPILLOVER
  candidate is in the V1 universe.
- ``FORMAL_PROPAGATION_LIMITS_RESPECTED`` — per-event candidate
  count is bounded.

V2 also inherits the V1 proxy-ablation E0..E4 scheme, plus a new
ablation pair:

- ``E5_industry_off`` — drop co-industry candidates.
- ``E6_concept_off`` — drop co-concept candidates.

## 8. File layout

```
datasets/seatflow_isd_v2/<version>/
  bronze/source_inventory.json
  silver/
    events.jsonl           (re-published from V1)
    participations.jsonl   (re-published from V1)
    market_bars.jsonl      (re-published from V1)
    daily_basic.jsonl      (re-published from V1)
    propagation_candidates.jsonl   (new)
  gold/
    lhb_stock_universe.json (re-published from V1)
    cycles.jsonl           (re-published from V1)
    spillover_cycles.jsonl (new)
  samples/
    direct_samples.jsonl   (re-published from V1)
    spillover_samples.jsonl (new)
  labels/
    label_rows.jsonl       (re-published from V1)
    spillover_label_rows.jsonl (new)
  splits/walk_forward_plan.json
  schemas/dataset_schema.json
  manifests/dataset_manifest.json
  quality_reports/quality_report.json
```

V2 emits ``dataset_seal.json`` only when **both** V1 gates and the
V2-specific gates pass.

## 9. Why V2 is its own dataset, not a V1 amendment

V1 explicitly forbids SPILLOVER. Pushing SPILLOVER into V1 would:

- Re-open the ``dataset_valid`` semantics on a version that is
  already deployed.
- Couple the V1 universe membership logic to propagation rules
  that do not exist today.
- Break the E0..E4 ablation scheme, which assumes the seat
  participation rows are unambiguous.

V2 as a separate build is the cleanest way to preserve V1's
DIRECT-only guarantee while extending the dataset for Diffusion
and spillover research.

## 10. Open questions deferred to a V2 build

- Exact S-score formulation.
- Co-concept source acquisition path.
- Whether SPILLOVER labels share the V1 benchmark or have a
  per-candidate benchmark.
- Whether V2 needs a wider window (10-day or 20-day) than V1's
  strict 5-day horizon.

These questions are out of scope for the V1.0/V1.1 handoff and
must be resolved before any V2 build starts.
