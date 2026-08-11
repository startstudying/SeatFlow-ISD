"""SeatFlow-ISD V1.0 ingestion package.

Reads the verified engineering dataset at
``<dataset_root>/<dataset_version>/`` and converts the V1 records into
the V3 domain contracts used by the rest of the SeatFlow research
engine.

Submodules:

- :mod:`contracts` — V3 dataclasses for events, participations, market
  bars, daily basics, cycles, samples, and labels.
- :mod:`reader` — verified reader that recomputes every file hash.
- :mod:`validator` — row-level integrity and provenance checks.
- :mod:`adapter` — V1 → V3 row conversion.
"""

from seatflow.ingestion.seatflow_isd.contracts import (
    ISDBenchmarkPath,
    ISDCycleV3,
    ISDDailyBasicV3,
    ISDEventV3,
    ISDLabelRowV3,
    ISDLoadResult,
    ISDLhbUniverseV3,
    ISDMaturity,
    ISDMarketBarV3,
    ISDSampleV3,
    ISDSeatParticipationV3,
)
from seatflow.ingestion.seatflow_isd.adjustments import (
    AdjFactorRow,
    AdjustmentSource,
    CompanyActionRow,
    adjust_price,
    discover_adjustment_records,
    load_adj_factor,
    load_company_actions,
)
from seatflow.ingestion.seatflow_isd.benchmark import (
    BenchmarkBar,
    BenchmarkSource,
    discover_benchmark_records,
    load_benchmark,
    to_isd_market_bars,
)
from seatflow.ingestion.seatflow_isd.industry import (
    IndustryMembershipRow,
    IndustrySource,
    load_industry_membership,
)
from seatflow.ingestion.seatflow_isd.proxy_ablation import (
    ALL_SCHEMES,
    AblationScheme,
    build_all_schemes,
    scheme_e0_named_only,
    scheme_e1_named_plus_proxy,
    scheme_e2_anonymous_only,
    scheme_e3_shuffled_seats,
    scheme_e4_proxy_only,
)
from seatflow.ingestion.seatflow_isd.tradeability import (
    TradeabilityRow,
    TradeabilitySource,
    discover_tradeability_records,
    load_tradeability,
)
from seatflow.ingestion.seatflow_isd.loader import ISDLoadReport, load_isd_v1

__all__ = [
    "ALL_SCHEMES",
    "AblationScheme",
    "AdjFactorRow",
    "AdjustmentSource",
    "BenchmarkBar",
    "BenchmarkSource",
    "CompanyActionRow",
    "ISDBenchmarkPath",
    "ISDCycleV3",
    "ISDDailyBasicV3",
    "ISDEventV3",
    "ISDLabelRowV3",
    "ISDLoadReport",
    "ISDLoadResult",
    "ISDLhbUniverseV3",
    "ISDMaturity",
    "ISDMarketBarV3",
    "ISDSampleV3",
    "ISDSeatParticipationV3",
    "IndustryMembershipRow",
    "IndustrySource",
    "TradeabilityRow",
    "TradeabilitySource",
    "adjust_price",
    "build_all_schemes",
    "discover_adjustment_records",
    "discover_benchmark_records",
    "discover_tradeability_records",
    "load_adj_factor",
    "load_benchmark",
    "load_company_actions",
    "load_industry_membership",
    "load_isd_v1",
    "load_tradeability",
    "scheme_e0_named_only",
    "scheme_e1_named_plus_proxy",
    "scheme_e2_anonymous_only",
    "scheme_e3_shuffled_seats",
    "scheme_e4_proxy_only",
    "to_isd_market_bars",
]
