import json
import unittest
from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from typing import cast
from unittest.mock import patch

from seatflow.data.contracts import serialize_suspension_evidence_jsonl
from seatflow.data.providers import (
    AkshareProvider,
    ProviderCapabilityError,
    ProviderNotConfiguredError,
    ProviderPermissionError,
    ProviderResponseError,
    SampleProvider,
    TushareHttpProvider,
    _post_json,
    _to_ts_code,
)
from seatflow.data.records import (
    INSTITUTIONAL_FLOW_PROXY_IDENTITY_SCOPE,
    NAMED_SEAT_IDENTITY_SCOPE,
    TradeSide,
)


class TushareHttpProviderTest(unittest.TestCase):
    def test_requires_runtime_token(self) -> None:
        provider = TushareHttpProvider(token="", post_json=lambda *_: {})

        with self.assertRaises(ProviderNotConfiguredError):
            provider.fetch_daily_bars(date(2026, 1, 15))

    def test_normalizes_daily_response(self) -> None:
        captured: dict[str, object] = {}

        def fake_post(url: str, payload: object, timeout: float) -> dict[str, object]:
            captured["url"] = url
            captured["payload"] = payload
            captured["timeout"] = timeout
            return {
                "code": 0,
                "data": {
                    "fields": [
                        "ts_code",
                        "trade_date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "pre_close",
                        "vol",
                        "amount",
                    ],
                    "items": [["000001.SZ", "20260115", 10, 11, 9, 10.5, 9.8, 1000, 10500]],
                },
            }

        provider = TushareHttpProvider(
            token="test-token", allow_insecure_http=True, post_json=fake_post
        )
        bars = provider.fetch_daily_bars(date(2026, 1, 15))

        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].stock_id, "000001.SZ")
        self.assertEqual(str(bars[0].close_price), "10.5")
        self.assertEqual(bars[0].volume_unit, "hand")
        self.assertEqual(bars[0].amount_unit, "thousand_cny")
        self.assertEqual(captured["url"], "http://api.tushare.pro")
        self.assertEqual(captured["timeout"], 30.0)

    def test_fetches_hashed_adjustment_factor_evidence(self) -> None:
        captured: dict[str, object] = {}

        def fake_post(_: str, payload: object, __: float) -> dict[str, object]:
            captured["payload"] = payload
            return {
                "code": 0,
                "data": {
                    "fields": ["ts_code", "trade_date", "adj_factor"],
                    "items": [
                        ["000001.SZ", "20260103", "1.2"],
                        ["000001.SZ", "20260102", "1.1"],
                    ],
                },
            }

        provider = TushareHttpProvider(
            token="test-token", allow_insecure_http=True, post_json=fake_post
        )
        evidence = provider.fetch_adjustment_factor_evidence(
            "000001.SZ", date(2026, 1, 2), date(2026, 1, 3)
        )

        self.assertEqual(
            captured["payload"],
            {
                "api_name": "adj_factor",
                "token": "test-token",
                "params": {
                    "ts_code": "000001.SZ",
                    "start_date": "20260102",
                    "end_date": "20260103",
                },
                "fields": "ts_code,trade_date,adj_factor",
            },
        )
        self.assertEqual(evidence.source, "tushare:adj_factor")
        self.assertEqual(
            [record.trade_date for record in evidence.records],
            [date(2026, 1, 2), date(2026, 1, 3)],
        )
        self.assertRegex(evidence.query_sha256, r"^[0-9a-f]{64}$")
        self.assertRegex(evidence.records[0].row_payload_sha256, r"^[0-9a-f]{64}$")

    def test_fetches_index_open_history_for_executable_benchmark(self) -> None:
        def fake_post(_: str, __: object, ___: float) -> dict[str, object]:
            return {
                "code": 0,
                "data": {
                    "fields": ["ts_code", "trade_date", "open"],
                    "items": [["000300.SH", "20260103", "101"], ["000300.SH", "20260102", "100"]],
                },
            }

        provider = TushareHttpProvider(
            token="test-token", allow_insecure_http=True, post_json=fake_post
        )

        rows = provider.fetch_index_open_history(
            "000300.SH", date(2026, 1, 2), date(2026, 1, 3)
        )

        self.assertEqual(
            rows,
            ((date(2026, 1, 2), Decimal("100")), (date(2026, 1, 3), Decimal("101"))),
        )

    def test_fetches_source_hashed_stock_status_evidence(self) -> None:
        def fake_post(_: str, payload: object, __: float) -> dict[str, object]:
            api_name = cast(dict[str, object], payload)["api_name"]
            if api_name == "stock_basic":
                return {
                    "code": 0,
                    "data": {
                        "fields": ["ts_code", "name", "list_date"],
                        "items": [["000001.SZ", "Ping An Bank", "19910403"]],
                    },
                }
            return {
                "code": 0,
                "data": {
                    "fields": ["ts_code", "start_date", "end_date", "name"],
                    "items": [["000001.SZ", "20260101", "", "Ping An Bank"]],
                },
            }

        provider = TushareHttpProvider(
            token="test-token", allow_insecure_http=True, post_json=fake_post
        )
        evidence = provider.fetch_stock_status_evidence(
            "000001.SZ", date(2026, 1, 1), date(2026, 1, 3)
        )

        self.assertEqual(evidence["stock_id"], "000001.SZ")
        self.assertEqual(
            evidence["stock_basic"],
            [{"stock_id": "000001.SZ", "name": "Ping An Bank", "list_date": "19910403"}],
        )
        self.assertRegex(str(evidence["query_sha256"]), r"^[0-9a-f]{64}$")

    def test_fetches_auditable_suspension_evidence_for_s_rows(self) -> None:
        captured: dict[str, object] = {}

        def fake_post(_: str, payload: object, __: float) -> dict[str, object]:
            captured["payload"] = payload
            return {
                "code": 0,
                "data": {
                    "fields": ["ts_code", "trade_date", "suspend_timing", "suspend_type"],
                    "items": [
                        ["000001.SZ", "20260115", "上午停牌", "S"],
                        ["000002.SZ", "20260115", None, "S"],
                        ["000003.SZ", "20260115", "", "S"],
                        ["000004.SZ", "20260115", "09:30", "R"],
                    ],
                },
            }

        provider = TushareHttpProvider(
            token="test-token", allow_insecure_http=True, post_json=fake_post
        )
        evidence = provider.fetch_suspension_evidence(date(2026, 1, 15))

        self.assertEqual(
            captured["payload"],
            {
                "api_name": "suspend_d",
                "token": "test-token",
                "params": {"trade_date": "20260115"},
                "fields": "ts_code,trade_date,suspend_timing,suspend_type",
            },
        )
        self.assertEqual(evidence.source, "tushare:suspend_d")
        self.assertEqual(evidence.source_row_count, 4)
        self.assertEqual(len(evidence.records), 3)
        self.assertEqual(evidence.records[0].stock_id, "000001.SZ")
        self.assertEqual(evidence.records[0].suspend_timing, "上午停牌")
        self.assertIsNone(evidence.records[1].suspend_timing)
        self.assertEqual(evidence.records[2].suspend_timing, "")
        self.assertEqual(evidence.records[0].trade_date, date(2026, 1, 15))
        self.assertRegex(evidence.records[0].row_payload_sha256, r"^[0-9a-f]{64}$")
        self.assertRegex(evidence.daily_query_sha256, r"^[0-9a-f]{64}$")
        serialized = json.loads(serialize_suspension_evidence_jsonl(evidence))
        self.assertEqual(serialized["daily_query_sha256"], evidence.daily_query_sha256)
        self.assertEqual(serialized["records"][0]["suspend_timing"], "上午停牌")
        self.assertIsNone(serialized["records"][1]["suspend_timing"])
        self.assertEqual(serialized["records"][2]["suspend_timing"], "")

    def test_suspension_evidence_hashes_are_stable_across_row_order(self) -> None:
        response_rows = [
            ["000002.SZ", "20260115", None, "S"],
            ["000001.SZ", "20260115", "", "S"],
        ]
        calls = 0

        def fake_post(_: str, __: object, ___: float) -> dict[str, object]:
            nonlocal calls
            calls += 1
            rows = response_rows if calls == 1 else list(reversed(response_rows))
            return {
                "code": 0,
                "data": {
                    "fields": ["ts_code", "trade_date", "suspend_timing", "suspend_type"],
                    "items": rows,
                },
            }

        provider = TushareHttpProvider(
            token="test-token", allow_insecure_http=True, post_json=fake_post
        )
        first = provider.fetch_suspension_evidence(date(2026, 1, 15))
        second = provider.fetch_suspension_evidence(date(2026, 1, 15))

        self.assertEqual(first.daily_query_sha256, second.daily_query_sha256)
        self.assertEqual(
            sorted(record.row_payload_sha256 for record in first.records),
            sorted(record.row_payload_sha256 for record in second.records),
        )
        self.assertEqual({record.suspend_timing for record in first.records}, {None, ""})

    def test_suspension_evidence_rejects_invalid_rows_and_duplicates(self) -> None:
        invalid_rows = (
            ["000001.SZ", "20260115", "09:30", "X"],
            ["000001.SZ", "20260230", "09:30", "S"],
            ["000001.SZ", "20260115", 930, "S"],
        )
        for row in invalid_rows:
            with self.subTest(row=row):
                provider = TushareHttpProvider(
                    token="test-token",
                    allow_insecure_http=True,
                    post_json=lambda *_, response_row=row: {
                        "code": 0,
                        "data": {
                            "fields": [
                                "ts_code",
                                "trade_date",
                                "suspend_timing",
                                "suspend_type",
                            ],
                            "items": [response_row],
                        },
                    },
                )
                with self.assertRaises(ProviderResponseError):
                    provider.fetch_suspension_evidence(date(2026, 1, 15))

        provider = TushareHttpProvider(
            token="test-token",
            allow_insecure_http=True,
            post_json=lambda *_: {
                "code": 0,
                "data": {
                    "fields": ["ts_code", "trade_date", "suspend_timing", "suspend_type"],
                    "items": [
                        ["000001.SZ", "20260115", "09:30", "S"],
                        ["SZ.000001", "2026-01-15", "09:30", "S"],
                    ],
                },
            },
        )
        with self.assertRaisesRegex(ProviderResponseError, "duplicate canonical row"):
            provider.fetch_suspension_evidence(date(2026, 1, 15))

    def test_non_tushare_and_unconfigured_providers_fail_closed_for_suspension_evidence(
        self,
    ) -> None:
        with self.assertRaisesRegex(ProviderNotConfiguredError, "TUSHARE_TOKEN"):
            TushareHttpProvider(token="", post_json=lambda *_: {}).fetch_suspension_evidence(
                date(2026, 1, 15)
            )
        for provider in (AkshareProvider(), SampleProvider()):
            with self.subTest(provider=provider.name):
                with self.assertRaises(ProviderCapabilityError):
                    provider.fetch_suspension_evidence(date(2026, 1, 15))

    def test_fetches_bounded_calendar_and_stock_history(self) -> None:
        def fake_post(_: str, payload: object, __: float) -> dict[str, object]:
            if not isinstance(payload, Mapping):
                raise AssertionError("provider payload must be a mapping")
            if payload["api_name"] == "trade_cal":
                self.assertEqual(
                    payload["params"],
                    {
                        "exchange": "SSE",
                        "start_date": "20230103",
                        "end_date": "20230104",
                        "is_open": "1",
                    },
                )
                return {
                    "code": 0,
                    "data": {
                        "fields": ["exchange", "cal_date", "is_open"],
                        "items": [["SSE", "20230104", 1], ["SSE", "20230103", 1]],
                    },
                }
            if payload["api_name"] == "index_daily":
                self.assertEqual(
                    payload["params"],
                    {"ts_code": "000300.SH", "start_date": "20230103", "end_date": "20230104"},
                )
                return {
                    "code": 0,
                    "data": {
                        "fields": ["ts_code", "trade_date", "close"],
                        "items": [["000300.SH", "20230104", 4000]],
                    },
                }
            self.assertEqual(payload["api_name"], "daily")
            self.assertEqual(
                payload["params"],
                {"ts_code": "000001.SZ", "start_date": "20230103", "end_date": "20230104"},
            )
            return {
                "code": 0,
                "data": {
                    "fields": [
                        "ts_code",
                        "trade_date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "pre_close",
                        "vol",
                        "amount",
                    ],
                    "items": [["000001.SZ", "20230104", 10, 11, 9, 10.5, 9.8, 1000, 10500]],
                },
            }

        provider = TushareHttpProvider(
            token="test-token", allow_insecure_http=True, post_json=fake_post
        )
        calendar = provider.fetch_trading_calendar(date(2023, 1, 3), date(2023, 1, 4))
        bars = provider.fetch_daily_bar_history(
            "000001.SZ", date(2023, 1, 3), date(2023, 1, 4)
        )
        index_closes = provider.fetch_index_close_history(
            "000300.SH", date(2023, 1, 3), date(2023, 1, 4)
        )

        self.assertEqual(calendar, (date(2023, 1, 3), date(2023, 1, 4)))
        self.assertEqual(bars[0].trade_date, date(2023, 1, 4))
        self.assertEqual(bars[0].volume_unit, "hand")
        self.assertEqual(index_closes, ((date(2023, 1, 4), Decimal("4000")),))

    def test_expands_institution_row_to_buy_and_sell_events(self) -> None:
        def fake_post(_: str, __: object, ___: float) -> dict[str, object]:
            return {
                "code": 0,
                "data": {
                    "fields": [
                        "trade_date",
                        "ts_code",
                        "exalter",
                        "side",
                        "buy",
                        "sell",
                        "net_buy",
                        "reason",
                    ],
                    "items": [
                        [
                            "20260115",
                            "000001.SZ",
                            "\u673a\u6784\u4e13\u7528",
                            "\u4e70\u5165",
                            120,
                            20,
                            100,
                            "\u504f\u79bb\u503c",
                        ]
                    ],
                },
            }

        provider = TushareHttpProvider(
            token="test-token", allow_insecure_http=True, post_json=fake_post
        )
        events = provider.fetch_dragon_tiger_seat_events(date(2026, 1, 15))

        self.assertEqual([event.side for event in events], [TradeSide.BUY, TradeSide.SELL])
        self.assertEqual([event.amount for event in events], [120.0, 20.0])
        self.assertTrue(all(event.available_time is not None for event in events))

    def test_deduplicates_mirrored_top_inst_payloads_without_double_counting(self) -> None:
        def fake_post(_: str, __: object, ___: float) -> dict[str, object]:
            return {
                "code": 0,
                "data": {
                    "fields": [
                        "trade_date",
                        "ts_code",
                        "exalter",
                        "side",
                        "buy",
                        "sell",
                        "net_buy",
                        "reason",
                    ],
                    "items": [
                        ["20260115", "000001.SZ", "named-seat", "buy", 120, 20, 100, "reason"],
                        ["20260115", "000001.SZ", "named-seat", "sell", 120, 20, 100, "reason"],
                    ],
                },
            }

        provider = TushareHttpProvider(
            token="test-token", allow_insecure_http=True, post_json=fake_post
        )
        events = provider.fetch_dragon_tiger_seat_events(date(2026, 1, 15))

        self.assertEqual([event.side for event in events], [TradeSide.BUY, TradeSide.SELL])
        self.assertEqual([event.amount for event in events], [120.0, 20.0])
        self.assertTrue(all(event.source_row_count == 2 for event in events))
        self.assertTrue(all(event.source_side_set == ("buy", "sell") for event in events))
        self.assertTrue(all(event.identity_scope == NAMED_SEAT_IDENTITY_SCOPE for event in events))

    def test_aggregates_distinct_named_payloads_by_true_trade_direction(self) -> None:
        def fake_post(_: str, __: object, ___: float) -> dict[str, object]:
            return {
                "code": 0,
                "data": {
                    "fields": [
                        "trade_date",
                        "ts_code",
                        "exalter",
                        "side",
                        "buy",
                        "sell",
                        "net_buy",
                        "reason",
                    ],
                    "items": [
                        ["20260115", "000001.SZ", "named-seat", "sell", 30, 0, 30, "reason"],
                        ["20260115", "000001.SZ", "named-seat", "buy", 70, 0, 70, "reason"],
                    ],
                },
            }

        provider = TushareHttpProvider(
            token="test-token", allow_insecure_http=True, post_json=fake_post
        )
        events = provider.fetch_dragon_tiger_seat_events(date(2026, 1, 15))

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].side, TradeSide.BUY)
        self.assertEqual(events[0].amount, 100.0)
        self.assertEqual(events[0].source_row_count, 2)
        self.assertEqual(events[0].identity_scope, NAMED_SEAT_IDENTITY_SCOPE)

    def test_anonymous_institutional_rows_use_date_scoped_flow_proxy_ids(self) -> None:
        def fake_post(_: str, payload: object, __: float) -> dict[str, object]:
            if not isinstance(payload, Mapping):
                raise AssertionError("provider payload must be a mapping")
            trade_date = str(payload["params"]["trade_date"])
            return {
                "code": 0,
                "data": {
                    "fields": [
                        "trade_date",
                        "ts_code",
                        "exalter",
                        "side",
                        "buy",
                        "sell",
                        "net_buy",
                        "reason",
                    ],
                    "items": [
                        [
                            trade_date,
                            "000001.SZ",
                            "\u673a\u6784\u4e13\u7528",
                            "buy",
                            120,
                            0,
                            120,
                            "reason",
                        ],
                    ],
                },
            }

        provider = TushareHttpProvider(
            token="test-token", allow_insecure_http=True, post_json=fake_post
        )
        first = provider.fetch_dragon_tiger_seat_events(date(2026, 1, 15))
        second = provider.fetch_dragon_tiger_seat_events(date(2026, 1, 16))

        self.assertEqual(first[0].identity_scope, INSTITUTIONAL_FLOW_PROXY_IDENTITY_SCOPE)
        self.assertEqual(second[0].identity_scope, INSTITUTIONAL_FLOW_PROXY_IDENTITY_SCOPE)
        self.assertNotEqual(first[0].seat_id, second[0].seat_id)
        self.assertIn("2026-01-15", first[0].seat_id)
        self.assertIn("2026-01-16", second[0].seat_id)

    def test_skips_non_equity_dragon_tiger_instruments(self) -> None:
        def fake_post(_: str, payload: object, __: float) -> dict[str, object]:
            if not isinstance(payload, Mapping):
                raise AssertionError("provider payload must be a mapping")
            api_name = str(payload.get("api_name"))
            if api_name == "top_list":
                return {
                    "code": 0,
                    "data": {
                        "fields": [
                            "trade_date",
                            "ts_code",
                            "name",
                            "l_buy",
                            "l_sell",
                            "net_amount",
                            "reason",
                        ],
                        "items": [
                            ["20260115", "118071.SH", "convertible-bond", 10, 1, 9, "reason"],
                            ["20260115", "000002.SZ", "incomplete", "--", 2, 1, "reason"],
                            ["20260115", "000001.SZ", "equity", 12, 2, 10, "reason"],
                        ],
                    },
                }
            return {
                "code": 0,
                "data": {
                    "fields": [
                        "trade_date",
                        "ts_code",
                        "exalter",
                        "side",
                        "buy",
                        "sell",
                        "net_buy",
                        "reason",
                    ],
                    "items": [["20260115", "118071.SH", "institution", "buy", 20, 0, 20, "reason"]],
                },
            }

        provider = TushareHttpProvider(
            token="test-token", allow_insecure_http=True, post_json=fake_post
        )

        summaries = provider.fetch_dragon_tiger_summaries(date(2026, 1, 15))
        events = provider.fetch_dragon_tiger_seat_events(date(2026, 1, 15))

        self.assertEqual([summary.stock_id for summary in summaries], ["000001.SZ"])
        self.assertEqual(events, ())

    def test_sanitizes_permission_failure(self) -> None:
        def fake_post(_: str, __: object, ___: float) -> dict[str, object]:
            return {"code": 2002, "msg": "token=test-token must not be echoed"}

        provider = TushareHttpProvider(
            token="test-token", allow_insecure_http=True, post_json=fake_post
        )

        with self.assertRaises(ProviderPermissionError) as context:
            provider.fetch_dragon_tiger_summaries(date(2026, 1, 15))

        self.assertNotIn("test-token", str(context.exception))

    def test_rejects_insecure_http_without_explicit_opt_in(self) -> None:
        provider = TushareHttpProvider(
            token="test-token", allow_insecure_http=False, post_json=lambda *_: {}
        )

        with self.assertRaisesRegex(ProviderNotConfiguredError, "ALLOW_INSECURE_HTTP"):
            provider.fetch_daily_bars(date(2026, 1, 15))

    def test_normalizes_beijing_exchange_codes(self) -> None:
        self.assertEqual(_to_ts_code("920082"), "920082.BJ")
        self.assertEqual(_to_ts_code("SH.600000"), "600000.SH")

    def test_converts_free_provider_network_errors(self) -> None:
        class BrokenAkshare:
            def stock_zh_a_hist(self, **_: object) -> object:
                raise RuntimeError("network unavailable")

        provider = AkshareProvider()
        provider._load_akshare = lambda: BrokenAkshare()  # type: ignore[method-assign]

        with self.assertRaisesRegex(ProviderResponseError, "AkShare daily request failed"):
            provider.fetch_daily_bars(date(2026, 1, 15), "000001.SZ")

    def test_converts_tushare_timeout_to_provider_error(self) -> None:
        with patch("seatflow.data.providers.urlopen", side_effect=TimeoutError):
            with self.assertRaisesRegex(ProviderResponseError, "timed out"):
                _post_json("http://example.invalid", {}, 1.0)
