"""Week-2 fix: SW2021 industry resolver point-in-time semantics."""

from __future__ import annotations

import csv
import unittest
from datetime import date
from tempfile import NamedTemporaryFile

from seatflow.data.industry_resolver import (
    IndustryMembership,
    IndustryResolver,
    load_industry_resolver,
)


def _membership(
    stock_id: str,
    industry: str,
    in_date: str,
    out_date: str | None = None,
) -> IndustryMembership:
    return IndustryMembership(
        stock_id=stock_id,
        industry_code="801030.SI",
        industry_name=industry,
        in_date=date.fromisoformat(in_date),
        out_date=date.fromisoformat(out_date) if out_date else None,
    )


class IndustryResolverTest(unittest.TestCase):
    def test_active_membership_returns_industry(self) -> None:
        resolver = IndustryResolver(
            memberships=(_membership("600000.SH", "银行", "2020-01-01"),)
        )
        self.assertEqual(
            resolver.industry_for("600000.SH", date(2023, 6, 15)), "银行"
        )

    def test_before_in_date_returns_none(self) -> None:
        resolver = IndustryResolver(
            memberships=(_membership("600000.SH", "银行", "2023-06-01"),)
        )
        self.assertIsNone(resolver.industry_for("600000.SH", date(2023, 5, 31)))

    def test_after_out_date_returns_none(self) -> None:
        resolver = IndustryResolver(
            memberships=(
                _membership("600000.SH", "银行", "2020-01-01", "2023-06-01"),
            )
        )
        self.assertIsNone(resolver.industry_for("600000.SH", date(2023, 6, 2)))

    def test_out_date_is_exclusive(self) -> None:
        # Active on the out_date's eve; out_date itself is no longer active.
        resolver = IndustryResolver(
            memberships=(
                _membership("600000.SH", "银行", "2020-01-01", "2023-06-01"),
            )
        )
        self.assertEqual(
            resolver.industry_for("600000.SH", date(2023, 5, 31)), "银行"
        )
        self.assertIsNone(resolver.industry_for("600000.SH", date(2023, 6, 1)))

    def test_overlapping_memberships_latest_in_date_wins(self) -> None:
        # A stock reclassified from 银行 to 非银金融 with overlapping snapshots.
        resolver = IndustryResolver(
            memberships=(
                _membership("600000.SH", "银行", "2020-01-01"),
                _membership("600000.SH", "非银金融", "2023-01-01"),
            )
        )
        self.assertEqual(
            resolver.industry_for("600000.SH", date(2023, 6, 15)), "非银金融"
        )

    def test_unknown_stock_returns_none(self) -> None:
        resolver = IndustryResolver(memberships=())
        self.assertIsNone(resolver.industry_for("999999.SH", date(2023, 6, 15)))

    def test_resolve_records_bulk(self) -> None:
        resolver = IndustryResolver(
            memberships=(_membership("600000.SH", "银行", "2020-01-01"),)
        )
        resolved = resolver.resolve_records(
            [("600000.SH", date(2023, 6, 15)), ("000001.SZ", date(2023, 6, 15))]
        )
        self.assertEqual(resolved["600000.SH"]["2023-06-15"], "银行")
        self.assertIsNone(resolved["000001.SZ"]["2023-06-15"])

    def test_load_from_csv_round_trip(self) -> None:
        with NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
        ) as tmp:
            writer = csv.DictWriter(
                tmp,
                fieldnames=["ts_code", "l1_code", "l1_name", "in_date", "out_date"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "ts_code": "600000.SH",
                    "l1_code": "801780.SI",
                    "l1_name": "银行",
                    "in_date": "2020-01-01",
                    "out_date": "",
                }
            )
            writer.writerow(
                {
                    "ts_code": "000001.SZ",
                    "l1_code": "801780.SI",
                    "l1_name": "银行",
                    "in_date": "2020-01-01",
                    "out_date": "2023-06-01",
                }
            )
            path = tmp.name
        resolver = load_industry_resolver(path)
        self.assertEqual(
            resolver.industry_for("600000.SH", date(2023, 6, 15)), "银行"
        )
        self.assertIsNone(
            resolver.industry_for("000001.SZ", date(2023, 6, 15))
        )


if __name__ == "__main__":
    unittest.main()
