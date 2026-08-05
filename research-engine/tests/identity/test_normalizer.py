"""Tests for name normalizer."""

from seatflow.identity.normalizer import normalize_seat_name, normalize_stock_code


class TestNormalizeSeatName:
    def test_strip_whitespace(self):
        result = normalize_seat_name("  华泰证券  ")
        assert result.normalized == "华泰证券"

    def test_remove_branch_suffix(self):
        result = normalize_seat_name("华泰证券#1")
        assert result.normalized == "华泰证券"
        assert result.branch_suffix_removed is True

    def test_remove_parenthetical_suffix(self):
        result = normalize_seat_name("华泰证券(第1营业部)")
        assert result.normalized == "华泰证券"

    def test_normalize_full_width_punctuation(self):
        result = normalize_seat_name("华泰证券（深圳）")
        assert "（" not in result.normalized
        assert "）" not in result.normalized

    def test_collapse_internal_whitespace(self):
        result = normalize_seat_name("华泰 证券 深圳")
        assert result.normalized == "华泰证券深圳"


class TestNormalizeStockCode:
    def test_already_has_suffix(self):
        assert normalize_stock_code("000001.SZ") == "000001.SZ"

    def test_infer_sh(self):
        assert normalize_stock_code("600000") == "600000.SH"

    def test_infer_sz(self):
        assert normalize_stock_code("000001") == "000001.SZ"

    def test_infer_bj(self):
        assert normalize_stock_code("430047") == "430047.BJ"

    def test_uppercase(self):
        assert normalize_stock_code("000001.sz") == "000001.SZ"
