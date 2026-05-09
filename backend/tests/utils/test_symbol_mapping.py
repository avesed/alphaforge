"""Tests for symbol mapping between StockPulse and Qlib formats."""
import pytest

from app.utils.symbol_mapping import (
    normalize_symbol_for_qlib,
    qlib_to_stockpulse,
    stockpulse_to_qlib,
)


class TestStockpulseToQlib:
    """StockPulse → Qlib conversion."""

    @pytest.mark.parametrize(
        "sp_symbol, expected",
        [
            ("600000.SS", "SH600000"),
            ("601318.SS", "SH601318"),
            ("000001.SZ", "SZ000001"),
            ("300750.SZ", "SZ300750"),
        ],
    )
    def test_cn_symbols(self, sp_symbol, expected):
        assert stockpulse_to_qlib(sp_symbol) == expected

    @pytest.mark.parametrize(
        "sp_symbol, expected",
        [
            ("0700.HK", "HK0700"),
            ("9988.HK", "HK9988"),
            ("0005.HK", "HK0005"),
        ],
    )
    def test_hk_symbols(self, sp_symbol, expected):
        assert stockpulse_to_qlib(sp_symbol) == expected

    @pytest.mark.parametrize(
        "sp_symbol",
        ["AAPL", "MSFT", "GOOGL", "TSLA", "BRK.B"],
    )
    def test_us_symbols_passthrough(self, sp_symbol):
        assert stockpulse_to_qlib(sp_symbol) == sp_symbol

    @pytest.mark.parametrize(
        "sp_symbol, expected",
        [
            ("GC=F", "GCF"),
            ("SI=F", "SIF"),
            ("PL=F", "PLF"),
            ("PA=F", "PAF"),
        ],
    )
    def test_metal_symbols(self, sp_symbol, expected):
        assert stockpulse_to_qlib(sp_symbol) == expected

    def test_case_insensitive_suffix(self):
        assert stockpulse_to_qlib("600000.ss") == "SH600000"

    def test_empty_string(self):
        assert stockpulse_to_qlib("") == ""


class TestQlibToStockpulse:
    """Qlib → StockPulse conversion."""

    @pytest.mark.parametrize(
        "qlib_symbol, expected",
        [
            ("SH600000", "600000.SS"),
            ("SZ000001", "000001.SZ"),
            ("HK0700", "0700.HK"),
        ],
    )
    def test_prefixed_symbols(self, qlib_symbol, expected):
        assert qlib_to_stockpulse(qlib_symbol) == expected

    @pytest.mark.parametrize(
        "qlib_symbol",
        ["AAPL", "MSFT", "GOOGL"],
    )
    def test_us_symbols_passthrough(self, qlib_symbol):
        assert qlib_to_stockpulse(qlib_symbol) == qlib_symbol

    @pytest.mark.parametrize(
        "qlib_symbol, expected",
        [
            ("GCF", "GC=F"),
            ("SIF", "SI=F"),
            ("PLF", "PL=F"),
            ("PAF", "PA=F"),
        ],
    )
    def test_metal_symbols(self, qlib_symbol, expected):
        assert qlib_to_stockpulse(qlib_symbol) == expected

    def test_non_numeric_prefix_no_false_match(self):
        """SHOP should NOT match SH prefix (remaining 'OP' is not numeric)."""
        assert qlib_to_stockpulse("SHOP") == "SHOP"

    def test_prefix_only_no_match(self):
        """Bare prefix like 'SH' without trailing digits should pass through."""
        assert qlib_to_stockpulse("SH") == "SH"


class TestNormalizeSymbolForQlib:
    """normalize_symbol_for_qlib handles bare codes, suffixed, and already-Qlib formats."""

    def test_already_qlib_format(self):
        assert normalize_symbol_for_qlib("SH600000", "cn") == "SH600000"
        assert normalize_symbol_for_qlib("SZ000001", "cn") == "SZ000001"

    def test_stockpulse_suffix(self):
        assert normalize_symbol_for_qlib("600000.SS", "cn") == "SH600000"
        assert normalize_symbol_for_qlib("0700.HK", "hk") == "HK0700"

    @pytest.mark.parametrize(
        "code, market, expected",
        [
            ("600000", "cn", "SH600000"),
            ("601318", "sh", "SH601318"),
            ("000001", "sz", "SZ000001"),
            ("300750", "cn", "SZ300750"),
            ("900001", "cn", "SH900001"),
            ("200001", "cn", "SZ200001"),
        ],
    )
    def test_bare_numeric_cn_inference(self, code, market, expected):
        assert normalize_symbol_for_qlib(code, market) == expected

    def test_us_symbol_passthrough(self):
        assert normalize_symbol_for_qlib("AAPL", "us") == "AAPL"

    def test_metal_symbol(self):
        assert normalize_symbol_for_qlib("GC=F", "metal") == "GCF"

    def test_bare_numeric_us_passthrough(self):
        """Bare numeric code with US market should not infer SH/SZ prefix."""
        assert normalize_symbol_for_qlib("600000", "us") == "600000"


class TestRoundtrip:
    """Verify bidirectional conversion is lossless."""

    @pytest.mark.parametrize(
        "sp_symbol",
        ["600000.SS", "000001.SZ", "0700.HK", "GC=F"],
    )
    def test_roundtrip_sp_qlib_sp(self, sp_symbol):
        qlib = stockpulse_to_qlib(sp_symbol)
        back = qlib_to_stockpulse(qlib)
        assert back == sp_symbol

    @pytest.mark.parametrize(
        "qlib_symbol",
        ["SH600000", "SZ000001", "HK0700", "GCF"],
    )
    def test_roundtrip_qlib_sp_qlib(self, qlib_symbol):
        sp = qlib_to_stockpulse(qlib_symbol)
        back = stockpulse_to_qlib(sp)
        assert back == qlib_symbol
