"""Tests for QlibContext — mapping and reset logic (no actual Qlib import)."""
import pytest

from app.context import QlibContext


class TestMarketToRegion:

    @pytest.mark.parametrize(
        "market, expected_region",
        [
            ("us", "us"),
            ("hk", "us"),
            ("sh", "cn"),
            ("sz", "cn"),
            ("cn", "cn"),
            ("metal", "us"),
        ],
    )
    def test_mapping(self, market, expected_region):
        assert QlibContext.MARKET_TO_REGION[market] == expected_region

    def test_unknown_market_not_in_mapping(self):
        assert "jp" not in QlibContext.MARKET_TO_REGION


class TestMarketToDataDir:

    @pytest.mark.parametrize(
        "market, expected_dir",
        [
            ("us", "us_data"),
            ("hk", "hk_data"),
            ("sh", "cn_data"),
            ("sz", "cn_data"),
            ("cn", "cn_data"),
            ("metal", "metal_data"),
        ],
    )
    def test_data_dir_mapping(self, market, expected_dir):
        assert QlibContext.MARKET_TO_DATA_DIR[market] == expected_dir


class TestStatusMarkets:

    def test_canonical_markets(self):
        assert set(QlibContext.STATUS_MARKETS.keys()) == {"us", "hk", "cn", "metal"}

    def test_distinct_data_dirs(self):
        dirs = list(QlibContext.STATUS_MARKETS.values())
        assert len(dirs) == len(set(dirs))


class TestReset:

    def test_reset_clears_state(self):
        QlibContext._current_region = "us"
        QlibContext._current_provider_uri = "/some/path"
        QlibContext._initialized = True

        QlibContext.reset()

        assert QlibContext._current_region is None
        assert QlibContext._current_provider_uri is None
        assert QlibContext._initialized is False

    def test_is_initialized_after_reset(self):
        QlibContext._initialized = True
        QlibContext.reset()
        assert QlibContext.is_initialized() is False

    def test_get_current_region_after_reset(self):
        QlibContext._current_region = "cn"
        QlibContext.reset()
        assert QlibContext.get_current_region() is None


class TestEnsureInitValidation:

    def test_unknown_market_raises(self):
        with pytest.raises(ValueError, match="Unknown market"):
            QlibContext.ensure_init("jp")
