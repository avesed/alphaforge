"""Tests for per-market ML configuration."""
import dataclasses

import pytest

from app.services.market_config import (
    MARKET_CONFIGS,
    MarketConfig,
    apply_override,
    get_market_config,
)


class TestMarketConfigs:
    """Verify all three market configs are correctly defined."""

    def test_all_markets_present(self):
        assert set(MARKET_CONFIGS.keys()) == {"us", "cn", "hk"}

    def test_us_config_values(self):
        cfg = MARKET_CONFIGS["us"]
        assert cfg.use_temporal_sort is False
        assert cfg.use_sector_neutral_labels is True
        assert cfg.use_balanced_quintiles is True
        assert cfg.use_sector_rank is True
        assert cfg.use_interactions is True
        assert cfg.nan_threshold == 0.75
        assert cfg.ffill_limit == 45

    def test_cn_config_values(self):
        cfg = MARKET_CONFIGS["cn"]
        assert cfg.use_temporal_sort is True
        assert cfg.use_sector_neutral_labels is False
        assert cfg.nan_threshold == 0.90
        assert cfg.ffill_limit == 90
        assert cfg.index_symbol == "000300.SS"

    def test_hk_config_values(self):
        cfg = MARKET_CONFIGS["hk"]
        assert cfg.use_temporal_sort is True
        assert cfg.min_ic_threshold == 0.005
        assert cfg.min_icir_threshold == 0.05
        assert cfg.index_symbol == "^HSI"

    def test_frozen_immutability(self):
        cfg = MARKET_CONFIGS["us"]
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.nan_threshold = 0.5


class TestGetMarketConfig:

    @pytest.mark.parametrize("market", ["us", "cn", "hk"])
    def test_known_markets(self, market):
        cfg = get_market_config(market)
        assert cfg is MARKET_CONFIGS[market]

    def test_case_insensitive(self):
        cfg = get_market_config("US")
        assert cfg is MARKET_CONFIGS["us"]

    def test_unknown_market_falls_back_to_us(self):
        cfg = get_market_config("jp")
        assert cfg is MARKET_CONFIGS["us"]

    def test_empty_string_falls_back_to_us(self):
        cfg = get_market_config("")
        assert cfg is MARKET_CONFIGS["us"]


class TestApplyOverride:

    def test_no_overrides_returns_base(self):
        cfg = apply_override("us")
        assert cfg is MARKET_CONFIGS["us"]

    def test_none_overrides_returns_base(self):
        cfg = apply_override("us", None)
        assert cfg is MARKET_CONFIGS["us"]

    def test_empty_overrides_returns_base(self):
        cfg = apply_override("us", {})
        assert cfg is MARKET_CONFIGS["us"]

    def test_valid_override_applied(self):
        cfg = apply_override("us", {"nan_threshold": 0.5})
        assert cfg.nan_threshold == 0.5
        assert cfg.use_temporal_sort is False  # unchanged

    def test_override_returns_new_instance(self):
        original = MARKET_CONFIGS["us"]
        overridden = apply_override("us", {"ffill_limit": 30})
        assert overridden is not original
        assert overridden.ffill_limit == 30
        assert original.ffill_limit == 45  # original unchanged

    def test_unknown_fields_ignored(self):
        cfg = apply_override("us", {"nonexistent_field": 42, "nan_threshold": 0.6})
        assert cfg.nan_threshold == 0.6
        assert not hasattr(cfg, "nonexistent_field")

    def test_all_unknown_fields_returns_base(self):
        cfg = apply_override("cn", {"bad_field": 1})
        assert cfg is MARKET_CONFIGS["cn"]

    def test_multiple_overrides(self):
        cfg = apply_override("hk", {
            "nan_threshold": 0.80,
            "ffill_limit": 60,
            "num_boost_round": 800,
        })
        assert cfg.nan_threshold == 0.80
        assert cfg.ffill_limit == 60
        assert cfg.num_boost_round == 800

    def test_override_on_unknown_market(self):
        cfg = apply_override("jp", {"nan_threshold": 0.5})
        assert cfg.nan_threshold == 0.5
        assert cfg.use_temporal_sort is False  # from US fallback
