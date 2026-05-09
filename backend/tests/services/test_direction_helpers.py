"""Tests for pure helper functions in direction_service.py.

Covers _walk_forward_splits, _get_direction_lgb_params, _fit_calibrator,
_apply_calibrator, and _numpy_default.

No LightGBM training, no Qlib, no external API calls.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from app.services.direction_service import (
    _DIRECTION_LGB_PARAMS,
    _ISOTONIC_MIN_SAMPLES,
    _apply_calibrator,
    _fit_calibrator,
    _get_direction_lgb_params,
    _numpy_default,
    _walk_forward_splits,
)
from app.services.market_config import MarketConfig, get_market_config


# -----------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------


def _make_date_strings(n: int) -> list[str]:
    """Create a list of n unique date strings starting from 2023-01-01."""
    from datetime import date, timedelta
    start = date(2023, 1, 1)
    return [(start + timedelta(days=i)).isoformat() for i in range(n)]


# -----------------------------------------------------------------------
# _walk_forward_splits (module-level function)
# -----------------------------------------------------------------------


class TestDirectionWalkForwardSplits:

    def test_single_fold(self):
        dates = _make_date_strings(200)
        splits = _walk_forward_splits(dates, n_folds=1, forward_days=0)
        assert len(splits) == 1
        train, val = splits[0]
        assert len(train) == 160

    def test_too_few_dates_empty(self):
        dates = _make_date_strings(50)
        assert _walk_forward_splits(dates, n_folds=1, forward_days=0) == []

    def test_multi_fold(self):
        dates = _make_date_strings(200)
        splits = _walk_forward_splits(dates, n_folds=3, forward_days=5)
        assert len(splits) == 3

    def test_agrees_with_prediction_service(self):
        """Direction's _walk_forward_splits should produce identical results
        to PredictionService._walk_forward_splits for the same inputs."""
        from app.services.prediction_service import PredictionService

        dates = _make_date_strings(200)
        for n_folds in (1, 3):
            for fwd in (0, 5):
                dir_splits = _walk_forward_splits(dates, n_folds, fwd)
                pred_splits = PredictionService._walk_forward_splits(
                    dates, n_folds, fwd,
                )
                assert len(dir_splits) == len(pred_splits), (
                    f"Split count mismatch for n_folds={n_folds}, fwd={fwd}"
                )
                for (dt, dv), (pt, pv) in zip(dir_splits, pred_splits):
                    assert dt == pt
                    assert dv == pv


# -----------------------------------------------------------------------
# _get_direction_lgb_params
# -----------------------------------------------------------------------


class TestGetDirectionLgbParams:

    def test_us_defaults(self):
        params = _get_direction_lgb_params("us")
        assert params["objective"] == "binary"
        assert "auc" in params["metric"]
        # US direction overrides should be applied
        assert params["learning_rate"] == 0.005
        assert params["num_leaves"] == 15

    def test_base_always_binary(self):
        for market in ("us", "cn", "hk"):
            params = _get_direction_lgb_params(market)
            assert params["objective"] == "binary"

    def test_custom_cfg_overrides(self):
        cfg = MarketConfig(
            use_temporal_sort=False,
            use_sector_neutral_labels=False,
            use_balanced_quintiles=False,
            use_sector_rank=False,
            use_interactions=False,
            nan_threshold=0.75,
            ffill_limit=45,
            lgb_overrides={"learning_rate": 0.1, "num_leaves": 50},
            direction_lgb_overrides={"lambda_l2": 99.0},
        )
        params = _get_direction_lgb_params("us", cfg=cfg)
        # direction_lgb_overrides applied
        assert params["lambda_l2"] == 99.0
        # lgb_overrides keys should also be merged (learning_rate, num_leaves)
        assert params["learning_rate"] == 0.1
        assert params["num_leaves"] == 50


# -----------------------------------------------------------------------
# _fit_calibrator
# -----------------------------------------------------------------------


class TestFitCalibrator:

    def test_isotonic_for_large_sample(self):
        rng = np.random.RandomState(42)
        n = _ISOTONIC_MIN_SAMPLES + 100
        probs = rng.rand(n)
        labels = (probs > 0.5).astype(float)
        calibrator, method = _fit_calibrator(probs, labels)
        assert method == "isotonic"
        assert isinstance(calibrator, IsotonicRegression)

    def test_platt_for_small_sample(self):
        rng = np.random.RandomState(42)
        n = _ISOTONIC_MIN_SAMPLES - 1
        probs = rng.rand(n)
        labels = (probs > 0.5).astype(float)
        calibrator, method = _fit_calibrator(probs, labels)
        assert method == "platt"
        assert isinstance(calibrator, LogisticRegression)


# -----------------------------------------------------------------------
# _apply_calibrator
# -----------------------------------------------------------------------


class TestApplyCalibrator:

    def test_isotonic_output_clipped(self):
        rng = np.random.RandomState(42)
        n = 1200
        probs = rng.rand(n)
        labels = (probs > 0.5).astype(float)
        calibrator, _ = _fit_calibrator(probs, labels)
        raw = rng.rand(50)
        calibrated = _apply_calibrator(calibrator, raw)
        assert calibrated.min() >= 0.0
        assert calibrated.max() <= 1.0

    def test_platt_output_clipped(self):
        rng = np.random.RandomState(42)
        n = 500
        probs = rng.rand(n)
        labels = (probs > 0.5).astype(float)
        calibrator, _ = _fit_calibrator(probs, labels)
        raw = rng.rand(50)
        calibrated = _apply_calibrator(calibrator, raw)
        assert calibrated.min() >= 0.0
        assert calibrated.max() <= 1.0

    def test_unknown_calibrator_passthrough(self):
        raw = np.array([0.1, 0.9, 0.5])
        result = _apply_calibrator("not_a_calibrator", raw)
        np.testing.assert_array_equal(result, raw)


# -----------------------------------------------------------------------
# _numpy_default (direction_service copy)
# -----------------------------------------------------------------------


class TestDirectionNumpyDefault:

    def test_np_int64(self):
        assert _numpy_default(np.int64(7)) == 7
        assert isinstance(_numpy_default(np.int64(7)), int)

    def test_np_float64(self):
        result = _numpy_default(np.float64(2.5))
        assert result == pytest.approx(2.5)
        assert isinstance(result, float)

    def test_np_float64_nan(self):
        assert _numpy_default(np.float64("nan")) is None

    def test_np_bool(self):
        assert _numpy_default(np.bool_(True)) is True

    def test_unknown_raises(self):
        with pytest.raises(TypeError):
            _numpy_default(set())
