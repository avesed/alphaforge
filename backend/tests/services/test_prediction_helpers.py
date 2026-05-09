"""Tests for pure helper functions in prediction_service.py.

Covers _safe_round, _numpy_default, _rank_auc, _walk_forward_splits,
_compute_ic_metrics, and the _ENSEMBLE_SEEDS constant.

No LightGBM training, no Qlib, no external API calls.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.services.prediction_service import (
    PredictionService,
    _ENSEMBLE_SEEDS,
    _numpy_default,
    _rank_auc,
    _safe_round,
)


# -----------------------------------------------------------------------
# _safe_round
# -----------------------------------------------------------------------


class TestSafeRound:

    def test_nan_returns_none(self):
        assert _safe_round(float("nan"), 2) is None

    def test_inf_returns_none(self):
        assert _safe_round(float("inf"), 2) is None

    def test_neg_inf_returns_none(self):
        assert _safe_round(float("-inf"), 2) is None

    def test_normal_float_rounded(self):
        assert _safe_round(3.14159, 2) == 3.14

    def test_zero_decimals(self):
        assert _safe_round(3.7, 0) == 4.0

    def test_negative_float_rounded(self):
        assert _safe_round(-1.555, 1) == -1.6


# -----------------------------------------------------------------------
# _numpy_default
# -----------------------------------------------------------------------


class TestNumpyDefault:

    def test_np_int64_to_int(self):
        result = _numpy_default(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_np_float64_to_float(self):
        result = _numpy_default(np.float64(3.14))
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_np_float64_nan_to_none(self):
        assert _numpy_default(np.float64("nan")) is None

    def test_np_float64_inf_to_none(self):
        assert _numpy_default(np.float64("inf")) is None

    def test_np_bool_true(self):
        result = _numpy_default(np.bool_(True))
        assert result is True
        assert isinstance(result, bool)

    def test_np_bool_false(self):
        result = _numpy_default(np.bool_(False))
        assert result is False

    def test_ndarray_to_list(self):
        arr = np.array([1, 2, 3])
        result = _numpy_default(arr)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_unknown_type_raises(self):
        with pytest.raises(TypeError, match="not JSON serializable"):
            _numpy_default({"key": "value"})


# -----------------------------------------------------------------------
# _rank_auc
# -----------------------------------------------------------------------


class TestRankAuc:

    def test_perfect_ranking(self):
        """When probs perfectly separate labels, AUC should be 1.0."""
        probs = np.array([0.1, 0.2, 0.3, 0.9, 0.95, 0.99])
        labels = np.array([0, 0, 0, 1, 1, 1])
        auc = _rank_auc(probs, labels)
        assert auc == pytest.approx(1.0)

    def test_inverted_ranking(self):
        """When probs are inverted, AUC should be 0.0."""
        probs = np.array([0.9, 0.8, 0.7, 0.1, 0.05, 0.01])
        labels = np.array([0, 0, 0, 1, 1, 1])
        auc = _rank_auc(probs, labels)
        assert auc == pytest.approx(0.0)

    def test_random_ranking(self):
        """Random ranking should give AUC around 0.5."""
        rng = np.random.RandomState(42)
        n = 1000
        labels = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        probs = rng.rand(n)
        auc = _rank_auc(probs, labels)
        assert 0.4 < auc < 0.6

    def test_all_positive_returns_none(self):
        probs = np.array([0.5, 0.6, 0.7])
        labels = np.array([1, 1, 1])
        assert _rank_auc(probs, labels) is None

    def test_all_negative_returns_none(self):
        probs = np.array([0.5, 0.6, 0.7])
        labels = np.array([0, 0, 0])
        assert _rank_auc(probs, labels) is None


# -----------------------------------------------------------------------
# PredictionService._walk_forward_splits
# -----------------------------------------------------------------------


def _make_date_strings(n: int) -> list[str]:
    """Create a list of n unique date strings starting from 2023-01-01."""
    from datetime import date, timedelta
    start = date(2023, 1, 1)
    return [(start + timedelta(days=i)).isoformat() for i in range(n)]


class TestWalkForwardSplits:

    def test_single_fold_80_20(self):
        """n_folds=1 should produce an 80/20 split."""
        dates = _make_date_strings(200)
        splits = PredictionService._walk_forward_splits(dates, n_folds=1, forward_days=0)
        assert len(splits) == 1
        train, val = splits[0]
        assert len(train) == 160  # 80% of 200

    def test_single_fold_with_forward_gap(self):
        """forward_days should create a gap between train end and val start."""
        dates = _make_date_strings(200)
        splits = PredictionService._walk_forward_splits(dates, n_folds=1, forward_days=5)
        assert len(splits) == 1
        train, val = splits[0]
        assert len(train) == 160
        # val should start at index 165 (160 + 5)
        assert val[0] == dates[165]

    def test_multi_fold_returns_multiple_splits(self):
        """n_folds=3 with 200 dates should produce 3 splits."""
        dates = _make_date_strings(200)
        splits = PredictionService._walk_forward_splits(dates, n_folds=3, forward_days=5)
        assert len(splits) == 3

    def test_multi_fold_train_val_no_overlap(self):
        """Train and val sets in each fold should not overlap."""
        dates = _make_date_strings(200)
        splits = PredictionService._walk_forward_splits(dates, n_folds=3, forward_days=5)
        for train, val in splits:
            train_set = set(train)
            val_set = set(val)
            assert train_set.isdisjoint(val_set)

    def test_too_few_dates_empty(self):
        """Fewer than _MIN_TRAIN_DATES=60 should return empty list."""
        dates = _make_date_strings(50)
        splits = PredictionService._walk_forward_splits(dates, n_folds=1, forward_days=0)
        assert splits == []

    def test_multi_fold_purging_gap(self):
        """With forward_days=5, there should be a gap between train end and val start."""
        dates = _make_date_strings(200)
        splits = PredictionService._walk_forward_splits(dates, n_folds=3, forward_days=5)
        for train, val in splits:
            train_end_idx = dates.index(train[-1])
            val_start_idx = dates.index(val[0])
            # Gap should be at least forward_days
            assert val_start_idx - train_end_idx >= 5


# -----------------------------------------------------------------------
# PredictionService._compute_ic_metrics
# -----------------------------------------------------------------------


class TestComputeIcMetrics:

    @staticmethod
    def _make_ic_data(
        n_dates: int = 10,
        n_symbols: int = 20,
        corr: float = 0.0,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Build synthetic val_df, predicted_scores, actual_returns.

        corr > 0 produces positively correlated predictions.
        """
        rng = np.random.RandomState(42)
        dates = [f"2024-01-{d+1:02d}" for d in range(n_dates)]
        rows = []
        for d in dates:
            for s in range(n_symbols):
                rows.append({"date": d, "symbol": f"SYM{s:03d}"})
        val_df = pd.DataFrame(rows)
        n = len(val_df)
        noise = rng.randn(n)
        signal = rng.randn(n)
        predicted = signal
        actual = corr * signal + (1 - abs(corr)) * noise
        return val_df, predicted, actual

    def test_perfect_positive_correlation(self):
        val_df, pred, _ = self._make_ic_data(n_dates=10, n_symbols=20, corr=1.0)
        actual = pred.copy()  # perfect correlation
        ic_series, ic_mean, icir = PredictionService._compute_ic_metrics(
            val_df, pred, actual,
        )
        assert ic_mean == pytest.approx(1.0, abs=0.01)
        # Perfect constant IC -> std=0 -> ICIR=0 by convention
        assert icir == 0.0

    def test_high_but_varying_correlation(self):
        """Strong positive signal with noise gives IC > 0 and ICIR > 0."""
        val_df, pred, actual = self._make_ic_data(n_dates=10, n_symbols=20, corr=0.9)
        ic_series, ic_mean, icir = PredictionService._compute_ic_metrics(
            val_df, pred, actual,
        )
        assert ic_mean > 0.3
        assert icir > 0

    def test_random_data_low_ic(self):
        val_df, pred, actual = self._make_ic_data(n_dates=10, n_symbols=20, corr=0.0)
        ic_series, ic_mean, icir = PredictionService._compute_ic_metrics(
            val_df, pred, actual,
        )
        assert -0.3 < ic_mean < 0.3

    def test_empty_data(self):
        empty_df = pd.DataFrame(columns=["date"])
        ic_series, ic_mean, icir = PredictionService._compute_ic_metrics(
            empty_df, np.array([]), np.array([]),
        )
        assert len(ic_series) == 0
        assert ic_mean == 0.0
        assert icir == 0.0

    def test_few_samples_per_date_skipped(self):
        """Dates with < 5 samples should be excluded from IC computation."""
        val_df = pd.DataFrame({
            "date": ["2024-01-01"] * 3,  # Only 3 samples
            "symbol": ["A", "B", "C"],
        })
        pred = np.array([1.0, 2.0, 3.0])
        actual = np.array([3.0, 2.0, 1.0])
        ic_series, ic_mean, icir = PredictionService._compute_ic_metrics(
            val_df, pred, actual,
        )
        # All dates skipped -> empty result
        assert len(ic_series) == 0
        assert ic_mean == 0.0
        assert icir == 0.0

    def test_zero_ic_std_icir_zero(self):
        """When all per-date ICs are identical, std=0 -> ICIR=0.0."""
        # Create data where each date has exactly the same rank order
        n_dates, n_symbols = 5, 10
        dates = [f"2024-01-{d+1:02d}" for d in range(n_dates)]
        rows = []
        pred_list = []
        actual_list = []
        for d in dates:
            for s in range(n_symbols):
                rows.append({"date": d, "symbol": f"S{s}"})
                pred_list.append(float(s))
                actual_list.append(float(s))
        val_df = pd.DataFrame(rows)
        pred = np.array(pred_list)
        actual = np.array(actual_list)
        ic_series, ic_mean, icir = PredictionService._compute_ic_metrics(
            val_df, pred, actual,
        )
        # Each date has perfect IC=1.0, std=0 -> ICIR=0
        assert ic_mean == pytest.approx(1.0, abs=0.01)
        assert icir == 0.0


# -----------------------------------------------------------------------
# _ENSEMBLE_SEEDS
# -----------------------------------------------------------------------


class TestEnsembleSeeds:

    def test_length_is_10(self):
        assert len(_ENSEMBLE_SEEDS) == 10

    def test_all_unique(self):
        assert len(set(_ENSEMBLE_SEEDS)) == len(_ENSEMBLE_SEEDS)

    def test_first_seed_is_42(self):
        assert _ENSEMBLE_SEEDS[0] == 42
