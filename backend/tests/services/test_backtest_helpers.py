"""Tests for pure helper functions in ml_backtest_service.py.

Covers MLBacktestService._calc_progress and
MLBacktestService._compute_validation_metrics.

No LightGBM training, no Qlib, no external API calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.ml_backtest_service import MLBacktestService


# -----------------------------------------------------------------------
# _calc_progress
# -----------------------------------------------------------------------


class TestCalcProgress:

    def test_zero_progress(self):
        assert MLBacktestService._calc_progress(0, 10, 0.0) == 0.0

    def test_mid_progress(self):
        # per_iter = 100/10 = 10.  5*10 + 50*10/100 = 50 + 5 = 55.0
        assert MLBacktestService._calc_progress(5, 10, 50.0) == pytest.approx(55.0)

    def test_clamped_at_99_9(self):
        # iteration=10, max=10, phase=100 -> 10*10 + 100*10/100 = 100+10 = 110
        # but clamped to 99.9
        assert MLBacktestService._calc_progress(10, 10, 100.0) == 99.9

    def test_never_exceeds_99_9(self):
        result = MLBacktestService._calc_progress(100, 5, 100.0)
        assert result == 99.9

    def test_single_iteration(self):
        # per_iter = 100/1 = 100.  0*100 + 50*100/100 = 50
        assert MLBacktestService._calc_progress(0, 1, 50.0) == pytest.approx(50.0)


# -----------------------------------------------------------------------
# _compute_validation_metrics
# -----------------------------------------------------------------------


def _make_backtest_instance() -> MLBacktestService:
    """Create a bare MLBacktestService instance.

    _compute_validation_metrics is an instance method but never uses ``self``,
    so a bare instance is sufficient.
    """
    return object.__new__(MLBacktestService)


class TestComputeValidationMetrics:

    def test_empty_predictions(self):
        svc = _make_backtest_instance()
        result = svc._compute_validation_metrics(
            predictions_df=pd.DataFrame(),
            close_df=pd.DataFrame(),
            forward_days=5,
            val_dates=[],
        )
        assert result["val_ic"] is None
        assert result["val_icir"] is None

    def test_fewer_than_50_merged_rows(self):
        """Merged rows < 50 should return null IC."""
        svc = _make_backtest_instance()
        # 3 dates x 10 symbols = 30 rows -- below 50 threshold
        dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
        symbols = [f"S{i}" for i in range(10)]
        pred_rows = []
        close_rows = []
        for d in dates:
            for s in symbols:
                pred_rows.append(
                    {"symbol": s, "date": d, "predicted_score": np.random.rand()}
                )
                close_rows.append({"symbol": s, "date": d, "close": 100.0})

        result = svc._compute_validation_metrics(
            predictions_df=pd.DataFrame(pred_rows),
            close_df=pd.DataFrame(close_rows),
            forward_days=1,
            val_dates=dates,
        )
        assert result["val_ic"] is None
        assert result["val_icir"] is None

    def test_output_keys(self):
        """Result dict should contain all expected metric keys."""
        svc = _make_backtest_instance()
        n_dates = 8
        n_symbols = 20
        rng = np.random.RandomState(42)

        dates = [f"2024-01-{d+1:02d}" for d in range(n_dates + 5)]
        symbols = [f"SYM{i:03d}" for i in range(n_symbols)]

        pred_rows = []
        close_rows = []
        for d in dates:
            for s in symbols:
                pred_rows.append(
                    {"symbol": s, "date": d, "predicted_score": rng.rand()}
                )
                close_rows.append(
                    {"symbol": s, "date": d, "close": 100 + rng.randn() * 5}
                )

        result = svc._compute_validation_metrics(
            predictions_df=pd.DataFrame(pred_rows),
            close_df=pd.DataFrame(close_rows),
            forward_days=1,
            val_dates=dates[:n_dates],
        )
        expected_keys = {
            "val_ic",
            "val_icir",
            "val_spread",
            "val_q1_return",
            "val_q5_return",
            "val_direction_accuracy",
            "val_hit_rate",
            "val_max_drawdown",
            "ic_curve",
            "quintile_returns",
        }
        assert expected_keys.issubset(result.keys())

    def test_correlated_predictions_positive_ic(self):
        """When predicted scores correlate with future returns, IC should be positive."""
        svc = _make_backtest_instance()
        n_dates = 8
        n_symbols = 25
        rng = np.random.RandomState(42)

        # We need n_dates + forward_days dates in close_df to compute forward returns
        forward_days = 1
        all_dates = [f"2024-01-{d+1:02d}" for d in range(n_dates + forward_days + 1)]
        symbols = [f"SYM{i:03d}" for i in range(n_symbols)]

        # Build close prices where the forward return is correlated with a signal
        close_rows = []
        signal = {}
        for s in symbols:
            sig = rng.rand()
            signal[s] = sig
            base_price = 100.0
            for i, d in enumerate(all_dates):
                # Price goes up for high-signal stocks
                price = base_price + sig * (i + 1) * 2 + rng.randn() * 0.5
                close_rows.append({"symbol": s, "date": d, "close": price})

        close_df = pd.DataFrame(close_rows)

        # Predictions are the signal itself (perfectly correlated with returns direction)
        pred_dates = all_dates[:n_dates]
        pred_rows = []
        for d in pred_dates:
            for s in symbols:
                pred_rows.append(
                    {"symbol": s, "date": d, "predicted_score": signal[s]}
                )

        result = svc._compute_validation_metrics(
            predictions_df=pd.DataFrame(pred_rows),
            close_df=close_df,
            forward_days=forward_days,
            val_dates=pred_dates,
        )
        assert result["val_ic"] is not None
        assert result["val_ic"] > 0.5
