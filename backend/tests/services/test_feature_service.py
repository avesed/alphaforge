"""Tests for feature_service.py pure/static parts.

Validates feature column definitions, counts, and name helpers
without requiring the full ML pipeline or external data sources.
"""

from __future__ import annotations

import pytest

from app.services.feature_service import (
    ALPHA158_FEATURES,
    ANALYST_FEATURES,
    EARNINGS_CALENDAR_FEATURES,
    EARNINGS_FEATURES,
    FUNDAMENTAL_FEATURES,
    INSIDER_FEATURES,
    INTERACTION_FEATURES,
    OPTIONS_FEATURES,
    SENTIMENT_FEATURES,
    FeatureService,
)


class TestFeatureColumnDefinitions:

    def test_alpha158_feature_count(self):
        """Alpha158 features should have 86 columns (from factor_service.FEATURE_NAMES)."""
        assert len(ALPHA158_FEATURES) == 86

    def test_fundamental_feature_count(self):
        """Fundamental features should have 18 columns."""
        assert len(FUNDAMENTAL_FEATURES) == 18

    def test_sentiment_feature_count(self):
        """Sentiment features should have 11 columns."""
        assert len(SENTIMENT_FEATURES) == 11

    def test_analyst_feature_count(self):
        """Analyst features should have 5 columns."""
        assert len(ANALYST_FEATURES) == 5

    def test_no_duplicate_feature_names(self):
        """All feature lists should have unique names within each list."""
        for name, features in [
            ("ALPHA158", ALPHA158_FEATURES),
            ("FUNDAMENTAL", FUNDAMENTAL_FEATURES),
            ("SENTIMENT", SENTIMENT_FEATURES),
            ("ANALYST", ANALYST_FEATURES),
            ("INSIDER", INSIDER_FEATURES),
            ("OPTIONS", OPTIONS_FEATURES),
            ("INTERACTION", INTERACTION_FEATURES),
        ]:
            assert len(features) == len(set(features)), (
                f"{name} has duplicate names"
            )


class TestFeatureServiceHelpers:

    def test_get_feature_names_all(self):
        """get_feature_names with all sources includes alpha158 + fundamental + sentiment."""
        svc = FeatureService()
        names = svc.get_feature_names(
            include_fundamental=True, include_sentiment=True,
        )
        # Should include all three categories
        assert len(names) == len(ALPHA158_FEATURES) + len(FUNDAMENTAL_FEATURES) + len(SENTIMENT_FEATURES)
        # First entries should be alpha158
        assert names[0] == ALPHA158_FEATURES[0]
        # Last entries should be sentiment
        assert names[-1] == SENTIMENT_FEATURES[-1]

    def test_get_feature_names_no_fundamental(self):
        """get_feature_names without fundamentals excludes those columns."""
        svc = FeatureService()
        names = svc.get_feature_names(
            include_fundamental=False, include_sentiment=True,
        )
        assert len(names) == len(ALPHA158_FEATURES) + len(SENTIMENT_FEATURES)
        for f in FUNDAMENTAL_FEATURES:
            assert f not in names

    def test_get_feature_names_no_sentiment(self):
        """get_feature_names without sentiment excludes those columns."""
        svc = FeatureService()
        names = svc.get_feature_names(
            include_fundamental=True, include_sentiment=False,
        )
        assert len(names) == len(ALPHA158_FEATURES) + len(FUNDAMENTAL_FEATURES)
        for f in SENTIMENT_FEATURES:
            assert f not in names

    def test_get_feature_count_matches_names(self):
        """get_feature_count returns the same as len(get_feature_names)."""
        svc = FeatureService()
        for fund, sent in [(True, True), (True, False), (False, True), (False, False)]:
            count = svc.get_feature_count(include_fundamental=fund, include_sentiment=sent)
            names = svc.get_feature_names(include_fundamental=fund, include_sentiment=sent)
            assert count == len(names)

    def test_earnings_features_empty(self):
        """Earnings features are currently excluded (empty list)."""
        assert EARNINGS_FEATURES == []

    def test_earnings_calendar_has_days_to_earnings(self):
        """Earnings calendar features include days_to_earnings."""
        assert "days_to_earnings" in EARNINGS_CALENDAR_FEATURES


class TestDropSparseFeatures:

    def test_drops_high_nan_columns(self):
        """Columns with >90% NaN should be dropped."""
        import pandas as pd
        import numpy as np

        df = pd.DataFrame({
            "symbol": ["A"] * 10,
            "date": pd.date_range("2024-01-01", periods=10),
            "good_feature": range(10),
            "sparse_feature": [np.nan] * 10,  # 100% NaN
        })

        result, dropped = FeatureService._drop_sparse_features(df, max_nan_ratio=0.90)
        assert dropped == 1
        assert "sparse_feature" not in result.columns
        assert "good_feature" in result.columns

    def test_keeps_features_below_threshold(self):
        """Columns with acceptable NaN rates are retained."""
        import pandas as pd
        import numpy as np

        df = pd.DataFrame({
            "symbol": ["A"] * 10,
            "date": pd.date_range("2024-01-01", periods=10),
            "ok_feature": [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],  # 10% NaN
        })

        result, dropped = FeatureService._drop_sparse_features(df, max_nan_ratio=0.90)
        assert dropped == 0
        assert "ok_feature" in result.columns


class TestComputeFeaturePSI:

    def test_psi_identical_distributions(self):
        """PSI between identical distributions should be near zero."""
        import pandas as pd
        import numpy as np

        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, 200)
        train_df = pd.DataFrame({"feat1": data})
        infer_df = pd.DataFrame({"feat1": data})

        scores = FeatureService.compute_feature_psi(
            train_df, infer_df, ["feat1"], bins=10,
        )
        assert "feat1" in scores
        assert scores["feat1"] < 0.05  # should be very close to 0

    def test_psi_shifted_distribution(self):
        """PSI between shifted distributions should be positive."""
        import pandas as pd
        import numpy as np

        rng = np.random.RandomState(42)
        train_df = pd.DataFrame({"feat1": rng.normal(0, 1, 200)})
        infer_df = pd.DataFrame({"feat1": rng.normal(3, 1, 200)})

        scores = FeatureService.compute_feature_psi(
            train_df, infer_df, ["feat1"], bins=10,
        )
        assert scores["feat1"] > 0.1  # significant shift
