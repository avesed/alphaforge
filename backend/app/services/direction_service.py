"""Binary direction prediction service -- LightGBM classification + calibration.

Trains a separate binary classification model that predicts whether each stock
will go up or down over the forward horizon, outputting a calibrated probability.

This complements the ranking model in prediction_service.py:
- Ranking model: optimises stock ordering (lambdarank objective).
- Direction model: predicts binary up/down probability (binary objective).

AlphaForge adaptation:
- Prediction/model read/write operations go through local PredictionStore.
- Market data (prices, fundamentals) still via StockPulseAsyncClient.
- Model files remain on local disk (joblib.dump/load).
"""

import asyncio
import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from app.config import get_settings
from app.services.feature_service import (
    ALPHA158_FEATURES,
    FUNDAMENTAL_FEATURES,
    SENTIMENT_FEATURES,
    feature_service,
)
from app.services.market_config import MarketConfig, get_market_config
from app.services.market_features_service import (
    MARKET_FEATURE_COLUMNS,
    build_market_features,
)
from app.services.prediction_store import prediction_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRAIN_LOOKBACK_DAYS = 730
_MIN_TRAIN_DATES = 60
_MIN_SYMBOLS_PER_DATE = 25


def _direction_model_filename(forward_days: int) -> str:
    """Per-horizon direction model artifact filename.

    5d keeps the legacy ``direction_model.pkl`` so existing single-horizon
    directories remain valid; other horizons use ``direction_model.{fwd}d.pkl``.
    """
    return (
        "direction_model.pkl" if forward_days == 5
        else f"direction_model.{forward_days}d.pkl"
    )


def _direction_calibrator_filename(forward_days: int) -> str:
    return "calibrator.pkl" if forward_days == 5 else f"calibrator.{forward_days}d.pkl"


def _direction_features_filename(forward_days: int) -> str:
    return (
        "direction_features.json" if forward_days == 5
        else f"direction_features.{forward_days}d.json"
    )

_ENSEMBLE_SEEDS: list[int] = [42, 137, 271, 419, 503, 631, 769, 887, 953, 1031]

_DIRECTION_LGB_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "is_unbalance": True,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}

_ISOTONIC_MIN_SAMPLES = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _numpy_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        import math
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _get_direction_lgb_params(
    market: str, cfg: MarketConfig | None = None,
) -> dict[str, Any]:
    params = dict(_DIRECTION_LGB_PARAMS)
    resolved = cfg or get_market_config(market)
    for key in ("learning_rate", "num_leaves", "min_child_samples", "lambda_l2"):
        if key in resolved.lgb_overrides:
            params[key] = resolved.lgb_overrides[key]
    params.update(resolved.direction_lgb_overrides)
    return params


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def train_and_predict_direction(
    market: str,
    forward_days: int = 5,
    force_retrain: bool = False,
    prediction_date: Optional[date] = None,
) -> Optional[dict]:
    """Full direction model pipeline: train (if needed) + inference.

    Returns a summary dict on success, None on failure.
    The function never raises -- direction model failure should not
    block ranking predictions.
    """
    try:
        return await _direction_pipeline(
            market, forward_days, force_retrain, prediction_date,
        )
    except Exception as e:
        logger.error(
            "Direction model pipeline failed for market=%s (non-fatal): %s",
            market, e, exc_info=True,
        )
        return None


async def _direction_pipeline(
    market: str,
    forward_days: int,
    force_retrain: bool,
    prediction_date: Optional[date] = None,
) -> dict:
    """Internal direction pipeline -- may raise on failure."""
    settings = get_settings()
    binding = settings.QUALITY_GATE_BINDING
    cfg = get_market_config(market)
    today = prediction_date or date.today()

    logger.info(
        "Direction model pipeline start: market=%s, forward_days=%d",
        market, forward_days,
    )

    # Step 1: Check for existing direction model (per-horizon artifact)
    model_dir = _get_model_dir(market, today)
    direction_model_path = os.path.join(
        model_dir, _direction_model_filename(forward_days),
    )

    trained_this_run = False
    # quality_passed reflects today's candidate model (or reused on-disk one).
    quality_passed = True
    auc_score = 0.0
    brier = 0.0

    if not force_retrain and os.path.exists(direction_model_path):
        logger.info(
            "Existing direction model found at %s, skipping training",
            direction_model_path,
        )
        if binding:
            # Resolve the reused model's quality so a rejected on-disk model
            # is subject to the serving policy below.
            existing_quality = await prediction_store.get_model_quality(
                market, today, forward_days, "direction",
            )
            if existing_quality == "rejected":
                quality_passed = False
    else:
        # Step 2: Train the direction model
        train_result = await _train_direction_model(
            market, today, forward_days, cfg,
        )
        trained_this_run = True
        quality_passed = train_result["quality_passed"]
        auc_score = train_result["auc"]
        brier = train_result["brier_score"]

        if not quality_passed:
            logger.warning(
                "Direction model quality gate FAILED: AUC=%.4f (min=%.4f), "
                "Brier=%.4f (max=%.4f)",
                auc_score, cfg.direction_min_auc,
                brier, cfg.direction_max_brier,
            )

    # Step 3: Decide which model serves (decision 5).
    #   - gate passed (or binding off): use today's model, normal confidence.
    #   - rejected + binding + prior approved exists: serve prior approved.
    #   - rejected + binding + no approved ever: serve today tagged low_conf.
    serving_quality = "approved"
    use_approved_only = False
    exclude_today: Optional[date] = None

    if binding and not quality_passed:
        approved = await prediction_store.get_latest_approved_model(
            market, model_type="direction", forward_days=forward_days,
            before_date=today,
        )
        if approved is not None:
            # Load the prior approved direction model (never today's rejected).
            use_approved_only = True
            exclude_today = today
            serving_quality = "approved"
            logger.warning(
                "Direction gate failed for %s/%s -- serving prior approved "
                "direction model (date=%s)",
                market, today.isoformat(), approved.get("model_date"),
            )
        else:
            # No approved direction model ever -> serve today tagged low_conf.
            serving_quality = "rejected"
            logger.warning(
                "Direction gate failed for %s/%s and no prior approved model "
                "-- serving today's rejected direction model (low_confidence)",
                market, today.isoformat(),
            )

    # Step 4: Inference using the resolved serving model.
    n_updated = await _run_direction_inference(
        market, today, forward_days, cfg,
        approved_only=use_approved_only,
        exclude_date=exclude_today,
    )

    summary = {
        "market": market,
        "trained": trained_this_run,
        "quality_passed": quality_passed,
        "serving_quality": serving_quality,
        "low_confidence": serving_quality == "rejected",
        "auc": auc_score if trained_this_run else None,
        "brier_score": brier if trained_this_run else None,
        "predictions_updated": n_updated,
    }

    logger.info(
        "Direction model pipeline completed: market=%s, trained=%s, "
        "quality_passed=%s, serving_quality=%s, predictions_updated=%d",
        market, trained_this_run, quality_passed, serving_quality, n_updated,
    )

    return summary


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


async def _train_direction_model(
    market: str,
    model_date: date,
    forward_days: int,
    cfg: MarketConfig,
    persist: bool = True,
) -> dict:
    """Train a binary direction model with walk-forward validation."""
    settings = get_settings()
    ensemble_size = settings.ENSEMBLE_SIZE
    n_folds = settings.WALKFORWARD_FOLDS

    train_end_date = model_date - timedelta(days=forward_days)
    train_start_date = model_date - timedelta(days=_TRAIN_LOOKBACK_DAYS)

    train_start_str = train_start_date.isoformat()
    train_end_str = model_date.isoformat()

    # Step 1: Build per-stock feature matrix
    symbols = await _resolve_symbols(market)
    feature_df = await feature_service.build_feature_matrix(
        market=market,
        symbols=symbols,
        start_date=train_start_str,
        end_date=train_end_str,
    )

    if feature_df.empty:
        raise RuntimeError(
            f"Feature matrix is empty for direction model (market={market})"
        )

    feature_df["date"] = pd.to_datetime(feature_df["date"])

    # Step 2: Build market-level features and merge
    market_features_df = await build_market_features(
        market=market,
        start_date=train_start_date,
        end_date=model_date,
    )

    if not market_features_df.empty:
        market_features_df["date"] = pd.to_datetime(market_features_df["date"])
        feature_df = feature_df.merge(
            market_features_df, on="date", how="left",
        )
        logger.info(
            "Merged %d market features into direction training data",
            len(MARKET_FEATURE_COLUMNS),
        )
    else:
        for col in MARKET_FEATURE_COLUMNS:
            feature_df[col] = np.nan

    # Step 2b: Stacking -- ranking model scores as a feature
    try:
        from app.services.prediction_service import prediction_service
        ranking_history = await prediction_service.get_prediction_history(
            market=market, days=_TRAIN_LOOKBACK_DAYS,
        )
        if ranking_history:
            ranking_df = pd.DataFrame(ranking_history)
            if "prediction_date" in ranking_df.columns:
                ranking_df.rename(columns={"prediction_date": "date"}, inplace=True)
            if "predicted_score" in ranking_df.columns:
                ranking_df.rename(columns={"predicted_score": "ranking_score"}, inplace=True)
            if "date" in ranking_df.columns and "ranking_score" in ranking_df.columns:
                ranking_df["date"] = pd.to_datetime(ranking_df["date"])
                ranking_df = ranking_df[["symbol", "date", "ranking_score"]].drop_duplicates(
                    subset=["symbol", "date"],
                )
                n_before = len(feature_df)
                feature_df = feature_df.merge(ranking_df, on=["symbol", "date"], how="left")
                n_matched = int(feature_df["ranking_score"].notna().sum())
                coverage = n_matched / n_before if n_before > 0 else 0
                logger.info(
                    "Stacking: merged ranking scores, %d/%d rows matched (%.1f%%)",
                    n_matched, n_before, coverage * 100,
                )
                if coverage < 0.05:
                    feature_df.drop(columns=["ranking_score"], inplace=True)
    except Exception as e:
        logger.warning("Stacking: failed to fetch ranking scores: %s", e)

    # Step 3: Fetch close prices for label computation
    close_df = await _fetch_close_prices(market, train_start_str, train_end_str)

    if close_df.empty:
        raise RuntimeError("Close price data is empty for direction model")

    close_df["date"] = pd.to_datetime(close_df["date"])

    df = feature_df.merge(
        close_df[["symbol", "date", "close"]],
        on=["symbol", "date"],
        how="left",
    )

    # Step 4: Compute forward returns and binary labels
    df = df.sort_values(["symbol", "date"])
    df["forward_return"] = df.groupby("symbol")["close"].transform(
        lambda x: x.shift(-forward_days) / x - 1
    )
    df["forward_return"] = df.groupby("date")["forward_return"].transform(
        lambda x: x.clip(x.quantile(0.01), x.quantile(0.99))
    )

    df = df.dropna(subset=["forward_return"])
    df["label"] = (df["forward_return"] > 0).astype(float)

    date_counts = df.groupby("date")["symbol"].nunique()
    valid_dates = date_counts[date_counts >= _MIN_SYMBOLS_PER_DATE].index
    df = df[df["date"].isin(valid_dates)]

    if len(df) < _MIN_TRAIN_DATES * _MIN_SYMBOLS_PER_DATE:
        raise RuntimeError(f"Insufficient labeled data for direction model: {len(df)} rows")

    class_balance = df["label"].mean()
    logger.info(
        "Direction labels: %d rows, %.1f%% positive, %.1f%% negative",
        len(df), class_balance * 100, (1 - class_balance) * 100,
    )

    # Step 5: Determine feature columns
    meta_cols = {"symbol", "date", "close", "forward_return", "label"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # Step 6: Walk-forward training with ensemble
    unique_dates = sorted(df["date"].unique())
    sort_cols = ["symbol", "date"] if cfg.use_temporal_sort else ["date", "symbol"]

    splits = _walk_forward_splits(unique_dates, n_folds=n_folds, forward_days=forward_days)
    if not splits:
        raise RuntimeError("Could not generate walk-forward splits for direction model")

    fold_aucs: list[float] = []
    fold_briers: list[float] = []
    final_models: list[lgb.Booster] = []
    final_val_df: pd.DataFrame = pd.DataFrame()
    final_val_probs: np.ndarray = np.array([])
    final_train_dates: list = []
    final_val_dates: list = []

    for fold_idx, (tr_dates, va_dates) in enumerate(splits):
        is_final_fold = fold_idx == len(splits) - 1

        tr_mask = df["date"].isin(tr_dates)
        va_mask = df["date"].isin(va_dates)
        tr_df = df[tr_mask].copy().sort_values(sort_cols).reset_index(drop=True)
        va_df = df[va_mask].copy().sort_values(sort_cols).reset_index(drop=True)

        X_tr = tr_df[feature_cols].values
        y_tr = tr_df["label"].values
        X_va = va_df[feature_cols].values
        y_va = va_df["label"].values

        tr_set = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
        va_set = lgb.Dataset(X_va, label=y_va, feature_name=feature_cols, reference=tr_set)

        models = await asyncio.to_thread(
            _train_direction_ensemble_sync,
            tr_set, va_set, market, ensemble_size, cfg,
        )

        va_probs = np.mean([m.predict(X_va) for m in models], axis=0)

        try:
            fold_auc = float(roc_auc_score(y_va, va_probs))
        except ValueError:
            fold_auc = 0.5
        fold_brier = float(brier_score_loss(y_va, va_probs))

        fold_aucs.append(fold_auc)
        fold_briers.append(fold_brier)

        logger.info("  Fold %d AUC=%.4f, Brier=%.4f", fold_idx + 1, fold_auc, fold_brier)

        if is_final_fold:
            final_models = models
            final_val_df = va_df
            final_val_probs = va_probs
            final_train_dates = list(tr_dates)
            final_val_dates = list(va_dates)

    # Step 7: Calibration
    y_val_final = final_val_df["label"].values
    calibrator, calibration_method = _fit_calibrator(final_val_probs, y_val_final)

    calibrated_probs = _apply_calibrator(calibrator, final_val_probs)
    try:
        calibrated_auc = float(roc_auc_score(y_val_final, calibrated_probs))
    except ValueError:
        calibrated_auc = 0.5
    calibrated_brier = float(brier_score_loss(y_val_final, calibrated_probs))

    mean_auc = float(np.mean(fold_aucs))
    mean_brier = float(np.mean(fold_briers))

    logger.info(
        "Direction walk-forward summary: mean_AUC=%.4f, mean_Brier=%.4f, "
        "calibrated_AUC=%.4f, calibrated_Brier=%.4f, method=%s",
        mean_auc, mean_brier, calibrated_auc, calibrated_brier, calibration_method,
    )

    quality_passed = (
        mean_auc > cfg.direction_min_auc
        and mean_brier < cfg.direction_max_brier
    )

    # Step 8-9: Save model + record
    model_dir = _get_model_dir(market, model_date)
    model_id = None

    if persist:
        from app.services.prediction_service import _write_quality_marker

        _save_direction_model(
            final_models, calibrator, feature_cols, model_dir,
            forward_days=forward_days,
        )

        # Rewrite the marker now that the gate has decided (DB authoritative).
        _write_quality_marker(
            model_dir, "direction", forward_days,
            "approved" if quality_passed else "rejected",
        )

        model_id = await _record_direction_model(
            market=market,
            model_date=model_date,
            train_start=pd.Timestamp(final_train_dates[0]).date(),
            train_end=pd.Timestamp(final_train_dates[-1]).date(),
            val_start=pd.Timestamp(final_val_dates[0]).date(),
            val_end=pd.Timestamp(final_val_dates[-1]).date(),
            forward_days=forward_days,
            feature_count=len(feature_cols),
            symbol_count=final_val_df["symbol"].nunique(),
            auc=mean_auc,
            brier_score=mean_brier,
            model_path=os.path.join(
                model_dir, _direction_model_filename(forward_days),
            ),
            quality_passed=quality_passed,
            extra_metadata={
                "ensemble_size": ensemble_size,
                "walkforward_folds": len(splits),
                "fold_aucs": [round(a, 6) for a in fold_aucs],
                "fold_briers": [round(b, 6) for b in fold_briers],
                "calibrated_auc": round(calibrated_auc, 6),
                "calibrated_brier": round(calibrated_brier, 6),
                "calibration_method": calibration_method,
                "class_balance": round(class_balance, 4),
                "has_market_features": not market_features_df.empty,
            },
        )

    return {
        "quality_passed": quality_passed,
        "auc": mean_auc,
        "brier_score": mean_brier,
        "calibrated_auc": calibrated_auc,
        "calibrated_brier": calibrated_brier,
        "calibration_method": calibration_method,
        "model_path": model_dir,
        "model_id": str(model_id) if model_id else None,
        "feature_count": len(feature_cols),
        "symbol_count": final_val_df["symbol"].nunique(),
        "ensemble_size": ensemble_size,
        "models": final_models,
        "feature_cols": feature_cols,
        "calibrator": calibrator,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


async def _run_direction_inference(
    market: str,
    prediction_date: date,
    forward_days: int,
    cfg: MarketConfig,
    approved_only: bool = False,
    exclude_date: Optional[date] = None,
) -> int:
    """Run direction model inference and update up_probability via StockPulse.

    When ``approved_only`` is set (serving policy fallback after a rejected
    gate), the model loader skips non-approved and ``exclude_date`` models so
    a just-rejected model is never served over a prior approved one
    (decision 5).
    """
    model_dir, models, calibrator, feature_cols = await _load_direction_model(
        market, prediction_date, forward_days=forward_days,
        approved_only=approved_only, exclude_date=exclude_date,
    )

    if models is None:
        logger.warning(
            "No direction model found for market=%s (approved_only=%s)",
            market, approved_only,
        )
        return 0

    logger.info(
        "Direction inference: market=%s, %d ensemble members, %d features",
        market, len(models), len(feature_cols),
    )

    # Build per-stock features
    symbols = await _resolve_symbols(market)
    inference_end = prediction_date.isoformat()
    inference_start = (prediction_date - timedelta(days=90)).isoformat()

    inference_df = await feature_service.build_feature_matrix(
        market=market, symbols=symbols,
        start_date=inference_start, end_date=inference_end,
    )

    if inference_df.empty:
        logger.warning("Inference feature matrix is empty for direction model")
        return 0

    inference_df["date"] = pd.to_datetime(inference_df["date"])

    # Merge market features
    market_features_df = await build_market_features(
        market=market,
        start_date=prediction_date - timedelta(days=90),
        end_date=prediction_date,
    )

    if not market_features_df.empty:
        market_features_df["date"] = pd.to_datetime(market_features_df["date"])
        inference_df = inference_df.merge(market_features_df, on="date", how="left")
    else:
        for col in MARKET_FEATURE_COLUMNS:
            if col not in inference_df.columns:
                inference_df[col] = np.nan

    # Stacking: merge latest ranking scores
    try:
        from app.services.prediction_service import prediction_service
        ranking_latest = await prediction_service.get_latest_predictions(market=market, top_n=500)
        if ranking_latest:
            rdf = pd.DataFrame(ranking_latest)
            if "predicted_score" in rdf.columns:
                rdf.rename(columns={"predicted_score": "ranking_score"}, inplace=True)
            if "ranking_score" in rdf.columns and "symbol" in rdf.columns:
                rdf = rdf[["symbol", "ranking_score"]].drop_duplicates(subset=["symbol"])
                inference_df = inference_df.merge(rdf, on="symbol", how="left")
    except Exception as e:
        logger.warning("Stacking inference: failed: %s", e)

    # Pick latest date with adequate coverage
    settings = get_settings()
    date_symbol_counts = inference_df.groupby("date")["symbol"].nunique().sort_index()
    if date_symbol_counts.empty:
        return 0

    max_date = date_symbol_counts.index.max()
    max_date_count = date_symbol_counts.loc[max_date]
    total_symbols = inference_df["symbol"].nunique()
    min_coverage = settings.INFERENCE_MIN_COVERAGE

    if max_date_count >= total_symbols * min_coverage:
        latest_date = max_date
    else:
        threshold = total_symbols * min_coverage
        candidates = date_symbol_counts[date_symbol_counts >= threshold]
        if candidates.empty:
            logger.warning("Direction inference: insufficient coverage")
            return 0
        latest_date = candidates.index.max()

    latest_df = inference_df[inference_df["date"] == latest_date].copy()

    # Align feature columns
    missing_features = [c for c in feature_cols if c not in latest_df.columns]
    if missing_features:
        for col in missing_features:
            latest_df[col] = np.nan

    X_inference = latest_df[feature_cols].values

    # Predict
    def _ensemble_predict() -> np.ndarray:
        return np.mean([m.predict(X_inference) for m in models], axis=0)

    raw_probs = await asyncio.to_thread(_ensemble_predict)
    raw_mean = float(np.mean(raw_probs))
    raw_std = float(np.std(raw_probs))

    # Calibrate
    if calibrator is not None:
        calibrated_mean = float(_apply_calibrator(calibrator, np.array([raw_mean]))[0])
    else:
        calibrated_mean = raw_mean

    if raw_std < 0.02 and len(raw_probs) > 10:
        from scipy.stats import rankdata
        n = len(raw_probs)
        ranks = rankdata(raw_probs, method="ordinal")
        normalised = (ranks - 1) / max(n - 1, 1)
        half_spread = 0.15
        calibrated_probs = calibrated_mean + half_spread * (2.0 * normalised - 1.0)
        calibrated_probs = np.clip(calibrated_probs, 0.05, 0.95)
    elif calibrator is not None:
        calibrated_probs = _apply_calibrator(calibrator, raw_probs)
    else:
        calibrated_probs = raw_probs

    # Update up_probability via StockPulse API
    n_updated = await _update_up_probabilities(
        market=market,
        prediction_date=prediction_date,
        forward_days=forward_days,
        symbols=latest_df["symbol"].values,
        up_probabilities=calibrated_probs,
    )

    logger.info(
        "Direction inference complete: market=%s, %d predictions updated, "
        "mean up_probability=%.4f",
        market, n_updated, float(np.mean(calibrated_probs)),
    )

    return n_updated


# ---------------------------------------------------------------------------
# Ensemble training (synchronous, runs via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _train_direction_ensemble_sync(
    train_set: lgb.Dataset,
    val_set: lgb.Dataset,
    market: str = "us",
    ensemble_size: int = 5,
    cfg: MarketConfig | None = None,
) -> list[lgb.Booster]:
    resolved = cfg or get_market_config(market)
    seeds = _ENSEMBLE_SEEDS[:ensemble_size]
    models: list[lgb.Booster] = []

    num_boost_round = resolved.num_boost_round
    early_stopping = resolved.early_stopping_rounds

    for i, seed in enumerate(seeds):
        params = _get_direction_lgb_params(market, resolved)
        params["seed"] = seed
        params["feature_fraction_seed"] = seed
        params["bagging_seed"] = seed

        callbacks = [lgb.early_stopping(early_stopping), lgb.log_evaluation(period=50)]

        model = lgb.train(
            params, train_set,
            valid_sets=[val_set], valid_names=["valid_0"],
            num_boost_round=num_boost_round, callbacks=callbacks,
        )
        models.append(model)

    return models


# ---------------------------------------------------------------------------
# Walk-forward splits
# ---------------------------------------------------------------------------


def _walk_forward_splits(
    unique_dates: list, n_folds: int, forward_days: int,
) -> list[tuple[list, list]]:
    total = len(unique_dates)
    if n_folds <= 1:
        split_idx = int(total * 0.8)
        if split_idx < _MIN_TRAIN_DATES:
            return []
        val_start = min(split_idx + forward_days, total - 1)
        train_dates = unique_dates[:split_idx]
        val_dates = unique_dates[val_start:]
        if len(val_dates) < 5:
            return []
        return [(train_dates, val_dates)]

    val_size = max(total // (n_folds + 2), 10)
    splits: list[tuple[list, list]] = []

    for i in range(n_folds):
        val_end_idx = total - (n_folds - 1 - i) * val_size
        val_start_idx = val_end_idx - val_size
        train_end_idx = val_start_idx - forward_days

        if train_end_idx < _MIN_TRAIN_DATES:
            continue
        if val_start_idx < 0 or val_end_idx > total:
            continue

        train_dates = unique_dates[:train_end_idx]
        val_dates = unique_dates[val_start_idx:val_end_idx]

        if len(val_dates) < 5:
            continue
        splits.append((train_dates, val_dates))

    return splits


# ---------------------------------------------------------------------------
# Probability calibration
# ---------------------------------------------------------------------------


def _fit_calibrator(
    predicted_probs: np.ndarray,
    true_labels: np.ndarray,
) -> tuple[Any, str]:
    n_samples = len(true_labels)

    if n_samples >= _ISOTONIC_MIN_SAMPLES:
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(predicted_probs, true_labels)
        method = "isotonic"
    else:
        calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        calibrator.fit(predicted_probs.reshape(-1, 1), true_labels.astype(int))
        method = "platt"

    return calibrator, method


def _apply_calibrator(calibrator: Any, raw_probs: np.ndarray) -> np.ndarray:
    if isinstance(calibrator, IsotonicRegression):
        calibrated = calibrator.predict(raw_probs)
    elif isinstance(calibrator, LogisticRegression):
        calibrated = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    else:
        return raw_probs
    return np.clip(calibrated, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------


def _get_model_dir(market: str, model_date: date) -> str:
    settings = get_settings()
    date_str = model_date.strftime("%Y%m%d")
    model_dir = str(Path(settings.PREDICTION_DATA_DIR) / market / date_str)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def _save_direction_model(
    models: list[lgb.Booster],
    calibrator: Any,
    feature_cols: list[str],
    model_dir: str,
    forward_days: int = 5,
) -> None:
    from app.services.prediction_service import _write_quality_marker

    model_path = os.path.join(model_dir, _direction_model_filename(forward_days))
    joblib.dump(models, model_path)

    calibrator_path = os.path.join(
        model_dir, _direction_calibrator_filename(forward_days),
    )
    joblib.dump(calibrator, calibrator_path)

    features_meta = {
        "features": feature_cols,
        "count": len(feature_cols),
        "ensemble_size": len(models),
    }
    features_path = os.path.join(
        model_dir, _direction_features_filename(forward_days),
    )
    with open(features_path, "w") as f:
        json.dump(features_meta, f, default=_numpy_default)

    # Quality marker defaults to "pending"; rewritten after the gate. The
    # direction marker (quality.direction.{fwd}.json) is independent of the
    # ranking marker and never overwrites it.
    _write_quality_marker(model_dir, "direction", forward_days, "pending")


async def _load_direction_model(
    market: str,
    target_date: date,
    forward_days: int = 5,
    approved_only: bool = False,
    exclude_date: Optional[date] = None,
) -> tuple[Optional[str], Optional[list[lgb.Booster]], Any, list[str]]:
    """Load a direction model, scanning back up to 31 days.

    When ``approved_only`` is True (Batch A serving policy), only approved
    models are loaded: quality is read from the on-disk marker first, then
    the DB (decision 3) for marker-less legacy directories. ``exclude_date``
    skips a specific model date (e.g. today's just-rejected model).
    """
    from app.services.prediction_service import _read_quality_marker

    settings = get_settings()
    base_dir = Path(settings.PREDICTION_DATA_DIR) / market
    exclude_str = exclude_date.strftime("%Y%m%d") if exclude_date is not None else None

    for days_back in range(31):
        check_date = target_date - timedelta(days=days_back)
        date_str = check_date.strftime("%Y%m%d")
        if exclude_str is not None and date_str == exclude_str:
            continue
        model_dir = str(base_dir / date_str)
        model_path = os.path.join(
            model_dir, _direction_model_filename(forward_days),
        )

        if not os.path.exists(model_path):
            continue

        if approved_only:
            quality = _read_quality_marker(model_dir, "direction", forward_days)
            if quality not in ("approved", "rejected"):
                try:
                    quality = await prediction_store.get_model_quality(
                        market, check_date, forward_days, "direction",
                    )
                except Exception as e:
                    logger.warning(
                        "DB direction quality lookup failed for %s/%s: %s",
                        market, date_str, e,
                    )
                    quality = None
            if quality != "approved":
                continue

        try:
            loaded = await asyncio.to_thread(joblib.load, model_path)
            models = loaded if isinstance(loaded, list) else [loaded]

            calibrator_path = os.path.join(
                model_dir, _direction_calibrator_filename(forward_days),
            )
            calibrator = None
            if os.path.exists(calibrator_path):
                calibrator = await asyncio.to_thread(joblib.load, calibrator_path)

            features_path = os.path.join(
                model_dir, _direction_features_filename(forward_days),
            )
            feature_cols: list[str] = []
            if os.path.exists(features_path):
                def _read():
                    with open(features_path) as f:
                        return json.load(f)
                meta = await asyncio.to_thread(_read)
                feature_cols = meta.get("features", [])
            else:
                feature_cols = feature_service.get_feature_names()
                feature_cols.extend(MARKET_FEATURE_COLUMNS)

            logger.info(
                "Loaded direction model from %s: %d members, %d features",
                model_path, len(models), len(feature_cols),
            )
            return model_dir, models, calibrator, feature_cols

        except Exception as e:
            logger.warning("Failed to load direction model from %s: %s", model_path, e)
            continue

    return None, None, None, []


# ---------------------------------------------------------------------------
# DB operations (via StockPulse API)
# ---------------------------------------------------------------------------


async def _record_direction_model(
    market: str,
    model_date: date,
    train_start: date,
    train_end: date,
    val_start: date,
    val_end: date,
    forward_days: int,
    feature_count: int,
    symbol_count: int,
    auc: float,
    brier_score: float,
    model_path: str,
    quality_passed: bool,
    extra_metadata: dict[str, Any] | None = None,
) -> Any:
    """Write direction model metadata via local PredictionStore."""
    import hashlib

    metadata: dict[str, Any] = {"model_type": "direction"}
    if extra_metadata:
        metadata.update(extra_metadata)

    # Generate a deterministic model ID
    id_seed = f"{market}:{model_date.isoformat()}:{forward_days}:direction"
    model_hash = hashlib.sha256(id_seed.encode()).hexdigest()[:32]

    model_data = {
        "id": model_hash,
        "market": market,
        "model_date": model_date.isoformat(),
        "forward_days": forward_days,
        "feature_count": feature_count,
        "symbol_count": symbol_count,
        "ensemble_size": extra_metadata.get("ensemble_size", 1) if extra_metadata else 1,
        "ic": None,
        "icir": None,
        "ndcg": None,
        "file_path": model_path,
        "training_config": json.dumps(metadata, default=_numpy_default),
        "quality": "approved" if quality_passed else "rejected",
        "model_type": "direction",
    }

    try:
        model_id = await prediction_store.write_model(model_data)
        return model_id
    except Exception as e:
        logger.error("Failed to record direction model: %s", e)
        raise RuntimeError(f"Direction model recording failed: {e}") from e


async def _update_up_probabilities(
    market: str,
    prediction_date: date,
    forward_days: int,
    symbols: np.ndarray,
    up_probabilities: np.ndarray,
) -> int:
    """Batch update up_probability via local PredictionStore."""
    updates = [
        {"symbol": str(symbol), "up_probability": float(prob)}
        for symbol, prob in zip(symbols, up_probabilities)
    ]

    try:
        n_updated = await prediction_store.update_up_probabilities(
            market=market,
            prediction_date=prediction_date.isoformat(),
            forward_days=forward_days,
            updates=updates,
        )
        logger.info(
            "Updated %d up_probability values: market=%s, date=%s",
            n_updated, market, prediction_date.isoformat(),
        )
        return n_updated
    except Exception as e:
        logger.error("Failed to update up_probability: %s", e)
        return 0


# ---------------------------------------------------------------------------
# Symbol resolution (reuse from prediction_service)
# ---------------------------------------------------------------------------


async def _resolve_symbols(market: str) -> list[str]:
    from app.services.prediction_service import prediction_service
    return await prediction_service._resolve_symbols(market)


# ---------------------------------------------------------------------------
# Close price fetch (reuse from prediction_service)
# ---------------------------------------------------------------------------


async def _fetch_close_prices(
    market: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    from app.services.prediction_service import prediction_service
    symbols = await _resolve_symbols(market)
    return await prediction_service._fetch_close_prices(
        market, symbols, start_date, end_date,
    )
