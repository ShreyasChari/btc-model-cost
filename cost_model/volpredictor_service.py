"""
Integration layer between the legacy VolPredictor script and the FastAPI app.

It executes the prediction cycle in a background thread (to avoid blocking the
event loop) and converts the resulting dataframe into dashboard-friendly stats.
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Dict

import numpy as np
import pandas as pd

from VolPredictor import VolPredictor as volpredictor_module


def _classify_regime(value: float) -> str:
    if value >= 0.9:
        return "High Stress"
    if value >= 0.7:
        return "Elevated"
    if value <= 0.45:
        return "Calm"
    return "Neutral"


def build_dashboard_dataframe(df: pd.DataFrame) -> Dict:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("VolPredictor returned invalid data")
    required_cols = {'DVOL', 'Predicted DVOL'}
    if not required_cols.issubset(df.columns):
        raise ValueError("Prediction dataframe missing DVOL columns")
    
    frame = df.copy()
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    frame = frame.dropna(subset=['DVOL', 'Predicted DVOL'])
    # Convert from numeric percent (e.g., 55) to decimals (0.55) for display
    frame['DVOL'] = frame['DVOL'] / 100.0
    frame['Predicted DVOL'] = frame['Predicted DVOL'] / 100.0
    if frame.empty:
        raise ValueError("No DVOL data available after cleaning")
    
    errors = frame['DVOL'] - frame['Predicted DVOL']
    mae = float(errors.abs().mean())
    rmse = float(np.sqrt((errors ** 2).mean()))
    bias = float(errors.mean())
    
    actual_changes = frame['DVOL'].diff().dropna()
    pred_changes = frame['Predicted DVOL'].diff().dropna()
    if len(actual_changes) and len(pred_changes):
        min_len = min(len(actual_changes), len(pred_changes))
        direction_hits = (
            (np.sign(actual_changes.tail(min_len)) == np.sign(pred_changes.tail(min_len)))
            .sum()
            / float(min_len)
        )
    else:
        direction_hits = 0.0
    
    vol_of_vol = float(frame['DVOL'].std())
    
    latest = frame.iloc[-1]
    latest_actual = float(latest['DVOL'])
    latest_predicted = float(latest['Predicted DVOL'])
    latest_error = float(latest_actual - latest_predicted)
    regime = _classify_regime(latest_actual)
    
    timeseries = [
        {
            "timestamp": idx.isoformat(),
            "actual": float(row['DVOL']),
            "predicted": float(row['Predicted DVOL']),
            "error": float(row['DVOL'] - row['Predicted DVOL'])
        }
        for idx, row in frame.iterrows()
    ]
    
    ref_idx = max(0, len(frame) - 24)
    change = (frame['DVOL'].iloc[-1] - frame['DVOL'].iloc[ref_idx]) * 100
    direction_word = "higher" if frame['DVOL'].iloc[-1] >= frame['DVOL'].iloc[ref_idx] else "lower"
    corr_value = frame['DVOL'].corr(frame['Predicted DVOL'])
    if pd.isna(corr_value):
        corr_value = 0.0
    
    insights = [
        f"DVOL is {abs(change):.1f} vol pts {direction_word} over the past day.",
        f"Model bias {'over' if bias > 0 else 'under'}-estimating realized vol by {abs(bias) * 100:.2f} vol pts.",
        f"Direction hit-rate: {direction_hits * 100:.0f}% | Regime: {regime}"
    ]
    
    dashboard = {
        "latest": {
            "timestamp": frame.index[-1].isoformat(),
            "actual": latest_actual,
            "predicted": latest_predicted,
            "forecast_next": latest_predicted,  # GBM already forecasts 1w ahead
            "error": latest_error,
        },
        "stats": {
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "direction_hit_rate": direction_hits,
            "correlation": float(corr_value),
            "vol_of_vol": vol_of_vol,
            "avg_actual": float(frame['DVOL'].mean()),
            "avg_predicted": float(frame['Predicted DVOL'].mean()),
            "regime": regime,
            "sample_size": int(len(frame))
        },
        "timeseries": timeseries,
        "insights": insights,
        "model": {
            "type": "Gradient Boosting Regressor",
            "source": "VolPredictor.py (TradingView + macro liquidity)",
            "refresh_minutes": 15
        }
    }
    return dashboard


def run_cycle(send_notifications: bool = False) -> pd.DataFrame:
    """Run the predictor once synchronously."""
    return volpredictor_module.run_cycle(send_notifications=send_notifications, save_outputs=False)


async def run_cycle_async() -> pd.DataFrame:
    """Run the predictor in a background thread to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(run_cycle, False))
