"""
Lightweight DVOL prediction analytics used by the frontend "Vol Predictor" tab.

The goal is not to be a perfect model but to provide actionable diagnostics:
- Latest predicted vs actual DVOL
- Forecast for the next bar (uses EMA style smoothing with drift)
- Backtest style statistics (MAE, RMSE, bias, hit-rate, correlation)
- Simple qualitative regime assessment + human readable insights
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import mean, pstdev
from typing import Dict, List


@dataclass
class VolPredictorConfig:
    alpha: float = 0.35  # weight for most recent observation
    window: int = 120    # number of samples to keep (hourly -> 5 days)


class VolPredictorAnalytics:
    """
    Builds a simple exponential smoothing forecast with a momentum tilt.
    The intent is to keep math transparent for traders looking at the UI.
    """

    def __init__(self, config: VolPredictorConfig | None = None):
        self.config = config or VolPredictorConfig()

    # ------------------------------------------------------------------
    def analyze(self, history: List[Dict]) -> Dict:
        """
        history: list of dicts with keys:
            - timestamp (seconds since epoch)
            - value (DVOL in decimal form, e.g. 0.65 for 65%)
        """
        if not history:
            raise ValueError("No DVOL history available")

        ordered = sorted(history, key=lambda x: x['timestamp'])
        trimmed = ordered[-self.config.window :]

        timestamps = [item['timestamp'] for item in trimmed]
        actuals = [float(item['value']) for item in trimmed]

        if len(actuals) < 5:
            raise ValueError("Insufficient DVOL history for prediction")

        predictions = self._build_predictions(actuals)
        stats = self._compute_stats(actuals, predictions)

        insight_text = self._build_insights(actuals, stats)

        timeseries = [
            {
                "timestamp": self._format_ts(ts),
                "actual": act,
                "predicted": pred,
                "error": act - pred,
            }
            for ts, act, pred in zip(timestamps, actuals, predictions)
        ]

        latest_actual = actuals[-1]
        latest_prediction = predictions[-1]
        next_forecast = self._forecast_next(actuals, predictions)

        return {
            "latest": {
                "timestamp": timeseries[-1]["timestamp"],
                "actual": latest_actual,
                "predicted": latest_prediction,
                "forecast_next": next_forecast,
                "error": latest_actual - latest_prediction,
            },
            "stats": stats,
            "timeseries": timeseries,
            "insights": insight_text,
            "model": {
                "type": "EMA + drift filter",
                "alpha": self.config.alpha,
                "window": len(actuals),
            },
        }

    # ------------------------------------------------------------------
    def _build_predictions(self, actuals: List[float]) -> List[float]:
        preds: List[float] = []
        alpha = self.config.alpha

        for idx, value in enumerate(actuals):
            if idx == 0:
                preds.append(value)
                continue

            previous_pred = preds[-1]
            momentum = (actuals[idx - 1] - actuals[idx - 2]) if idx >= 2 else 0.0

            ema_component = alpha * actuals[idx - 1] + (1 - alpha) * previous_pred
            drift_adjusted = ema_component + 0.5 * alpha * momentum
            preds.append(drift_adjusted)

        return preds

    def _compute_stats(self, actuals: List[float], preds: List[float]) -> Dict:
        errors = [act - pred for act, pred in zip(actuals[1:], preds[1:])]
        abs_errors = [abs(e) for e in errors]
        sq_errors = [e ** 2 for e in errors]

        mae = mean(abs_errors)
        rmse = (mean(sq_errors)) ** 0.5
        bias = mean(errors)

        actual_changes = [
            actuals[i] - actuals[i - 1] for i in range(1, len(actuals))
        ]
        pred_changes = [
            preds[i] - actuals[i - 1] for i in range(1, len(preds))
        ]
        direction_hits = sum(
            1 for a, p in zip(actual_changes, pred_changes) if a == 0 or (a > 0) == (p > 0)
        )
        hit_rate = direction_hits / len(actual_changes)

        corr = self._correlation(actuals[1:], preds[1:])

        vol_of_vol = pstdev(actuals) if len(actuals) > 1 else 0.0
        avg_actual = mean(actuals)
        avg_pred = mean(preds)

        regime = self._regime_label(actuals[-1])

        return {
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "direction_hit_rate": hit_rate,
            "correlation": corr,
            "vol_of_vol": vol_of_vol,
            "avg_actual": avg_actual,
            "avg_predicted": avg_pred,
            "regime": regime,
            "sample_size": len(actuals),
        }

    def _forecast_next(self, actuals: List[float], preds: List[float]) -> float:
        alpha = self.config.alpha
        latest_actual = actuals[-1]
        latest_pred = preds[-1]
        return alpha * latest_actual + (1 - alpha) * latest_pred

    @staticmethod
    def _correlation(x: List[float], y: List[float]) -> float:
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        mean_x = mean(x)
        mean_y = mean(y)

        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denom_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
        denom_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5

        if denom_x == 0 or denom_y == 0:
            return 0.0
        return num / (denom_x * denom_y)

    @staticmethod
    def _format_ts(timestamp: float) -> str:
        return datetime.utcfromtimestamp(timestamp).isoformat() + "Z"

    @staticmethod
    def _regime_label(latest: float) -> str:
        if latest >= 0.9:
            return "High Stress"
        if latest >= 0.7:
            return "Elevated"
        if latest <= 0.45:
            return "Calm"
        return "Neutral"

    def _build_insights(self, actuals: List[float], stats: Dict) -> List[str]:
        insights: List[str] = []
        latest = actuals[-1]
        week_change = latest - actuals[max(0, len(actuals) - 24)]
        trend = "higher" if week_change > 0 else "lower" if week_change < 0 else "flat"
        insights.append(f"DVOL is {abs(week_change)*100:.1f} vol pts {trend} over the past day.")

        if abs(stats["bias"]) < 0.005:
            insights.append("Model bias is well centered (<0.5 vol pts).")
        else:
            direction = "over" if stats["bias"] > 0 else "under"
            insights.append(f"Model has been {direction}-estimating realized vol by {abs(stats['bias'])*100:.2f} vol pts.")

        insights.append(f"Direction hit-rate: {stats['direction_hit_rate']*100:.0f}% "
                        f"| Regime: {stats['regime']}")
        return insights
