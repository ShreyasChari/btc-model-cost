# VolPredictor Package

## Overview
`VolPredictor.py` automates the full DVOL forecasting workflow for Bitcoin options implied volatility. It blends high-frequency market data with macro liquidity metrics, trains a Gradient Boosting model, and continually refreshes the prediction while notifying a Telegram channel.

## End-to-End Flow
1. **Historical baseline** – Loads `Vol Prediction Data_new.csv` (30‑minute bars) to preserve feature history.
2. **Live market fetch** – Pulls the latest data for S&P 500, Nasdaq, US 10Y yield, VIX, BTCUSD, and Deribit DVOL via `tvDatafeed`. Each series is aligned to 30-minute UTC timestamps and forward-filled off the S&P 500 clock.
3. **Global liquidity construction** – Calls `get_total_liquidity()` to download central-bank balance sheets, FX pairs, RRP, and TGA levels. It currency-adjusts BOJ/ECB assets, then builds a composite "Global Net Liquidity" signal that is merged back into the trading dataset by Year-Month.
4. **Feature engineering** –
   - Converts timestamps to datetime index and adds `hour` and `dayofweek` categorical features.
   - Builds 35-interval rolling means for S&P 500, Nasdaq, VIX, and BTCUSD to capture short-term trends.
   - Creates the supervised label `DVOL_1w_ahead` by shifting DVOL forward 5 trading days × 7 intervals per day (approx. one week on the 30-minute grid).
   - Drops any rows with insufficient history to ensure clean training data.
5. **Model training & tuning** – Pipelines `StandardScaler` + `GradientBoostingRegressor`. Performs 5-fold `GridSearchCV` across `(n_estimators, max_depth, learning_rate)` to pick the best hyperparameters before re-fitting on the latest history (all but the most recent row).
6. **Prediction & monitoring** –
   - Uses the most recent feature row to forecast next-week DVOL, appends `Predicted DVOL` back into the dataframe, and logs the value.
   - Persists refreshed datasets to CSV (`Vol Prediction Data_new.csv`, `Vol Prediction Data_to_plot.csv`) and exports `Actual_vs_Predicted_DVOL.png`.
   - Pushes a concise telegram notification via `send_msg_on_telegram()` and records the run in `VolPredictor.log`.
7. **Automation** – `main()` executes once when the script starts and is then scheduled to repeat at the top of every hour using the `schedule` package.

## Files Included
- `VolPredictor.py` – Core orchestration script for data ingest, modeling, and alerting.
- `README.md` – This document summarizing the workflow and usage.
- Supporting CSV/PKL/log assets stay alongside the script so the model can reuse cached history between runs.

## Setup
1. **Python**: 3.9+ recommended.
2. **Install deps**: `pip install -r requirements.txt`.
3. **TradingView credentials**: Set `TV_USERNAME` and `TV_PASSWORD` env vars (update `VolPredictor.py` to read from them) or edit the placeholders directly if running in a secure environment.
4. **Telegram tokens**: Replace the placeholder bot tokens or load them from env vars before sharing the script externally.

## Running
```bash
python VolPredictor.py
```
The script immediately refreshes the dataset, produces a forecast, and then keeps running to update the prediction every hour.

## Outputs
- `VolPredictor.log` captures informational messages.
- `Vol Prediction Data_new.csv` holds the consolidated feature set.
- `Vol Prediction Data_to_plot.csv` & `Actual_vs_Predicted_DVOL.png` help visualize the model fit.
- Telegram message confirms each successful prediction cycle.

## Notes
- Ensure the machine running the script has stable network access; each cycle depends on multiple TradingView/FRED pulls.
- If you plan to distribute the project, scrub or externalize any sensitive API keys before zipping.
