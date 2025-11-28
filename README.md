# 🔷 Bitcoin Cost Model

Implied Volatility Surface & Arbitrage Detection based on Gilbert Eid's Cost Model framework.

## Overview

This tool builds a Bitcoin implied volatility surface and identifies arbitrage opportunities using the Cost Model's theta decomposition:

```
θ = Γ × Ω_G + Va × Ω_Va + Vo × Ω_Vo
```

Where:
- **Ω_G** = Cost of Gamma (volatility carry)
- **Ω_Va** = Cost of Vanna (skew carry)  
- **Ω_Vo** = Cost of Volga (smile carry)

## Features

- 📊 **Live Data**: Fetch options data from Deribit
- ✏️ **Manual Input**: Enter custom surface parameters
- 📈 **3D Surface Visualization**: Interactive volatility surface
- ⚖️ **Break-Even Analysis**: Gamma, Vanna, Volga break-evens
- 🔬 **Theta Decomposition**: See exactly what you're paying for
- 🎯 **Arbitrage Detection**: Calendar spreads, skew trades, butterflies
- 💰 **Trade Recommendations**: Specific executable trades

## Quick Start

### Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Open http://localhost:8000
```

### Run on Replit

1. Create a new Replit
2. Import from GitHub
3. Click "Run"

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend UI |
| `/api/surface/manual` | POST | Create surface from manual inputs |
| `/api/surface/fetch-deribit` | GET | Fetch live Deribit data |
| `/api/surface` | GET | Get current surface data |
| `/api/break-evens` | GET | Get break-evens table |
| `/api/theta-decomposition` | GET | Decompose theta for specific option |
| `/api/arbitrage` | GET | Find arbitrage opportunities |
| `/api/trades` | GET | Get trade recommendations |

## Manual Input Format

```json
{
  "spot": 95000,
  "maturities": [
    {"maturity": 0.0833, "atm_vol": 0.52, "skew": -0.06, "curvature": 0.40},
    {"maturity": 0.25, "atm_vol": 0.47, "skew": -0.045, "curvature": 0.32},
    {"maturity": 0.5, "atm_vol": 0.44, "skew": -0.04, "curvature": 0.28},
    {"maturity": 1.0, "atm_vol": 0.42, "skew": -0.035, "curvature": 0.25}
  ]
}
```

## Trading Signals

### Term Structure
- **Backwardation** (short > long): Sell front, buy back → Calendar spreads
- **Contango** (long > short): Normal regime

### Skew
- **Negative** (puts expensive): Sell put skew → Risk reversals
- **Positive** (calls expensive): Sell call skew

### Smile
- **Volga/Gamma < 0.6**: Wings cheap → Buy butterflies
- **Volga/Gamma > 1.0**: Wings expensive → Sell butterflies

## Break-Even Interpretation

| Metric | Meaning | Trade Signal |
|--------|---------|--------------|
| Gamma BE | Daily % move to break even on gamma | Compare to historical vol |
| Volga BE | Daily % vol move to break even | Compare to vol-of-vol |
| Vanna BE | Vol change per 1% spot move | Implied spot-vol correlation |

## Architecture

```
btc-cost-model/
├── main.py              # FastAPI application
├── cost_model/          # Core package
│   ├── __init__.py
│   ├── greeks.py        # Black-Scholes Greeks
│   ├── surface.py       # Vol surface & Cost Model
│   ├── analyzer.py      # Arbitrage detection
│   └── deribit.py       # Deribit API client
├── static/              # Frontend
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── requirements.txt
├── .replit
└── replit.nix
```

## References

- Gilbert Eid, "The Cost Model for Deriving Implied Volatility Surfaces"
- Baird, "Option Market Making"
- Max Dama, "Automated Trading"
- Santander, "Volatility Trading Primer"

## License

MIT
