# Cost Model Gamma & Variance Tracker

Full integration of Gilbert Eid's Cost Model framework with real-time Deribit portfolio analysis and dynamic hedging P&L attribution.

## The Master Equation

```
θ = Γ × Ω_G + Va × Ω_Va + Vo × Ω_Vo
```

Where:
- **Γ × Ω_G**: Gamma cost (volatility carry) - what you pay for potential gamma scalping revenue
- **Va × Ω_Va**: Vanna cost (skew carry) - what you pay for spot-vol correlation exposure
- **Vo × Ω_Vo**: Volga cost (smile carry) - what you pay for vol-of-vol exposure

## Why This Matters

Traditional P&L attribution shows "gamma P&L" vs "theta cost" - but that's incomplete. The Cost Model tells you *exactly* where your theta is going:

| If you're long... | You're paying for... | You profit when... |
|-------------------|---------------------|-------------------|
| Gamma | Volatility carry | Realized vol > Implied vol |
| Vanna | Skew carry | Spot-vol correlation behaves as expected |
| Volga | Smile carry | Vol moves a lot (vol of vol) |

## Features

### 1. Cost Model Decomposition
- Breaks down daily theta into gamma/vanna/volga components
- Shows percentage attribution of each cost
- Compares Cost Model theta to Black-Scholes theta

### 2. Dynamic P&L Simulation
- Simulates gamma scalping with full Cost Model attribution
- Models vol changes (for vanna/volga P&L)
- Tracks cumulative P&L by Greek source

### 3. Variance Budget Tracking
- Are you harvesting more variance than you paid for?
- Visual comparison of realized vs implied variance paths
- AHEAD/BEHIND/ON TRACK status

### 4. DVOL Prediction Integration
- Input your predicted DVOL from your GBR model
- Generates LONG VOL / SHORT VOL / NEUTRAL signals
- Compares prediction to current market

### 5. Model vs Market Arbitrage
- Calculates theoretical Cost Model price
- Compares to market price
- Flags mispricings > 5% with BUY/SELL signals

## Installation

```bash
cd cost-model-gamma-tracker
pip install -r requirements.txt
python app.py
# Open http://localhost:8000
```

### Saving Your Deribit Keys

Set environment variables once and the app will use them automatically (leave the UI fields empty):

```bash
export DERIBIT_CLIENT_ID="your_client_id"
export DERIBIT_CLIENT_SECRET="your_secret_key"
# Optional
export DERIBIT_TESTNET=true
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/market-data` | GET | Public BTC data, realized vol, DVOL |
| `/api/portfolio` | POST | Full portfolio with Cost Model analysis |
| `/api/analyze-manual` | POST | Analyze manually entered positions |
| `/api/cost-model-surface` | GET | Generate Cost Model IV surface |

## Usage Examples

### 1. Fetch Portfolio with DVOL Prediction
```python
import requests

response = requests.post('http://localhost:8000/api/portfolio', json={
    'credentials': {
        'client_id': 'your_client_id',
        'client_secret': 'your_secret',
        'testnet': False,
        'predicted_dvol': 52.5
    },
    'params': {}
})

data = response.json()
print(f"Vol Signal: {data['portfolio']['vol_signal']}")
print(f"Dollar Gamma: ${data['portfolio']['dollar_gamma']:.0f}")
print(f"Break-Even Move: {data['portfolio']['break_even_move']:.2f}%")
```

### 2. Generate Cost Model Surface
```python
response = requests.get('http://localhost:8000/api/cost-model-surface', params={
    'btc_price': 100000,
    'base_iv': 50,
    'vol_of_vol': 60,
    'spot_vol_corr': -0.5
})

surface = response.json()['surface']
# Returns theta decomposition across strikes and expiries
```

## Cost Model Mathematics

### Omega_Gamma (Ω_G) - Volatility Carry
```
Ω_G = 0.5 × σ² / 365
```
This is the daily variance you need to realize to break even on gamma.

### Omega_Vanna (Ω_Va) - Skew Carry
```
Ω_Va = ρ × σ_S × σ_σ / 365
```
Where ρ is spot-vol correlation (typically -0.5 to -0.7 for BTC).

### Omega_Volga (Ω_Vo) - Smile Carry
```
Ω_Vo = 0.5 × σ_σ² / 365
```
Where σ_σ is vol of vol (typically 50-100% for BTC).

## Integration with Your Existing Tools

### DVOL Prediction Model
Pass your GBR model's predicted DVOL:
```python
predicted_dvol = your_gbr_model.predict(features)
# Pass to API
```

### Cost Model IV Surface
The `/api/cost-model-surface` endpoint generates data compatible with your existing surface visualization.

## Key Insights

1. **Short-dated ATM options**: High gamma cost, low vanna/volga cost
2. **Long-dated OTM options**: Lower gamma cost, higher vanna/volga cost
3. **Skew trades**: Dominated by vanna cost
4. **Vol-of-vol trades**: Dominated by volga cost

## Files

- `app.py` - FastAPI backend with Cost Model calculations
- `CostModelGammaTracker.jsx` - React component for browser
- `requirements.txt` - Python dependencies

## References

- Gilbert Eid's Cost Model framework
- Santander Volatility Trading Primer
- Takaishi's Rough Volatility research for BTC

## License

MIT
