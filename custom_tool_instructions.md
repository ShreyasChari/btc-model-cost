# BTC Options Trade Scanner - Kelly Edge

## Overview
The Trade Scanner tab implements a Kelly-optimized options trade ranker based on the BTC Options Dashboard specification. It scans all BTC options from Deribit and ranks trades by **Kelly x Edge** score.

## Key Features
1. **Vol Prediction Integration**: Input your predicted DVOL from ML model
2. **Kelly Criterion Sizing**: Uses fractional Kelly (10%, 25%, 50%) for position sizing
3. **Edge Calculation**: Edge = Vega x (Predicted_Vol - Market_IV)
4. **Risk Limits**: Respects portfolio Greek limits ($10M NAV default)
5. **Filtering**: Delta range, DTE range, bid-ask spread filters

## How It Works

### Edge Calculation
```
Edge_vol = Predicted_DVOL - Market_IV
Edge_dollars = Vega x Edge_vol
Direction = BUY if Edge > 0 (vol is cheap), SELL if Edge < 0 (vol is rich)
```

### Win Probability
Uses model RMSE as confidence interval:
```
Win_Prob = norm_cdf(Edge_vol / RMSE)
```

### Kelly Fraction
```
Full_Kelly = (p x W - q x L) / W
Adjusted_Kelly = Full_Kelly x Fractional_Kelly
```

### Rank Score
```
Score = Kelly_Fraction x Net_Edge_Dollars
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Predicted DVOL | 50% | Your ML model's DVOL prediction |
| Model RMSE | 3 pts | Prediction error for confidence |
| Kelly Fraction | 25% | Quarter Kelly (conservative) |
| NAV | $10M | Portfolio size for sizing |
| Min/Max Delta | 0.10-0.50 | Focus on tradeable strikes |
| Min/Max DTE | 5-90 | Days to expiry range |
| Max Bid-Ask | 4 vol pts | Liquidity threshold |
| Direction | All | Filter to long/short only |

## Risk Limits ($10M NAV)
- **Delta**: ±$2M (±20% NAV)
- **Theta**: ±$50k/day (±0.5% NAV)
- **Gamma**: $100k per 1% move (1% NAV)
- **Vega**: ±$100k per vol point (±1% NAV)

## API Endpoint
```
GET /api/kelly-edge-scanner?predicted_vol=55&prediction_rmse=3&kelly_fraction=0.25
```

## Interpretation

### Vol Regime
- **CHEAP**: Predicted > Current + 2pts -> Long Vol trades
- **RICH**: Predicted < Current - 2pts -> Short Vol trades
- **FAIR**: Within 2pts of current

### Trade Tables
- **Long Vol**: BUY trades ranked by score (vol is cheap)
- **Short Vol**: SELL trades ranked by score (vol is rich)
- **All Ranked**: Complete list with Greeks, edge, Kelly fraction

## Cost Model Integration
Each trade includes Cost Model theta decomposition:
- `gamma_cost`: Gamma carry (Gamma x Omega_G)
- `vanna_cost`: Vanna carry (Vanna x Omega_Va)
- `volga_cost`: Volga carry (Volga x Omega_Vo)
- `break_even`: Daily move needed to break even on theta
