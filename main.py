"""
Bitcoin Cost Model - FastAPI Application
Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio

from cost_model import BitcoinVolSurface, CostModelAnalyzer
from cost_model.surface import MarketComparison
from cost_model.deribit import DeribitClient

app = FastAPI(
    title="Bitcoin Cost Model",
    description="Implied Volatility Surface & Arbitrage Detection using Eid's Cost Model",
    version="1.0.0"
)

# Global state
surface: Optional[BitcoinVolSurface] = None
analyzer: Optional[CostModelAnalyzer] = None
comparison: Optional[MarketComparison] = None
market_options: List[Dict] = []  # Raw market data
deribit = DeribitClient()


# ============================================================================
# Pydantic Models
# ============================================================================

class ManualMaturity(BaseModel):
    maturity: float  # Years
    atm_vol: float   # e.g., 0.52 for 52%
    skew: float      # e.g., -0.06
    curvature: float # e.g., 0.40

class ManualSurfaceInput(BaseModel):
    spot: float
    maturities: List[ManualMaturity]

class OptionData(BaseModel):
    strike: float
    maturity: float
    bid_iv: Optional[float] = None
    ask_iv: Optional[float] = None
    mid_iv: Optional[float] = None
    volume: Optional[float] = None


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Serve the frontend"""
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health():
    """Health check"""
    return {"status": "ok", "surface_loaded": surface is not None}


@app.post("/api/surface/manual")
async def create_manual_surface(data: ManualSurfaceInput):
    """Create surface from manual inputs"""
    global surface, analyzer
    
    surface = BitcoinVolSurface(spot=data.spot)
    
    for m in data.maturities:
        surface.add_maturity(m.maturity, m.atm_vol, m.skew, m.curvature)
    
    analyzer = CostModelAnalyzer(surface)
    
    return {
        "status": "ok",
        "spot": data.spot,
        "maturities": len(data.maturities),
        "break_evens": surface.get_break_evens()
    }


@app.get("/api/surface/fetch-deribit")
async def fetch_deribit_surface():
    """Fetch live data from Deribit and build surface"""
    global surface, analyzer, comparison, market_options
    
    try:
        # Get current BTC price
        spot = await deribit.get_index_price("BTC")
        
        # Get options data
        options = await deribit.get_options_data("BTC")
        
        if not options:
            raise HTTPException(status_code=500, detail="No options data received")
        
        # Store raw market data
        market_options = options
        
        # Build surface from options
        surface = BitcoinVolSurface(spot=spot)
        
        # Group by maturity and calibrate
        by_maturity = {}
        for opt in options:
            T = opt['maturity']
            if T not in by_maturity:
                by_maturity[T] = {'strikes': [], 'vols': []}
            by_maturity[T]['strikes'].append(opt['strike'])
            by_maturity[T]['vols'].append(opt['mid_iv'])
        
        import numpy as np
        for T, data in sorted(by_maturity.items()):
            if len(data['strikes']) >= 3:  # Need at least 3 points
                strikes = np.array(data['strikes'])
                vols = np.array(data['vols'])
                surface.calibrate_from_options(T, strikes, vols)
        
        analyzer = CostModelAnalyzer(surface)
        
        # Create comparison object
        comparison = MarketComparison(surface)
        comparison.load_market_data(options)
        
        # Get comparison summary
        comp_result = comparison.compare()
        
        return {
            "status": "ok",
            "spot": spot,
            "maturities": len(surface.maturities),
            "options_count": len(options),
            "break_evens": surface.get_break_evens(),
            "comparison_summary": comp_result.get('summary', {})
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/surface")
async def get_surface():
    """Get current surface data"""
    if surface is None:
        raise HTTPException(status_code=400, detail="Surface not initialized")
    
    return surface.to_dict()


@app.get("/api/break-evens")
async def get_break_evens():
    """Get break-evens for all maturities"""
    if surface is None:
        raise HTTPException(status_code=400, detail="Surface not initialized")
    
    return surface.get_break_evens()


@app.get("/api/theta-decomposition")
async def get_theta_decomposition(strike: float, maturity: float):
    """Get theta decomposition for a specific option"""
    if surface is None:
        raise HTTPException(status_code=400, detail="Surface not initialized")
    
    return surface.get_theta_decomposition(strike, maturity)


@app.get("/api/arbitrage")
async def get_arbitrage_opportunities():
    """Analyze and return arbitrage opportunities"""
    if analyzer is None:
        raise HTTPException(status_code=400, detail="Analyzer not initialized")
    
    return analyzer.find_opportunities()


@app.get("/api/trades")
async def get_trade_recommendations():
    """Get specific trade recommendations"""
    if analyzer is None:
        raise HTTPException(status_code=400, detail="Analyzer not initialized")
    
    return analyzer.get_trade_recommendations()


@app.get("/api/vol")
async def get_vol(strike: float, maturity: float):
    """Get implied vol for specific strike/maturity"""
    if surface is None:
        raise HTTPException(status_code=400, detail="Surface not initialized")
    
    vol = surface.get_vol(strike, maturity)
    return {"strike": strike, "maturity": maturity, "implied_vol": vol}


@app.get("/api/deribit/instruments")
async def get_deribit_instruments():
    """Get available Deribit instruments"""
    try:
        instruments = await deribit.get_instruments("BTC")
        return {"instruments": instruments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/deribit/ticker/{instrument}")
async def get_deribit_ticker(instrument: str):
    """Get ticker for specific instrument"""
    try:
        ticker = await deribit.get_ticker(instrument)
        return ticker
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/comparison")
async def get_market_comparison(threshold: float = 0.01):
    """
    Compare market IVs to Cost Model IVs
    
    threshold: minimum spread (in vol terms) to flag as mispricing
               0.01 = 1 vol point, 0.02 = 2 vol points
    """
    if comparison is None:
        raise HTTPException(status_code=400, detail="No market data loaded. Fetch Deribit data first.")
    
    return comparison.compare(threshold=threshold)


@app.get("/api/comparison/signals")
async def get_mispricing_signals(threshold: float = 0.01, limit: int = 20):
    """Get top mispricings sorted by magnitude"""
    if comparison is None:
        raise HTTPException(status_code=400, detail="No market data loaded. Fetch Deribit data first.")
    
    result = comparison.compare(threshold=threshold)
    return {
        'signals': result.get('signals', [])[:limit],
        'summary': result.get('summary', {})
    }


@app.get("/api/comparison/heatmap")
async def get_mispricing_heatmap():
    """Get heatmap data for mispricing visualization"""
    if comparison is None:
        raise HTTPException(status_code=400, detail="No market data loaded. Fetch Deribit data first.")
    
    return comparison.get_mispricing_heatmap()


@app.get("/api/comparison/by-maturity/{maturity}")
async def get_comparison_by_maturity(maturity: float):
    """Get all comparisons for a specific maturity (in years)"""
    if comparison is None:
        raise HTTPException(status_code=400, detail="No market data loaded. Fetch Deribit data first.")
    
    result = comparison.compare()
    mat_key = str(round(maturity, 3))
    
    if mat_key in result.get('by_maturity', {}):
        return result['by_maturity'][mat_key]
    
    # Try to find closest maturity
    all_mats = result.get('by_maturity', {})
    if all_mats:
        closest = min(all_mats.keys(), key=lambda x: abs(float(x) - maturity))
        return {
            'requested': maturity,
            'closest_available': float(closest),
            'data': all_mats[closest]
        }
    
    raise HTTPException(status_code=404, detail=f"No data for maturity {maturity}")


@app.get("/api/market-data")
async def get_raw_market_data():
    """Get raw market data from last Deribit fetch"""
    if not market_options:
        raise HTTPException(status_code=400, detail="No market data loaded. Fetch Deribit data first.")
    
    return {
        'count': len(market_options),
        'options': market_options
    }


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
