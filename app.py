"""
Cost Model Gamma & Variance Tracker
Full integration with Eid's Cost Model framework
θ = Γ × Ω_G + Va × Ω_Va + Vo × Ω_Vo
"""

import asyncio
import aiohttp
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from math import log, sqrt, exp, pi
import numpy as np

# ============================================================
# CONFIGURATION - Set via environment variables or defaults
# ============================================================
DEFAULT_DERIBIT_CLIENT_ID ="ltwn4DUq"
DEFAULT_DERIBIT_CLIENT_SECRET ="5xPmDo8epKVXJkSKv79wvB3_Foka7wwypGN0Ryqu_Ww"
DEFAULT_DERIBIT_TESTNET = os.getenv("DERIBIT_TESTNET", "false").lower() in {"1", "true", "yes"}

# ============================================================
# COST MODEL MATHEMATICS
# ============================================================

def norm_cdf(x: float) -> float:
    """Standard normal CDF"""
    a1, a2, a3 = 0.254829592, -0.284496736, 1.421413741
    a4, a5, p = -1.453152027, 1.061405429, 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x) / sqrt(2)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x)
    return 0.5 * (1.0 + sign * y)

def norm_pdf(x: float) -> float:
    """Standard normal PDF"""
    return exp(-0.5 * x * x) / sqrt(2 * pi)


@dataclass
class FullGreeks:
    """Extended Greeks including higher-order"""
    delta: float
    gamma: float
    theta: float
    vega: float
    vanna: float  # d(delta)/d(sigma) = d(vega)/d(S)
    volga: float  # d(vega)/d(sigma)
    price: float
    d1: float
    d2: float


@dataclass
class CostModelOmegas:
    """Cost of carry for each Greek component"""
    omega_gamma: float   # Ω_G: volatility carry cost
    omega_vanna: float   # Ω_Va: skew carry cost
    omega_volga: float   # Ω_Vo: smile carry cost


@dataclass
class ThetaDecomposition:
    """Decomposition of theta into Cost Model components"""
    gamma_cost: float    # Γ × Ω_G
    vanna_cost: float    # Va × Ω_Va  
    volga_cost: float    # Vo × Ω_Vo
    total_cost_model: float
    bs_theta: float
    residual: float
    omegas: CostModelOmegas


def calculate_full_greeks(S: float, K: float, T: float, r: float, sigma: float, 
                          option_type: str = 'call') -> FullGreeks:
    """
    Calculate all Greeks including Vanna and Volga
    """
    if T <= 0.001 or sigma <= 0.001:
        intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        delta = 1.0 if option_type == 'call' and S > K else (-1.0 if option_type == 'put' and S < K else 0.0)
        return FullGreeks(delta=delta, gamma=0, theta=0, vega=0, vanna=0, volga=0, 
                         price=intrinsic, d1=0, d2=0)
    
    sqrt_T = sqrt(T)
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    nd1 = norm_cdf(d1)
    nd2 = norm_cdf(d2)
    npd1 = norm_pdf(d1)
    
    if option_type == 'call':
        delta = nd1
        price = S * nd1 - K * exp(-r * T) * nd2
        theta = (-(S * npd1 * sigma) / (2 * sqrt_T) - r * K * exp(-r * T) * nd2) / 365
    else:
        delta = nd1 - 1
        price = K * exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
        theta = (-(S * npd1 * sigma) / (2 * sqrt_T) + r * K * exp(-r * T) * norm_cdf(-d2)) / 365
    
    gamma = npd1 / (S * sigma * sqrt_T)
    vega = S * npd1 * sqrt_T / 100  # per 1% vol move
    
    # Vanna: d(delta)/d(sigma) = -d2 * npd1 / sigma
    # Also equals d(vega)/d(S)
    vanna = -npd1 * d2 / sigma
    
    # Volga: d(vega)/d(sigma) = vega * d1 * d2 / sigma  
    volga = vega * d1 * d2 / sigma
    
    return FullGreeks(delta=delta, gamma=gamma, theta=theta, vega=vega,
                     vanna=vanna, volga=volga, price=price, d1=d1, d2=d2)


def calculate_cost_model_omegas(S: float, T: float, sigma: float, 
                                 vol_of_vol: float = 0.6,
                                 skew_slope: float = -0.1,
                                 spot_vol_corr: float = -0.5) -> CostModelOmegas:
    """
    Calculate Omega values (cost of carry for each Greek)
    
    Omega_Gamma (Ω_G): Cost of gamma carry
        - Represents realized variance
        - For ATM: Ω_G ≈ 0.5 * σ² (annualized)
        - Daily: Ω_G ≈ 0.5 * σ² / 365
        
    Omega_Vanna (Ω_Va): Cost of skew carry
        - Related to spot-vol correlation (typically negative for crypto)
        - Ω_Va ≈ ρ * σ_S * σ_σ
        - Simplified: Ω_Va ≈ skew_slope * σ
        
    Omega_Volga (Ω_Vo): Cost of smile carry (vol of vol)
        - Ω_Vo ≈ 0.5 * σ_σ² (variance of implied vol)
    
    Args:
        S: Spot price
        T: Time to expiry (years)
        sigma: Implied volatility (decimal)
        vol_of_vol: Volatility of implied vol (annualized, decimal)
        skew_slope: Skew per unit log-moneyness
        spot_vol_corr: Correlation between spot and vol changes
    """
    # Omega_Gamma: Daily variance (what you need to realize to break even)
    omega_gamma = 0.5 * sigma ** 2 / 365
    
    # Omega_Vanna: Skew carry
    # When spot falls, vol rises (negative correlation)
    # This creates P&L through vanna exposure
    omega_vanna = spot_vol_corr * sigma * vol_of_vol / 365
    
    # Omega_Volga: Smile carry / vol of vol carry
    omega_volga = 0.5 * vol_of_vol ** 2 / 365
    
    return CostModelOmegas(
        omega_gamma=omega_gamma,
        omega_vanna=omega_vanna, 
        omega_volga=omega_volga
    )


def decompose_theta_cost_model(greeks: FullGreeks, S: float, T: float, sigma: float,
                                vol_of_vol: float = 0.6,
                                spot_vol_corr: float = -0.5) -> ThetaDecomposition:
    """
    Decompose theta into Cost Model components:
    θ = Γ × Ω_G + Va × Ω_Va + Vo × Ω_Vo
    
    This tells you exactly where your theta cost is coming from:
    - Gamma cost: Paying for potential gamma scalping revenue
    - Vanna cost: Paying for skew exposure
    - Volga cost: Paying for vol-of-vol exposure
    """
    omegas = calculate_cost_model_omegas(S, T, sigma, vol_of_vol, spot_vol_corr=spot_vol_corr)
    
    # Gamma cost: Γ × S² × Ω_G
    gamma_cost = greeks.gamma * S * S * omegas.omega_gamma
    
    # Vanna cost: Va × S × Ω_Va
    vanna_cost = greeks.vanna * S * omegas.omega_vanna
    
    # Volga cost: Vo × Ω_Vo  
    volga_cost = greeks.volga * omegas.omega_volga
    
    total_cost_model = gamma_cost + vanna_cost + volga_cost
    
    return ThetaDecomposition(
        gamma_cost=gamma_cost,
        vanna_cost=vanna_cost,
        volga_cost=volga_cost,
        total_cost_model=total_cost_model,
        bs_theta=greeks.theta,
        residual=greeks.theta - total_cost_model,
        omegas=omegas
    )


def cost_model_theoretical_price(S: float, K: float, T: float, r: float, sigma: float,
                                  option_type: str, market_skew: Optional[Dict] = None) -> Dict:
    """
    Calculate theoretical price using Cost Model adjustments
    
    The Cost Model can adjust BS prices for:
    1. Skew effects (via vanna)
    2. Smile effects (via volga)
    
    Returns both BS price and adjusted model price
    """
    greeks = calculate_full_greeks(S, K, T, r, sigma, option_type)
    decomposition = decompose_theta_cost_model(greeks, S, T, sigma)
    
    model_price = greeks.price
    
    if market_skew:
        # Adjust for observed skew
        # Price adjustment = Vanna * skew_adjustment
        log_moneyness = log(K / S) / (sigma * sqrt(T)) if T > 0 else 0
        skew_adjustment = market_skew.get('slope', 0) * log_moneyness
        model_price += greeks.vanna * skew_adjustment * S
    
    return {
        'bs_price': greeks.price,
        'model_price': model_price,
        'greeks': asdict(greeks),
        'decomposition': asdict(decomposition)
    }


# ============================================================
# REALIZED VOLATILITY CALCULATIONS
# ============================================================

def calculate_realized_volatility(closes: List[float], window: int = None) -> Tuple[float, List[float]]:
    """Calculate realized volatility from close prices"""
    if len(closes) < 2:
        return 0.0, []
    
    returns = [log(closes[i] / closes[i-1]) for i in range(1, len(closes))]
    
    if window:
        returns = returns[-window:]
    
    if len(returns) < 2:
        return 0.0, returns
    
    variance = np.var(returns, ddof=1)
    daily_vol = sqrt(variance)
    annualized_vol = daily_vol * sqrt(365)  # crypto 365 days
    
    return annualized_vol * 100, returns


def calculate_vol_of_vol(ivs: List[float], window: int = None) -> float:
    """Calculate volatility of implied volatility"""
    if len(ivs) < 2:
        return 0.0
    
    if window:
        ivs = ivs[-window:]
    
    # Log returns of IV
    returns = [log(ivs[i] / ivs[i-1]) for i in range(1, len(ivs)) if ivs[i-1] > 0]
    
    if len(returns) < 2:
        return 0.0
    
    variance = np.var(returns, ddof=1)
    return sqrt(variance) * sqrt(365) * 100


def calculate_spot_vol_correlation(spot_returns: List[float], vol_changes: List[float], 
                                    window: int = None) -> float:
    """Calculate correlation between spot returns and vol changes"""
    if len(spot_returns) != len(vol_changes) or len(spot_returns) < 5:
        return -0.5  # Default assumption for crypto
    
    if window:
        spot_returns = spot_returns[-window:]
        vol_changes = vol_changes[-window:]
    
    return np.corrcoef(spot_returns, vol_changes)[0, 1]


# ============================================================
# DERIBIT CLIENT
# ============================================================

DERIBIT_BASE_URL = "https://www.deribit.com/api/v2"
DERIBIT_TEST_URL = "https://test.deribit.com/api/v2"


class DeribitClient:
    def __init__(self, client_id: str = None, client_secret: str = None, testnet: bool = False):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = DERIBIT_TEST_URL if testnet else DERIBIT_BASE_URL
        self.access_token = None
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        if self.client_id and self.client_secret:
            await self.authenticate()
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
            
    async def authenticate(self):
        params = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        result = await self._request("public/auth", params)
        self.access_token = result.get("access_token")
        return result
    
    async def _request(self, method: str, params: dict = None) -> dict:
        url = f"{self.base_url}/{method}"
        headers = {}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
            
        async with self.session.get(url, params=params, headers=headers) as response:
            data = await response.json()
            if "result" in data:
                return data["result"]
            elif "error" in data:
                raise Exception(f"API Error: {data['error']}")
            return data
    
    async def get_index_price(self, currency: str = "BTC") -> float:
        result = await self._request("public/get_index_price", {"index_name": f"{currency.lower()}_usd"})
        return result["index_price"]
    
    async def get_positions(self, currency: str = "BTC") -> List[Dict]:
        if not self.access_token:
            raise Exception("Authentication required")
        return await self._request("private/get_positions", {"currency": currency})
    
    async def get_order_book(self, instrument_name: str) -> Dict:
        return await self._request("public/get_order_book", {"instrument_name": instrument_name})
    
    async def get_tradingview_chart_data(self, instrument_name: str, resolution: str = "D",
                                          start: int = None, end: int = None) -> Dict:
        if start is None:
            start = int((datetime.now() - timedelta(days=60)).timestamp() * 1000)
        if end is None:
            end = int(datetime.now().timestamp() * 1000)
        return await self._request("public/get_tradingview_chart_data", {
            "instrument_name": instrument_name,
            "resolution": resolution,
            "start_timestamp": start,
            "end_timestamp": end
        })
    
    async def get_volatility_index(self, currency: str = "BTC") -> Dict:
        return await self._request("public/get_volatility_index_data", {
            "currency": currency,
            "resolution": "1D",
            "start_timestamp": int((datetime.now() - timedelta(days=60)).timestamp() * 1000),
            "end_timestamp": int(datetime.now().timestamp() * 1000)
        })

    async def get_instruments(self, currency: str = "BTC", kind: str = "option", expired: bool = False) -> List[Dict]:
        """Fetch all available instruments"""
        return await self._request("public/get_instruments", {
            "currency": currency,
            "kind": kind,
            "expired": expired
        })

    async def get_book_summary_by_currency(self, currency: str = "BTC", kind: str = "option") -> List[Dict]:
        """Fetch order book summaries for all instruments - includes mark_price, mark_iv, bid/ask"""
        return await self._request("public/get_book_summary_by_currency", {
            "currency": currency,
            "kind": kind
        })

    async def get_ticker(self, instrument_name: str) -> Dict:
        """Fetch ticker data for an instrument - includes Greeks (delta, gamma, theta, vega)"""
        return await self._request("public/ticker", {"instrument_name": instrument_name})

    async def get_forwards_by_expiry(self, currency: str = "BTC") -> Dict[str, float]:
        """
        Get synthetic forward price for each expiry.

        Deribit calculates synthetic forwards via put-call parity across strikes.
        The forward is returned as 'underlying_price' in order_book/ticker responses.

        We only need to query one option per expiry to get the forward for that maturity.

        Returns:
            Dict mapping expiry string (e.g., '27DEC24') to forward price
        """
        forwards = {}

        # Get all option instruments
        instruments = await self.get_instruments(currency, "option")

        # Group by expiry - we only need one option per expiry
        expiries_seen = set()
        sample_instruments = []

        for inst in instruments:
            # Extract expiry from instrument name (e.g., BTC-27DEC24-100000-C -> 27DEC24)
            parts = inst['instrument_name'].split('-')
            if len(parts) >= 2:
                expiry = parts[1]
                if expiry not in expiries_seen:
                    expiries_seen.add(expiry)
                    sample_instruments.append(inst['instrument_name'])

        # Fetch order book for one option per expiry to get forward price
        for instrument_name in sample_instruments:
            try:
                book = await self.get_order_book(instrument_name)
                expiry = instrument_name.split('-')[1]
                forward = book.get('underlying_price')
                if forward:
                    forwards[expiry] = forward
            except Exception as e:
                print(f"Failed to fetch forward for {instrument_name}: {e}")

        return forwards


# ============================================================
# PORTFOLIO ANALYSIS WITH COST MODEL
# ============================================================

@dataclass
class PositionAnalysis:
    instrument: str
    size: float
    direction: str
    strike: float
    expiry: str
    dte: int
    option_type: str
    iv: float  # IV as percentage (e.g., 55.5)
    # Greeks
    delta: float
    gamma: float
    theta: float  # Our BS-calculated theta (USD/day)
    vega: float
    vanna: float
    volga: float
    # Theta comparison (BS vs Deribit)
    theta_deribit: float  # Deribit's reported theta (USD/day)
    theta_diff: float  # Our theta - Deribit theta (positive = Deribit more negative)
    theta_diff_pct: float  # % difference from Deribit
    # Cost Model decomposition
    gamma_cost: float
    vanna_cost: float
    volga_cost: float
    # Pricing
    mark_price: float  # Total position value in USD
    per_contract_price: float  # Per-contract price in USD (for parity calc)
    model_price: float
    mispricing_pct: float
    signal: str


@dataclass
class PortfolioAnalysis:
    # Aggregate Greeks
    total_delta: float
    total_gamma: float
    total_theta: float  # USD/day (our BS calculation)
    total_vega: float
    total_vanna: float
    total_volga: float
    # Cost Model aggregates
    total_gamma_cost: float
    total_vanna_cost: float
    total_volga_cost: float
    # Key metrics
    gamma_1pct: float  # Delta change (in BTC) per 1% spot move
    dollar_gamma: float
    daily_theta_usd: float
    break_even_move: float
    avg_iv: float
    # Expected gamma P&L and net theta (accounts for expected spot movement)
    expected_gamma_pnl: float  # Expected daily gamma P&L based on realized vol
    fair_gamma_pnl: float  # Expected daily gamma P&L at implied vol (fair value)
    net_theta_usd: float  # Theta + expected gamma P&L (what you actually expect to earn/lose)
    # Theta/Gamma Analysis - bumped theta vs BSM theta
    bumped_theta: float  # Theta by repricing: T-1 day, forward moved by 1 std dev (at IV)
    theta_gamma_ratio: float  # Ratio: bumped_theta / bsm_theta (should be ~0 at IV)
    # Market data
    btc_price: float
    realized_vol_30d: float
    vol_of_vol: float
    spot_vol_corr: float
    dvol: float
    # Positions
    positions: List[PositionAnalysis]
    # Trade signals
    vol_signal: str
    vol_spread: float
    # Put-Call Parity violations
    parity_violations: List[Dict]
    # Vol surface data (IV by strike/expiry)
    vol_surface: List[Dict]
    # Box Spread arbitrage opportunities
    box_spread_arbs: List[Dict]
    # Butterfly arbitrage (convexity violations)
    butterfly_arbs: List[Dict]
    # Term structure analysis (√T normalized)
    term_structure: List[Dict]
    # Theta comparison (BS vs Deribit) - sorted by largest variance
    theta_comparison: List[Dict]


def parse_instrument(name: str) -> Optional[Dict]:
    """Parse Deribit instrument name"""
    parts = name.upper().split('-')
    if len(parts) < 4:
        return None
    
    try:
        expiry = parts[1]
        strike = float(parts[2])
        option_type = 'call' if parts[3] == 'C' else 'put'
        
        months = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                  'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
        day = int(expiry[:2])
        month = months.get(expiry[2:5], 1)
        year = 2000 + int(expiry[5:7])
        
        expiry_date = datetime(year, month, day)
        dte = max(0, (expiry_date - datetime.now()).days)
        
        return {
            'strike': strike,
            'option_type': option_type,
            'dte': dte,
            'expiry': expiry,
            'expiry_date': expiry_date
        }
    except:
        return None


async def analyze_portfolio_with_cost_model(
    positions: List[Dict],
    btc_price: float,
    realized_vol: float,
    vol_of_vol: float = 60.0,
    spot_vol_corr: float = -0.5,
    dvol: float = 50.0,
    predicted_dvol: float = None,
    client: 'DeribitClient' = None  # Pass client to fetch order books
) -> PortfolioAnalysis:
    """
    Comprehensive portfolio analysis with Cost Model decomposition
    """
    position_analyses = []

    # Aggregate values
    total_delta = 0
    total_gamma = 0
    total_theta = 0.0
    total_vega = 0
    total_vanna = 0
    total_volga = 0
    total_gamma_cost = 0
    total_vanna_cost = 0
    total_volga_cost = 0
    iv_sum = 0
    iv_count = 0

    # Pre-fetch synthetic forward prices for each expiry
    # Deribit uses synthetic forwards (computed via put-call parity) for option pricing
    # This is more efficient than fetching per-instrument
    forwards_by_expiry = {}
    if client:
        try:
            forwards_by_expiry = await client.get_forwards_by_expiry("BTC")
        except Exception as e:
            print(f"Failed to fetch forwards by expiry: {e}")

    for pos in positions:
        parsed = parse_instrument(pos.get('instrument_name', ''))
        if not parsed:
            continue
        
        size = pos.get('size', 0)
        if size == 0:
            continue

        direction = pos.get('direction', 'buy')
        # Deribit's size is already signed (negative for shorts)
        # So we use abs(size) and apply sign from direction
        abs_size = abs(size)
        sign = 1 if direction == 'buy' else -1

        T = parsed['dte'] / 365

        # Get synthetic forward price for this expiry from pre-fetched cache
        # Deribit uses synthetic forwards (via put-call parity) for option pricing
        # Using forward with r=0 in BS aligns our Greeks with Deribit's
        forward_price = forwards_by_expiry.get(parsed['expiry'], btc_price)

        # Fetch ticker for IV and Deribit's Greeks (theta comparison)
        ticker = None
        if client:
            try:
                ticker = await client.get_ticker(pos['instrument_name'])
            except:
                pass

        # Fetch IV from ticker or order book
        iv_raw = pos.get('mark_iv') or pos.get('implied_volatility')
        if not iv_raw and ticker:
            iv_raw = ticker.get('mark_iv')
        if not iv_raw and client:
            try:
                order_book = await client.get_order_book(pos['instrument_name'])
                iv_raw = order_book.get('mark_iv')
            except:
                pass
        iv = (iv_raw / 100) if iv_raw else 0.50  # Default 50% if not provided

        # Calculate Greeks using FORWARD price with r=0
        # When pricing off forwards, the carry cost is already embedded in the forward
        # so we set r=0 in the BS formula
        greeks = calculate_full_greeks(forward_price, parsed['strike'], T, 0.0, iv, parsed['option_type'])
        
        # Cost Model decomposition
        decomp = decompose_theta_cost_model(
            greeks, btc_price, T, iv,
            vol_of_vol=vol_of_vol / 100,
            spot_vol_corr=spot_vol_corr
        )
        
        # Mark price in USD
        mark_price = pos.get('mark_price', 0) * btc_price
        signal = 'FAIR'
        
        # Use Deribit's delta when available, but calculate our own gamma/theta for consistency
        # Deribit delta is generally accurate
        delta_contribution = pos.get('delta', greeks.delta * abs_size * sign)
        # Always use our calculated BS gamma for consistency with scenario analysis
        # BS gamma = d(delta)/dS, units: delta change per $1 move
        gamma_contribution = greeks.gamma * abs_size * sign
        # Always use our calculated BS theta - Deribit's theta is known to be inflated
        # Our BS theta formula uses S (USD spot), so theta is already in USD/day
        theta_contribution = greeks.theta * abs_size * sign  # Already in USD/day
        # Use Deribit's vega when available
        vega_contribution = pos.get('vega', greeks.vega * abs_size * sign)

        total_delta += delta_contribution
        total_gamma += gamma_contribution
        total_theta += theta_contribution
        total_vega += vega_contribution
        total_vanna += greeks.vanna * abs_size * sign  # Deribit doesn't provide vanna
        total_volga += greeks.volga * abs_size * sign  # Deribit doesn't provide volga

        # Cost model costs - scale to position size
        # Note: decomp costs are positive for a long option (you pay theta)
        # For short positions (sign=-1), we negate so costs become negative (you receive theta)
        # But theta itself is negative for long options, so to match:
        # Long option: theta < 0, cost > 0... we want cost = -theta (approximately)
        # So we negate to match the sign of theta
        total_gamma_cost += -decomp.gamma_cost * abs_size * sign
        total_vanna_cost += -decomp.vanna_cost * abs_size * sign
        total_volga_cost += -decomp.volga_cost * abs_size * sign
        
        iv_sum += iv_raw if iv_raw else 50
        iv_count += 1

        # Store per-contract price in USD for parity calculation
        per_contract_price_usd = pos.get('mark_price', 0) * btc_price

        # Our BS-calculated theta (already in USD/day)
        our_theta = greeks.theta * abs_size * sign

        # Deribit's theta - need to fetch from ticker API (positions API doesn't include theta)
        # Ticker greeks.theta is in USD per day per 1 contract
        deribit_theta_usd = 0
        if client:
            try:
                ticker = await client.get_ticker(pos['instrument_name'])
                ticker_greeks = ticker.get('greeks', {})
                # Deribit ticker theta is USD/day per 1 contract
                deribit_theta_per_contract = ticker_greeks.get('theta', 0)
                # Scale by position size and direction
                deribit_theta_usd = deribit_theta_per_contract * abs_size * sign
            except Exception as e:
                print(f"Failed to fetch ticker for {pos['instrument_name']}: {e}")

        # Calculate difference: our theta - Deribit theta
        # Positive diff means Deribit shows more negative theta (inflated)
        theta_diff = our_theta - deribit_theta_usd
        theta_diff_pct = (theta_diff / abs(deribit_theta_usd) * 100) if deribit_theta_usd != 0 else 0

        position_analyses.append(PositionAnalysis(
            instrument=pos['instrument_name'],
            size=abs_size,
            direction=direction,
            strike=parsed['strike'],
            expiry=parsed['expiry'],
            dte=parsed['dte'],
            option_type=parsed['option_type'],
            iv=iv_raw if iv_raw else 50,  # Store IV as percentage
            delta=greeks.delta * abs_size * sign,
            gamma=greeks.gamma * abs_size * sign,
            theta=our_theta,
            vega=greeks.vega * abs_size * sign,
            vanna=greeks.vanna * abs_size * sign,
            volga=greeks.volga * abs_size * sign,
            theta_deribit=deribit_theta_usd,
            theta_diff=theta_diff,
            theta_diff_pct=theta_diff_pct,
            gamma_cost=-decomp.gamma_cost * abs_size * sign,
            vanna_cost=-decomp.vanna_cost * abs_size * sign,
            volga_cost=-decomp.volga_cost * abs_size * sign,
            mark_price=mark_price,
            per_contract_price=per_contract_price_usd,  # Per-contract for parity
            model_price=mark_price,
            mispricing_pct=0,
            signal=signal
        ))
    
    # Calculate key metrics
    avg_iv = iv_sum / max(iv_count, 1)
    
    # Gamma 1%: Delta change (in BTC) per 1% spot move
    # Deribit gamma is delta change per $1 move
    # So gamma_1pct = total_gamma * (1% of spot) = total_gamma * btc_price * 0.01
    gamma_1pct = total_gamma * btc_price * 0.01
    
    # Dollar gamma: P&L per 1% squared move
    # gamma is d²V/dS², so for 1% move: P&L = 0.5 * gamma * (0.01*S)² = 0.5 * gamma * S² * 0.0001
    # Deribit gamma is already position-level
    dollar_gamma = 0.5 * total_gamma * btc_price * btc_price * 0.0001
    
    # Theta already aggregated in USD/day
    daily_theta_usd = total_theta

    # Expected gamma P&L from realized volatility
    # Formula: Expected P&L = 0.5 × Gamma × S² × σ_daily²
    # dollar_gamma is already 0.5 * gamma * S² * 0.0001 (P&L per 1% squared move)
    # Daily variance in %² = (RV_annual%)² / 365
    # Expected gamma P&L = dollar_gamma * daily_variance
    daily_variance_pct_sq = (realized_vol ** 2) / 365  # Daily variance in %²
    expected_gamma_pnl = dollar_gamma * daily_variance_pct_sq

    # Also compute at implied vol for comparison (fair value gamma P&L)
    iv_daily_variance = (avg_iv ** 2) / 365
    fair_gamma_pnl = dollar_gamma * iv_daily_variance

    # Net theta = BS theta + expected gamma P&L (at realized vol)
    # For short gamma: gamma P&L is negative (you lose when spot moves)
    # For long gamma: gamma P&L is positive (you profit when spot moves)
    net_theta_usd = daily_theta_usd + expected_gamma_pnl

    # Bumped Theta: Reprice portfolio with T-1 day to see actual time decay
    # We compute:
    # 1. Pure time decay: V(S, T-1) - V(S, T) = bumped theta from time only
    # 2. Compare to BSM theta
    # The difference shows how well the Greek approximation works
    current_portfolio_value = 0
    tomorrow_portfolio_value = 0  # Same spot, T-1 day
    up_move_value = 0  # S+1σ, T-1 day
    down_move_value = 0  # S-1σ, T-1 day

    daily_vol = avg_iv / 100 / sqrt(365)  # Daily vol in decimal
    expected_spot_move = btc_price * daily_vol  # 1 std dev daily move in $

    for pa in position_analyses:
        # Current T
        T = pa.dte / 365
        # Bumped T (1 day forward)
        T_bumped = max((pa.dte - 1) / 365, 0.001)  # Avoid division by zero
        iv_decimal = pa.iv / 100

        # Get forward for this position's expiry
        forward = forwards_by_expiry.get(pa.expiry, btc_price)

        # Sign: direction determines if we're long (+) or short (-) this option
        sign = 1 if pa.direction == 'buy' else -1

        # Current value at (S, T)
        current_greeks = calculate_full_greeks(forward, pa.strike, T, 0.0, iv_decimal, pa.option_type)
        current_portfolio_value += current_greeks.price * pa.size * sign

        # Tomorrow value at (S, T-1) - pure time decay, no spot move
        tomorrow_greeks = calculate_full_greeks(forward, pa.strike, T_bumped, 0.0, iv_decimal, pa.option_type)
        tomorrow_portfolio_value += tomorrow_greeks.price * pa.size * sign

        # Up move: (S+1σ, T-1)
        up_greeks = calculate_full_greeks(forward + expected_spot_move, pa.strike, T_bumped, 0.0, iv_decimal, pa.option_type)
        up_move_value += up_greeks.price * pa.size * sign

        # Down move: (S-1σ, T-1)
        down_greeks = calculate_full_greeks(forward - expected_spot_move, pa.strike, T_bumped, 0.0, iv_decimal, pa.option_type)
        down_move_value += down_greeks.price * pa.size * sign

    # Bumped theta = pure time decay (spot unchanged)
    bumped_theta = tomorrow_portfolio_value - current_portfolio_value

    # Expected P&L with variance: average of up and down moves (captures gamma effect)
    # This is what you'd expect to make/lose over 1 day if spot moves by 1 std dev randomly
    expected_pnl_with_variance = 0.5 * (up_move_value + down_move_value) - current_portfolio_value

    # Theta/Gamma ratio: bumped_theta / BSM_theta
    # Should be close to 1.0 if BS theta is accurate (both measure pure time decay)
    # Deviation indicates model error or numerical precision issues
    theta_gamma_ratio = bumped_theta / daily_theta_usd if abs(daily_theta_usd) > 0.01 else 0

    # Break-even: daily move needed for gamma P&L to offset theta
    # 0.5 * gamma * S² * move² = |theta|
    # move = sqrt(2 * |theta_usd| / (0.5 * gamma * S²))
    if total_gamma > 0 and abs(daily_theta_usd) > 0:
        break_even_move = sqrt(2 * abs(daily_theta_usd) / (0.5 * total_gamma * btc_price * btc_price)) * 100
    else:
        break_even_move = 0
    
    # Vol signal from prediction
    if predicted_dvol is not None:
        vol_spread = predicted_dvol - dvol
        if vol_spread > 3:
            vol_signal = 'LONG VOL'
        elif vol_spread < -3:
            vol_signal = 'SHORT VOL'
        else:
            vol_signal = 'NEUTRAL'
    else:
        vol_spread = realized_vol - avg_iv
        vol_signal = 'LONG VOL' if vol_spread > 3 else ('SHORT VOL' if vol_spread < -3 else 'NEUTRAL')

    # Build vol surface from positions (IV by strike and expiry)
    vol_surface = []
    for pa in position_analyses:
        vol_surface.append({
            'strike': pa.strike,
            'expiry': pa.expiry,
            'dte': pa.dte,
            'iv': pa.iv,
            'option_type': pa.option_type,
            'moneyness': (pa.strike / btc_price - 1) * 100  # % OTM/ITM
        })

    # Put-Call Parity Check
    # For same strike/expiry: C - P = S - K*e^(-rT)
    # Group positions by strike and expiry to find put-call pairs
    parity_violations = []
    positions_by_strike_expiry = {}
    for pa in position_analyses:
        key = (pa.strike, pa.expiry)
        if key not in positions_by_strike_expiry:
            positions_by_strike_expiry[key] = {}
        positions_by_strike_expiry[key][pa.option_type] = pa

    r = 0.05  # risk-free rate assumption
    for (strike, expiry), opts in positions_by_strike_expiry.items():
        if 'call' in opts and 'put' in opts:
            call = opts['call']
            put = opts['put']
            T = call.dte / 365

            # Theoretical parity: C - P = S - K*e^(-rT)
            # Use per-contract prices (already in USD per contract)
            call_price = call.per_contract_price
            put_price = put.per_contract_price

            theoretical_diff = btc_price - strike * exp(-r * T)
            actual_diff = call_price - put_price
            parity_error = actual_diff - theoretical_diff
            parity_error_pct = (parity_error / btc_price) * 100 if btc_price > 0 else 0

            # Flag if parity error > 0.5% of spot
            if abs(parity_error_pct) > 0.5:
                parity_violations.append({
                    'strike': strike,
                    'expiry': expiry,
                    'dte': call.dte,
                    'call_price': call_price,
                    'put_price': put_price,
                    'theoretical_diff': theoretical_diff,
                    'actual_diff': actual_diff,
                    'parity_error': parity_error,
                    'parity_error_pct': parity_error_pct,
                    'action': 'BUY CALL + SELL PUT' if parity_error < 0 else 'SELL CALL + BUY PUT'
                })

    # ========================================
    # BOX SPREAD ARBITRAGE DETECTION
    # ========================================
    # Box Spread: Buy Call Spread K1→K2 + Buy Put Spread K2→K1 = (K2-K1)×e^(-rT)
    # If market cost < theoretical value → buy the box (risk-free profit)
    # If market cost > theoretical value → sell the box (risk-free profit)
    box_spread_arbs = []

    # Group options by expiry for box spread analysis
    options_by_expiry = {}
    for pa in position_analyses:
        if pa.expiry not in options_by_expiry:
            options_by_expiry[pa.expiry] = {'calls': {}, 'puts': {}}
        options_by_expiry[pa.expiry][pa.option_type + 's'][pa.strike] = pa

    for expiry, opts in options_by_expiry.items():
        calls = opts['calls']
        puts = opts['puts']

        # Find strikes where we have both calls and puts
        common_strikes = sorted(set(calls.keys()) & set(puts.keys()))

        # Check all pairs of strikes for box spread opportunities
        for i, k1 in enumerate(common_strikes):
            for k2 in common_strikes[i+1:]:
                if k2 <= k1:
                    continue

                c1 = calls.get(k1)
                c2 = calls.get(k2)
                p1 = puts.get(k1)
                p2 = puts.get(k2)

                if not all([c1, c2, p1, p2]):
                    continue

                T = c1.dte / 365
                # Theoretical value of box spread: (K2 - K1) * e^(-rT)
                theoretical_value = (k2 - k1) * exp(-r * T)

                # Market cost of box: Buy C(K1) - Sell C(K2) + Sell P(K1) - Buy P(K2)
                # = C(K1) - C(K2) - P(K1) + P(K2)
                # Using per-contract prices
                market_cost = (c1.per_contract_price - c2.per_contract_price
                              - p1.per_contract_price + p2.per_contract_price)

                arb_profit = theoretical_value - market_cost
                arb_pct = (arb_profit / theoretical_value) * 100 if theoretical_value > 0 else 0

                # Flag if arb > 0.5% of box value
                if abs(arb_pct) > 0.5:
                    box_spread_arbs.append({
                        'expiry': expiry,
                        'dte': c1.dte,
                        'k1': k1,
                        'k2': k2,
                        'theoretical_value': theoretical_value,
                        'market_cost': market_cost,
                        'arb_profit': arb_profit,
                        'arb_pct': arb_pct,
                        'action': 'BUY BOX' if arb_profit > 0 else 'SELL BOX'
                    })

    # ========================================
    # BUTTERFLY CONVEXITY VIOLATION DETECTION
    # ========================================
    # No-arbitrage condition: C(K1) + C(K3) >= 2×C(K2) where K2 = (K1+K3)/2
    # If violated: Buy wings (K1, K3), Sell body (2×K2) for risk-free profit
    butterfly_arbs = []

    for expiry, opts in options_by_expiry.items():
        for opt_type in ['calls', 'puts']:
            options = opts[opt_type]
            strikes = sorted(options.keys())

            # Only check consecutive triplets for proper butterflies
            # K1-K2 and K2-K3 should have equal spacing
            for i in range(len(strikes) - 2):
                k1 = strikes[i]
                k2 = strikes[i + 1]
                k3 = strikes[i + 2]

                # Check if strikes are evenly spaced (within 1%)
                spread1 = k2 - k1
                spread2 = k3 - k2
                if abs(spread1 - spread2) / min(spread1, spread2) > 0.01:
                    continue

                o1 = options.get(k1)
                o2 = options.get(k2)
                o3 = options.get(k3)

                if not all([o1, o2, o3]):
                    continue

                # Convexity condition: C(K1) + C(K3) >= 2*C(K2)
                wings_cost = o1.per_contract_price + o3.per_contract_price
                body_cost = 2 * o2.per_contract_price

                # Violation = body_cost - wings_cost (positive = violation)
                convexity_violation = body_cost - wings_cost

                # Flag if violation > $50 (meaningful arb after transaction costs)
                if convexity_violation > 50:
                    butterfly_arbs.append({
                        'expiry': expiry,
                        'dte': o1.dte,
                        'option_type': opt_type[:-1],  # Remove 's'
                        'k1': k1,
                        'k2': k2,
                        'k3': k3,
                        'wings_cost': wings_cost,
                        'body_cost': body_cost,
                        'convexity_violation': convexity_violation,
                        'action': f'BUY {k1}/{k3} SELL 2x{k2}'
                    })

    # ========================================
    # √T NORMALIZED TERM STRUCTURE ANALYSIS
    # ========================================
    # Santander approach: Compare IVs directly and calculate forward vol
    # Forward vol between T1 and T2: σ_fwd² = (σ2²×T2 - σ1²×T1) / (T2 - T1)
    # Richness = how much IV deviates from what forward vol implies
    term_structure = []

    # Get ATM IV for each expiry (closest to spot)
    expiry_atm_ivs = {}
    for pa in position_analyses:
        moneyness = abs(pa.strike / btc_price - 1)
        if pa.expiry not in expiry_atm_ivs or moneyness < expiry_atm_ivs[pa.expiry]['moneyness']:
            expiry_atm_ivs[pa.expiry] = {
                'iv': pa.iv,
                'dte': pa.dte,
                'moneyness': moneyness,
                'strike': pa.strike
            }

    # Sort by DTE
    sorted_expiries = sorted(expiry_atm_ivs.items(), key=lambda x: x[1]['dte'])

    if len(sorted_expiries) > 0:
        # Calculate average IV as baseline for richness
        avg_iv = sum(d['iv'] for _, d in sorted_expiries) / len(sorted_expiries)

        for i, (expiry, data) in enumerate(sorted_expiries):
            dte = data['dte']
            T = dte / 365
            iv = data['iv']

            # Calculate forward vol if we have a previous expiry
            fwd_vol = None
            if i > 0:
                _, prev_data = sorted_expiries[i-1]
                T1 = prev_data['dte'] / 365
                T2 = T
                iv1 = prev_data['iv'] / 100  # Convert to decimal
                iv2 = iv / 100

                # Forward variance: σ_fwd² = (σ2²×T2 - σ1²×T1) / (T2 - T1)
                if T2 > T1:
                    fwd_var = (iv2**2 * T2 - iv1**2 * T1) / (T2 - T1)
                    if fwd_var > 0:
                        fwd_vol = sqrt(fwd_var) * 100  # Convert back to %

            # Richness: how much this IV deviates from average
            # Positive = this tenor is expensive, Negative = this tenor is cheap
            richness = ((iv - avg_iv) / avg_iv) * 100

            # Signal based on richness
            if richness > 3:
                signal = 'RICH - SELL'
            elif richness < -3:
                signal = 'CHEAP - BUY'
            else:
                signal = 'FAIR'

            term_structure.append({
                'expiry': expiry,
                'dte': dte,
                'atm_iv': iv,
                'strike': data['strike'],
                'fwd_vol': fwd_vol,  # Forward vol to this expiry from previous
                'richness': richness,
                'signal': signal
            })

    # Build theta comparison table - sorted by absolute difference (largest variance first)
    theta_comparison = []
    total_theta_deribit = 0
    for pa in position_analyses:
        theta_comparison.append({
            'instrument': pa.instrument,
            'direction': pa.direction,
            'size': pa.size,
            'strike': pa.strike,
            'expiry': pa.expiry,
            'dte': pa.dte,
            'theta_bs': pa.theta,  # Our BS-calculated theta
            'theta_deribit': pa.theta_deribit,  # Deribit's theta
            'theta_diff': pa.theta_diff,  # Difference (our - deribit)
            'theta_diff_pct': pa.theta_diff_pct,  # % difference
            'abs_diff': abs(pa.theta_diff)  # For sorting
        })
        total_theta_deribit += pa.theta_deribit

    # Sort by absolute difference (largest variance first)
    theta_comparison.sort(key=lambda x: x['abs_diff'], reverse=True)

    # Add totals summary at the end
    theta_comparison_summary = {
        'total_theta_bs': total_theta,
        'total_theta_deribit': total_theta_deribit,
        'total_diff': total_theta - total_theta_deribit,
        'total_diff_pct': ((total_theta - total_theta_deribit) / abs(total_theta_deribit) * 100) if total_theta_deribit != 0 else 0
    }

    return PortfolioAnalysis(
        total_delta=total_delta,
        total_gamma=total_gamma,
        total_theta=total_theta,
        total_vega=total_vega,
        total_vanna=total_vanna,
        total_volga=total_volga,
        total_gamma_cost=total_gamma_cost,
        total_vanna_cost=total_vanna_cost,
        total_volga_cost=total_volga_cost,
        gamma_1pct=gamma_1pct,
        dollar_gamma=dollar_gamma,
        daily_theta_usd=daily_theta_usd,
        break_even_move=break_even_move,
        avg_iv=avg_iv,
        expected_gamma_pnl=expected_gamma_pnl,
        fair_gamma_pnl=fair_gamma_pnl,
        net_theta_usd=net_theta_usd,
        bumped_theta=bumped_theta,
        theta_gamma_ratio=theta_gamma_ratio,
        btc_price=btc_price,
        realized_vol_30d=realized_vol,
        vol_of_vol=vol_of_vol,
        spot_vol_corr=spot_vol_corr,
        dvol=dvol,
        positions=[asdict(p) for p in position_analyses],
        vol_signal=vol_signal,
        vol_spread=vol_spread,
        parity_violations=parity_violations,
        vol_surface=vol_surface,
        box_spread_arbs=box_spread_arbs,
        butterfly_arbs=butterfly_arbs,
        term_structure=term_structure,
        theta_comparison={'positions': theta_comparison, 'summary': theta_comparison_summary}
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Cost Model Gamma Tracker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CredentialsInput(BaseModel):
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    testnet: Optional[bool] = None
    predicted_dvol: Optional[float] = None


class ManualPositionInput(BaseModel):
    instrument_name: str
    size: float
    direction: str
    mark_iv: float = 50.0
    mark_price: float = 0.0


class ManualPortfolioInput(BaseModel):
    positions: List[ManualPositionInput]
    predicted_dvol: Optional[float] = None
    btc_price: Optional[float] = None  # For scenario analysis


class CostModelParams(BaseModel):
    vol_of_vol: float = 60.0
    spot_vol_corr: float = -0.5


@app.get("/api/market-data")
async def get_market_data():
    """Get public market data with realized vol calculations"""
    try:
        async with DeribitClient() as client:
            btc_price = await client.get_index_price("BTC")
            ohlc = await client.get_tradingview_chart_data("BTC-PERPETUAL", "D")
            
            closes = ohlc.get("close", [])
            realized_vol, returns = calculate_realized_volatility(closes, window=30)
            
            try:
                dvol_data = await client.get_volatility_index("BTC")
                dvol_values = [d[1] for d in dvol_data.get("data", []) if d[1] > 0]
                current_dvol = dvol_values[-1] if dvol_values else 50
                vol_of_vol = calculate_vol_of_vol(dvol_values, window=30)
            except:
                current_dvol = 50
                vol_of_vol = 60
            
            return {
                "btc_price": btc_price,
                "realized_vol_30d": realized_vol,
                "dvol": current_dvol,
                "vol_of_vol": vol_of_vol,
                "daily_returns": returns[-30:],
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/portfolio")
async def get_portfolio(credentials: CredentialsInput, params: CostModelParams = CostModelParams()):
    """Get portfolio with full Cost Model analysis"""
    # Use defaults from env vars if not provided
    client_id = credentials.client_id or DEFAULT_DERIBIT_CLIENT_ID
    client_secret = credentials.client_secret or DEFAULT_DERIBIT_CLIENT_SECRET
    testnet_flag = credentials.testnet if credentials.testnet is not None else DEFAULT_DERIBIT_TESTNET
    
    if not client_id or not client_secret:
        raise HTTPException(status_code=400, detail="Deribit credentials required. Set DERIBIT_CLIENT_ID and DERIBIT_CLIENT_SECRET env vars or provide in request.")
    
    try:
        async with DeribitClient(client_id, client_secret, testnet_flag) as client:
            # Fetch data
            btc_price = await client.get_index_price("BTC")
            positions = await client.get_positions("BTC")
            ohlc = await client.get_tradingview_chart_data("BTC-PERPETUAL", "D")
            
            closes = ohlc.get("close", [])
            realized_vol, returns = calculate_realized_volatility(closes, window=30)
            
            try:
                dvol_data = await client.get_volatility_index("BTC")
                dvol_values = [d[1] for d in dvol_data.get("data", []) if d[1] > 0]
                current_dvol = dvol_values[-1] if dvol_values else 50
                vol_of_vol_calc = calculate_vol_of_vol(dvol_values, window=30)
            except:
                current_dvol = 50
                vol_of_vol_calc = params.vol_of_vol
            
            # Analyze with Cost Model
            analysis = await analyze_portfolio_with_cost_model(
                positions=positions,
                btc_price=btc_price,
                realized_vol=realized_vol,
                vol_of_vol=vol_of_vol_calc or params.vol_of_vol,
                spot_vol_corr=params.spot_vol_corr,
                dvol=current_dvol,
                predicted_dvol=credentials.predicted_dvol,
                client=client  # Pass client to fetch order book IVs
            )
            
            return {
                "portfolio": asdict(analysis),
                "daily_returns": returns[-30:],
                "ohlc": {
                    "timestamps": ohlc.get("ticks", [])[-60:],
                    "closes": closes[-60:]
                },
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-manual")
async def analyze_manual(portfolio: ManualPortfolioInput, params: CostModelParams = CostModelParams()):
    """Analyze manually entered positions with Cost Model"""
    try:
        async with DeribitClient() as client:
            btc_price = await client.get_index_price("BTC")
            ohlc = await client.get_tradingview_chart_data("BTC-PERPETUAL", "D")
            
            closes = ohlc.get("close", [])
            realized_vol, returns = calculate_realized_volatility(closes, window=30)
            
            # Convert to expected format
            positions = [{
                'instrument_name': p.instrument_name,
                'size': p.size,
                'direction': p.direction,
                'mark_iv': p.mark_iv,
                'mark_price': p.mark_price / btc_price if p.mark_price > 0 else 0
            } for p in portfolio.positions]
            
            try:
                dvol_data = await client.get_volatility_index("BTC")
                dvol_values = [d[1] for d in dvol_data.get("data", []) if d[1] > 0]
                current_dvol = dvol_values[-1] if dvol_values else 50
            except:
                current_dvol = 50
            
            analysis = await analyze_portfolio_with_cost_model(
                positions=positions,
                btc_price=btc_price,
                realized_vol=realized_vol,
                vol_of_vol=params.vol_of_vol,
                spot_vol_corr=params.spot_vol_corr,
                dvol=current_dvol,
                predicted_dvol=portfolio.predicted_dvol,
                client=client  # Pass client to fetch order book IVs
            )
            
            return {
                "portfolio": asdict(analysis),
                "daily_returns": returns[-30:],
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cost-model-surface")
async def get_cost_model_surface(
    btc_price: float = Query(100000),
    base_iv: float = Query(50),
    vol_of_vol: float = Query(60),
    spot_vol_corr: float = Query(-0.5)
):
    """
    Generate Cost Model implied volatility surface
    Returns theta decomposition across strikes and expiries
    """
    strikes = [btc_price * m for m in [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]]
    dtes = [7, 14, 30, 60, 90, 180]
    
    surface_data = []
    
    for dte in dtes:
        T = dte / 365
        for strike in strikes:
            moneyness = log(strike / btc_price) / (base_iv / 100 * sqrt(T)) if T > 0 else 0
            
            # Simple skew model: IV = base_iv - skew_slope * moneyness
            skew_slope = 0.05  # 5% per unit moneyness
            iv = base_iv / 100 - skew_slope * moneyness
            iv = max(0.1, min(2.0, iv))  # Bound IV
            
            greeks = calculate_full_greeks(btc_price, strike, T, 0.05, iv, 'call')
            decomp = decompose_theta_cost_model(
                greeks, btc_price, T, iv,
                vol_of_vol=vol_of_vol / 100,
                spot_vol_corr=spot_vol_corr
            )
            
            surface_data.append({
                'strike': strike,
                'dte': dte,
                'moneyness': moneyness,
                'iv': iv * 100,
                'delta': greeks.delta,
                'gamma': greeks.gamma,
                'vanna': greeks.vanna,
                'volga': greeks.volga,
                'theta': greeks.theta,
                'gamma_cost': decomp.gamma_cost,
                'vanna_cost': decomp.vanna_cost,
                'volga_cost': decomp.volga_cost,
                'total_cost_model': decomp.total_cost_model
            })
    
    return {
        'surface': surface_data,
        'parameters': {
            'btc_price': btc_price,
            'base_iv': base_iv,
            'vol_of_vol': vol_of_vol,
            'spot_vol_corr': spot_vol_corr
        }
    }


@app.get("/api/market-arb-scan")
async def scan_market_arbitrage():
    """
    Scan entire Deribit BTC options market for arbitrage opportunities:
    1. Butterfly convexity violations
    2. Box spread mispricings
    3. Term structure analysis across all expiries
    """
    try:
        async with DeribitClient() as client:
            # Fetch all option book summaries (public endpoint - no auth needed)
            btc_price = await client.get_index_price("BTC")
            book_summaries = await client.get_book_summary_by_currency("BTC", "option")

            # Parse and organize options data
            options_by_expiry = {}  # {expiry: {strike: {'call': {...}, 'put': {...}}}}
            expiry_atm_ivs = {}  # For term structure

            for book in book_summaries:
                instrument = book.get('instrument_name', '')
                parsed = parse_instrument(instrument)
                if not parsed:
                    continue

                mark_price_btc = book.get('mark_price', 0)
                mark_iv = book.get('mark_iv', 0)
                bid_price = book.get('bid_price') or 0
                ask_price = book.get('ask_price') or 0

                # Skip options with no market (no IV or very wide/no quotes)
                if mark_iv is None or mark_iv <= 0:
                    continue

                # Convert mark price to USD
                mark_price_usd = mark_price_btc * btc_price

                expiry = parsed['expiry']
                strike = parsed['strike']
                opt_type = parsed['option_type']
                dte = parsed['dte']

                if expiry not in options_by_expiry:
                    options_by_expiry[expiry] = {}
                if strike not in options_by_expiry[expiry]:
                    options_by_expiry[expiry][strike] = {'call': None, 'put': None, 'dte': dte}

                options_by_expiry[expiry][strike][opt_type] = {
                    'instrument': instrument,
                    'mark_price_usd': mark_price_usd,
                    'mark_iv': mark_iv,
                    'bid_price': bid_price * btc_price if bid_price else 0,
                    'ask_price': ask_price * btc_price if ask_price else 0,
                    'dte': dte
                }

                # Track ATM IV for term structure using OTM options only
                # For reliable IV: use OTM call (strike > spot) and OTM put (strike < spot)
                if expiry not in expiry_atm_ivs:
                    expiry_atm_ivs[expiry] = {
                        'dte': dte,
                        'otm_call': None,  # Closest OTM call (strike > spot)
                        'otm_put': None,   # Closest OTM put (strike < spot)
                    }

                # Track closest OTM call (above spot)
                if opt_type == 'call' and strike > btc_price:
                    call_moneyness = strike / btc_price - 1
                    if expiry_atm_ivs[expiry]['otm_call'] is None or call_moneyness < expiry_atm_ivs[expiry]['otm_call']['moneyness']:
                        expiry_atm_ivs[expiry]['otm_call'] = {
                            'iv': mark_iv,
                            'strike': strike,
                            'moneyness': call_moneyness
                        }

                # Track closest OTM put (below spot)
                if opt_type == 'put' and strike < btc_price:
                    put_moneyness = 1 - strike / btc_price
                    if expiry_atm_ivs[expiry]['otm_put'] is None or put_moneyness < expiry_atm_ivs[expiry]['otm_put']['moneyness']:
                        expiry_atm_ivs[expiry]['otm_put'] = {
                            'iv': mark_iv,
                            'strike': strike,
                            'moneyness': put_moneyness
                        }

            # ========================================
            # MARKET-WIDE BUTTERFLY SCAN
            # ========================================
            butterfly_arbs = []
            r = 0.05  # Risk-free rate

            for expiry, strikes_data in options_by_expiry.items():
                strikes = sorted(strikes_data.keys())

                for opt_type in ['call', 'put']:
                    # Get all strikes that have this option type with valid prices
                    # IMPORTANT: Only use OTM options for reliable pricing
                    # Puts: strikes BELOW spot (OTM puts)
                    # Calls: strikes ABOVE spot (OTM calls)
                    if opt_type == 'put':
                        valid_strikes = [k for k in strikes
                                        if k < btc_price  # OTM puts only
                                        and strikes_data[k].get(opt_type)
                                        and strikes_data[k][opt_type]['mark_price_usd'] > 0]
                    else:  # call
                        valid_strikes = [k for k in strikes
                                        if k > btc_price  # OTM calls only
                                        and strikes_data[k].get(opt_type)
                                        and strikes_data[k][opt_type]['mark_price_usd'] > 0]

                    # Check consecutive triplets
                    for i in range(len(valid_strikes) - 2):
                        k1 = valid_strikes[i]
                        k2 = valid_strikes[i + 1]
                        k3 = valid_strikes[i + 2]

                        # Check if evenly spaced (within 5% for market-wide scan)
                        spread1 = k2 - k1
                        spread2 = k3 - k2
                        if spread1 <= 0 or spread2 <= 0:
                            continue
                        if abs(spread1 - spread2) / min(spread1, spread2) > 0.05:
                            continue

                        o1 = strikes_data[k1][opt_type]
                        o2 = strikes_data[k2][opt_type]
                        o3 = strikes_data[k3][opt_type]

                        # Use mid prices if available, else mark
                        p1 = (o1['bid_price'] + o1['ask_price']) / 2 if o1['bid_price'] > 0 and o1['ask_price'] > 0 else o1['mark_price_usd']
                        p2 = (o2['bid_price'] + o2['ask_price']) / 2 if o2['bid_price'] > 0 and o2['ask_price'] > 0 else o2['mark_price_usd']
                        p3 = (o3['bid_price'] + o3['ask_price']) / 2 if o3['bid_price'] > 0 and o3['ask_price'] > 0 else o3['mark_price_usd']

                        # Convexity check: C(K1) + C(K3) >= 2*C(K2)
                        wings_cost = p1 + p3
                        body_cost = 2 * p2
                        violation = body_cost - wings_cost

                        # Flag significant violations (>$20 for market scan)
                        if violation > 20:
                            butterfly_arbs.append({
                                'expiry': expiry,
                                'dte': o1['dte'],
                                'option_type': opt_type,
                                'k1': k1,
                                'k2': k2,
                                'k3': k3,
                                'p1': p1,
                                'p2': p2,
                                'p3': p3,
                                'wings_cost': wings_cost,
                                'body_cost': body_cost,
                                'violation': violation,
                                'action': f'BUY {k1}/{k3} SELL 2x{k2}'
                            })

            # Sort by violation size (biggest first)
            butterfly_arbs.sort(key=lambda x: -x['violation'])

            # ========================================
            # MARKET-WIDE BOX SPREAD SCAN
            # ========================================
            box_arbs = []

            for expiry, strikes_data in options_by_expiry.items():
                # Find strikes with both calls and puts
                strikes_with_both = [k for k in strikes_data.keys()
                                     if strikes_data[k].get('call') and strikes_data[k].get('put')
                                     and strikes_data[k]['call']['mark_price_usd'] > 0
                                     and strikes_data[k]['put']['mark_price_usd'] > 0]
                strikes_with_both = sorted(strikes_with_both)

                dte = strikes_data[strikes_with_both[0]]['dte'] if strikes_with_both else 0
                T = dte / 365

                # Check pairs of strikes
                for i, k1 in enumerate(strikes_with_both):
                    for k2 in strikes_with_both[i+1:]:
                        c1 = strikes_data[k1]['call']
                        c2 = strikes_data[k2]['call']
                        p1 = strikes_data[k1]['put']
                        p2 = strikes_data[k2]['put']

                        # Use mid prices
                        c1_price = (c1['bid_price'] + c1['ask_price']) / 2 if c1['bid_price'] > 0 and c1['ask_price'] > 0 else c1['mark_price_usd']
                        c2_price = (c2['bid_price'] + c2['ask_price']) / 2 if c2['bid_price'] > 0 and c2['ask_price'] > 0 else c2['mark_price_usd']
                        p1_price = (p1['bid_price'] + p1['ask_price']) / 2 if p1['bid_price'] > 0 and p1['ask_price'] > 0 else p1['mark_price_usd']
                        p2_price = (p2['bid_price'] + p2['ask_price']) / 2 if p2['bid_price'] > 0 and p2['ask_price'] > 0 else p2['mark_price_usd']

                        # Box value: (K2-K1) * e^(-rT)
                        theoretical = (k2 - k1) * exp(-r * T)

                        # Market cost: C(K1) - C(K2) - P(K1) + P(K2)
                        market_cost = c1_price - c2_price - p1_price + p2_price

                        arb_profit = theoretical - market_cost
                        arb_pct = (arb_profit / theoretical) * 100 if theoretical > 0 else 0

                        # Flag if > 0.3% arb (tighter threshold for market scan)
                        if abs(arb_pct) > 0.3:
                            box_arbs.append({
                                'expiry': expiry,
                                'dte': dte,
                                'k1': k1,
                                'k2': k2,
                                'theoretical': theoretical,
                                'market_cost': market_cost,
                                'arb_profit': arb_profit,
                                'arb_pct': arb_pct,
                                'action': 'BUY BOX' if arb_profit > 0 else 'SELL BOX'
                            })

            # Sort by absolute arb percentage
            box_arbs.sort(key=lambda x: -abs(x['arb_pct']))

            # ========================================
            # MARKET-WIDE TERM STRUCTURE (using OTM options only)
            # ========================================
            # First, compute synthetic ATM IV from OTM call + OTM put average
            processed_expiries = []
            for expiry, data in expiry_atm_ivs.items():
                otm_call = data.get('otm_call')
                otm_put = data.get('otm_put')

                # Need at least one OTM option, prefer average of both
                if otm_call and otm_put:
                    # Average of closest OTM call and put = synthetic ATM
                    atm_iv = (otm_call['iv'] + otm_put['iv']) / 2
                    atm_strike = (otm_call['strike'] + otm_put['strike']) / 2
                elif otm_call:
                    atm_iv = otm_call['iv']
                    atm_strike = otm_call['strike']
                elif otm_put:
                    atm_iv = otm_put['iv']
                    atm_strike = otm_put['strike']
                else:
                    continue  # Skip if no OTM options

                processed_expiries.append({
                    'expiry': expiry,
                    'dte': data['dte'],
                    'atm_iv': atm_iv,
                    'atm_strike': atm_strike,
                    'otm_call_iv': otm_call['iv'] if otm_call else None,
                    'otm_put_iv': otm_put['iv'] if otm_put else None
                })

            sorted_expiries = sorted(processed_expiries, key=lambda x: x['dte'])
            term_structure = []

            if len(sorted_expiries) > 0:
                avg_iv = sum(d['atm_iv'] for d in sorted_expiries) / len(sorted_expiries)

                for i, data in enumerate(sorted_expiries):
                    dte = data['dte']
                    T = dte / 365
                    iv = data['atm_iv']

                    # Forward vol calculation
                    fwd_vol = None
                    if i > 0:
                        prev_data = sorted_expiries[i-1]
                        T1 = prev_data['dte'] / 365
                        T2 = T
                        iv1 = prev_data['atm_iv'] / 100
                        iv2 = iv / 100

                        if T2 > T1:
                            fwd_var = (iv2**2 * T2 - iv1**2 * T1) / (T2 - T1)
                            if fwd_var > 0:
                                fwd_vol = sqrt(fwd_var) * 100

                    richness = ((iv - avg_iv) / avg_iv) * 100

                    if richness > 3:
                        signal = 'RICH - SELL'
                    elif richness < -3:
                        signal = 'CHEAP - BUY'
                    else:
                        signal = 'FAIR'

                    term_structure.append({
                        'expiry': data['expiry'],
                        'dte': dte,
                        'atm_iv': iv,
                        'atm_strike': data['atm_strike'],
                        'otm_call_iv': data['otm_call_iv'],
                        'otm_put_iv': data['otm_put_iv'],
                        'fwd_vol': fwd_vol,
                        'richness': richness,
                        'signal': signal
                    })

            return {
                'btc_price': btc_price,
                'butterfly_arbs': butterfly_arbs[:20],  # Top 20
                'box_arbs': box_arbs[:20],  # Top 20
                'term_structure': term_structure,
                'stats': {
                    'total_options_scanned': len(book_summaries),
                    'expiries_analyzed': len(options_by_expiry),
                    'butterfly_violations_found': len(butterfly_arbs),
                    'box_arb_opportunities': len(box_arbs)
                },
                'timestamp': datetime.now().isoformat()
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/strategy-scanner")
async def strategy_scanner():
    """
    Cost Model Strategy Scanner

    Scans the market for optimal structures based on greek exposures:
    1. CHEAP GAMMA: High gamma/theta ratio options (efficient gamma exposure)
    2. RICH THETA: Options where theta is rich relative to expected variance
    3. VANNA TRADES: Risk reversals and skew plays
    4. VOLGA TRADES: Butterfly and strangle ideas for vol-of-vol exposure
    5. CALENDAR SPREADS: Term structure opportunities
    """
    try:
        async with DeribitClient() as client:
            # Get BTC price
            btc_price = await client.get_index_price("BTC")

            # Get DVOL for market vol estimate
            dvol_data = await client.get_volatility_index("BTC")
            dvol = dvol_data.get('volatility', 50) / 100  # Convert to decimal

            # Fetch all options
            book_summaries = await client.get_book_summary_by_currency("BTC", "option")

            # Process each option with Cost Model
            options_analyzed = []
            r = 0.05
            vol_of_vol = 0.6  # ~60% vol of vol for crypto
            spot_vol_corr = -0.5  # Negative correlation

            for opt in book_summaries:
                instrument = opt.get('instrument_name', '')
                if not instrument.startswith('BTC-'):
                    continue

                mark_iv = opt.get('mark_iv')
                mark_price = opt.get('mark_price')
                bid_iv = opt.get('bid_iv')
                ask_iv = opt.get('ask_iv')

                if not mark_iv or mark_iv <= 0 or not mark_price:
                    continue

                # Parse instrument
                parts = instrument.split('-')
                if len(parts) < 4:
                    continue

                expiry_str = parts[1]
                try:
                    strike = float(parts[2])
                    opt_type = 'call' if parts[3] == 'C' else 'put'
                except:
                    continue

                # Calculate DTE
                try:
                    expiry_date = datetime.strptime(expiry_str, "%d%b%y")
                    dte = (expiry_date - datetime.now()).days
                    if dte <= 0:
                        continue
                except:
                    continue

                T = dte / 365
                sigma = mark_iv / 100

                # Only analyze OTM options for reliability
                if opt_type == 'call' and strike <= btc_price:
                    continue
                if opt_type == 'put' and strike >= btc_price:
                    continue

                # Calculate Greeks
                greeks = calculate_full_greeks(btc_price, strike, T, r, sigma, opt_type)

                # Calculate Cost Model decomposition
                decomp = decompose_theta_cost_model(greeks, btc_price, T, sigma, vol_of_vol, spot_vol_corr)

                # Calculate efficiency metrics
                mark_price_usd = mark_price * btc_price
                # greeks.theta is already in USD (BSM computed with S in USD)
                daily_theta_usd = greeks.theta

                # Gamma efficiency: gamma per dollar of daily theta
                gamma_efficiency = abs(greeks.gamma / daily_theta_usd) if daily_theta_usd != 0 else 0

                # Break-even move (%) - from Cost Model: theta = 0.5 * gamma * S^2 * (move%)^2
                # So move% = sqrt(2 * |theta| / (gamma * S^2))
                if greeks.gamma != 0 and btc_price != 0:
                    break_even = sqrt(abs(2 * daily_theta_usd / (greeks.gamma * btc_price * btc_price))) * 100
                else:
                    break_even = 0

                # Moneyness
                moneyness = (strike / btc_price - 1) * 100

                # IV spread (if bid/ask available)
                iv_spread = (ask_iv - bid_iv) if bid_iv and ask_iv else None

                # ========================================
                # BUMPED THETA WITH VARIANCE
                # Expected daily P&L = E[V(S±σ, T-1)] - V(S, T)
                # This captures time decay + expected gamma P&L from 1-day variance
                # ========================================
                T_bumped = max((dte - 1) / 365, 0.001)  # Tomorrow
                daily_vol = sigma / sqrt(365)  # Daily vol in decimal
                expected_spot_move = btc_price * daily_vol  # 1 std dev daily move

                # Current value
                current_value = greeks.price

                # Up move: V(S+σ, T-1)
                up_greeks = calculate_full_greeks(btc_price + expected_spot_move, strike, T_bumped, r, sigma, opt_type)
                up_value = up_greeks.price

                # Down move: V(S-σ, T-1)
                down_greeks = calculate_full_greeks(btc_price - expected_spot_move, strike, T_bumped, r, sigma, opt_type)
                down_value = down_greeks.price

                # Expected P&L with variance: average of up/down minus current
                expected_daily_pnl = 0.5 * (up_value + down_value) - current_value

                # Theta/Variance Ratio: expected_daily_pnl / BS_theta
                # < 1.0 = cheap (gamma earns more than theta costs)
                # > 1.0 = rich (theta costs more than gamma earns)
                # ~1.0 = fair
                if abs(daily_theta_usd) > 0.01:
                    theta_variance_ratio = expected_daily_pnl / daily_theta_usd
                else:
                    theta_variance_ratio = 1.0

                options_analyzed.append({
                    'instrument': instrument,
                    'expiry': expiry_str,
                    'dte': dte,
                    'strike': strike,
                    'type': opt_type,
                    'moneyness': moneyness,
                    'mark_iv': mark_iv,
                    'mark_price_usd': mark_price_usd,
                    'iv_spread': iv_spread,
                    # Greeks
                    'delta': greeks.delta,
                    'gamma': greeks.gamma,
                    'theta': greeks.theta,
                    'vega': greeks.vega,
                    'vanna': greeks.vanna,
                    'volga': greeks.volga,
                    # Cost Model
                    'gamma_cost': decomp.gamma_cost,
                    'vanna_cost': decomp.vanna_cost,
                    'volga_cost': decomp.volga_cost,
                    # Efficiency metrics
                    'gamma_efficiency': gamma_efficiency,
                    'break_even': break_even,
                    'daily_theta_usd': daily_theta_usd,
                    # Bumped theta with variance
                    'expected_daily_pnl': expected_daily_pnl,
                    'theta_variance_ratio': theta_variance_ratio
                })

            # ========================================
            # STRATEGY 1: CHEAP GAMMA (BUY candidates)
            # Lowest theta_variance_ratio = best value for long gamma
            # Ratio < 1.0 means gamma earns more than theta costs
            # ========================================
            cheap_gamma = sorted(
                [o for o in options_analyzed if o['dte'] <= 30 and abs(o['moneyness']) < 15],
                key=lambda x: x['theta_variance_ratio']  # Lowest ratio = cheapest
            )[:10]

            # ========================================
            # STRATEGY 2: RICH THETA (SELL candidates)
            # Highest theta_variance_ratio = best for selling
            # Ratio > 1.0 means theta collects more than gamma gives up
            # ========================================
            rich_theta = sorted(
                [o for o in options_analyzed
                 if o['dte'] >= 14 and o['dte'] <= 60
                 and abs(o['moneyness']) > 10 and abs(o['moneyness']) < 30
                 and abs(o['daily_theta_usd']) > 5],
                key=lambda x: -x['theta_variance_ratio']  # Highest ratio = richest
            )[:10]

            # ========================================
            # STRATEGY 3: VANNA TRADES (Risk Reversals)
            # Risk reversals: Long OTM call + Short OTM put = net positive vanna
            # Vanna = dDelta/dVol = dVega/dSpot
            # With negative spot-vol correlation (spot up -> vol down):
            # - Short put benefits from vol drop + delta moving toward 0
            # - Long call: delta gain partially offset by vol drop
            # Net effect: structure profits from the correlation
            # Really trading skew differential (put IV vs call IV)
            # ========================================
            vanna_trades = []

            # Group by expiry
            by_expiry = {}
            for opt in options_analyzed:
                if opt['expiry'] not in by_expiry:
                    by_expiry[opt['expiry']] = {'calls': [], 'puts': []}
                if opt['type'] == 'call':
                    by_expiry[opt['expiry']]['calls'].append(opt)
                else:
                    by_expiry[opt['expiry']]['puts'].append(opt)

            for expiry, data in by_expiry.items():
                calls = sorted(data['calls'], key=lambda x: x['moneyness'])
                puts = sorted(data['puts'], key=lambda x: -x['moneyness'])

                # Find ~10% OTM call and ~10% OTM put
                otm_call = next((c for c in calls if c['moneyness'] > 8 and c['moneyness'] < 15), None)
                otm_put = next((p for p in puts if p['moneyness'] < -8 and p['moneyness'] > -15), None)

                if otm_call and otm_put:
                    # Risk reversal: Long call, short put
                    net_vanna = otm_call['vanna'] - otm_put['vanna']
                    net_theta = otm_call['daily_theta_usd'] - otm_put['daily_theta_usd']
                    net_delta = otm_call['delta'] - otm_put['delta']
                    call_iv = otm_call['mark_iv']
                    put_iv = otm_put['mark_iv']
                    skew = put_iv - call_iv  # Positive = puts richer

                    vanna_trades.append({
                        'expiry': expiry,
                        'dte': otm_call['dte'],
                        'call_strike': otm_call['strike'],
                        'put_strike': otm_put['strike'],
                        'call_iv': call_iv,
                        'put_iv': put_iv,
                        'skew': skew,
                        'net_vanna': net_vanna,
                        'net_theta_usd': net_theta,
                        'net_delta': net_delta,
                        'structure': f"Long {int(otm_call['strike'])}C / Short {int(otm_put['strike'])}P",
                        'rationale': 'Long skew' if skew > 3 else 'Short skew' if skew < -3 else 'Neutral skew'
                    })

            vanna_trades = sorted(vanna_trades, key=lambda x: -abs(x['skew']))[:10]

            # ========================================
            # STRATEGY 4: VOLGA TRADES (Vol of Vol exposure)
            # Strangles and butterflies for smile convexity
            # ========================================
            volga_trades = []

            for expiry, data in by_expiry.items():
                calls = sorted(data['calls'], key=lambda x: x['moneyness'])
                puts = sorted(data['puts'], key=lambda x: -x['moneyness'])

                # Find ~15-20% OTM strangle
                wide_call = next((c for c in calls if c['moneyness'] > 15 and c['moneyness'] < 25), None)
                wide_put = next((p for p in puts if p['moneyness'] < -15 and p['moneyness'] > -25), None)

                if wide_call and wide_put:
                    # Long strangle = long volga
                    total_volga = wide_call['volga'] + wide_put['volga']
                    total_theta = wide_call['daily_theta_usd'] + wide_put['daily_theta_usd']
                    total_cost = wide_call['mark_price_usd'] + wide_put['mark_price_usd']
                    avg_iv = (wide_call['mark_iv'] + wide_put['mark_iv']) / 2

                    # Volga efficiency: volga per dollar of theta decay
                    volga_efficiency = total_volga / abs(total_theta) if total_theta != 0 else 0

                    volga_trades.append({
                        'expiry': expiry,
                        'dte': wide_call['dte'],
                        'call_strike': wide_call['strike'],
                        'put_strike': wide_put['strike'],
                        'call_iv': wide_call['mark_iv'],
                        'put_iv': wide_put['mark_iv'],
                        'avg_iv': avg_iv,
                        'total_volga': total_volga,
                        'total_theta_usd': total_theta,
                        'total_cost_usd': total_cost,
                        'volga_efficiency': volga_efficiency,
                        'structure': f"Long {int(wide_put['strike'])}P / {int(wide_call['strike'])}C Strangle",
                        'rationale': 'Long vol-of-vol' if volga_efficiency > 0.1 else 'Expensive vol-of-vol'
                    })

            volga_trades = sorted(volga_trades, key=lambda x: -x['volga_efficiency'])[:10]

            # ========================================
            # STRATEGY 5: CALENDAR SPREADS
            # Based on term structure from earlier scan
            # ========================================
            calendar_ideas = []

            # Find ATM options across expiries for calendar analysis
            atm_by_expiry = {}
            for opt in options_analyzed:
                if abs(opt['moneyness']) < 8:  # Near ATM
                    exp = opt['expiry']
                    if exp not in atm_by_expiry or abs(opt['moneyness']) < abs(atm_by_expiry[exp]['moneyness']):
                        atm_by_expiry[exp] = opt

            sorted_atm = sorted(atm_by_expiry.values(), key=lambda x: x['dte'])

            for i in range(len(sorted_atm) - 1):
                front = sorted_atm[i]
                back = sorted_atm[i + 1]

                iv_diff = back['mark_iv'] - front['mark_iv']
                theta_diff = back['daily_theta_usd'] - front['daily_theta_usd']

                # Calendar: sell front, buy back
                net_theta = -front['daily_theta_usd'] + back['daily_theta_usd']  # Positive if front decays faster

                calendar_ideas.append({
                    'front_expiry': front['expiry'],
                    'back_expiry': back['expiry'],
                    'front_dte': front['dte'],
                    'back_dte': back['dte'],
                    'front_iv': front['mark_iv'],
                    'back_iv': back['mark_iv'],
                    'iv_diff': iv_diff,
                    'net_theta_usd': net_theta,
                    'strike': front['strike'],
                    'structure': f"Sell {front['expiry']} / Buy {back['expiry']} {int(front['strike'])}",
                    'rationale': 'Front IV rich' if iv_diff < -2 else 'Back IV rich' if iv_diff > 2 else 'Neutral'
                })

            return {
                'btc_price': btc_price,
                'dvol': dvol * 100,
                'options_analyzed': len(options_analyzed),
                'strategies': {
                    'cheap_gamma': {
                        'description': 'Efficient long gamma - high gamma per theta dollar',
                        'use_case': 'Long gamma with minimal bleed',
                        'trades': cheap_gamma
                    },
                    'rich_theta': {
                        'description': 'Short vol candidates - high theta relative to gamma risk',
                        'use_case': 'Harvest theta premium on OTM options',
                        'trades': rich_theta
                    },
                    'vanna_trades': {
                        'description': 'Risk reversals for skew exposure',
                        'use_case': 'Trade spot-vol correlation',
                        'trades': vanna_trades
                    },
                    'volga_trades': {
                        'description': 'Strangles for vol-of-vol exposure',
                        'use_case': 'Long convexity / smile exposure',
                        'trades': volga_trades
                    },
                    'calendar_spreads': {
                        'description': 'Term structure plays',
                        'use_case': 'Trade rich/cheap maturities',
                        'trades': calendar_ideas
                    }
                },
                'timestamp': datetime.now().isoformat()
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# KELLY-EDGE TRADE SCANNER (BTC Options Dashboard)
# ============================================================

@dataclass
class RiskLimits:
    """Portfolio risk limits configuration"""
    nav: float = 10_000_000  # $10M NAV
    max_delta_pct: float = 0.20  # ±20% of NAV
    max_theta_daily: float = 50_000  # ±$50k/day
    max_gamma_1pct: float = 100_000  # $100k per 1% move
    max_vega_1vol: float = 100_000  # ±$100k per vol point


@dataclass
class TradingFilters:
    """Option filtering parameters"""
    min_delta: float = 0.10
    max_delta: float = 0.50
    max_bid_ask_vol: float = 4.0  # vol points
    min_open_interest: int = 100
    min_daily_volume: int = 50
    min_dte: int = 5
    max_dte: int = 90
    otm_only: bool = True


class DynamicOmegaEstimator:
    """
    Dynamic Omega Estimator following the spec:
    - Ω_G uses predicted vol vs implied vol (not just implied)
    - Ω_Va uses actual spot-vol correlation
    - Ω_Vo uses actual vol-of-vol
    """

    def __init__(self, predicted_vol: float, vol_of_vol: float = 0.60,
                 spot_vol_corr: float = -0.50):
        """
        Args:
            predicted_vol: Predicted DVOL from ML model (percentage)
            vol_of_vol: Vol-of-vol estimate (default 60% for BTC)
            spot_vol_corr: Spot-vol correlation (default -0.5 for BTC)
        """
        self.predicted_vol = predicted_vol / 100  # Convert to decimal
        self.vol_of_vol = vol_of_vol
        self.spot_vol_corr = spot_vol_corr

    def estimate_omega_gamma(self, spot: float, sigma_implied: float) -> float:
        """
        Ω_G = 0.5 × S² × (σ_predicted² - σ_implied²)

        This is the KEY difference from static omega:
        - Positive when predicted > implied (vol is cheap, long gamma profitable)
        - Negative when predicted < implied (vol is rich, short gamma profitable)
        """
        sigma_pred = self.predicted_vol
        sigma_impl = sigma_implied / 100 if sigma_implied > 1 else sigma_implied

        return 0.5 * spot**2 * (sigma_pred**2 - sigma_impl**2)

    def estimate_omega_vanna(self, spot: float) -> float:
        """
        Ω_Va = ρ × σ_spot × σ_DVOL × S

        Using simplified estimate with default parameters.
        In production, would use rolling correlation from historical data.
        """
        # Approximate: σ_spot ≈ predicted_vol, σ_DVOL ≈ vol_of_vol
        return self.spot_vol_corr * self.predicted_vol * self.vol_of_vol * spot

    def estimate_omega_volga(self) -> float:
        """
        Ω_Vo = 0.5 × σ_DVOL²
        """
        return 0.5 * self.vol_of_vol**2

    def get_all_omegas(self, spot: float, sigma_implied: float) -> dict:
        """Get all omega parameters"""
        return {
            'omega_g': self.estimate_omega_gamma(spot, sigma_implied),
            'omega_va': self.estimate_omega_vanna(spot),
            'omega_vo': self.estimate_omega_volga()
        }

    def calculate_expected_daily_pnl(self, spot: float, sigma_implied: float,
                                      gamma: float, vanna: float, volga: float,
                                      theta_market: float) -> dict:
        """
        Expected daily P&L from theta decomposition (Method 3 from spec):

        Expected_PnL = -θ_market + Γ×Ω_G + Va×Ω_Va + Vo×Ω_Vo

        Positive = position generates alpha
        Negative = position bleeds
        """
        omegas = self.get_all_omegas(spot, sigma_implied)

        theta_gamma = gamma * omegas['omega_g']
        theta_vanna = vanna * omegas['omega_va']
        theta_volga = volga * omegas['omega_vo']

        # For a long position: pay theta, earn carry
        expected_pnl = -theta_market + theta_gamma + theta_vanna + theta_volga

        return {
            'expected_daily_pnl': expected_pnl,
            'theta_gamma': theta_gamma,
            'theta_vanna': theta_vanna,
            'theta_volga': theta_volga,
            'theta_market': theta_market,
            'omegas': omegas
        }


class KellyEdgeTradeScanner:
    """
    Trade scanner that ranks options by Kelly-implied optimal edge
    Score = Kelly_fraction × Edge_dollars

    Now uses DynamicOmegaEstimator for spec-compliant omega calculation.
    """

    def __init__(self, predicted_vol: float, prediction_rmse: float = 3.0,
                 kelly_fraction: float = 0.25, nav: float = 10_000_000):
        self.predicted_vol = predicted_vol  # Predicted DVOL from ML model
        self.prediction_rmse = prediction_rmse  # Model RMSE for confidence
        self.kelly_fraction = kelly_fraction  # Fractional Kelly (0.25 = quarter)
        self.nav = nav
        self.risk_limits = RiskLimits(nav=nav)
        self.filters = TradingFilters()

        # Initialize dynamic omega estimator with predicted vol
        self.omega_estimator = DynamicOmegaEstimator(predicted_vol=predicted_vol)

    def calculate_win_probability(self, edge_vol_points: float) -> float:
        """
        Calculate probability that trade is profitable
        Based on prediction confidence interval
        """
        if self.prediction_rmse <= 0:
            return 0.5

        # Z-score of edge relative to prediction uncertainty
        z_score = edge_vol_points / self.prediction_rmse
        return norm_cdf(z_score)

    def calculate_kelly_fraction(self, edge_dollars: float, max_loss: float,
                                  win_probability: float) -> float:
        """
        Kelly fraction = (p × W - q × L) / W
        Using fractional Kelly for safety
        """
        if max_loss <= 0 or edge_dollars <= 0:
            return 0.0

        q = 1 - win_probability
        full_kelly = (win_probability * edge_dollars - q * max_loss) / edge_dollars

        return max(0, full_kelly * self.kelly_fraction)

    def calculate_edge(self, option_data: dict) -> dict:
        """
        Calculate edge using predicted vol vs market IV

        Method: Theoretical vs Market Price via Vega approximation
        Edge_dollars ≈ Vega × (predicted_vol - market_IV)
        """
        market_iv = option_data['mark_iv']
        vega = option_data['vega']

        # Edge in vol points
        edge_vol = self.predicted_vol - market_iv

        # Edge in dollars (vega is per 1% vol move)
        edge_dollars = vega * edge_vol

        # Direction: BUY if vol is cheap (predicted > market), SELL if rich
        direction = 'BUY' if edge_vol > 0 else 'SELL'

        return {
            'edge_vol_points': edge_vol,
            'edge_dollars': edge_dollars,
            'edge_pct': edge_vol / market_iv * 100 if market_iv > 0 else 0,
            'direction': direction
        }

    def calculate_transaction_costs(self, premium: float) -> float:
        """Calculate transaction costs (1.75 bps taker fee)"""
        return abs(premium) * 0.000175

    def score_trade(self, option_data: dict, spot: float) -> dict:
        """
        Score a trade using Kelly × Edge ranking

        Now includes dynamic omega calculations for expected P&L (Method 3 from spec).
        """
        edge = self.calculate_edge(option_data)

        # Transaction costs
        premium = option_data.get('mark_price_usd', 0)
        tx_cost = self.calculate_transaction_costs(premium)

        net_edge = abs(edge['edge_dollars']) - tx_cost

        # Skip if edge doesn't exceed 2x costs
        if net_edge < tx_cost * 2:
            return None

        # Win probability from prediction confidence
        win_prob = self.calculate_win_probability(abs(edge['edge_vol_points']))

        # Adjust for direction
        if edge['direction'] == 'SELL':
            win_prob = 1 - win_prob if edge['edge_vol_points'] > 0 else win_prob

        # Max loss estimation (premium for long, larger for short)
        if edge['direction'] == 'BUY':
            max_loss = premium
        else:
            # For short, use 3x premium as rough max loss estimate
            max_loss = premium * 3

        # Kelly fraction
        kelly = self.calculate_kelly_fraction(net_edge, max_loss, win_prob)

        # Rank score: Kelly × Edge
        rank_score = kelly * net_edge

        # Suggested position size
        max_position_value = self.nav * 0.05  # 5% max per trade
        suggested_contracts = int(max_position_value / max_loss * kelly) if max_loss > 0 else 0

        # Calculate dynamic omega-based expected P&L (Method 3 from spec)
        expected_pnl_data = self.omega_estimator.calculate_expected_daily_pnl(
            spot=spot,
            sigma_implied=option_data['mark_iv'],
            gamma=option_data['gamma'],
            vanna=option_data.get('vanna', 0),
            volga=option_data.get('volga', 0),
            theta_market=option_data['theta']
        )

        # Bumped Theta calculation: V(S, T-1) - V(S, T) for pure time decay
        # This is for a SINGLE option, not a portfolio
        dte = option_data['dte']
        strike = option_data['strike']
        opt_type = option_data['type']
        iv_decimal = option_data['mark_iv'] / 100

        T = dte / 365
        T_bumped = max((dte - 1) / 365, 0.001)  # 1 day forward, avoid div by zero

        # Calculate option value at T (current) and T-1 (tomorrow)
        current_greeks = calculate_full_greeks(spot, strike, T, 0.0, iv_decimal, opt_type)
        tomorrow_greeks = calculate_full_greeks(spot, strike, T_bumped, 0.0, iv_decimal, opt_type)

        # Bumped theta = tomorrow - current (negative for long, time decays value)
        bumped_theta = tomorrow_greeks.price - current_greeks.price

        # BS theta for comparison (already in USD per day)
        bs_theta = option_data['theta']

        # Theta/Gamma ratio: bumped_theta / BS_theta
        # Should be ~1.0x if BS theta is accurate
        # Values far from 1.0 indicate mispricing
        if abs(bs_theta) > 0.001:
            theta_gamma_ratio = bumped_theta / bs_theta
        else:
            theta_gamma_ratio = 0.0

        # Sanity check: for reasonable options, ratio should be 0.8-1.2x
        # If it's way off, something is wrong with the option data

        return {
            'instrument': option_data['instrument'],
            'expiry': option_data['expiry'],
            'dte': option_data['dte'],
            'strike': option_data['strike'],
            'type': option_data['type'],
            'moneyness': option_data['moneyness'],
            'mark_iv': option_data['mark_iv'],
            'mark_price_usd': premium,
            'bid_iv': option_data.get('bid_iv'),
            'ask_iv': option_data.get('ask_iv'),
            'iv_spread': option_data.get('iv_spread'),
            # Greeks
            'delta': option_data['delta'],
            'gamma': option_data['gamma'],
            'theta': option_data['theta'],
            'vega': option_data['vega'],
            'vanna': option_data.get('vanna', 0),
            'volga': option_data.get('volga', 0),
            # Edge calculation (Method 2 from spec)
            'edge_vol_points': edge['edge_vol_points'],
            'edge_dollars': edge['edge_dollars'],
            'edge_pct': edge['edge_pct'],
            'direction': edge['direction'],
            'net_edge': net_edge,
            'tx_cost': tx_cost,
            # Kelly sizing
            'win_probability': win_prob,
            'kelly_fraction': kelly,
            'rank_score': rank_score,
            'suggested_contracts': suggested_contracts,
            # Dynamic Omega-based expected P&L (Method 3 from spec)
            'expected_daily_pnl': expected_pnl_data['expected_daily_pnl'],
            'theta_gamma_dynamic': expected_pnl_data['theta_gamma'],
            'theta_vanna_dynamic': expected_pnl_data['theta_vanna'],
            'theta_volga_dynamic': expected_pnl_data['theta_volga'],
            'omega_g': expected_pnl_data['omegas']['omega_g'],
            'omega_va': expected_pnl_data['omegas']['omega_va'],
            'omega_vo': expected_pnl_data['omegas']['omega_vo'],
            # Static cost model (for comparison)
            'gamma_cost': option_data.get('gamma_cost', 0),
            'vanna_cost': option_data.get('vanna_cost', 0),
            'volga_cost': option_data.get('volga_cost', 0),
            'break_even': option_data.get('break_even', 0),
            # Bumped theta analysis
            'bumped_theta': bumped_theta,
            'theta_gamma_ratio': theta_gamma_ratio
        }

    def passes_filters(self, option_data: dict) -> bool:
        """Check if option passes trading filters"""
        delta = abs(option_data.get('delta', 0))
        if delta < self.filters.min_delta or delta > self.filters.max_delta:
            return False

        dte = option_data.get('dte', 0)
        if dte < self.filters.min_dte or dte > self.filters.max_dte:
            return False

        iv_spread = option_data.get('iv_spread')
        if iv_spread and iv_spread > self.filters.max_bid_ask_vol:
            return False

        if self.filters.otm_only:
            moneyness = option_data.get('moneyness', 0)
            opt_type = option_data.get('type', '')
            if opt_type == 'call' and moneyness < 0:
                return False
            if opt_type == 'put' and moneyness > 0:
                return False

        return True

    def check_risk_limits(self, trade: dict, portfolio_greeks: dict) -> dict:
        """Check if trade would breach portfolio risk limits"""
        sign = 1 if trade['direction'] == 'BUY' else -1
        contracts = trade['suggested_contracts']

        combined = {
            'delta': portfolio_greeks.get('delta', 0) + (trade['delta'] * contracts * sign * 100000),  # Rough BTC delta
            'gamma': portfolio_greeks.get('gamma', 0) + (trade['gamma'] * contracts * sign * 100000),
            'theta': portfolio_greeks.get('theta', 0) + (trade['theta'] * contracts * sign),
            'vega': portfolio_greeks.get('vega', 0) + (trade['vega'] * contracts * sign)
        }

        breaches = []
        limits = self.risk_limits

        if abs(combined['delta']) > limits.nav * limits.max_delta_pct:
            breaches.append('delta')
        if abs(combined['theta']) > limits.max_theta_daily:
            breaches.append('theta')
        if abs(combined['gamma']) > limits.max_gamma_1pct:
            breaches.append('gamma')
        if abs(combined['vega']) > limits.max_vega_1vol:
            breaches.append('vega')

        return {
            'passed': len(breaches) == 0,
            'breaches': breaches,
            'combined_greeks': combined
        }


@app.get("/api/kelly-edge-scanner")
async def kelly_edge_scanner(
    predicted_vol: float = 50.0,
    prediction_rmse: float = 3.0,
    kelly_fraction: float = 0.25,
    nav: float = 10_000_000,
    min_delta: float = 0.10,
    max_delta: float = 0.50,
    min_dte: int = 5,
    max_dte: int = 90,
    max_bid_ask: float = 4.0,
    direction_filter: str = "all"  # "all", "long", "short"
):
    """
    Kelly-Edge Trade Scanner

    Scans all BTC options and ranks them by Kelly × Edge score.
    Uses predicted volatility from ML model to calculate edge.

    Parameters:
    - predicted_vol: Predicted DVOL from your model (%)
    - prediction_rmse: Model RMSE for confidence interval (vol points)
    - kelly_fraction: Fractional Kelly to use (0.25 = quarter Kelly)
    - nav: Portfolio NAV for position sizing ($)
    - min_delta/max_delta: Delta range filter
    - min_dte/max_dte: DTE range filter
    - max_bid_ask: Max bid-ask spread in vol points
    - direction_filter: "all", "long" (buy vol), "short" (sell vol)
    """
    try:
        scanner = KellyEdgeTradeScanner(
            predicted_vol=predicted_vol,
            prediction_rmse=prediction_rmse,
            kelly_fraction=kelly_fraction,
            nav=nav
        )

        # Update filters
        scanner.filters.min_delta = min_delta
        scanner.filters.max_delta = max_delta
        scanner.filters.min_dte = min_dte
        scanner.filters.max_dte = max_dte
        scanner.filters.max_bid_ask_vol = max_bid_ask

        async with DeribitClient() as client:
            # Get BTC price
            btc_price = await client.get_index_price("BTC")

            # Get current DVOL - parse from historical data to get latest value
            dvol_data = await client.get_volatility_index("BTC")
            dvol_values = [d[1] for d in dvol_data.get("data", []) if d[1] > 0]
            current_dvol = dvol_values[-1] if dvol_values else 50

            # Fetch all options
            book_summaries = await client.get_book_summary_by_currency("BTC", "option")

            # Process each option
            r = 0.05
            vol_of_vol = 0.6
            spot_vol_corr = -0.5

            scored_trades = []
            total_scanned = 0
            passed_filters = 0

            for opt in book_summaries:
                instrument = opt.get('instrument_name', '')
                if not instrument.startswith('BTC-'):
                    continue

                mark_iv = opt.get('mark_iv')
                mark_price = opt.get('mark_price')
                bid_iv = opt.get('bid_iv')
                ask_iv = opt.get('ask_iv')

                if not mark_iv or mark_iv <= 0 or not mark_price:
                    continue

                total_scanned += 1

                # Parse instrument
                parts = instrument.split('-')
                if len(parts) < 4:
                    continue

                expiry_str = parts[1]
                try:
                    strike = float(parts[2])
                    opt_type = 'call' if parts[3] == 'C' else 'put'
                except:
                    continue

                # Calculate DTE
                try:
                    expiry_date = datetime.strptime(expiry_str, "%d%b%y")
                    dte = (expiry_date - datetime.now()).days
                    if dte <= 0:
                        continue
                except:
                    continue

                T = dte / 365
                sigma = mark_iv / 100

                # Calculate Greeks
                greeks = calculate_full_greeks(btc_price, strike, T, r, sigma, opt_type)

                # Calculate Cost Model decomposition
                decomp = decompose_theta_cost_model(greeks, btc_price, T, sigma, vol_of_vol, spot_vol_corr)

                # Build option data
                mark_price_usd = mark_price * btc_price
                moneyness = (strike / btc_price - 1) * 100
                iv_spread = (ask_iv - bid_iv) if bid_iv and ask_iv else None

                # Break-even calculation
                if greeks.gamma != 0 and btc_price != 0:
                    break_even = sqrt(abs(2 * greeks.theta / (greeks.gamma * btc_price * btc_price))) * 100
                else:
                    break_even = 0

                option_data = {
                    'instrument': instrument,
                    'expiry': expiry_str,
                    'dte': dte,
                    'strike': strike,
                    'type': opt_type,
                    'moneyness': moneyness,
                    'mark_iv': mark_iv,
                    'mark_price_usd': mark_price_usd,
                    'bid_iv': bid_iv,
                    'ask_iv': ask_iv,
                    'iv_spread': iv_spread,
                    'delta': greeks.delta,
                    'gamma': greeks.gamma,
                    'theta': greeks.theta,
                    'vega': greeks.vega,
                    'vanna': greeks.vanna,
                    'volga': greeks.volga,
                    'gamma_cost': decomp.gamma_cost,
                    'vanna_cost': decomp.vanna_cost,
                    'volga_cost': decomp.volga_cost,
                    'break_even': break_even,
                    'open_interest': opt.get('open_interest', 0),
                    'volume_24h': opt.get('volume', 0)
                }

                # Apply filters
                if not scanner.passes_filters(option_data):
                    continue

                passed_filters += 1

                # Score the trade
                scored = scanner.score_trade(option_data, btc_price)
                if scored:
                    # Apply direction filter
                    if direction_filter == "long" and scored['direction'] != 'BUY':
                        continue
                    if direction_filter == "short" and scored['direction'] != 'SELL':
                        continue

                    scored_trades.append(scored)

            # Sort by rank score (Kelly × Edge)
            scored_trades.sort(key=lambda x: x['rank_score'], reverse=True)

            # Vol regime determination
            vol_edge = predicted_vol - current_dvol
            if vol_edge > 2:
                vol_regime = "CHEAP"
                vol_signal = f"Vol is CHEAP by {vol_edge:.1f} pts - Consider LONG VOL"
            elif vol_edge < -2:
                vol_regime = "RICH"
                vol_signal = f"Vol is RICH by {abs(vol_edge):.1f} pts - Consider SHORT VOL"
            else:
                vol_regime = "FAIR"
                vol_signal = "Vol is FAIRLY PRICED"

            # Separate by direction
            long_trades = [t for t in scored_trades if t['direction'] == 'BUY'][:20]
            short_trades = [t for t in scored_trades if t['direction'] == 'SELL'][:20]

            # Calculate dynamic omega values for visualization
            sigma_pred = predicted_vol / 100
            sigma_impl = current_dvol / 100

            # Raw omega per unit (for reference)
            omega_g_per_unit = 0.5 * btc_price**2 * (sigma_pred**2 - sigma_impl**2) / 365
            omega_va_per_unit = spot_vol_corr * btc_price * sigma_impl * vol_of_vol / 365
            omega_vo_per_unit = 0.5 * vol_of_vol**2 / 365

            # Aggregate expected P&L from TOP 20 TRADES (actual gamma × omega)
            # This is the real expected daily P&L if you traded the top 20
            total_gamma_pnl = 0
            total_vanna_pnl = 0
            total_volga_pnl = 0
            total_market_theta = 0

            for t in scored_trades[:20]:
                # Each trade's expected P&L = Greek × Omega_per_unit
                # Sign depends on direction: BUY = long gamma (+), SELL = short gamma (-)
                direction_sign = 1 if t['direction'] == 'BUY' else -1
                contracts = t.get('suggested_contracts', 1)

                # Gamma P&L: gamma × omega_g × contracts × direction
                gamma = t.get('gamma', 0)
                total_gamma_pnl += gamma * omega_g_per_unit * contracts * direction_sign

                # Vanna P&L
                vanna = t.get('vanna', 0)
                total_vanna_pnl += vanna * omega_va_per_unit * contracts * direction_sign

                # Volga P&L
                volga = t.get('volga', 0)
                total_volga_pnl += volga * omega_vo_per_unit * contracts * direction_sign

                # Market theta (what you pay/receive)
                theta = t.get('theta', 0)
                total_market_theta += theta * contracts * direction_sign

            # Also calculate per-contract omega for a "typical" ATM option
            # Normalized to 1 BTC notional for easier interpretation
            atm_gamma_1btc = 2 / (btc_price * sigma_impl * sqrt(30/365))  # Approx ATM gamma for 30d
            omega_g_per_btc = atm_gamma_1btc * omega_g_per_unit

            # Theta decomposition from cost model (what trades are paying/receiving)
            total_theta_gamma = sum(t.get('gamma_cost', 0) for t in scored_trades[:20])
            total_theta_vanna = sum(t.get('vanna_cost', 0) for t in scored_trades[:20])
            total_theta_volga = sum(t.get('volga_cost', 0) for t in scored_trades[:20])

            # Expected P&L breakdown - ACTUAL aggregate from top trades
            expected_pnl_components = {
                'gamma_carry': total_gamma_pnl,
                'vanna_carry': total_vanna_pnl,
                'volga_carry': total_volga_pnl,
                'total_expected': total_gamma_pnl + total_vanna_pnl + total_volga_pnl
            }

            return {
                'btc_price': btc_price,
                'current_dvol': current_dvol,
                'predicted_vol': predicted_vol,
                'vol_edge': vol_edge,
                'vol_regime': vol_regime,
                'vol_signal': vol_signal,
                'prediction_rmse': prediction_rmse,
                'kelly_fraction': kelly_fraction,
                'nav': nav,
                'summary': {
                    'total_scanned': total_scanned,
                    'passed_filters': passed_filters,
                    'scored_trades': len(scored_trades),
                    'long_vol_trades': len(long_trades),
                    'short_vol_trades': len(short_trades)
                },
                'filters_applied': {
                    'delta_range': [min_delta, max_delta],
                    'dte_range': [min_dte, max_dte],
                    'max_bid_ask_vol': max_bid_ask,
                    'direction': direction_filter
                },
                'risk_limits': {
                    'max_delta': nav * 0.20,
                    'max_theta': 50000,
                    'max_gamma': 100000,
                    'max_vega': 100000
                },
                'omega_analysis': {
                    'omega_g': omega_g_per_unit,  # Per unit gamma (for reference)
                    'omega_va': omega_va_per_unit,
                    'omega_vo': omega_vo_per_unit,
                    'vol_of_vol': vol_of_vol,
                    'spot_vol_corr': spot_vol_corr,
                    'sigma_pred': sigma_pred,
                    'sigma_impl': sigma_impl,
                    'expected_pnl': expected_pnl_components,  # Actual aggregate P&L from top 20 trades
                    'aggregate_gamma_pnl': total_gamma_pnl,
                    'aggregate_vanna_pnl': total_vanna_pnl,
                    'aggregate_volga_pnl': total_volga_pnl
                },
                'theta_decomposition': {
                    'gamma_cost': total_theta_gamma,
                    'vanna_cost': total_theta_vanna,
                    'volga_cost': total_theta_volga,
                    'market_theta': total_market_theta
                },
                'trades': {
                    'long_vol': long_trades,
                    'short_vol': short_trades,
                    'all_ranked': scored_trades[:50]  # Top 50
                },
                'timestamp': datetime.now().isoformat()
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/scenario-analysis")
async def scenario_analysis(portfolio: ManualPortfolioInput):
    """
    Portfolio Scenario Analysis - Shock grid across spot and time

    Creates a 2D grid showing P&L changes:
    - X-axis: Spot price changes (e.g., -5% to +5%)
    - Y-axis: Days forward (e.g., 0 to 7 days)

    Shows actual theta being consumed as time passes.
    """
    try:
        async with DeribitClient() as client:
            # Use provided btc_price or fetch from Deribit
            btc_price = portfolio.btc_price if portfolio.btc_price else await client.get_index_price("BTC")

            # Define grid parameters
            spot_shocks = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]  # Percentage changes
            days_forward = [0, 1, 2, 3, 4, 5, 6, 7]  # Days

            # Parse positions
            positions_data = []
            for p in portfolio.positions:
                parsed = parse_instrument(p.instrument_name)
                if not parsed:
                    continue

                abs_size = abs(p.size)
                sign = 1 if p.direction == 'buy' else -1

                # Get IV from input or fetch from order book
                iv = p.mark_iv
                if not iv:
                    try:
                        order_book = await client.get_order_book(p.instrument_name)
                        iv = order_book.get('mark_iv', 50)
                    except:
                        iv = 50

                positions_data.append({
                    'instrument': p.instrument_name,
                    'strike': parsed['strike'],
                    'dte': parsed['dte'],
                    'option_type': parsed['option_type'],
                    'size': abs_size,
                    'sign': sign,
                    'iv': iv / 100,  # Convert to decimal
                    'direction': p.direction
                })

            if not positions_data:
                raise HTTPException(status_code=400, detail="No valid positions to analyze")

            # Calculate current portfolio value (baseline)
            def calculate_portfolio_value(spot: float, days_passed: int) -> dict:
                """Calculate portfolio value at a given spot and time"""
                total_value = 0
                total_delta = 0
                total_gamma = 0
                total_theta = 0

                for pos in positions_data:
                    # Adjust DTE for days passed
                    dte_adjusted = max(pos['dte'] - days_passed, 0.01)
                    T = dte_adjusted / 365

                    # Calculate option price using BSM
                    greeks = calculate_full_greeks(
                        spot, pos['strike'], T, 0.05, pos['iv'], pos['option_type']
                    )

                    # Position value (price * size * sign)
                    option_value = greeks.price * pos['size'] * pos['sign']
                    total_value += option_value

                    # Greeks contribution
                    total_delta += greeks.delta * pos['size'] * pos['sign']
                    total_gamma += greeks.gamma * pos['size'] * pos['sign']
                    total_theta += greeks.theta * pos['size'] * pos['sign']

                return {
                    'value': total_value,
                    'delta': total_delta,
                    'gamma': total_gamma,
                    'theta': total_theta
                }

            # Calculate baseline (current value)
            baseline = calculate_portfolio_value(btc_price, 0)
            baseline_value = baseline['value']

            # Build the scenario grid
            grid = []
            theta_consumed = []  # Track actual theta consumed per day

            for day in days_forward:
                row = []
                theta_row = []

                for spot_pct in spot_shocks:
                    shocked_spot = btc_price * (1 + spot_pct / 100)
                    result = calculate_portfolio_value(shocked_spot, day)
                    pnl = result['value'] - baseline_value

                    row.append({
                        'spot_pct': spot_pct,
                        'day': day,
                        'pnl': pnl,
                        'value': result['value'],
                        'delta': result['delta'],
                        'gamma': result['gamma'],
                        'theta': result['theta']
                    })

                    # Calculate theta consumed (difference from previous day at same spot)
                    if day > 0:
                        prev_day_result = calculate_portfolio_value(shocked_spot, day - 1)
                        actual_theta = result['value'] - prev_day_result['value']
                        theta_row.append(actual_theta)
                    else:
                        theta_row.append(0)

                grid.append(row)
                theta_consumed.append(theta_row)

            # Extract just P&L values for heatmap (already in USD since BS uses S in USD)
            pnl_matrix = [[cell['pnl'] for cell in row] for row in grid]

            # Calculate summary stats (in USD)
            max_loss = min(min(row) for row in pnl_matrix)
            max_gain = max(max(row) for row in pnl_matrix)

            # Theta at spot=0% for each day (pure time decay)
            spot_zero_idx = spot_shocks.index(0)
            theta_at_zero = [row[spot_zero_idx]['pnl'] for row in grid]

            return {
                'btc_price': btc_price,
                'baseline_value': baseline_value,  # USD (BS price uses S in USD)
                'spot_shocks': spot_shocks,
                'days_forward': days_forward,
                'pnl_matrix': pnl_matrix,  # USD
                'theta_consumed': theta_consumed,  # USD
                'grid': grid,
                'summary': {
                    'max_loss': max_loss,  # USD
                    'max_gain': max_gain,  # USD
                    'theta_at_zero': theta_at_zero,  # USD - P&L at spot=0% for each day
                    'daily_theta_estimate': baseline['theta']  # USD (BS theta uses S in USD)
                },
                'positions_analyzed': len(positions_data),
                'timestamp': datetime.now().isoformat()
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# HTML frontend (embedded)
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Cost Model Gamma Tracker</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body class="bg-gray-950 text-gray-100 p-6">
    <div class="max-w-6xl mx-auto">
        <h1 class="text-3xl font-bold text-center mb-2 bg-gradient-to-r from-green-400 via-blue-500 to-purple-500 bg-clip-text text-transparent">
            Cost Model Gamma & Variance Tracker
        </h1>
        <p class="text-center text-gray-400 mb-4">θ = Γ × Ω_G + Va × Ω_Va + Vo × Ω_Vo</p>

        <!-- Tab Navigation -->
        <div class="flex space-x-2 mb-6 border-b border-gray-800">
            <button onclick="switchTab('portfolio')" id="tab-portfolio" class="tab-btn px-6 py-3 font-medium rounded-t-lg border-b-2 border-blue-500 text-blue-400 bg-gray-900">
                Portfolio Analysis
            </button>
            <button onclick="switchTab('custom')" id="tab-custom" class="tab-btn px-6 py-3 font-medium rounded-t-lg border-b-2 border-transparent text-gray-400 hover:text-gray-200">
                Trade Scanner
            </button>
        </div>

        <!-- Portfolio Analysis Tab Content -->
        <div id="content-portfolio" class="tab-content">

        <div class="bg-gray-900 rounded-xl p-6 mb-6 border border-gray-800">
            <h2 class="text-xl font-semibold mb-4 text-blue-400">Connect to Deribit</h2>
            <div class="grid grid-cols-4 gap-4 mb-4">
                <input type="text" id="clientId" placeholder="Client ID" class="bg-gray-800 border border-gray-700 rounded px-3 py-2">
                <input type="password" id="clientSecret" placeholder="Client Secret" class="bg-gray-800 border border-gray-700 rounded px-3 py-2">
                <input type="number" id="predictedDvol" placeholder="Predicted DVOL" class="bg-gray-800 border border-gray-700 rounded px-3 py-2">
                <button onclick="fetchPortfolio()" class="bg-blue-600 hover:bg-blue-700 rounded font-medium">Fetch Portfolio</button>
            </div>
        </div>
        
        <div id="results" class="hidden space-y-6">
            <!-- Key Metrics -->
            <div class="grid grid-cols-6 gap-3">
                <div class="bg-gray-900 rounded-lg p-3 border border-gray-800">
                    <div class="text-xs text-gray-400">BTC Price</div>
                    <div id="btcPrice" class="text-xl font-bold">-</div>
                </div>
                <div class="bg-gray-900 rounded-lg p-3 border border-gray-800" style="box-shadow: 0 0 10px rgba(34,197,94,0.2)">
                    <div class="text-xs text-gray-400">Gamma (Δ per 1%)</div>
                    <div id="gamma1pct" class="text-xl font-bold text-green-400">-</div>
                    <div class="text-xs text-gray-500">BTC delta change</div>
                </div>
                <div class="bg-gray-900 rounded-lg p-3 border border-gray-800" style="box-shadow: 0 0 10px rgba(239,68,68,0.2)">
                    <div class="text-xs text-gray-400">Daily Theta (BS)</div>
                    <div id="dailyTheta" class="text-xl font-bold text-red-400">-</div>
                </div>
                <div class="bg-gray-900 rounded-lg p-3 border border-gray-800" style="box-shadow: 0 0 10px rgba(249,115,22,0.2)">
                    <div class="text-xs text-gray-400">Gamma P&L (IV)</div>
                    <div id="fairGammaPnl" class="text-xl font-bold text-orange-400">-</div>
                    <div class="text-xs text-gray-500">at implied vol</div>
                </div>
                <div class="bg-gray-900 rounded-lg p-3 border border-gray-800" style="box-shadow: 0 0 10px rgba(6,182,212,0.3)">
                    <div class="text-xs text-gray-400">Net Theta (IV)</div>
                    <div id="netThetaIV" class="text-xl font-bold text-cyan-400">-</div>
                    <div class="text-xs text-gray-500">theta + gamma</div>
                </div>
                <div class="bg-gray-900 rounded-lg p-3 border border-gray-800" style="box-shadow: 0 0 10px rgba(168,85,247,0.2)">
                    <div class="text-xs text-gray-400">Bumped Theta</div>
                    <div id="bumpedTheta" class="text-xl font-bold text-purple-400">-</div>
                    <div class="text-xs text-gray-500">T-1, S+1σ</div>
                </div>
                <div class="bg-gray-900 rounded-lg p-3 border border-gray-800" style="box-shadow: 0 0 10px rgba(236,72,153,0.2)">
                    <div class="text-xs text-gray-400">θ/Γ Ratio</div>
                    <div id="thetaGammaRatio" class="text-xl font-bold text-pink-400">-</div>
                    <div class="text-xs text-gray-500">bumped/BS</div>
                </div>
                <div class="bg-gray-900 rounded-lg p-3 border border-gray-800">
                    <div class="text-xs text-gray-400">Break-Even</div>
                    <div id="breakEven" class="text-xl font-bold text-yellow-400">-</div>
                </div>
                <div class="bg-gray-900 rounded-lg p-3 border border-gray-800">
                    <div class="text-xs text-gray-400">Vol Signal</div>
                    <div id="volSignal" class="text-xl font-bold">-</div>
                </div>
                <div class="bg-gray-900 rounded-lg p-3 border border-gray-800">
                    <div class="text-xs text-gray-400">RV vs IV</div>
                    <div id="volSpread" class="text-xl font-bold">-</div>
                </div>
            </div>
            
            <!-- Cost Model Breakdown -->
            <div class="bg-gray-900 rounded-xl p-4 border border-gray-800">
                <h3 class="text-lg font-semibold mb-3 text-green-400">Cost Model Decomposition</h3>
                <p class="text-xs text-gray-500 mb-3">θ = Γ×Ω_G + Va×Ω_Va + Vo×Ω_Vo | Sum should ≈ Daily Theta</p>
                <div class="grid grid-cols-4 gap-4">
                    <div>
                        <div class="flex justify-between text-sm mb-1">
                            <span class="text-green-400">Gamma Cost</span>
                            <span id="gammaCost">-</span>
                        </div>
                        <div class="h-3 bg-gray-800 rounded-full overflow-hidden">
                            <div id="gammaCostBar" class="h-full bg-green-500" style="width:0%"></div>
                        </div>
                    </div>
                    <div>
                        <div class="flex justify-between text-sm mb-1">
                            <span class="text-blue-400">Vanna Cost</span>
                            <span id="vannaCost">-</span>
                        </div>
                        <div class="h-3 bg-gray-800 rounded-full overflow-hidden">
                            <div id="vannaCostBar" class="h-full bg-blue-500" style="width:0%"></div>
                        </div>
                    </div>
                    <div>
                        <div class="flex justify-between text-sm mb-1">
                            <span class="text-purple-400">Volga Cost</span>
                            <span id="volgaCost">-</span>
                        </div>
                        <div class="h-3 bg-gray-800 rounded-full overflow-hidden">
                            <div id="volgaCostBar" class="h-full bg-purple-500" style="width:0%"></div>
                        </div>
                    </div>
                    <div>
                        <div class="flex justify-between text-sm mb-1">
                            <span class="text-yellow-400">= Sum</span>
                            <span id="costModelSum">-</span>
                        </div>
                        <div class="text-xs text-gray-500 mt-1">
                            vs Theta: <span id="costModelResidual">-</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Gamma Profile by Strike -->
            <div class="bg-gray-900 rounded-xl p-4 border border-gray-800">
                <h3 class="text-lg font-semibold mb-3 text-green-400">Gamma Profile by Strike</h3>
                <p class="text-xs text-gray-500 mb-2">Gamma (Δ per 1% move) in USD by strike. Green = long gamma, Red = short gamma.</p>
                <div id="gammaProfileChart" style="height:280px"></div>
            </div>

            <!-- Scenario Analysis / P&L Grid -->
            <div class="bg-gray-900 rounded-xl p-4 border border-gray-800">
                <div class="flex justify-between items-center mb-3">
                    <div>
                        <h3 class="text-lg font-semibold text-orange-400">Scenario Analysis (Spot × Time)</h3>
                        <p class="text-xs text-gray-500">P&L heatmap across spot shocks and time decay. Shows actual theta being consumed.</p>
                    </div>
                    <button onclick="runScenarioAnalysis()" id="scenarioBtn" class="bg-orange-600 hover:bg-orange-700 px-4 py-2 rounded font-medium text-sm">
                        Run Scenario
                    </button>
                </div>
                <div id="scenarioResults" class="hidden">
                    <div class="grid grid-cols-4 gap-3 mb-4">
                        <div class="bg-gray-800 rounded-lg p-3">
                            <div class="text-xs text-gray-400">Max Gain</div>
                            <div id="scenarioMaxGain" class="text-lg font-bold text-green-400">-</div>
                        </div>
                        <div class="bg-gray-800 rounded-lg p-3">
                            <div class="text-xs text-gray-400">Max Loss</div>
                            <div id="scenarioMaxLoss" class="text-lg font-bold text-red-400">-</div>
                        </div>
                        <div class="bg-gray-800 rounded-lg p-3">
                            <div class="text-xs text-gray-400">Daily θ (Est.)</div>
                            <div id="scenarioDailyTheta" class="text-lg font-bold text-yellow-400">-</div>
                        </div>
                        <div class="bg-gray-800 rounded-lg p-3">
                            <div class="text-xs text-gray-400">7-Day θ at Spot</div>
                            <div id="scenario7DayTheta" class="text-lg font-bold text-cyan-400">-</div>
                        </div>
                    </div>
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <h4 class="text-sm font-semibold mb-2 text-orange-300">P&L Heatmap (Spot × Days)</h4>
                            <div id="scenarioHeatmap" style="height:300px"></div>
                        </div>
                        <div>
                            <h4 class="text-sm font-semibold mb-2 text-cyan-300">Actual Theta Consumed per Day</h4>
                            <div id="thetaConsumedChart" style="height:300px"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Charts -->
            <div class="grid grid-cols-2 gap-4">
                <div class="bg-gray-900 rounded-xl p-4 border border-gray-800">
                    <h3 class="text-lg font-semibold mb-2 text-yellow-400">Variance Path</h3>
                    <div id="varianceChart" style="height:250px"></div>
                </div>
                <div class="bg-gray-900 rounded-xl p-4 border border-gray-800">
                    <h3 class="text-lg font-semibold mb-2 text-cyan-400">Daily Returns vs Break-Even</h3>
                    <div id="returnsChart" style="height:250px"></div>
                </div>
            </div>

            <!-- Vol Surface -->
            <div class="bg-gray-900 rounded-xl p-4 border border-gray-800">
                <h3 class="text-lg font-semibold mb-3 text-purple-400">Implied Volatility Surface</h3>
                <div id="volSurfaceChart" style="height:300px"></div>
            </div>

            <!-- Put-Call Parity Violations -->
            <div id="paritySection" class="bg-gray-900 rounded-xl p-4 border border-gray-800 hidden">
                <h3 class="text-lg font-semibold mb-3 text-orange-400">Put-Call Parity Violations</h3>
                <p class="text-xs text-gray-500 mb-2">C - P should equal S - K×e^(-rT). Deviations may indicate arbitrage.</p>
                <table class="w-full text-sm">
                    <thead>
                        <tr class="text-gray-400 border-b border-gray-700">
                            <th class="text-left py-2">Strike</th>
                            <th class="text-left py-2">Expiry</th>
                            <th class="text-right py-2">Call</th>
                            <th class="text-right py-2">Put</th>
                            <th class="text-right py-2">Error</th>
                            <th class="text-right py-2">Error %</th>
                            <th class="text-right py-2">Action</th>
                        </tr>
                    </thead>
                    <tbody id="parityTable"></tbody>
                </table>
            </div>

            <!-- Box Spread Arbitrage -->
            <div id="boxSpreadSection" class="bg-gray-900 rounded-xl p-4 border border-gray-800 hidden">
                <h3 class="text-lg font-semibold mb-3 text-emerald-400">Box Spread Arbitrage</h3>
                <p class="text-xs text-gray-500 mb-2">Box = Call Spread + Put Spread. Should equal (K2-K1)×e^(-rT). Deviations = risk-free profit.</p>
                <table class="w-full text-sm">
                    <thead>
                        <tr class="text-gray-400 border-b border-gray-700">
                            <th class="text-left py-2">Expiry</th>
                            <th class="text-right py-2">K1</th>
                            <th class="text-right py-2">K2</th>
                            <th class="text-right py-2">Theoretical</th>
                            <th class="text-right py-2">Market</th>
                            <th class="text-right py-2">Profit</th>
                            <th class="text-right py-2">Profit %</th>
                            <th class="text-right py-2">Action</th>
                        </tr>
                    </thead>
                    <tbody id="boxSpreadTable"></tbody>
                </table>
            </div>

            <!-- Butterfly Arbitrage -->
            <div id="butterflySection" class="bg-gray-900 rounded-xl p-4 border border-gray-800 hidden">
                <h3 class="text-lg font-semibold mb-3 text-pink-400">Butterfly Convexity Violations</h3>
                <p class="text-xs text-gray-500 mb-2">C(K1) + C(K3) ≥ 2×C(K2) must hold. Violations = risk-free butterfly profit.</p>
                <table class="w-full text-sm">
                    <thead>
                        <tr class="text-gray-400 border-b border-gray-700">
                            <th class="text-left py-2">Expiry</th>
                            <th class="text-left py-2">Type</th>
                            <th class="text-right py-2">K1</th>
                            <th class="text-right py-2">K2</th>
                            <th class="text-right py-2">K3</th>
                            <th class="text-right py-2">Violation $</th>
                            <th class="text-right py-2">Action</th>
                        </tr>
                    </thead>
                    <tbody id="butterflyTable"></tbody>
                </table>
            </div>

            <!-- Term Structure Analysis -->
            <div id="termStructureSection" class="bg-gray-900 rounded-xl p-4 border border-gray-800 hidden">
                <h3 class="text-lg font-semibold mb-3 text-cyan-400">Term Structure Analysis</h3>
                <p class="text-xs text-gray-500 mb-2">Compares ATM IV across expiries. Forward vol shows implied vol between tenors. Richness = deviation from avg IV.</p>
                <div id="termStructureChart" style="height:250px" class="mb-4"></div>
                <table class="w-full text-sm">
                    <thead>
                        <tr class="text-gray-400 border-b border-gray-700">
                            <th class="text-left py-2">Expiry</th>
                            <th class="text-right py-2">DTE</th>
                            <th class="text-right py-2">ATM IV</th>
                            <th class="text-right py-2">Fwd Vol</th>
                            <th class="text-right py-2">Richness</th>
                            <th class="text-right py-2">Signal</th>
                        </tr>
                    </thead>
                    <tbody id="termStructureTable"></tbody>
                </table>
            </div>

            <!-- Theta Comparison: BS vs Deribit -->
            <div id="thetaComparisonSection" class="bg-gray-900 rounded-xl p-4 border border-gray-800">
                <h3 class="text-lg font-semibold mb-3 text-amber-400">Theta Comparison: BS Model vs Deribit</h3>
                <p class="text-xs text-gray-500 mb-2">Compares our Black-Scholes theta calculation with Deribit's reported theta. Sorted by largest variance. Deribit tends to overstate theta.</p>

                <!-- Summary Stats -->
                <div class="grid grid-cols-4 gap-3 mb-4">
                    <div class="bg-gray-800 rounded-lg p-2 border border-gray-700">
                        <div class="text-xs text-gray-400">BS Theta (Ours)</div>
                        <div id="thetaBsTotal" class="text-lg font-bold text-amber-400">-</div>
                    </div>
                    <div class="bg-gray-800 rounded-lg p-2 border border-gray-700">
                        <div class="text-xs text-gray-400">Deribit Theta</div>
                        <div id="thetaDeribitTotal" class="text-lg font-bold text-gray-300">-</div>
                    </div>
                    <div class="bg-gray-800 rounded-lg p-2 border border-gray-700">
                        <div class="text-xs text-gray-400">Difference</div>
                        <div id="thetaDiffTotal" class="text-lg font-bold text-emerald-400">-</div>
                    </div>
                    <div class="bg-gray-800 rounded-lg p-2 border border-gray-700">
                        <div class="text-xs text-gray-400">Diff %</div>
                        <div id="thetaDiffPctTotal" class="text-lg font-bold text-cyan-400">-</div>
                    </div>
                </div>

                <table class="w-full text-sm">
                    <thead>
                        <tr class="text-gray-400 border-b border-gray-700">
                            <th class="text-left py-2">Instrument</th>
                            <th class="text-center py-2">Dir</th>
                            <th class="text-right py-2">Size</th>
                            <th class="text-right py-2">DTE</th>
                            <th class="text-right py-2">BS Theta</th>
                            <th class="text-right py-2">Deribit Theta</th>
                            <th class="text-right py-2">Diff $</th>
                            <th class="text-right py-2">Diff %</th>
                        </tr>
                    </thead>
                    <tbody id="thetaComparisonTable"></tbody>
                </table>
            </div>
        </div>

        <!-- MARKET-WIDE ARBITRAGE SCANNER -->
        <div class="mt-8 border-t-2 border-gray-700 pt-6">
            <div class="flex justify-between items-center mb-4">
                <h2 class="text-2xl font-bold bg-gradient-to-r from-yellow-400 via-orange-500 to-red-500 bg-clip-text text-transparent">
                    Market-Wide Arbitrage Scanner
                </h2>
                <button onclick="scanMarketArbitrage()" id="scanBtn" class="bg-orange-600 hover:bg-orange-700 px-4 py-2 rounded font-medium">
                    Scan Deribit Market
                </button>
            </div>
            <p class="text-gray-400 text-sm mb-4">Scans ALL BTC options on Deribit for butterfly convexity violations, box spread mispricings, and term structure opportunities.</p>

            <div id="marketScanResults" class="hidden space-y-6">
                <!-- Scan Stats -->
                <div class="grid grid-cols-4 gap-3">
                    <div class="bg-gray-900 rounded-lg p-3 border border-gray-800">
                        <div class="text-xs text-gray-400">Options Scanned</div>
                        <div id="optionsScanned" class="text-xl font-bold text-yellow-400">-</div>
                    </div>
                    <div class="bg-gray-900 rounded-lg p-3 border border-gray-800">
                        <div class="text-xs text-gray-400">Expiries Analyzed</div>
                        <div id="expiriesAnalyzed" class="text-xl font-bold text-orange-400">-</div>
                    </div>
                    <div class="bg-gray-900 rounded-lg p-3 border border-gray-800">
                        <div class="text-xs text-gray-400">Butterfly Violations</div>
                        <div id="butterflyCount" class="text-xl font-bold text-pink-400">-</div>
                    </div>
                    <div class="bg-gray-900 rounded-lg p-3 border border-gray-800">
                        <div class="text-xs text-gray-400">Box Arb Opps</div>
                        <div id="boxArbCount" class="text-xl font-bold text-emerald-400">-</div>
                    </div>
                </div>

                <!-- Market Term Structure -->
                <div class="bg-gray-900 rounded-xl p-4 border border-gray-800">
                    <h3 class="text-lg font-semibold mb-3 text-cyan-400">Market-Wide Term Structure</h3>
                    <div id="mktTermStructureChart" style="height:300px" class="mb-4"></div>
                    <table class="w-full text-sm">
                        <thead>
                            <tr class="text-gray-400 border-b border-gray-700">
                                <th class="text-left py-2">Expiry</th>
                                <th class="text-right py-2">DTE</th>
                                <th class="text-right py-2">ATM Strike</th>
                                <th class="text-right py-2">ATM IV</th>
                                <th class="text-right py-2">Fwd Vol</th>
                                <th class="text-right py-2">Richness</th>
                                <th class="text-right py-2">Signal</th>
                            </tr>
                        </thead>
                        <tbody id="mktTermStructureTable"></tbody>
                    </table>
                </div>

                <!-- Market Butterfly Violations -->
                <div id="mktButterflySection" class="bg-gray-900 rounded-xl p-4 border border-gray-800 hidden">
                    <h3 class="text-lg font-semibold mb-3 text-pink-400">Butterfly Convexity Violations (Top 20)</h3>
                    <p class="text-xs text-gray-500 mb-2">Buy wings + Sell 2x body for risk-free profit when C(K1) + C(K3) &lt; 2×C(K2)</p>
                    <table class="w-full text-sm">
                        <thead>
                            <tr class="text-gray-400 border-b border-gray-700">
                                <th class="text-left py-2">Expiry</th>
                                <th class="text-left py-2">Type</th>
                                <th class="text-right py-2">K1</th>
                                <th class="text-right py-2">K2</th>
                                <th class="text-right py-2">K3</th>
                                <th class="text-right py-2">Wings</th>
                                <th class="text-right py-2">Body</th>
                                <th class="text-right py-2">Profit $</th>
                                <th class="text-right py-2">Action</th>
                            </tr>
                        </thead>
                        <tbody id="mktButterflyTable"></tbody>
                    </table>
                </div>

                <!-- Market Box Spreads -->
                <div id="mktBoxSection" class="bg-gray-900 rounded-xl p-4 border border-gray-800 hidden">
                    <h3 class="text-lg font-semibold mb-3 text-emerald-400">Box Spread Arbitrage (Top 20)</h3>
                    <p class="text-xs text-gray-500 mb-2">Box should equal (K2-K1)×e^(-rT). Deviations = risk-free profit.</p>
                    <table class="w-full text-sm">
                        <thead>
                            <tr class="text-gray-400 border-b border-gray-700">
                                <th class="text-left py-2">Expiry</th>
                                <th class="text-right py-2">K1</th>
                                <th class="text-right py-2">K2</th>
                                <th class="text-right py-2">Theoretical</th>
                                <th class="text-right py-2">Market</th>
                                <th class="text-right py-2">Profit $</th>
                                <th class="text-right py-2">Profit %</th>
                                <th class="text-right py-2">Action</th>
                            </tr>
                        </thead>
                        <tbody id="mktBoxTable"></tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- COST MODEL STRATEGY SCANNER -->
        <div class="mt-8 border-t-2 border-gray-700 pt-6">
            <div class="flex justify-between items-center mb-4">
                <h2 class="text-2xl font-bold bg-gradient-to-r from-green-400 via-cyan-500 to-blue-500 bg-clip-text text-transparent">
                    Cost Model Strategy Scanner
                </h2>
                <button onclick="scanStrategies()" id="strategyBtn" class="bg-cyan-600 hover:bg-cyan-700 px-4 py-2 rounded font-medium">
                    Scan Strategies
                </button>
            </div>
            <p class="text-gray-400 text-sm mb-4">Analyzes all OTM options using Cost Model decomposition to find optimal structures for gamma, vanna, volga, and term structure trades.</p>

            <div id="strategyResults" class="hidden space-y-6">
                <!-- Strategy Stats -->
                <div class="grid grid-cols-3 gap-3">
                    <div class="bg-gray-900 rounded-lg p-3 border border-gray-800">
                        <div class="text-xs text-gray-400">BTC Price</div>
                        <div id="stratBtcPrice" class="text-xl font-bold text-white">-</div>
                    </div>
                    <div class="bg-gray-900 rounded-lg p-3 border border-gray-800">
                        <div class="text-xs text-gray-400">DVOL</div>
                        <div id="stratDvol" class="text-xl font-bold text-purple-400">-</div>
                    </div>
                    <div class="bg-gray-900 rounded-lg p-3 border border-gray-800">
                        <div class="text-xs text-gray-400">Options Analyzed</div>
                        <div id="stratOptionsCount" class="text-xl font-bold text-cyan-400">-</div>
                    </div>
                </div>

                <!-- Cheap Gamma -->
                <div class="bg-gray-900 rounded-xl p-4 border border-gray-800">
                    <h3 class="text-lg font-semibold mb-2 text-green-400">Cheap Gamma</h3>
                    <p class="text-xs text-gray-500 mb-3">Efficient long gamma - lowest break-even moves. Short-dated near-ATM options.</p>
                    <table class="w-full text-sm">
                        <thead>
                            <tr class="text-gray-400 border-b border-gray-700">
                                <th class="text-left py-2">Instrument</th>
                                <th class="text-right py-2">IV</th>
                                <th class="text-right py-2">Price</th>
                                <th class="text-right py-2">Daily Theta</th>
                                <th class="text-right py-2" title="Expected P&L with variance / BS Theta. Negative = gamma profit exceeds theta cost">θ/Var</th>
                                <th class="text-right py-2">Delta</th>
                            </tr>
                        </thead>
                        <tbody id="cheapGammaTable"></tbody>
                    </table>
                </div>

                <!-- Rich Theta -->
                <div class="bg-gray-900 rounded-xl p-4 border border-gray-800">
                    <h3 class="text-lg font-semibold mb-2 text-red-400">Rich Theta (Short Vol)</h3>
                    <p class="text-xs text-gray-500 mb-3">High theta relative to gamma risk. OTM options for premium selling.</p>
                    <table class="w-full text-sm">
                        <thead>
                            <tr class="text-gray-400 border-b border-gray-700">
                                <th class="text-left py-2">Instrument</th>
                                <th class="text-right py-2">IV</th>
                                <th class="text-right py-2">Price</th>
                                <th class="text-right py-2">Daily Theta</th>
                                <th class="text-right py-2" title="Expected P&L with variance / BS Theta. Near 0 = theta dominates gamma">θ/Var</th>
                                <th class="text-right py-2">Moneyness</th>
                            </tr>
                        </thead>
                        <tbody id="richThetaTable"></tbody>
                    </table>
                </div>

                <!-- Vanna Trades -->
                <div class="bg-gray-900 rounded-xl p-4 border border-gray-800">
                    <h3 class="text-lg font-semibold mb-2 text-blue-400">Vanna Trades (Risk Reversals)</h3>
                    <p class="text-xs text-gray-500 mb-3">Skew exposure via risk reversals. Long OTM call + short OTM put = positive net vanna. Profits when negative spot-vol correlation holds (spot↑ vol↓). Short put gains from vol drop; long call gains from delta.</p>
                    <table class="w-full text-sm">
                        <thead>
                            <tr class="text-gray-400 border-b border-gray-700">
                                <th class="text-left py-2">Expiry</th>
                                <th class="text-left py-2">Structure</th>
                                <th class="text-right py-2">Call IV</th>
                                <th class="text-right py-2">Put IV</th>
                                <th class="text-right py-2">Skew</th>
                                <th class="text-right py-2">Net Theta</th>
                                <th class="text-left py-2">Signal</th>
                            </tr>
                        </thead>
                        <tbody id="vannaTradesTable"></tbody>
                    </table>
                </div>

                <!-- Volga Trades -->
                <div class="bg-gray-900 rounded-xl p-4 border border-gray-800">
                    <h3 class="text-lg font-semibold mb-2 text-purple-400">Volga Trades (Vol-of-Vol)</h3>
                    <p class="text-xs text-gray-500 mb-3">Wide strangles for smile/vol-of-vol exposure. Profits if vol moves significantly either way.</p>
                    <table class="w-full text-sm">
                        <thead>
                            <tr class="text-gray-400 border-b border-gray-700">
                                <th class="text-left py-2">Expiry</th>
                                <th class="text-left py-2">Structure</th>
                                <th class="text-right py-2">Avg IV</th>
                                <th class="text-right py-2">Cost</th>
                                <th class="text-right py-2">Daily Theta</th>
                                <th class="text-right py-2">DTE</th>
                            </tr>
                        </thead>
                        <tbody id="volgaTradesTable"></tbody>
                    </table>
                </div>

                <!-- Calendar Spreads -->
                <div class="bg-gray-900 rounded-xl p-4 border border-gray-800">
                    <h3 class="text-lg font-semibold mb-2 text-yellow-400">Calendar Spreads</h3>
                    <p class="text-xs text-gray-500 mb-3">Term structure plays. Sell front month, buy back month when IV diff is favorable.</p>
                    <table class="w-full text-sm">
                        <thead>
                            <tr class="text-gray-400 border-b border-gray-700">
                                <th class="text-left py-2">Structure</th>
                                <th class="text-right py-2">Front IV</th>
                                <th class="text-right py-2">Back IV</th>
                                <th class="text-right py-2">IV Diff</th>
                                <th class="text-right py-2">Net Theta</th>
                                <th class="text-left py-2">Signal</th>
                            </tr>
                        </thead>
                        <tbody id="calendarTable"></tbody>
                    </table>
                </div>
            </div>
        </div>
        </div> <!-- End Portfolio Analysis Tab Content -->

        <!-- Custom Tool Tab Content: Kelly-Edge Trade Scanner -->
        <div id="content-custom" class="tab-content hidden">
            <!-- Header and Configuration -->
            <div class="bg-gray-900 rounded-xl p-6 mb-6 border border-gray-800">
                <h2 class="text-xl font-semibold mb-2 text-purple-400">BTC Options Trade Scanner</h2>
                <p class="text-gray-400 mb-4 text-sm">Ranks trades by Kelly x Edge score using predicted volatility from your ML model</p>

                <!-- Configuration Inputs -->
                <div class="grid grid-cols-6 gap-4 mb-4">
                    <div>
                        <label class="block text-xs text-gray-400 mb-1">Predicted DVOL (%)</label>
                        <input type="number" id="kellyPredictedVol" value="50" step="0.5" class="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">
                    </div>
                    <div>
                        <label class="block text-xs text-gray-400 mb-1">Model RMSE (vol pts)</label>
                        <input type="number" id="kellyRmse" value="3" step="0.5" class="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">
                    </div>
                    <div>
                        <label class="block text-xs text-gray-400 mb-1">Kelly Fraction</label>
                        <select id="kellyFraction" class="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">
                            <option value="0.10">10% (Conservative)</option>
                            <option value="0.25" selected>25% (Quarter Kelly)</option>
                            <option value="0.50">50% (Half Kelly)</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs text-gray-400 mb-1">NAV ($)</label>
                        <input type="number" id="kellyNav" value="10000000" step="100000" class="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">
                    </div>
                    <div>
                        <label class="block text-xs text-gray-400 mb-1">Direction</label>
                        <select id="kellyDirection" class="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">
                            <option value="all" selected>All Trades</option>
                            <option value="long">Long Vol Only</option>
                            <option value="short">Short Vol Only</option>
                        </select>
                    </div>
                    <div class="flex items-end">
                        <button onclick="runKellyScanner()" id="kellyScanBtn" class="w-full bg-purple-600 hover:bg-purple-700 rounded font-medium py-2 text-sm">
                            Scan Market
                        </button>
                    </div>
                </div>

                <!-- Advanced Filters (collapsible) -->
                <details class="mt-2">
                    <summary class="text-sm text-gray-400 cursor-pointer hover:text-gray-200">Advanced Filters</summary>
                    <div class="grid grid-cols-5 gap-4 mt-3 pt-3 border-t border-gray-700">
                        <div>
                            <label class="block text-xs text-gray-400 mb-1">Min Delta</label>
                            <input type="number" id="kellyMinDelta" value="0.10" step="0.05" class="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">
                        </div>
                        <div>
                            <label class="block text-xs text-gray-400 mb-1">Max Delta</label>
                            <input type="number" id="kellyMaxDelta" value="0.50" step="0.05" class="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">
                        </div>
                        <div>
                            <label class="block text-xs text-gray-400 mb-1">Min DTE</label>
                            <input type="number" id="kellyMinDte" value="5" class="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">
                        </div>
                        <div>
                            <label class="block text-xs text-gray-400 mb-1">Max DTE</label>
                            <input type="number" id="kellyMaxDte" value="90" class="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">
                        </div>
                        <div>
                            <label class="block text-xs text-gray-400 mb-1">Max Bid-Ask (vol)</label>
                            <input type="number" id="kellyMaxSpread" value="4" step="0.5" class="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">
                        </div>
                    </div>
                </details>
            </div>

            <!-- Results Section (hidden until scan runs) -->
            <div id="kellyResults" class="hidden space-y-6">
                <!-- Summary Metrics -->
                <div class="grid grid-cols-6 gap-3">
                    <div class="bg-gray-900 rounded-lg p-3 border border-gray-800">
                        <div class="text-xs text-gray-400">BTC Price</div>
                        <div id="kellyBtcPrice" class="text-xl font-bold">-</div>
                    </div>
                    <div class="bg-gray-900 rounded-lg p-3 border border-gray-800">
                        <div class="text-xs text-gray-400">Current DVOL</div>
                        <div id="kellyCurrentDvol" class="text-xl font-bold text-blue-400">-</div>
                    </div>
                    <div class="bg-gray-900 rounded-lg p-3 border border-gray-800">
                        <div class="text-xs text-gray-400">Predicted DVOL</div>
                        <div id="kellyPredDvol" class="text-xl font-bold text-purple-400">-</div>
                    </div>
                    <div class="bg-gray-900 rounded-lg p-3 border border-gray-800">
                        <div class="text-xs text-gray-400">Vol Edge</div>
                        <div id="kellyVolEdge" class="text-xl font-bold">-</div>
                    </div>
                    <div class="bg-gray-900 rounded-lg p-3 border border-gray-800">
                        <div class="text-xs text-gray-400">Vol Regime</div>
                        <div id="kellyVolRegime" class="text-xl font-bold">-</div>
                    </div>
                    <div class="bg-gray-900 rounded-lg p-3 border border-gray-800">
                        <div class="text-xs text-gray-400">Trades Found</div>
                        <div id="kellyTradesFound" class="text-xl font-bold text-green-400">-</div>
                    </div>
                </div>

                <!-- Vol Signal Banner -->
                <div id="kellyVolSignal" class="rounded-lg p-4 text-center font-medium"></div>

                <!-- Dynamic Omega Analysis Section -->
                <div class="bg-gray-900 rounded-xl p-4 border border-gray-800">
                    <h3 class="text-lg font-semibold mb-3 text-emerald-400">Expected Daily P&L (Top 20 Trades)</h3>
                    <p class="text-xs text-gray-500 mb-4">Aggregate expected daily P&L if you traded all top 20 recommendations at suggested size</p>

                    <!-- Aggregate P&L from Top 20 Trades -->
                    <div class="grid grid-cols-4 gap-4 mb-4">
                        <div class="bg-gray-800 rounded-lg p-3">
                            <div class="text-xs text-gray-400 mb-1">Gamma P&L</div>
                            <div id="kellyOmegaG" class="text-xl font-bold text-green-400">-</div>
                            <div class="text-xs text-gray-500 mt-1">Σ(Γ × Ω_G × contracts)</div>
                        </div>
                        <div class="bg-gray-800 rounded-lg p-3">
                            <div class="text-xs text-gray-400 mb-1">Vanna P&L</div>
                            <div id="kellyOmegaVa" class="text-xl font-bold text-blue-400">-</div>
                            <div class="text-xs text-gray-500 mt-1">Σ(Va × Ω_Va × contracts)</div>
                        </div>
                        <div class="bg-gray-800 rounded-lg p-3">
                            <div class="text-xs text-gray-400 mb-1">Volga P&L</div>
                            <div id="kellyOmegaVo" class="text-xl font-bold text-purple-400">-</div>
                            <div class="text-xs text-gray-500 mt-1">Σ(Vo × Ω_Vo × contracts)</div>
                        </div>
                        <div class="bg-gray-800 rounded-lg p-3 border-2 border-yellow-600">
                            <div class="text-xs text-yellow-400 mb-1">Total Expected P&L</div>
                            <div id="kellyTotalPnl" class="text-xl font-bold text-yellow-400">-</div>
                            <div class="text-xs text-gray-500 mt-1">Daily expected if vol = predicted</div>
                        </div>
                    </div>

                    <!-- Charts -->
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <h4 class="text-sm font-semibold mb-2 text-yellow-400">Expected P&L Breakdown (Top 20)</h4>
                            <div id="kellyExpectedPnlChart" style="height:200px"></div>
                        </div>
                        <div>
                            <h4 class="text-sm font-semibold mb-2 text-cyan-400">Theta Cost Decomposition</h4>
                            <div id="kellyThetaDecompChart" style="height:200px"></div>
                        </div>
                    </div>
                </div>

                <!-- Risk Limits Display -->
                <div class="bg-gray-900 rounded-xl p-4 border border-gray-800">
                    <h3 class="text-sm font-semibold mb-3 text-yellow-400">Risk Limits ($10M NAV)</h3>
                    <div class="grid grid-cols-4 gap-4">
                        <div>
                            <div class="flex justify-between text-sm mb-1">
                                <span class="text-gray-400">Delta</span>
                                <span id="kellyDeltaLimit" class="text-gray-300">$2M max</span>
                            </div>
                            <div class="h-2 bg-gray-800 rounded-full overflow-hidden">
                                <div id="kellyDeltaBar" class="h-full bg-blue-500" style="width:0%"></div>
                            </div>
                        </div>
                        <div>
                            <div class="flex justify-between text-sm mb-1">
                                <span class="text-gray-400">Theta</span>
                                <span id="kellyThetaLimit" class="text-gray-300">$50k/day max</span>
                            </div>
                            <div class="h-2 bg-gray-800 rounded-full overflow-hidden">
                                <div id="kellyThetaBar" class="h-full bg-red-500" style="width:0%"></div>
                            </div>
                        </div>
                        <div>
                            <div class="flex justify-between text-sm mb-1">
                                <span class="text-gray-400">Gamma</span>
                                <span id="kellyGammaLimit" class="text-gray-300">$100k/1% max</span>
                            </div>
                            <div class="h-2 bg-gray-800 rounded-full overflow-hidden">
                                <div id="kellyGammaBar" class="h-full bg-green-500" style="width:0%"></div>
                            </div>
                        </div>
                        <div>
                            <div class="flex justify-between text-sm mb-1">
                                <span class="text-gray-400">Vega</span>
                                <span id="kellyVegaLimit" class="text-gray-300">$100k/vol max</span>
                            </div>
                            <div class="h-2 bg-gray-800 rounded-full overflow-hidden">
                                <div id="kellyVegaBar" class="h-full bg-purple-500" style="width:0%"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Trade Tables -->
                <div class="grid grid-cols-2 gap-4">
                    <!-- Long Vol Trades -->
                    <div class="bg-gray-900 rounded-xl p-4 border border-gray-800">
                        <h3 class="text-lg font-semibold mb-3 text-green-400">Long Vol Trades (BUY)</h3>
                        <p class="text-xs text-gray-500 mb-2">Predicted vol > market IV. Ranked by Kelly x Edge.</p>
                        <div class="overflow-x-auto">
                            <table class="w-full text-xs">
                                <thead>
                                    <tr class="text-gray-400 border-b border-gray-700">
                                        <th class="text-left py-2">Instrument</th>
                                        <th class="text-right py-2">DTE</th>
                                        <th class="text-right py-2">IV</th>
                                        <th class="text-right py-2">Edge</th>
                                        <th class="text-right py-2">Kelly</th>
                                        <th class="text-right py-2">Score</th>
                                        <th class="text-right py-2">Qty</th>
                                    </tr>
                                </thead>
                                <tbody id="kellyLongTable"></tbody>
                            </table>
                        </div>
                    </div>

                    <!-- Short Vol Trades -->
                    <div class="bg-gray-900 rounded-xl p-4 border border-gray-800">
                        <h3 class="text-lg font-semibold mb-3 text-red-400">Short Vol Trades (SELL)</h3>
                        <p class="text-xs text-gray-500 mb-2">Predicted vol < market IV. Ranked by Kelly x Edge.</p>
                        <div class="overflow-x-auto">
                            <table class="w-full text-xs">
                                <thead>
                                    <tr class="text-gray-400 border-b border-gray-700">
                                        <th class="text-left py-2">Instrument</th>
                                        <th class="text-right py-2">DTE</th>
                                        <th class="text-right py-2">IV</th>
                                        <th class="text-right py-2">Edge</th>
                                        <th class="text-right py-2">Kelly</th>
                                        <th class="text-right py-2">Score</th>
                                        <th class="text-right py-2">Qty</th>
                                    </tr>
                                </thead>
                                <tbody id="kellyShortTable"></tbody>
                            </table>
                        </div>
                    </div>
                </div>

                <!-- Full Ranked Table -->
                <div class="bg-gray-900 rounded-xl p-4 border border-gray-800">
                    <h3 class="text-lg font-semibold mb-3 text-cyan-400">All Trades Ranked by Kelly x Edge</h3>
                    <p class="text-xs text-gray-500 mb-2">Complete list of scored trades. Score = Kelly_fraction x Edge_dollars.</p>
                    <div class="overflow-x-auto max-h-96 overflow-y-auto">
                        <table class="w-full text-xs">
                            <thead class="sticky top-0 bg-gray-900">
                                <tr class="text-gray-400 border-b border-gray-700">
                                    <th class="text-left py-2 px-1">Rank</th>
                                    <th class="text-left py-2 px-1">Instrument</th>
                                    <th class="text-center py-2 px-1">Dir</th>
                                    <th class="text-right py-2 px-1">DTE</th>
                                    <th class="text-right py-2 px-1">Delta</th>
                                    <th class="text-right py-2 px-1">Mark IV</th>
                                    <th class="text-right py-2 px-1">Edge (vol)</th>
                                    <th class="text-right py-2 px-1">Edge ($)</th>
                                    <th class="text-right py-2 px-1">Win Prob</th>
                                    <th class="text-right py-2 px-1">Kelly %</th>
                                    <th class="text-right py-2 px-1">Score</th>
                                    <th class="text-right py-2 px-1">Qty</th>
                                    <th class="text-right py-2 px-1">Break-Even</th>
                                    <th class="text-right py-2 px-1" title="Bumped θ / BS θ ratio. ~1.0 = fair, <1 = cheap theta, >1 = rich theta">θ/Γ</th>
                                </tr>
                            </thead>
                            <tbody id="kellyAllTable"></tbody>
                        </table>
                    </div>
                </div>

                <!-- Scan Stats -->
                <div class="text-xs text-gray-500 text-center" id="kellyScanStats"></div>
            </div>
        </div>

    </div>

    <script>
        // Tab switching function
        function switchTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
            // Remove active state from all tabs
            document.querySelectorAll('.tab-btn').forEach(el => {
                el.classList.remove('border-blue-500', 'text-blue-400', 'bg-gray-900');
                el.classList.add('border-transparent', 'text-gray-400');
            });
            // Show selected tab content
            document.getElementById('content-' + tabName).classList.remove('hidden');
            // Activate selected tab
            const activeTab = document.getElementById('tab-' + tabName);
            activeTab.classList.remove('border-transparent', 'text-gray-400');
            activeTab.classList.add('border-blue-500', 'text-blue-400', 'bg-gray-900');
        }

        // Store current positions for scenario analysis
        let currentPositions = [];
        let currentBtcPrice = 0;

        async function runScenarioAnalysis() {
            if (currentPositions.length === 0) {
                alert('Please fetch portfolio first to run scenario analysis');
                return;
            }

            const btn = document.getElementById('scenarioBtn');
            btn.disabled = true;
            btn.textContent = 'Calculating...';

            try {
                const res = await fetch('/api/scenario-analysis', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        positions: currentPositions,
                        btc_price: currentBtcPrice,
                        predicted_dvol: 50
                    })
                });
                const data = await res.json();

                if (!res.ok) {
                    let errMsg = 'Scenario analysis failed';
                    if (typeof data.detail === 'string') {
                        errMsg = data.detail;
                    } else if (Array.isArray(data.detail)) {
                        errMsg = data.detail.map(d => d.msg || JSON.stringify(d)).join('; ');
                    }
                    throw new Error(errMsg);
                }

                document.getElementById('scenarioResults').classList.remove('hidden');

                // Summary stats
                const formatUsd = (v) => (v >= 0 ? '+$' : '-$') + Math.abs(v).toFixed(0);
                document.getElementById('scenarioMaxGain').textContent = formatUsd(data.summary.max_gain);
                document.getElementById('scenarioMaxLoss').textContent = formatUsd(data.summary.max_loss);
                document.getElementById('scenarioDailyTheta').textContent = formatUsd(data.summary.daily_theta_estimate);

                // 7-day theta at spot (P&L at day 7, spot 0%)
                const theta7Day = data.summary.theta_at_zero[7] || data.summary.theta_at_zero[data.summary.theta_at_zero.length - 1];
                document.getElementById('scenario7DayTheta').textContent = formatUsd(theta7Day);

                // P&L Heatmap
                const spotLabels = data.spot_shocks.map(s => s + '%');
                const dayLabels = data.days_forward.map(d => 'Day ' + d);

                // Transpose matrix for heatmap (Plotly wants z[y][x])
                const zData = data.pnl_matrix;

                Plotly.newPlot('scenarioHeatmap', [{
                    z: zData,
                    x: spotLabels,
                    y: dayLabels,
                    type: 'heatmap',
                    colorscale: [
                        [0, '#ef4444'],      // Red for losses
                        [0.5, '#1f2937'],    // Gray for breakeven
                        [1, '#22c55e']       // Green for gains
                    ],
                    zmid: 0,
                    text: zData.map(row => row.map(v => '$' + v.toFixed(0))),
                    texttemplate: '%{text}',
                    textfont: { size: 9, color: '#fff' },
                    hovertemplate: 'Spot: %{x}<br>Time: %{y}<br>P&L: $%{z:.0f}<extra></extra>',
                    colorbar: {
                        title: 'P&L ($)',
                        titlefont: { color: '#9ca3af' },
                        tickfont: { color: '#9ca3af' }
                    }
                }], {
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#9ca3af' },
                    margin: { t: 30, b: 50, l: 60, r: 30 },
                    xaxis: { title: 'Spot Change', tickfont: { size: 10 } },
                    yaxis: { title: 'Days Forward', tickfont: { size: 10 } }
                });

                // Theta Consumed Chart - show actual theta per day at different spot levels
                const thetaTraces = [];
                const spotColors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#06b6d4', '#8b5cf6'];
                const selectedSpots = [-3, -1, 0, 1, 3];  // Show subset of spots

                selectedSpots.forEach((spotPct, idx) => {
                    const spotIdx = data.spot_shocks.indexOf(spotPct);
                    if (spotIdx >= 0) {
                        const thetaValues = data.theta_consumed.map(row => row[spotIdx]);
                        thetaTraces.push({
                            x: dayLabels.slice(1),  // Skip day 0
                            y: thetaValues.slice(1),
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: spotPct + '% spot',
                            line: { color: spotColors[idx % spotColors.length] }
                        });
                    }
                });

                Plotly.newPlot('thetaConsumedChart', thetaTraces, {
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#9ca3af' },
                    margin: { t: 30, b: 50, l: 60, r: 30 },
                    xaxis: { title: 'Day', gridcolor: '#374151' },
                    yaxis: { title: 'θ Consumed ($)', gridcolor: '#374151', zerolinecolor: '#6b7280' },
                    legend: { orientation: 'h', y: 1.1 },
                    showlegend: true
                });

            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Run Scenario';
            }
        }

        // ============================================================
        // Kelly-Edge Trade Scanner
        // ============================================================
        async function runKellyScanner() {
            const btn = document.getElementById('kellyScanBtn');
            btn.disabled = true;
            btn.textContent = 'Scanning...';

            try {
                // Gather parameters
                const params = new URLSearchParams({
                    predicted_vol: document.getElementById('kellyPredictedVol').value,
                    prediction_rmse: document.getElementById('kellyRmse').value,
                    kelly_fraction: document.getElementById('kellyFraction').value,
                    nav: document.getElementById('kellyNav').value,
                    min_delta: document.getElementById('kellyMinDelta').value,
                    max_delta: document.getElementById('kellyMaxDelta').value,
                    min_dte: document.getElementById('kellyMinDte').value,
                    max_dte: document.getElementById('kellyMaxDte').value,
                    max_bid_ask: document.getElementById('kellyMaxSpread').value,
                    direction_filter: document.getElementById('kellyDirection').value
                });

                const res = await fetch('/api/kelly-edge-scanner?' + params.toString());
                const data = await res.json();

                if (!res.ok) {
                    throw new Error(data.detail || 'Scan failed');
                }

                // Show results section
                document.getElementById('kellyResults').classList.remove('hidden');

                // Summary Metrics
                document.getElementById('kellyBtcPrice').textContent = '$' + data.btc_price.toLocaleString(undefined, {maximumFractionDigits: 0});
                document.getElementById('kellyCurrentDvol').textContent = data.current_dvol.toFixed(1) + '%';
                document.getElementById('kellyPredDvol').textContent = data.predicted_vol.toFixed(1) + '%';

                const volEdge = data.vol_edge;
                const volEdgeEl = document.getElementById('kellyVolEdge');
                volEdgeEl.textContent = (volEdge >= 0 ? '+' : '') + volEdge.toFixed(1) + ' pts';
                volEdgeEl.className = 'text-xl font-bold ' + (volEdge > 0 ? 'text-green-400' : volEdge < 0 ? 'text-red-400' : 'text-gray-400');

                const volRegimeEl = document.getElementById('kellyVolRegime');
                volRegimeEl.textContent = data.vol_regime;
                if (data.vol_regime === 'CHEAP') {
                    volRegimeEl.className = 'text-xl font-bold text-green-400';
                } else if (data.vol_regime === 'RICH') {
                    volRegimeEl.className = 'text-xl font-bold text-red-400';
                } else {
                    volRegimeEl.className = 'text-xl font-bold text-gray-400';
                }

                document.getElementById('kellyTradesFound').textContent = data.summary.scored_trades;

                // Vol Signal Banner
                const signalEl = document.getElementById('kellyVolSignal');
                if (data.vol_regime === 'CHEAP') {
                    signalEl.className = 'rounded-lg p-4 text-center font-medium bg-green-900/50 text-green-300 border border-green-700';
                    signalEl.textContent = data.vol_signal;
                } else if (data.vol_regime === 'RICH') {
                    signalEl.className = 'rounded-lg p-4 text-center font-medium bg-red-900/50 text-red-300 border border-red-700';
                    signalEl.textContent = data.vol_signal;
                } else {
                    signalEl.className = 'rounded-lg p-4 text-center font-medium bg-gray-800 text-gray-300 border border-gray-700';
                    signalEl.textContent = data.vol_signal;
                }

                // Risk Limits
                const nav = parseFloat(document.getElementById('kellyNav').value);
                document.getElementById('kellyDeltaLimit').textContent = '$' + (nav * 0.2 / 1000000).toFixed(1) + 'M max';
                document.getElementById('kellyThetaLimit').textContent = '$50k/day max';
                document.getElementById('kellyGammaLimit').textContent = '$100k/1% max';
                document.getElementById('kellyVegaLimit').textContent = '$100k/vol max';

                // Long Vol Table
                const longTrades = data.trades.long_vol;
                document.getElementById('kellyLongTable').innerHTML = longTrades.slice(0, 10).map(t => `
                    <tr class="border-b border-gray-800 hover:bg-gray-800">
                        <td class="py-2 text-green-300 text-xs">${t.instrument}</td>
                        <td class="py-2 text-right">${t.dte}d</td>
                        <td class="py-2 text-right">${t.mark_iv.toFixed(1)}%</td>
                        <td class="py-2 text-right text-green-400">${t.edge_vol_points >= 0 ? '+' : ''}${t.edge_vol_points.toFixed(1)}</td>
                        <td class="py-2 text-right">${(t.kelly_fraction * 100).toFixed(1)}%</td>
                        <td class="py-2 text-right text-yellow-400">${t.rank_score.toFixed(2)}</td>
                        <td class="py-2 text-right font-medium">${t.suggested_contracts}</td>
                    </tr>
                `).join('') || '<tr><td colspan="7" class="py-4 text-center text-gray-500">No long vol trades found</td></tr>';

                // Short Vol Table
                const shortTrades = data.trades.short_vol;
                document.getElementById('kellyShortTable').innerHTML = shortTrades.slice(0, 10).map(t => `
                    <tr class="border-b border-gray-800 hover:bg-gray-800">
                        <td class="py-2 text-red-300 text-xs">${t.instrument}</td>
                        <td class="py-2 text-right">${t.dte}d</td>
                        <td class="py-2 text-right">${t.mark_iv.toFixed(1)}%</td>
                        <td class="py-2 text-right text-red-400">${t.edge_vol_points >= 0 ? '+' : ''}${t.edge_vol_points.toFixed(1)}</td>
                        <td class="py-2 text-right">${(t.kelly_fraction * 100).toFixed(1)}%</td>
                        <td class="py-2 text-right text-yellow-400">${t.rank_score.toFixed(2)}</td>
                        <td class="py-2 text-right font-medium">${t.suggested_contracts}</td>
                    </tr>
                `).join('') || '<tr><td colspan="7" class="py-4 text-center text-gray-500">No short vol trades found</td></tr>';

                // All Ranked Table
                const allTrades = data.trades.all_ranked;
                document.getElementById('kellyAllTable').innerHTML = allTrades.map((t, idx) => `
                    <tr class="border-b border-gray-800 hover:bg-gray-800">
                        <td class="py-2 px-1 text-gray-400">#${idx + 1}</td>
                        <td class="py-2 px-1 ${t.direction === 'BUY' ? 'text-green-300' : 'text-red-300'}">${t.instrument}</td>
                        <td class="py-2 px-1 text-center">
                            <span class="px-2 py-0.5 rounded text-xs font-medium ${t.direction === 'BUY' ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}">
                                ${t.direction}
                            </span>
                        </td>
                        <td class="py-2 px-1 text-right">${t.dte}d</td>
                        <td class="py-2 px-1 text-right">${t.delta.toFixed(3)}</td>
                        <td class="py-2 px-1 text-right">${t.mark_iv.toFixed(1)}%</td>
                        <td class="py-2 px-1 text-right ${t.edge_vol_points > 0 ? 'text-green-400' : 'text-red-400'}">${t.edge_vol_points >= 0 ? '+' : ''}${t.edge_vol_points.toFixed(1)}</td>
                        <td class="py-2 px-1 text-right">${t.edge_dollars >= 0 ? '+' : ''}$${t.edge_dollars.toFixed(2)}</td>
                        <td class="py-2 px-1 text-right">${(t.win_probability * 100).toFixed(0)}%</td>
                        <td class="py-2 px-1 text-right">${(t.kelly_fraction * 100).toFixed(1)}%</td>
                        <td class="py-2 px-1 text-right font-medium text-yellow-400">${t.rank_score.toFixed(2)}</td>
                        <td class="py-2 px-1 text-right font-bold">${t.suggested_contracts}</td>
                        <td class="py-2 px-1 text-right text-gray-400">${t.break_even.toFixed(2)}%</td>
                        <td class="py-2 px-1 text-right ${t.theta_gamma_ratio < 0.9 ? 'text-green-400' : t.theta_gamma_ratio > 1.1 ? 'text-red-400' : 'text-gray-400'}">${t.theta_gamma_ratio.toFixed(2)}x</td>
                    </tr>
                `).join('') || '<tr><td colspan="14" class="py-4 text-center text-gray-500">No trades found matching criteria</td></tr>';

                // Scan Stats
                document.getElementById('kellyScanStats').textContent =
                    `Scanned ${data.summary.total_scanned} options | ${data.summary.passed_filters} passed filters | ${data.summary.scored_trades} scored | Last updated: ${new Date(data.timestamp).toLocaleTimeString()}`;

                // Expected P&L Analysis - Using AGGREGATE values from top 20 trades
                const omega = data.omega_analysis;
                const formatPnl = (v) => (v >= 0 ? '+$' : '-$') + Math.abs(v).toFixed(2);

                // Use aggregate P&L values (actual dollar amounts from top 20 trades)
                const gammaPnl = omega.aggregate_gamma_pnl || 0;
                const vannaPnl = omega.aggregate_vanna_pnl || 0;
                const volgaPnl = omega.aggregate_volga_pnl || 0;
                const totalPnl = gammaPnl + vannaPnl + volgaPnl;

                // Populate P&L values
                const omegaGEl = document.getElementById('kellyOmegaG');
                omegaGEl.textContent = formatPnl(gammaPnl);
                omegaGEl.className = 'text-xl font-bold ' + (gammaPnl >= 0 ? 'text-green-400' : 'text-red-400');

                const omegaVaEl = document.getElementById('kellyOmegaVa');
                omegaVaEl.textContent = formatPnl(vannaPnl);
                omegaVaEl.className = 'text-xl font-bold ' + (vannaPnl >= 0 ? 'text-blue-400' : 'text-orange-400');

                const omegaVoEl = document.getElementById('kellyOmegaVo');
                omegaVoEl.textContent = formatPnl(volgaPnl);
                omegaVoEl.className = 'text-xl font-bold ' + (volgaPnl >= 0 ? 'text-purple-400' : 'text-orange-400');

                // Total Expected P&L
                const totalPnlEl = document.getElementById('kellyTotalPnl');
                totalPnlEl.textContent = formatPnl(totalPnl);
                totalPnlEl.className = 'text-xl font-bold ' + (totalPnl >= 0 ? 'text-yellow-400' : 'text-red-400');

                // Expected P&L Chart (Bar chart - aggregate from top 20 trades)
                Plotly.newPlot('kellyExpectedPnlChart', [{
                    x: ['Gamma P&L', 'Vanna P&L', 'Volga P&L', 'Total'],
                    y: [gammaPnl, vannaPnl, volgaPnl, totalPnl],
                    type: 'bar',
                    marker: {
                        color: [
                            gammaPnl >= 0 ? '#22c55e' : '#ef4444',
                            vannaPnl >= 0 ? '#3b82f6' : '#f97316',
                            volgaPnl >= 0 ? '#a855f7' : '#f97316',
                            totalPnl >= 0 ? '#eab308' : '#f43f5e'
                        ]
                    },
                    text: [
                        formatPnl(gammaPnl),
                        formatPnl(vannaPnl),
                        formatPnl(volgaPnl),
                        formatPnl(totalPnl)
                    ],
                    textposition: 'outside',
                    textfont: { size: 10 }
                }], {
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#9ca3af' },
                    margin: { t: 20, b: 60, l: 50, r: 20 },
                    xaxis: { tickangle: -30 },
                    yaxis: { title: 'Daily P&L ($)', zeroline: true, zerolinecolor: '#4b5563' }
                });

                // Theta Decomposition Chart (Pie chart)
                const thetaDecomp = data.theta_decomposition;
                const thetaTotal = Math.abs(thetaDecomp.gamma_cost) + Math.abs(thetaDecomp.vanna_cost) + Math.abs(thetaDecomp.volga_cost);
                if (thetaTotal > 0) {
                    Plotly.newPlot('kellyThetaDecompChart', [{
                        values: [Math.abs(thetaDecomp.gamma_cost), Math.abs(thetaDecomp.vanna_cost), Math.abs(thetaDecomp.volga_cost)],
                        labels: ['Gamma Cost', 'Vanna Cost', 'Volga Cost'],
                        type: 'pie',
                        marker: {
                            colors: ['#22c55e', '#3b82f6', '#a855f7']
                        },
                        textinfo: 'label+percent',
                        textposition: 'inside',
                        hole: 0.4,
                        textfont: { size: 10, color: '#fff' }
                    }], {
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        font: { color: '#9ca3af' },
                        margin: { t: 20, b: 20, l: 20, r: 20 },
                        showlegend: false,
                        annotations: [{
                            text: '$' + thetaDecomp.market_theta.toFixed(0),
                            showarrow: false,
                            font: { size: 14, color: '#eab308' }
                        }]
                    });
                }

            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Scan Market';
            }
        }

        async function scanStrategies() {
            const btn = document.getElementById('strategyBtn');
            btn.disabled = true;
            btn.textContent = 'Scanning...';

            try {
                const res = await fetch('/api/strategy-scanner');
                const data = await res.json();

                if (!res.ok) {
                    throw new Error(data.detail || 'Scan failed');
                }

                document.getElementById('strategyResults').classList.remove('hidden');

                // Stats
                document.getElementById('stratBtcPrice').textContent = '$' + data.btc_price.toLocaleString(undefined, {maximumFractionDigits: 0});
                document.getElementById('stratDvol').textContent = data.dvol.toFixed(1) + '%';
                document.getElementById('stratOptionsCount').textContent = data.options_analyzed;

                // Cheap Gamma Table
                const cheapGamma = data.strategies.cheap_gamma.trades;
                document.getElementById('cheapGammaTable').innerHTML = cheapGamma.slice(0, 8).map(t => `
                    <tr class="border-b border-gray-800">
                        <td class="py-2 text-green-300">${t.instrument}</td>
                        <td class="py-2 text-right">${t.mark_iv.toFixed(1)}%</td>
                        <td class="py-2 text-right">$${t.mark_price_usd.toFixed(0)}</td>
                        <td class="py-2 text-right text-red-400">$${t.daily_theta_usd.toFixed(2)}</td>
                        <td class="py-2 text-right ${t.theta_variance_ratio < 0 ? 'text-green-400' : t.theta_variance_ratio < 0.5 ? 'text-yellow-400' : 'text-gray-400'}">${t.theta_variance_ratio.toFixed(2)}</td>
                        <td class="py-2 text-right">${t.delta.toFixed(3)}</td>
                    </tr>
                `).join('');

                // Rich Theta Table
                const richTheta = data.strategies.rich_theta.trades;
                document.getElementById('richThetaTable').innerHTML = richTheta.slice(0, 8).map(t => `
                    <tr class="border-b border-gray-800">
                        <td class="py-2 text-red-300">${t.instrument}</td>
                        <td class="py-2 text-right">${t.mark_iv.toFixed(1)}%</td>
                        <td class="py-2 text-right">$${t.mark_price_usd.toFixed(0)}</td>
                        <td class="py-2 text-right text-green-400">$${t.daily_theta_usd.toFixed(2)}</td>
                        <td class="py-2 text-right ${t.theta_variance_ratio < 0.1 ? 'text-green-400' : t.theta_variance_ratio < 0.3 ? 'text-yellow-400' : 'text-gray-400'}">${t.theta_variance_ratio.toFixed(2)}</td>
                        <td class="py-2 text-right">${t.moneyness > 0 ? '+' : ''}${t.moneyness.toFixed(1)}%</td>
                    </tr>
                `).join('');

                // Vanna Trades Table
                const vannaTrades = data.strategies.vanna_trades.trades;
                document.getElementById('vannaTradesTable').innerHTML = vannaTrades.slice(0, 8).map(t => `
                    <tr class="border-b border-gray-800">
                        <td class="py-2">${t.expiry}</td>
                        <td class="py-2 text-blue-300 text-xs">${t.structure}</td>
                        <td class="py-2 text-right">${t.call_iv.toFixed(1)}%</td>
                        <td class="py-2 text-right">${t.put_iv.toFixed(1)}%</td>
                        <td class="py-2 text-right ${t.skew > 0 ? 'text-red-400' : 'text-green-400'}">${t.skew > 0 ? '+' : ''}${t.skew.toFixed(1)}</td>
                        <td class="py-2 text-right">${t.net_theta_usd >= 0 ? '+' : ''}$${t.net_theta_usd.toFixed(2)}</td>
                        <td class="py-2 text-xs ${t.rationale.includes('Long') ? 'text-yellow-400' : ''}">${t.rationale}</td>
                    </tr>
                `).join('');

                // Volga Trades Table
                const volgaTrades = data.strategies.volga_trades.trades;
                document.getElementById('volgaTradesTable').innerHTML = volgaTrades.slice(0, 8).map(t => `
                    <tr class="border-b border-gray-800">
                        <td class="py-2">${t.expiry}</td>
                        <td class="py-2 text-purple-300 text-xs">${t.structure}</td>
                        <td class="py-2 text-right">${t.avg_iv.toFixed(1)}%</td>
                        <td class="py-2 text-right">$${t.total_cost_usd.toLocaleString(undefined, {maximumFractionDigits: 0})}</td>
                        <td class="py-2 text-right text-red-400">$${t.total_theta_usd.toFixed(2)}</td>
                        <td class="py-2 text-right">${t.dte}d</td>
                    </tr>
                `).join('');

                // Calendar Spreads Table
                const calendars = data.strategies.calendar_spreads.trades;
                document.getElementById('calendarTable').innerHTML = calendars.slice(0, 8).map(t => `
                    <tr class="border-b border-gray-800">
                        <td class="py-2 text-yellow-300 text-xs">${t.structure}</td>
                        <td class="py-2 text-right">${t.front_iv.toFixed(1)}%</td>
                        <td class="py-2 text-right">${t.back_iv.toFixed(1)}%</td>
                        <td class="py-2 text-right ${t.iv_diff > 0 ? 'text-red-400' : 'text-green-400'}">${t.iv_diff > 0 ? '+' : ''}${t.iv_diff.toFixed(1)}%</td>
                        <td class="py-2 text-right">${t.net_theta_usd >= 0 ? '+' : ''}$${t.net_theta_usd.toFixed(2)}</td>
                        <td class="py-2 text-xs ${t.rationale.includes('rich') ? 'text-yellow-400' : ''}">${t.rationale}</td>
                    </tr>
                `).join('');

            } catch (e) {
                alert('Error: ' + e.message);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Scan Strategies';
            }
        }

        async function scanMarketArbitrage() {
            const btn = document.getElementById('scanBtn');
            btn.disabled = true;
            btn.textContent = 'Scanning...';

            try {
                const res = await fetch('/api/market-arb-scan');
                const data = await res.json();

                if (!res.ok) {
                    throw new Error(data.detail || 'Scan failed');
                }

                document.getElementById('marketScanResults').classList.remove('hidden');

                // Stats
                document.getElementById('optionsScanned').textContent = data.stats.total_options_scanned;
                document.getElementById('expiriesAnalyzed').textContent = data.stats.expiries_analyzed;
                document.getElementById('butterflyCount').textContent = data.stats.butterfly_violations_found;
                document.getElementById('boxArbCount').textContent = data.stats.box_arb_opportunities;

                // Term Structure Chart
                if (data.term_structure && data.term_structure.length > 0) {
                    const traces = [
                        {
                            x: data.term_structure.map(t => t.dte),
                            y: data.term_structure.map(t => t.atm_iv),
                            mode: 'lines+markers',
                            name: 'ATM IV (%)',
                            marker: {size: 10, color: '#22d3ee'}
                        },
                        {
                            x: data.term_structure.filter(t => t.fwd_vol).map(t => t.dte),
                            y: data.term_structure.filter(t => t.fwd_vol).map(t => t.fwd_vol),
                            mode: 'lines+markers',
                            name: 'Forward Vol (%)',
                            marker: {size: 10, symbol: 'diamond', color: '#a855f7'}
                        }
                    ];

                    // Add average line
                    const avgIv = data.term_structure.reduce((s, t) => s + t.atm_iv, 0) / data.term_structure.length;
                    traces.push({
                        x: [data.term_structure[0].dte, data.term_structure[data.term_structure.length-1].dte],
                        y: [avgIv, avgIv],
                        mode: 'lines',
                        name: 'Avg IV',
                        line: {color: '#eab308', dash: 'dash'}
                    });

                    Plotly.newPlot('mktTermStructureChart', traces, {
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        font: {color: '#9ca3af'},
                        margin: {t:10,b:40,l:50,r:50},
                        xaxis: {title: 'Days to Expiry', type: 'log'},
                        yaxis: {title: 'Volatility (%)'},
                        legend: {orientation: 'h', y: -0.15}
                    });

                    // Term Structure Table
                    const tbody = document.getElementById('mktTermStructureTable');
                    tbody.innerHTML = data.term_structure.map(t => `
                        <tr class="border-b border-gray-800">
                            <td class="py-2">${t.expiry}</td>
                            <td class="py-2 text-right">${t.dte}</td>
                            <td class="py-2 text-right">$${t.atm_strike.toLocaleString()}</td>
                            <td class="py-2 text-right">${t.atm_iv.toFixed(1)}%</td>
                            <td class="py-2 text-right">${t.fwd_vol ? t.fwd_vol.toFixed(1) + '%' : '-'}</td>
                            <td class="py-2 text-right ${t.richness > 0 ? 'text-red-400' : 'text-green-400'}">${t.richness > 0 ? '+' : ''}${t.richness.toFixed(1)}%</td>
                            <td class="py-2 text-xs ${t.signal.includes('BUY') ? 'text-green-400' : (t.signal.includes('SELL') ? 'text-red-400' : '')}">${t.signal}</td>
                        </tr>
                    `).join('');
                }

                // Butterfly Violations
                if (data.butterfly_arbs && data.butterfly_arbs.length > 0) {
                    document.getElementById('mktButterflySection').classList.remove('hidden');
                    const tbody = document.getElementById('mktButterflyTable');
                    tbody.innerHTML = data.butterfly_arbs.map(b => `
                        <tr class="border-b border-gray-800">
                            <td class="py-2">${b.expiry}</td>
                            <td class="py-2">${b.option_type}</td>
                            <td class="py-2 text-right">$${b.k1.toLocaleString()}</td>
                            <td class="py-2 text-right">$${b.k2.toLocaleString()}</td>
                            <td class="py-2 text-right">$${b.k3.toLocaleString()}</td>
                            <td class="py-2 text-right">$${b.wings_cost.toFixed(2)}</td>
                            <td class="py-2 text-right">$${b.body_cost.toFixed(2)}</td>
                            <td class="py-2 text-right text-green-400">$${b.violation.toFixed(2)}</td>
                            <td class="py-2 text-xs text-yellow-400">${b.action}</td>
                        </tr>
                    `).join('');
                } else {
                    document.getElementById('mktButterflySection').classList.add('hidden');
                }

                // Box Spread Arbitrage
                if (data.box_arbs && data.box_arbs.length > 0) {
                    document.getElementById('mktBoxSection').classList.remove('hidden');
                    const tbody = document.getElementById('mktBoxTable');
                    tbody.innerHTML = data.box_arbs.map(b => `
                        <tr class="border-b border-gray-800">
                            <td class="py-2">${b.expiry}</td>
                            <td class="py-2 text-right">$${b.k1.toLocaleString()}</td>
                            <td class="py-2 text-right">$${b.k2.toLocaleString()}</td>
                            <td class="py-2 text-right">$${b.theoretical.toFixed(2)}</td>
                            <td class="py-2 text-right">$${b.market_cost.toFixed(2)}</td>
                            <td class="py-2 text-right ${b.arb_profit > 0 ? 'text-green-400' : 'text-red-400'}">$${Math.abs(b.arb_profit).toFixed(2)}</td>
                            <td class="py-2 text-right ${b.arb_pct > 0 ? 'text-green-400' : 'text-red-400'}">${b.arb_pct.toFixed(2)}%</td>
                            <td class="py-2 text-xs text-yellow-400">${b.action}</td>
                        </tr>
                    `).join('');
                } else {
                    document.getElementById('mktBoxSection').classList.add('hidden');
                }

            } catch (e) {
                alert('Error: ' + e.message);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Scan Deribit Market';
            }
        }

        async function fetchPortfolio() {
            const clientId = document.getElementById('clientId').value;
            const clientSecret = document.getElementById('clientSecret').value;
            const predictedDvol = parseFloat(document.getElementById('predictedDvol').value) || null;
            const clientIdValue = clientId.trim() || null;
            const clientSecretValue = clientSecret.trim() || null;
            
            try {
                const payload = {
                    credentials: {
                        client_id: clientIdValue,
                        client_secret: clientSecretValue,
                        testnet: false,
                        predicted_dvol: predictedDvol
                    },
                    params: {}
                };
                
                const res = await fetch('/api/portfolio', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (res.ok) {
                    displayResults(data);
                } else {
                    console.error('Portfolio request failed', data);
                    const detail = data?.detail;
                    let message = 'Unknown error';
                    if (typeof detail === 'string') {
                        message = detail;
                    } else if (Array.isArray(detail)) {
                        message = detail.map(d => d.msg || JSON.stringify(d)).join('; ');
                    } else if (detail && typeof detail === 'object') {
                        message = detail.message || detail.error || JSON.stringify(detail);
                    }
                    alert('Error: ' + message);
                }
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }
        
        function displayResults(data) {
            document.getElementById('results').classList.remove('hidden');
            const p = data.portfolio;
            
            document.getElementById('btcPrice').textContent = '$' + p.btc_price.toLocaleString();
            document.getElementById('gamma1pct').textContent = p.gamma_1pct.toFixed(4);
            document.getElementById('dailyTheta').textContent = (p.daily_theta_usd >= 0 ? '+$' : '-$') + Math.abs(p.daily_theta_usd).toFixed(0);
            document.getElementById('dailyTheta').className = 'text-xl font-bold ' + (p.daily_theta_usd >= 0 ? 'text-green-400' : 'text-red-400');

            // Gamma P&L at implied vol (fair value) - negative for short gamma
            document.getElementById('fairGammaPnl').textContent = (p.fair_gamma_pnl >= 0 ? '+$' : '-$') + Math.abs(p.fair_gamma_pnl).toFixed(0);
            document.getElementById('fairGammaPnl').className = 'text-xl font-bold ' + (p.fair_gamma_pnl >= 0 ? 'text-green-400' : 'text-orange-400');

            // Net theta at IV = BS theta + fair gamma P&L (expected at implied vol)
            const netThetaIV = p.daily_theta_usd + p.fair_gamma_pnl;
            document.getElementById('netThetaIV').textContent = (netThetaIV >= 0 ? '+$' : '-$') + Math.abs(netThetaIV).toFixed(0);
            document.getElementById('netThetaIV').className = 'text-xl font-bold ' + (netThetaIV >= 0 ? 'text-cyan-400' : 'text-red-400');

            // Bumped Theta: repriced portfolio value after T-1 and S+1σ
            document.getElementById('bumpedTheta').textContent = (p.bumped_theta >= 0 ? '+$' : '-$') + Math.abs(p.bumped_theta).toFixed(0);
            document.getElementById('bumpedTheta').className = 'text-xl font-bold ' + (p.bumped_theta >= 0 ? 'text-green-400' : 'text-purple-400');

            // Theta/Gamma Ratio: bumped_theta / BS_theta (should be ~0 at IV)
            const ratioText = p.theta_gamma_ratio.toFixed(2) + 'x';
            document.getElementById('thetaGammaRatio').textContent = ratioText;
            // Color based on how far from 0 (ideal is 0 at implied vol)
            const absRatio = Math.abs(p.theta_gamma_ratio);
            document.getElementById('thetaGammaRatio').className = 'text-xl font-bold ' +
                (absRatio < 0.2 ? 'text-green-400' : absRatio < 0.5 ? 'text-yellow-400' : 'text-pink-400');

            document.getElementById('breakEven').textContent = p.break_even_move.toFixed(2) + '%';
            
            const volSignalEl = document.getElementById('volSignal');
            volSignalEl.textContent = p.vol_signal;
            volSignalEl.className = 'text-xl font-bold ' + 
                (p.vol_signal === 'LONG VOL' ? 'text-green-400' : 
                 p.vol_signal === 'SHORT VOL' ? 'text-red-400' : 'text-gray-400');
            
            const spreadEl = document.getElementById('volSpread');
            spreadEl.textContent = (p.vol_spread > 0 ? '+' : '') + p.vol_spread.toFixed(1) + '%';
            spreadEl.className = 'text-xl font-bold ' + (p.vol_spread > 0 ? 'text-green-400' : 'text-red-400');
            
            // Cost breakdown - show signed values that sum to theta
            // For short gamma: gamma_cost is negative (you receive theta from gamma exposure)
            const costModelSum = p.total_gamma_cost + p.total_vanna_cost + p.total_volga_cost;
            const total = Math.abs(p.total_gamma_cost) + Math.abs(p.total_vanna_cost) + Math.abs(p.total_volga_cost) + 0.01;

            // Format with sign
            const formatCost = (v) => (v >= 0 ? '+$' : '-$') + Math.abs(v).toFixed(2);
            document.getElementById('gammaCost').textContent = formatCost(p.total_gamma_cost);
            document.getElementById('vannaCost').textContent = formatCost(p.total_vanna_cost);
            document.getElementById('volgaCost').textContent = formatCost(p.total_volga_cost);
            document.getElementById('gammaCostBar').style.width = (Math.abs(p.total_gamma_cost) / total * 100) + '%';
            document.getElementById('vannaCostBar').style.width = (Math.abs(p.total_vanna_cost) / total * 100) + '%';
            document.getElementById('volgaCostBar').style.width = (Math.abs(p.total_volga_cost) / total * 100) + '%';

            // Display sum and residual
            document.getElementById('costModelSum').textContent = formatCost(costModelSum);
            const residual = p.daily_theta_usd - costModelSum;
            const residualPct = Math.abs(residual / p.daily_theta_usd * 100);
            document.getElementById('costModelResidual').textContent =
                (Math.abs(residual) < 100 ? 'OK' : formatCost(residual)) +
                ' (' + residualPct.toFixed(0) + '% diff)';
            document.getElementById('costModelResidual').className =
                residualPct < 20 ? 'text-green-400' : 'text-yellow-400';

            console.log('Cost Model Decomposition:', {
                gamma_cost: p.total_gamma_cost,
                vanna_cost: p.total_vanna_cost,
                volga_cost: p.total_volga_cost,
                sum: costModelSum,
                daily_theta: p.daily_theta_usd,
                residual: residual
            });

            // Gamma Profile Chart by Strike
            if (p.positions && p.positions.length > 0) {
                // Store positions and btc price for scenario analysis
                // Format to match ManualPositionInput schema: instrument_name, size, direction, mark_iv
                // Note: PositionAnalysis uses 'instrument', API expects 'instrument_name'
                currentBtcPrice = p.btc_price;
                currentPositions = p.positions.map(pos => ({
                    instrument_name: pos.instrument || pos.instrument_name,  // Handle both field names
                    size: pos.size,
                    direction: pos.direction,
                    mark_iv: pos.iv || 50,  // API expects mark_iv
                    mark_price: pos.mark_price || 0
                }));
                console.log('Stored positions for scenario analysis:', currentPositions.length, 'BTC:', currentBtcPrice);

                // Aggregate gamma by strike - gamma is already signed from position direction
                const gammaByStrike = {};
                p.positions.forEach(pos => {
                    const strike = pos.strike;
                    // Gamma 1% in USD = gamma * btc_price * 0.01 * btc_price
                    // pos.gamma is already signed (negative for short positions)
                    const gamma1pctUSD = pos.gamma * p.btc_price * 0.01 * p.btc_price;
                    if (!gammaByStrike[strike]) {
                        gammaByStrike[strike] = 0;
                    }
                    gammaByStrike[strike] += gamma1pctUSD;
                });

                // Sort by strike
                const strikes = Object.keys(gammaByStrike).map(Number).sort((a, b) => a - b);
                const strikeLabels = strikes.map(s => '$' + s.toLocaleString());
                const gammaValues = strikes.map(s => gammaByStrike[s]);

                // Color bars based on sign: green for long (positive), red for short (negative)
                const barColors = gammaValues.map(v => v >= 0 ? '#22c55e' : '#ef4444');
                const textColors = gammaValues.map(v => v >= 0 ? '#22c55e' : '#ef4444');

                Plotly.newPlot('gammaProfileChart', [{
                    x: strikeLabels,
                    y: gammaValues,
                    type: 'bar',
                    name: 'Gamma 1%',
                    marker: { color: barColors },
                    text: gammaValues.map(v => (v >= 0 ? '+$' : '-$') + Math.abs(v).toFixed(0)),
                    textposition: 'outside',
                    textfont: { color: textColors, size: 10 }
                }], {
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#9ca3af' },
                    margin: { t: 30, b: 60, l: 70, r: 30 },
                    xaxis: {
                        title: 'Strike (USD)',
                        tickangle: -45,
                        gridcolor: '#374151'
                    },
                    yaxis: {
                        title: 'Gamma 1% (USD)',
                        gridcolor: '#374151',
                        zerolinecolor: '#9ca3af',
                        zerolinewidth: 2
                    },
                    annotations: [{
                        xref: 'paper',
                        yref: 'paper',
                        x: 0.5,
                        y: 1.05,
                        xanchor: 'center',
                        text: 'Above zero = Long Γ | Below zero = Short Γ',
                        showarrow: false,
                        font: { color: '#6b7280', size: 10 }
                    }]
                });
            }

            // Variance chart
            const returns = data.daily_returns;
            let cumVar = 0;
            const cumVariance = returns.map(r => { cumVar += r*r; return cumVar; });
            const expectedDailyVar = Math.pow(p.avg_iv/100, 2) / 365;
            const expectedPath = returns.map((_, i) => expectedDailyVar * (i+1));
            
            Plotly.newPlot('varianceChart', [
                {y: cumVariance, name: 'Realized', line: {color: '#22c55e'}},
                {y: expectedPath, name: 'Expected', line: {color: '#eab308', dash: 'dash'}}
            ], {
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                font: {color: '#9ca3af'}, margin: {t:10,b:30,l:40,r:10}
            });
            
            // Returns chart
            const absReturns = returns.map(r => Math.abs(r) * 100);
            Plotly.newPlot('returnsChart', [
                {y: absReturns, type: 'bar', name: 'Daily |Return|', 
                 marker: {color: absReturns.map(r => r > p.break_even_move ? '#22c55e' : '#ef4444')}},
                {y: Array(returns.length).fill(p.break_even_move), name: 'Break-Even',
                 line: {color: '#eab308', dash: 'dash'}}
            ], {
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                font: {color: '#9ca3af'}, margin: {t:10,b:30,l:40,r:10}
            });
            
            // Vol Surface Chart
            if (p.vol_surface && p.vol_surface.length > 0) {
                // Group by expiry for different traces
                const byExpiry = {};
                p.vol_surface.forEach(v => {
                    if (!byExpiry[v.expiry]) byExpiry[v.expiry] = [];
                    byExpiry[v.expiry].push(v);
                });

                const traces = Object.entries(byExpiry).map(([expiry, points]) => {
                    points.sort((a, b) => a.strike - b.strike);
                    return {
                        x: points.map(pt => pt.moneyness),
                        y: points.map(pt => pt.iv),
                        mode: 'lines+markers',
                        name: expiry + ' (' + points[0].dte + 'd)',
                        marker: {size: 6}
                    };
                });

                Plotly.newPlot('volSurfaceChart', traces, {
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: {color: '#9ca3af'},
                    margin: {t:10,b:40,l:50,r:10},
                    xaxis: {title: 'Moneyness (%)', zeroline: true, zerolinecolor: '#4b5563'},
                    yaxis: {title: 'IV (%)'},
                    legend: {orientation: 'h', y: -0.2}
                });
            }

            // Parity Violations
            if (p.parity_violations && p.parity_violations.length > 0) {
                document.getElementById('paritySection').classList.remove('hidden');
                const tbody = document.getElementById('parityTable');
                tbody.innerHTML = p.parity_violations.map(v => `
                    <tr class="border-b border-gray-800">
                        <td class="py-2">$${v.strike.toLocaleString()}</td>
                        <td class="py-2">${v.expiry}</td>
                        <td class="py-2 text-right">$${v.call_price.toFixed(2)}</td>
                        <td class="py-2 text-right">$${v.put_price.toFixed(2)}</td>
                        <td class="py-2 text-right ${v.parity_error > 0 ? 'text-red-400' : 'text-green-400'}">$${v.parity_error.toFixed(2)}</td>
                        <td class="py-2 text-right ${v.parity_error_pct > 0 ? 'text-red-400' : 'text-green-400'}">${v.parity_error_pct.toFixed(2)}%</td>
                        <td class="py-2 text-right text-xs">${v.action}</td>
                    </tr>
                `).join('');
            } else {
                document.getElementById('paritySection').classList.add('hidden');
            }

            // Box Spread Arbitrage
            if (p.box_spread_arbs && p.box_spread_arbs.length > 0) {
                document.getElementById('boxSpreadSection').classList.remove('hidden');
                const tbody = document.getElementById('boxSpreadTable');
                tbody.innerHTML = p.box_spread_arbs.map(b => `
                    <tr class="border-b border-gray-800">
                        <td class="py-2">${b.expiry}</td>
                        <td class="py-2 text-right">$${b.k1.toLocaleString()}</td>
                        <td class="py-2 text-right">$${b.k2.toLocaleString()}</td>
                        <td class="py-2 text-right">$${b.theoretical_value.toFixed(2)}</td>
                        <td class="py-2 text-right">$${b.market_cost.toFixed(2)}</td>
                        <td class="py-2 text-right ${b.arb_profit > 0 ? 'text-green-400' : 'text-red-400'}">$${b.arb_profit.toFixed(2)}</td>
                        <td class="py-2 text-right ${b.arb_pct > 0 ? 'text-green-400' : 'text-red-400'}">${b.arb_pct.toFixed(2)}%</td>
                        <td class="py-2 text-right text-xs">${b.action}</td>
                    </tr>
                `).join('');
            } else {
                document.getElementById('boxSpreadSection').classList.add('hidden');
            }

            // Butterfly Arbitrage
            if (p.butterfly_arbs && p.butterfly_arbs.length > 0) {
                document.getElementById('butterflySection').classList.remove('hidden');
                const tbody = document.getElementById('butterflyTable');
                tbody.innerHTML = p.butterfly_arbs.map(b => `
                    <tr class="border-b border-gray-800">
                        <td class="py-2">${b.expiry}</td>
                        <td class="py-2">${b.option_type}</td>
                        <td class="py-2 text-right">$${b.k1.toLocaleString()}</td>
                        <td class="py-2 text-right">$${b.k2.toLocaleString()}</td>
                        <td class="py-2 text-right">$${b.k3.toLocaleString()}</td>
                        <td class="py-2 text-right text-green-400">$${b.convexity_violation.toFixed(2)}</td>
                        <td class="py-2 text-right text-xs">${b.action}</td>
                    </tr>
                `).join('');
            } else {
                document.getElementById('butterflySection').classList.add('hidden');
            }

            // Term Structure Analysis
            if (p.term_structure && p.term_structure.length > 0) {
                document.getElementById('termStructureSection').classList.remove('hidden');
                // Plot term structure chart
                const traces = [
                    {
                        x: p.term_structure.map(t => t.dte),
                        y: p.term_structure.map(t => t.atm_iv),
                        mode: 'lines+markers',
                        name: 'ATM IV (%)',
                        marker: {size: 8, color: '#22d3ee'}
                    },
                    {
                        x: p.term_structure.filter(t => t.fwd_vol).map(t => t.dte),
                        y: p.term_structure.filter(t => t.fwd_vol).map(t => t.fwd_vol),
                        mode: 'lines+markers',
                        name: 'Forward Vol (%)',
                        marker: {size: 8, symbol: 'diamond', color: '#a855f7'}
                    }
                ];
                Plotly.newPlot('termStructureChart', traces, {
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: {color: '#9ca3af'},
                    margin: {t:10,b:40,l:50,r:50},
                    xaxis: {title: 'Days to Expiry'},
                    yaxis: {title: 'Volatility (%)'},
                    legend: {orientation: 'h', y: -0.2}
                });

                // Populate table
                const tbody = document.getElementById('termStructureTable');
                tbody.innerHTML = p.term_structure.map(t => `
                    <tr class="border-b border-gray-800">
                        <td class="py-2">${t.expiry}</td>
                        <td class="py-2 text-right">${t.dte}</td>
                        <td class="py-2 text-right">${t.atm_iv.toFixed(1)}%</td>
                        <td class="py-2 text-right">${t.fwd_vol ? t.fwd_vol.toFixed(1) + '%' : '-'}</td>
                        <td class="py-2 text-right ${t.richness > 0 ? 'text-red-400' : 'text-green-400'}">${t.richness > 0 ? '+' : ''}${t.richness.toFixed(1)}%</td>
                        <td class="py-2 text-right text-xs ${t.signal.includes('BUY') ? 'text-green-400' : (t.signal.includes('SELL') ? 'text-red-400' : '')}">${t.signal}</td>
                    </tr>
                `).join('');
            } else {
                document.getElementById('termStructureSection').classList.add('hidden');
            }

            // Theta Comparison table
            if (p.theta_comparison && p.theta_comparison.positions && p.theta_comparison.positions.length > 0) {
                const tc = p.theta_comparison;
                const summary = tc.summary;

                // Update summary stats
                document.getElementById('thetaBsTotal').textContent = '$' + summary.total_theta_bs.toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0});
                document.getElementById('thetaDeribitTotal').textContent = '$' + summary.total_theta_deribit.toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0});
                document.getElementById('thetaDiffTotal').textContent = (summary.total_diff > 0 ? '+' : '') + '$' + summary.total_diff.toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0});
                document.getElementById('thetaDiffPctTotal').textContent = (summary.total_diff_pct > 0 ? '+' : '') + summary.total_diff_pct.toFixed(1) + '%';

                // Populate table
                const thetaTbody = document.getElementById('thetaComparisonTable');
                thetaTbody.innerHTML = tc.positions.map(pos => `
                    <tr class="border-b border-gray-800 hover:bg-gray-800">
                        <td class="py-2 text-xs font-mono">${pos.instrument}</td>
                        <td class="py-2 text-center ${pos.direction === 'buy' ? 'text-green-400' : 'text-red-400'}">${pos.direction === 'buy' ? 'LONG' : 'SHORT'}</td>
                        <td class="py-2 text-right">${pos.size.toFixed(1)}</td>
                        <td class="py-2 text-right">${pos.dte}</td>
                        <td class="py-2 text-right text-amber-400">$${pos.theta_bs.toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0})}</td>
                        <td class="py-2 text-right text-gray-300">$${pos.theta_deribit.toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0})}</td>
                        <td class="py-2 text-right ${pos.theta_diff > 0 ? 'text-emerald-400' : 'text-red-400'}">${pos.theta_diff > 0 ? '+' : ''}$${pos.theta_diff.toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0})}</td>
                        <td class="py-2 text-right ${pos.theta_diff_pct > 0 ? 'text-emerald-400' : 'text-red-400'}">${pos.theta_diff_pct > 0 ? '+' : ''}${pos.theta_diff_pct.toFixed(1)}%</td>
                    </tr>
                `).join('');
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_TEMPLATE


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
