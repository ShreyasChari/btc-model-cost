"""
Cost Model Volatility Surface
Based on Gilbert Eid's framework: θ = Γ×Ω_G + Va×Ω_Va + Vo×Ω_Vo
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from .greeks import calculate_greeks, bs_price


@dataclass
class CostParams:
    """
    Cost Model parameters for a single maturity
    
    Normalized costs:
    - omega_g: Cost per unit of S²×Gamma
    - omega_va: Cost per unit of S×Vanna  
    - omega_vo: Cost per unit of Volga
    """
    omega_g: float
    omega_va: float
    omega_vo: float
    maturity: float
    spot: float = 95000
    
    @property
    def gamma_be(self) -> float:
        """Daily Gamma break-even (% spot move)"""
        if self.omega_g >= 0:
            return 0.0
        return float(np.sqrt(-2 * self.omega_g / self.spot))
    
    @property
    def vanna_be(self) -> float:
        """Daily Vanna break-even (vol change per 1% spot)"""
        return float(-self.omega_va / self.spot)
    
    @property
    def volga_be(self) -> float:
        """Daily Volga break-even (% vol move)"""
        if self.omega_vo >= 0:
            return 0.0
        return float(np.sqrt(-2 * self.omega_vo / self.spot))
    
    @property
    def spot_vol_corr(self) -> float:
        """Implied spot-vol correlation"""
        gbe = self.gamma_be
        vobe = self.volga_be
        if gbe == 0 or vobe == 0:
            return 0.0
        return float(-self.omega_va / self.spot / (gbe * vobe))
    
    def to_dict(self) -> dict:
        return {
            'maturity': self.maturity,
            'omega_g': float(self.omega_g),
            'omega_va': float(self.omega_va),
            'omega_vo': float(self.omega_vo),
            'gamma_be': self.gamma_be,
            'vanna_be': self.vanna_be,
            'volga_be': self.volga_be,
            'spot_vol_corr': self.spot_vol_corr
        }


def solve_costs(spot: float, strikes: np.ndarray, vols: np.ndarray,
                maturity: float, rate: float = 0.0) -> CostParams:
    """
    Solve for Cost Model parameters from options
    
    Master equation: Γ×Ω_G + Va×Ω_Va + Vo×Ω_Vo = θ
    """
    n = len(strikes)
    
    greeks_matrix = np.zeros((n, 3))
    theta_vector = np.zeros(n)
    
    for i, (K, sigma) in enumerate(zip(strikes, vols)):
        opt_type = 'put' if K < spot else 'call'
        g = calculate_greeks(spot, K, maturity, rate, sigma, opt_type)
        
        # Normalize Greeks
        gamma_norm = g.gamma * spot * spot
        vanna_norm = g.vanna * spot
        volga_norm = g.volga
        theta_norm = g.theta * spot
        
        greeks_matrix[i] = [gamma_norm, vanna_norm, volga_norm]
        theta_vector[i] = theta_norm
    
    # Solve via least squares
    costs, _, _, _ = np.linalg.lstsq(greeks_matrix, theta_vector, rcond=None)
    
    return CostParams(
        omega_g=costs[0],
        omega_va=costs[1],
        omega_vo=costs[2],
        maturity=maturity,
        spot=spot
    )


def quadratic_smile(log_moneyness: np.ndarray, atm_vol: float, 
                   skew: float, curvature: float) -> np.ndarray:
    """Simple quadratic smile: σ(k) = σ_ATM + skew*k + curvature*k²"""
    return atm_vol + skew * log_moneyness + curvature * log_moneyness**2


class BitcoinVolSurface:
    """
    Bitcoin Implied Volatility Surface using Cost Model framework
    """
    
    def __init__(self, spot: float, rate: float = 0.0):
        self.spot = spot
        self.rate = rate
        self.maturities: List[float] = []
        self.smile_params: Dict[float, Dict] = {}
        self.cost_params: Dict[float, CostParams] = {}
        self._spline: Optional[RectBivariateSpline] = None
        
    def add_maturity(self, T: float, atm_vol: float, skew: float, curvature: float):
        """Add a maturity with smile parameters"""
        if T not in self.maturities:
            self.maturities.append(T)
            self.maturities.sort()
        
        self.smile_params[T] = {
            'atm_vol': atm_vol,
            'skew': skew,
            'curvature': curvature
        }
        
        self._calibrate_costs(T)
        self._build_spline()
    
    def calibrate_from_options(self, T: float, strikes: np.ndarray, 
                               vols: np.ndarray):
        """Calibrate from actual option data"""
        if T not in self.maturities:
            self.maturities.append(T)
            self.maturities.sort()
        
        # Fit quadratic smile
        log_m = np.log(strikes / self.spot)
        
        # Least squares fit: vol = a + b*k + c*k²
        A = np.column_stack([np.ones_like(log_m), log_m, log_m**2])
        coeffs, _, _, _ = np.linalg.lstsq(A, vols, rcond=None)
        
        self.smile_params[T] = {
            'atm_vol': float(coeffs[0]),
            'skew': float(coeffs[1]),
            'curvature': float(coeffs[2])
        }
        
        self.cost_params[T] = solve_costs(self.spot, strikes, vols, T, self.rate)
        self._build_spline()
    
    def _calibrate_costs(self, T: float):
        """Calibrate Cost Model parameters from smile"""
        params = self.smile_params[T]
        
        moneyness = np.array([0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15])
        strikes = self.spot * moneyness
        log_m = np.log(moneyness)
        
        vols = quadratic_smile(log_m, params['atm_vol'], 
                              params['skew'], params['curvature'])
        
        self.cost_params[T] = solve_costs(self.spot, strikes, vols, T, self.rate)
    
    def _build_spline(self):
        """Build 2D spline for interpolation"""
        if len(self.maturities) < 2:
            return
        
        moneyness = np.linspace(0.7, 1.3, 31)
        log_m = np.log(moneyness)
        
        vol_grid = np.zeros((len(self.maturities), len(moneyness)))
        
        for i, T in enumerate(self.maturities):
            p = self.smile_params[T]
            vol_grid[i] = quadratic_smile(log_m, p['atm_vol'], p['skew'], p['curvature'])
        
        self._spline = RectBivariateSpline(
            self.maturities, log_m, vol_grid, 
            kx=min(3, len(self.maturities)-1), ky=3
        )
    
    def get_vol(self, K: float, T: float) -> float:
        """Get implied vol for any strike/maturity"""
        log_m = np.log(K / self.spot)
        
        if T in self.smile_params:
            p = self.smile_params[T]
            return float(quadratic_smile(np.array([log_m]), p['atm_vol'], 
                                        p['skew'], p['curvature'])[0])
        
        if self._spline is not None:
            T_clamped = np.clip(T, self.maturities[0], self.maturities[-1])
            return float(self._spline(T_clamped, log_m)[0, 0])
        
        nearest_T = min(self.maturities, key=lambda t: abs(t - T))
        p = self.smile_params[nearest_T]
        return float(quadratic_smile(np.array([log_m]), p['atm_vol'], 
                                    p['skew'], p['curvature'])[0])
    
    def get_theta_decomposition(self, K: float, T: float, notional: float = 1.0) -> dict:
        """Decompose theta into Gamma/Vanna/Volga components"""
        sigma = self.get_vol(K, T)
        opt_type = 'put' if K < self.spot else 'call'
        greeks = calculate_greeks(self.spot, K, T, self.rate, sigma, opt_type)
        
        # Get costs (interpolate if needed)
        costs = self._get_costs_for_maturity(T)
        
        # Normalized Greeks
        gamma_norm = greeks.gamma * self.spot * self.spot
        vanna_norm = greeks.vanna * self.spot
        volga_norm = greeks.volga
        
        # Cost decomposition
        gamma_cost = gamma_norm * costs.omega_g / self.spot * notional
        vanna_cost = vanna_norm * costs.omega_va / self.spot * notional
        volga_cost = volga_norm * costs.omega_vo / self.spot * notional
        model_theta = gamma_cost + vanna_cost + volga_cost
        bs_theta = greeks.theta * notional
        
        total_abs = abs(gamma_cost) + abs(vanna_cost) + abs(volga_cost)
        
        return {
            'strike': K,
            'maturity': T,
            'moneyness': K / self.spot,
            'implied_vol': sigma,
            'option_type': opt_type,
            'price': bs_price(self.spot, K, T, self.rate, sigma, opt_type) * notional,
            'greeks': greeks.to_dict(),
            'bs_theta': float(bs_theta),
            'gamma_cost': float(gamma_cost),
            'vanna_cost': float(vanna_cost),
            'volga_cost': float(volga_cost),
            'model_theta': float(model_theta),
            'gamma_pct': float(abs(gamma_cost) / total_abs * 100) if total_abs > 0 else 0,
            'vanna_pct': float(abs(vanna_cost) / total_abs * 100) if total_abs > 0 else 0,
            'volga_pct': float(abs(volga_cost) / total_abs * 100) if total_abs > 0 else 0,
        }
    
    def _get_costs_for_maturity(self, T: float) -> CostParams:
        """Get or interpolate costs for a maturity"""
        if T in self.cost_params:
            return self.cost_params[T]
        
        mats = sorted(self.cost_params.keys())
        if not mats:
            return CostParams(0, 0, 0, T, self.spot)
        
        if T <= mats[0]:
            return self.cost_params[mats[0]]
        if T >= mats[-1]:
            return self.cost_params[mats[-1]]
        
        for i in range(len(mats)-1):
            if mats[i] <= T <= mats[i+1]:
                w = (T - mats[i]) / (mats[i+1] - mats[i])
                c1, c2 = self.cost_params[mats[i]], self.cost_params[mats[i+1]]
                return CostParams(
                    omega_g=c1.omega_g*(1-w) + c2.omega_g*w,
                    omega_va=c1.omega_va*(1-w) + c2.omega_va*w,
                    omega_vo=c1.omega_vo*(1-w) + c2.omega_vo*w,
                    maturity=T,
                    spot=self.spot
                )
        
        return self.cost_params[mats[-1]]
    
    def get_surface_data(self, moneyness_range: Tuple[float, float] = (0.7, 1.3),
                        num_strikes: int = 25) -> dict:
        """Get full surface data for visualization"""
        moneyness = np.linspace(moneyness_range[0], moneyness_range[1], num_strikes)
        strikes = self.spot * moneyness
        
        vols = []
        for T in self.maturities:
            row = []
            for K in strikes:
                row.append(self.get_vol(K, T))
            vols.append(row)
        
        return {
            'moneyness': moneyness.tolist(),
            'strikes': strikes.tolist(),
            'maturities': [float(m) for m in self.maturities],
            'vols': vols
        }
    
    def get_break_evens(self) -> list:
        """Get break-evens for all maturities"""
        result = []
        for T in sorted(self.cost_params.keys()):
            c = self.cost_params[T]
            p = self.smile_params.get(T, {'atm_vol': 0, 'skew': 0, 'curvature': 0})
            result.append({
                'maturity': float(T),
                'maturity_label': f"{int(T*12)}M" if T < 1 else f"{T:.1f}Y",
                'atm_vol': float(p['atm_vol']),
                'skew': float(p['skew']),
                'curvature': float(p['curvature']),
                'gamma_be': c.gamma_be,
                'vanna_be': c.vanna_be,
                'volga_be': c.volga_be,
                'spot_vol_corr': c.spot_vol_corr
            })
        return result
    
    def to_dict(self) -> dict:
        """Serialize surface to dictionary"""
        return {
            'spot': self.spot,
            'rate': self.rate,
            'maturities': [float(m) for m in self.maturities],
            'smile_params': {str(k): v for k, v in self.smile_params.items()},
            'break_evens': self.get_break_evens(),
            'surface_data': self.get_surface_data()
        }


class MarketComparison:
    """
    Compares Market IVs to Cost Model fitted IVs
    Identifies mispricings at each strike/maturity
    """
    
    def __init__(self, surface: BitcoinVolSurface):
        self.surface = surface
        self.market_data: List[Dict] = []  # Raw market observations
    
    def load_market_data(self, options: List[Dict]):
        """
        Load market data from Deribit format
        options: list of dicts with 'strike', 'maturity', 'mid_iv', 'bid_iv', 'ask_iv'
        """
        self.market_data = options
    
    def compare(self, threshold: float = 0.01) -> Dict:
        """
        Compare market IVs to model IVs
        
        Returns dict with:
        - comparisons: list of individual option comparisons
        - summary: aggregate statistics
        - signals: filtered list of actionable mispricings
        """
        if not self.market_data:
            return {'error': 'No market data loaded'}
        
        comparisons = []
        rich_count = 0
        cheap_count = 0
        total_spread = 0
        
        for opt in self.market_data:
            K = opt['strike']
            T = opt['maturity']
            market_iv = opt.get('mid_iv') or opt.get('mark_iv')
            
            if market_iv is None or T <= 0:
                continue
            
            # Get model IV
            model_iv = self.surface.get_vol(K, T)
            
            # Calculate spread
            spread = market_iv - model_iv  # Positive = market rich, Negative = market cheap
            spread_bps = spread * 10000  # Convert to basis points
            
            # Determine signal
            if spread > threshold:
                signal = 'RICH'
                rich_count += 1
            elif spread < -threshold:
                signal = 'CHEAP'
                cheap_count += 1
            else:
                signal = 'FAIR'
            
            # Bid/ask context
            bid_iv = opt.get('bid_iv')
            ask_iv = opt.get('ask_iv')
            
            # Check if model is outside bid/ask
            if bid_iv and ask_iv:
                if model_iv > ask_iv:
                    edge = 'MODEL_ABOVE_ASK'  # Market cheap, buy
                elif model_iv < bid_iv:
                    edge = 'MODEL_BELOW_BID'  # Market rich, sell
                else:
                    edge = 'WITHIN_SPREAD'
            else:
                edge = 'NO_SPREAD_DATA'
            
            comparisons.append({
                'instrument': opt.get('instrument', f"K={K}, T={T:.2f}"),
                'strike': K,
                'maturity': T,
                'maturity_label': f"{int(T*12)}M" if T < 1 else f"{T:.1f}Y",
                'moneyness': K / self.surface.spot,
                'option_type': opt.get('option_type', 'unknown'),
                'market_iv': market_iv,
                'model_iv': model_iv,
                'spread': spread,
                'spread_bps': spread_bps,
                'spread_pct': spread / model_iv * 100 if model_iv > 0 else 0,
                'signal': signal,
                'edge': edge,
                'bid_iv': bid_iv,
                'ask_iv': ask_iv,
                'volume': opt.get('volume', 0),
                'open_interest': opt.get('open_interest', 0),
            })
            
            total_spread += abs(spread)
        
        # Filter actionable signals
        signals = [c for c in comparisons if c['signal'] != 'FAIR']
        signals_sorted = sorted(signals, key=lambda x: abs(x['spread']), reverse=True)
        
        # Group by maturity for heatmap
        by_maturity = {}
        for c in comparisons:
            T = round(c['maturity'], 3)
            if T not in by_maturity:
                by_maturity[T] = []
            by_maturity[T].append(c)
        
        return {
            'comparisons': comparisons,
            'signals': signals_sorted[:20],  # Top 20 mispricings
            'summary': {
                'total_options': len(comparisons),
                'rich_count': rich_count,
                'cheap_count': cheap_count,
                'fair_count': len(comparisons) - rich_count - cheap_count,
                'avg_abs_spread': total_spread / len(comparisons) if comparisons else 0,
                'avg_abs_spread_bps': (total_spread / len(comparisons) * 10000) if comparisons else 0,
            },
            'by_maturity': {str(k): v for k, v in sorted(by_maturity.items())}
        }
    
    def get_mispricing_heatmap(self) -> Dict:
        """
        Generate heatmap data for visualization
        X-axis: Moneyness, Y-axis: Maturity, Color: Spread
        """
        comparison = self.compare()
        if 'error' in comparison:
            return comparison
        
        # Build grid
        maturities = sorted(set(c['maturity'] for c in comparison['comparisons']))
        moneyness_vals = sorted(set(round(c['moneyness'], 2) for c in comparison['comparisons']))
        
        # Create lookup
        lookup = {}
        for c in comparison['comparisons']:
            key = (round(c['maturity'], 3), round(c['moneyness'], 2))
            if key not in lookup:
                lookup[key] = []
            lookup[key].append(c['spread'])
        
        # Build heatmap grid
        heatmap = []
        for T in maturities:
            row = []
            for m in moneyness_vals:
                spreads = lookup.get((round(T, 3), m), [])
                avg_spread = sum(spreads) / len(spreads) if spreads else None
                row.append(avg_spread)
            heatmap.append(row)
        
        return {
            'maturities': maturities,
            'moneyness': moneyness_vals,
            'spreads': heatmap,  # 2D grid of spreads (None where no data)
            'spread_range': {
                'min': min(c['spread'] for c in comparison['comparisons']),
                'max': max(c['spread'] for c in comparison['comparisons'])
            }
        }
