"""
Black-Scholes Greeks Calculator
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Literal


@dataclass
class Greeks:
    """Container for option Greeks"""
    delta: float
    gamma: float
    vega: float
    theta: float      # Daily theta
    vanna: float      # dDelta/dVol
    volga: float      # dVega/dVol (Vomma)
    
    def to_dict(self) -> dict:
        return {
            'delta': float(self.delta),
            'gamma': float(self.gamma),
            'vega': float(self.vega),
            'theta': float(self.theta),
            'vanna': float(self.vanna),
            'volga': float(self.volga)
        }


def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d1 in Black-Scholes"""
    return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))


def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d2 in Black-Scholes"""
    return d1(S, K, T, r, sigma) - sigma*np.sqrt(T)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, 
             option_type: Literal['call', 'put'] = 'call') -> float:
    """
    Black-Scholes option price
    
    Parameters:
    - S: Spot price
    - K: Strike price
    - T: Time to expiry (years)
    - r: Risk-free rate
    - sigma: Volatility (annualized)
    - option_type: 'call' or 'put'
    """
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    
    if option_type.lower() == 'call':
        return S * norm.cdf(d1_val) - K * np.exp(-r*T) * norm.cdf(d2_val)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)


def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float,
                    option_type: Literal['call', 'put'] = 'call') -> Greeks:
    """
    Calculate all Greeks needed for Cost Model
    """
    if T <= 0 or sigma <= 0:
        return Greeks(0, 0, 0, 0, 0, 0)
    
    sqrt_T = np.sqrt(T)
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    
    n_d1 = norm.pdf(d1_val)
    N_d1 = norm.cdf(d1_val)
    N_d2 = norm.cdf(d2_val)
    
    # Delta
    if option_type.lower() == 'call':
        delta = N_d1
        theta = (-S * n_d1 * sigma / (2*sqrt_T) - r * K * np.exp(-r*T) * N_d2)
    else:
        delta = N_d1 - 1
        theta = (-S * n_d1 * sigma / (2*sqrt_T) + r * K * np.exp(-r*T) * norm.cdf(-d2_val))
    
    # Gamma (same for call/put)
    gamma = n_d1 / (S * sigma * sqrt_T)
    
    # Vega
    vega = S * n_d1 * sqrt_T
    
    # Vanna = dDelta/dSigma = dVega/dS
    vanna = -n_d1 * d2_val / sigma
    
    # Volga = dVega/dSigma (Vomma)
    volga = vega * d1_val * d2_val / sigma
    
    return Greeks(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta / 365,  # Daily theta
        vanna=vanna,
        volga=volga
    )


def implied_volatility(price: float, S: float, K: float, T: float, r: float,
                      option_type: Literal['call', 'put'] = 'call',
                      tol: float = 1e-6, max_iter: int = 100) -> float:
    """
    Calculate implied volatility using Newton-Raphson method
    """
    if T <= 0:
        return 0.0
    
    # Initial guess
    sigma = np.sqrt(2 * np.pi / T) * price / S
    sigma = max(0.01, min(sigma, 5.0))
    
    for _ in range(max_iter):
        bs = bs_price(S, K, T, r, sigma, option_type)
        vega = calculate_greeks(S, K, T, r, sigma, option_type).vega
        
        if abs(vega) < 1e-10:
            break
            
        diff = bs - price
        if abs(diff) < tol:
            break
            
        sigma = sigma - diff / vega
        sigma = max(0.01, min(sigma, 5.0))
    
    return sigma
