"""
Bitcoin Cost Model Package
Based on Gilbert Eid's "The Cost Model for Deriving Implied Volatility Surfaces"
"""

from .greeks import calculate_greeks, bs_price, implied_volatility, Greeks
from .surface import BitcoinVolSurface, CostParams, MarketComparison
from .analyzer import CostModelAnalyzer

__version__ = "1.0.0"
__all__ = [
    "calculate_greeks",
    "bs_price",
    "implied_volatility",
    "Greeks",
    "BitcoinVolSurface",
    "CostParams",
    "MarketComparison",
    "CostModelAnalyzer"
]
