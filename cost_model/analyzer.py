"""
Cost Model Arbitrage Analyzer
Identifies trading opportunities based on Cost Model mispricings
"""

from typing import List, Dict, Optional
from .surface import BitcoinVolSurface
from .greeks import bs_price, calculate_greeks


class CostModelAnalyzer:
    """
    Analyzes volatility surface for arbitrage opportunities
    """
    
    # Historical average break-evens for comparison
    HISTORICAL_AVG = {
        'gamma_be': 0.028,   # ~2.8% daily
        'volga_be': 0.020,   # ~2.0% daily
        'vanna_be': -0.008,  # -0.8% vol per 1% spot
    }
    
    def __init__(self, surface: BitcoinVolSurface):
        self.surface = surface
        self.spot = surface.spot
    
    def find_opportunities(self) -> Dict:
        """Find all arbitrage opportunities"""
        return {
            'term_structure': self._analyze_term_structure(),
            'skew': self._analyze_skew(),
            'smile': self._analyze_smile(),
            'summary': self._get_regime_summary()
        }
    
    def _analyze_term_structure(self) -> List[Dict]:
        """Analyze term structure for calendar spread opportunities"""
        opportunities = []
        
        mats = sorted(self.surface.cost_params.keys())
        if len(mats) < 2:
            return opportunities
        
        for i in range(len(mats) - 1):
            T1, T2 = mats[i], mats[i+1]
            p1 = self.surface.smile_params.get(T1, {})
            p2 = self.surface.smile_params.get(T2, {})
            c1 = self.surface.cost_params[T1]
            c2 = self.surface.cost_params[T2]
            
            vol_spread = p1.get('atm_vol', 0) - p2.get('atm_vol', 0)
            gamma_spread = c1.gamma_be - c2.gamma_be
            
            mat1_label = f"{int(T1*12)}M" if T1 < 1 else f"{T1:.1f}Y"
            mat2_label = f"{int(T2*12)}M" if T2 < 1 else f"{T2:.1f}Y"
            
            if vol_spread > 0.02:  # >2 vol points backwardation
                opportunities.append({
                    'type': 'calendar_spread',
                    'direction': 'sell_short_buy_long',
                    'short_maturity': T1,
                    'long_maturity': T2,
                    'label': f"{mat1_label} vs {mat2_label}",
                    'vol_spread': vol_spread,
                    'gamma_spread': gamma_spread,
                    'conviction': 'HIGH' if vol_spread > 0.04 else 'MEDIUM',
                    'rationale': f"Backwardation: {mat1_label} vol {vol_spread*100:.1f} points above {mat2_label}"
                })
            elif vol_spread < -0.02:  # >2 vol points contango
                opportunities.append({
                    'type': 'calendar_spread',
                    'direction': 'buy_short_sell_long',
                    'short_maturity': T1,
                    'long_maturity': T2,
                    'label': f"{mat1_label} vs {mat2_label}",
                    'vol_spread': vol_spread,
                    'gamma_spread': gamma_spread,
                    'conviction': 'MEDIUM',
                    'rationale': f"Steep contango: {mat2_label} vol {-vol_spread*100:.1f} points above {mat1_label}"
                })
        
        return opportunities
    
    def _analyze_skew(self) -> List[Dict]:
        """Analyze skew for risk reversal opportunities"""
        opportunities = []
        
        for T in self.surface.maturities:
            p = self.surface.smile_params.get(T, {})
            skew = p.get('skew', 0)
            
            # Calculate actual vol skew
            put_iv = self.surface.get_vol(self.spot * 0.90, T)
            call_iv = self.surface.get_vol(self.spot * 1.10, T)
            actual_skew = put_iv - call_iv
            
            mat_label = f"{int(T*12)}M" if T < 1 else f"{T:.1f}Y"
            
            if actual_skew > 0.025:  # Puts 2.5+ vol points expensive
                opportunities.append({
                    'type': 'skew_trade',
                    'direction': 'sell_put_skew',
                    'maturity': T,
                    'label': mat_label,
                    'skew': actual_skew,
                    'put_iv': put_iv,
                    'call_iv': call_iv,
                    'conviction': 'HIGH' if actual_skew > 0.04 else 'MEDIUM',
                    'rationale': f"Puts expensive: 90/110 skew = {actual_skew*100:.1f} vol points"
                })
            elif actual_skew < -0.025:  # Calls expensive
                opportunities.append({
                    'type': 'skew_trade',
                    'direction': 'sell_call_skew',
                    'maturity': T,
                    'label': mat_label,
                    'skew': actual_skew,
                    'put_iv': put_iv,
                    'call_iv': call_iv,
                    'conviction': 'MEDIUM',
                    'rationale': f"Calls expensive: 90/110 skew = {actual_skew*100:.1f} vol points"
                })
        
        return opportunities
    
    def _analyze_smile(self) -> List[Dict]:
        """Analyze smile for butterfly opportunities"""
        opportunities = []
        
        for T in self.surface.maturities:
            c = self.surface.cost_params.get(T)
            if not c:
                continue
            
            ratio = c.volga_be / c.gamma_be if c.gamma_be != 0 else 0
            mat_label = f"{int(T*12)}M" if T < 1 else f"{T:.1f}Y"
            
            if ratio < 0.60:  # Wings cheap
                opportunities.append({
                    'type': 'smile_trade',
                    'direction': 'buy_wings',
                    'maturity': T,
                    'label': mat_label,
                    'volga_gamma_ratio': ratio,
                    'conviction': 'MEDIUM' if ratio < 0.55 else 'LOW',
                    'rationale': f"Wings cheap: Volga/Gamma ratio = {ratio:.2f}"
                })
            elif ratio > 1.0:  # Wings expensive
                opportunities.append({
                    'type': 'smile_trade',
                    'direction': 'sell_wings',
                    'maturity': T,
                    'label': mat_label,
                    'volga_gamma_ratio': ratio,
                    'conviction': 'MEDIUM' if ratio > 1.2 else 'LOW',
                    'rationale': f"Wings expensive: Volga/Gamma ratio = {ratio:.2f}"
                })
        
        return opportunities
    
    def _get_regime_summary(self) -> Dict:
        """Summarize market regime"""
        be = self.surface.get_break_evens()
        if not be:
            return {'regime': 'unknown'}
        
        # Check term structure
        if len(be) >= 2:
            front_vol = be[0]['atm_vol']
            back_vol = be[-1]['atm_vol']
            if front_vol > back_vol + 0.02:
                term_structure = 'backwardation'
            elif back_vol > front_vol + 0.02:
                term_structure = 'contango'
            else:
                term_structure = 'flat'
        else:
            term_structure = 'insufficient_data'
        
        # Check skew
        avg_corr = sum(b['spot_vol_corr'] for b in be) / len(be)
        if avg_corr < -0.15:
            skew_regime = 'deeply_negative'
        elif avg_corr < -0.05:
            skew_regime = 'negative'
        elif avg_corr > 0.05:
            skew_regime = 'positive'
        else:
            skew_regime = 'neutral'
        
        # Determine overall regime
        if term_structure == 'backwardation' and skew_regime in ['negative', 'deeply_negative']:
            regime = 'distressed'
        elif term_structure == 'contango' and skew_regime in ['neutral', 'positive']:
            regime = 'complacent'
        else:
            regime = 'mixed'
        
        return {
            'regime': regime,
            'term_structure': term_structure,
            'skew_regime': skew_regime,
            'avg_spot_vol_corr': avg_corr,
            'front_vol': be[0]['atm_vol'] if be else None,
            'back_vol': be[-1]['atm_vol'] if be else None
        }
    
    def get_trade_recommendations(self) -> List[Dict]:
        """Get specific executable trade recommendations"""
        opps = self.find_opportunities()
        trades = []
        
        # Calendar spread trades
        for opp in opps['term_structure']:
            if opp['conviction'] in ['HIGH', 'MEDIUM']:
                trade = self._build_calendar_trade(opp)
                if trade:
                    trades.append(trade)
        
        # Skew trades
        for opp in opps['skew']:
            if opp['conviction'] in ['HIGH', 'MEDIUM']:
                trade = self._build_skew_trade(opp)
                if trade:
                    trades.append(trade)
        
        # Butterfly trades
        for opp in opps['smile']:
            if opp['conviction'] in ['HIGH', 'MEDIUM']:
                trade = self._build_butterfly_trade(opp)
                if trade:
                    trades.append(trade)
        
        return trades
    
    def _build_calendar_trade(self, opp: Dict) -> Optional[Dict]:
        """Build calendar spread trade details"""
        T1, T2 = opp['short_maturity'], opp['long_maturity']
        
        iv1 = self.surface.get_vol(self.spot, T1)
        iv2 = self.surface.get_vol(self.spot, T2)
        
        call1 = bs_price(self.spot, self.spot, T1, 0, iv1, 'call')
        put1 = bs_price(self.spot, self.spot, T1, 0, iv1, 'put')
        straddle1 = call1 + put1
        
        call2 = bs_price(self.spot, self.spot, T2, 0, iv2, 'call')
        put2 = bs_price(self.spot, self.spot, T2, 0, iv2, 'put')
        straddle2 = call2 + put2
        
        mat1_label = f"{int(T1*12)}M" if T1 < 1 else f"{T1:.1f}Y"
        mat2_label = f"{int(T2*12)}M" if T2 < 1 else f"{T2:.1f}Y"
        
        return {
            'strategy': 'Calendar Spread',
            'conviction': opp['conviction'],
            'rationale': opp['rationale'],
            'legs': [
                {
                    'action': 'SELL',
                    'instrument': f'{mat1_label} ATM Straddle',
                    'strike': self.spot,
                    'maturity': T1,
                    'iv': iv1,
                    'price': straddle1,
                    'details': f"Sell {mat1_label} ${self.spot:,.0f} Call @ ${call1:,.0f} + Put @ ${put1:,.0f}"
                },
                {
                    'action': 'BUY',
                    'instrument': f'{mat2_label} ATM Straddle',
                    'strike': self.spot,
                    'maturity': T2,
                    'iv': iv2,
                    'price': straddle2,
                    'details': f"Buy {mat2_label} ${self.spot:,.0f} Call @ ${call2:,.0f} + Put @ ${put2:,.0f}"
                }
            ],
            'net_debit': straddle2 - straddle1,
            'greeks': {
                'net_gamma': 'near zero',
                'net_theta': 'positive',
                'net_vega': 'long'
            },
            'scenarios': {
                'win': 'Vol normalizes, theta decay',
                'lose': 'Front-end vol spikes further'
            }
        }
    
    def _build_skew_trade(self, opp: Dict) -> Optional[Dict]:
        """Build risk reversal trade details"""
        T = opp['maturity']
        
        put_strike = int(self.spot * 0.90)
        call_strike = int(self.spot * 1.10)
        
        put_iv = self.surface.get_vol(put_strike, T)
        call_iv = self.surface.get_vol(call_strike, T)
        
        put_price = bs_price(self.spot, put_strike, T, 0, put_iv, 'put')
        call_price = bs_price(self.spot, call_strike, T, 0, call_iv, 'call')
        
        mat_label = f"{int(T*12)}M" if T < 1 else f"{T:.1f}Y"
        
        if opp['direction'] == 'sell_put_skew':
            return {
                'strategy': 'Risk Reversal (Sell Put Skew)',
                'conviction': opp['conviction'],
                'rationale': opp['rationale'],
                'legs': [
                    {
                        'action': 'SELL',
                        'instrument': f'{mat_label} ${put_strike:,} Put',
                        'strike': put_strike,
                        'maturity': T,
                        'iv': put_iv,
                        'price': put_price,
                        'details': f"Sell {mat_label} ${put_strike:,} Put @ ${put_price:,.0f} (IV: {put_iv*100:.1f}%)"
                    },
                    {
                        'action': 'BUY',
                        'instrument': f'{mat_label} ${call_strike:,} Call',
                        'strike': call_strike,
                        'maturity': T,
                        'iv': call_iv,
                        'price': call_price,
                        'details': f"Buy {mat_label} ${call_strike:,} Call @ ${call_price:,.0f} (IV: {call_iv*100:.1f}%)"
                    }
                ],
                'net_credit': put_price - call_price,
                'skew_pickup': put_iv - call_iv,
                'greeks': {
                    'delta': 'positive (bullish)',
                    'vanna': 'positive',
                    'gamma': 'near zero'
                },
                'scenarios': {
                    'win': 'Skew normalizes, BTC rallies',
                    'lose': 'Further crash spikes put vol'
                }
            }
        return None
    
    def _build_butterfly_trade(self, opp: Dict) -> Optional[Dict]:
        """Build butterfly trade details"""
        T = opp['maturity']
        
        K_low = int(self.spot * 0.85)
        K_mid = int(self.spot)
        K_high = int(self.spot * 1.15)
        
        iv_low = self.surface.get_vol(K_low, T)
        iv_mid = self.surface.get_vol(K_mid, T)
        iv_high = self.surface.get_vol(K_high, T)
        
        put_low = bs_price(self.spot, K_low, T, 0, iv_low, 'put')
        put_mid = bs_price(self.spot, K_mid, T, 0, iv_mid, 'put')
        call_mid = bs_price(self.spot, K_mid, T, 0, iv_mid, 'call')
        call_high = bs_price(self.spot, K_high, T, 0, iv_high, 'call')
        
        straddle_mid = put_mid + call_mid
        wing_cost = put_low + call_high
        
        mat_label = f"{int(T*12)}M" if T < 1 else f"{T:.1f}Y"
        
        if opp['direction'] == 'buy_wings':
            return {
                'strategy': 'Iron Butterfly (Long Wings)',
                'conviction': opp['conviction'],
                'rationale': opp['rationale'],
                'legs': [
                    {
                        'action': 'BUY',
                        'instrument': f'{mat_label} ${K_low:,} Put',
                        'strike': K_low,
                        'maturity': T,
                        'iv': iv_low,
                        'price': put_low
                    },
                    {
                        'action': 'SELL',
                        'instrument': f'{mat_label} ${K_mid:,} Put',
                        'strike': K_mid,
                        'maturity': T,
                        'iv': iv_mid,
                        'price': put_mid
                    },
                    {
                        'action': 'SELL',
                        'instrument': f'{mat_label} ${K_mid:,} Call',
                        'strike': K_mid,
                        'maturity': T,
                        'iv': iv_mid,
                        'price': call_mid
                    },
                    {
                        'action': 'BUY',
                        'instrument': f'{mat_label} ${K_high:,} Call',
                        'strike': K_high,
                        'maturity': T,
                        'iv': iv_high,
                        'price': call_high
                    }
                ],
                'net_credit': straddle_mid - wing_cost,
                'max_profit': straddle_mid - wing_cost,
                'greeks': {
                    'delta': 'near zero',
                    'gamma': 'negative',
                    'volga': 'positive',
                    'theta': 'positive'
                },
                'scenarios': {
                    'win': f'Spot pins near ${K_mid:,}',
                    'lose': 'Large move either way'
                }
            }
        return None
