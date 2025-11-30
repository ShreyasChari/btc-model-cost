"""
Deribit API Client
Fetches live options data from Deribit exchange
"""

import aiohttp
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DeribitClient:
    """
    Async client for Deribit public API
    No authentication needed for public endpoints
    """
    
    BASE_URL = "https://www.deribit.com/api/v2"
    TEST_URL = "https://test.deribit.com/api/v2"
    
    def __init__(self, testnet: bool = False):
        self.base_url = self.TEST_URL if testnet else self.BASE_URL
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _request(self, method: str, params: dict = None) -> dict:
        """Make API request"""
        session = await self._get_session()
        url = f"{self.base_url}/public/{method}"
        
        try:
            async with session.get(url, params=params) as response:
                data = await response.json()
                if 'error' in data:
                    raise Exception(f"Deribit API error: {data['error']}")
                return data.get('result', data)
        except aiohttp.ClientError as e:
            logger.error(f"Deribit request failed: {e}")
            raise
    
    async def get_index_price(self, currency: str = "BTC") -> float:
        """Get current index price"""
        result = await self._request("get_index_price", {
            "index_name": f"{currency.lower()}_usd"
        })
        return result['index_price']
    
    async def get_instruments(self, currency: str = "BTC", kind: str = "option") -> List[Dict]:
        """Get available instruments"""
        result = await self._request("get_instruments", {
            "currency": currency,
            "kind": kind,
            "expired": "false"
        })
        return result
    
    async def get_ticker(self, instrument_name: str) -> Dict:
        """Get ticker for specific instrument"""
        return await self._request("ticker", {
            "instrument_name": instrument_name
        })
    
    async def get_book_summary_by_currency(self, currency: str = "BTC", 
                                           kind: str = "option") -> List[Dict]:
        """Get book summary for all instruments"""
        return await self._request("get_book_summary_by_currency", {
            "currency": currency,
            "kind": kind
        })
    
    async def get_options_data(self, currency: str = "BTC") -> List[Dict]:
        """
        Fetch all options data and format for Cost Model
        Returns list of dicts with strike, maturity, implied vols
        """
        try:
            # Get all option instruments
            instruments = await self.get_instruments(currency, "option")
            
            # Get book summaries (includes mark_iv)
            summaries = await self.get_book_summary_by_currency(currency, "option")
            
            # Create lookup by instrument name
            summary_lookup = {s['instrument_name']: s for s in summaries}
            
            options = []
            now = datetime.utcnow()
            
            for inst in instruments:
                name = inst['instrument_name']
                summary = summary_lookup.get(name, {})
                
                # Parse expiration
                exp_timestamp = inst.get('expiration_timestamp', 0) / 1000
                exp_date = datetime.fromtimestamp(exp_timestamp)
                days_to_expiry = (exp_date - now).days
                
                if days_to_expiry < 3:
                    continue
                
                # Get mark IV
                mark_iv = summary.get('mark_iv')
                if mark_iv is None or mark_iv <= 0:
                    continue
                
                options.append({
                    'instrument': name,
                    'strike': inst['strike'],
                    'maturity': days_to_expiry / 365,  # Convert to years
                    'days_to_expiry': days_to_expiry,
                    'expiration': exp_date.strftime('%Y-%m-%d'),
                    'option_type': inst['option_type'],
                    'mid_iv': mark_iv / 100,  # Convert from % to decimal
                    'bid_iv': summary.get('bid_iv', mark_iv) / 100 if summary.get('bid_iv') else None,
                    'ask_iv': summary.get('ask_iv', mark_iv) / 100 if summary.get('ask_iv') else None,
                    'volume': summary.get('volume', 0),
                    'open_interest': summary.get('open_interest', 0),
                    'underlying_price': summary.get('underlying_price', 0),
                    'mark_price': summary.get('mark_price', 0),
                })
            
            return sorted(options, key=lambda x: (x['maturity'], x['strike']))
            
        except Exception as e:
            logger.error(f"Failed to fetch options data: {e}")
            raise
    
    async def get_dvol(self, currency: str = "BTC") -> float:
        """Get DVOL (30-day implied volatility index)"""
        try:
            result = await self._request("get_volatility_index_data", {
                "currency": currency,
                "resolution": "1"
            })
            if result and 'data' in result and result['data']:
                return result['data'][-1][4] / 100  # Close price, convert to decimal
            return None
        except Exception:
            return None


# Synchronous wrapper for non-async contexts
def get_deribit_data_sync(currency: str = "BTC") -> Dict:
    """
    Synchronous function to fetch all Deribit data
    Use this if you're not in an async context
    """
    async def fetch():
        client = DeribitClient()
        try:
            spot = await client.get_index_price(currency)
            options = await client.get_options_data(currency)
            dvol = await client.get_dvol(currency)
            return {
                'spot': spot,
                'options': options,
                'dvol': dvol,
                'timestamp': datetime.utcnow().isoformat()
            }
        finally:
            await client.close()
    
    return asyncio.run(fetch())
