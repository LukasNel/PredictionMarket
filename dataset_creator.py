import json
import requests
import time
from abc import ABC, abstractmethod
from typing import Optional
from datasets import Dataset, DatasetDict, Features, Value, load_dataset
from pprint import pprint

from tqdm import tqdm

class PredictionMarketDataExtractor(ABC):
    """Abstract base class for prediction market data extractors."""
    
    @abstractmethod
    def get_market_name(self) -> str:
        """Return the name of the prediction market platform."""
        pass
    
    @abstractmethod
    def fetch_closed_markets(self, max_markets: int = 100) -> list[dict]:
        """
        Fetch closed/resolved markets from the platform.
        
        Args:
            max_markets: Maximum number of markets to fetch
            
        Returns:
            List of market dictionaries with standardized fields
        """
        pass
    
    @abstractmethod
    def get_dataset_features(self) -> Features:
        """Return the HuggingFace Features schema for this platform's data."""
        pass
    
    @abstractmethod
    def to_unified_format(self, market: dict) -> dict:
        """
        Convert a platform-specific market dict to unified schema format.
        
        Args:
            market: Platform-specific market dictionary
            
        Returns:
            Market dictionary in unified schema format
        """
        pass
    
    def create_dataset(self, max_markets: int = 100) -> Dataset:
        """
        Create a HuggingFace Dataset from fetched markets.
        
        Args:
            max_markets: Maximum number of markets to fetch
            
        Returns:
            HuggingFace Dataset object
        """
        print(f"Fetching {self.get_market_name()} closed markets...")
        markets = self.fetch_closed_markets(max_markets=max_markets)
        print(f"  Fetched {len(markets)} {self.get_market_name()} markets")
        
        if not markets:
            # Return empty dataset with correct schema
            return Dataset.from_list([], features=self.get_dataset_features())
        
        return Dataset.from_list(markets, features=self.get_dataset_features())


class PolymarketDataExtractor(PredictionMarketDataExtractor):
    """Polymarket-specific data extractor."""
    
    BASE_URL = "https://gamma-api.polymarket.com"
    
    def get_market_name(self) -> str:
        return "polymarket"
    
    @staticmethod
    def parse_winning_outcome(outcomes_str: str, outcome_prices_str: str) -> tuple[str, list[str], list[float]]:
        """
        Parse Polymarket outcomes and prices to determine the winning outcome.
        
        Args:
            outcomes_str: JSON string of outcomes
            outcome_prices_str: JSON string of prices
            
        Returns:
            Tuple of (winning_outcome, outcomes_list, prices_list)
        """
        try:
            outcomes = json.loads(outcomes_str) if outcomes_str else []
            prices = json.loads(outcome_prices_str) if outcome_prices_str else []
            prices_float = [float(p) for p in prices]
            
            if outcomes and prices_float:
                max_idx = prices_float.index(max(prices_float))
                winning_outcome = outcomes[max_idx]
            else:
                winning_outcome = ""
                
            return winning_outcome, outcomes, prices_float
        except (json.JSONDecodeError, ValueError, IndexError):
            return "", [], []
    
    def fetch_closed_markets(self, max_markets: int = 100) -> list[dict]:
        """Fetch closed markets from Polymarket's Gamma API."""
        all_markets = []
        offset = 0
        limit = 100
        
        while len(all_markets) < max_markets:
            markets = self._fetch_page(limit=limit, offset=offset)
            
            if not markets:
                break
                
            all_markets.extend(markets)
            
            if len(markets) < limit:
                break
                
            offset += limit
            time.sleep(0.5)  # Rate limiting
        
        return all_markets[:max_markets]
    
    def _fetch_page(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """Fetch a single page of markets."""
        url = f"{self.BASE_URL}/markets"
        params = {
            "closed": "true",
            "limit": limit,
            "offset": offset,
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            markets = []
            for market in data:
                # Parse outcomes
                outcomes_str = market.get("outcomes", "[]")
                outcome_prices_str = market.get("outcomePrices", "[]")
                winning_outcome, outcomes_list, prices_list = self.parse_winning_outcome(
                    outcomes_str, outcome_prices_str
                )
                if sum(prices_list) == 0:
                  print("The market has no outcome")
                  continue
                
                # Extract event info
                events = market.get("events", [])
                event_ticker = events[0].get("ticker", "") if events else ""
                event_title = events[0].get("title", "") if events else ""
                
                market_info = {
                    "platform": self.get_market_name(),
                    "market_id": str(market.get("id", "")),
                    "condition_id": market.get("conditionId", ""),
                    "question": market.get("question", ""),
                    "description": market.get("description", ""),
                    "market_slug": market.get("slug", ""),
                    "resolution_source": market.get("resolutionSource", ""),
                    "start_date": market.get("startDate", ""),
                    "end_date": market.get("endDate", ""),
                    "created_at": market.get("createdAt", ""),
                    "updated_at": market.get("updatedAt", ""),
                    "closed_at": market.get("closedTime", ""),
                    "category": market.get("category", ""),
                    "outcomes": json.dumps(outcomes_list) if outcomes_list else outcomes_str,
                    "outcome_prices": json.dumps(prices_list) if prices_list else outcome_prices_str,
                    "winning_outcome": winning_outcome,
                    "volume": float(market.get("volumeNum", market.get("volume", 0)) or 0),
                    "liquidity": float(market.get("liquidityNum", market.get("liquidity", 0)) or 0),
                    "volume_24h": float(market.get("volume24hr", 0) or 0),
                    "volume_1wk": float(market.get("volume1wk", 0) or 0),
                    "volume_1mo": float(market.get("volume1mo", 0) or 0),
                    "volume_1yr": float(market.get("volume1yr", 0) or 0),
                    "open_interest": float(market.get("openInterest", 0) or 0),
                    "market_type": market.get("marketType", ""),
                    "active": market.get("active", False),
                    "closed": market.get("closed", False),
                    "archived": market.get("archived", False),
                    "featured": market.get("featured", False),
                    "restricted": market.get("restricted", False),
                    "wide_format": market.get("wideFormat", False),
                    "event_ticker": event_ticker,
                    "event_title": event_title,
                    "market_maker_address": market.get("marketMakerAddress", ""),
                    "clob_token_ids": market.get("clobTokenIds", "[]"),
                    "fee": market.get("fee", ""),
                    "spread": float(market.get("spread", 0) or 0),
                    "best_bid": float(market.get("bestBid", 0) or 0),
                    "best_ask": float(market.get("bestAsk", 0) or 0),
                    "last_trade_price": float(market.get("lastTradePrice", 0) or 0),
                    "one_day_price_change": float(market.get("oneDayPriceChange", 0) or 0),
                    "one_week_price_change": float(market.get("oneWeekPriceChange", 0) or 0),
                    # Store original full response for future feature recovery
                    "original": json.dumps(market),
                }
                markets.append(market_info)
                
            return markets
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Polymarket markets: {e}")
            return []
    
    def get_dataset_features(self) -> Features:
        """Return the HuggingFace Features schema for Polymarket data."""
        return Features({
            "platform": Value("string"),
            "market_id": Value("string"),
            "condition_id": Value("string"),
            "question": Value("string"),
            "description": Value("string"),
            "market_slug": Value("string"),
            "resolution_source": Value("string"),
            "start_date": Value("string"),
            "end_date": Value("string"),
            "created_at": Value("string"),
            "updated_at": Value("string"),
            "closed_at": Value("string"),
            "category": Value("string"),
            "outcomes": Value("string"),
            "outcome_prices": Value("string"),
            "winning_outcome": Value("string"),
            "volume": Value("float64"),
            "liquidity": Value("float64"),
            "volume_24h": Value("float64"),
            "volume_1wk": Value("float64"),
            "volume_1mo": Value("float64"),
            "volume_1yr": Value("float64"),
            "open_interest": Value("float64"),
            "market_type": Value("string"),
            "active": Value("bool"),
            "closed": Value("bool"),
            "archived": Value("bool"),
            "featured": Value("bool"),
            "restricted": Value("bool"),
            "wide_format": Value("bool"),
            "event_ticker": Value("string"),
            "event_title": Value("string"),
            "market_maker_address": Value("string"),
            "clob_token_ids": Value("string"),
            "fee": Value("string"),
            "spread": Value("float64"),
            "best_bid": Value("float64"),
            "best_ask": Value("float64"),
            "last_trade_price": Value("float64"),
            "one_day_price_change": Value("float64"),
            "one_week_price_change": Value("float64"),
            "original": Value("string"),  # Full original JSON response
        })
    
    def to_unified_format(self, market: dict) -> dict:
        """Convert Polymarket market data to unified schema format."""
        # Parse outcomes and outcome_prices if they're JSON strings
        outcomes = market.get("outcomes", "")
        outcome_prices = market.get("outcome_prices", "")
        
        # If already JSON strings, keep them; otherwise convert
        try:
            # Test if it's valid JSON
            json.loads(outcomes)
            outcomes_json = outcomes
        except (json.JSONDecodeError, TypeError):
            # If it's a comma-separated string, convert to JSON
            if outcomes and isinstance(outcomes, str):
                outcomes_json = json.dumps(outcomes.split(","))
            else:
                outcomes_json = json.dumps([])
        
        try:
            # Test if it's valid JSON
            json.loads(outcome_prices)
            prices_json = outcome_prices
        except (json.JSONDecodeError, TypeError):
            # If it's a comma-separated string, convert to JSON
            if outcome_prices and isinstance(outcome_prices, str):
                prices_json = json.dumps([float(p) for p in outcome_prices.split(",")])
            else:
                prices_json = json.dumps([])
        
        return {
            "platform": "polymarket",
            "market_id": str(market.get("market_id", "")),
            "question": str(market.get("question", "")),
            "description": str(market.get("description", "")),
            "category": str(market.get("category", "")),
            "winning_outcome": str(market.get("winning_outcome", "")),
            "outcomes": outcomes_json,
            "outcome_prices": prices_json,
            "resolution_source": str(market.get("resolution_source", "")),
            "close_time": str(market.get("closed_at", "")),
            "volume": float(market.get("volume", 0) or 0),
            "liquidity": float(market.get("liquidity", 0) or 0),
            "open_interest": float(market.get("open_interest", 0) or 0),
            "market_type": str(market.get("market_type", "")),
            "original": str(market.get("original", "")),
        }


class KalshiDataExtractor(PredictionMarketDataExtractor):
    """Kalshi-specific data extractor."""
    
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    
    def get_market_name(self) -> str:
        return "kalshi"
    
    def fetch_closed_markets(self, max_markets: int = 100) -> list[dict]:
        """Fetch closed markets from Kalshi API by fetching events with nested markets."""
        all_markets = []
        cursor = None
        limit = 200  # Max allowed by API
        pbar = tqdm(total=max_markets, desc="Fetching markets")
        while len(all_markets) < max_markets:
            markets, next_cursor = self._fetch_page(limit=limit, cursor=cursor)
            
            if not markets:
                break
            pbar.update(len(markets))
            all_markets.extend(markets)
            
            if not next_cursor or len(markets) == 0:
                break
                
            cursor = next_cursor
            time.sleep(0.1)  # Rate limiting
        
        return all_markets[:max_markets]
    
    def _fetch_page(self, limit: int = 200, cursor: Optional[str] = None) -> tuple[list[dict], Optional[str]]:
        """Fetch a single page of events with nested markets."""
        url = f"{self.BASE_URL}/events"
        params = {
            "status": "settled",
            "limit": limit,
            "with_nested_markets": "true",
        }
        
        if cursor:
            params["cursor"] = cursor
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            markets = []
            for event in data.get("events", []):
                event_ticker = event.get("event_ticker", "")
                event_category = event.get("category", "")
                event_title = event.get("title", "")
                
                # Parse nested markets
                for market in event.get("markets", []):
                    market_info = {
                        "platform": self.get_market_name(),
                        "market_id": market.get("ticker", ""),
                        "event_ticker": event_ticker,\
                        "series_ticker": event.get("series_ticker", ""),
                        "event_title": event_title,
                        "question": market.get("title", ""),
                        "description": market.get("rules_primary", "") + "\n" + market.get("rules_secondary", ""),
                        "rules_secondary": market.get("rules_secondary", ""),
                        "subtitle": market.get("subtitle", ""),
                        "yes_sub_title": market.get("yes_sub_title", ""),
                        "no_sub_title": market.get("no_sub_title", ""),
                        "market_slug": market.get("ticker", ""),
                        "category": event_category,  # Get category from event
                        "market_type": market.get("market_type", ""),
                        "status": market.get("status", ""),
                        "result": market.get("result", ""),
                        "winning_outcome": market.get("result", ""),
                        "open_time": market.get("open_time", ""),
                        "close_time": market.get("close_time", ""),
                        "expected_expiration_time": market.get("expected_expiration_time", ""),
                        "expiration_time": market.get("expiration_time", ""),
                        "latest_expiration_time": market.get("latest_expiration_time", ""),
                        "expiration_value": market.get("expiration_value", ""),
                        "settlement_value": float(market.get("settlement_value", 0) or 0),
                        "settlement_value_dollars": market.get("settlement_value_dollars", ""),
                        "settlement_timer_seconds": int(market.get("settlement_timer_seconds", 0) or 0),
                        "can_close_early": market.get("can_close_early", False),
                        "early_close_condition": market.get("early_close_condition", ""),
                        "volume": float(market.get("volume", 0) or 0),
                        "volume_24h": float(market.get("volume_24h", 0) or 0),
                        "open_interest": float(market.get("open_interest", 0) or 0),
                        "liquidity": float(market.get("liquidity", 0) or 0),
                        "liquidity_dollars": market.get("liquidity_dollars", ""),
                        "last_price": float(market.get("last_price", 0) or 0),
                        "last_price_dollars": market.get("last_price_dollars", ""),
                        "previous_price": float(market.get("previous_price", 0) or 0),
                        "previous_price_dollars": market.get("previous_price_dollars", ""),
                        "yes_bid": float(market.get("yes_bid", 0) or 0),
                        "yes_bid_dollars": market.get("yes_bid_dollars", ""),
                        "yes_ask": float(market.get("yes_ask", 0) or 0),
                        "yes_ask_dollars": market.get("yes_ask_dollars", ""),
                        "no_bid": float(market.get("no_bid", 0) or 0),
                        "no_bid_dollars": market.get("no_bid_dollars", ""),
                        "no_ask": float(market.get("no_ask", 0) or 0),
                        "no_ask_dollars": market.get("no_ask_dollars", ""),
                        "previous_yes_bid": float(market.get("previous_yes_bid", 0) or 0),
                        "previous_yes_bid_dollars": market.get("previous_yes_bid_dollars", ""),
                        "previous_yes_ask": float(market.get("previous_yes_ask", 0) or 0),
                        "previous_yes_ask_dollars": market.get("previous_yes_ask_dollars", ""),
                        "notional_value": float(market.get("notional_value", 0) or 0),
                        "notional_value_dollars": market.get("notional_value_dollars", ""),
                        "tick_size": float(market.get("tick_size", 0) or 0),
                        "risk_limit_cents": float(market.get("risk_limit_cents", 0) or 0),
                        "strike_type": market.get("strike_type", ""),
                        "floor_strike": float(market.get("floor_strike", 0) or 0),
                        "cap_strike": float(market.get("cap_strike", 0) or 0),
                        "functional_strike": market.get("functional_strike", ""),
                        "custom_strike": str(market.get("custom_strike", {})) if market.get("custom_strike") else "",
                        "price_level_structure": market.get("price_level_structure", ""),
                        "price_ranges": str(market.get("price_ranges", [])) if market.get("price_ranges") else "",
                        "response_price_units": market.get("response_price_units", ""),
                        "mve_collection_ticker": market.get("mve_collection_ticker", ""),
                        "mve_selected_legs": str(market.get("mve_selected_legs", [])) if market.get("mve_selected_legs") else "",
                        # Store original full response for future feature recovery
                        "original": {"market": json.dumps(market), "event": json.dumps(event)},
                        "event_sub_title": event.get("sub_title", ""),
                        "event_collateral_return_type": event.get("collateral_return_type", ""),
                        "event_mutually_exclusive": event.get("mutually_exclusive", False),
                        "event_strike_date": event.get("strike_date", ""),
                        "event_strike_period": market.get("strike_period", ""),
                    }
                    markets.append(market_info)
            
            next_cursor = data.get("cursor")
            return markets, next_cursor
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Kalshi events: {e}")
            return [], None
    
    def get_dataset_features(self) -> Features:
        """Return the HuggingFace Features schema for Kalshi data."""
        return Features({
            "platform": Value("string"),
            "series_ticker": Value("string"),
            "market_id": Value("string"),
            "event_ticker": Value("string"),
            "event_title": Value("string"),
            "question": Value("string"),
            "description": Value("string"),
            "rules_secondary": Value("string"),
            "subtitle": Value("string"),
            "yes_sub_title": Value("string"),
            "no_sub_title": Value("string"),
            "market_slug": Value("string"),
            "category": Value("string"),
            "market_type": Value("string"),
            "status": Value("string"),
            "result": Value("string"),
            "winning_outcome": Value("string"),
            "open_time": Value("string"),
            "close_time": Value("string"),
            "expected_expiration_time": Value("string"),
            "expiration_time": Value("string"),
            "latest_expiration_time": Value("string"),
            "expiration_value": Value("string"),
            "settlement_value": Value("float64"),
            "settlement_value_dollars": Value("string"),
            "settlement_timer_seconds": Value("int64"),
            "can_close_early": Value("bool"),
            "early_close_condition": Value("string"),
            "volume": Value("float64"),
            "volume_24h": Value("float64"),
            "open_interest": Value("float64"),
            "liquidity": Value("float64"),
            "liquidity_dollars": Value("string"),
            "last_price": Value("float64"),
            "last_price_dollars": Value("string"),
            "previous_price": Value("float64"),
            "previous_price_dollars": Value("string"),
            "yes_bid": Value("float64"),
            "yes_bid_dollars": Value("string"),
            "yes_ask": Value("float64"),
            "yes_ask_dollars": Value("string"),
            "no_bid": Value("float64"),
            "no_bid_dollars": Value("string"),
            "no_ask": Value("float64"),
            "no_ask_dollars": Value("string"),
            "previous_yes_bid": Value("float64"),
            "previous_yes_bid_dollars": Value("string"),
            "previous_yes_ask": Value("float64"),
            "previous_yes_ask_dollars": Value("string"),
            "notional_value": Value("float64"),
            "notional_value_dollars": Value("string"),
            "tick_size": Value("float64"),
            "risk_limit_cents": Value("float64"),
            "strike_type": Value("string"),
            "floor_strike": Value("float64"),
            "cap_strike": Value("float64"),
            "functional_strike": Value("string"),
            "custom_strike": Value("string"),
            "price_level_structure": Value("string"),
            "price_ranges": Value("string"),
            "response_price_units": Value("string"),
            "mve_collection_ticker": Value("string"),
            "mve_selected_legs": Value("string"),
            "original": Value("string"),  # Full original JSON response
            "event_sub_title": Value("string"),
            "event_collateral_return_type": Value("string"),
            "event_mutually_exclusive": Value("bool"),
            "event_strike_date": Value("string"),
            "event_strike_period": Value("string"),
        })
    
    def to_unified_format(self, market: dict) -> dict:
        """Convert Kalshi market data to unified schema format."""
        # Calculate outcome prices from bid/ask midpoints
        yes_bid = float(market.get("yes_bid", 0) or 0)
        yes_ask = float(market.get("yes_ask", 0) or 0)
        no_bid = float(market.get("no_bid", 0) or 0)
        no_ask = float(market.get("no_ask", 0) or 0)
        
        # Use midpoint of bid/ask or just bid if ask is 0
        yes_price = (yes_bid + yes_ask) / 2 if yes_ask > 0 else yes_bid
        no_price = (no_bid + no_ask) / 2 if no_ask > 0 else no_bid
        
        # Normalize to 0-1 range (Kalshi prices are in cents, 0-100)
        yes_price = yes_price / 100 if yes_price > 0 else 0
        no_price = no_price / 100 if no_price > 0 else 0
        
        return {
            "platform": "kalshi",
            "market_id": str(market.get("market_id", "")),
            "question": str(market.get("question", "")),
            "description": str(market.get("description", "")),
            "category": str(market.get("category", "")),
            "winning_outcome": str(market.get("winning_outcome", "")),
            "outcomes": json.dumps(["Yes", "No"]),  # Kalshi is always binary
            "outcome_prices": json.dumps([yes_price, no_price]),
            "resolution_source": "",  # Kalshi doesn't have this field
            "close_time": str(market.get("close_time", "")),
            "volume": float(market.get("volume", 0) or 0),
            "liquidity": float(market.get("liquidity", 0) or 0),
            "open_interest": float(market.get("open_interest", 0) or 0),
            "market_type": str(market.get("market_type", "")),
            "original": str(market.get("original", "")),
            "series_ticker": str(market.get("series_ticker", "")),
            "event_sub_title": str(market.get("event_sub_title", "")),
            "event_collateral_return_type": str(market.get("event_collateral_return_type", "")),
            "event_mutually_exclusive": bool(market.get("event_mutually_exclusive", False)),
            "event_strike_date": str(market.get("event_strike_date", "")),
            "event_strike_period": str(market.get("event_strike_period", "")),
        }


class PredictionMarketDataOrchestrator:
    """Orchestrates multiple prediction market data extractors into a unified dataset."""
    
    def __init__(self, extractors: list[PredictionMarketDataExtractor] = None):
        """
        Initialize the orchestrator with a list of extractors.
        
        Args:
            extractors: List of PredictionMarketDataExtractor instances.
                       If None, uses default extractors (Polymarket, Kalshi).
        """
        self.extractors = extractors
        # Create extractors map: platform_name -> extractor
        self.extractors_map = {
            extractor.get_market_name(): extractor 
            for extractor in self.extractors
        }
    
    def add_extractor(self, extractor: PredictionMarketDataExtractor):
        """Add an extractor to the orchestrator."""
        self.extractors.append(extractor)
        # Update the extractors map
        self.extractors_map[extractor.get_market_name()] = extractor
    
    def create_unified_schema(self) -> Features:
        """Return the unified schema for cross-platform analysis."""
        return Features({
            "platform": Value("string"),
            "market_id": Value("string"),
            "question": Value("string"),
            "description": Value("string"),
            "category": Value("string"),
            "winning_outcome": Value("string"),
            "outcomes": Value("string"),
            "outcome_prices": Value("string"),
            "resolution_source": Value("string"),
            "close_time": Value("string"),
            "volume": Value("float64"),
            "liquidity": Value("float64"),
            "open_interest": Value("float64"),
            "market_type": Value("string"),
            "original": Value("string"),  # Full original JSON response
            "series_ticker": Value("string"),
            "event_sub_title": Value("string"),
            "event_collateral_return_type": Value("string"),
            "event_mutually_exclusive": Value("bool"),
            "event_strike_date": Value("string"),
            "event_strike_period": Value("string"),
        })
    
    def _convert_to_unified(self, datasets: dict[str, Dataset], extractors_map: dict[str, PredictionMarketDataExtractor]) -> Dataset:
        """
        Convert platform-specific datasets to unified schema.
        
        Args:
            datasets: Dictionary of platform name to Dataset
            extractors_map: Dictionary mapping platform names to their extractors
            
        Returns:
            Unified Dataset with common fields
        """
        unified_markets = []
        
        for platform_name, dataset in datasets.items():
            extractor = extractors_map.get(platform_name)
            if not extractor:
                continue
                
            for idx in range(len(dataset)):
                market = dataset[idx]
                unified = extractor.to_unified_format(market)
                unified_markets.append(unified)
        
        return Dataset.from_list(unified_markets, features=self.create_unified_schema())
    
    def fetch_all(self, max_markets_per_platform: int = 100) -> DatasetDict:
        """
        Fetch data from all extractors and create a unified DatasetDict.
        
        Args:
            max_markets_per_platform: Maximum markets to fetch per platform
            
        Returns:
            DatasetDict with 'unified' split and individual platform splits
        """
        print("=" * 60)
        print("Prediction Markets Data Orchestrator")
        print("=" * 60)
        
        # Fetch from each extractor
        platform_datasets = {}
        
        for extractor in self.extractors:
            platform_name = extractor.get_market_name()
            print(f"\nProcessing {platform_name}...")
            
            dataset = extractor.create_dataset(max_markets=max_markets_per_platform)
            platform_datasets[platform_name] = dataset
            
            print(f"  {platform_name}: {len(dataset)} markets")
        
        # Create unified dataset
        print("\nCreating unified dataset...")
        unified_dataset = self._convert_to_unified(platform_datasets, self.extractors_map)
        print(f"  Unified: {len(unified_dataset)} total markets")
        
        # Build DatasetDict
        dataset_dict = {"unified": unified_dataset}
        dataset_dict.update(platform_datasets)
        
        return DatasetDict(dataset_dict)
    
    def save_datasets(self, dataset_dict: DatasetDict, output_dir: str = "."):
        """
        Save all datasets to disk in various formats.
        
        Args:
            dataset_dict: The DatasetDict to save
            output_dir: Directory to save files to
        """
        # Save full DatasetDict
        dataset_path = f"{output_dir}/prediction_markets_dataset"
        print(f"\nSaving DatasetDict to {dataset_path}...")
        dataset_dict.save_to_disk(dataset_path)
        print("  DatasetDict saved!")
        
        # Save individual splits as parquet
        print("\nSaving individual splits as parquet files...")
        for split_name in dataset_dict.keys():
            parquet_path = f"{output_dir}/prediction_markets_{split_name}.parquet"
            dataset_dict[split_name].to_parquet(parquet_path)
            print(f"  Saved {parquet_path}")
        
        print("\nAll files saved!")



def main():
    """Main function demonstrating the orchestrator pattern."""
    import sys
    
    # Initialize extractors
    polymarket_extractor = PolymarketDataExtractor()
    kalshi_extractor = KalshiDataExtractor()

    orchestrator = PredictionMarketDataOrchestrator([kalshi_extractor])
    
    # Fetch all data
    max_markets = 1000000
    dataset_dict = orchestrator.fetch_all(max_markets_per_platform=max_markets)
    
    # Display results
    print("\n" + "=" * 60)
    print("Dataset Summary:")
    print("=" * 60)
    for split_name in dataset_dict.keys():
        print(f"  {split_name}: {len(dataset_dict[split_name])} rows")
    
    # Show sample with original field
    print("\n" + "=" * 60)
    print("Sample from UNIFIED dataset (with original field):")
    print("=" * 60)
    if len(dataset_dict['unified']) > 0:
        sample = dataset_dict['unified'][0]
        print(f"Platform: {sample['platform']}")
        print(f"Question: {sample['question'][:80]}...")
        print(f"Winning Outcome: {sample['winning_outcome']}")
        print(f"Original (first 200 chars): {sample['original'][:200]}...")
    
    # Save datasets
    orchestrator.save_datasets(dataset_dict, output_dir=".")
    
    
    print("\n" + "=" * 60)
    print("To push to HuggingFace Hub:")
    print("=" * 60)
    print("""
# huggingface-cli login
# dataset_dict.push_to_hub("your-username/prediction-markets-historical")
""")
    
    return dataset_dict


if __name__ == "__main__":
    dataset = main()
    dataset['unified'].push_to_hub("2084Collective/prediction-markets-historical-v5")
    for data in dataset['unified']:
        pprint(data)
        break
    # for data in dataset['polymarket']:
    #     pprint(data)
    #     break
    for data in dataset['kalshi']:
        pprint(data)
        break
    uploaded_dataset = load_dataset("2084Collective/prediction-markets-historical-v5")
    def clean_dataset(data_item):
        accept = True
        accept = accept and data_item['platform'] == 'kalshi'
        accept = accept and (data_item['winning_outcome'].lower() == 'yes' or data_item['winning_outcome'].lower() == 'no')
        return accept
    print("Prior to cleaning: ", len(uploaded_dataset['train']))
    cleaned_dataset = uploaded_dataset.filter(clean_dataset)
    print("After cleaning: ", len(cleaned_dataset['train']))
    cleaned_dataset.push_to_hub("2084Collective/prediction-markets-historical-v5-cleaned")