"""Configuration loader for the Kalshi trading bot."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

KALSHI_PROD_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"
KALSHI_PROD_WS = "wss://api.elections.kalshi.com/trade-api/ws/v2"
KALSHI_DEMO_WS = "wss://demo-api.kalshi.co/trade-api/ws/v2"


@dataclass
class Config:
    api_key_id: str = os.getenv("KALSHI_API_KEY_ID", "")
    private_key_path: str = os.getenv("KALSHI_PRIVATE_KEY_PATH", "./private_key.pem")
    env: str = os.getenv("KALSHI_ENV", "demo")

    bankroll: float = float(os.getenv("BANKROLL", "10.0"))
    max_bet_fraction: float = float(os.getenv("MAX_BET_FRACTION", "0.15"))
    min_edge: float = float(os.getenv("MIN_EDGE", "0.01"))

    # Risk management — aggressive capital deployment
    daily_loss_cap_pct: float = float(os.getenv("DAILY_LOSS_CAP_PCT", "0.25"))
    cash_reserve_pct: float = float(os.getenv("CASH_RESERVE_PCT", "0.15"))
    max_market_exposure_pct: float = float(os.getenv("MAX_MARKET_EXPOSURE_PCT", "0.30"))
    survival_floor: float = float(os.getenv("SURVIVAL_FLOOR", "1.00"))

    # Strategy toggles
    strategy_dutch_book: bool = os.getenv("STRATEGY_DUTCH_BOOK", "1") == "1"
    strategy_tail_bets: bool = os.getenv("STRATEGY_TAIL_BETS", "1") == "1"
    strategy_spread: bool = os.getenv("STRATEGY_SPREAD", "1") == "1"
    strategy_momentum: bool = os.getenv("STRATEGY_MOMENTUM", "1") == "1"

    # Time filters — AGGRESSIVE: only fast-resolving markets (1-3 days)
    max_days_to_resolve: int = int(os.getenv("MAX_DAYS_TO_RESOLVE", "3"))
    # Minimum 24h volume to consider a market (in contracts)
    min_volume_24h: int = int(os.getenv("MIN_VOLUME_24H", "25"))

    poll_interval: int = int(os.getenv("POLL_INTERVAL", "10"))
    dry_run: bool = os.getenv("DRY_RUN", "1") == "1"

    @property
    def base_url(self) -> str:
        return KALSHI_PROD_URL if self.env == "production" else KALSHI_DEMO_URL

    @property
    def ws_url(self) -> str:
        return KALSHI_PROD_WS if self.env == "production" else KALSHI_DEMO_WS

    @property
    def max_bet_size(self) -> float:
        return self.bankroll * self.max_bet_fraction

    @property
    def daily_loss_cap(self) -> float:
        return self.bankroll * self.daily_loss_cap_pct

    @property
    def cash_reserve(self) -> float:
        return self.bankroll * self.cash_reserve_pct

    @property
    def max_market_exposure(self) -> float:
        return self.bankroll * self.max_market_exposure_pct
