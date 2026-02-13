"""Configuration loader for the Polymarket trading bot."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    private_key: str = os.getenv("PRIVATE_KEY", "")
    chain_id: int = int(os.getenv("CHAIN_ID", "137"))
    clob_url: str = os.getenv("CLOB_URL", "https://clob.polymarket.com")
    gamma_url: str = os.getenv("GAMMA_URL", "https://gamma-api.polymarket.com")

    bankroll: float = float(os.getenv("BANKROLL", "10.0"))
    max_bet_fraction: float = float(os.getenv("MAX_BET_FRACTION", "0.05"))
    min_edge: float = float(os.getenv("MIN_EDGE", "0.03"))

    # Risk management
    daily_loss_cap_pct: float = float(os.getenv("DAILY_LOSS_CAP_PCT", "0.20"))
    cash_reserve_pct: float = float(os.getenv("CASH_RESERVE_PCT", "0.30"))
    max_market_exposure_pct: float = float(os.getenv("MAX_MARKET_EXPOSURE_PCT", "0.20"))

    # Strategy toggles
    strategy_no_bets: bool = os.getenv("STRATEGY_NO_BETS", "1") == "1"
    strategy_arbitrage: bool = os.getenv("STRATEGY_ARBITRAGE", "1") == "1"
    strategy_spread: bool = os.getenv("STRATEGY_SPREAD", "1") == "1"

    poll_interval: int = int(os.getenv("POLL_INTERVAL", "30"))
    dry_run: bool = os.getenv("DRY_RUN", "1") == "1"

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
