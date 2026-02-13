"""Risk manager — gates every trade with bankroll and exposure checks."""

import logging

from config import Config
from positions import PositionTracker

logger = logging.getLogger(__name__)


class RiskManager:
    """Pre-trade risk checks for the trading bot."""

    def __init__(self, cfg: Config, tracker: PositionTracker):
        self.cfg = cfg
        self.tracker = tracker

    def can_trade(self, size: float, market_id: str) -> tuple[bool, str]:
        """Check if a trade is allowed under current risk limits.

        Returns:
            (allowed, reason_if_denied)
        """
        # 1. Position size limit
        if size > self.cfg.max_bet_size:
            return False, (f"Size ${size:.2f} exceeds max bet "
                           f"${self.cfg.max_bet_size:.2f}")

        # 2. Market exposure limit
        current_market_exp = self.tracker.get_market_exposure(market_id)
        if current_market_exp + size > self.cfg.max_market_exposure:
            return False, (f"Market exposure ${current_market_exp + size:.2f} "
                           f"would exceed limit ${self.cfg.max_market_exposure:.2f}")

        # 3. Daily loss cap
        daily_pnl = self.tracker.get_daily_pnl()
        if daily_pnl < -self.cfg.daily_loss_cap:
            return False, (f"Daily loss cap hit: ${daily_pnl:.2f} "
                           f"(limit: -${self.cfg.daily_loss_cap:.2f})")

        # 4. Cash reserve check
        total_exposure = self.tracker.get_exposure()
        available_cash = self.cfg.bankroll - total_exposure
        if available_cash - size < self.cfg.cash_reserve:
            return False, (f"Insufficient cash: ${available_cash:.2f} available, "
                           f"need ${size:.2f} + ${self.cfg.cash_reserve:.2f} reserve")

        return True, ""

    def adjust_size(self, requested_size: float, market_id: str) -> float:
        """Clamp trade size down to what risk limits allow."""
        size = min(requested_size, self.cfg.max_bet_size)

        # Don't exceed market exposure
        current_market_exp = self.tracker.get_market_exposure(market_id)
        max_additional = self.cfg.max_market_exposure - current_market_exp
        size = min(size, max(0, max_additional))

        # Don't dip below cash reserve
        total_exposure = self.tracker.get_exposure()
        available = self.cfg.bankroll - total_exposure - self.cfg.cash_reserve
        size = min(size, max(0, available))

        return round(size, 2)

    def is_trading_halted(self) -> tuple[bool, str]:
        """Check if trading should be halted entirely."""
        daily_pnl = self.tracker.get_daily_pnl()
        if daily_pnl < -self.cfg.daily_loss_cap:
            return True, f"Daily loss cap reached: ${daily_pnl:.2f}"
        return False, ""
