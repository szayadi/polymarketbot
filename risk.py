"""Risk manager with self-preservation logic.

The bot's PRIMARY directive is survival. If the balance approaches zero,
all trading halts. Every trade must pass through risk gates.
"""

import logging

from config import Config
from positions import PositionTracker

logger = logging.getLogger(__name__)


class RiskManager:
    """Pre-trade risk checks. Prioritizes survival above all else."""

    def __init__(self, cfg: Config, tracker: PositionTracker):
        self.cfg = cfg
        self.tracker = tracker
        self._halted = False
        self._halt_reason = ""

    def update_balance(self, balance_dollars: float) -> None:
        """Update the live balance and check survival floor.

        If balance drops below survival_floor, halt ALL trading permanently
        until manual intervention.
        """
        if balance_dollars <= self.cfg.survival_floor:
            self._halted = True
            self._halt_reason = (
                f"SURVIVAL FLOOR BREACHED: ${balance_dollars:.2f} <= "
                f"${self.cfg.survival_floor:.2f} — all trading halted"
            )
            logger.critical(self._halt_reason)

    def can_trade(self, cost_cents: int, ticker: str,
                  event_ticker: str = "") -> tuple[bool, str]:
        """Check if a trade is allowed. Returns (allowed, reason)."""
        cost_dollars = cost_cents / 100.0

        # 0. Survival check — absolute priority
        if self._halted:
            return False, self._halt_reason

        # 1. Position size limit
        max_bet_cents = int(self.cfg.max_bet_size * 100)
        if cost_cents > max_bet_cents:
            return False, (f"Cost {cost_cents}c exceeds max bet "
                           f"{max_bet_cents}c")

        # 2. Market exposure limit
        current_exp = self.tracker.get_market_exposure_cents(ticker)
        max_exp_cents = int(self.cfg.max_market_exposure * 100)
        if current_exp + cost_cents > max_exp_cents:
            return False, (f"Market exposure {(current_exp + cost_cents)}c "
                           f"would exceed {max_exp_cents}c")

        # 3. Event exposure (for arbitrage — don't over-concentrate)
        if event_ticker:
            event_exp = self.tracker.get_event_exposure_cents(event_ticker)
            if event_exp + cost_cents > max_exp_cents * 2:
                return False, f"Event exposure would exceed limit"

        # 4. Daily loss cap
        daily_pnl = self.tracker.get_daily_pnl_cents()
        daily_cap_cents = int(self.cfg.daily_loss_cap * 100)
        if daily_pnl < -daily_cap_cents:
            return False, (f"Daily loss cap hit: {daily_pnl}c "
                           f"(limit: -{daily_cap_cents}c)")

        # 5. Cash reserve check
        total_exposure = self.tracker.get_exposure_cents()
        bankroll_cents = int(self.cfg.bankroll * 100)
        reserve_cents = int(self.cfg.cash_reserve * 100)
        available = bankroll_cents - total_exposure
        if available - cost_cents < reserve_cents:
            return False, (f"Cash reserve: {available}c available, "
                           f"need {cost_cents}c + {reserve_cents}c reserve")

        return True, ""

    def adjust_count(self, requested_count: int, price_cents: int,
                     ticker: str) -> int:
        """Clamp contract count to what risk limits allow."""
        cost_per = price_cents
        max_cost_cents = int(self.cfg.max_bet_size * 100)
        max_count_by_bet = max_cost_cents // cost_per if cost_per > 0 else 0

        # Don't exceed market exposure
        current_exp = self.tracker.get_market_exposure_cents(ticker)
        max_exp_cents = int(self.cfg.max_market_exposure * 100)
        remaining_exp = max(0, max_exp_cents - current_exp)
        max_count_by_exp = remaining_exp // cost_per if cost_per > 0 else 0

        # Don't breach cash reserve
        total_exp = self.tracker.get_exposure_cents()
        bankroll_cents = int(self.cfg.bankroll * 100)
        reserve_cents = int(self.cfg.cash_reserve * 100)
        available = bankroll_cents - total_exp - reserve_cents
        max_count_by_cash = max(0, available) // cost_per if cost_per > 0 else 0

        return max(0, min(requested_count, max_count_by_bet,
                          max_count_by_exp, max_count_by_cash))

    def is_halted(self) -> tuple[bool, str]:
        """Check if trading is completely halted."""
        if self._halted:
            return True, self._halt_reason

        daily_pnl = self.tracker.get_daily_pnl_cents()
        daily_cap = int(self.cfg.daily_loss_cap * 100)
        if daily_pnl < -daily_cap:
            return True, f"Daily loss cap: {daily_pnl}c (limit: -{daily_cap}c)"

        return False, ""
