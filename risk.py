"""Risk manager with self-preservation and dynamic scaling.

The bot's PRIMARY directive is survival. If the balance approaches zero,
all trading halts. Every trade must pass through risk gates.

Dynamic scaling: as the bankroll grows, bet sizes grow proportionally.
When losing, the bot shrinks aggressively to protect capital.
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
        self._initial_bankroll = cfg.bankroll
        self._peak_bankroll = cfg.bankroll

    def update_balance(self, balance_dollars: float) -> None:
        """Update the live balance, track high-water mark, check survival."""
        # Track peak for drawdown calculation
        if balance_dollars > self._peak_bankroll:
            self._peak_bankroll = balance_dollars

        if balance_dollars <= self.cfg.survival_floor:
            self._halted = True
            self._halt_reason = (
                f"SURVIVAL FLOOR BREACHED: ${balance_dollars:.2f} <= "
                f"${self.cfg.survival_floor:.2f} — all trading halted"
            )
            logger.critical(self._halt_reason)

    def get_growth_factor(self) -> float:
        """Scale bets based on bankroll growth from initial.

        If bankroll doubled: bet 1.5x. If bankroll halved: bet 0.5x.
        This compounds winners and protects during drawdowns.
        """
        if self._initial_bankroll <= 0:
            return 1.0
        ratio = self.cfg.bankroll / self._initial_bankroll
        # Square root scaling — aggressive but not reckless
        if ratio >= 1.0:
            return min(2.0, ratio ** 0.5)
        else:
            # Shrink faster when losing
            return max(0.3, ratio ** 1.5)

    def get_drawdown_pct(self) -> float:
        """Current drawdown from peak as a percentage."""
        if self._peak_bankroll <= 0:
            return 0.0
        return 1.0 - (self.cfg.bankroll / self._peak_bankroll)

    def can_trade(self, cost_cents: int, ticker: str,
                  event_ticker: str = "") -> tuple[bool, str]:
        """Check if a trade is allowed. Returns (allowed, reason)."""

        # 0. Survival check — absolute priority
        if self._halted:
            return False, self._halt_reason

        # Dynamic max bet based on current bankroll
        growth = self.get_growth_factor()
        max_bet_cents = int(self.cfg.max_bet_size * 100 * growth)

        # 1. Position size limit
        if cost_cents > max_bet_cents:
            return False, (f"Cost {cost_cents}c exceeds max bet "
                           f"{max_bet_cents}c (growth={growth:.2f}x)")

        # 2. Drawdown protection — reduce exposure during drawdowns
        drawdown = self.get_drawdown_pct()
        if drawdown > 0.20:
            # More than 20% drawdown from peak — reduce max exposure
            max_bet_cents = int(max_bet_cents * 0.5)
            if cost_cents > max_bet_cents:
                return False, f"Drawdown protection: {drawdown:.0%} from peak"

        # 3. Market exposure limit
        max_exp_cents = int(self.cfg.max_market_exposure * 100 * growth)
        current_exp = self.tracker.get_market_exposure_cents(ticker)
        if current_exp + cost_cents > max_exp_cents:
            return False, (f"Market exposure {(current_exp + cost_cents)}c "
                           f"would exceed {max_exp_cents}c")

        # 4. Event exposure
        if event_ticker:
            event_exp = self.tracker.get_event_exposure_cents(event_ticker)
            if event_exp + cost_cents > max_exp_cents * 2:
                return False, f"Event exposure would exceed limit"

        # 5. Daily loss cap
        daily_pnl = self.tracker.get_daily_pnl_cents()
        daily_cap_cents = int(self.cfg.daily_loss_cap * 100)
        if daily_pnl < -daily_cap_cents:
            return False, (f"Daily loss cap hit: {daily_pnl}c "
                           f"(limit: -{daily_cap_cents}c)")

        # 6. Cash reserve check
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
        if price_cents <= 0:
            return 0

        growth = self.get_growth_factor()
        cost_per = price_cents
        max_cost_cents = int(self.cfg.max_bet_size * 100 * growth)
        max_count_by_bet = max_cost_cents // cost_per

        # Drawdown reduction
        drawdown = self.get_drawdown_pct()
        if drawdown > 0.20:
            max_count_by_bet = max(1, max_count_by_bet // 2)

        # Don't exceed market exposure
        max_exp_cents = int(self.cfg.max_market_exposure * 100 * growth)
        current_exp = self.tracker.get_market_exposure_cents(ticker)
        remaining_exp = max(0, max_exp_cents - current_exp)
        max_count_by_exp = remaining_exp // cost_per

        # Don't breach cash reserve
        total_exp = self.tracker.get_exposure_cents()
        bankroll_cents = int(self.cfg.bankroll * 100)
        reserve_cents = int(self.cfg.cash_reserve * 100)
        available = bankroll_cents - total_exp - reserve_cents
        max_count_by_cash = max(0, available) // cost_per

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
