"""Position tracker with JSON persistence for the Kalshi bot."""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

STATE_FILE = os.path.join(os.path.dirname(__file__), "state.json")


@dataclass
class Position:
    ticker: str             # Kalshi market ticker
    event_ticker: str       # Parent event
    side: str               # "yes" or "no"
    action: str             # "buy" or "sell"
    entry_price_cents: int  # Price paid per contract in cents
    count: int              # Number of contracts
    current_price_cents: int  # Latest market price in cents
    timestamp: str          # ISO timestamp of entry
    strategy: str           # Which strategy opened this
    order_id: str = ""      # Kalshi order ID
    status: str = "open"    # "open" | "closed"
    pnl_cents: int = 0      # Realized P&L in cents
    exit_price_cents: int = 0
    question: str = ""      # Market question for logging
    edge_predicted: float = 0.0  # Edge we expected

    @property
    def cost_cents(self) -> int:
        """Total cost of this position in cents."""
        return self.entry_price_cents * self.count

    @property
    def cost_dollars(self) -> float:
        return self.cost_cents / 100.0

    @property
    def unrealized_pnl_cents(self) -> int:
        """Unrealized P&L in cents."""
        return (self.current_price_cents - self.entry_price_cents) * self.count

    @property
    def unrealized_pnl_dollars(self) -> float:
        return self.unrealized_pnl_cents / 100.0


class PositionTracker:
    """Tracks all positions and persists to JSON."""

    def __init__(self, state_path: str = STATE_FILE):
        self.state_path = state_path
        self.positions: list[Position] = []
        self.daily_realized_pnl_cents: int = 0
        self.pnl_date: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.load()

    def add(self, position: Position) -> None:
        self.positions.append(position)
        logger.info("Position opened: %s %s %s @%dc x%d [%s]",
                     position.action, position.side, position.ticker,
                     position.entry_price_cents, position.count,
                     position.strategy)
        self.save()

    def close(self, ticker: str, pnl_cents: int,
              exit_price_cents: int = 0) -> None:
        for pos in self.positions:
            if pos.ticker == ticker and pos.status == "open":
                pos.status = "closed"
                pos.pnl_cents = pnl_cents
                pos.exit_price_cents = exit_price_cents
                self._add_daily_pnl(pnl_cents)
                logger.info("Position closed: %s PnL=%dc ($%.2f)",
                            ticker, pnl_cents, pnl_cents / 100.0)
                break
        self.save()

    def update_price(self, ticker: str, new_price_cents: int) -> None:
        for pos in self.positions:
            if pos.ticker == ticker and pos.status == "open":
                pos.current_price_cents = new_price_cents

    def get_open(self) -> list[Position]:
        return [p for p in self.positions if p.status == "open"]

    def get_open_by_strategy(self, strategy: str) -> list[Position]:
        return [p for p in self.positions
                if p.status == "open" and p.strategy == strategy]

    def get_exposure_cents(self) -> int:
        """Total capital locked in open positions (cents)."""
        return sum(p.cost_cents for p in self.get_open())

    def get_exposure_dollars(self) -> float:
        return self.get_exposure_cents() / 100.0

    def get_market_exposure_cents(self, ticker: str) -> int:
        return sum(p.cost_cents for p in self.get_open() if p.ticker == ticker)

    def get_event_exposure_cents(self, event_ticker: str) -> int:
        return sum(p.cost_cents for p in self.get_open()
                   if p.event_ticker == event_ticker)

    def has_position(self, ticker: str) -> bool:
        return any(p.ticker == ticker and p.status == "open"
                   for p in self.positions)

    def get_daily_pnl_cents(self) -> int:
        self._check_date_rollover()
        return self.daily_realized_pnl_cents

    def get_daily_pnl_dollars(self) -> float:
        return self.get_daily_pnl_cents() / 100.0

    def get_total_unrealized_pnl_cents(self) -> int:
        return sum(p.unrealized_pnl_cents for p in self.get_open())

    def _add_daily_pnl(self, cents: int) -> None:
        self._check_date_rollover()
        self.daily_realized_pnl_cents += cents

    def _check_date_rollover(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.pnl_date:
            logger.info("Daily P&L reset (was %dc)", self.daily_realized_pnl_cents)
            self.daily_realized_pnl_cents = 0
            self.pnl_date = today

    def save(self) -> None:
        data = {
            "positions": [asdict(p) for p in self.positions],
            "daily_realized_pnl_cents": self.daily_realized_pnl_cents,
            "pnl_date": self.pnl_date,
        }
        try:
            with open(self.state_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save state: %s", e)

    def load(self) -> None:
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path) as f:
                data = json.load(f)
            self.positions = [Position(**p) for p in data.get("positions", [])]
            self.daily_realized_pnl_cents = data.get("daily_realized_pnl_cents", 0)
            self.pnl_date = data.get("pnl_date", "")
            logger.info("Loaded %d positions from state", len(self.positions))
        except Exception as e:
            logger.error("Failed to load state: %s", e)

    def clear_stale(self, max_age_days: int = 14) -> int:
        """Close positions older than max_age_days as stale.

        Returns the number of positions closed.
        """
        now = datetime.now(timezone.utc)
        closed = 0
        for pos in self.positions:
            if pos.status != "open":
                continue
            try:
                entry_time = datetime.fromisoformat(pos.timestamp)
                if entry_time.tzinfo is None:
                    entry_time = entry_time.replace(tzinfo=timezone.utc)
                age_days = (now - entry_time).total_seconds() / 86400.0
                if age_days > max_age_days:
                    pos.status = "closed"
                    pos.pnl_cents = 0  # Unknown outcome — mark as break-even
                    closed += 1
                    logger.info("Cleared stale position: %s (%.0f days old)", pos.ticker, age_days)
            except (ValueError, TypeError):
                # Can't parse timestamp — close it as stale
                pos.status = "closed"
                pos.pnl_cents = 0
                closed += 1
                logger.info("Cleared stale position (bad timestamp): %s", pos.ticker)
        if closed:
            self.save()
        return closed

    def reset(self) -> int:
        """Close ALL open positions (for fresh start). Returns count closed."""
        count = sum(1 for p in self.positions if p.status == "open")
        for pos in self.positions:
            if pos.status == "open":
                pos.status = "closed"
                pos.pnl_cents = 0
        self.daily_realized_pnl_cents = 0
        self.save()
        return count

    def summary(self) -> str:
        open_pos = self.get_open()
        exposure = self.get_exposure_dollars()
        unrealized = self.get_total_unrealized_pnl_cents() / 100.0
        daily = self.get_daily_pnl_dollars()
        return (
            f"Positions: {len(open_pos)} open | "
            f"Exposure: ${exposure:.2f} | "
            f"Unrealized: ${unrealized:+.2f} | "
            f"Daily realized: ${daily:+.2f}"
        )
