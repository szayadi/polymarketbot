"""Position tracker with JSON persistence."""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

STATE_FILE = os.path.join(os.path.dirname(__file__), "state.json")


@dataclass
class Position:
    market_id: str          # condition_id
    token_id: str           # YES or NO token
    side: str               # "YES" or "NO"
    entry_price: float      # price paid per share
    size: float             # number of shares
    current_price: float    # latest market price
    timestamp: str          # ISO timestamp of entry
    strategy: str           # which strategy opened this
    order_id: str = ""      # CLOB order ID (empty for dry-run)
    status: str = "open"    # "pending" | "open" | "closed"
    pnl: float = 0.0       # realized P&L (0 until closed)
    close_price: float = 0.0
    question: str = ""      # human-readable market question

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L based on current price."""
        if self.side == "NO":
            # Bought NO at entry_price, current NO price = 1 - current_yes_price
            return (self.current_price - self.entry_price) * self.size
        else:
            return (self.current_price - self.entry_price) * self.size

    @property
    def cost_basis(self) -> float:
        """Total cost of this position."""
        return self.entry_price * self.size


class PositionTracker:
    """Tracks all positions and persists to JSON."""

    def __init__(self, state_path: str = STATE_FILE):
        self.state_path = state_path
        self.positions: list[Position] = []
        self.daily_realized_pnl: float = 0.0
        self.pnl_date: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.load()

    def add(self, position: Position) -> None:
        """Add a new position."""
        self.positions.append(position)
        logger.info("Position opened: %s %s @ %.4f x%.2f [%s]",
                     position.side, position.token_id[:12],
                     position.entry_price, position.size, position.strategy)
        self.save()

    def close(self, market_id: str, realized_pnl: float,
              close_price: float = 0.0) -> None:
        """Close a position by market_id."""
        for pos in self.positions:
            if pos.market_id == market_id and pos.status == "open":
                pos.status = "closed"
                pos.pnl = realized_pnl
                pos.close_price = close_price
                self._add_daily_pnl(realized_pnl)
                logger.info("Position closed: %s PnL=%.4f",
                            market_id[:12], realized_pnl)
                break
        self.save()

    def update_price(self, token_id: str, new_price: float) -> None:
        """Update current price for a position."""
        for pos in self.positions:
            if pos.token_id == token_id and pos.status == "open":
                pos.current_price = new_price

    def get_open(self) -> list[Position]:
        """Get all open positions."""
        return [p for p in self.positions if p.status == "open"]

    def get_open_by_strategy(self, strategy: str) -> list[Position]:
        """Get open positions for a specific strategy."""
        return [p for p in self.positions
                if p.status == "open" and p.strategy == strategy]

    def get_exposure(self) -> float:
        """Total capital locked in open positions."""
        return sum(p.cost_basis for p in self.get_open())

    def get_market_exposure(self, market_id: str) -> float:
        """Total capital in a specific market."""
        return sum(p.cost_basis for p in self.get_open()
                   if p.market_id == market_id)

    def get_daily_pnl(self) -> float:
        """Sum of today's realized P&L."""
        self._check_date_rollover()
        return self.daily_realized_pnl

    def get_total_unrealized_pnl(self) -> float:
        """Sum of unrealized P&L across open positions."""
        return sum(p.unrealized_pnl for p in self.get_open())

    def _add_daily_pnl(self, amount: float) -> None:
        self._check_date_rollover()
        self.daily_realized_pnl += amount

    def _check_date_rollover(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.pnl_date:
            logger.info("Daily P&L reset (was %.4f)", self.daily_realized_pnl)
            self.daily_realized_pnl = 0.0
            self.pnl_date = today

    def save(self) -> None:
        """Persist state to JSON."""
        data = {
            "positions": [asdict(p) for p in self.positions],
            "daily_realized_pnl": self.daily_realized_pnl,
            "pnl_date": self.pnl_date,
        }
        try:
            with open(self.state_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save state: %s", e)

    def load(self) -> None:
        """Load state from JSON."""
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path) as f:
                data = json.load(f)
            self.positions = [Position(**p) for p in data.get("positions", [])]
            self.daily_realized_pnl = data.get("daily_realized_pnl", 0.0)
            self.pnl_date = data.get("pnl_date", "")
            logger.info("Loaded %d positions from state", len(self.positions))
        except Exception as e:
            logger.error("Failed to load state: %s", e)

    def summary(self) -> str:
        """Human-readable summary."""
        open_pos = self.get_open()
        exposure = self.get_exposure()
        unrealized = self.get_total_unrealized_pnl()
        daily = self.get_daily_pnl()
        return (
            f"Positions: {len(open_pos)} open | "
            f"Exposure: ${exposure:.2f} | "
            f"Unrealized P&L: ${unrealized:+.4f} | "
            f"Daily realized: ${daily:+.4f}"
        )
