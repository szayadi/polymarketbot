"""Base strategy interface."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from client import PolymarketClient
from config import Config
from positions import Position, PositionTracker
from risk import RiskManager

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Abstract base for all trading strategies."""

    name: str = "base"

    def __init__(self, client: PolymarketClient, cfg: Config,
                 risk: RiskManager, tracker: PositionTracker):
        self.client = client
        self.cfg = cfg
        self.risk = risk
        self.tracker = tracker

    @abstractmethod
    def scan(self) -> list[dict]:
        """Scan markets and return trade signals.

        Each signal dict should contain:
            token_id: str       — YES or NO token to trade
            side: str           — "BUY" or "SELL"
            price: float        — limit price
            size: float         — share quantity
            market_id: str      — condition_id
            edge: float         — expected profit margin
            reason: str         — human-readable explanation
            question: str       — market question text
        """

    def execute(self, signals: list[dict]) -> list[dict]:
        """Execute signals after risk checks. Returns results."""
        results = []
        for sig in signals:
            # Risk gate
            size = self.risk.adjust_size(sig["size"], sig["market_id"])
            if size <= 0:
                logger.info("[%s] SKIP (risk limit): %s", self.name, sig["reason"])
                continue

            allowed, deny_reason = self.risk.can_trade(size, sig["market_id"])
            if not allowed:
                logger.info("[%s] BLOCKED: %s", self.name, deny_reason)
                continue

            # Place order
            sig["size"] = size
            if sig["side"] == "BUY":
                result = self.client.buy(
                    sig["token_id"], sig["price"], size,
                    dry_run=self.cfg.dry_run,
                )
            else:
                result = self.client.sell(
                    sig["token_id"], sig["price"], size,
                    dry_run=self.cfg.dry_run,
                )

            if result:
                # Track position
                pos = Position(
                    market_id=sig["market_id"],
                    token_id=sig["token_id"],
                    side="NO" if "NO" in sig.get("reason", "").upper() else "YES",
                    entry_price=sig["price"],
                    size=size,
                    current_price=sig["price"],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    strategy=self.name,
                    order_id=result.get("order_id", ""),
                    status="open",
                    question=sig.get("question", ""),
                )
                self.tracker.add(pos)
                results.append({"signal": sig, "result": result})
                logger.info("[%s] EXECUTED: %s %s @ %.4f x%.2f | edge=%.2f%% | %s",
                            self.name, sig["side"], sig["token_id"][:12],
                            sig["price"], size, sig["edge"] * 100,
                            sig["reason"])

        return results
