"""Base strategy interface with adaptive learning integration."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

from client import KalshiClient
from config import Config
from learner import AdaptiveLearner
from positions import Position, PositionTracker
from risk import RiskManager

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Abstract base for all Kalshi trading strategies."""

    name: str = "base"

    def __init__(self, client: KalshiClient, cfg: Config,
                 risk: RiskManager, tracker: PositionTracker,
                 learner: Optional[AdaptiveLearner] = None):
        self.client = client
        self.cfg = cfg
        self.risk = risk
        self.tracker = tracker
        self.learner = learner

    @abstractmethod
    def scan(self) -> list[dict]:
        """Scan markets and return trade signals.

        Each signal dict:
            ticker: str
            event_ticker: str
            side: str           - "yes" or "no"
            action: str         - "buy" or "sell"
            price_cents: int    - limit price in cents
            count: int          - contracts
            edge: float         - expected profit margin
            reason: str
            question: str
            category: str
            series_ticker: str
        """

    def get_min_edge(self) -> float:
        """Effective minimum edge, adjusted by learner."""
        base = self.cfg.min_edge
        if self.learner:
            mult = self.learner.get_edge_multiplier(self.name)
            return base * mult
        return base

    def get_adjusted_count(self, count: int) -> int:
        """Adjust contract count by learner confidence."""
        if self.learner:
            mult = self.learner.get_size_multiplier(self.name)
            return max(1, int(count * mult))
        return count

    def execute(self, signals: list[dict]) -> list[dict]:
        """Execute signals after learner + risk checks."""
        results = []
        for sig in signals:
            # Learner gate
            if self.learner:
                ok, reason = self.learner.should_trade(
                    self.name, sig.get("category", ""), sig["price_cents"])
                if not ok:
                    logger.info("[%s] LEARNER BLOCKED: %s | %s",
                                self.name, reason, sig["reason"])
                    continue

            # Adjust count
            count = self.get_adjusted_count(sig["count"])
            cost_cents = count * sig["price_cents"]

            # Risk gate
            count = self.risk.adjust_count(count, sig["price_cents"], sig["ticker"])
            if count <= 0:
                logger.info("[%s] SKIP (risk): %s", self.name, sig["reason"])
                continue

            cost_cents = count * sig["price_cents"]
            ok, deny = self.risk.can_trade(
                cost_cents, sig["ticker"], sig.get("event_ticker", ""))
            if not ok:
                logger.info("[%s] BLOCKED: %s", self.name, deny)
                continue

            # Place order
            result = self.client.place_order(
                ticker=sig["ticker"],
                side=sig["side"],
                action=sig["action"],
                count=count,
                yes_price_cents=sig["price_cents"] if sig["side"] == "yes" else None,
                no_price_cents=sig["price_cents"] if sig["side"] == "no" else None,
                dry_run=self.cfg.dry_run,
            )

            if result:
                pos = Position(
                    ticker=sig["ticker"],
                    event_ticker=sig.get("event_ticker", ""),
                    side=sig["side"],
                    action=sig["action"],
                    entry_price_cents=sig["price_cents"],
                    count=count,
                    current_price_cents=sig["price_cents"],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    strategy=self.name,
                    order_id=result.get("order_id", ""),
                    question=sig.get("question", ""),
                    edge_predicted=sig["edge"],
                )
                self.tracker.add(pos)
                results.append({"signal": sig, "result": result})

                logger.info(
                    "[%s] EXEC: %s %s %s x%d @%dc | edge=%.1f%% | %s",
                    self.name, sig["action"], sig["side"], sig["ticker"],
                    count, sig["price_cents"], sig["edge"] * 100,
                    sig["reason"],
                )

        return results
