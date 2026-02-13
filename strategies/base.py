"""Base strategy interface with adaptive learning integration."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

from client import PolymarketClient
from config import Config
from learner import AdaptiveLearner
from positions import Position, PositionTracker
from risk import RiskManager

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Abstract base for all trading strategies.

    Integrates with the AdaptiveLearner to:
    - Check if a trade should be taken (learned blacklists, win rates)
    - Adjust edge requirements based on historical calibration
    - Scale position sizes based on strategy confidence
    - Record outcomes for future learning
    """

    name: str = "base"

    def __init__(self, client: PolymarketClient, cfg: Config,
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

        Each signal dict should contain:
            token_id: str       - YES or NO token to trade
            side: str           - "BUY" or "SELL"
            price: float        - limit price
            size: float         - share quantity
            market_id: str      - condition_id
            edge: float         - expected profit margin
            reason: str         - human-readable explanation
            question: str       - market question text
            category: str       - (optional) market category
            spread: float       - (optional) bid-ask spread at signal time
            liquidity: float    - (optional) market liquidity
        """

    def get_min_edge(self) -> float:
        """Get the effective minimum edge, adjusted by learner.

        If the learner has determined this strategy overestimates edge,
        it raises the bar. If the strategy is well-calibrated, it may
        lower it slightly.
        """
        base_edge = self.cfg.min_edge
        if self.learner:
            multiplier = self.learner.get_edge_multiplier(self.name)
            adjusted = base_edge * multiplier
            if multiplier != 1.0:
                logger.debug("[%s] Edge adjusted: %.2f%% -> %.2f%% (mult=%.2f)",
                             self.name, base_edge * 100, adjusted * 100, multiplier)
            return adjusted
        return base_edge

    def get_adjusted_size(self, base_size: float) -> float:
        """Get position size adjusted by learner confidence.

        High-performing strategies get larger sizes.
        Struggling strategies get smaller sizes.
        """
        if self.learner:
            multiplier = self.learner.get_size_multiplier(self.name)
            adjusted = base_size * multiplier
            if multiplier != 1.0:
                logger.debug("[%s] Size adjusted: %.2f -> %.2f (mult=%.2f)",
                             self.name, base_size, adjusted, multiplier)
            return adjusted
        return base_size

    def execute(self, signals: list[dict]) -> list[dict]:
        """Execute signals after risk checks and learner gate. Returns results."""
        results = []
        for sig in signals:
            # Learner gate: should we trade this at all?
            if self.learner:
                should, reason = self.learner.should_trade(
                    strategy=self.name,
                    category=sig.get("category", ""),
                    price=sig["price"],
                )
                if not should:
                    logger.info("[%s] LEARNER BLOCKED: %s | %s",
                                self.name, reason, sig["reason"])
                    continue

            # Adjust size based on learner confidence
            raw_size = sig["size"]
            adjusted_size = self.get_adjusted_size(raw_size)

            # Risk gate
            size = self.risk.adjust_size(adjusted_size, sig["market_id"])
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

                size_note = ""
                if self.learner and raw_size != size:
                    size_note = f" (learner: {raw_size:.2f}->{size:.2f})"

                logger.info("[%s] EXECUTED: %s %s @ %.4f x%.2f%s | edge=%.2f%% | %s",
                            self.name, sig["side"], sig["token_id"][:12],
                            sig["price"], size, size_note,
                            sig["edge"] * 100, sig["reason"])

        return results
