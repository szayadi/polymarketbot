"""Adaptive learning engine.

Tracks every trade outcome and builds a statistical model of what works.
Adjusts strategy parameters dynamically based on rolling performance.

The learner answers three questions:
  1. Should I trade this market? (market scoring)
  2. How much should I bet? (confidence-weighted sizing)
  3. Should I change my thresholds? (parameter adaptation)
"""

import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

LEARN_FILE = os.path.join(os.path.dirname(__file__), "learned.json")

# Minimum trades before the learner starts influencing decisions
MIN_TRADES_FOR_CONFIDENCE = 5

# How fast old data decays (0 = no decay, 1 = forget everything)
# 0.05 = recent trades weigh ~5% more per trade than old ones
DECAY_RATE = 0.05

# Blacklist threshold: if win rate drops below this, avoid the category
BLACKLIST_WIN_RATE = 0.30


@dataclass
class TradeRecord:
    """A single completed trade for learning."""
    timestamp: str
    strategy: str
    market_id: str
    token_id: str
    side: str               # "YES" or "NO"
    entry_price: float
    exit_price: float       # 1.0 if resolved in our favor, 0.0 if against
    size: float
    pnl: float              # realized profit/loss
    edge_predicted: float   # edge we expected when entering
    edge_actual: float      # actual edge realized
    category: str           # market category (sports, politics, etc.)
    question: str
    price_range: str        # bucketed: "0-20", "20-40", "40-60", "60-80", "80-100"
    spread_at_entry: float  # bid-ask spread when we entered
    liquidity_at_entry: float
    won: bool               # did the trade make money?


@dataclass
class StrategyStats:
    """Rolling statistics for a strategy."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_edge_predicted: float = 0.0
    avg_edge_actual: float = 0.0
    win_rate: float = 0.0
    # Weighted stats (recent trades matter more)
    weighted_win_rate: float = 0.0
    weighted_avg_pnl: float = 0.0


@dataclass
class CategoryStats:
    """Performance by market category."""
    total_trades: int = 0
    wins: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    blacklisted: bool = False


@dataclass
class PriceRangeStats:
    """Performance by entry price range."""
    total_trades: int = 0
    wins: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0


class AdaptiveLearner:
    """Learns from trade outcomes and adjusts strategy behavior."""

    def __init__(self, learn_path: str = LEARN_FILE):
        self.learn_path = learn_path
        self.trades: list[TradeRecord] = []

        # Aggregated stats
        self.strategy_stats: dict[str, StrategyStats] = defaultdict(StrategyStats)
        self.category_stats: dict[str, CategoryStats] = defaultdict(CategoryStats)
        self.price_range_stats: dict[str, dict[str, PriceRangeStats]] = defaultdict(
            lambda: defaultdict(PriceRangeStats)
        )

        # Learned parameter adjustments
        self.edge_multipliers: dict[str, float] = {}  # strategy -> multiplier
        self.size_multipliers: dict[str, float] = {}   # strategy -> multiplier

        self.load()

    # ── Recording outcomes ───────────────────────────────────────

    def record_trade(self, strategy: str, market_id: str, token_id: str,
                     side: str, entry_price: float, exit_price: float,
                     size: float, pnl: float, edge_predicted: float,
                     category: str = "", question: str = "",
                     spread_at_entry: float = 0.0,
                     liquidity_at_entry: float = 0.0) -> None:
        """Record a completed trade outcome for learning."""
        won = pnl > 0
        edge_actual = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
        price_range = self._price_bucket(entry_price)

        record = TradeRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            strategy=strategy,
            market_id=market_id,
            token_id=token_id,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            size=size,
            pnl=pnl,
            edge_predicted=edge_predicted,
            edge_actual=edge_actual,
            category=category,
            question=question,
            price_range=price_range,
            spread_at_entry=spread_at_entry,
            liquidity_at_entry=liquidity_at_entry,
            won=won,
        )
        self.trades.append(record)

        logger.info("[learner] Recorded: %s %s %s pnl=%.4f (predicted=%.2f%% actual=%.2f%%)",
                    strategy, "WIN" if won else "LOSS", market_id[:12],
                    pnl, edge_predicted * 100, edge_actual * 100)

        # Recompute stats
        self._recompute_stats()
        self._adapt_parameters()
        self.save()

    # ── Querying learned intelligence ────────────────────────────

    def should_trade(self, strategy: str, category: str = "",
                     price: float = 0.0) -> tuple[bool, str]:
        """Should we take this trade based on historical performance?

        Returns (should_trade, reason).
        """
        # Not enough data yet — allow all trades
        stats = self.strategy_stats.get(strategy)
        if not stats or stats.total_trades < MIN_TRADES_FOR_CONFIDENCE:
            return True, "insufficient data — allowing trade"

        # Check if category is blacklisted
        if category:
            cat_stats = self.category_stats.get(category)
            if cat_stats and cat_stats.blacklisted:
                return False, (f"category '{category}' blacklisted "
                               f"(win rate {cat_stats.win_rate:.0%})")

        # Check if this price range is historically bad for this strategy
        price_range = self._price_bucket(price)
        pr_stats = self.price_range_stats.get(strategy, {}).get(price_range)
        if (pr_stats and pr_stats.total_trades >= MIN_TRADES_FOR_CONFIDENCE
                and pr_stats.win_rate < BLACKLIST_WIN_RATE):
            return False, (f"price range {price_range} has {pr_stats.win_rate:.0%} "
                           f"win rate for {strategy}")

        # Check if strategy overall is performing badly
        if (stats.weighted_win_rate < BLACKLIST_WIN_RATE
                and stats.total_trades >= MIN_TRADES_FOR_CONFIDENCE * 2):
            return False, (f"strategy {strategy} weighted win rate "
                           f"{stats.weighted_win_rate:.0%} below threshold")

        return True, "trade approved by learner"

    def get_edge_multiplier(self, strategy: str) -> float:
        """Get the learned edge requirement multiplier.

        > 1.0 = require more edge (strategy has been overconfident)
        < 1.0 = require less edge (strategy has been too conservative)
        Default: 1.0
        """
        return self.edge_multipliers.get(strategy, 1.0)

    def get_size_multiplier(self, strategy: str) -> float:
        """Get the learned position size multiplier.

        Scales bet size based on strategy confidence.
        High win rate + good calibration → bet more.
        Low win rate → bet less.
        Default: 1.0
        """
        return self.size_multipliers.get(strategy, 1.0)

    def get_report(self) -> str:
        """Human-readable performance report."""
        lines = ["=== Learner Report ==="]
        lines.append(f"Total trades recorded: {len(self.trades)}")

        for name, stats in self.strategy_stats.items():
            lines.append(f"\n[{name}]")
            lines.append(f"  Trades: {stats.total_trades} "
                         f"(W:{stats.wins} L:{stats.losses})")
            lines.append(f"  Win rate: {stats.win_rate:.0%} "
                         f"(weighted: {stats.weighted_win_rate:.0%})")
            lines.append(f"  Total P&L: ${stats.total_pnl:+.4f}")
            lines.append(f"  Avg predicted edge: {stats.avg_edge_predicted:.2%}")
            lines.append(f"  Avg actual edge: {stats.avg_edge_actual:.2%}")
            mult_e = self.edge_multipliers.get(name, 1.0)
            mult_s = self.size_multipliers.get(name, 1.0)
            lines.append(f"  Edge multiplier: {mult_e:.2f}x "
                         f"| Size multiplier: {mult_s:.2f}x")

        if self.category_stats:
            lines.append("\n[Categories]")
            for cat, cs in sorted(self.category_stats.items()):
                flag = " BLACKLISTED" if cs.blacklisted else ""
                lines.append(f"  {cat}: {cs.total_trades} trades, "
                             f"{cs.win_rate:.0%} win rate, "
                             f"${cs.total_pnl:+.4f}{flag}")

        return "\n".join(lines)

    # ── Internal computation ─────────────────────────────────────

    def _recompute_stats(self) -> None:
        """Recompute all aggregate statistics from trade records."""
        # Reset
        self.strategy_stats = defaultdict(StrategyStats)
        self.category_stats = defaultdict(CategoryStats)
        self.price_range_stats = defaultdict(lambda: defaultdict(PriceRangeStats))

        for i, trade in enumerate(self.trades):
            weight = math.exp(DECAY_RATE * (i - len(self.trades)))  # recent = higher

            # Strategy stats
            ss = self.strategy_stats[trade.strategy]
            ss.total_trades += 1
            if trade.won:
                ss.wins += 1
            else:
                ss.losses += 1
            ss.total_pnl += trade.pnl
            ss.weighted_win_rate += weight * (1.0 if trade.won else 0.0)
            ss.weighted_avg_pnl += weight * trade.pnl

            # Category stats
            if trade.category:
                cs = self.category_stats[trade.category]
                cs.total_trades += 1
                if trade.won:
                    cs.wins += 1
                cs.total_pnl += trade.pnl

            # Price range stats
            pr = self.price_range_stats[trade.strategy][trade.price_range]
            pr.total_trades += 1
            if trade.won:
                pr.wins += 1

        # Normalize
        for name, ss in self.strategy_stats.items():
            if ss.total_trades > 0:
                ss.win_rate = ss.wins / ss.total_trades
                ss.avg_edge_predicted = sum(
                    t.edge_predicted for t in self.trades if t.strategy == name
                ) / ss.total_trades
                ss.avg_edge_actual = sum(
                    t.edge_actual for t in self.trades if t.strategy == name
                ) / ss.total_trades

                # Normalize weighted stats
                total_weight = sum(
                    math.exp(DECAY_RATE * (i - len(self.trades)))
                    for i, t in enumerate(self.trades) if t.strategy == name
                )
                if total_weight > 0:
                    ss.weighted_win_rate /= total_weight
                    ss.weighted_avg_pnl /= total_weight

        for cat, cs in self.category_stats.items():
            if cs.total_trades > 0:
                cs.win_rate = cs.wins / cs.total_trades
                cs.blacklisted = (cs.win_rate < BLACKLIST_WIN_RATE
                                  and cs.total_trades >= MIN_TRADES_FOR_CONFIDENCE)

        for strat, ranges in self.price_range_stats.items():
            for pr_name, pr in ranges.items():
                if pr.total_trades > 0:
                    pr.win_rate = pr.wins / pr.total_trades
                    relevant = [t for t in self.trades
                                if t.strategy == strat and t.price_range == pr_name]
                    pr.avg_pnl = sum(t.pnl for t in relevant) / len(relevant)

    def _adapt_parameters(self) -> None:
        """Adjust strategy parameters based on learned performance."""
        for name, stats in self.strategy_stats.items():
            if stats.total_trades < MIN_TRADES_FOR_CONFIDENCE:
                continue

            # Edge multiplier: if we're consistently overestimating edge,
            # require more edge before trading
            if stats.avg_edge_predicted > 0 and stats.avg_edge_actual != 0:
                calibration = stats.avg_edge_actual / stats.avg_edge_predicted
                # Clamp between 0.5x and 2.0x
                # If calibration < 1: we overestimate → need higher edge → multiplier > 1
                # If calibration > 1: we underestimate → can accept lower edge → multiplier < 1
                raw_mult = 1.0 / max(calibration, 0.01)
                self.edge_multipliers[name] = max(0.5, min(2.0, raw_mult))
            else:
                self.edge_multipliers[name] = 1.0

            # Size multiplier: scale bet size by win rate confidence
            # High win rate → bet up to 1.5x
            # Low win rate → shrink to 0.3x
            if stats.weighted_win_rate >= 0.7:
                self.size_multipliers[name] = min(1.5, 0.5 + stats.weighted_win_rate)
            elif stats.weighted_win_rate >= 0.5:
                self.size_multipliers[name] = 1.0
            elif stats.weighted_win_rate >= 0.3:
                self.size_multipliers[name] = 0.6
            else:
                self.size_multipliers[name] = 0.3

            logger.info("[learner] %s: win_rate=%.0f%% edge_mult=%.2f size_mult=%.2f",
                        name, stats.weighted_win_rate * 100,
                        self.edge_multipliers[name], self.size_multipliers[name])

    @staticmethod
    def _price_bucket(price: float) -> str:
        """Bucket a price into a range for analysis."""
        if price <= 0.20:
            return "0-20"
        elif price <= 0.40:
            return "20-40"
        elif price <= 0.60:
            return "40-60"
        elif price <= 0.80:
            return "60-80"
        else:
            return "80-100"

    # ── Persistence ──────────────────────────────────────────────

    def save(self) -> None:
        """Persist learned data to JSON."""
        data = {
            "trades": [asdict(t) for t in self.trades],
            "edge_multipliers": self.edge_multipliers,
            "size_multipliers": self.size_multipliers,
        }
        try:
            with open(self.learn_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save learned data: %s", e)

    def load(self) -> None:
        """Load learned data from JSON."""
        if not os.path.exists(self.learn_path):
            return
        try:
            with open(self.learn_path) as f:
                data = json.load(f)
            self.trades = [TradeRecord(**t) for t in data.get("trades", [])]
            self.edge_multipliers = data.get("edge_multipliers", {})
            self.size_multipliers = data.get("size_multipliers", {})
            self._recompute_stats()
            logger.info("Loaded %d trade records from learned data", len(self.trades))
        except Exception as e:
            logger.error("Failed to load learned data: %s", e)
