"""Adaptive learning engine for the Kalshi trading bot.

Tracks every trade outcome and adjusts strategy behavior:
- Scores strategies, categories, and price ranges by win rate
- Adjusts edge requirements based on prediction calibration
- Scales position sizes by confidence (aggressive when hot, conservative when cold)
- Blacklists consistently losing categories/price ranges
- Tracks streak data to detect hot/cold runs
- Faster decay to adapt quickly to changing market conditions
"""

import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

LEARN_FILE = os.path.join(os.path.dirname(__file__), "learned.json")
MIN_TRADES_FOR_CONFIDENCE = 3  # Reduced from 5 — learn faster
DECAY_RATE = 0.10  # Doubled from 0.05 — weight recent trades much more
BLACKLIST_WIN_RATE = 0.25  # Slightly more forgiving — give strategies more runway
HOT_STREAK_THRESHOLD = 3  # Consecutive wins to boost sizing
COLD_STREAK_THRESHOLD = 2  # Consecutive losses to reduce sizing


@dataclass
class TradeRecord:
    timestamp: str
    strategy: str
    ticker: str
    event_ticker: str
    side: str
    entry_price_cents: int
    exit_price_cents: int
    count: int
    pnl_cents: int
    edge_predicted: float
    edge_actual: float
    category: str
    series_ticker: str
    question: str
    price_bucket: str
    won: bool
    volume_at_entry: int = 0
    hold_time_hours: float = 0.0


@dataclass
class StrategyStats:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl_cents: int = 0
    avg_edge_predicted: float = 0.0
    avg_edge_actual: float = 0.0
    win_rate: float = 0.0
    weighted_win_rate: float = 0.0
    weighted_avg_pnl: float = 0.0
    current_streak: int = 0  # positive = wins, negative = losses
    best_streak: int = 0
    worst_streak: int = 0
    avg_hold_hours: float = 0.0
    roi_per_hour: float = 0.0


class AdaptiveLearner:
    """Learns from outcomes and adjusts strategy parameters."""

    def __init__(self, learn_path: str = LEARN_FILE):
        self.learn_path = learn_path
        self.trades: list[TradeRecord] = []
        self.strategy_stats: dict[str, StrategyStats] = {}
        self.category_blacklist: set[str] = set()
        self.price_bucket_win_rates: dict[str, dict[str, float]] = {}
        self.edge_multipliers: dict[str, float] = {}
        self.size_multipliers: dict[str, float] = {}
        # Track which tickers have been profitable
        self.ticker_scores: dict[str, float] = {}
        self.load()

    def record_trade(self, strategy: str, ticker: str, event_ticker: str,
                     side: str, entry_price_cents: int, exit_price_cents: int,
                     count: int, pnl_cents: int, edge_predicted: float,
                     category: str = "", series_ticker: str = "",
                     question: str = "", volume_at_entry: int = 0,
                     hold_time_hours: float = 0.0) -> None:
        """Record a completed trade for learning."""
        won = pnl_cents > 0
        edge_actual = 0.0
        if entry_price_cents > 0:
            edge_actual = (exit_price_cents - entry_price_cents) / entry_price_cents

        record = TradeRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            strategy=strategy,
            ticker=ticker,
            event_ticker=event_ticker,
            side=side,
            entry_price_cents=entry_price_cents,
            exit_price_cents=exit_price_cents,
            count=count,
            pnl_cents=pnl_cents,
            edge_predicted=edge_predicted,
            edge_actual=edge_actual,
            category=category,
            series_ticker=series_ticker,
            question=question,
            price_bucket=self._price_bucket(entry_price_cents),
            won=won,
            volume_at_entry=volume_at_entry,
            hold_time_hours=hold_time_hours,
        )
        self.trades.append(record)

        result_str = "WIN" if won else "LOSS"
        logger.info("[learner] %s %s %s pnl=%dc (pred=%.1f%% actual=%.1f%%) hold=%.1fh",
                    strategy, result_str, ticker,
                    pnl_cents, edge_predicted * 100, edge_actual * 100,
                    hold_time_hours)

        self._recompute()
        self.save()

    def should_trade(self, strategy: str, category: str = "",
                     price_cents: int = 0) -> tuple[bool, str]:
        """Should we take this trade based on history?"""
        stats = self.strategy_stats.get(strategy)
        if not stats or stats.total_trades < MIN_TRADES_FOR_CONFIDENCE:
            return True, "insufficient data"

        if category and category in self.category_blacklist:
            return False, f"category '{category}' blacklisted"

        bucket = self._price_bucket(price_cents)
        bucket_rates = self.price_bucket_win_rates.get(strategy, {})
        bucket_data = bucket_rates.get(bucket)
        if bucket_data is not None and bucket_data < BLACKLIST_WIN_RATE:
            return False, f"price bucket {bucket} win rate {bucket_data:.0%}"

        if (stats.weighted_win_rate < BLACKLIST_WIN_RATE
                and stats.total_trades >= MIN_TRADES_FOR_CONFIDENCE * 2):
            return False, f"strategy win rate {stats.weighted_win_rate:.0%}"

        return True, "approved"

    def get_edge_multiplier(self, strategy: str) -> float:
        return self.edge_multipliers.get(strategy, 1.0)

    def get_size_multiplier(self, strategy: str) -> float:
        base = self.size_multipliers.get(strategy, 1.0)
        # Streak adjustment — be more aggressive on hot streaks
        stats = self.strategy_stats.get(strategy)
        if stats:
            if stats.current_streak >= HOT_STREAK_THRESHOLD:
                streak_boost = min(0.5, 0.1 * (stats.current_streak - HOT_STREAK_THRESHOLD + 1))
                base = min(2.0, base + streak_boost)
            elif stats.current_streak <= -COLD_STREAK_THRESHOLD:
                streak_cut = min(0.5, 0.15 * abs(stats.current_streak + COLD_STREAK_THRESHOLD - 1))
                base = max(0.2, base - streak_cut)
        return base

    def get_report(self) -> str:
        lines = [f"=== Learner: {len(self.trades)} trades ==="]
        total_pnl = sum(t.pnl_cents for t in self.trades)
        lines.append(f"  Total PnL: {total_pnl}c (${total_pnl/100:.2f})")

        for name, stats in self.strategy_stats.items():
            em = self.edge_multipliers.get(name, 1.0)
            sm = self.get_size_multiplier(name)
            streak_str = f"+{stats.current_streak}" if stats.current_streak > 0 else str(stats.current_streak)
            lines.append(
                f"  [{name}] {stats.total_trades} trades | "
                f"W:{stats.wins} L:{stats.losses} | "
                f"WR:{stats.win_rate:.0%} (wt:{stats.weighted_win_rate:.0%}) | "
                f"PnL:{stats.total_pnl_cents}c | streak:{streak_str} | "
                f"edge_mult:{em:.2f} size_mult:{sm:.2f}"
            )
            if stats.avg_hold_hours > 0:
                lines.append(
                    f"    avg_hold:{stats.avg_hold_hours:.1f}h | "
                    f"roi/hr:{stats.roi_per_hour:.2f}c"
                )
        if self.category_blacklist:
            lines.append(f"  Blacklisted categories: {self.category_blacklist}")
        return "\n".join(lines)

    def _recompute(self) -> None:
        self.strategy_stats = {}
        cat_stats: dict[str, dict] = defaultdict(lambda: {"wins": 0, "total": 0})
        bucket_stats: dict[str, dict[str, dict]] = defaultdict(
            lambda: defaultdict(lambda: {"wins": 0, "total": 0})
        )
        # Track streaks per strategy
        streaks: dict[str, int] = {}

        for i, t in enumerate(self.trades):
            weight = math.exp(DECAY_RATE * (i - len(self.trades)))

            if t.strategy not in self.strategy_stats:
                self.strategy_stats[t.strategy] = StrategyStats()
            ss = self.strategy_stats[t.strategy]
            ss.total_trades += 1
            if t.won:
                ss.wins += 1
            else:
                ss.losses += 1
            ss.total_pnl_cents += t.pnl_cents
            ss.weighted_win_rate += weight * (1.0 if t.won else 0.0)
            ss.weighted_avg_pnl += weight * t.pnl_cents

            # Streak tracking
            if t.strategy not in streaks:
                streaks[t.strategy] = 0
            if t.won:
                streaks[t.strategy] = max(1, streaks[t.strategy] + 1)
            else:
                streaks[t.strategy] = min(-1, streaks[t.strategy] - 1)
            ss.current_streak = streaks[t.strategy]
            ss.best_streak = max(ss.best_streak, streaks[t.strategy])
            ss.worst_streak = min(ss.worst_streak, streaks[t.strategy])

            if t.category:
                cat_stats[t.category]["total"] += 1
                if t.won:
                    cat_stats[t.category]["wins"] += 1

            bucket_stats[t.strategy][t.price_bucket]["total"] += 1
            if t.won:
                bucket_stats[t.strategy][t.price_bucket]["wins"] += 1

            # Ticker scoring
            if t.ticker not in self.ticker_scores:
                self.ticker_scores[t.ticker] = 0.0
            self.ticker_scores[t.ticker] += t.pnl_cents * weight

        # Normalize
        for name, ss in self.strategy_stats.items():
            if ss.total_trades > 0:
                ss.win_rate = ss.wins / ss.total_trades
                strat_trades = [t for t in self.trades if t.strategy == name]
                ss.avg_edge_predicted = sum(t.edge_predicted for t in strat_trades) / len(strat_trades)
                ss.avg_edge_actual = sum(t.edge_actual for t in strat_trades) / len(strat_trades)
                total_w = sum(math.exp(DECAY_RATE * (i - len(self.trades)))
                              for i, t in enumerate(self.trades) if t.strategy == name)
                if total_w > 0:
                    ss.weighted_win_rate /= total_w
                    ss.weighted_avg_pnl /= total_w

                # Hold time and ROI per hour
                hold_trades = [t for t in strat_trades if t.hold_time_hours > 0]
                if hold_trades:
                    ss.avg_hold_hours = sum(t.hold_time_hours for t in hold_trades) / len(hold_trades)
                    if ss.avg_hold_hours > 0:
                        avg_pnl = ss.total_pnl_cents / ss.total_trades
                        ss.roi_per_hour = avg_pnl / ss.avg_hold_hours

        # Category blacklist
        self.category_blacklist = set()
        for cat, data in cat_stats.items():
            if data["total"] >= MIN_TRADES_FOR_CONFIDENCE:
                wr = data["wins"] / data["total"]
                if wr < BLACKLIST_WIN_RATE:
                    self.category_blacklist.add(cat)

        # Price bucket win rates
        self.price_bucket_win_rates = {}
        for strat, buckets in bucket_stats.items():
            self.price_bucket_win_rates[strat] = {}
            for bucket, data in buckets.items():
                if data["total"] >= MIN_TRADES_FOR_CONFIDENCE:
                    self.price_bucket_win_rates[strat][bucket] = (
                        data["wins"] / data["total"]
                    )

        # Adapt multipliers — more aggressive scaling
        for name, ss in self.strategy_stats.items():
            if ss.total_trades < MIN_TRADES_FOR_CONFIDENCE:
                continue
            # Edge multiplier — calibrate predicted vs actual edge
            if ss.avg_edge_predicted > 0 and ss.avg_edge_actual != 0:
                cal = ss.avg_edge_actual / ss.avg_edge_predicted
                self.edge_multipliers[name] = max(0.3, min(3.0, 1.0 / max(cal, 0.01)))
            else:
                self.edge_multipliers[name] = 1.0
            # Size multiplier — scale with win rate, wider range
            wr = ss.weighted_win_rate
            if wr >= 0.8:
                self.size_multipliers[name] = 2.0
            elif wr >= 0.7:
                self.size_multipliers[name] = min(1.8, 0.5 + wr)
            elif wr >= 0.5:
                self.size_multipliers[name] = 1.0
            elif wr >= 0.3:
                self.size_multipliers[name] = 0.5
            else:
                self.size_multipliers[name] = 0.2

    @staticmethod
    def _price_bucket(price_cents: int) -> str:
        if price_cents <= 20:
            return "0-20"
        elif price_cents <= 40:
            return "20-40"
        elif price_cents <= 60:
            return "40-60"
        elif price_cents <= 80:
            return "60-80"
        else:
            return "80-100"

    def save(self) -> None:
        data = {
            "trades": [asdict(t) for t in self.trades],
            "edge_multipliers": self.edge_multipliers,
            "size_multipliers": self.size_multipliers,
            "ticker_scores": self.ticker_scores,
        }
        try:
            with open(self.learn_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save learned data: %s", e)

    def load(self) -> None:
        if not os.path.exists(self.learn_path):
            return
        try:
            with open(self.learn_path) as f:
                data = json.load(f)
            trades_raw = data.get("trades", [])
            self.trades = []
            for t in trades_raw:
                # Handle old records missing new fields
                t.setdefault("volume_at_entry", 0)
                t.setdefault("hold_time_hours", 0.0)
                self.trades.append(TradeRecord(**t))
            self.edge_multipliers = data.get("edge_multipliers", {})
            self.size_multipliers = data.get("size_multipliers", {})
            self.ticker_scores = data.get("ticker_scores", {})
            self._recompute()
            logger.info("Loaded %d trade records", len(self.trades))
        except Exception as e:
            logger.error("Failed to load learned data: %s", e)
