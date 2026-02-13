"""Strategy 3: Sports & Politics Spread Trading.

These markets are flooded with retail money and delayed reactions.
The bot places limit orders on both sides of wide spreads to capture
the bid-ask gap. Also detects stale prices relative to order book depth.
"""

import logging
import time
from datetime import datetime, timezone

from strategies.base import BaseStrategy
from positions import Position

logger = logging.getLogger(__name__)

# Configuration
MIN_SPREAD_PCT = 0.03         # Minimum 3% spread to be worth trading
MIN_VOLUME_24H = 5000.0       # Minimum 24h volume
MIN_LIQUIDITY = 1000.0        # Minimum market liquidity
CATEGORIES = ["sports", "politics"]
ORDER_STALE_SECONDS = 300     # Cancel unfilled orders after 5 minutes


class SpreadStrategy(BaseStrategy):
    name = "spread"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track our pending spread orders: {market_id: {"bid_time": t, "ask_time": t}}
        self._pending_orders: dict[str, dict] = {}

    def scan(self) -> list[dict]:
        """Find sports/politics markets with wide bid-ask spreads."""
        signals = []

        for category in CATEGORIES:
            try:
                markets = self.client.get_all_markets(
                    active=True, category=category,
                    min_liquidity=MIN_LIQUIDITY, max_pages=2,
                )
            except Exception as e:
                logger.error("[spread] Failed to fetch %s markets: %s",
                             category, e)
                continue

            logger.info("[spread] Scanning %d %s markets for spread opps",
                        len(markets), category)

            for market in markets:
                try:
                    market_signals = self._evaluate_market(market)
                    signals.extend(market_signals)
                except Exception as e:
                    logger.debug("[spread] Error evaluating market: %s", e)

        signals.sort(key=lambda s: s["edge"], reverse=True)
        logger.info("[spread] Found %d signals", len(signals))
        return signals

    def _evaluate_market(self, market: dict) -> list[dict]:
        """Check if a market has a tradeable spread."""
        signals = []

        # Check volume
        volume_24h = float(market.get("volume24hr", 0) or 0)
        if volume_24h < MIN_VOLUME_24H:
            return signals

        # Need token IDs
        token_ids = market.get("clobTokenIds") or market.get("clob_token_ids", [])
        if not token_ids:
            return signals

        condition_id = market.get("condition_id", market.get("id", ""))
        question = market.get("question", market.get("title", "Unknown"))

        # Check if already have exposure in this market
        existing = self.tracker.get_market_exposure(condition_id)
        if existing > 0:
            return signals

        # Check YES token spread
        yes_token = token_ids[0]
        best_bid, best_ask = self.client.get_best_bid_ask(yes_token)
        if best_bid is None or best_ask is None:
            return signals

        spread = best_ask - best_bid
        mid = (best_ask + best_bid) / 2.0

        # Avoid extremes — spread trading works best in the 0.20-0.80 range
        if mid < 0.15 or mid > 0.85:
            return signals

        spread_pct = spread / mid if mid > 0 else 0

        if spread_pct < MIN_SPREAD_PCT:
            return signals

        # Get tick size for proper pricing
        tick_size_str = self.client.get_tick_size(yes_token)
        tick = float(tick_size_str)

        # Our bid: slightly above current best bid
        our_bid = round(best_bid + tick, len(tick_size_str.split(".")[-1]))
        # Our ask: slightly below current best ask
        our_ask = round(best_ask - tick, len(tick_size_str.split(".")[-1]))

        # Make sure our spread is still profitable after fees
        our_spread = our_ask - our_bid
        fee_estimate = 0.02
        net_profit = our_spread - fee_estimate

        if net_profit < self.get_min_edge():
            return signals

        # Size: half the max bet per side (we need capital for both legs)
        half_size = self.cfg.max_bet_size / 2.0

        category = market.get("category", "")

        # Generate BUY signal (bid side)
        signals.append({
            "token_id": yes_token,
            "side": "BUY",
            "price": our_bid,
            "size": half_size,
            "market_id": condition_id,
            "edge": net_profit,
            "reason": f"Spread BUY: bid@{our_bid:.4f} "
                      f"(spread={spread:.4f}, {spread_pct:.1%})",
            "question": question,
            "category": category,
            "spread": spread,
            "liquidity": float(market.get("liquidity", 0) or 0),
        })

        # Generate SELL signal (ask side)
        signals.append({
            "token_id": yes_token,
            "side": "SELL",
            "price": our_ask,
            "size": half_size,
            "market_id": condition_id,
            "edge": net_profit,
            "reason": f"Spread SELL: ask@{our_ask:.4f} "
                      f"(spread={spread:.4f}, {spread_pct:.1%})",
            "question": question,
            "category": category,
            "spread": spread,
            "liquidity": float(market.get("liquidity", 0) or 0),
        })

        return signals

    def cleanup_stale_orders(self) -> None:
        """Cancel orders that have been sitting too long without filling.

        Called from the main loop to free up capital tied in stale spread orders.
        """
        now = time.time()
        stale_markets = []

        for market_id, times in self._pending_orders.items():
            for side, placed_time in times.items():
                if now - placed_time > ORDER_STALE_SECONDS:
                    stale_markets.append(market_id)
                    break

        if stale_markets:
            logger.info("[spread] Cancelling stale orders in %d markets",
                        len(stale_markets))
            # In a real implementation, we'd cancel specific orders
            # For now, track that these positions should be closed
            for mid in stale_markets:
                self._pending_orders.pop(mid, None)

    def execute(self, signals: list[dict]) -> list[dict]:
        """Override to track pending spread orders."""
        results = super().execute(signals)

        # Track when we placed spread orders
        for r in results:
            sig = r["signal"]
            mid = sig["market_id"]
            if mid not in self._pending_orders:
                self._pending_orders[mid] = {}
            self._pending_orders[mid][sig["side"]] = time.time()

        return results
