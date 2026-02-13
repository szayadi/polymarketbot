"""Strategy 3: Spread Capture / Market Making.

Places limit orders on both sides of the bid-ask spread.
When both fill, the spread becomes profit.

On Kalshi, maker orders earn rebates rather than paying fees,
making this strategy fee-efficient.
"""

import logging
import time
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

MIN_SPREAD_CENTS = 4          # Minimum 4c spread to be worth it
MIN_VOLUME = 1000             # Minimum market volume
STALE_ORDER_SECONDS = 300     # Cancel after 5 minutes


class SpreadStrategy(BaseStrategy):
    name = "spread"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending: dict[str, float] = {}  # ticker -> placed_time

    def scan(self) -> list[dict]:
        """Find markets with wide bid-ask spreads."""
        signals = []

        try:
            markets = self.client.get_all_markets(status="open", max_pages=3)
        except Exception as e:
            logger.error("[spread] Failed to fetch markets: %s", e)
            return signals

        logger.info("[spread] Scanning %d markets for spread opps", len(markets))

        for market in markets:
            try:
                sigs = self._evaluate_market(market)
                signals.extend(sigs)
            except Exception as e:
                logger.debug("[spread] Error: %s", e)

        signals.sort(key=lambda s: s["edge"], reverse=True)
        logger.info("[spread] Found %d signals", len(signals))
        return signals

    def _evaluate_market(self, market: dict) -> list[dict]:
        """Check if a market has a tradeable spread."""
        ticker = market.get("ticker", "")
        status = market.get("status", "")
        if status not in ("open", "active"):
            return []

        # Check volume
        volume = int(market.get("volume", 0) or 0)
        if volume < MIN_VOLUME:
            return []

        # Already in this market?
        if self.tracker.has_position(ticker):
            return []
        if ticker in self._pending:
            return []

        # Get orderbook for accurate bid/ask
        yes_bid, yes_ask = self.client.get_best_bid_ask(ticker)
        if yes_bid is None or yes_ask is None:
            return []

        spread = yes_ask - yes_bid
        if spread < MIN_SPREAD_CENTS:
            return []

        mid = (yes_ask + yes_bid) / 2.0

        # Avoid extremes — spread trading works in the 20-80 range
        if mid < 20 or mid > 80:
            return []

        # Our orders: improve the bid/ask by 1 cent
        our_bid = yes_bid + 1
        our_ask = yes_ask - 1
        our_spread = our_ask - our_bid

        if our_spread < 2:
            return []

        # As maker orders, fees are minimal/rebated
        # Net profit per round trip ≈ our_spread cents
        net_profit_pct = our_spread / mid / 100.0

        if net_profit_pct < self.get_min_edge():
            return []

        event_ticker = market.get("event_ticker", "")
        category = market.get("category", "")
        series_ticker = market.get("series_ticker", "")
        question = market.get("yes_sub_title", market.get("title", ticker))

        # Half the max bet per side
        max_bet_cents = int(self.cfg.max_bet_size * 100)
        count_per_side = max(1, (max_bet_cents // 2) // our_bid)

        signals = []

        # BUY YES at our_bid (improving best bid)
        signals.append({
            "ticker": ticker,
            "event_ticker": event_ticker,
            "side": "yes",
            "action": "buy",
            "price_cents": our_bid,
            "count": count_per_side,
            "edge": net_profit_pct,
            "reason": (f"Spread BUY: YES@{our_bid}c "
                       f"(spread={spread}c, net={our_spread}c)"),
            "question": question,
            "category": category,
            "series_ticker": series_ticker,
        })

        # SELL YES at our_ask (improving best ask)
        # On Kalshi: selling YES = buying NO at (100 - our_ask)
        signals.append({
            "ticker": ticker,
            "event_ticker": event_ticker,
            "side": "no",
            "action": "buy",
            "price_cents": 100 - our_ask,
            "count": count_per_side,
            "edge": net_profit_pct,
            "reason": (f"Spread SELL: NO@{100 - our_ask}c "
                       f"(spread={spread}c, net={our_spread}c)"),
            "question": question,
            "category": category,
            "series_ticker": series_ticker,
        })

        return signals

    def cleanup_stale_orders(self) -> None:
        """Cancel stale unfilled spread orders."""
        now = time.time()
        stale = [t for t, placed in self._pending.items()
                 if now - placed > STALE_ORDER_SECONDS]
        for ticker in stale:
            self._pending.pop(ticker, None)
            logger.info("[spread] Stale order timeout: %s", ticker)

    def execute(self, signals: list[dict]) -> list[dict]:
        results = super().execute(signals)
        for r in results:
            self._pending[r["signal"]["ticker"]] = time.time()
        return results
