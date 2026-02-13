"""Strategy 3: Aggressive Spread Capture / Market Making.

Places limit orders on both sides of the bid-ask spread.
When both fill, the spread becomes profit.

AGGRESSIVE MODE:
- Tighter minimum spread (3c vs 4c)
- Wider price range (15-85 vs 20-80)
- Fast stale cancellation (60s vs 5min)
- Lower volume requirements
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

# ── AGGRESSIVE thresholds ──
MIN_SPREAD_CENTS = 3              # Trade 3c+ spreads (was 4)
MIN_VOLUME = 200                  # Lower volume bar (was 1000)
STALE_ORDER_SECONDS = 60          # Cancel unfilled after 60s (was 300)
MIN_NET_SPREAD = 1                # Capture even 1c net (was 2)
PRICE_RANGE_LOW = 15              # Accept prices from 15c (was 20)
PRICE_RANGE_HIGH = 85             # Accept prices up to 85c (was 80)


class SpreadStrategy(BaseStrategy):
    name = "spread"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending: dict[str, float] = {}

    def scan(self) -> list[dict]:
        """Find markets with wide bid-ask spreads."""
        signals = []

        try:
            markets = self.client.get_all_markets(status="open", max_pages=8)
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

    def _resolves_soon(self, market: dict) -> bool:
        """Check if market resolves within our time window."""
        cutoff = datetime.now(timezone.utc) + timedelta(days=self.cfg.max_days_to_resolve)
        for field in ("expected_expiration_time", "close_time", "latest_expiration_time",
                      "expiration_time", "end_date_time", "settlement_timer_expiration_time"):
            ts = market.get(field)
            if ts:
                try:
                    if isinstance(ts, str):
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    else:
                        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    if dt <= cutoff:
                        return True
                except (ValueError, TypeError, OSError):
                    continue
        return False

    def _evaluate_market(self, market: dict) -> list[dict]:
        """Check if a market has a tradeable spread."""
        ticker = market.get("ticker", "")
        status = market.get("status", "")
        if status not in ("open", "active"):
            return []

        # Time filter
        if not self._resolves_soon(market):
            return []

        # Check volume — lowered
        volume = int(market.get("volume", 0) or 0)
        vol24 = int(market.get("volume_24h", 0) or 0)
        if volume < MIN_VOLUME and vol24 < self.cfg.min_volume_24h:
            return []

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

        # Wider acceptable price range
        if mid < PRICE_RANGE_LOW or mid > PRICE_RANGE_HIGH:
            return []

        our_bid = yes_bid + 1
        our_ask = yes_ask - 1
        our_spread = our_ask - our_bid

        if our_spread < MIN_NET_SPREAD:
            return []

        net_profit_pct = our_spread / mid / 100.0

        if net_profit_pct < self.get_min_edge():
            return []

        event_ticker = market.get("event_ticker", "")
        category = market.get("category", "")
        series_ticker = market.get("series_ticker", "")
        question = market.get("yes_sub_title", market.get("title", ticker))

        max_bet_cents = int(self.cfg.max_bet_size * 100)
        count_per_side = max(1, (max_bet_cents // 2) // our_bid)

        signals = []

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
        """Cancel stale unfilled spread orders — fast timeout."""
        now = time.time()
        stale = [t for t, placed in self._pending.items()
                 if now - placed > STALE_ORDER_SECONDS]
        for ticker in stale:
            self._pending.pop(ticker, None)
            logger.info("[spread] Stale order timeout (60s): %s", ticker)

    def execute(self, signals: list[dict]) -> list[dict]:
        results = super().execute(signals)
        for r in results:
            self._pending[r["signal"]["ticker"]] = time.time()
        return results
