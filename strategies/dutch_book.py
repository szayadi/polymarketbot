"""Strategy 1: Dutch Book Arbitrage (GUARANTEED profit).

In multi-outcome events on Kalshi, all YES prices must sum to ~$1.00.
When they don't, we can buy all outcomes and guarantee a profit.

Now filtered to only trade events that resolve SOON (within max_days_to_resolve)
and have real liquidity, so capital isn't locked up for years.
"""

import logging
from datetime import datetime, timezone, timedelta
from client import KalshiClient
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

MIN_ARB_PROFIT_PCT = 0.02


class DutchBookStrategy(BaseStrategy):
    name = "dutch_book"

    def scan(self) -> list[dict]:
        """Scan multi-outcome events for Dutch book opportunities."""
        signals = []

        try:
            events = self.client.get_all_events(status="open", max_pages=5)
        except Exception as e:
            logger.error("[dutch_book] Failed to fetch events: %s", e)
            return signals

        logger.info("[dutch_book] Scanning %d events", len(events))

        for event in events:
            try:
                event_signals = self._evaluate_event(event)
                signals.extend(event_signals)
            except Exception as e:
                logger.debug("[dutch_book] Error on event: %s", e)

        signals.sort(key=lambda s: s["edge"], reverse=True)
        logger.info("[dutch_book] Found %d arb signals", len(signals))
        return signals

    def _is_fast_resolving(self, markets: list[dict]) -> bool:
        """Check if at least one market resolves within our time window."""
        cutoff = datetime.now(timezone.utc) + timedelta(days=self.cfg.max_days_to_resolve)
        for m in markets:
            # Check expected_expiration_time, close_time, latest_expiration_time
            for field in ("expected_expiration_time", "close_time", "latest_expiration_time"):
                ts = m.get(field)
                if ts:
                    try:
                        if isinstance(ts, str):
                            # Handle ISO format
                            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        else:
                            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                        if dt <= cutoff:
                            return True
                    except (ValueError, TypeError, OSError):
                        continue
        return False

    def _has_volume(self, markets: list[dict]) -> bool:
        """Check if the event has meaningful trading volume."""
        total_vol = 0
        for m in markets:
            vol24 = int(m.get("volume_24h", 0) or 0)
            vol = int(m.get("volume", 0) or 0)
            total_vol += max(vol24, vol)
        return total_vol >= self.cfg.min_volume_24h

    def _evaluate_event(self, event: dict) -> list[dict]:
        """Check if an event's markets have a Dutch book opportunity."""
        markets = event.get("markets", [])
        if not markets or len(markets) < 2:
            return []

        if not event.get("mutually_exclusive", False):
            return []

        # FILTER: Only fast-resolving events
        if not self._is_fast_resolving(markets):
            return []

        # FILTER: Must have real volume
        if not self._has_volume(markets):
            return []

        event_ticker = event.get("event_ticker", "")
        event_title = event.get("title", "Unknown")
        category = event.get("category", "")
        series_ticker = event.get("series_ticker", "")

        # Collect YES ask prices for each outcome market
        market_data = []
        total_yes_ask = 0

        for m in markets:
            ticker = m.get("ticker", "")
            status = m.get("status", "")
            if status != "open" and status != "active":
                continue

            yes_ask = m.get("yes_ask")
            if yes_ask is None:
                bid, ask = self.client.get_best_bid_ask(ticker)
                if ask is None:
                    return []
                yes_ask = ask
            else:
                yes_ask = int(yes_ask)

            if yes_ask <= 0 or yes_ask >= 100:
                return []

            market_data.append({
                "ticker": ticker,
                "yes_ask": yes_ask,
                "question": m.get("yes_sub_title", m.get("title", "")),
            })
            total_yes_ask += yes_ask

        if len(market_data) < 2:
            return []

        gross_profit_cents = 100 - total_yes_ask

        total_fee_cents = sum(
            self.client.calc_taker_fee(1, md["yes_ask"])
            for md in market_data
        )

        net_profit_cents = gross_profit_cents - total_fee_cents
        net_profit_pct = net_profit_cents / total_yes_ask if total_yes_ask > 0 else 0

        if net_profit_pct < max(MIN_ARB_PROFIT_PCT, self.get_min_edge()):
            return []

        signals = []

        max_cost_per_set = total_yes_ask
        max_bet_cents = int(self.cfg.max_bet_size * 100)
        sets = max(1, max_bet_cents // max_cost_per_set)

        existing = self.tracker.get_event_exposure_cents(event_ticker)
        if existing > 0:
            return []

        for md in market_data:
            signals.append({
                "ticker": md["ticker"],
                "event_ticker": event_ticker,
                "side": "yes",
                "action": "buy",
                "price_cents": md["yes_ask"],
                "count": sets,
                "edge": net_profit_pct,
                "reason": (f"Dutch book: {event_title} | "
                           f"sum={total_yes_ask}c<100c | "
                           f"net profit={net_profit_cents}c/set "
                           f"({net_profit_pct:.1%})"),
                "question": md["question"],
                "category": category,
                "series_ticker": series_ticker,
            })

        return signals
