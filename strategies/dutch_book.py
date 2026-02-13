"""Strategy 1: Dutch Book Arbitrage (GUARANTEED profit).

In multi-outcome events on Kalshi, all YES prices must sum to ~$1.00.
When they don't, we can buy all outcomes and guarantee a profit.

This is the bot's primary money-making strategy because it's
mathematically risk-free (ignoring execution risk).

Examples:
  - "GDP 0-2%" YES=25c + "GDP 2-4%" YES=30c + "GDP 4%+" YES=35c = 90c
  - Buy all three for 90c total, one MUST resolve YES → receive 100c
  - Guaranteed profit: 10c per set (minus fees)
"""

import logging
from client import KalshiClient
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

# Only trade if profit after fees > this percentage
MIN_ARB_PROFIT_PCT = 0.02


class DutchBookStrategy(BaseStrategy):
    name = "dutch_book"

    def scan(self) -> list[dict]:
        """Scan multi-outcome events for Dutch book opportunities."""
        signals = []

        try:
            events = self.client.get_all_events(status="open", max_pages=3)
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

    def _evaluate_event(self, event: dict) -> list[dict]:
        """Check if an event's markets have a Dutch book opportunity."""
        markets = event.get("markets", [])
        if not markets or len(markets) < 2:
            return []

        # Only look at mutually exclusive events (one outcome must be YES)
        if not event.get("mutually_exclusive", False):
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

            # Get the YES ask price (what we'd pay to buy YES)
            yes_ask = m.get("yes_ask")
            if yes_ask is None:
                # Try to get from orderbook
                bid, ask = self.client.get_best_bid_ask(ticker)
                if ask is None:
                    return []  # Can't price all outcomes → skip event
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

        # Dutch book check: if total cost < 100 cents → guaranteed profit
        # Payout is always 100c (one outcome resolves YES)
        gross_profit_cents = 100 - total_yes_ask

        # Estimate total fees for buying all outcomes
        total_fee_cents = sum(
            self.client.calc_taker_fee(1, md["yes_ask"])
            for md in market_data
        )

        net_profit_cents = gross_profit_cents - total_fee_cents
        net_profit_pct = net_profit_cents / total_yes_ask if total_yes_ask > 0 else 0

        if net_profit_pct < max(MIN_ARB_PROFIT_PCT, self.get_min_edge()):
            return []

        # We have an arb! Generate buy signals for every outcome
        signals = []

        # How many sets can we afford?
        max_cost_per_set = total_yes_ask  # cents per complete set
        max_bet_cents = int(self.cfg.max_bet_size * 100)
        sets = max(1, max_bet_cents // max_cost_per_set)

        # Already have exposure in this event?
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
