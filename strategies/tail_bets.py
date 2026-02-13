"""Strategy 2: Aggressive High-Probability Bets.

Buy contracts that are very likely to resolve in our favor:
- YES contracts priced 85-99c (high-probability YES outcome)
- NO contracts priced 1-15c (buy YES's complement cheaply)

WIDENED from the old 95/5 thresholds to 85/15 — catches 5-10x more
opportunities while still trading high-probability outcomes.

Urgency scoring: markets resolving sooner are prioritized.
"""

import logging
from datetime import datetime, timezone, timedelta
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

# ── AGGRESSIVE thresholds ──
MAX_YES_PRICE_FOR_NO_BET = 15     # Buy NO when YES ≤15c (was 5c)
MIN_YES_PRICE_FOR_YES_BET = 85    # Buy YES when bid ≥85c (was 95c)
MIN_LIQUIDITY_CENTS = 10000       # $100 min liquidity (was $500)
URGENCY_BOOST_HOURS = 24          # Markets within 24h get edge boost


class TailBetsStrategy(BaseStrategy):
    name = "tail_bets"
    use_research = True  # Validate with weather/crypto/finance data before betting

    def scan(self) -> list[dict]:
        """Find markets with near-certain outcomes."""
        signals = []

        try:
            markets = self.client.get_all_markets(status="open", max_pages=8)
        except Exception as e:
            logger.error("[tail_bets] Failed to fetch markets: %s", e)
            return signals

        logger.info("[tail_bets] Scanning %d markets for tail bets", len(markets))

        for market in markets:
            try:
                sig = self._evaluate_market(market)
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.debug("[tail_bets] Error: %s", e)

        signals.sort(key=lambda s: s["edge"], reverse=True)
        logger.info("[tail_bets] Found %d signals", len(signals))
        return signals

    def _resolves_soon(self, market: dict) -> bool:
        """Check if market resolves within our time window.

        PERMISSIVE: if no time field is found, accept the market.
        Only reject if we can confirm it resolves too far in the future.
        """
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
                    else:
                        return False
                except (ValueError, TypeError, OSError):
                    continue
        return True

    def _hours_to_resolve(self, market: dict) -> float:
        """Estimate hours until market resolves."""
        now = datetime.now(timezone.utc)
        for field in ("expected_expiration_time", "close_time", "latest_expiration_time",
                      "expiration_time", "end_date_time", "settlement_timer_expiration_time"):
            ts = market.get(field)
            if ts:
                try:
                    if isinstance(ts, str):
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    else:
                        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    hours = (dt - now).total_seconds() / 3600.0
                    if hours > 0:
                        return hours
                except (ValueError, TypeError, OSError):
                    continue
        return 9999.0

    def _evaluate_market(self, market: dict) -> dict | None:
        """Check if a market qualifies for a tail bet."""
        ticker = market.get("ticker", "")
        status = market.get("status", "")
        if status not in ("open", "active"):
            return None

        # Time filter
        if not self._resolves_soon(market):
            return None

        # Get prices — Kalshi listing may not include bid/ask
        yes_bid = market.get("yes_bid")
        yes_ask = market.get("yes_ask")
        if yes_bid is not None and yes_ask is not None:
            yes_bid = int(yes_bid)
            yes_ask = int(yes_ask)
        else:
            yes_bid, yes_ask = self.client.get_best_bid_ask(ticker)
            if yes_bid is None or yes_ask is None:
                return None

        if yes_bid <= 0 or yes_ask <= 0 or yes_ask >= 100:
            return None

        # Check liquidity — lowered threshold
        liquidity = int(market.get("liquidity", 0) or 0)
        volume = int(market.get("volume", 0) or 0)
        vol24 = int(market.get("volume_24h", 0) or 0)
        if liquidity < MIN_LIQUIDITY_CENTS and volume < MIN_LIQUIDITY_CENTS and vol24 < self.cfg.min_volume_24h:
            return None

        if self.tracker.has_position(ticker):
            return None

        event_ticker = market.get("event_ticker", "")
        category = market.get("category", "")
        series_ticker = market.get("series_ticker", "")
        question = market.get("yes_sub_title", market.get("title", ticker))
        hours_left = self._hours_to_resolve(market)

        # Case 1: YES is cheap → buy NO (high probability NO wins)
        if yes_ask <= MAX_YES_PRICE_FOR_NO_BET:
            no_price = 100 - yes_bid
            if no_price >= 100 or no_price <= 0:
                return None

            fee = self.client.calc_taker_fee(1, no_price)
            edge = (100 - no_price - fee) / no_price

            # Urgency boost for fast-resolving markets
            if hours_left <= URGENCY_BOOST_HOURS:
                urgency_mult = 1.0 + (URGENCY_BOOST_HOURS - hours_left) / URGENCY_BOOST_HOURS
                edge *= min(urgency_mult, 1.5)

            if edge < self.get_min_edge():
                return None

            max_bet_cents = int(self.cfg.max_bet_size * 100)
            count = max(1, max_bet_cents // no_price)

            resolve_str = f"{hours_left:.0f}h" if hours_left < 9999 else "?"

            return {
                "ticker": ticker,
                "event_ticker": event_ticker,
                "side": "no",
                "action": "buy",
                "price_cents": no_price,
                "count": count,
                "edge": edge,
                "reason": (f"Tail NO bet: YES@{yes_ask}c → "
                           f"buy NO@{no_price}c (edge {edge:.1%}) "
                           f"resolves={resolve_str}"),
                "question": question,
                "category": category,
                "series_ticker": series_ticker,
            }

        # Case 2: YES is expensive → buy YES (high probability YES wins)
        if yes_bid >= MIN_YES_PRICE_FOR_YES_BET:
            buy_price = yes_ask
            if buy_price >= 100 or buy_price <= 0:
                return None

            fee = self.client.calc_taker_fee(1, buy_price)
            edge = (100 - buy_price - fee) / buy_price

            # Urgency boost for fast-resolving markets
            if hours_left <= URGENCY_BOOST_HOURS:
                urgency_mult = 1.0 + (URGENCY_BOOST_HOURS - hours_left) / URGENCY_BOOST_HOURS
                edge *= min(urgency_mult, 1.5)

            if edge < self.get_min_edge():
                return None

            max_bet_cents = int(self.cfg.max_bet_size * 100)
            count = max(1, max_bet_cents // buy_price)

            resolve_str = f"{hours_left:.0f}h" if hours_left < 9999 else "?"

            return {
                "ticker": ticker,
                "event_ticker": event_ticker,
                "side": "yes",
                "action": "buy",
                "price_cents": buy_price,
                "count": count,
                "edge": edge,
                "reason": (f"Tail YES bet: YES@{yes_ask}c → "
                           f"buy YES@{buy_price}c (edge {edge:.1%}) "
                           f"resolves={resolve_str}"),
                "question": question,
                "category": category,
                "series_ticker": series_ticker,
            }

        return None
