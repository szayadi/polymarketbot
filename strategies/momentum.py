"""Strategy 4: Aggressive Momentum / Fast Scalping.

Detects markets where price is moving directionally with volume.
Buys in the direction of the move and takes profits FAST.

Optimized for same-day and next-day markets with rapid turnover.
Trailing stop locks in gains once a position is profitable.
"""

import logging
from datetime import datetime, timezone, timedelta
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

# ── Aggressive thresholds for fast trading ──
MIN_PRICE_MOVE_CENTS = 2          # Trigger on 2c moves (was 3)
MIN_VOLUME_24H = 50               # Lower volume bar (was 100)
PROFIT_TARGET_PCT = 0.08          # Take profits at 8% (was 15%)
STOP_LOSS_PCT = 0.05              # Cut losses at 5% (was 8%)
TRAILING_STOP_PCT = 0.03          # 3% trailing stop once profitable
CONSISTENCY_MIN = 0.50            # 50% directional consistency (was 60%)
MIN_SNAPSHOTS = 2                 # React after just 2 snapshots (was 3)
VOLUME_FULL_SCORE = 200           # Full volume score at 200 (was 500)
# Time-based urgency: markets resolving sooner get higher edge boost
URGENCY_BOOST_HOURS = 48          # Markets resolving within 48h get a boost


class MomentumStrategy(BaseStrategy):
    name = "momentum"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track price snapshots across cycles for trend detection
        self._price_history: dict[str, list[tuple[float, int]]] = {}  # ticker -> [(timestamp, yes_price)]
        # Track high-water mark for trailing stops
        self._high_water: dict[str, int] = {}  # ticker -> best price seen

    def scan(self) -> list[dict]:
        """Find markets with strong price momentum."""
        signals = []

        try:
            markets = self.client.get_all_markets(status="open", max_pages=8)
        except Exception as e:
            logger.error("[momentum] Failed to fetch markets: %s", e)
            return signals

        # Filter to tradeable markets first
        candidates = []
        for m in markets:
            if self._is_candidate(m):
                candidates.append(m)

        logger.info("[momentum] Scanning %d candidates (from %d markets)",
                    len(candidates), len(markets))

        for market in candidates:
            try:
                sig = self._evaluate_market(market)
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.debug("[momentum] Error on %s: %s",
                             market.get("ticker", ""), e)

        signals.sort(key=lambda s: s["edge"], reverse=True)
        logger.info("[momentum] Found %d signals", len(signals))
        return signals

    def _resolves_soon(self, market: dict) -> bool:
        """Check if market resolves within our time window."""
        cutoff = datetime.now(timezone.utc) + timedelta(days=self.cfg.max_days_to_resolve)
        for field in ("expected_expiration_time", "close_time", "latest_expiration_time"):
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

    def _hours_to_resolve(self, market: dict) -> float:
        """Estimate hours until market resolves. Returns 9999 if unknown."""
        now = datetime.now(timezone.utc)
        for field in ("expected_expiration_time", "close_time", "latest_expiration_time"):
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

    def _is_candidate(self, market: dict) -> bool:
        """Quick filter before expensive analysis."""
        status = market.get("status", "")
        if status not in ("open", "active"):
            return False

        # Must resolve soon
        if not self._resolves_soon(market):
            return False

        # Must have volume — lowered threshold
        vol24 = int(market.get("volume_24h", 0) or 0)
        volume = int(market.get("volume", 0) or 0)
        if vol24 < MIN_VOLUME_24H and volume < self.cfg.min_volume_24h:
            return False

        # Skip if already holding
        ticker = market.get("ticker", "")
        if self.tracker.has_position(ticker):
            return False

        # Need reasonable pricing (avoid deep extremes)
        yes_bid = market.get("yes_bid")
        yes_ask = market.get("yes_ask")
        if yes_bid is None or yes_ask is None:
            return False
        yes_bid = int(yes_bid)
        yes_ask = int(yes_ask)
        if yes_bid < 10 or yes_ask > 90:
            return False

        return True

    def _evaluate_market(self, market: dict) -> dict | None:
        """Analyze recent trades for momentum signal."""
        ticker = market.get("ticker", "")

        yes_bid = int(market.get("yes_bid", 0) or 0)
        yes_ask = int(market.get("yes_ask", 0) or 0)
        if yes_bid <= 0 or yes_ask <= 0:
            return None

        mid = (yes_bid + yes_ask) / 2.0
        spread = yes_ask - yes_bid

        # Record current price for cross-cycle tracking
        now = datetime.now(timezone.utc).timestamp()
        if ticker not in self._price_history:
            self._price_history[ticker] = []
        self._price_history[ticker].append((now, int(mid)))
        # Keep last 30 snapshots (more data for faster polling)
        self._price_history[ticker] = self._price_history[ticker][-30:]

        history = self._price_history[ticker]
        if len(history) < MIN_SNAPSHOTS:
            return None

        # Calculate price trend across our snapshots
        prices = [p for _, p in history]
        recent_move = prices[-1] - prices[0]
        # Check consistency — are prices moving in one direction?
        up_moves = sum(1 for i in range(1, len(prices)) if prices[i] > prices[i-1])
        down_moves = sum(1 for i in range(1, len(prices)) if prices[i] < prices[i-1])
        total_moves = len(prices) - 1

        if abs(recent_move) < MIN_PRICE_MOVE_CENTS:
            return None

        # Require directional consistency
        if recent_move > 0:
            consistency = up_moves / total_moves
        else:
            consistency = down_moves / total_moves

        if consistency < CONSISTENCY_MIN:
            return None

        # Volume confirmation
        vol24 = int(market.get("volume_24h", 0) or 0)

        # Calculate edge based on momentum strength, volume, and urgency
        momentum_strength = abs(recent_move) / mid  # Price move as % of price
        volume_score = min(1.0, vol24 / VOLUME_FULL_SCORE)
        edge = momentum_strength * volume_score * consistency

        # URGENCY BOOST: markets resolving sooner get higher edge
        hours_left = self._hours_to_resolve(market)
        if hours_left <= URGENCY_BOOST_HOURS:
            urgency_mult = 1.0 + (URGENCY_BOOST_HOURS - hours_left) / URGENCY_BOOST_HOURS
            edge *= min(urgency_mult, 2.0)  # Up to 2x boost for imminent resolution

        if edge < self.get_min_edge():
            return None

        # Trade direction: follow the momentum
        if recent_move > 0:
            # Price going UP → buy YES
            side = "yes"
            price = yes_ask
        else:
            # Price going DOWN → buy NO
            side = "no"
            price = 100 - yes_bid

        if price <= 0 or price >= 100:
            return None

        # Size based on confidence — bigger bets on stronger signals
        max_bet_cents = int(self.cfg.max_bet_size * 100)
        count = max(1, max_bet_cents // price)

        event_ticker = market.get("event_ticker", "")
        category = market.get("category", "")
        series_ticker = market.get("series_ticker", "")
        question = market.get("yes_sub_title", market.get("title", ticker))

        direction = "UP" if recent_move > 0 else "DOWN"
        resolve_str = f"{hours_left:.0f}h" if hours_left < 9999 else "?"

        return {
            "ticker": ticker,
            "event_ticker": event_ticker,
            "side": side,
            "action": "buy",
            "price_cents": price,
            "count": count,
            "edge": edge,
            "reason": (f"Momentum {direction}: {ticker} moved {recent_move:+d}c "
                       f"over {len(history)} snapshots | "
                       f"consistency={consistency:.0%} vol24={vol24} "
                       f"resolves={resolve_str} | "
                       f"{question}"),
            "question": question,
            "category": category,
            "series_ticker": series_ticker,
        }

    def check_exits(self) -> list[dict]:
        """Check open momentum positions for exit signals.

        Uses profit target, stop loss, AND trailing stop for fast profit capture.
        """
        exits = []
        for pos in self.tracker.get_open_by_strategy(self.name):
            entry = pos.entry_price_cents
            current = pos.current_price_cents
            if entry <= 0:
                continue

            pnl_pct = (current - entry) / entry
            ticker = pos.ticker

            # Update high-water mark for trailing stop
            if ticker not in self._high_water:
                self._high_water[ticker] = current
            if current > self._high_water[ticker]:
                self._high_water[ticker] = current

            # PROFIT TARGET — take the win
            if pnl_pct >= PROFIT_TARGET_PCT:
                exits.append({
                    "ticker": ticker,
                    "reason": f"PROFIT TARGET: {ticker} entry={entry}c now={current}c ({pnl_pct:+.1%})",
                    "pnl_pct": pnl_pct,
                })
                self._high_water.pop(ticker, None)

            # TRAILING STOP — lock in gains once profitable
            elif pnl_pct > 0 and ticker in self._high_water:
                peak = self._high_water[ticker]
                drawback = (peak - current) / peak if peak > 0 else 0
                if drawback >= TRAILING_STOP_PCT:
                    exits.append({
                        "ticker": ticker,
                        "reason": (f"TRAILING STOP: {ticker} peak={peak}c now={current}c "
                                   f"(entry={entry}c, drew back {drawback:.1%})"),
                        "pnl_pct": pnl_pct,
                    })
                    self._high_water.pop(ticker, None)

            # STOP LOSS — cut losses fast
            elif pnl_pct <= -STOP_LOSS_PCT:
                exits.append({
                    "ticker": ticker,
                    "reason": f"STOP LOSS: {ticker} entry={entry}c now={current}c ({pnl_pct:+.1%})",
                    "pnl_pct": pnl_pct,
                })
                self._high_water.pop(ticker, None)

        return exits
