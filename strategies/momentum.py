"""Strategy 4: Momentum / Volume Surge Trading.

Detects markets where price is moving directionally with increasing volume.
Buys in the direction of the move and exits when momentum fades or
at a profit target / stop loss.

Only trades fast-resolving, high-volume markets for quick capital turnover.
"""

import logging
from datetime import datetime, timezone, timedelta
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

# Minimum price move in last N trades to trigger
MIN_PRICE_MOVE_CENTS = 3
# Minimum 24h volume
MIN_VOLUME_24H = 100
# Profit target and stop loss as fraction of entry
PROFIT_TARGET_PCT = 0.15
STOP_LOSS_PCT = 0.08


class MomentumStrategy(BaseStrategy):
    name = "momentum"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track price snapshots across cycles for trend detection
        self._price_history: dict[str, list[tuple[float, int]]] = {}  # ticker -> [(timestamp, yes_price)]

    def scan(self) -> list[dict]:
        """Find markets with strong price momentum."""
        signals = []

        try:
            markets = self.client.get_all_markets(status="open", max_pages=5)
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

    def _is_candidate(self, market: dict) -> bool:
        """Quick filter before expensive analysis."""
        status = market.get("status", "")
        if status not in ("open", "active"):
            return False

        # Must resolve soon
        if not self._resolves_soon(market):
            return False

        # Must have volume
        vol24 = int(market.get("volume_24h", 0) or 0)
        volume = int(market.get("volume", 0) or 0)
        if vol24 < MIN_VOLUME_24H and volume < self.cfg.min_volume_24h:
            return False

        # Skip if already holding
        ticker = market.get("ticker", "")
        if self.tracker.has_position(ticker):
            return False

        # Need reasonable pricing (not at extremes)
        yes_bid = market.get("yes_bid")
        yes_ask = market.get("yes_ask")
        if yes_bid is None or yes_ask is None:
            return False
        yes_bid = int(yes_bid)
        yes_ask = int(yes_ask)
        if yes_bid < 5 or yes_ask > 95:
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
        # Keep last 20 snapshots
        self._price_history[ticker] = self._price_history[ticker][-20:]

        history = self._price_history[ticker]
        if len(history) < 3:
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

        # Require at least 60% of moves in the same direction
        if recent_move > 0:
            consistency = up_moves / total_moves
        else:
            consistency = down_moves / total_moves

        if consistency < 0.6:
            return None

        # Volume confirmation — need real 24h volume
        vol24 = int(market.get("volume_24h", 0) or 0)

        # Calculate edge based on momentum strength and volume
        momentum_strength = abs(recent_move) / mid  # Price move as % of price
        volume_score = min(1.0, vol24 / 500)  # Normalized volume (500+ = full score)
        edge = momentum_strength * volume_score * consistency

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

        # Size based on confidence
        max_bet_cents = int(self.cfg.max_bet_size * 100)
        count = max(1, max_bet_cents // price)

        event_ticker = market.get("event_ticker", "")
        category = market.get("category", "")
        series_ticker = market.get("series_ticker", "")
        question = market.get("yes_sub_title", market.get("title", ticker))

        direction = "UP" if recent_move > 0 else "DOWN"

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
                       f"consistency={consistency:.0%} vol24={vol24} | "
                       f"{question}"),
            "question": question,
            "category": category,
            "series_ticker": series_ticker,
        }

    def check_exits(self) -> list[dict]:
        """Check open momentum positions for exit signals (profit target or stop loss)."""
        exits = []
        for pos in self.tracker.get_open_by_strategy(self.name):
            entry = pos.entry_price_cents
            current = pos.current_price_cents
            if entry <= 0:
                continue

            pnl_pct = (current - entry) / entry

            if pnl_pct >= PROFIT_TARGET_PCT:
                exits.append({
                    "ticker": pos.ticker,
                    "reason": f"PROFIT TARGET: {pos.ticker} entry={entry}c now={current}c ({pnl_pct:+.1%})",
                    "pnl_pct": pnl_pct,
                })
            elif pnl_pct <= -STOP_LOSS_PCT:
                exits.append({
                    "ticker": pos.ticker,
                    "reason": f"STOP LOSS: {pos.ticker} entry={entry}c now={current}c ({pnl_pct:+.1%})",
                    "pnl_pct": pnl_pct,
                })

        return exits
