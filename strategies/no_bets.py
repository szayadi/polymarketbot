"""Strategy 1: High-Probability NO Bets.

Targets near-impossible outcomes where YES is priced at 1-5 cents.
Buys NO shares at 95-99 cents to collect $1.00 on resolution.
This is systematic risk underwriting — not gambling.
"""

import logging
from datetime import datetime, timezone

from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

# YES price thresholds
MAX_YES_PRICE = 0.05      # Only consider markets where YES <= 5 cents
MIN_LIQUIDITY = 500.0     # Minimum market liquidity in USD
MIN_HOURS_TO_CLOSE = 24   # Don't trade markets closing within 24h


class NoBetsStrategy(BaseStrategy):
    name = "no_bets"

    def scan(self) -> list[dict]:
        """Find markets with near-impossible YES outcomes."""
        signals = []

        try:
            markets = self.client.get_all_markets(
                active=True, min_liquidity=MIN_LIQUIDITY, max_pages=3,
            )
        except Exception as e:
            logger.error("[no_bets] Failed to fetch markets: %s", e)
            return signals

        logger.info("[no_bets] Scanning %d markets for high-prob NO bets", len(markets))

        for market in markets:
            try:
                signal = self._evaluate_market(market)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.debug("[no_bets] Error evaluating market: %s", e)

        # Sort by edge descending — best opportunities first
        signals.sort(key=lambda s: s["edge"], reverse=True)
        logger.info("[no_bets] Found %d signals", len(signals))
        return signals

    def _evaluate_market(self, market: dict) -> dict | None:
        """Check if a market qualifies for a NO bet."""
        # Need outcome prices
        outcome_prices = market.get("outcomePrices")
        if not outcome_prices or len(outcome_prices) < 2:
            return None

        try:
            yes_price = float(outcome_prices[0])
            no_price = float(outcome_prices[1])
        except (ValueError, TypeError):
            return None

        # Filter: YES must be very cheap (near-impossible event)
        if yes_price > MAX_YES_PRICE or yes_price <= 0:
            return None

        # Need token IDs
        token_ids = market.get("clobTokenIds") or market.get("clob_token_ids", [])
        if len(token_ids) < 2:
            return None

        no_token_id = token_ids[1]  # NO token is index 1

        # Check liquidity
        liquidity = float(market.get("liquidity", 0) or 0)
        if liquidity < MIN_LIQUIDITY:
            return None

        # Check time to close — skip markets about to resolve
        end_date_str = market.get("endDate") or market.get("end_date")
        if end_date_str:
            try:
                end_date = datetime.fromisoformat(
                    end_date_str.replace("Z", "+00:00"))
                hours_left = (end_date - datetime.now(timezone.utc)).total_seconds() / 3600
                if hours_left < MIN_HOURS_TO_CLOSE:
                    return None
            except (ValueError, TypeError):
                pass

        # Check the NO token's actual ask price from the order book
        best_bid, best_ask = self.client.get_best_bid_ask(no_token_id)
        if best_ask is None:
            return None

        buy_price = best_ask  # We'd buy NO at the ask

        # Calculate edge: profit per share = $1.00 - buy_price (on resolution)
        # Subtract estimated fee (~2% conservative)
        fee_estimate = 0.02
        edge = (1.0 - buy_price) - fee_estimate

        if edge < self.get_min_edge():
            return None

        # Check that we're not already in this market
        existing = self.tracker.get_market_exposure(
            market.get("condition_id", market.get("id", "")))
        if existing > 0:
            return None

        question = market.get("question", market.get("title", "Unknown"))
        condition_id = market.get("condition_id", market.get("id", ""))

        category = market.get("category", market.get("groupItemTitle", ""))

        return {
            "token_id": no_token_id,
            "side": "BUY",
            "price": buy_price,
            "size": self.cfg.max_bet_size,
            "market_id": condition_id,
            "edge": edge,
            "reason": f"NO bet: YES@{yes_price:.2f} → buy NO@{buy_price:.4f} "
                      f"(edge {edge:.1%})",
            "question": question,
            "category": category,
            "liquidity": liquidity,
        }
