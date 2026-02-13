"""Polymarket API client wrapper.

Wraps py-clob-client and the Gamma API for market discovery,
orderbook access, and order placement.
"""

import logging
import time
from typing import Optional

import requests
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType

from config import Config

logger = logging.getLogger(__name__)

# Simple rate limiter: track request timestamps
_request_times: list[float] = []
MAX_REQUESTS_PER_MINUTE = 80  # Stay under 100 limit


def _rate_limit():
    """Sleep if approaching rate limit."""
    now = time.time()
    _request_times[:] = [t for t in _request_times if now - t < 60]
    if len(_request_times) >= MAX_REQUESTS_PER_MINUTE:
        sleep_for = 60 - (now - _request_times[0]) + 0.5
        logger.debug("Rate limit: sleeping %.1fs", sleep_for)
        time.sleep(max(sleep_for, 1.0))
    _request_times.append(time.time())


class PolymarketClient:
    """Unified client for Polymarket CLOB + Gamma APIs."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.clob = ClobClient(
            cfg.clob_url,
            key=cfg.private_key,
            chain_id=cfg.chain_id,
        )
        # Derive API creds for authenticated endpoints
        try:
            self.clob.set_api_creds(self.clob.create_or_derive_api_creds())
            logger.info("Authenticated with CLOB API")
        except Exception as e:
            logger.warning("Could not derive API creds (dry-run ok): %s", e)

    # ── Market discovery (Gamma API) ─────────────────────────────

    def get_markets(self, limit: int = 100, offset: int = 0,
                    active: bool = True, closed: bool = False,
                    category: Optional[str] = None,
                    order: str = "volume",
                    min_liquidity: Optional[float] = None) -> list[dict]:
        """Fetch markets from the Gamma API with filtering."""
        _rate_limit()
        params: dict = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "order": order,
            "ascending": "false",
        }
        if category:
            params["category"] = category
        if min_liquidity is not None:
            params["liquidity_num_min"] = min_liquidity
        resp = requests.get(f"{self.cfg.gamma_url}/markets", params=params,
                            timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_all_markets(self, active: bool = True,
                        category: Optional[str] = None,
                        min_liquidity: Optional[float] = None,
                        max_pages: int = 5) -> list[dict]:
        """Paginate through all markets."""
        all_markets = []
        for page in range(max_pages):
            batch = self.get_markets(
                limit=100, offset=page * 100, active=active,
                category=category, min_liquidity=min_liquidity,
            )
            if not batch:
                break
            all_markets.extend(batch)
            if len(batch) < 100:
                break
        return all_markets

    def get_events(self, limit: int = 50) -> list[dict]:
        """Fetch events (groups of related markets) from Gamma."""
        _rate_limit()
        resp = requests.get(f"{self.cfg.gamma_url}/events",
                            params={"limit": limit, "active": "true"},
                            timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_event(self, event_id: str) -> Optional[dict]:
        """Fetch a single event by ID."""
        _rate_limit()
        try:
            resp = requests.get(f"{self.cfg.gamma_url}/events/{event_id}",
                                timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    # ── Orderbook & pricing ──────────────────────────────────────

    def get_orderbook(self, token_id: str) -> Optional[dict]:
        """Get the full orderbook for a token (YES or NO side)."""
        _rate_limit()
        try:
            return self.clob.get_order_book(token_id)
        except Exception as e:
            logger.debug("Failed to get orderbook for %s: %s", token_id[:12], e)
            return None

    def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get the midpoint price for a token."""
        _rate_limit()
        try:
            mid = self.clob.get_midpoint(token_id)
            return float(mid) if mid else None
        except Exception:
            return None

    def get_price(self, token_id: str, side: str = "BUY") -> Optional[float]:
        """Get the current price for a token on a given side."""
        _rate_limit()
        try:
            price = self.clob.get_price(token_id, side)
            return float(price) if price else None
        except Exception:
            return None

    def get_spread(self, token_id: str) -> Optional[dict]:
        """Get the best bid/ask spread for a token."""
        _rate_limit()
        try:
            return self.clob.get_spread(token_id)
        except Exception:
            return None

    def get_tick_size(self, token_id: str) -> str:
        """Get the tick size for a token. Defaults to '0.01'."""
        _rate_limit()
        try:
            return self.clob.get_tick_size(token_id)
        except Exception:
            return "0.01"

    def get_fee_rate(self, token_id: str) -> float:
        """Get the fee rate in basis points for a token."""
        _rate_limit()
        try:
            rate = self.clob.get_fee_rate(token_id)
            return float(rate) if rate else 0.0
        except Exception:
            return 0.0

    def get_best_bid_ask(self, token_id: str) -> tuple[Optional[float], Optional[float]]:
        """Return (best_bid, best_ask) for a token."""
        book = self.get_orderbook(token_id)
        if not book:
            return None, None
        best_bid = float(book["bids"][0]["price"]) if book.get("bids") else None
        best_ask = float(book["asks"][0]["price"]) if book.get("asks") else None
        return best_bid, best_ask

    def get_last_trade_price(self, token_id: str) -> Optional[float]:
        """Get the last trade price for a token."""
        _rate_limit()
        try:
            result = self.clob.get_last_trade_price(token_id)
            return float(result) if result else None
        except Exception:
            return None

    # ── Order placement ──────────────────────────────────────────

    def buy(self, token_id: str, price: float, size: float,
            dry_run: bool = True) -> Optional[dict]:
        """Place a limit buy order for a token.

        Args:
            token_id: The YES or NO token to buy.
            price: Limit price (0.01 - 0.99).
            size: Number of shares.
            dry_run: If True, only log but don't submit.
        """
        logger.info("BUY %s | price=%.4f size=%.2f %s",
                     token_id[:12], price, size,
                     "[DRY RUN]" if dry_run else "[LIVE]")
        if dry_run:
            return {"status": "dry_run", "side": "BUY",
                    "price": price, "size": size, "token_id": token_id}

        _rate_limit()
        order_args = OrderArgs(
            price=price,
            size=size,
            side="BUY",
            token_id=token_id,
        )
        signed = self.clob.create_order(order_args)
        result = self.clob.post_order(signed, OrderType.GTC)
        logger.info("Order placed: %s", result)
        return result

    def sell(self, token_id: str, price: float, size: float,
             dry_run: bool = True) -> Optional[dict]:
        """Place a limit sell order for a token."""
        logger.info("SELL %s | price=%.4f size=%.2f %s",
                     token_id[:12], price, size,
                     "[DRY RUN]" if dry_run else "[LIVE]")
        if dry_run:
            return {"status": "dry_run", "side": "SELL",
                    "price": price, "size": size, "token_id": token_id}

        _rate_limit()
        order_args = OrderArgs(
            price=price,
            size=size,
            side="SELL",
            token_id=token_id,
        )
        signed = self.clob.create_order(order_args)
        result = self.clob.post_order(signed, OrderType.GTC)
        logger.info("Order placed: %s", result)
        return result

    def cancel_all(self) -> None:
        """Cancel all open orders."""
        try:
            _rate_limit()
            self.clob.cancel_all()
            logger.info("All orders cancelled")
        except Exception as e:
            logger.error("Failed to cancel orders: %s", e)
