"""Kalshi API client with RSA-PSS authentication.

Handles all interaction with the Kalshi REST API:
- Market discovery (events, markets, orderbooks)
- Order placement and management
- Portfolio and balance queries
- Rate limiting to stay under tier limits
"""

import base64
import logging
import time
import uuid
from typing import Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from config import Config

logger = logging.getLogger(__name__)

# Rate limiting: basic tier = 20 read/sec, 10 write/sec
_request_times: list[float] = []
MAX_REQUESTS_PER_SECOND = 8  # Stay safely under 10 write/sec


def _rate_limit():
    """Sleep if approaching rate limit."""
    now = time.time()
    _request_times[:] = [t for t in _request_times if now - t < 1.0]
    if len(_request_times) >= MAX_REQUESTS_PER_SECOND:
        sleep_for = 1.0 - (now - _request_times[0]) + 0.05
        time.sleep(max(sleep_for, 0.1))
    _request_times.append(time.time())


class KalshiClient:
    """Client for the Kalshi REST API v2."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.base_url = cfg.base_url
        self._private_key = None
        self._load_private_key()

    def _load_private_key(self):
        """Load RSA private key from PEM file."""
        try:
            with open(self.cfg.private_key_path, "rb") as f:
                self._private_key = serialization.load_pem_private_key(
                    f.read(), password=None
                )
            logger.info("Loaded private key from %s", self.cfg.private_key_path)
        except FileNotFoundError:
            logger.warning("Private key not found at %s (dry-run ok)",
                           self.cfg.private_key_path)
        except Exception as e:
            logger.warning("Failed to load private key: %s", e)

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        """Create RSA-PSS signature for a request."""
        # Kalshi requires the full path including /trade-api/v2 prefix
        message = f"{timestamp_ms}{method.upper()}/trade-api/v2{path}"
        signature = self._private_key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _auth_headers(self, method: str, path: str) -> dict:
        """Generate authentication headers for a request."""
        timestamp_ms = str(int(time.time() * 1000))
        # Strip query params from path for signing
        sign_path = path.split("?")[0]
        signature = self._sign(timestamp_ms, method, sign_path)
        return {
            "KALSHI-ACCESS-KEY": self.cfg.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, params: dict = None,
                 json_data: dict = None, auth: bool = True) -> Optional[dict]:
        """Make an authenticated API request."""
        _rate_limit()
        url = f"{self.base_url}{path}"
        headers = self._auth_headers(method, path) if auth and self._private_key else {}

        try:
            resp = requests.request(
                method, url, headers=headers, params=params,
                json=json_data, timeout=15,
            )
            if resp.status_code == 429:
                logger.warning("Rate limited — backing off 2s")
                time.sleep(2)
                return self._request(method, path, params, json_data, auth)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            logger.error("API error %s %s: %s", method, path, e)
            if resp is not None:
                logger.debug("Response body: %s", resp.text[:500])
            return None
        except Exception as e:
            logger.error("Request failed %s %s: %s", method, path, e)
            return None

    # ── Market Discovery ─────────────────────────────────────────

    def get_markets(self, limit: int = 100, cursor: str = "",
                    status: str = "open", event_ticker: str = "",
                    series_ticker: str = "") -> Optional[dict]:
        """Fetch markets with filtering."""
        params: dict = {"limit": limit, "status": status}
        if cursor:
            params["cursor"] = cursor
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        return self._request("GET", "/markets", params=params, auth=False)

    def get_all_markets(self, status: str = "open",
                        max_pages: int = 5) -> list[dict]:
        """Paginate through all markets."""
        all_markets = []
        cursor = ""
        for _ in range(max_pages):
            result = self.get_markets(limit=200, cursor=cursor, status=status)
            if not result or not result.get("markets"):
                break
            all_markets.extend(result["markets"])
            cursor = result.get("cursor", "")
            if not cursor:
                break
        return all_markets

    def get_market(self, ticker: str) -> Optional[dict]:
        """Get details for a single market."""
        result = self._request("GET", f"/markets/{ticker}", auth=False)
        return result.get("market") if result else None

    def get_events(self, limit: int = 100, cursor: str = "",
                   status: str = "open",
                   with_nested_markets: bool = True) -> Optional[dict]:
        """Fetch events (groups of related markets)."""
        params: dict = {
            "limit": limit,
            "status": status,
            "with_nested_markets": str(with_nested_markets).lower(),
        }
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/events", params=params, auth=False)

    def get_all_events(self, status: str = "open",
                       max_pages: int = 5) -> list[dict]:
        """Paginate through all events with nested markets."""
        all_events = []
        cursor = ""
        for _ in range(max_pages):
            result = self.get_events(limit=200, cursor=cursor, status=status,
                                     with_nested_markets=True)
            if not result or not result.get("events"):
                break
            all_events.extend(result["events"])
            cursor = result.get("cursor", "")
            if not cursor:
                break
        return all_events

    def get_event(self, event_ticker: str) -> Optional[dict]:
        """Get a single event with its markets."""
        result = self._request("GET", f"/events/{event_ticker}", auth=False)
        return result.get("event") if result else None

    # ── Orderbook & Pricing ──────────────────────────────────────

    def get_orderbook(self, ticker: str, depth: int = 10) -> Optional[dict]:
        """Get the orderbook for a market."""
        result = self._request("GET", f"/markets/{ticker}/orderbook",
                               params={"depth": depth}, auth=False)
        return result.get("orderbook") if result else None

    def get_best_bid_ask(self, ticker: str) -> tuple[Optional[int], Optional[int]]:
        """Get best YES bid and ask prices in cents.

        Returns (best_yes_bid_cents, best_yes_ask_cents).
        """
        book = self.get_orderbook(ticker, depth=1)
        if not book:
            return None, None

        yes_bid = None
        yes_ask = None

        # yes_bids: [[price_str, quantity], ...]
        if book.get("yes") and book["yes"]:
            yes_bid = int(book["yes"][0][0])
        # Compute ask from NO bids: yes_ask = 100 - best_no_bid
        if book.get("no") and book["no"]:
            best_no_bid = int(book["no"][0][0])
            yes_ask = 100 - best_no_bid

        return yes_bid, yes_ask

    def get_trades(self, ticker: str, limit: int = 50) -> Optional[list]:
        """Get recent trades for a market."""
        result = self._request("GET", f"/markets/{ticker}/trades",
                               params={"limit": limit}, auth=False)
        return result.get("trades") if result else None

    # ── Portfolio ────────────────────────────────────────────────

    def get_balance(self) -> Optional[int]:
        """Get account balance in cents."""
        result = self._request("GET", "/portfolio/balance")
        return result.get("balance") if result else None

    def get_balance_dollars(self) -> Optional[float]:
        """Get account balance in dollars."""
        cents = self.get_balance()
        return cents / 100.0 if cents is not None else None

    def get_positions(self, ticker: str = "",
                      event_ticker: str = "") -> Optional[list]:
        """Get open positions."""
        params: dict = {"limit": 200}
        if ticker:
            params["ticker"] = ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        result = self._request("GET", "/portfolio/positions", params=params)
        return result.get("market_positions") if result else None

    def get_fills(self, ticker: str = "", limit: int = 100) -> Optional[list]:
        """Get executed trades (fills)."""
        params: dict = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        result = self._request("GET", "/portfolio/fills", params=params)
        return result.get("fills") if result else None

    def get_settlements(self, limit: int = 100) -> Optional[list]:
        """Get settlement history."""
        result = self._request("GET", "/portfolio/settlements",
                               params={"limit": limit})
        return result.get("settlements") if result else None

    def get_orders(self, ticker: str = "",
                   status: str = "resting") -> Optional[list]:
        """Get orders filtered by status."""
        params: dict = {"limit": 200, "status": status}
        if ticker:
            params["ticker"] = ticker
        result = self._request("GET", "/portfolio/orders", params=params)
        return result.get("orders") if result else None

    # ── Order Placement ──────────────────────────────────────────

    def place_order(self, ticker: str, side: str, action: str,
                    count: int, yes_price_cents: Optional[int] = None,
                    no_price_cents: Optional[int] = None,
                    time_in_force: str = "good_till_canceled",
                    dry_run: bool = True) -> Optional[dict]:
        """Place a limit order.

        Args:
            ticker: Market ticker.
            side: "yes" or "no".
            action: "buy" or "sell".
            count: Number of contracts.
            yes_price_cents: Price in cents (1-99) for YES side.
            no_price_cents: Price in cents (1-99) for NO side.
            time_in_force: "good_till_canceled", "fill_or_kill", "immediate_or_cancel".
            dry_run: If True, only log.
        """
        price_str = ""
        if yes_price_cents:
            price_str = f"yes@{yes_price_cents}c"
        elif no_price_cents:
            price_str = f"no@{no_price_cents}c"

        logger.info("ORDER: %s %s %s x%d %s %s",
                     action.upper(), side.upper(), ticker, count,
                     price_str, "[DRY RUN]" if dry_run else "[LIVE]")

        if dry_run:
            return {
                "status": "dry_run", "ticker": ticker, "side": side,
                "action": action, "count": count,
                "yes_price": yes_price_cents, "no_price": no_price_cents,
            }

        body: dict = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "client_order_id": str(uuid.uuid4()),
            "time_in_force": time_in_force,
        }
        if yes_price_cents is not None:
            body["yes_price"] = yes_price_cents
        if no_price_cents is not None:
            body["no_price"] = no_price_cents

        result = self._request("POST", "/portfolio/orders", json_data=body)
        if result and "order" in result:
            logger.info("Order placed: %s", result["order"].get("order_id", ""))
        return result.get("order") if result else None

    def cancel_order(self, order_id: str) -> Optional[dict]:
        """Cancel a single order."""
        result = self._request("DELETE", f"/portfolio/orders/{order_id}")
        if result:
            logger.info("Order cancelled: %s", order_id)
        return result

    def cancel_all_orders(self) -> None:
        """Cancel all resting orders."""
        orders = self.get_orders(status="resting")
        if not orders:
            return
        for order in orders:
            oid = order.get("order_id", "")
            if oid:
                self.cancel_order(oid)
        logger.info("Cancelled %d orders", len(orders))

    # ── Fee Calculation ──────────────────────────────────────────

    @staticmethod
    def calc_taker_fee(count: int, price_cents: int) -> float:
        """Calculate taker fee in cents.

        Formula: ceil(0.07 * count * P * (1 - P))
        where P = price_cents / 100
        """
        import math
        p = price_cents / 100.0
        fee = 0.07 * count * p * (1.0 - p)
        return math.ceil(fee * 100) / 100.0  # Round up to nearest cent

    @staticmethod
    def calc_fee_pct(price_cents: int) -> float:
        """Fee as a percentage of contract value at a given price."""
        p = price_cents / 100.0
        # Fee per contract = 0.07 * P * (1-P)
        # As percentage of cost: fee / price
        if p <= 0 or p >= 1:
            return 0.0
        fee_per_contract = 0.07 * p * (1.0 - p)
        return fee_per_contract / p

    # ── Exchange Status ──────────────────────────────────────────

    def get_exchange_status(self) -> Optional[dict]:
        """Check if the exchange is operational."""
        return self._request("GET", "/exchange/status", auth=False)
