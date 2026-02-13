#!/usr/bin/env python3
"""Polymarket Trading Bot — Main Runner.

Three strategies running on a $10 bankroll:
  1. High-probability NO bets on near-impossible outcomes
  2. Logic/conditional arbitrage across related markets
  3. Spread trading on sports & politics markets

Usage:
    # Dry run (default — no real money):
    python main.py

    # Live trading (set DRY_RUN=0 in .env):
    python main.py

    # See .env.example for all configuration options.
"""

import logging
import signal
import sys
import time

from config import Config
from client import PolymarketClient
from positions import PositionTracker
from risk import RiskManager
from strategies.no_bets import NoBetsStrategy
from strategies.arbitrage import ArbitrageStrategy
from strategies.spread import SpreadStrategy

# ── Logging ──────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log", mode="a"),
    ],
)
logger = logging.getLogger("polybot")


# ── Globals for signal handling ──────────────────────────────────

_shutdown = False


def _handle_shutdown(signum, frame):
    global _shutdown
    logger.info("Shutdown signal received — cleaning up...")
    _shutdown = True


def main():
    cfg = Config()

    # ── Banner ───────────────────────────────────────────────────
    mode = "DRY RUN" if cfg.dry_run else "LIVE TRADING"
    logger.info("=" * 60)
    logger.info("  Polymarket Trading Bot")
    logger.info("  Mode: %s", mode)
    logger.info("  Bankroll: $%.2f", cfg.bankroll)
    logger.info("  Max bet: $%.2f (%.0f%% of bankroll)",
                cfg.max_bet_size, cfg.max_bet_fraction * 100)
    logger.info("  Min edge: %.1f%%", cfg.min_edge * 100)
    logger.info("  Poll interval: %ds", cfg.poll_interval)
    logger.info("  Strategies: NO=%s  ARB=%s  SPREAD=%s",
                cfg.strategy_no_bets, cfg.strategy_arbitrage,
                cfg.strategy_spread)
    logger.info("=" * 60)

    if not cfg.private_key and not cfg.dry_run:
        logger.error("PRIVATE_KEY required for live trading. Set it in .env")
        sys.exit(1)

    # ── Initialize components ────────────────────────────────────
    client = PolymarketClient(cfg)
    tracker = PositionTracker()
    risk = RiskManager(cfg, tracker)

    strategies = []
    if cfg.strategy_no_bets:
        strategies.append(NoBetsStrategy(client, cfg, risk, tracker))
    if cfg.strategy_arbitrage:
        strategies.append(ArbitrageStrategy(client, cfg, risk, tracker))
    if cfg.strategy_spread:
        strategies.append(SpreadStrategy(client, cfg, risk, tracker))

    if not strategies:
        logger.error("No strategies enabled. Set STRATEGY_* in .env")
        sys.exit(1)

    logger.info("Loaded %d strategies: %s",
                len(strategies), [s.name for s in strategies])

    # ── Signal handlers ──────────────────────────────────────────
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    # ── Main loop ────────────────────────────────────────────────
    cycle = 0
    while not _shutdown:
        cycle += 1
        logger.info("─── Cycle %d ───", cycle)

        # Check if trading is halted
        halted, reason = risk.is_trading_halted()
        if halted:
            logger.warning("Trading halted: %s", reason)
            logger.info("Waiting for next day or manual intervention...")
            _sleep(cfg.poll_interval * 4)
            continue

        # Run each strategy
        total_signals = 0
        total_trades = 0
        for strategy in strategies:
            try:
                signals = strategy.scan()
                total_signals += len(signals)
                if signals:
                    results = strategy.execute(signals)
                    total_trades += len(results)
            except Exception as e:
                logger.error("[%s] Error: %s", strategy.name, e, exc_info=True)

        # Clean up stale spread orders
        for strategy in strategies:
            if hasattr(strategy, "cleanup_stale_orders"):
                strategy.cleanup_stale_orders()

        # Update position prices
        _update_positions(client, tracker)

        # Log summary
        logger.info("Cycle %d complete: %d signals → %d trades",
                     cycle, total_signals, total_trades)
        logger.info("Portfolio: %s", tracker.summary())

        # Save state
        tracker.save()

        # Sleep
        _sleep(cfg.poll_interval)

    # ── Shutdown ─────────────────────────────────────────────────
    logger.info("Shutting down...")
    if not cfg.dry_run:
        logger.info("Cancelling all open orders...")
        client.cancel_all()
    tracker.save()
    logger.info("State saved. Bot stopped.")


def _update_positions(client: PolymarketClient, tracker: PositionTracker):
    """Refresh current prices for all open positions."""
    for pos in tracker.get_open():
        try:
            price = client.get_midpoint(pos.token_id)
            if price is not None:
                tracker.update_price(pos.token_id, price)
        except Exception:
            pass


def _sleep(seconds: int):
    """Interruptible sleep that respects shutdown flag."""
    for _ in range(seconds):
        if _shutdown:
            return
        time.sleep(1)


if __name__ == "__main__":
    main()
