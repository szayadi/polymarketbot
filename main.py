#!/usr/bin/env python3
"""Polymarket Trading Bot — Main Runner.

Three strategies running on a $10 bankroll:
  1. High-probability NO bets on near-impossible outcomes
  2. Logic/conditional arbitrage across related markets
  3. Spread trading on sports & politics markets

Adaptive learning: The bot tracks every trade outcome and adjusts
its behavior over time — scaling up what works, avoiding what doesn't.

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
from learner import AdaptiveLearner
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
    logger.info("  Adaptive learning: ENABLED")
    logger.info("=" * 60)

    if not cfg.private_key and not cfg.dry_run:
        logger.error("PRIVATE_KEY required for live trading. Set it in .env")
        sys.exit(1)

    # ── Initialize components ────────────────────────────────────
    client = PolymarketClient(cfg)
    tracker = PositionTracker()
    learner = AdaptiveLearner()
    risk = RiskManager(cfg, tracker)

    # Log learner state
    if learner.trades:
        logger.info("Learner loaded: %d historical trades", len(learner.trades))
        logger.info("\n%s", learner.get_report())

    strategies = []
    if cfg.strategy_no_bets:
        strategies.append(NoBetsStrategy(client, cfg, risk, tracker, learner))
    if cfg.strategy_arbitrage:
        strategies.append(ArbitrageStrategy(client, cfg, risk, tracker, learner))
    if cfg.strategy_spread:
        strategies.append(SpreadStrategy(client, cfg, risk, tracker, learner))

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
    REPORT_EVERY = 20  # Print learner report every N cycles

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

        # Update position prices and check for resolved markets
        _update_positions(client, tracker)
        _check_resolved_positions(client, tracker, learner)

        # Log summary
        logger.info("Cycle %d complete: %d signals -> %d trades",
                     cycle, total_signals, total_trades)
        logger.info("Portfolio: %s", tracker.summary())

        # Periodic learner report
        if cycle % REPORT_EVERY == 0 and learner.trades:
            logger.info("\n%s", learner.get_report())

        # Save state
        tracker.save()
        learner.save()

        # Sleep
        _sleep(cfg.poll_interval)

    # ── Shutdown ─────────────────────────────────────────────────
    logger.info("Shutting down...")
    if not cfg.dry_run:
        logger.info("Cancelling all open orders...")
        client.cancel_all()
    tracker.save()
    learner.save()

    if learner.trades:
        logger.info("\n%s", learner.get_report())

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


def _check_resolved_positions(client: PolymarketClient,
                               tracker: PositionTracker,
                               learner: AdaptiveLearner):
    """Check if any open positions have resolved and feed results to learner.

    A position is considered resolved when:
    - The market has closed (price moved to 0 or 1)
    - For dry-run: simulate resolution based on current price movement
    """
    for pos in tracker.get_open():
        price = pos.current_price

        # Check for resolution: price at 0 or 1 (or very close)
        resolved = False
        exit_price = 0.0

        if price >= 0.99:
            resolved = True
            exit_price = 1.0
        elif price <= 0.01:
            resolved = True
            exit_price = 0.0

        if not resolved:
            continue

        # Calculate realized P&L
        if pos.side == "YES":
            pnl = (exit_price - pos.entry_price) * pos.size
        else:  # NO side
            pnl = (exit_price - pos.entry_price) * pos.size

        # Close position
        tracker.close(pos.market_id, pnl, close_price=exit_price)

        # Feed outcome to learner
        learner.record_trade(
            strategy=pos.strategy,
            market_id=pos.market_id,
            token_id=pos.token_id,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            pnl=pnl,
            edge_predicted=0.0,  # We don't store this on the position
            category="",
            question=pos.question,
        )

        logger.info("Position resolved: %s %s | entry=%.4f exit=%.4f | P&L=$%.4f",
                     pos.strategy, pos.market_id[:12],
                     pos.entry_price, exit_price, pnl)


def _sleep(seconds: int):
    """Interruptible sleep that respects shutdown flag."""
    for _ in range(seconds):
        if _shutdown:
            return
        time.sleep(1)


if __name__ == "__main__":
    main()
