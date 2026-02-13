#!/usr/bin/env python3
"""Kalshi Trading Bot — Self-Preserving Automated Trader.

Four strategies focused on fast, profitable trades:
  1. Dutch Book Arbitrage — guaranteed profit on mispriced multi-outcome events
  2. Tail Bets — buy near-certain outcomes for reliable gains
  3. Spread Capture — place maker orders on both sides of wide spreads
  4. Momentum — ride price moves on high-volume, fast-resolving markets

All strategies filter to SHORT-TERM markets only (default 30 days max).
The bot learns from every outcome and adapts its behavior.

SURVIVAL DIRECTIVE:
  The bot's #1 priority is staying alive. If the balance ever drops below
  the survival floor ($1.00 default), ALL trading halts permanently.
  It grows aggressively when winning, shrinks defensively when losing.
"""

import logging
import signal
import sys
import time
from datetime import datetime, timezone

from config import Config
from client import KalshiClient
from learner import AdaptiveLearner
from positions import PositionTracker
from risk import RiskManager
from strategies.dutch_book import DutchBookStrategy
from strategies.tail_bets import TailBetsStrategy
from strategies.spread import SpreadStrategy
from strategies.momentum import MomentumStrategy

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
logger = logging.getLogger("kalshi-bot")

_shutdown = False


def _handle_shutdown(signum, frame):
    global _shutdown
    logger.info("Shutdown signal received")
    _shutdown = True


def main():
    cfg = Config()

    # ── Banner ───────────────────────────────────────────────────
    mode = "DRY RUN" if cfg.dry_run else "LIVE"
    env_label = cfg.env.upper()
    logger.info("=" * 60)
    logger.info("  Kalshi Trading Bot v2")
    logger.info("  Environment: %s | Mode: %s", env_label, mode)
    logger.info("  Bankroll: $%.2f | Max bet: $%.2f", cfg.bankroll, cfg.max_bet_size)
    logger.info("  Min edge: %.1f%% | Survival floor: $%.2f",
                cfg.min_edge * 100, cfg.survival_floor)
    logger.info("  Max resolve time: %d days | Min 24h volume: %d",
                cfg.max_days_to_resolve, cfg.min_volume_24h)
    logger.info("  Strategies: DUTCH=%s  TAIL=%s  SPREAD=%s  MOMENTUM=%s",
                cfg.strategy_dutch_book, cfg.strategy_tail_bets,
                cfg.strategy_spread, cfg.strategy_momentum)
    logger.info("  Adaptive learning: ENABLED (fast decay)")
    logger.info("=" * 60)

    if not cfg.api_key_id and not cfg.dry_run:
        logger.error("KALSHI_API_KEY_ID required for live trading")
        sys.exit(1)

    # ── Initialize ───────────────────────────────────────────────
    client = KalshiClient(cfg)
    tracker = PositionTracker()
    learner = AdaptiveLearner()
    risk = RiskManager(cfg, tracker)

    # Check exchange status
    exchange = client.get_exchange_status()
    if exchange:
        logger.info("Exchange status: %s", exchange)

    # Sync balance from Kalshi
    live_balance = client.get_balance_dollars()
    if live_balance is not None:
        logger.info("Live balance: $%.2f", live_balance)
        cfg.bankroll = live_balance
        risk.update_balance(live_balance)
    else:
        logger.info("Could not fetch balance (using config: $%.2f)", cfg.bankroll)

    if learner.trades:
        logger.info("Learner: %d historical trades", len(learner.trades))
        logger.info("\n%s", learner.get_report())

    # Build strategy list — priority order
    strategies = []
    momentum_strategy = None

    if cfg.strategy_dutch_book:
        strategies.append(DutchBookStrategy(client, cfg, risk, tracker, learner))
    if cfg.strategy_tail_bets:
        strategies.append(TailBetsStrategy(client, cfg, risk, tracker, learner))
    if cfg.strategy_spread:
        strategies.append(SpreadStrategy(client, cfg, risk, tracker, learner))
    if cfg.strategy_momentum:
        momentum_strategy = MomentumStrategy(client, cfg, risk, tracker, learner)
        strategies.append(momentum_strategy)

    if not strategies:
        logger.error("No strategies enabled")
        sys.exit(1)

    logger.info("Loaded %d strategies: %s",
                len(strategies), [s.name for s in strategies])

    # ── Signal handlers ──────────────────────────────────────────
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    # ── Main loop ────────────────────────────────────────────────
    cycle = 0
    REPORT_EVERY = 10  # Report more frequently

    while not _shutdown:
        cycle += 1
        logger.info("--- Cycle %d ---", cycle)

        # Sync balance every cycle for survival checks
        balance = client.get_balance_dollars()
        if balance is not None:
            cfg.bankroll = balance
            risk.update_balance(balance)
            drawdown = risk.get_drawdown_pct()
            growth = risk.get_growth_factor()
            logger.info("Balance: $%.2f | Growth: %.2fx | Drawdown: %.1f%%",
                        balance, growth, drawdown * 100)

        # Halt check
        halted, reason = risk.is_halted()
        if halted:
            logger.warning("HALTED: %s", reason)
            if "SURVIVAL" in reason:
                logger.critical("Bot is dead. Balance at survival floor. Exiting.")
                break
            _sleep(cfg.poll_interval * 4)
            continue

        # Check momentum exit signals (profit target / stop loss)
        if momentum_strategy:
            exits = momentum_strategy.check_exits()
            for exit_sig in exits:
                logger.info("EXIT SIGNAL: %s", exit_sig["reason"])

        # Run strategies in priority order
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

        # Cleanup stale orders
        for s in strategies:
            if hasattr(s, "cleanup_stale_orders"):
                s.cleanup_stale_orders()

        # Check settlements and feed to learner
        _check_settlements(client, tracker, learner)

        # Update prices
        _update_prices(client, tracker)

        # Summary
        logger.info("Cycle %d: %d signals -> %d trades", cycle,
                     total_signals, total_trades)
        logger.info("Portfolio: %s", tracker.summary())

        if cycle % REPORT_EVERY == 0 and learner.trades:
            logger.info("\n%s", learner.get_report())

        # Save
        tracker.save()
        learner.save()

        _sleep(cfg.poll_interval)

    # ── Shutdown ─────────────────────────────────────────────────
    logger.info("Shutting down...")
    if not cfg.dry_run:
        client.cancel_all_orders()
    tracker.save()
    learner.save()
    if learner.trades:
        logger.info("\n%s", learner.get_report())
    logger.info("Bot stopped.")


def _check_settlements(client: KalshiClient, tracker: PositionTracker,
                        learner: AdaptiveLearner):
    """Check for newly settled markets and feed outcomes to learner."""
    settlements = client.get_settlements(limit=50)
    if not settlements:
        return

    settled_tickers = {s.get("ticker") for s in settlements}

    for pos in tracker.get_open():
        if pos.ticker not in settled_tickers:
            continue

        for s in settlements:
            if s.get("ticker") != pos.ticker:
                continue

            revenue_cents = int(s.get("revenue", 0))
            cost_cents = pos.cost_cents
            pnl_cents = revenue_cents - cost_cents

            result = s.get("market_result", "")
            if pos.side == "yes":
                exit_price = 100 if result == "yes" else 0
            else:
                exit_price = 100 if result == "no" else 0

            # Calculate hold time
            hold_hours = 0.0
            try:
                entry_time = datetime.fromisoformat(pos.timestamp)
                hold_hours = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600.0
            except (ValueError, TypeError):
                pass

            tracker.close(pos.ticker, pnl_cents, exit_price)

            learner.record_trade(
                strategy=pos.strategy,
                ticker=pos.ticker,
                event_ticker=pos.event_ticker,
                side=pos.side,
                entry_price_cents=pos.entry_price_cents,
                exit_price_cents=exit_price,
                count=pos.count,
                pnl_cents=pnl_cents,
                edge_predicted=pos.edge_predicted,
                category="",
                question=pos.question,
                hold_time_hours=hold_hours,
            )

            logger.info("SETTLED: %s %s | entry=%dc exit=%dc | PnL=%dc ($%.2f) | hold=%.1fh",
                         pos.strategy, pos.ticker, pos.entry_price_cents,
                         exit_price, pnl_cents, pnl_cents / 100.0, hold_hours)
            break


def _update_prices(client: KalshiClient, tracker: PositionTracker):
    """Refresh current prices for open positions."""
    for pos in tracker.get_open():
        try:
            market = client.get_market(pos.ticker)
            if market:
                if pos.side == "yes":
                    price = int(market.get("yes_bid", 0) or 0)
                else:
                    price = int(market.get("no_bid", 0) or 0)
                if price > 0:
                    tracker.update_price(pos.ticker, price)
        except Exception:
            pass


def _sleep(seconds: int):
    for _ in range(seconds):
        if _shutdown:
            return
        time.sleep(1)


if __name__ == "__main__":
    main()
