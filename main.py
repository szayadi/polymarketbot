#!/usr/bin/env python3
"""Kalshi Trading Bot — Aggressive Fast-Resolution Trader.

Four strategies optimized for DAILY profits on fast-resolving markets:
  1. Dutch Book Arbitrage — capture even 0.5% arbs (lowered from 2%)
  2. Tail Bets — widened to 85/15 bands for 5-10x more opportunities
  3. Spread Capture — tighter spreads, 60s stale timeout, wider price range
  4. Momentum — 8% profit target, 5% stop loss, trailing stop, urgency boost

DEFAULT: Only markets resolving within 3 DAYS (not 30).
Polls every 10 seconds. Deploys 85% of capital (15% reserve).
15% max bet per trade. 30% max per market. 25% daily loss cap.

SURVIVAL DIRECTIVE:
  The bot's #1 priority is staying alive. If the balance ever drops below
  the survival floor ($1.00 default), ALL trading halts permanently.
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone

from config import Config
from client import KalshiClient
from learner import AdaptiveLearner
from positions import PositionTracker
from researcher import MarketResearcher
from risk import RiskManager
from strategies.dutch_book import DutchBookStrategy
from strategies.tail_bets import TailBetsStrategy
from strategies.spread import SpreadStrategy
from strategies.momentum import MomentumStrategy

# ── Logging ──────────────────────────────────────────────────────
# File handler gets everything (verbose, for debugging)
# Console gets ONLY our clean formatted output via print()

file_handler = logging.FileHandler("bot.log", mode="a")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))

# Console handler — only WARNING+ from sub-modules (errors/criticals)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter(
    "  !! %(message)s",
))

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[console_handler, file_handler],
)

# Our main logger — set to INFO so we can still log to file
logger = logging.getLogger("kalshi-bot")
logger.setLevel(logging.INFO)

_shutdown = False

# ── Clean Console Output ──────────────────────────────────────────

DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
WHITE = "\033[37m"
RESET = "\033[0m"
CLEAR_LINE = "\033[2K"


def _print(msg: str = ""):
    """Print to console (bypasses logging noise)."""
    print(msg, flush=True)


def _color_pnl(cents: int) -> str:
    """Color-code a P&L value."""
    dollars = cents / 100.0
    if cents > 0:
        return f"{GREEN}+${dollars:.2f}{RESET}"
    elif cents < 0:
        return f"{RED}-${abs(dollars):.2f}{RESET}"
    return f"$0.00"


def _color_pct(pct: float, invert: bool = False) -> str:
    """Color-code a percentage."""
    if invert:
        pct = -pct
    if pct > 0:
        return f"{GREEN}{pct:.1f}%{RESET}"
    elif pct < 0:
        return f"{RED}{pct:.1f}%{RESET}"
    return f"{pct:.1f}%"


def _print_banner(cfg: Config):
    """Print startup banner."""
    mode = f"{RED}LIVE{RESET}" if not cfg.dry_run else f"{YELLOW}DRY RUN{RESET}"
    env_label = f"{RED}PRODUCTION{RESET}" if cfg.env == "production" else f"{CYAN}DEMO{RESET}"

    _print()
    _print(f"{BOLD}{'=' * 58}{RESET}")
    _print(f"{BOLD}  KALSHI TRADING BOT v3{RESET}  {DIM}aggressive mode{RESET}")
    _print(f"{'=' * 58}")
    _print()
    _print(f"  {DIM}Environment{RESET}  {env_label}    {DIM}Mode{RESET}  {mode}")
    _print(f"  {DIM}Bankroll{RESET}     ${cfg.bankroll:.2f}    {DIM}Max bet{RESET}  ${cfg.max_bet_size:.2f}")
    _print(f"  {DIM}Min edge{RESET}     {cfg.min_edge*100:.1f}%       {DIM}Reserve{RESET}  {cfg.cash_reserve_pct*100:.0f}%")
    _print(f"  {DIM}Resolve{RESET}      {cfg.max_days_to_resolve}d max     {DIM}Poll{RESET}     {cfg.poll_interval}s")
    _print()

    strats = []
    if cfg.strategy_dutch_book:
        strats.append("Dutch Book")
    if cfg.strategy_tail_bets:
        strats.append("Tail Bets")
    if cfg.strategy_spread:
        strats.append("Spread")
    if cfg.strategy_momentum:
        strats.append("Momentum")
    _print(f"  {DIM}Strategies{RESET}   {', '.join(strats)}")

    features = []
    if cfg.enable_research:
        features.append("Research")
    features.append("Learner")
    _print(f"  {DIM}Features{RESET}     {', '.join(features)}")
    _print()
    _print(f"{'=' * 58}")
    _print()


def _print_cycle_header(cycle: int, balance: float, growth: float,
                        drawdown: float, daily_pnl_cents: int):
    """Print compact cycle header."""
    now = datetime.now().strftime("%H:%M:%S")
    pnl_str = _color_pnl(daily_pnl_cents)
    dd_str = _color_pct(-drawdown * 100) if drawdown > 0.01 else f"{DIM}0%{RESET}"

    _print(f"{DIM}{'-' * 58}{RESET}")
    _print(f"  {BOLD}Cycle {cycle}{RESET}  {DIM}{now}{RESET}"
           f"    {DIM}Balance{RESET} ${balance:.2f}"
           f"    {DIM}Daily{RESET} {pnl_str}"
           f"    {DIM}DD{RESET} {dd_str}")


def _print_scan_results(strategy_results: list[tuple[str, int, int]]):
    """Print strategy scan summary — one line per strategy."""
    for name, signals, trades in strategy_results:
        if signals == 0 and trades == 0:
            continue
        trade_str = f"{GREEN}{trades} traded{RESET}" if trades > 0 else f"{DIM}0 traded{RESET}"
        _print(f"  {DIM}|{RESET} {name:<12} {signals:>3} signals  {trade_str}")


def _print_trade(strategy: str, action: str, side: str, ticker: str,
                 count: int, price: int, edge: float, reason: str):
    """Print a single trade execution."""
    side_color = GREEN if side == "yes" else CYAN
    cost = count * price
    _print(f"  {BOLD}>>>{RESET} {strategy:<12} {action.upper()} "
           f"{side_color}{side.upper()}{RESET} {ticker} "
           f"x{count} @{price}c "
           f"{DIM}(${cost/100:.2f}, edge {edge:.1f}%){RESET}")


def _print_research(strategy: str, reason: str, confidence: float,
                    old_edge: float, new_edge: float):
    """Print research adjustment."""
    if confidence > 1.0:
        conf_str = f"{GREEN}+{(confidence-1)*100:.0f}%{RESET}"
    else:
        conf_str = f"{RED}{(confidence-1)*100:.0f}%{RESET}"
    _print(f"  {DIM}|{RESET} {strategy:<12} Research: {reason} [{conf_str}]")


def _print_exit(reason: str):
    """Print momentum exit signal."""
    if "PROFIT" in reason:
        _print(f"  {GREEN}$$${RESET}  {reason}")
    elif "TRAILING" in reason:
        _print(f"  {YELLOW}~~~{RESET}  {reason}")
    else:
        _print(f"  {RED}xxx{RESET}  {reason}")


def _print_settlement(strategy: str, ticker: str, pnl_cents: int,
                      hold_hours: float):
    """Print a settlement."""
    pnl = _color_pnl(pnl_cents)
    _print(f"  {BOLD}SETTLED{RESET}  {strategy:<12} {ticker}  {pnl}  {DIM}({hold_hours:.1f}h){RESET}")


def _print_portfolio(tracker: PositionTracker):
    """Print portfolio summary line."""
    open_pos = tracker.get_open()
    exposure = tracker.get_exposure_dollars()
    unrealized = tracker.get_total_unrealized_pnl_cents()

    if not open_pos:
        _print(f"  {DIM}Portfolio: no open positions{RESET}")
        return

    _print(f"  {DIM}Portfolio:{RESET} {len(open_pos)} open"
           f"  {DIM}Exposure{RESET} ${exposure:.2f}"
           f"  {DIM}Unrealized{RESET} {_color_pnl(unrealized)}")


def _print_learner_report(learner: AdaptiveLearner):
    """Print clean learner stats."""
    if not learner.trades:
        return

    total_pnl = sum(t.pnl_cents for t in learner.trades)
    _print()
    _print(f"  {BOLD}Learning Report{RESET}  {DIM}({len(learner.trades)} trades, "
           f"total {_color_pnl(total_pnl)}{DIM}){RESET}")
    _print()

    for name, stats in learner.strategy_stats.items():
        em = learner.edge_multipliers.get(name, 1.0)
        sm = learner.get_size_multiplier(name)
        wr_color = GREEN if stats.win_rate >= 0.5 else RED
        streak = stats.current_streak
        streak_str = f"{GREEN}+{streak}{RESET}" if streak > 0 else f"{RED}{streak}{RESET}" if streak < 0 else "0"

        _print(f"  {name:<12}  "
               f"W {stats.wins:>2} / L {stats.losses:>2}  "
               f"{wr_color}{stats.win_rate:>4.0%}{RESET}  "
               f"PnL {_color_pnl(stats.total_pnl_cents)}  "
               f"streak {streak_str}  "
               f"{DIM}edge:{em:.1f}x size:{sm:.1f}x{RESET}")

    if learner.category_blacklist:
        _print(f"  {DIM}Blacklisted: {', '.join(learner.category_blacklist)}{RESET}")
    _print()


# ── Signal handler ────────────────────────────────────────────────

def _handle_shutdown(signum, frame):
    global _shutdown
    logger.info("Shutdown signal received")
    _print(f"\n  {YELLOW}Shutting down...{RESET}")
    _shutdown = True


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kalshi Trading Bot")
    parser.add_argument("--reset", action="store_true",
                        help="Clear all open positions and start fresh")
    args = parser.parse_args()

    cfg = Config()
    _print_banner(cfg)

    if not cfg.api_key_id and not cfg.dry_run:
        _print(f"  {RED}ERROR: KALSHI_API_KEY_ID required for live trading{RESET}")
        sys.exit(1)

    # ── Initialize ───────────────────────────────────────────────
    client = KalshiClient(cfg)
    tracker = PositionTracker()
    learner = AdaptiveLearner()

    # Handle --reset flag
    if args.reset:
        count = tracker.reset()
        _print(f"  {YELLOW}RESET: Closed {count} open positions. Starting fresh.{RESET}")

    # Auto-cleanup: close positions older than 14 days (definitely stale)
    stale_closed = tracker.clear_stale(max_age_days=14)
    if stale_closed:
        _print(f"  {YELLOW}Cleaned {stale_closed} stale positions (>14 days old){RESET}")
    risk = RiskManager(cfg, tracker)

    # Check exchange status
    exchange = client.get_exchange_status()
    if exchange:
        logger.info("Exchange status: %s", exchange)

    # Sync balance from Kalshi
    live_balance = client.get_balance_dollars()
    if live_balance is not None:
        _print(f"  {DIM}Live balance:{RESET} ${live_balance:.2f}")
        cfg.bankroll = live_balance
        risk.update_balance(live_balance)
    else:
        _print(f"  {DIM}Using config balance:{RESET} ${cfg.bankroll:.2f}")

    if learner.trades:
        _print_learner_report(learner)

    # Initialize researcher for external data validation
    researcher = MarketResearcher() if cfg.enable_research else None

    # Build strategy list — priority order
    strategies = []
    momentum_strategy = None

    if cfg.strategy_dutch_book:
        strategies.append(DutchBookStrategy(client, cfg, risk, tracker, learner, researcher))
    if cfg.strategy_tail_bets:
        strategies.append(TailBetsStrategy(client, cfg, risk, tracker, learner, researcher))
    if cfg.strategy_spread:
        strategies.append(SpreadStrategy(client, cfg, risk, tracker, learner, researcher))
    if cfg.strategy_momentum:
        momentum_strategy = MomentumStrategy(client, cfg, risk, tracker, learner, researcher)
        strategies.append(momentum_strategy)

    if not strategies:
        _print(f"  {RED}ERROR: No strategies enabled{RESET}")
        sys.exit(1)

    _print(f"  {DIM}Ready. Polling every {cfg.poll_interval}s...{RESET}")
    _print()

    # ── Signal handlers ──────────────────────────────────────────
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    # ── Main loop ────────────────────────────────────────────────
    cycle = 0
    REPORT_EVERY = 10

    while not _shutdown:
        cycle += 1

        # Sync balance every cycle for survival checks
        balance = client.get_balance_dollars()
        if balance is not None:
            cfg.bankroll = balance
            risk.update_balance(balance)

        drawdown = risk.get_drawdown_pct()
        growth = risk.get_growth_factor()
        daily_pnl = tracker.get_daily_pnl_cents()

        _print_cycle_header(cycle, cfg.bankroll, growth, drawdown, daily_pnl)

        # Halt check
        halted, reason = risk.is_halted()
        if halted:
            if "SURVIVAL" in reason:
                _print(f"  {RED}{BOLD}DEAD — balance hit survival floor. Exiting.{RESET}")
                break
            _print(f"  {YELLOW}HALTED: {reason}{RESET}")
            _sleep(cfg.poll_interval * 4)
            continue

        # Check momentum exit signals
        if momentum_strategy:
            exits = momentum_strategy.check_exits()
            for exit_sig in exits:
                _print_exit(exit_sig["reason"])

        # Run strategies in priority order
        strategy_results = []
        cycle_trades = []

        for strategy in strategies:
            try:
                signals = strategy.scan()
                results = strategy.execute(signals) if signals else []
                strategy_results.append((strategy.name, len(signals), len(results)))

                # Collect trades for display
                for r in results:
                    sig = r["signal"]
                    cycle_trades.append((
                        strategy.name, sig["action"], sig["side"],
                        sig["ticker"], sig.get("count", 1),
                        sig["price_cents"], sig["edge"], sig["reason"],
                    ))
            except Exception as e:
                logger.error("[%s] Error: %s", strategy.name, e, exc_info=True)
                strategy_results.append((strategy.name, 0, 0))

        # Print scan summary (only if there were signals)
        total_signals = sum(s for _, s, _ in strategy_results)
        total_trades = sum(t for _, _, t in strategy_results)

        if total_signals > 0:
            _print_scan_results(strategy_results)

        # Print each trade
        for trade in cycle_trades:
            _print_trade(*trade)

        # Cleanup stale orders
        for s in strategies:
            if hasattr(s, "cleanup_stale_orders"):
                s.cleanup_stale_orders()

        # Check settlements and feed to learner
        _check_settlements(client, tracker, learner)

        # Update prices
        _update_prices(client, tracker)

        # Portfolio summary
        _print_portfolio(tracker)

        # Periodic learner report
        if cycle % REPORT_EVERY == 0 and learner.trades:
            _print_learner_report(learner)

        # Log to file (not console)
        logger.info("Cycle %d: %d signals -> %d trades | balance=$%.2f | daily=%dc",
                     cycle, total_signals, total_trades, cfg.bankroll, daily_pnl)

        # Save
        tracker.save()
        learner.save()

        _sleep(cfg.poll_interval)

    # ── Shutdown ─────────────────────────────────────────────────
    if not cfg.dry_run:
        client.cancel_all_orders()
    tracker.save()
    learner.save()

    _print()
    if learner.trades:
        _print_learner_report(learner)

    _print(f"  {DIM}Bot stopped.{RESET}")
    _print()


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

            _print_settlement(pos.strategy, pos.ticker, pnl_cents, hold_hours)

            logger.info("SETTLED: %s %s | entry=%dc exit=%dc | PnL=%dc | hold=%.1fh",
                         pos.strategy, pos.ticker, pos.entry_price_cents,
                         exit_price, pnl_cents, hold_hours)
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
