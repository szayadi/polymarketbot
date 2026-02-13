"""Strategy 2: Logic / Conditional Arbitrage.

Detects mispricings across related markets within the same event:
  A) Dutch-book: All YES prices in a multi-outcome event sum to != $1
  B) Conditional: Event A implies event B, but prices don't reflect it
"""

import logging
import re
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

MIN_EVENT_LIQUIDITY = 1000.0
MIN_ARB_PROFIT_PCT = 0.03     # 3% minimum profit after fees
FEE_ESTIMATE = 0.02           # Conservative 2% fee estimate


class ArbitrageStrategy(BaseStrategy):
    name = "arbitrage"

    def scan(self) -> list[dict]:
        """Scan events for arbitrage opportunities."""
        signals = []

        try:
            events = self.client.get_events(limit=50)
        except Exception as e:
            logger.error("[arbitrage] Failed to fetch events: %s", e)
            return signals

        logger.info("[arbitrage] Scanning %d events for arb opportunities",
                     len(events))

        for event in events:
            try:
                markets = event.get("markets", [])
                if not markets or len(markets) < 2:
                    continue

                # Strategy A: Dutch-book across multi-outcome markets
                dutch_signals = self._check_dutch_book(event, markets)
                signals.extend(dutch_signals)

                # Strategy B: Conditional/implication arbitrage
                cond_signals = self._check_conditional(event, markets)
                signals.extend(cond_signals)

            except Exception as e:
                logger.debug("[arbitrage] Error evaluating event: %s", e)

        signals.sort(key=lambda s: s["edge"], reverse=True)
        logger.info("[arbitrage] Found %d signals", len(signals))
        return signals

    def _check_dutch_book(self, event: dict, markets: list[dict]) -> list[dict]:
        """Check if YES prices across outcomes sum to != $1.

        If sum of all YES prices < 1.00: buy all YES → guaranteed $1 payout
        If sum of all YES prices > 1.00: buy all NO → guaranteed profit
        """
        signals = []

        # Each market in the event is one outcome
        yes_prices = []
        market_tokens = []  # (condition_id, yes_token, no_token, yes_price, question)

        for m in markets:
            outcome_prices = m.get("outcomePrices")
            if not outcome_prices or len(outcome_prices) < 2:
                continue

            try:
                yes_p = float(outcome_prices[0])
                no_p = float(outcome_prices[1])
            except (ValueError, TypeError):
                continue

            token_ids = m.get("clobTokenIds") or m.get("clob_token_ids", [])
            if len(token_ids) < 2:
                continue

            liquidity = float(m.get("liquidity", 0) or 0)
            if liquidity < MIN_EVENT_LIQUIDITY:
                continue

            yes_prices.append(yes_p)
            condition_id = m.get("condition_id", m.get("id", ""))
            market_tokens.append((
                condition_id,
                token_ids[0],   # YES token
                token_ids[1],   # NO token
                yes_p,
                m.get("question", ""),
            ))

        if len(yes_prices) < 2:
            return signals

        total_yes = sum(yes_prices)
        event_title = event.get("title", "Unknown event")

        # Case 1: Sum < 1.0 → buy all YES tokens for guaranteed profit
        if total_yes < (1.0 - MIN_ARB_PROFIT_PCT - FEE_ESTIMATE):
            profit = 1.0 - total_yes - FEE_ESTIMATE
            if profit >= MIN_ARB_PROFIT_PCT:
                # Generate buy signal for each YES token
                per_market_size = self.cfg.max_bet_size / len(market_tokens)
                for cid, yes_tok, _, yes_p, question in market_tokens:
                    # Verify ask price from order book
                    _, best_ask = self.client.get_best_bid_ask(yes_tok)
                    if best_ask is None:
                        continue
                    signals.append({
                        "token_id": yes_tok,
                        "side": "BUY",
                        "price": best_ask,
                        "size": per_market_size,
                        "market_id": cid,
                        "edge": profit,
                        "reason": f"Dutch book (YES sum={total_yes:.3f}<1): "
                                  f"buy YES@{best_ask:.4f} in [{event_title}]",
                        "question": question,
                    })

        # Case 2: Sum > 1.0 → buy all NO tokens for guaranteed profit
        elif total_yes > (1.0 + MIN_ARB_PROFIT_PCT + FEE_ESTIMATE):
            profit = total_yes - 1.0 - FEE_ESTIMATE
            if profit >= MIN_ARB_PROFIT_PCT:
                per_market_size = self.cfg.max_bet_size / len(market_tokens)
                for cid, _, no_tok, yes_p, question in market_tokens:
                    _, best_ask = self.client.get_best_bid_ask(no_tok)
                    if best_ask is None:
                        continue
                    signals.append({
                        "token_id": no_tok,
                        "side": "BUY",
                        "price": best_ask,
                        "size": per_market_size,
                        "market_id": cid,
                        "edge": profit,
                        "reason": f"Dutch book (YES sum={total_yes:.3f}>1): "
                                  f"buy NO@{best_ask:.4f} in [{event_title}]",
                        "question": question,
                    })

        return signals

    def _check_conditional(self, event: dict, markets: list[dict]) -> list[dict]:
        """Detect logical implications between markets and find mispricings.

        Patterns detected:
        - "X by more than Y" implies "X" (subset implies superset)
        - "X before date1" implies "X before date2" if date1 < date2
        - Stronger claim (higher threshold) implies weaker claim
        """
        signals = []
        parsed = []

        for m in markets:
            outcome_prices = m.get("outcomePrices")
            if not outcome_prices or len(outcome_prices) < 2:
                continue
            token_ids = m.get("clobTokenIds") or m.get("clob_token_ids", [])
            if len(token_ids) < 2:
                continue

            try:
                yes_p = float(outcome_prices[0])
            except (ValueError, TypeError):
                continue

            question = (m.get("question") or m.get("title") or "").lower()
            condition_id = m.get("condition_id", m.get("id", ""))
            liquidity = float(m.get("liquidity", 0) or 0)
            if liquidity < MIN_EVENT_LIQUIDITY:
                continue

            parsed.append({
                "question": question,
                "yes_price": yes_p,
                "yes_token": token_ids[0],
                "no_token": token_ids[1],
                "condition_id": condition_id,
                "original_question": m.get("question", m.get("title", "")),
            })

        # Check all pairs for implications
        for i, a in enumerate(parsed):
            for j, b in enumerate(parsed):
                if i == j:
                    continue

                # Does A imply B? (A is stronger, B is weaker)
                if self._implies(a["question"], b["question"]):
                    # If A implies B, then P(A) <= P(B) must hold
                    # If P(A) > P(B), there's a mispricing
                    if a["yes_price"] > b["yes_price"] + MIN_ARB_PROFIT_PCT + FEE_ESTIMATE:
                        edge = a["yes_price"] - b["yes_price"] - FEE_ESTIMATE

                        # Buy YES on B (underpriced) and/or sell YES on A (overpriced)
                        # Simpler: just buy the underpriced side
                        _, best_ask = self.client.get_best_bid_ask(b["yes_token"])
                        if best_ask is None:
                            continue

                        signals.append({
                            "token_id": b["yes_token"],
                            "side": "BUY",
                            "price": best_ask,
                            "size": self.cfg.max_bet_size,
                            "market_id": b["condition_id"],
                            "edge": edge,
                            "reason": (f"Conditional arb: '{a['original_question']}' "
                                       f"(YES@{a['yes_price']:.2f}) implies "
                                       f"'{b['original_question']}' "
                                       f"(YES@{b['yes_price']:.2f}) — B underpriced"),
                            "question": b["original_question"],
                        })

        return signals

    @staticmethod
    def _implies(question_a: str, question_b: str) -> bool:
        """Heuristic: does question_a logically imply question_b?

        Examples:
          "will X win by more than 10?" implies "will X win by more than 5?"
          "will X win by more than 10?" implies "will X win?"
          "will X happen before march?" implies "will X happen before june?"
        """
        # Pattern 1: "by more than N" — higher threshold implies lower
        pattern_more = r"by more than (\d+)"
        match_a = re.search(pattern_more, question_a)
        match_b = re.search(pattern_more, question_b)
        if match_a and match_b:
            # Same base question? (strip the number part)
            base_a = re.sub(pattern_more, "", question_a).strip()
            base_b = re.sub(pattern_more, "", question_b).strip()
            if base_a == base_b and int(match_a.group(1)) > int(match_b.group(1)):
                return True

        # Pattern 2: "more than N" implies the base question without threshold
        if match_a and not match_b:
            base_a = re.sub(pattern_more, "", question_a).strip().rstrip("?").strip()
            base_b = question_b.strip().rstrip("?").strip()
            # Check if they share enough words (same root question)
            words_a = set(base_a.split())
            words_b = set(base_b.split())
            overlap = len(words_a & words_b)
            if overlap >= min(len(words_a), len(words_b)) * 0.7:
                return True

        # Pattern 3: "before [month/date]" — earlier date implies later date
        months = ["january", "february", "march", "april", "may", "june",
                  "july", "august", "september", "october", "november", "december"]
        pattern_before = r"before (\w+)"
        match_a = re.search(pattern_before, question_a)
        match_b = re.search(pattern_before, question_b)
        if match_a and match_b:
            month_a = match_a.group(1).lower()
            month_b = match_b.group(1).lower()
            if month_a in months and month_b in months:
                base_a = re.sub(pattern_before, "", question_a).strip()
                base_b = re.sub(pattern_before, "", question_b).strip()
                if base_a == base_b and months.index(month_a) < months.index(month_b):
                    return True

        # Pattern 4: "over N%" / "over N points" — higher threshold implies lower
        pattern_over = r"over (\d+(?:\.\d+)?)"
        match_a = re.search(pattern_over, question_a)
        match_b = re.search(pattern_over, question_b)
        if match_a and match_b:
            base_a = re.sub(pattern_over, "", question_a).strip()
            base_b = re.sub(pattern_over, "", question_b).strip()
            if base_a == base_b and float(match_a.group(1)) > float(match_b.group(1)):
                return True

        return False
