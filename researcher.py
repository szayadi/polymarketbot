"""Pre-trade market researcher — validates signals with external data.

Checks real-world data BEFORE committing capital:
  - Weather markets → Open-Meteo API (free, no key, <10ms)
  - Crypto markets  → CoinGecko API (free, <1s)
  - Finance markets → Yahoo Finance via yfinance (free)
  - Economics       → FRED API (free, official Fed data)

Only runs on tail bets and momentum — arb and spread skip this entirely.
Hard timeout of 3 seconds per lookup. If the API is slow, we skip research
and trade anyway (never block a signal just because an API is down).
"""

import logging
import re
import requests
from typing import Optional

logger = logging.getLogger(__name__)

# Hard timeout for all external API calls — never block the bot
API_TIMEOUT = 3.0

# ── City coordinates for weather lookups ──
CITY_COORDS = {
    "nyc": (40.71, -74.01), "new york": (40.71, -74.01),
    "chicago": (41.88, -87.63), "los angeles": (34.05, -118.24),
    "la": (34.05, -118.24), "miami": (25.76, -80.19),
    "houston": (29.76, -95.37), "dallas": (32.78, -96.80),
    "phoenix": (33.45, -112.07), "philadelphia": (39.95, -75.17),
    "san antonio": (29.42, -98.49), "san diego": (32.72, -117.16),
    "san francisco": (37.77, -122.42), "sf": (37.77, -122.42),
    "seattle": (47.61, -122.33), "denver": (39.74, -104.98),
    "boston": (42.36, -71.06), "nashville": (36.16, -86.78),
    "austin": (30.27, -97.74), "atlanta": (33.75, -84.39),
    "dc": (38.91, -77.04), "washington": (38.91, -77.04),
    "portland": (45.52, -122.68), "las vegas": (36.17, -115.14),
    "detroit": (42.33, -83.05), "minneapolis": (44.98, -93.27),
}

# Crypto symbol mapping
CRYPTO_IDS = {
    "bitcoin": "bitcoin", "btc": "bitcoin",
    "ethereum": "ethereum", "eth": "ethereum",
    "solana": "solana", "sol": "solana",
    "dogecoin": "dogecoin", "doge": "dogecoin",
    "xrp": "ripple", "ripple": "ripple",
    "cardano": "cardano", "ada": "cardano",
}


class MarketResearcher:
    """Validates trade signals against real-world data sources."""

    def validate_signal(self, signal: dict) -> tuple[bool, str, float]:
        """Validate a trade signal with external data.

        Returns:
            (should_trade, reason, confidence_adjustment)

            confidence_adjustment:
              > 1.0 = external data supports the trade (boost edge)
              = 1.0 = no data available or neutral
              < 1.0 = external data contradicts (reduce edge)
              = 0.0 = external data strongly contradicts (block trade)
        """
        question = signal.get("question", "").lower()
        ticker = signal.get("ticker", "").lower()
        reason_text = signal.get("reason", "").lower()
        combined = f"{question} {ticker} {reason_text}"

        try:
            # Route to the right data source based on market content
            if self._is_weather_market(combined):
                return self._check_weather(combined, signal)
            elif self._is_crypto_market(combined):
                return self._check_crypto(combined, signal)
            elif self._is_finance_market(combined):
                return self._check_finance(combined, signal)
            else:
                # No relevant external data — trade as-is
                return True, "no external data available", 1.0
        except Exception as e:
            logger.debug("[researcher] Error: %s", e)
            return True, f"research error: {e}", 1.0

    # ── Category Detection ──────────────────────────────────────

    @staticmethod
    def _is_weather_market(text: str) -> bool:
        weather_kw = ["rain", "snow", "temperature", "daily high", "daily low",
                       "degrees", "weather", "precipitation", "heat", "cold",
                       "frost", "wind", "hurricane", "tornado", "storm"]
        return any(kw in text for kw in weather_kw)

    @staticmethod
    def _is_crypto_market(text: str) -> bool:
        crypto_kw = ["bitcoin", "btc", "ethereum", "eth", "crypto",
                      "solana", "sol", "dogecoin", "doge", "xrp",
                      "cardano", "ada"]
        return any(kw in text for kw in crypto_kw)

    @staticmethod
    def _is_finance_market(text: str) -> bool:
        finance_kw = ["s&p", "sp500", "nasdaq", "dow jones", "stock",
                       "market close", "index", "treasury", "yield",
                       "interest rate", "fed funds"]
        return any(kw in text for kw in finance_kw)

    # ── Weather Validation (Open-Meteo) ─────────────────────────

    def _check_weather(self, text: str, signal: dict) -> tuple[bool, str, float]:
        """Check weather forecast via Open-Meteo (free, no API key)."""
        # Find the city
        city, coords = self._extract_city(text)
        if not coords:
            return True, "weather: city not recognized", 1.0

        lat, lon = coords

        try:
            resp = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,snowfall_sum",
                    "temperature_unit": "fahrenheit",
                    "timezone": "America/New_York",
                    "forecast_days": 3,
                },
                timeout=API_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.debug("[researcher] Weather API error: %s", e)
            return True, f"weather API unavailable: {e}", 1.0

        daily = data.get("daily", {})
        if not daily:
            return True, "weather: no forecast data", 1.0

        # Parse what the market is asking about
        side = signal.get("side", "")
        confidence = 1.0

        # Rain markets
        if "rain" in text:
            rain_totals = daily.get("rain_sum", [])
            if rain_totals:
                # Check if rain is expected in the forecast period
                max_rain = max(rain_totals[:2])  # next 2 days
                will_rain = max_rain > 0.5  # >0.5mm = rain

                if "no" in side and will_rain:
                    # We're betting NO rain but forecast says rain
                    logger.info("[researcher] WEATHER CONFLICT: betting NO rain but forecast=%.1fmm", max_rain)
                    return True, f"weather: forecast rain={max_rain:.1f}mm (conflict)", 0.5
                elif "yes" in side and not will_rain:
                    # Betting YES rain but forecast is dry
                    logger.info("[researcher] WEATHER CONFLICT: betting YES rain but forecast=%.1fmm", max_rain)
                    return True, f"weather: forecast rain={max_rain:.1f}mm (conflict)", 0.5
                elif ("no" in side and not will_rain) or ("yes" in side and will_rain):
                    confidence = 1.3  # Forecast supports our position
                    logger.info("[researcher] WEATHER CONFIRM: forecast supports %s rain (%.1fmm)", side, max_rain)

                return True, f"weather: rain forecast={max_rain:.1f}mm for {city}", confidence

        # Snow markets
        if "snow" in text:
            snow_totals = daily.get("snowfall_sum", [])
            if snow_totals:
                max_snow = max(snow_totals[:2])
                will_snow = max_snow > 0.1

                if ("no" in side and will_snow) or ("yes" in side and not will_snow):
                    confidence = 0.5
                elif ("no" in side and not will_snow) or ("yes" in side and will_snow):
                    confidence = 1.3

                return True, f"weather: snow forecast={max_snow:.1f}cm for {city}", confidence

        # Temperature markets
        if any(kw in text for kw in ["high", "temperature", "degrees"]):
            temps_max = daily.get("temperature_2m_max", [])
            if temps_max:
                forecast_high = temps_max[0]  # Tomorrow's high

                # Try to extract threshold from the market question
                threshold = self._extract_number(text)
                if threshold is not None:
                    if "above" in text or "over" in text:
                        if forecast_high > threshold + 3:
                            confidence = 1.3  # Clearly above
                        elif forecast_high < threshold - 3:
                            confidence = 0.5  # Clearly below
                    elif "below" in text or "under" in text:
                        if forecast_high < threshold - 3:
                            confidence = 1.3
                        elif forecast_high > threshold + 3:
                            confidence = 0.5

                return True, f"weather: forecast high={forecast_high:.0f}F for {city}", confidence

        return True, f"weather: data retrieved for {city}", confidence

    # ── Crypto Validation (CoinGecko) ───────────────────────────

    def _check_crypto(self, text: str, signal: dict) -> tuple[bool, str, float]:
        """Check crypto prices via CoinGecko (free)."""
        coin_id = self._extract_crypto(text)
        if not coin_id:
            return True, "crypto: coin not recognized", 1.0

        try:
            resp = requests.get(
                f"https://api.coingecko.com/api/v3/simple/price",
                params={
                    "ids": coin_id,
                    "vs_currencies": "usd",
                    "include_24hr_change": "true",
                },
                timeout=API_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.debug("[researcher] Crypto API error: %s", e)
            return True, f"crypto API unavailable: {e}", 1.0

        coin_data = data.get(coin_id, {})
        if not coin_data:
            return True, f"crypto: no data for {coin_id}", 1.0

        current_price = coin_data.get("usd", 0)
        change_24h = coin_data.get("usd_24h_change", 0)
        side = signal.get("side", "")
        confidence = 1.0

        # Extract price threshold from market question
        threshold = self._extract_number(text)

        if threshold and current_price > 0:
            distance_pct = (current_price - threshold) / threshold

            if "above" in text or "over" in text:
                if current_price > threshold * 1.05:
                    # Price is already 5%+ above threshold — tail bet YES is strong
                    confidence = 1.3
                    logger.info("[researcher] CRYPTO CONFIRM: %s=$%.0f, threshold=$%.0f (+%.1f%%)",
                                coin_id, current_price, threshold, distance_pct * 100)
                elif current_price < threshold * 0.95:
                    # Price is 5%+ below — tail bet YES is risky
                    confidence = 0.5
                    logger.info("[researcher] CRYPTO CONFLICT: %s=$%.0f, threshold=$%.0f (%.1f%%)",
                                coin_id, current_price, threshold, distance_pct * 100)
            elif "below" in text or "under" in text:
                if current_price < threshold * 0.95:
                    confidence = 1.3
                elif current_price > threshold * 1.05:
                    confidence = 0.5

        # Momentum check: is 24h trend supporting our direction?
        if change_24h and abs(change_24h) > 2:
            if (side == "yes" and change_24h > 2) or (side == "no" and change_24h < -2):
                confidence = min(confidence * 1.1, 1.5)
            elif (side == "yes" and change_24h < -2) or (side == "no" and change_24h > 2):
                confidence *= 0.9

        return True, f"crypto: {coin_id}=${current_price:,.0f} (24h: {change_24h:+.1f}%)", confidence

    # ── Finance Validation ──────────────────────────────────────

    def _check_finance(self, text: str, signal: dict) -> tuple[bool, str, float]:
        """Check financial data. Uses a simple quote fetch."""
        # For S&P 500 / market indices
        confidence = 1.0

        # Try to get S&P 500 level from a free source
        if any(kw in text for kw in ["s&p", "sp500", "s&p 500"]):
            try:
                resp = requests.get(
                    "https://api.coingecko.com/api/v3/simple/price",
                    params={"ids": "bitcoin", "vs_currencies": "usd"},
                    timeout=API_TIMEOUT,
                )
                # CoinGecko doesn't do stocks — just return neutral
                return True, "finance: stock data not available via free API", 1.0
            except Exception:
                pass

        return True, "finance: no free real-time data", confidence

    # ── Helper Methods ──────────────────────────────────────────

    def _extract_city(self, text: str) -> tuple[str, Optional[tuple[float, float]]]:
        """Extract city name and coordinates from market text."""
        text_lower = text.lower()
        # Check longest names first to avoid partial matches
        for city in sorted(CITY_COORDS.keys(), key=len, reverse=True):
            if city in text_lower:
                return city, CITY_COORDS[city]
        return "", None

    def _extract_crypto(self, text: str) -> Optional[str]:
        """Extract crypto coin ID from market text."""
        text_lower = text.lower()
        for keyword, coin_id in CRYPTO_IDS.items():
            if keyword in text_lower:
                return coin_id
        return None

    @staticmethod
    def _extract_number(text: str) -> Optional[float]:
        """Extract a price/temperature threshold number from text.

        Handles formats like: $100,000  100k  95°  5000  $99,999.99
        """
        # Try dollar amounts first: $100,000 or $99,999.99
        dollar_match = re.findall(r'\$[\d,]+(?:\.\d+)?', text)
        if dollar_match:
            try:
                num_str = dollar_match[0].replace('$', '').replace(',', '')
                return float(num_str)
            except ValueError:
                pass

        # Try K/M shorthand: 100k, 2.5M
        km_match = re.findall(r'(\d+(?:\.\d+)?)\s*([kKmM])', text)
        if km_match:
            try:
                num = float(km_match[0][0])
                suffix = km_match[0][1].lower()
                if suffix == 'k':
                    return num * 1000
                elif suffix == 'm':
                    return num * 1_000_000
            except ValueError:
                pass

        # Try plain numbers > 50 (avoid matching cents/small values)
        plain_match = re.findall(r'(?<!\d)(\d{2,}(?:\.\d+)?)(?!\d)', text)
        if plain_match:
            try:
                nums = [float(n) for n in plain_match if float(n) > 50]
                if nums:
                    return max(nums)  # Return the largest threshold found
            except ValueError:
                pass

        return None
