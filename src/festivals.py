"""
Festival calendar feature. Diwali is, by a wide margin, the single most
documented driver of short-term AQI spikes in North India (firecracker
use spikes PM2.5 dramatically for several days around it) - a far
stronger, more direct signal than trying to get month/day-of-year cyclic
features to indirectly capture it.

Diwali follows the Hindu lunisolar calendar, so its Gregorian date shifts
unpredictably each year (it is NOT a fixed offset like "always October
20th") - it must be looked up per year, not computed or guessed. The
dates below were verified via web search against multiple consistent
sources (drikpanchang.com, publicholidays.com, and others) as of
September 2026.

Extend DIWALI_DATES with new years as the project's live history grows
past 2028 - the feature degrades gracefully (falls back to the nearest
known year) but accuracy for future years depends on this table being
kept current.

Other regionally-relevant festivals (Chhath Puja, New Year's Eve
fireworks) likely have a similar effect and would be reasonable
additions later - not included yet because their dates weren't
separately verified for this project.
"""

import pandas as pd

DIWALI_DATES = {
    2024: "2024-10-31",
    2025: "2025-10-20",
    2026: "2026-11-08",
    2027: "2027-10-29",
    2028: "2028-10-17",
}

# Firecracker-driven AQI spikes in Delhi are well documented to persist
# for several days after Diwali night itself (smog builds up and lingers
# in still winter air), not just on the day - so a window, not a single
# day flag, is more representative of the actual pollution pattern.
DIWALI_WINDOW_DAYS = 3

_DIWALI_TIMESTAMPS = [pd.Timestamp(d) for d in DIWALI_DATES.values()]


def days_to_nearest_diwali(ts: pd.Timestamp) -> float:
    """Absolute days between `ts` and the nearest known Diwali date.
    Degrades gracefully (but less accurately) outside the years covered
    by DIWALI_DATES - see module docstring."""
    ts = pd.Timestamp(ts).normalize()
    return min(abs((ts - d).days) for d in _DIWALI_TIMESTAMPS)


def add_festival_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `days_to_diwali` (continuous proximity) and `is_diwali_window`
    (binary, +/- DIWALI_WINDOW_DAYS) to a dataframe with a `timestamp`
    column."""
    df = df.copy()
    df["days_to_diwali"] = df["timestamp"].apply(days_to_nearest_diwali)
    df["is_diwali_window"] = (df["days_to_diwali"] <= DIWALI_WINDOW_DAYS).astype(int)
    return df
