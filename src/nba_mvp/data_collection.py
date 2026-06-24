"""Live nba_api pulls for per-game traditional + advanced player stats.

Hits stats.nba.com, so this is only used by scripts/refresh_predictions.py and the
notebook's optional "refresh from live data" cell — never by the Streamlit app.
"""

import datetime as dt
import time

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

ADVANCED_COLUMNS = [
    "PLAYER_ID", "OFF_RATING", "DEF_RATING", "NET_RATING",
    "AST_PCT", "REB_PCT", "TS_PCT", "USG_PCT", "PIE",
]


def season_string(year: int) -> str:
    """Convert 2000 -> "2000-01"."""
    return f"{year}-{str(year + 1)[-2:]}"


def current_season_year(today: dt.date | None = None) -> int:
    """NBA seasons start in October, so before October the "current" season started last year."""
    today = today or dt.date.today()
    return today.year if today.month >= 10 else today.year - 1


def collect_single_season(year: int) -> pd.DataFrame | None:
    """Download and merge per-game traditional + advanced stats for one season."""
    season = season_string(year)
    print(f"Collecting {season}...")
    try:
        df_trad = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season, per_mode_detailed="PerGame", season_type_all_star="Regular Season"
        ).get_data_frames()[0]
        time.sleep(0.6)

        df_adv = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season, per_mode_detailed="PerGame", season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
        ).get_data_frames()[0]

        df = pd.merge(df_trad, df_adv[ADVANCED_COLUMNS], on="PLAYER_ID", how="left")
        df["SEASON"] = season
        df["SEASON_YEAR"] = year
        print(f"  done: {len(df)} players")
        return df
    except Exception as e:
        print(f"  error collecting {season}: {e}")
        return None


def collect_seasons(start_year: int, end_year: int, pause: float = 1.0) -> pd.DataFrame:
    """Collect and concatenate per-game stats for every season from start_year to end_year (inclusive)."""
    all_seasons = []
    for year in range(start_year, end_year + 1):
        df_season = collect_single_season(year)
        if df_season is not None:
            all_seasons.append(df_season)
        time.sleep(pause)
    return pd.concat(all_seasons, ignore_index=True)
