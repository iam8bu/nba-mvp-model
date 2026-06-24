"""Merging in MVP voting shares and engineering box-score-sum + per-season Z-score features."""

import pandas as pd

Z_STATS = [
    "GP", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV",
    "FG_PCT", "FG3_PCT", "FT_PCT", "TS_PCT",
    "OFF_RATING", "DEF_RATING", "NET_RATING",
    "AST_PCT", "REB_PCT", "USG_PCT", "PIE",
    "PTS_REB_AST",
]

FEATURES = [
    "GP", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV",
    "FG_PCT", "FG3_PCT", "FT_PCT", "TS_PCT",
    "OFF_RATING", "DEF_RATING", "NET_RATING",
    "AST_PCT", "REB_PCT", "USG_PCT", "PIE",
    "PTS_REB_AST", "W_PCT",
    "PTS_Z", "REB_Z", "AST_Z", "STL_Z", "BLK_Z", "NET_RATING_Z", "PIE_Z", "USG_PCT_Z",
    "GP_Z", "MIN_Z", "TOV_Z",
    "FG_PCT_Z", "FG3_PCT_Z", "FT_PCT_Z", "TS_PCT_Z",
    "OFF_RATING_Z", "DEF_RATING_Z",
    "AST_PCT_Z", "REB_PCT_Z",
    "PTS_REB_AST_Z",
]

# Features with importance > 1% in the full-feature random forest (see modeling.select_important_features).
FINAL_FEATURES = [
    "PTS_REB_AST_Z", "W_PCT", "NET_RATING_Z", "GP_Z", "AST_Z",
    "PTS_Z", "DEF_RATING_Z", "FT_PCT_Z", "PIE_Z", "TS_PCT_Z",
]

def merge_mvp_voting(stats_df: pd.DataFrame, mvp_df: pd.DataFrame) -> pd.DataFrame:
    """Merge Basketball-Reference MVP vote share onto a player-season stats table by name + season."""
    mvp_df = mvp_df.copy()
    mvp_df["Player"] = mvp_df["Player"].str.strip()
    mvp_df["Season"] = mvp_df["Season"].astype(str).str.strip()
    mvp_df["Share"] = pd.to_numeric(mvp_df["Share"], errors="coerce")

    stats_df = stats_df.copy()
    stats_df["PLAYER_NAME"] = stats_df["PLAYER_NAME"].str.strip()
    stats_df["SEASON"] = stats_df["SEASON"].astype(str).str.strip()

    mvp_subset = mvp_df[["Player", "Season", "Share"]].rename(
        columns={"Player": "PLAYER_NAME", "Season": "SEASON", "Share": "MVP_SHARE"}
    )
    merged = stats_df.merge(mvp_subset, on=["PLAYER_NAME", "SEASON"], how="left")
    merged["MVP_SHARE"] = merged["MVP_SHARE"].fillna(0.0)
    return merged


def add_box_score_sum(df: pd.DataFrame) -> pd.DataFrame:
    """PTS + REB + AST, a quick approximation of overall box-score impact."""
    df = df.copy()
    df["PTS_REB_AST"] = df["PTS"] + df["REB"] + df["AST"]
    return df


def add_season_zscores(df: pd.DataFrame, stats: list[str] = Z_STATS) -> pd.DataFrame:
    """Z-score each stat within its own season, so eras with different scoring environments are comparable."""
    df = df.copy()
    for stat in stats:
        if stat in df.columns:
            df[f"{stat}_Z"] = df.groupby("SEASON")[stat].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
    return df


def build_training_dataset(raw_stats_df: pd.DataFrame, mvp_df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline: merge MVP shares, then add box-score-sum and per-season Z-scores."""
    df = merge_mvp_voting(raw_stats_df, mvp_df)
    df = add_box_score_sum(df)
    df = add_season_zscores(df)
    return df
