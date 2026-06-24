"""Paths and loaders for the cached CSV snapshots in data/."""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


def load_voting_data(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """MVP voting shares, manually exported from Basketball-Reference."""
    return pd.read_csv(Path(data_dir) / "NBAMVPVotingData.csv")


def load_raw_stats(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Cached per-game traditional + advanced stats, 2000-01..2024-25 (see data_collection.py to refresh)."""
    return pd.read_csv(Path(data_dir) / "nba_raw_stats.csv")


def load_processed_stats(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Cached training dataset: raw stats + MVP share + engineered features."""
    return pd.read_csv(Path(data_dir) / "nba_processed_stats.csv")


def load_predictions(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Latest current-season MVP race snapshot (player + predicted probability)."""
    return pd.read_csv(Path(data_dir) / "predictions.csv")


def load_current_season_stats(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Latest current-season per-player box score + engineered features (for app drill-downs)."""
    return pd.read_csv(Path(data_dir) / "current_season_stats.csv")
