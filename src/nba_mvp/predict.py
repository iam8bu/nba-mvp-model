"""Turning a trained model + current-season stats into a normalized MVP-race prediction."""

import numpy as np
import pandas as pd

# Z-scored within just the current season's qualified pool (no prior-season history needed).
CURRENT_SEASON_Z_STATS = ["PTS", "REB", "AST", "PTS_REB_AST", "GP", "NET_RATING", "DEF_RATING", "FT_PCT", "PIE", "TS_PCT"]


def engineer_current_season_features(df: pd.DataFrame, min_games: int = 10) -> pd.DataFrame:
    """Filter to qualified players and engineer the same features used in training.

    Z-scores are computed within just this season's pool, since there's no full season
    of results yet to standardize against like the historical training data has.
    """
    df = df[df["GP"] >= min_games].copy()
    df["PTS_REB_AST"] = df["PTS"] + df["REB"] + df["AST"]

    for stat in CURRENT_SEASON_Z_STATS:
        mean_val = df[stat].mean()
        std_val = df[stat].std()
        df[f"{stat}_Z"] = (df[stat] - mean_val) / std_val if std_val > 0 else 0.0

    return df


def prepare_current_season_features(df: pd.DataFrame, final_features: list[str], min_games: int = 10):
    """Engineer current-season features and reduce to the model's feature columns.

    Returns (X, player_names).
    """
    engineered = engineer_current_season_features(df, min_games=min_games)
    X = engineered.reindex(columns=final_features, fill_value=0)
    return X, engineered["PLAYER_NAME"].values


def predict_mvp_shares(model, X) -> np.ndarray:
    return model.predict(X)


def normalize_to_probability(shares) -> np.ndarray:
    """Rescale predicted shares across the field so they sum to 1, like a probability."""
    shares = np.asarray(shares, dtype=float)
    total = shares.sum()
    if total == 0:
        return shares
    return shares / total


def top_candidates(player_names, probabilities, threshold: float = 0.01) -> pd.DataFrame:
    """Players with at least `threshold` predicted MVP probability, highest first."""
    probabilities = np.asarray(probabilities)
    mask = probabilities >= threshold
    result = pd.DataFrame(
        {
            "PLAYER_NAME": np.asarray(player_names)[mask],
            "MVP_PROBABILITY": probabilities[mask],
        }
    ).sort_values("MVP_PROBABILITY", ascending=False).reset_index(drop=True)
    return result
