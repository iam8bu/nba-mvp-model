"""Refresh the cached MVP-race snapshot in data/.

By default this only re-pulls the *current* season's live stats from nba_api and
re-predicts using the existing cached training history (data/nba_raw_stats.csv).
Pass --full to also re-pull the full training history (slow: ~25 seasons x 2 API
calls each, with rate-limit pauses).

The Streamlit app never calls nba_api itself — run this script to update the
snapshot it reads.

Usage:
    python scripts/refresh_predictions.py
    python scripts/refresh_predictions.py --full
    python scripts/refresh_predictions.py --season-year 2025
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nba_mvp import data_collection, data_io, features, modeling, predict

HISTORY_START_YEAR = 2000
HISTORY_END_YEAR = 2024  # 2024-25 season; bump forward once later seasons are final


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="Also re-pull the full training history from nba_api (slow).")
    parser.add_argument("--season-year", type=int, default=None, help="Override the current season's start year (e.g. 2025 for 2025-26).")
    args = parser.parse_args()

    if args.full:
        print(f"Re-pulling full history {HISTORY_START_YEAR}-{HISTORY_END_YEAR}...")
        raw_stats = data_collection.collect_seasons(HISTORY_START_YEAR, HISTORY_END_YEAR)
        raw_stats.to_csv(data_io.DATA_DIR / "nba_raw_stats.csv", index=False)
    else:
        raw_stats = data_io.load_raw_stats()

    mvp_df = data_io.load_voting_data()
    training_df = features.build_training_dataset(raw_stats, mvp_df)
    training_df.to_csv(data_io.DATA_DIR / "nba_processed_stats.csv", index=False)
    print(f"Training dataset: {len(training_df)} player-seasons, {(training_df['MVP_SHARE'] > 0).sum()} with MVP votes")

    training_df["mvp_candidate"] = (training_df["MVP_SHARE"] > 0).astype(int)
    X = training_df[features.FINAL_FEATURES].fillna(training_df[features.FINAL_FEATURES].median())
    y = training_df["MVP_SHARE"]
    X_train, X_test, y_train, y_test = modeling.make_train_test_split(X, y, training_df["mvp_candidate"])

    print("Tuning random forest...")
    search = modeling.tune_random_forest(X_train, y_train)
    model = search.best_estimator_
    y_pred = model.predict(X_test)
    print("\nHeld-out performance:")
    print(modeling.evaluate_by_tier(y_test, y_pred).to_string(index=False))

    season_year = args.season_year if args.season_year is not None else data_collection.current_season_year()
    print(f"\nPredicting {data_collection.season_string(season_year)} MVP race...")
    current_df = data_collection.collect_single_season(season_year)
    current_engineered = predict.engineer_current_season_features(current_df)
    X_current = current_engineered.reindex(columns=features.FINAL_FEATURES, fill_value=0)
    player_names = current_engineered["PLAYER_NAME"].values

    shares = predict.predict_mvp_shares(model, X_current)
    probabilities = predict.normalize_to_probability(shares)
    results = predict.top_candidates(player_names, probabilities)

    current_engineered.to_csv(data_io.DATA_DIR / "current_season_stats.csv", index=False)
    results.to_csv(data_io.DATA_DIR / "predictions.csv", index=False)
    print(f"\nSaved data/current_season_stats.csv ({len(current_engineered)} qualified players)")
    print(f"Saved data/predictions.csv ({len(results)} candidates)")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
