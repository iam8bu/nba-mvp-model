"""Training, tuning, and evaluating the MVP-vote-share regression model."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

RF_PARAM_GRID = {
    "n_estimators": [200, 500, 800, 1000],
    "max_depth": [4, 6, 8, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    # 'auto' was removed as a valid max_features value in modern scikit-learn.
    "max_features": ["sqrt", "log2", None],
}

# MVP vote share tiers used for the breakdown in evaluate_by_tier.
SHARE_TIERS = {
    "All players": None,
    "MVP candidates (share > 0)": 0.0,
    "Strong candidates (share > 0.1)": 0.1,
    "Top contenders (share > 0.3)": 0.3,
    "Likely winners (share > 0.5)": 0.5,
}


def make_train_test_split(X: pd.DataFrame, y: pd.Series, candidate_mask: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Stratified split so MVP candidates are proportionally represented in train and test."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=candidate_mask)


def train_baseline_linear(X_train, y_train) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators: int = 500, max_depth: int = 8, random_state: int = 42) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def select_important_features(model: RandomForestRegressor, feature_names: list[str], threshold: float = 0.01) -> list[str]:
    """Feature names whose random-forest importance exceeds `threshold`, most important first."""
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    return importances[importances > threshold].index.tolist()


def tune_random_forest(X_train, y_train, param_grid: dict = RF_PARAM_GRID, n_iter: int = 30, cv: int = 5, random_state: int = 42) -> RandomizedSearchCV:
    """Randomized hyperparameter search over `param_grid`, scored on R^2."""
    rf = RandomForestRegressor(random_state=random_state)
    search = RandomizedSearchCV(
        estimator=rf, param_distributions=param_grid, n_iter=n_iter, scoring="r2", cv=cv, random_state=random_state, n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def evaluate_by_tier(y_test, y_pred) -> pd.DataFrame:
    """Regression metrics broken out by MVP-vote-share tier (all players, candidates, contenders, winners)."""
    y_test = pd.Series(y_test).reset_index(drop=True)
    y_pred = pd.Series(np.asarray(y_pred), index=y_test.index)

    rows = []
    for label, threshold in SHARE_TIERS.items():
        mask = y_test > threshold if threshold is not None else pd.Series(True, index=y_test.index)
        n = int(mask.sum())
        if n == 0:
            continue
        row = {"tier": label, "samples": n, "avg_share": float(y_test[mask].mean())}
        if n >= 2:
            row.update(regression_metrics(y_test[mask], y_pred[mask]))
        rows.append(row)
    return pd.DataFrame(rows)
