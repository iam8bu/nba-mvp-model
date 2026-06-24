# NBA MVP Prediction Model

Predicting NBA MVP vote share from box-score and advanced stats, trained on 25 seasons of data (2000-01 to 2024-25) and validated against real Basketball-Reference MVP voting. Built for the **2025 FAS Technical Challenge**.

Includes a live current-season MVP race prediction and an interactive "MVP Race Tracker" app.

## What's here

- **`src/nba_mvp/`** — the reusable pipeline: live stat collection (`data_collection.py`), feature engineering (`features.py`), model training/tuning/evaluation (`modeling.py`), and current-season prediction (`predict.py`). The notebook, the refresh script, and the app all build on this.
- **`notebooks/nba_mvp_model.ipynb`** — the full methodology and results writeup.
- **`scripts/refresh_predictions.py`** — a CLI to update the cached MVP-race snapshot (`python scripts/refresh_predictions.py`, or `--full` to also re-pull the entire training history).
- **`app.py`** — a Streamlit app showing the current MVP race leaderboard and a per-player stat profile. Reads only the cached snapshot in `data/` — no live API calls or model training at request time.
- **`archive/`** — the original unrefactored notebook, the original project presentation (`NBA MVP Model Presentation.pptx`), and an unused raw stats pull (`unused_nba_historical_stats.csv`) that was never actually loaded by the notebook.

## Data

- **NBA Stats API** (via [`nba_api`](https://github.com/swar/nba_api)): per-game traditional + advanced stats for every player-season, 2000-01 through 2024-25.
- **Basketball-Reference**: MVP voting results, manually exported as `data/NBAMVPVotingData.csv` since no API conveniently exposes historical voting. `MVP_SHARE` is a player's voting points divided by total possible points (e.g. a unanimous MVP scores 1.0).

## Methodology

1. **Data Acquisition** — merge MVP vote share onto every player-season's stats by player name + season.
2. **Feature Engineering & Selection** — compute `PTS_REB_AST` (a quick box-score-impact proxy) and Z-score every stat within its own season, so different scoring eras are comparable. Train a Random Forest on the full feature set, keep anything above 1% importance, then drop the non-standardized duplicates in favor of their Z-scored versions — the final model uses 9 Z-scored features plus team win percentage.
3. **Modeling** — baseline linear regression, then a Random Forest tuned via randomized search (5-fold CV, scored on R²), evaluated specifically on how well it separates real MVP contenders from the rest of the league, not just on overall R².

## Results

The tuned model explains roughly **65-70% of the variance** in MVP vote share overall and among players who received any MVP support — a solid result for surfacing real contenders out of ~12,000 player-seasons. It starts to struggle separating the *strongest* contenders from fringe ones, for two reasons: that tier has very few real examples to learn from, and MVP voting is partly narrative-driven — voters weigh storylines more heavily once they're down to the top 2-3 candidates, which a box-score-only model can't see. Despite that, its rankings hold up against domain knowledge: it consistently puts the league's true top tier (Jokić/SGA/Dončić/Giannis-caliber seasons) at the top, and its longer-shot picks tend to be reasonable.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the analysis notebook:

```bash
jupyter notebook notebooks/nba_mvp_model.ipynb
```

Run the MVP Race Tracker app:

```bash
streamlit run app.py
```

Refresh the cached snapshot the app and notebook read:

```bash
python scripts/refresh_predictions.py               # current season only (fast)
python scripts/refresh_predictions.py --full         # also re-pulls the full 2000-2025 training history (slow)
```

> **Note:** `nba_api` hits stats.nba.com directly, which is known to rate-limit or block requests from cloud hosts. The app intentionally never calls it — only `scripts/refresh_predictions.py` and the notebook's optional "refresh from live data" cell do, and both are meant to be run locally.

## Limitations & future work

- MVP voting is partly a narrative award, and that's exactly the part a box-score model can't capture.
- The strong/top-contender tiers train on only a few dozen real examples, so the model is more reliable at "is this player in the conversation" than at finely ranking the top 2-3 candidates against each other.
- Team success (win %) is included, but broader team context (a team's record without this player, supporting cast strength, season narratives) isn't — likely a meaningful share of the unexplained variance.
- Next steps: add team-level context features, track how predictions evolve game-by-game through a season, and backtest against past seasons the model wasn't trained on to see how early it converges on the eventual winner.

## Credits

Data from Basketball-Reference and the NBA Stats API (see citations in the notebook). ChatGPT, Claude, and Copilot were used at points during development.
