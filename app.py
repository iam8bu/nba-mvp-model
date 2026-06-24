"""Streamlit demo: current-season NBA MVP race, from a cached model snapshot.

Reads data/predictions.csv and data/current_season_stats.csv only — no live nba_api
calls or model training here. Run scripts/refresh_predictions.py to update the snapshot.

Run with: streamlit run app.py
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib.pyplot as plt
import streamlit as st

from nba_mvp import data_io

DISPLAY_STATS = {
    "PTS": "Points",
    "REB": "Rebounds",
    "AST": "Assists",
    "W_PCT": "Team Win %",
}
Z_PROFILE_STATS = ["PTS_REB_AST_Z", "NET_RATING_Z", "GP_Z", "AST_Z", "PTS_Z", "DEF_RATING_Z", "FT_PCT_Z", "PIE_Z", "TS_PCT_Z"]

st.set_page_config(page_title="NBA MVP Race Tracker", layout="centered")


@st.cache_data
def load_snapshot():
    predictions = data_io.load_predictions()
    current_stats = data_io.load_current_season_stats()
    return predictions, current_stats


st.title("NBA MVP Race Tracker")
st.write(
    "Current-season MVP probabilities from a Random Forest trained on 25 seasons "
    "(2000-01 to 2024-25) of box-score and advanced stats, validated against actual "
    "Basketball-Reference MVP voting."
)

predictions, current_stats = load_snapshot()

predictions_path = data_io.DATA_DIR / "predictions.csv"
last_refreshed = dt.datetime.fromtimestamp(predictions_path.stat().st_mtime).strftime("%B %-d, %Y")
st.caption(f"Snapshot last refreshed: {last_refreshed}. Run `python scripts/refresh_predictions.py` to update it.")

top_n = st.slider("Number of contenders to show", min_value=3, max_value=len(predictions), value=min(10, len(predictions)))
top_predictions = predictions.head(top_n)

fig, ax = plt.subplots(figsize=(8, 0.4 * top_n + 1))
ordered = top_predictions.iloc[::-1]
ax.barh(ordered["PLAYER_NAME"], ordered["MVP_PROBABILITY"] * 100)
ax.set_xlabel("Predicted MVP probability (%)")
ax.set_title(f"Top {top_n} MVP contenders")
plt.tight_layout()
st.pyplot(fig)

display_table = top_predictions.copy()
display_table["MVP_PROBABILITY"] = (display_table["MVP_PROBABILITY"] * 100).round(1).astype(str) + "%"
display_table = display_table.rename(columns={"PLAYER_NAME": "Player", "MVP_PROBABILITY": "MVP Probability"})
st.dataframe(display_table, width="stretch", hide_index=True)

st.divider()
st.subheader("Player profile")

selected_player = st.selectbox("Choose a contender to inspect", top_predictions["PLAYER_NAME"])
player_row = current_stats[current_stats["PLAYER_NAME"] == selected_player]

if player_row.empty:
    st.warning("No detailed stats found for this player in the current snapshot.")
else:
    player_row = player_row.iloc[0]
    team = player_row.get("TEAM_ABBREVIATION", "")
    st.write(f"**{selected_player}** ({team}) — {int(player_row['GP'])} games played")

    cols = st.columns(len(DISPLAY_STATS))
    for col, (stat, label) in zip(cols, DISPLAY_STATS.items()):
        value = player_row[stat]
        col.metric(label, f"{value:.1%}" if stat == "W_PCT" else f"{value:.1f}")

    z_values = player_row[Z_PROFILE_STATS]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in z_values]
    ax.barh(Z_PROFILE_STATS, z_values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Z-score vs. this season's qualified players (0 = league average)")
    ax.set_title(f"{selected_player}'s feature profile")
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("These are the exact features the model uses to predict MVP share — bars to the right of 0 are above this season's average.")

st.divider()
st.caption(
    "Data: NBA Stats API (nba_api) for box-score/advanced stats, Basketball-Reference for "
    "historical MVP voting. See the analysis notebook in `notebooks/` for the full methodology."
)
