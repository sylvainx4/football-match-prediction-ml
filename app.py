
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="wide"
)

PROJECT_FOLDER = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_FOLDER / "models" / "best_model.joblib"
DATA_PATH = (
    PROJECT_FOLDER / "data" / "raw" / "results.csv"
)


# ==========================================
# 2. LOAD THE TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ==========================================
# 3. LOAD HISTORICAL FOOTBALL DATA
# ==========================================

@st.cache_data
def load_team_history():

    df = pd.read_csv(DATA_PATH, encoding="cp1252")

    df["DateTime"] = pd.to_datetime(
        df["DateTime"],
        errors="coerce",
        utc=True
    )

    df = df.dropna(
        subset=[
            "DateTime", "HomeTeam", "AwayTeam",
            "FTHG", "FTAG", "FTR"
        ]
    )

    df = df.sort_values(
        "DateTime",
        kind="stable"
    )

    histories = {}

    for match in df.itertuples():

        home = match.HomeTeam
        away = match.AwayTeam

        # Determine points and results
        if match.FTR == "H":
            home_points, away_points = 3, 0
            home_result, away_result = "W", "L"

        elif match.FTR == "A":
            home_points, away_points = 0, 3
            home_result, away_result = "L", "W"

        else:
            home_points, away_points = 1, 1
            home_result, away_result = "D", "D"

        # Home team's match history
        histories.setdefault(home, []).append({
            "date": match.DateTime,
            "goals_for": float(match.FTHG),
            "goals_against": float(match.FTAG),
            "points": home_points,
            "result": home_result
        })

        # Away team's match history
        histories.setdefault(away, []).append({
            "date": match.DateTime,
            "goals_for": float(match.FTAG),
            "goals_against": float(match.FTHG),
            "points": away_points,
            "result": away_result
        })

    return histories, df["DateTime"].max()


# ==========================================
# 4. CALCULATE THE LAST FIVE MATCHES
# ==========================================

def calculate_team_features(history, prefix):

    recent_matches = history[-5:]

    count = len(recent_matches)

    return {
        f"{prefix}_avg_goals_scored":
            sum(m["goals_for"] for m in recent_matches) / count,

        f"{prefix}_avg_goals_conceded":
            sum(m["goals_against"] for m in recent_matches) / count,

        f"{prefix}_avg_points":
            sum(m["points"] for m in recent_matches) / count,

        f"{prefix}_win_rate":
            sum(m["points"] == 3 for m in recent_matches) / count,

        f"{prefix}_matches_available": count
    }


# ==========================================
# 5. APPLICATION INTERFACE
# ==========================================

st.title("⚽ Football Match Prediction System")

st.write(
    "An interactive football match outcome predictor "
    "powered by Machine Learning."
)

st.divider()

try:

    model = load_model()
    histories, latest_date = load_team_history()

except Exception as error:

    st.error(f"Could not load the model or dataset: {error}")

    st.info(
        "Check that best_model.joblib exists inside models "
        "and results.csv exists inside data/raw."
    )

    st.stop()


st.caption(
    f"Historical dataset available through "
    f"{latest_date.strftime('%d %B %Y')}"
)

st.warning(
    "This demonstration uses historical data ending in April 2022. "
    "It does not include live team results or current-season form. "
    "Predictions should not be treated as guaranteed outcomes."
)

st.subheader("Select your match")

teams = sorted(histories.keys())

col1, col2 = st.columns(2)

with col1:

    home_team = st.selectbox(
        "Home Team",
        teams,
        index=teams.index("Arsenal") if "Arsenal" in teams else 0
    )

with col2:

    away_options = [team for team in teams if team != home_team]

    away_team = st.selectbox(
        "Away Team",
        away_options,
        index=(
            away_options.index("Chelsea")
            if "Chelsea" in away_options else 0
        )
    )


# ==========================================
# 6. DISPLAY RECENT TEAM FORM
# ==========================================

st.divider()

st.subheader("Recent Team Performance")

form_col1, form_col2 = st.columns(2)

home_history = histories[home_team][-5:]
away_history = histories[away_team][-5:]

with form_col1:

    st.markdown(f"**{home_team}**")

    home_form = " ".join(
        match["result"] for match in reversed(home_history)
    )

    st.write("Recent form (newest first):", home_form)

    st.write(
        "Average goals scored:",
        round(
            sum(m["goals_for"] for m in home_history)
            / len(home_history), 2
        )
    )

with form_col2:

    st.markdown(f"**{away_team}**")

    away_form = " ".join(
        match["result"] for match in reversed(away_history)
    )

    st.write("Recent form (newest first):", away_form)

    st.write(
        "Average goals scored:",
        round(
            sum(m["goals_for"] for m in away_history)
            / len(away_history), 2
        )
    )


# ==========================================
# 7. GENERATE MATCH PREDICTION
# ==========================================

st.divider()

if st.button("Predict Match Outcome", type="primary"):

    features = {}

    features.update(
        calculate_team_features(
            histories[home_team],
            "home"
        )
    )

    features.update(
        calculate_team_features(
            histories[away_team],
            "away"
        )
    )

    input_data = pd.DataFrame([{
        "HomeTeam": home_team,
        "AwayTeam": away_team,
        **features
    }])

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    probability_map = dict(
        zip(model.classes_, probabilities)
    )

    outcome_names = {
        "H": "Home Win",
        "D": "Draw",
        "A": "Away Win"
    }

    st.subheader("Prediction Results")

    st.success(
        f"Predicted outcome: {outcome_names[prediction]}"
    )

    result_col1, result_col2, result_col3 = st.columns(3)

    result_col1.metric(
        "Home Win",
        f"{probability_map.get('H', 0) * 100:.1f}%"
    )

    result_col2.metric(
        "Draw",
        f"{probability_map.get('D', 0) * 100:.1f}%"
    )

    result_col3.metric(
        "Away Win",
        f"{probability_map.get('A', 0) * 100:.1f}%"
    )

    st.subheader("Outcome Probability Comparison")

    probability_chart = pd.DataFrame({
        "Outcome": ["Home Win", "Draw", "Away Win"],
        "Probability": [
            probability_map.get("H", 0),
            probability_map.get("D", 0),
            probability_map.get("A", 0)
        ]
    }).set_index("Outcome")

    st.bar_chart(probability_chart)

    st.caption(
        "Probabilities represent the model's estimates, "
        "not certainty or guaranteed match results."
    )

st.divider()

st.caption(
    "Football Match Prediction Using Machine Learning | DAI 022"
)
