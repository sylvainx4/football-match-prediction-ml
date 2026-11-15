
import pandas as pd
from pathlib import Path
from collections import defaultdict, deque

# 1. Locate project folders
project_folder = Path(__file__).resolve().parent.parent
raw_folder = project_folder / "data" / "raw"
processed_folder = project_folder / "data" / "processed"

processed_folder.mkdir(parents=True, exist_ok=True)

# 2. Load the football dataset
csv_file = list(raw_folder.glob("*.csv"))[0]
df = pd.read_csv(csv_file, encoding="cp1252")

# 3. Prepare dates and sort matches chronologically
df["DateTime"] = pd.to_datetime(
    df["DateTime"], errors="coerce", utc=True
)

df = df.dropna(
    subset=["DateTime", "HomeTeam", "AwayTeam", "FTR",
            "FTHG", "FTAG"]
)

df = df.sort_values("DateTime", kind="stable")

# 4. Store the previous five matches for every team
team_history = defaultdict(lambda: deque(maxlen=5))

feature_rows = []
updates = []

# Calculate features from a team's previous matches
def calculate_form(history, prefix):

    if not history:
        return {
            f"{prefix}_avg_goals_scored": 0,
            f"{prefix}_avg_goals_conceded": 0,
            f"{prefix}_avg_points": 0,
            f"{prefix}_win_rate": 0,
            f"{prefix}_matches_available": 0
        }

    matches = list(history)

    return {
        f"{prefix}_avg_goals_scored":
            sum(m["goals_for"] for m in matches) / len(matches),

        f"{prefix}_avg_goals_conceded":
            sum(m["goals_against"] for m in matches) / len(matches),

        f"{prefix}_avg_points":
            sum(m["points"] for m in matches) / len(matches),

        f"{prefix}_win_rate":
            sum(m["points"] == 3 for m in matches) / len(matches),

        f"{prefix}_matches_available": len(matches)
    }

# 5. Process matches in chronological batches.
# Matches with the same timestamp cannot use each
# other's results as previous-match information.

for match_time, batch in df.groupby("DateTime", sort=True):

    updates = []

    for _, match in batch.iterrows():

        home = match["HomeTeam"]
        away = match["AwayTeam"]

        features = {
            "DateTime": match_time,
            "HomeTeam": home,
            "AwayTeam": away
        }

        features.update(
            calculate_form(team_history[home], "home")
        )

        features.update(
            calculate_form(team_history[away], "away")
        )

        features["FTR"] = match["FTR"]

        feature_rows.append(features)

        # Record this match's result for future matches
        if match["FTR"] == "H":
            home_points, away_points = 3, 0
        elif match["FTR"] == "A":
            home_points, away_points = 0, 3
        else:
            home_points, away_points = 1, 1

        updates.append((
            home,
            away,
            {
                "goals_for": match["FTHG"],
                "goals_against": match["FTAG"],
                "points": home_points
            },
            {
                "goals_for": match["FTAG"],
                "goals_against": match["FTHG"],
                "points": away_points
            }
        ))

    # Update histories only after the whole timestamp batch
    for home, away, home_data, away_data in updates:
        team_history[home].append(home_data)
        team_history[away].append(away_data)

# 6. Create the processed dataset
features_df = pd.DataFrame(feature_rows)

output_file = processed_folder / "matches_features.csv"
features_df.to_csv(output_file, index=False)

print("\n========== FEATURE ENGINEERING COMPLETE ==========")
print("Total matches processed:", len(features_df))
print("Total features created:", len(features_df.columns))
print("\nFirst five processed matches:")
print(features_df.head())

print("\nFeature columns:")
print(features_df.columns.tolist())

print("\nSaved dataset to:")
print(output_file)
