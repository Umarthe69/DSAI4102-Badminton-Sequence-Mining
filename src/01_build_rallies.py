import glob
import os
import pandas as pd


def load_and_clean_strokes(data_dir=r"data\raw\ShuttleSet22\set"):
  """Loads all match set CSV files, logs data quality stats, and cleans stroke records."""
  csv_files = sorted(
      glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
  )
  match_csvs = [
      f
      for f in csv_files
      if os.path.basename(f).lower() not in ["match.csv", "homography.csv"]
  ]

  print(f"Found {len(match_csvs)} match set CSV files.")

  all_dfs = []
  for file_path in match_csvs:
    df = pd.read_csv(file_path)
    match_name = os.path.basename(os.path.dirname(file_path))
    set_name = os.path.splitext(os.path.basename(file_path))[0]
    df["match_id"] = match_name
    df["set_id"] = f"{match_name}_{set_name}"
    all_dfs.append(df)

  raw_strokes = pd.concat(all_dfs, ignore_index=True)
  print(f"Raw loaded strokes: {len(raw_strokes)}")
  print(f"Unique matches: {raw_strokes['match_id'].nunique()}")

  missing_before = (
      raw_strokes[["player", "type", "rally", "ball_round"]]
      .isna()
      .sum()
      .to_dict()
  )
  print(f"Missing values before cleaning: {missing_before}")

  cleaned = raw_strokes.dropna(
      subset=["player", "type", "rally", "ball_round"]
  ).copy()
  cleaned["type"] = cleaned["type"].astype(str).str.strip()
  cleaned["player"] = cleaned["player"].astype(str).str.strip()

  # Sort strictly by set, rally, and shot order (ball_round)
  cleaned = cleaned.sort_values(
      by=["set_id", "rally", "ball_round"]
  ).reset_index(drop=True)
  print(f"Cleaned strokes: {len(cleaned)}")
  return cleaned


def build_rally_sequences(strokes_df):
  """Reconstructs complete rallies and maps hitters/winners to consistent within-rally roles:

  - Server (S): player who hits ball_round == 1
  - Receiver (R): the opponent in that rally
  """
  rally_records = []
  dropped_rallies = 0

  grouped = strokes_df.groupby(["match_id", "set_id", "rally"], sort=False)

  for (match_id, set_id, rally_num), group in grouped:
    group = group.sort_values("ball_round")

    # Determine rally winner from 'getpoint_player' (populated on the final shot of the rally)
    winner_series = group["getpoint_player"].dropna()
    if winner_series.empty:
      dropped_rallies += 1
      continue

    winner_player = str(winner_series.iloc[-1]).strip()
    server_player = str(group["player"].iloc[0]).strip()

    role_shots = []
    for _, row in group.iterrows():
      hitter = str(row["player"]).strip()
      shot_type = row["type"]
      role = "S" if hitter == server_player else "R"
      role_shots.append(f"{role}:{shot_type}")

    outcome = (
        "Server_Win" if winner_player == server_player else "Receiver_Win"
    )
    win_reason = (
        str(group["win_reason"].dropna().iloc[-1]).strip()
        if not group["win_reason"].dropna().empty
        else "unknown"
    )
    lose_reason = (
        str(group["lose_reason"].dropna().iloc[-1]).strip()
        if not group["lose_reason"].dropna().empty
        else "unknown"
    )

    rally_records.append({
        "match_id": match_id,
        "set_id": set_id,
        "rally_num": rally_num,
        "rally_length": len(role_shots),
        "outcome": outcome,
        "win_reason": win_reason,
        "lose_reason": lose_reason,
        "sequence_full": " | ".join(role_shots),
        "sequence_no_final": (
            " | ".join(role_shots[:-1]) if len(role_shots) > 1 else ""
        ),
    })

  rallies_df = pd.DataFrame(rally_records)
  print(
      f"Reconstructed valid rallies: {len(rallies_df)} (Dropped incomplete:"
      f" {dropped_rallies})"
  )
  print(
      f"Mean rally length: {rallies_df['rally_length'].mean():.2f} shots"
      f" (Median: {rallies_df['rally_length'].median():.1f})"
  )
  print("\nOutcome distribution:")
  print(
      rallies_df["outcome"]
      .value_counts()
      .to_frame("count")
      .assign(rate=rallies_df["outcome"].value_counts(normalize=True).round(4))
  )
  return rallies_df


if __name__ == "__main__":
  strokes = load_and_clean_strokes(r"data\raw\ShuttleSet22\set")
  rallies = build_rally_sequences(strokes)

  os.makedirs(r"data\processed", exist_ok=True)
  out_path = r"data\processed\rallies_sequences.csv"
  rallies.to_csv(out_path, index=False)
  print(f"\nSaved processed rally dataset to {out_path}")