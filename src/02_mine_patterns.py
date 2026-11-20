from collections import Counter
from itertools import combinations
import os
import numpy as np
import pandas as pd


def split_by_match(df, train_ratio=0.70, val_ratio=0.15, seed=42):
  """Splits rallies into Train/Val/Test grouped by match_id to prevent match/player leakage."""
  rng = np.random.default_rng(seed)
  matches = np.array(sorted(df["match_id"].unique()))
  rng.shuffle(matches)

  n_total = len(matches)
  n_train = int(n_total * train_ratio)
  n_val = int(n_total * val_ratio)

  train_matches = set(matches[:n_train])
  val_matches = set(matches[n_train : n_train + n_val])
  test_matches = set(matches[n_train + n_val :])

  train_df = df[df["match_id"].isin(train_matches)].copy()
  val_df = df[df["match_id"].isin(val_matches)].copy()
  test_df = df[df["match_id"].isin(test_matches)].copy()

  print(
      f"Match Split -> Train: {len(train_matches)} matches"
      f" ({len(train_df)} rallies) | Val: {len(val_matches)} matches"
      f" ({len(val_df)} rallies) | Test: {len(test_matches)} matches"
      f" ({len(test_df)} rallies)"
  )
  return train_df, val_df, test_df


def extract_patterns_from_rally(tokens, n, mode="ordered"):
  """Extracts unique n-stroke patterns present in a single rally.

  - mode='ordered': contiguous n-grams (s_t, ..., s_{t+n-1})
  - mode='unordered': unique sorted combinations of n tokens anywhere in the
  rally
  """
  if len(tokens) < n:
    return set()

  if mode == "ordered":
    return {" -> ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}
  elif mode == "unordered":
    unique_tokens = sorted(set(tokens))
    if len(unique_tokens) < n:
      return set()
    return {" + ".join(comb) for comb in combinations(unique_tokens, n)}
  else:
    raise ValueError("mode must be 'ordered' or 'unordered'")


def mine_rules(df, seq_col="sequence_full", min_support=0.02, mode="ordered"):
  """Computes sequence support, outcome-specific support, confidence, and lift

  for 2-stroke and 3-stroke patterns against Server_Win and Receiver_Win.
  """
  n_rallies = len(df)
  outcome_counts = df["outcome"].value_counts().to_dict()
  base_rates = {k: v / n_rallies for k, v in outcome_counts.items()}

  pattern_total_counts = Counter()
  pattern_outcome_counts = {
      "Server_Win": Counter(),
      "Receiver_Win": Counter(),
  }
  pattern_match_sets = {}

  for _, row in df.iterrows():
    seq_str = str(row[seq_col])
    if not seq_str or seq_str == "nan":
      continue
    tokens = seq_str.split(" | ")
    outcome = row["outcome"]
    match_id = row["match_id"]

    patterns_in_rally = set()
    for n in [2, 3]:
      for pat in extract_patterns_from_rally(tokens, n=n, mode=mode):
        patterns_in_rally.add((n, pat))

    for n, pat in patterns_in_rally:
      pattern_total_counts[(n, pat)] += 1
      pattern_outcome_counts[outcome][(n, pat)] += 1
      pattern_match_sets.setdefault((n, pat), set()).add(match_id)

  results = []
  for (n, pat), count_total in pattern_total_counts.items():
    seq_support = count_total / n_rallies
    if seq_support < min_support:
      continue

    n_matches = len(pattern_match_sets[(n, pat)])

    for outcome in ["Server_Win", "Receiver_Win"]:
      count_out = pattern_outcome_counts[outcome][(n, pat)]
      outcome_support = count_out / n_rallies
      confidence = count_out / count_total
      lift = confidence / base_rates[outcome]

      results.append({
          "mode": mode,
          "n_gram": n,
          "pattern": pat,
          "target_outcome": outcome,
          "count_total": count_total,
          "count_outcome": count_out,
          "unique_matches": n_matches,
          "seq_support": round(seq_support, 4),
          "outcome_support": round(outcome_support, 4),
          "confidence": round(confidence, 4),
          "lift": round(lift, 4),
      })

  res_df = pd.DataFrame(results)
  if not res_df.empty:
    res_df = res_df.sort_values(
        by=["lift", "count_total"], ascending=[False, False]
    ).reset_index(drop=True)
  return res_df


if __name__ == "__main__":
  rallies = pd.read_csv(r"data\processed\rallies_sequences.csv")
  train_df, val_df, test_df = split_by_match(rallies)

  os.makedirs("results", exist_ok=True)
  train_df.to_csv(r"data\processed\train_rallies.csv", index=False)
  val_df.to_csv(r"data\processed\val_rallies.csv", index=False)
  test_df.to_csv(r"data\processed\test_rallies.csv", index=False)

  summary_rows = []
  for threshold in [0.01, 0.02, 0.05]:
    for mode in ["ordered", "unordered"]:
      rules = mine_rules(
          train_df, seq_col="sequence_full", min_support=threshold, mode=mode
      )
      rules.to_csv(f"results/rules_{mode}_sup_{threshold}.csv", index=False)

      high_lift = (
          rules[rules["lift"] >= 1.25] if not rules.empty else pd.DataFrame()
      )
      summary_rows.append({
          "min_support": threshold,
          "mode": mode,
          "total_candidate_rules": len(rules),
          "rules_lift_ge_1_25": len(high_lift),
          "max_lift": rules["lift"].max() if not rules.empty else 0.0,
      })

  summary_df = pd.DataFrame(summary_rows)
  print("\n--- Threshold & Baseline Comparison (Train Split) ---")
  print(summary_df.to_string(index=False))
  summary_df.to_csv("results/threshold_baseline_summary.csv", index=False)