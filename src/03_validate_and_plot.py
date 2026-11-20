from importlib.machinery import SourceFileLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

miner = SourceFileLoader("mine_patterns", r"src\02_mine_patterns.py").load_module()
extract_patterns_from_rally = miner.extract_patterns_from_rally
mine_rules = miner.mine_rules


def evaluate_on_holdout(candidate_rules, holdout_df, seq_col="sequence_full"):
  """Evaluates rules discovered on Train against unseen Val/Test matches."""
  n_rallies = len(holdout_df)
  base_rates = holdout_df["outcome"].value_counts(normalize=True).to_dict()

  rally_patterns = []
  for _, row in holdout_df.iterrows():
    seq_str = str(row[seq_col])
    tokens = seq_str.split(" | ") if seq_str and seq_str != "nan" else []
    pats = set()
    for n in [2, 3]:
      pats.update(extract_patterns_from_rally(tokens, n=n, mode="ordered"))
    rally_patterns.append((pats, row["outcome"]))

  eval_rows = []
  for _, rule in candidate_rules.iterrows():
    pat = rule["pattern"]
    target = rule["target_outcome"]

    tot = sum(1 for pats, _ in rally_patterns if pat in pats)
    out = sum(
        1 for pats, outc in rally_patterns if pat in pats and outc == target
    )

    sup = tot / n_rallies if n_rallies > 0 else 0.0
    conf = out / tot if tot > 0 else 0.0
    lift = conf / base_rates[target] if tot > 0 else 0.0

    eval_rows.append({
        "pattern": pat,
        "target_outcome": target,
        "train_count": rule["count_total"],
        "train_lift": rule["lift"],
        "holdout_count": tot,
        "holdout_support": round(sup, 4),
        "holdout_confidence": round(conf, 4),
        "holdout_lift": round(lift, 4),
    })

  return pd.DataFrame(eval_rows)


def run_permutation_test(
    train_df, candidate_rules, n_permutations=200, seed=42
):
  """Shuffles outcome labels n_permutations times to test whether observed max/rule lifts

  could occur by random chance across candidate rules.
  """
  rng = np.random.default_rng(seed)
  n_rallies = len(train_df)
  outcomes = train_df["outcome"].values.copy()
  base_rates = pd.Series(outcomes).value_counts(normalize=True).to_dict()

  target_pats = candidate_rules["pattern"].unique()
  pat_to_idx = {p: i for i, p in enumerate(target_pats)}
  presence = np.zeros((n_rallies, len(target_pats)), dtype=bool)

  for r_idx, seq_str in enumerate(train_df["sequence_full"].astype(str)):
    tokens = seq_str.split(" | ")
    pats = set()
    for n in [2, 3]:
      pats.update(extract_patterns_from_rally(tokens, n=n, mode="ordered"))
    for p in pats:
      if p in pat_to_idx:
        presence[r_idx, pat_to_idx[p]] = True

  pat_totals = presence.sum(axis=0)
  perm_max_lifts = []

  for _ in range(n_permutations):
    shuffled = rng.permutation(outcomes)
    max_lift_this_run = 0.0
    for outcome in ["Server_Win", "Receiver_Win"]:
      is_out = (shuffled == outcome)[:, None]
      out_counts = (presence & is_out).sum(axis=0)
      confs = np.divide(out_counts, pat_totals, where=pat_totals > 0)
      lifts = confs / base_rates[outcome]
      max_lift_this_run = max(max_lift_this_run, float(lifts.max()))
    perm_max_lifts.append(max_lift_this_run)

  perm_max_lifts = np.array(perm_max_lifts)

  p_values = []
  for _, rule in candidate_rules.iterrows():
    p_val = (np.sum(perm_max_lifts >= rule["lift"]) + 1) / (n_permutations + 1)
    p_values.append(round(float(p_val), 4))

  candidate_rules = candidate_rules.copy()
  candidate_rules["perm_p_value"] = p_values
  return candidate_rules, perm_max_lifts


def generate_figures(summary_df, holdout_eval_df, perm_max_lifts):
  """Generates clean figures formatted for an IEEE two-column paper."""
  plt.rcParams.update({"font.size": 10, "figure.dpi": 300})

  # Figure 1: Rule Collapse across Support Thresholds (Ordered vs Unordered)
  fig, ax = plt.subplots(figsize=(5.5, 3.4))
  for mode, marker in [("ordered", "o"), ("unordered", "s")]:
    sub = summary_df[summary_df["mode"] == mode]
    ax.plot(
        sub["min_support"].astype(str),
        sub["total_candidate_rules"],
        marker=marker,
        linewidth=2,
        label=f"{mode.capitalize()} Patterns",
    )
  ax.set_xlabel("Minimum Sequence Support Threshold")
  ax.set_ylabel("Number of Candidate Rules")
  ax.set_title("Candidate Rule Count Across Support Thresholds")
  ax.grid(True, linestyle="--", alpha=0.5)
  ax.legend()
  plt.tight_layout()
  plt.savefig("results/fig1_threshold_sensitivity.png")
  plt.close()

  # Figure 2: Train Lift vs. Held-Out Test Lift (Generalization Check)
  if not holdout_eval_df.empty:
    valid = holdout_eval_df[holdout_eval_df["holdout_count"] >= 5]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.scatter(
        valid["train_lift"], valid["holdout_lift"], alpha=0.75, edgecolors="k"
    )
    ax.axhline(
        1.0, color="gray", linestyle="--", linewidth=1, label="Null Lift (1.0)"
    )
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    lims = [
        min(valid["train_lift"].min(), valid["holdout_lift"].min()) - 0.1,
        max(valid["train_lift"].max(), valid["holdout_lift"].max()) + 0.1,
    ]
    ax.plot(lims, lims, "r:", linewidth=1.2, label="Perfect Stability (y = x)")
    ax.set_xlabel("Training Split Lift")
    ax.set_ylabel("Held-Out Test Split Lift")
    ax.set_title("Pattern Lift Generalization Across Unseen Matches")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/fig2_holdout_generalization.png")
    plt.close()

  # Figure 3: Permutation Test Null Distribution of Max Lift
  fig, ax = plt.subplots(figsize=(5.5, 3.2))
  ax.hist(
      perm_max_lifts, bins=20, color="#4C72B0", edgecolor="black", alpha=0.85
  )
  ax.axvline(
      np.percentile(perm_max_lifts, 95),
      color="red",
      linestyle="--",
      linewidth=1.5,
      label="95th Percentile Null Max Lift",
  )
  ax.set_xlabel("Maximum Lift Under Permuted Outcomes")
  ax.set_ylabel("Frequency (out of 200 permutations)")
  ax.set_title("Null Distribution of Maximum Lift (Chance Baseline)")
  ax.legend()
  plt.tight_layout()
  plt.savefig("results/fig3_permutation_null.png")
  plt.close()


if __name__ == "__main__":
  train_df = pd.read_csv(r"data\processed\train_rallies.csv")
  val_df = pd.read_csv(r"data\processed\val_rallies.csv")
  test_df = pd.read_csv(r"data\processed\test_rallies.csv")
  summary_df = pd.read_csv("results/threshold_baseline_summary.csv")

  rules_02 = mine_rules(
      train_df, seq_col="sequence_full", min_support=0.02, mode="ordered"
  )
  rules_02_perm, perm_null = run_permutation_test(
      train_df, rules_02, n_permutations=200
  )
  rules_02_perm.to_csv("results/train_rules_002_with_pvalues.csv", index=False)

  test_eval = evaluate_on_holdout(rules_02_perm, test_df, seq_col="sequence_full")
  test_eval.to_csv("results/test_holdout_evaluation_002.csv", index=False)

  print(
      "\n--- Top 10 Ordered Patterns (Support >= 0.02) Evaluated on Held-Out"
      " Test Matches ---"
  )
  print(test_eval.head(10).to_string(index=False))

  rules_no_final = mine_rules(
      train_df, seq_col="sequence_no_final", min_support=0.02, mode="ordered"
  )
  rules_no_final.to_csv(
      "results/train_rules_002_no_final_shot.csv", index=False
  )

  print("\n--- Top 10 Patterns EXCLUDING Final Rally-Ending Shot ---")
  print(rules_no_final.head(10).to_string(index=False))

  generate_figures(summary_df, test_eval, perm_null)
  print(
      "\nSaved figures to results/fig1_threshold_sensitivity.png,"
      " fig2_holdout_generalization.png, fig3_permutation_null.png"
  )