# Mining Badminton Rally Patterns: Discovering Ordered Stroke Sequences Associated with Rally Victory

**Course:** DSAI 4102 — Data Mining  
**Institution:** University of Doha for Science and Technology (UDST), Doha, Qatar  
**Authors:** Umar Yousaf (60302231), Mohammad (60307140), Anas (60104434)

---

## Overview
Single-shot frequency counts fail to capture the turn-by-turn tactical exchanges that decide rallies in competitive badminton. This repository implements an interpretable sequential pattern mining pipeline on the **ShuttleSet22** stroke-level badminton singles dataset to discover contiguous 2- and 3-stroke sequences associated with rally victory.

Key components of the pipeline:
- **Role-Normalized Rally Reconstruction:** Reconstructs **4,840 complete rallies** (52,356 strokes across 58 matches) and maps hitters to consistent within-rally roles (`Server` `S` vs. `Receiver` `R`).
- **Order-Sensitive vs. Unordered Baseline Comparison:** Evaluates contiguous sequence mining against an unordered co-occurrence baseline across minimum support thresholds ($\tau \in \{0.01, 0.02, 0.05\}$), reducing candidate rules tenfold at $\tau = 0.02$ (336 vs. 3,312 rules).
- **Multi-Stage Validity Filtering:** Validates candidate rules using match-disjoint splits (40 train / 8 validation / 10 test matches), 200-trial outcome-label permutation testing, and final-shot exclusion (`T_no_final`).

---

## Key Empirical Findings
- **Top Statistically Validated Tactical Pattern:** `R:Net_Shot -> S:Lift -> R:Transition_Drop` (`Server_Win`)
  - **Training Split:** Lift = `1.3825`, Confidence = `70.24%` (84 occurrences across 28 matches), family-wise permutation $p = 0.0348 < 0.05$.
  - **Held-Out Test Split (10 Unseen Matches):** Lift = `1.4034`, Confidence = `72.73%`.
  - **Final-Shot Exclusion (`T_no_final`):** Remains the #1 ranked sequence (Lift = `1.3374`, Confidence = `67.95%`), confirming mid-rally positional leverage rather than a terminal finishing stroke.
- **Terminal-Shot Artifact Identified:** `R:Net_Block -> S:Net_Kill` exhibits high training lift (`1.5254`, $p = 0.0050$) but drops to `1.1133` on unseen test matches and disappears when the rally-ending shot is removed.

---

## Repository Structure
```text
DSAI4102-Badminton-Sequence-Mining/
├── data/
│   └── processed/
│       └── rallies_sequences.csv            # Reconstructed 4,840 role-mapped rallies
├── results/
│   ├── fig1_threshold_sensitivity.png       # Ordered vs. unordered rule counts
│   ├── fig2_holdout_generalization.png      # Train lift vs. held-out test lift
│   ├── fig3_permutation_null.png            # 200-trial max-lift permutation null distribution
│   ├── threshold_baseline_summary.csv       # Summary across support thresholds (0.01, 0.02, 0.05)
│   ├── train_rules_002_with_pvalues.csv     # Candidate training rules with permutation p-values
│   ├── test_holdout_evaluation_002.csv      # Held-out test evaluation metrics
│   └── train_rules_002_no_final_shot.csv    # Final-shot exclusion ablation results
└── src/
    ├── check_data.py                        # Raw dataset schema and integrity inspection
    ├── 01_build_rallies.py                  # Rally reconstruction and Server/Receiver role mapping
    ├── 02_mine_patterns.py                  # Contiguous sequence mining vs. unordered baseline
    ├── 03_validate_and_plot.py              # Holdout testing, permutation test, ablation, and plots
    └── 04_export_english_tables.py          # English stroke-label translation and table export
