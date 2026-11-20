import pandas as pd

SHOT_MAP = {
    '殺球': 'Smash',
    '挑球': 'Lift',
    '放小球': 'Net_Shot',
    '擋小球': 'Net_Block',
    '長球': 'Clear',
    '切球': 'Drop',
    '過度切球': 'Transition_Drop',
    '撲球': 'Net_Kill',
    '發長球': 'Long_Serve',
    '發短球': 'Short_Serve',
    '防守回挑': 'Def_Lift',
    '防守回抽': 'Def_Drive',
    '推球': 'Push',
    '平球': 'Drive',
    '後場抽平球': 'Rear_Drive',
    '勾球': 'Cross_Net',
    '點扣': 'Stick_Smash',
    '小平球': 'Flat_Net',
    '未知球種': 'Unknown',
}


def translate_pattern(s):
  text = str(s)
  for zh, en in SHOT_MAP.items():
    text = text.replace(zh, en)
  return text


if __name__ == '__main__':
  train_df = pd.read_csv('results/train_rules_002_with_pvalues.csv')
  train_df['pattern_en'] = train_df['pattern'].apply(translate_pattern)
  train_df.to_csv('results/train_rules_002_english.csv', index=False)

  print('=== TOP 10 TRAIN RULES WITH PERMUTATION P-VALUES ===')
  print(
      train_df[[
          'pattern_en',
          'target_outcome',
          'count_total',
          'unique_matches',
          'seq_support',
          'confidence',
          'lift',
          'perm_p_value',
      ]]
      .head(10)
      .to_string(index=False)
  )

  test_df = pd.read_csv('results/test_holdout_evaluation_002.csv')
  test_df['pattern_en'] = test_df['pattern'].apply(translate_pattern)
  test_df.to_csv('results/test_holdout_evaluation_002_english.csv', index=False)

  print('\n=== TOP 10 HELD-OUT TEST EVALUATION (ENGLISH) ===')
  print(
      test_df[[
          'pattern_en',
          'target_outcome',
          'train_count',
          'train_lift',
          'holdout_count',
          'holdout_confidence',
          'holdout_lift',
      ]]
      .head(10)
      .to_string(index=False)
  )

  nf_df = pd.read_csv('results/train_rules_002_no_final_shot.csv')
  nf_df['pattern_en'] = nf_df['pattern'].apply(translate_pattern)
  nf_df.to_csv('results/train_rules_002_no_final_shot_english.csv', index=False)

  print('\n=== TOP 8 NO-FINAL-SHOT PATTERNS (ENGLISH) ===')
  print(
      nf_df[[
          'pattern_en',
          'target_outcome',
          'count_total',
          'unique_matches',
          'seq_support',
          'confidence',
          'lift',
      ]]
      .head(8)
      .to_string(index=False)
  )