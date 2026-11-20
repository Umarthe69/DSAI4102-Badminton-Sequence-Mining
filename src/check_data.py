import os, glob, pandas as pd
base_path = r'data\\raw'
csv_files = glob.glob(os.path.join(base_path, '**', '*.csv'), recursive=True)
print('Total CSV files found:', len(csv_files))
for f in csv_files[:5]: print(' -', f)
sample_files = [f for f in csv_files if os.path.basename(f).lower() not in ['match.csv', 'homography.csv']]
if sample_files:
    df = pd.read_csv(sample_files[0])
    print('\nSample file:', sample_files[0])
    print('Columns:', list(df.columns))
    print(df.head(3))
