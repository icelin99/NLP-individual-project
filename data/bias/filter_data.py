import pandas as pd

df = pd.read_csv('./data/bias/crows_pairs_anonymized.csv')

filtered_df = df[(df['bias_type'] == 'sexual-orientation')]

filtered_df.to_csv('./data/bias/sexual_crows_pairs.csv',index=False)