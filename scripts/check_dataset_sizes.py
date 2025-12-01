import os
import pandas as pd

files = [
    'data/hf_letijo03.csv',
    'data/marc3ee_datasetb_withpredictions.csv',
    'data/marc3ee_reviews.csv',
    'data/local_only/kornwtp_lazada_review_fil.csv',
    'data/local_only/mteb_filipino_shopee_reviews.csv',
    'data/local_only/scaredmeow_shopee_reviews_tl_stars.csv',
    'data/local_only/SEACrowd_shopee_reviews_tagalog.csv'
]

# Check seacrowd parts
seacrowd_parts = [
    'data/local_only/seacrowd_parts/train.csv',
    'data/local_only/seacrowd_parts/test.csv',
    'data/local_only/seacrowd_parts/validation.csv'
]

total_mb = 0
total_rows = 0

print('Individual dataset files:')
print('-' * 80)

for f in files:
    if os.path.exists(f):
        size = os.path.getsize(f)
        size_mb = size / (1024*1024)
        total_mb += size_mb
        try:
            df = pd.read_csv(f, encoding='utf-8', on_bad_lines='skip')
            rows = len(df)
            total_rows += rows
            print(f'{f:60s} | {size_mb:6.2f} MB | {rows:8,} rows')
        except Exception as e:
            print(f'{f:60s} | {size_mb:6.2f} MB | ERROR: {e}')
    else:
        print(f'{f:60s} | NOT FOUND')

print('\nSEACrowd parts:')
for f in seacrowd_parts:
    if os.path.exists(f):
        size = os.path.getsize(f)
        size_mb = size / (1024*1024)
        total_mb += size_mb
        try:
            df = pd.read_csv(f, encoding='utf-8', on_bad_lines='skip')
            rows = len(df)
            total_rows += rows
            print(f'{f:60s} | {size_mb:6.2f} MB | {rows:8,} rows')
        except Exception as e:
            print(f'{f:60s} | {size_mb:6.2f} MB | ERROR: {e}')

print('-' * 80)
print(f'TOTAL BEFORE DEDUP: {total_mb:6.2f} MB | {total_rows:8,} rows')
print()

combined_size = os.path.getsize('data/combined_taglish.csv')
combined_mb = combined_size / (1024*1024)
combined_df = pd.read_csv('data/combined_taglish.csv')

print(f'AFTER DEDUP (combined_taglish.csv): {combined_mb:6.2f} MB | {len(combined_df):8,} rows')
print()
print(f'REMOVED: {total_rows - len(combined_df):,} duplicate rows ({((total_rows - len(combined_df))/total_rows)*100:.1f}%)')
print(f'DEDUP saved: {total_mb - combined_mb:.2f} MB ({((total_mb - combined_mb)/total_mb)*100:.1f}%)')
