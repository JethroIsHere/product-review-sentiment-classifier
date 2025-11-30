import os
import pandas as pd
from pathlib import Path

DATA_DIRS = [Path('data'), Path('data') / 'local_only']
OUT_PATH = Path('data') / 'combined_taglish.csv'

def infer_and_extract(df, filename):
    # Return DataFrame with columns: text, rating (1-5), original_label, source
    src = filename.replace('\\', '/')
    cols = [c.lower() for c in df.columns]
    # determine text column
    if 'text' in df.columns:
        text_col = 'text'
    elif 'comment' in cols:
        # find original case
        text_col = df.columns[cols.index('comment')]
    elif 'comment' in df.columns:
        text_col = 'comment'
    elif 'comment' in cols:
        text_col = df.columns[cols.index('comment')]
    elif 'comment' in df.columns:
        text_col = 'Comment'
    else:
        # fallback: try common candidates
        for cand in ['review', 'comment_text', 'review_text']:
            if cand in df.columns:
                text_col = cand
                break
        else:
            # last resort: first string column
            text_col = None
            for c in df.columns:
                if df[c].dtype == object:
                    text_col = c
                    break
    # determine rating column
    rating_col = None
    if 'rating' in [c.lower() for c in df.columns]:
        # pick the first matching
        for c in df.columns:
            if c.lower() == 'rating':
                rating_col = c
                break
    elif 'label' in [c.lower() for c in df.columns]:
        for c in df.columns:
            if c.lower() == 'label':
                rating_col = c
                break
    elif 'score' in [c.lower() for c in df.columns]:
        for c in df.columns:
            if c.lower() == 'score':
                rating_col = c
                break

    records = []
    for _, row in df.iterrows():
        text = None
        if text_col is not None:
            text = row.get(text_col)
        else:
            # skip if no text identified
            continue
        if pd.isna(text):
            continue

        original_label = None
        rating = None
        if rating_col is not None:
            original_label = row.get(rating_col)
            # handle numeric labels stored as float/int
            try:
                if pd.isna(original_label):
                    rating = None
                else:
                    # many HF Tagalog sets use 0..4 -> map to 1..5
                    if isinstance(original_label, (int, float)):
                        val = int(original_label)
                        if 0 <= val <= 4:
                            rating = val + 1
                        else:
                            rating = val
                    else:
                        # string labels: try to extract digits
                        s = str(original_label).strip()
                        # handle labels like '1positive' or '0negative'
                        if s.endswith('positive') or s.endswith('negative'):
                            if s.startswith('1') or s.startswith('pos') or 'positive' in s:
                                rating = 5
                            else:
                                rating = 1
                        else:
                            import re
                            m = re.search(r"(\d+)", s)
                            if m:
                                v = int(m.group(1))
                                # if v in 0..4 map to 1..5
                                if 0 <= v <= 4:
                                    rating = v + 1
                                elif 1 <= v <= 5:
                                    rating = v
                                else:
                                    rating = None
                            else:
                                rating = None
            except Exception:
                rating = None

        rec = {
            'text': str(text).strip(),
            'rating': int(rating) if rating is not None else None,
            'original_label': original_label,
            'source': src,
        }
        records.append(rec)

    return pd.DataFrame(records)

def map_label_3class(rating):
    if rating is None:
        return None
    try:
        r = int(rating)
    except Exception:
        return None
    if r >= 4:
        return 2
    if r == 3:
        return 1
    return 0

def main():
    frames = []
    for d in DATA_DIRS:
        if not d.exists():
            continue
        for p in d.rglob('*.csv'):
            # skip combined output if re-run
            if p.resolve() == OUT_PATH.resolve():
                continue
            try:
                df = pd.read_csv(p)
            except Exception:
                try:
                    df = pd.read_csv(p, encoding='utf-8', engine='python')
                except Exception:
                    # fallback to latin-1 for files with non-utf8 encoding
                    try:
                        df = pd.read_csv(p, encoding='latin-1')
                        print(f'Read {p} with latin-1 fallback')
                    except Exception as e:
                        print('Failed to read', p, e)
                        continue
            print('Processing', p)
            extracted = infer_and_extract(df, str(p))
            # Skip files that don't provide any rating information (likely DB tables)
            if 'rating' in extracted.columns:
                nr = extracted['rating'].notna().sum()
            else:
                nr = 0
            if nr < 1:
                print(f"Skipping {p} — no rating values found ({nr})")
                continue
            if extracted.shape[0] > 0:
                frames.append(extracted)

    if len(frames) == 0:
        print('No data found to merge')
        return

    full = pd.concat(frames, ignore_index=True)
    # drop empty texts
    full['text'] = full['text'].astype(str).str.strip()
    full = full[full['text'] != '']
    # compute rating and label_3class
    full['label_3class'] = full['rating'].apply(map_label_3class)
    # dedupe by text
    before = len(full)
    full = full.drop_duplicates(subset=['text']).reset_index(drop=True)
    after = len(full)
    print(f'Deduplicated {before-after} rows; {after} remain')

    # Fill source where missing
    full['source'] = full['source'].fillna('local')

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUT_PATH, index=False)
    print('Wrote combined file to', OUT_PATH)

    # report stats
    print('\nClass distribution (label_3class):')
    print(full['label_3class'].value_counts(dropna=False))
    total_bytes = OUT_PATH.stat().st_size
    print('Combined CSV size bytes:', total_bytes, 'MB:', round(total_bytes/1024/1024,4))

if __name__ == '__main__':
    main()
