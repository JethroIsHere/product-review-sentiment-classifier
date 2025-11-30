import os
import sys
from datetime import datetime
from urllib.request import urlretrieve

try:
    import pandas as pd
except Exception as e:
    print('pandas not available:', e)
    raise

BASE_URL = "https://huggingface.co/datasets/scaredmeow/shopee-reviews-tl-stars/resolve/main"
FILES = {
    'train': f"{BASE_URL}/train.csv",
    'validation': f"{BASE_URL}/validation.csv",
    'test': f"{BASE_URL}/test.csv",
}

OUT_DIR = os.path.join('data', 'local_only')
OUT_COMBINED = os.path.join(OUT_DIR, 'SEACrowd_shopee_reviews_tagalog.csv')

def download_parts(tmp_dir):
    os.makedirs(tmp_dir, exist_ok=True)
    local_paths = {}
    for split, url in FILES.items():
        out_path = os.path.join(tmp_dir, f"{split}.csv")
        print('Downloading', url)
        urlretrieve(url, out_path)
        local_paths[split] = out_path
    return local_paths

def concat_and_save(local_paths, out_path):
    frames = []
    total = 0
    for split, p in local_paths.items():
        df = pd.read_csv(p)
        df['__hf_split'] = split
        frames.append(df)
        total += len(df)
    if len(frames) == 0:
        print('No data downloaded')
        sys.exit(1)
    full = pd.concat(frames, ignore_index=True)
    full['source'] = 'SEACrowd/shopee_reviews_tagalog'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    full.to_csv(out_path, index=False)
    return total

def append_log(out_path, rows, license_str='mpl-2.0'):
    logs_path = os.path.join('logs', 'downloaded_datasets.csv')
    ts = datetime.utcnow().isoformat()
    filename = out_path.replace('\\', '/')
    row = f"{ts},{filename},SEACrowd/shopee_reviews_tagalog,{rows},{license_str}\n"
    try:
        with open(logs_path, 'a', encoding='utf-8') as f:
            f.write(row)
        print('Appended log to', logs_path)
    except Exception as e:
        print('Failed to append log:', e)

if __name__ == '__main__':
    tmp = os.path.join('data', 'local_only', 'seacrowd_parts')
    paths = download_parts(tmp)
    total_rows = concat_and_save(paths, OUT_COMBINED)
    print(f'Saved {total_rows} rows to {OUT_COMBINED}')
    append_log(OUT_COMBINED, total_rows, license_str='mpl-2.0')
