import sys
import os
from datetime import datetime

try:
    from datasets import load_dataset
except Exception as e:
    print('datasets library not available:', e)
    raise

import pandas as pd

def sanitize_filename(name):
    return name.replace('/', '_').replace(' ', '_')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python download_hf.py <dataset_id> <out_csv_path>')
        sys.exit(1)

    dataset_id = sys.argv[1]
    out_path = sys.argv[2]

    print(f'Loading dataset: {dataset_id}')
    # Some community datasets (like SEACrowd) require executing remote dataset scripts.
    # Use `trust_remote_code=True` so datasets with custom loading code can be handled.
    # Callers should ensure they trust the dataset source before enabling this.
    try:
        ds = load_dataset(dataset_id, trust_remote_code=True)
    except TypeError:
        # Older datasets API may not accept the kwarg; fall back to default call
        ds = load_dataset(dataset_id)

    # try to get license if available
    info = None
    try:
        info = ds['train'].info if 'train' in ds else ds[list(ds.keys())[0]].info
        license_str = getattr(info, 'license', None)
    except Exception:
        license_str = None

    # concatenate all splits into one dataframe
    frames = []
    total_rows = 0
    for split_name, split in ds.items():
        try:
            df = pd.DataFrame(split)
        except Exception:
            # fallback: iterate manually
            df = pd.DataFrame(list(split))
        if df.shape[0] > 0:
            df['__hf_split'] = split_name
            frames.append(df)
            total_rows += df.shape[0]

    if len(frames) == 0:
        print('No rows found in dataset')
        sys.exit(1)

    full = pd.concat(frames, ignore_index=True)
    # add source column
    full['source'] = dataset_id

    # ensure the target folder exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    full.to_csv(out_path, index=False)

    print(f'Saved {total_rows} rows to {out_path}')

    # append to logs/downloaded_datasets.csv
    logs_path = os.path.join(os.path.dirname(os.path.dirname(out_path)), 'logs', 'downloaded_datasets.csv')
    ts = datetime.utcnow().isoformat()
    filename = out_path.replace('\\', '/')
    license_field = license_str if license_str is not None else 'unknown'

    log_row = f"{ts},{filename},{dataset_id},{total_rows},{license_field}\n"
    try:
        with open(logs_path, 'a', encoding='utf-8') as f:
            f.write(log_row)
        print('Appended log to', logs_path)
    except Exception as e:
        print('Failed to append log:', e)

    print('Done')
