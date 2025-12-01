import pandas as pd
from datetime import datetime
import os

log_entry = pd.DataFrame([{
    'dataset_name': 'kornwtp/lazada-review-fil-classification',
    'source': 'https://huggingface.co/datasets/kornwtp/lazada-review-fil-classification',
    'rows': 989,
    'size_kb': 84,
    'download_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'local_filename': 'kornwtp_lazada_review_fil.csv'
}])

log_path = 'logs/downloaded_datasets.csv'
os.makedirs('logs', exist_ok=True)

if os.path.exists(log_path):
    log = pd.read_csv(log_path)
    log = pd.concat([log, log_entry], ignore_index=True)
else:
    log = log_entry

log.to_csv(log_path, index=False)
print(f'Logged to {log_path}')
