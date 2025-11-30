# Taglish Product-Review Sentiment Classifier

A compact, notebook-first project for building and testing a sentiment classifier for Taglish product reviews. The repository contains a data normalizer, an end-to-end training notebook (preprocessing → tokenization → RNN training → evaluation), and quick inference helpers so you can test examples locally.

---

## What this repository contains

- `data/combined_taglish.csv` — the canonical, merged Taglish product-review CSV produced by the normalizer (if present).
- `scripts/normalize_merge.py` — dataset normalizer and merger. Reads dataset CSVs, infers text/rating columns, normalizes ratings, deduplicates, and writes `data/combined_taglish.csv`.
- `notebooks/taglish_rnn.ipynb` — the main, self-contained notebook: preprocessing, tokenizer, model build (RNN), experiments, training and evaluation, plus an export cell to save artifacts.
- `notebooks/artifacts/` — saved artifacts created by the notebook (model, tokenizer, processed sequences). Typical files: `taglish_rnn_model.h5`, `tokenizer.pickle`, `sequences.npy`, `labels.npy`.
- `scripts/run_prediction.py` — a small helper script for quickly running one-off predictions from the saved artifacts (requires TensorFlow installed).
- `logs/downloaded_datasets.csv` — dataset provenance and download logs (if present).
- `archive/` — an archive folder for raw/duplicate files and large downloads (ignored by `.gitignore`).

---

## Quick setup (local)

1. Open a PowerShell terminal and change to the project folder:

```powershell
cd "C:\Users\LENOVO\OneDrive\Desktop\Artificial Neural Networks\Final Project"
```

2. Create a virtual environment (recommended) and activate it, or use your preferred Python environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install required packages (only what you need). Installing TensorFlow is optional unless you want to run/train the model locally.

```powershell
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn seaborn matplotlib
# If you want to train or run the saved Keras model locally, install tensorflow
pip install tensorflow
```

---

## Running the notebook

- Launch your notebook server or open `notebooks/taglish_rnn.ipynb` in VS Code.
- The notebook is written to run end-to-end from `data/combined_taglish.csv` (no external downloads during execution). Steps included:
  - Preprocessing (lowercase, emoji removal, repeated-letter normalization, punctuation cleaning, stopword removal)
  - Tokenization (Keras Tokenizer, configurable vocabulary)
  - Model building (Embedding + Bidirectional GRU + Dense head) and training
  - Evaluation (classification report / confusion matrix)
  - Export (saves model and tokenizer to `notebooks/artifacts/`)
- If you only want to try inference, run the export cell first (or ensure artifacts are present), then use the quick prediction cell at the end of the notebook.

---

## Quick inference (two options)

Option A — Notebook interactive cell (recommended):
- Open `notebooks/taglish_rnn.ipynb` and run the last cell labeled "Quick Prediction (Test your own inputs)". You can either set the `SAMPLE` variable in that cell to a string and run, or run the cell and paste your review when prompted.

Option B — Script helper:
- Ensure `notebooks/artifacts/taglish_rnn_model.h5` and `notebooks/artifacts/tokenizer.pickle` exist.
- Run:

```powershell
python .\scripts\run_prediction.py
```

This script prints the predicted 1–5 rating, a 3-class mapping (Good / Neutral / Bad), and per-class probabilities.

---

## Reproducing the combined dataset

- If you need to regenerate `data/combined_taglish.csv`, run `scripts/normalize_merge.py` (ensure raw CSVs are present under `data/` or `data/local_only/`). The normalizer handles Latin-1 fallback for non-UTF-8 CSVs and logs provenance to `logs/downloaded_datasets.csv` when available.

---

## Files you may want to inspect

- `scripts/normalize_merge.py` — how different source CSVs are read and normalized.
- `notebooks/taglish_rnn.ipynb` — model-building and experiments; check the export and quick-prediction cells for inference helpers.
- `notebooks/artifacts/` — the exported model and tokenizer used for inference.

---

## Tips & troubleshooting

- If `run_prediction.py` fails with `ModuleNotFoundError: No module named 'tensorflow'`, install TensorFlow (see setup section) or use the notebook to run inference in a different environment.
- If the notebook complains about missing `data/combined_taglish.csv`, either run the normalizer to rebuild it or place your CSV at that path with a `rating` column (1..5) and a text column (`text`, `review`, `content` or the first column will be used).
- The notebook contains a `model.summary()` table cell (displays layer table up to "Non-trainable params") and a single `model.summary()` occurrence so outputs are not repeated.

---

## Attribution & notes

- This project aims to be reproducible and notebook-first: the main notebook runs end-to-end from the combined CSV and saves artifacts for inference.
- Keep the `archive/` and `data/local_only/` directories in `.gitignore` (they contain raw downloads and large files).

---

If you want, I can split this README into a shorter top-level summary and a longer `docs/` page with full experiment logs, hyperparameters and metric comparisons. Tell me which you'd prefer and I can add it next.
