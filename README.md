# Taglish Product Review Sentiment Classifier

This project trains an LSTM-based classifier from scratch to detect sentiment in Tagalog+English (code-switched) product reviews across general product categories.

- Target product category: General product reviews (mobile phones, chargers, headphones, beauty, home appliances, and other product types).

Notebooks:
- `data_processor.ipynb` — Taglish-safe text cleaning, tokenization, padding and train/val/test split.
- `model_builder.ipynb` — Builds the Keras LSTM model (Embedding + Bidirectional LSTM + Dropout + Dense).
- `train_runner.ipynb` — Training loop, evaluation, and experiment logging to `hyperparameter_logs.csv`.
- `predict.ipynb` — Demo notebook to load a saved model and run interactive predictions.

Dependencies:
Install the dependencies in `requirements.txt`:

```powershell
pip install -r requirements.txt
```

Quick start:
1. Open and run `data_processor.ipynb` to prepare your CSV dataset (`review_text`, `label`).
2. Run `model_builder.ipynb` to review model architecture.
3. Run `train_runner.ipynb` to train and log experiments.
4. Use `predict.ipynb` for interactive demo during your presentation.

Notes:
- Do not use pre-trained transformer models (BERT). Use the Keras Embedding layer (train-from-scratch).
- Ensure dataset contains at least ~3,000 Taglish reviews for meaningful results.
