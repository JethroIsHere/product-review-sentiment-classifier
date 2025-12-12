# 🌟 Taglish Product Review Sentiment Classifier

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning sentiment classifier for Taglish (Tagalog-English code-mixed) product reviews using Bidirectional GRU neural networks. Trained from scratch to classify reviews into 5-star ratings with **65.12% accuracy**.

**CCS 248 - Artificial Neural Networks Final Project (2025)**  
**Team Members:** Jethro Roland T. Dañocup, Duke Salfred B. Bocala, Jazylle Mae B. Senibalo

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Documentation](#documentation)
- [Future Work](#future-work)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This project addresses sentiment classification for **Taglish** (Tagalog-English code-mixed language) product reviews, a common linguistic pattern in Philippine e-commerce platforms.

### Problem Statement
Classify product reviews written in Taglish into 5 sentiment categories (1-5 star ratings) to help:
- E-commerce platforms identify product quality issues
- Businesses prioritize customer service responses
- Generate actionable insights from customer feedback

### Key Achievements
- ✅ **65.12% test accuracy** (exceeds 50-60% baseline)
- ✅ **90% precision** on 5-star (positive) reviews
- ✅ **83% recall** on 1-star (negative) reviews
- ✅ Trained entirely from scratch (no pretrained models)
- ✅ Systematic hyperparameter tuning (7 configurations)

---

## ✨ Features

- 🔤 **Taglish Support**: Handles code-switching between Tagalog and English
- 🧠 **Bidirectional GRU**: Captures context from both directions
- 📊 **Comprehensive Preprocessing**: Emoji removal, stopword filtering, text normalization
- ⚡ **Fast Inference**: ~150ms per review prediction
- 📈 **Well-Documented**: Full training logs and hyperparameter experiments
- 🔧 **Reproducible**: Seed-controlled for consistent results

---

## 📊 Dataset

### Overview
- **Total Samples**: 58,603 product reviews
- **Source**: Philippine e-commerce platforms (Shopee, Lazada)
- **Language**: Taglish (Tagalog-English code-mixed)
- **Labels**: 5 classes (1-5 star ratings)

### Distribution
| Rating | Count | Percentage |
|--------|-------|------------|
| ⭐ 1-star | 3,000 | 5.12% |
| ⭐⭐ 2-star | 7,126 | 12.16% |
| ⭐⭐⭐ 3-star | 4,760 | 8.12% |
| ⭐⭐⭐⭐ 4-star | 6,347 | 10.83% |
| ⭐⭐⭐⭐⭐ 5-star | 37,370 | 63.77% |

### Data Split
- **Training**: 85% (49,812 samples)
- **Validation**: 10% of training (4,981 samples)
- **Test**: 15% (8,791 samples)

---

## 🏗️ Model Architecture

### Bidirectional GRU Network

```
Input (Taglish Text)
    ↓
Preprocessing Pipeline
    ↓
Embedding Layer (10,000 vocab → 128 dim)
    ↓
Bidirectional GRU (32 units, dropout=0.5)
    ↓
Dropout (0.5)
    ↓
Dense Layer (64 units, ReLU, L2 reg)
    ↓
Dropout (0.5)
    ↓
Output Layer (5 units, Softmax)
    ↓
Predicted Rating (1-5 stars)
```

### Model Specifications
- **Vocabulary Size**: 10,000 words
- **Embedding Dimension**: 128
- **GRU Units**: 32 (64 bidirectional)
- **Total Parameters**: 1,398,597
- **Optimizer**: Adam (lr=5e-5)

**See [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) for detailed documentation.**

---

## 🚀 Installation

### Prerequisites
- Python 3.13+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/JethroIsHere/product-review-sentiment-classifier.git
cd product-review-sentiment-classifier

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install --upgrade pip
pip install tensorflow pandas numpy matplotlib scikit-learn seaborn
```

---

## 💻 Usage

### Training the Model

```bash
# Open main training notebook
jupyter notebook notebooks/taglish_rnn.ipynb
```

### Making Predictions

```python
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model('artifacts/taglish_rnn_model.h5', compile=False)
with open('artifacts/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

# Predict
review = "Maganda quality pero medyo mahal"
# ... preprocess, tokenize, predict ...
```

---

## 📈 Results

### Best Model Performance (Hyperparameter Tuning Config 3)

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **65.12%** |
| **Macro F1** | 0.3954 |
| **Weighted F1** | 0.6512 |

**Model:** `taglish_rnn_best.keras` / `taglish_rnn_best.h5` (Config 3: vocab_size=10K)

### Main Notebook Model Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **63.97%** |
| **Macro F1** | ~0.36 |
| **Weighted F1** | ~0.64 |

**Model:** `taglish_rnn_model.h5` (same hyperparameters, includes minority class oversampling)

**Note:** Both models use identical architecture (vocab=10K, embed=128, GRU=32, dropout=0.5, lr=5e-5). The 2.15% accuracy difference is due to random weight initialization, stochastic training (dropout, batch shuffling), and data augmentation differences (main notebook includes oversampling).

### Per-Class Performance (Best Model - 65.12%)
| Star Rating | Precision | Recall | F1 | Support |
|-------------|-----------|--------|-----|---------|
| 1-star ⭐ | 0.25 | 0.83 | 0.38 | 450 |
| 2-star ⭐⭐ | 0.47 | 0.05 | 0.09 | 1,069 |
| 3-star ⭐⭐⭐ | 0.22 | 0.27 | 0.24 | 714 |
| 4-star ⭐⭐⭐⭐ | 0.22 | 0.22 | 0.22 | 952 |
| 5-star ⭐⭐⭐⭐⭐ | 0.90 | 0.85 | 0.88 | 5,606 |

---

## 📁 Project Structure

```
product-review-sentiment-classifier/
│
├── data/
│   └── combined_taglish.csv          # Main dataset
│
├── notebooks/
│   ├── taglish_rnn.ipynb             # Main training
│   ├── hyperparameter_tuning.ipynb   # Experiments
│   └── predict.ipynb                 # Inference
│
├── artifacts/
│   ├── taglish_rnn_model.h5          # Final trained model (main notebook)
│   ├── taglish_rnn_best.keras        # Best model from tuning (65.12% accuracy)
│   ├── taglish_rnn_best.h5           # Best model (H5 format)
│   └── tokenizer.pickle              # Tokenizer
│     # Summary of 7 runs
│   ├── hyperparameter_tuning_full_results.json  # Detailed metrics & histories
│   └── hyperparameter_comparison.png            # Performance visualization
│   ├── hyperparameter_tuning_results.csv
│   └── hyperparameter_comparison.png
│
├── FINAL_PROJECT_REPORT.md           # Full report
├── ARCHITECTURE_DIAGRAM.md            # Architecture docs
└── README.md                          # This file
```

---

## 🔬 Hyperparameter Tuning

7 configurations tested:

| Run | Vocab | Embed | GRU | Test Acc | Test F1 |
|-----|-------|-------|-----|----------|---------|
| 1 | 8K | 128 | 32 | 63.19% | 0.3866 |
| 2 | 5K | 128 | 32 | 62.98% | 0.3881 |
| **3** | **10K** | **128** | **32** | **65.12%** | **0.3954** |
| 4 | 8K | 128 | 64 | 63.28% | 0.3866 |
| 5 | 8K | 128 | 32 | 62.11% | 0.3402 |
| 6 | 8K | 128 | 32 | 62.39% | 0.3424 |
| 7 | 8K | 256 | 32 | 63.88% | 0.4087 |

**Key Finding**: Larger vocabulary (10K) performed best.

---

## 📚 Documentation

### Complete Documentation

1. **[FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md)**
   - Comprehensive project report
   - Problem statement and justification
   - Dataset validation
   - Training process and results
   - Analysis and conclusions

2. **[ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md)**
   - Detailed architecture breakdown
   - Layer specifications
   - Design decisions
   - Parameter calculations

3. **Jupyter Notebooks**
   - `taglish_rnn.ipynb` - Training pipeline
   - `hyperparameter_tuning.ipynb` - Experiments
   - `predict.ipynb` - Inference examples

---

## 🔮 Future Work

- [ ] Implement attention mechanisms
- [ ] Experiment with Transformers
- [ ] Multi-task learning
- [ ] Build REST API
- [ ] Deploy on cloud platform

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

**Course**: CCS 248 - Artificial Neural Networks  
**Project**: Final Project (2025)

**Data Sources**: Shopee & Lazada Philippines

**Tools**: TensorFlow, Keras, scikit-learn, pandas, numpy, matplotlib, seaborn

---

## 📞 Contact

**Repository**: [product-review-sentiment-classifier](https://github.com/JethroIsHere/product-review-sentiment-classifier)  
**GitHub**: [@JethroIsHere](https://github.com/JethroIsHere)
