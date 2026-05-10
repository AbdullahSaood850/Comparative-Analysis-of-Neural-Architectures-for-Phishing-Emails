# NLP Assignment 2 — Phishing Email Detection

> Comparative Analysis of Neural Architectures for Phishing Email Classification

| Field | Details |
|---|---|
| **Course Code** | AI4001 |
| **Course Title** | Fundamentals of Natural Language Processing |
| **Assignment** | 2 |
| **Roll Number** | 23F-0014 |
| **Dataset** | Phishing Email Dataset (Kaggle — subhajournal) |
| **Task** | Binary Classification: Phishing vs Legitimate Email |

---

## Overview

This project compares four models for detecting phishing emails:

| Module | Model | Input Representation |
|---|---|---|
| 1A | Logistic Regression | TF-IDF (5000 features, bigrams) |
| 1B | Feedforward Neural Network (FNN) | TF-IDF (dense) |
| 1C | Recurrent Neural Network (RNN) | Learned Embeddings (10k vocab × 64D) |
| 1D | Long Short-Term Memory (LSTM) | Learned Embeddings (10k vocab × 64D) |

---

## Dataset

**Phishing Email Dataset** — [Kaggle (subhajournal)](https://www.kaggle.com/datasets/subhajournal/phishingemails)

- File: `Phishing_Email.csv`
- Columns: `Email Text`, `Email Type`
- Labels: `Phishing Email` → `phishing`, `Safe Email` → `legitimate`

---

## Project Structure

```
NLP_Assignment_2_23F-0014.ipynb   # Main notebook
README_NLP_A2.md                  # This file
NLP_A2_Report.pdf                 # Full project report
```

---

## Setup & Requirements

### Install Dependencies

```bash
pip install pandas numpy scikit-learn tensorflow keras matplotlib
```

### Run on Kaggle

1. Upload the notebook to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Add the dataset: **Phishing Email Dataset** (subhajournal/phishingemails)
3. Enable GPU accelerator (recommended for RNN/LSTM)
4. Run all cells top to bottom

---

## Preprocessing Pipeline

```
Raw Email Text
     │
     ▼
Lowercase → Remove digits → Remove punctuation → Strip whitespace
     │
     ▼
  clean_text
     │
     ├── TF-IDF Vectorizer (max_features=5000, ngram_range=(1,2))
     │         └──► Logistic Regression / FNN
     │
     └── Keras Tokenizer (num_words=10000) + Pad Sequences (maxlen=200)
               └──► RNN / LSTM
```

---

## Model Architectures

### Logistic Regression (Baseline)
```
TF-IDF (5000D) → LogisticRegression(C=1.0, solver='liblinear')
```

### Feedforward Neural Network (FNN)
```
TF-IDF (5000D) → Dense(256, ReLU) → Dropout(0.5)
              → Dense(128, ReLU) → Dropout(0.3)
              → Dense(1, sigmoid)
```

### Recurrent Neural Network (RNN)
```
Embedding(10000, 64, input_length=200)
    → SimpleRNN(64, tanh)
    → Dense(1, sigmoid)
```

### LSTM (Best Model)
```
Embedding(10000, 64, input_length=200)
    → LSTM(64, dropout=0.3, recurrent_dropout=0.2)
    → Dense(1, sigmoid)
```

---

## Training Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Loss | Binary Cross-Entropy |
| Batch Size | 32 |
| Max Epochs | 20 (FNN), 10 (RNN/LSTM) |
| Early Stopping | patience=6 (FNN), patience=3 (LSTM) |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Train/Val/Test Split | 80% / 10% / 10% (stratified) |

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.974 | ~0.975 | ~0.972 | ~0.973 | ~0.997 |
| FNN | ~0.978 | ~0.979 | ~0.976 | ~0.977 | ~0.998 |
| RNN | ~0.968 | ~0.965 | ~0.970 | ~0.967 | ~0.993 |
| **LSTM ★** | **~0.982** | **~0.981** | **~0.984** | **~0.982** | **~0.999** |

> ★ **LSTM selected for deployment** — highest recall (fewest missed phishing emails)

---

## Why Recall Matters Most

In phishing detection, **recall = most important metric**:

- **False Negative** (missed phishing) → user receives malicious email → security breach
- **False Positive** (legitimate flagged as phishing) → user inconvenience only

LSTM's superior recall (~98.4%) makes it the safest choice for real-world deployment.

---

## Adversarial Robustness

A manually crafted phishing email was tested:

```
"Your account has been suspended! Click this link to verify immediately:
 http://fakebank.com/login"
```

LSTM predicted phishing probability: **0.9814** ✓ Correct

---

## Key Findings

- **LSTM** is the best model — highest recall, AUC, and lowest Brier Score
- **Logistic Regression** is surprisingly competitive (AUC ~0.997) due to strong lexical signals in phishing emails
- **RNN** underperforms LSTM due to the vanishing gradient problem with long email sequences
- **FNN** bridges the gap between LR and sequence models but lacks positional awareness

---

## Evaluation Metrics Used

| Metric | Purpose |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | Of flagged phishing, how many are real? |
| Recall | Of all phishing, how many were caught? |
| F1-Score | Balance of Precision and Recall |
| AUC-ROC | Discrimination ability across thresholds |
| Brier Score | Probability calibration quality |
| Confusion Matrix | Breakdown of TP / FP / TN / FN |

---

## Limitations & Future Work

**Limitations:**
- Fixed vocabulary (10k words) may miss novel phishing terminology
- No URL, header, or HTML feature extraction
- LSTM not fine-tuned from a pretrained model

**Improvements:**
- Fine-tune BERT/RoBERTa on the phishing dataset
- Add URL and email metadata as additional features
- Threshold tuning based on acceptable false negative rate
- Ensemble LSTM + Logistic Regression for robustness

---

## References

- Subha Journal. *Phishing Email Dataset*. Kaggle. https://www.kaggle.com/datasets/subhajournal/phishingemails
- Hochreiter & Schmidhuber (1997). *Long Short-Term Memory*. Neural Computation.
- TensorFlow/Keras Documentation. https://www.tensorflow.org
- Scikit-learn Documentation. https://scikit-learn.org
