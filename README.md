# Ticket Classifier

Classify customer support tickets by **category** and **priority** using traditional ML (no LLM).

## How It Works

```
"My payment failed and I can't login"
         │
         ▼
┌─────────────────────┐
│  Text Preprocessing  │  → lowercase, remove stopwords, tokenize
├─────────────────────┤
│  Feature Extraction  │  → TF-IDF vectorization
├─────────────────────┤
│    ML Classifier     │  → Random Forest / SVM
└─────────┬───────────┘
          │
          ▼
Category: "billing"  |  Priority: "high"  |  Confidence: 0.92
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    FastAPI Server                      │
│                    (api.py)                           │
├──────────────────────────────────────────────────────┤
│  POST /predict     → classify a ticket               │
│  POST /train       → retrain the model               │
│  GET  /metrics     → model accuracy, confusion matrix │
├──────────────────────────────────────────────────────┤
│              ML Pipeline (src/)                       │
│  preprocessor.py → feature extraction (TF-IDF)        │
│  classifier.py   → train, predict, evaluate           │
│  data_loader.py  → load and split dataset             │
└──────────────────────────────────────────────────────┘
```

## ML Pipeline

```
Raw text → Clean → TF-IDF vectors → Train model → Save model → Serve via API
```

| Step | What it does | Code |
|------|-------------|------|
| 1. Load data | Read CSV with tickets + labels | `data_loader.py` |
| 2. Preprocess | Clean text, remove noise | `preprocessor.py` |
| 3. Feature extraction | Convert text → numbers (TF-IDF) | `preprocessor.py` |
| 4. Train | Fit Random Forest on features | `classifier.py` |
| 5. Evaluate | Accuracy, precision, recall, F1 | `classifier.py` |
| 6. Save | Persist model to disk (joblib) | `classifier.py` |
| 7. Serve | FastAPI endpoint for predictions | `api.py` |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Train the model
python -m scripts.train

# Start the API server
uvicorn api:app --reload

# Classify a ticket
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "My payment failed and I cannot login"}'
```

## Test

```bash
pytest
```

## Key ML Concepts (Interview Topics)

| Concept | What it means |
|---------|--------------|
| **TF-IDF** | Converts text to numbers based on word importance |
| **Train/Test Split** | 80% train, 20% test — prevents overfitting |
| **Precision** | Of all predicted "billing", how many were actually billing? |
| **Recall** | Of all actual "billing" tickets, how many did we find? |
| **F1 Score** | Balance between precision and recall |
| **Confusion Matrix** | Table showing correct vs wrong predictions |
| **Overfitting** | Model memorizes training data, fails on new data |
| **Cross-validation** | Train/test on different splits to get reliable metrics |

## Tech Stack

| Technology | What it does | Why not LLM? |
|-----------|-------------|-------------|
| scikit-learn | ML training & prediction | Faster, cheaper, deterministic |
| TF-IDF | Text → feature vectors | No GPU needed, works on CPU |
| Random Forest | Classification algorithm | Handles text well, interpretable |
| FastAPI | REST API server | Serve predictions |
| joblib | Save/load trained model | Persist model to disk |

---

## Results

389 hand-written tickets. Every figure is **stratified 5-fold cross-validation
over all rows**, not a single split - on a 78 row test set a four point
difference is three tickets, which is noise.

| Target | Accuracy | Baseline | Margin |
|---|---|---|---|
| Category | **86.9%** ± 3.8% | 25.7% | +61pp |
| Priority | **56.8%** ± 4.2% | 42.2% | +15pp |
| Sentiment | **66.8%** ± 4.6% | 51.7% | +15pp |

Baseline is "always guess the most common class". **Report it beside every
score** - the first sentiment model scored 58.7%, which reads as respectable
until you notice that answering "negative" to everything scored 56.5%. It had
learned nothing.

### How it got here

| | Category | Priority | Sentiment |
|---|---|---|---|
| 60 rows, RandomForest | 58.3% | 25.0% | crashed |
| 228 rows, LogisticRegression, tuned | 82.0% ± 9.5% | 53.9% | 67.5% |
| 389 rows, retuned | **86.9% ± 3.8%** | **56.8%** | **66.8%** |

Three changes, in order of how much they mattered:

1. **More data.** 60 to 389 rows. Note the error bar on category: ± 9.5% became
   ± 3.8%. More rows did not only raise the score, it made the score *mean
   something* - at 228 rows any single split landed ten points either side of
   the truth.
2. **The right algorithm.** LogisticRegression over RandomForest. TF-IDF is
   sparse and high dimensional; trees split one feature at a time, linear models
   weigh every word at once.
3. **Tuning.** Mostly `C` - the default of 1.0 regularises far too hard on a few
   hundred rows and the model was being penalised into underfitting.

### The sentiment number that did not move, and why that is good

Sentiment reads 67.5% then 66.8% - flat. But the **baseline fell from 61.0% to
51.7%**, because the extra data brought the negative class down from 61% of the
set to 52%.

Same accuracy against a harder problem. The margin over guessing went from
+6.5pp to +15.1pp. The earlier score was partly a skewed dataset flattering the
model.

### Confidence is actionable

`predict_proba` is well calibrated, so a threshold becomes a routing rule:
auto-file above 0.5, send the rest to a person. On the held-out set that
automated 11 of 46 tickets with no errors. A 24% automation rate is modest and
honest - the alternative is automating everything at 87%, which misfiles one
ticket in eight and cannot say which.

---

## Three mistakes, all the same shape

Each one produced a number that looked like a result and was not. None of them
threw an error.

**100% accuracy, scored on the training data.** The teaching script in `learn/`
trained and tested on the same rows. A model that memorises scores perfectly and
is useless on anything new. The honest figure was 58%.

**A grid search that saw its own test set.** Tuning ran over all 228 rows.
Cross-validation reported 83% for category while the held-out score *fell* to
70%. Re-run on the training split alone, the same search chose different
parameters and scored 84.8%.

> Any decision made by looking at the test set stops the test set being a test.

**A single split reported as a result.** That 84.8% was a lucky draw - a
stratified split of the same data gave 73.9%. Only cross-validation over every
row gives a number worth quoting.

---

## Benchmarked against real data

The 389 tickets here were hand-written, and synthetic data is usually easier
than reality. `scripts/benchmark_real_data.py` runs the identical pipeline
against **banking77** - 13,083 genuine customer service queries, public, no API
key:

```
OURS - hand written, broad categories
  389 rows, 4 classes                     86.9% +/-3.8%   baseline 25.7%

REAL - banking77, narrow intents, 100 rows each
  distinct intents                        99.5% +/-0.6%   baseline 25.0%
  four kinds of fee (confusable)          95.2% +/-2.2%   baseline 25.0%
  four kinds of failure (confusable)      96.5% +/-1.7%   baseline 25.0%

REAL - banking77, everything
  13,083 rows, 77 classes                 84.7% +/-0.7%   baseline  1.7%
```

**The prediction was wrong, and that is the finding.** Real data was expected to
score lower and expose the synthetic set as too clean. It scored *higher* -
even on four intents deliberately chosen to overlap.

The cause is class **breadth**, not provenance. banking77's intents are narrow:
`card_arrival` is one question asked many ways. Our `technical` covers CORS
errors, slow pages, failed exports and spelling mistakes - four unrelated
problems sharing one label, and no consistent vocabulary to learn from.

So: **how the classes are defined matters more than whether the text is real.**
These numbers are not inflated by synthetic data; they are limited by a harder
labelling scheme. Splitting `technical` into `api`, `performance` and `ui` would
probably do more for accuracy than another two hundred rows.

And the error bars make the sample size argument better than any explanation:
+/-3.8% on 389 rows against +/-0.7% on 13,083.

---

## What is still weak

**Priority, at 56.8%.** Fifteen points over baseline, and neither more data nor
tuning moved it much. Urgency is often not in the text - "my invoice is wrong" is
urgent or not depending on the amount, which the ticket does not say. That is a
finding, not a bug.

**Roughly one ticket in twenty is genuinely ambiguous.** "Can I get a copy of the
receipt sent to a second email" is billing or account depending on how you read
it. That puts a ceiling near 95% that no amount of data removes.

**Sentiment is the wrong tool for the job.** Category lives in nouns - refund,
parcel, password - which is exactly what TF-IDF represents. Sentiment lives in
tone and negation, which word counts cannot capture. A small pre-trained model
would beat this, still locally and still free.

Knowing which problem needs which tool is the point: **category is TF-IDF because
the signal is lexical; sentiment needs a language model because the signal is
grammatical.**

---

## Why no LLM

Every other project in this workspace calls a model. This one asks whether it
needs to.

For classification with clean labels, TF-IDF plus a linear model is often the
better answer: **2ms on CPU, no API key, no quota, no cost**, and you can name
the words that drove the decision. The interesting question in applied ML is not
"which model is strongest" but "what is the cheapest thing that is good enough".
