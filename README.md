# Ticket Classifier

**Interview reference: [docs/interview/README.md](docs/interview/README.md)** — the numbers, the four evaluation mistakes, and the questions to have ready.

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

# What did labelling 389 tickets actually buy? (trained vs zero-shot)
python -m scripts.zero_shot

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
| Category | **86.9%** ± 3.6% | 25.7% | +61pp |
| Priority | **57.6%** ± 4.7% | 42.2% | +15pp |
| Sentiment | **68.1%** ± 7.1% | 51.7% | +16pp |

Baseline is "always guess the most common class". **Report it beside every
score** - the first sentiment model scored 58.7%, which reads as respectable
until you notice that answering "negative" to everything scored 56.5%. It had
learned nothing.

### How it got here

| | Category | Priority | Sentiment |
|---|---|---|---|
| 60 rows, RandomForest | 58.3% | 25.0% | crashed |
| 228 rows, LogisticRegression, tuned | 82.0% ± 9.5% | 53.9% | 67.5% |
| 389 rows, retuned, cleaned | **86.9% ± 3.6%** | **57.6%** | **68.1%** |

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

Sentiment reads 67.5% then 68.1% - flat. But the **baseline fell from 61.0% to
51.7%**, because the extra data brought the negative class down from 61% of the
set to 52%.

Same accuracy against a harder problem. The margin over guessing went from
+6.5pp to +15.1pp. The earlier score was partly a skewed dataset flattering the
model.

### Confidence is actionable

`predict_proba` is well calibrated, so a threshold becomes a routing rule. On the
78 held-out tickets:

    auto-accept >=0.5   52 of 78 automated (67%),  98% accurate
    auto-accept >=0.6   42 of 78 automated (54%), 100% accurate

At 0.6 the model is right about every ticket it commits to, and hands the rest to
a person. The alternative is automating all of them at 87%, which misfiles one in
eight and cannot say which.

---

## Four mistakes, all the same shape

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
  389 rows, 4 classes                     86.9% +/-3.6%   baseline 25.7%

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
+/-3.6% on 389 rows against +/-0.7% on 13,083.

---

## What is still weak

**Priority, at 57.6%.** Fifteen points over baseline, and neither more data nor
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

## What did labelling 389 tickets actually buy?

```bash
python -m scripts.zero_shot
```

The two numbers above - 86.9% trained, 25.7% guessing - make the labelling effort
look worth 61 points. But a model that has never seen one of our labels is not
restricted to guessing. `facebook/bart-large-mnli` is handed the four category
names in English and asked, per ticket, which one it entails. **No training, no
labels, runs locally.**

| Approach | Labels it saw | Accuracy |
|---|---:|---|
| *guess the most common class* | *—* | *25.7%* |
| **zero-shot** (pre-trained, no training) | **0** | **75.6% ±0.038** |
| **trained** (TF-IDF + logistic regression) | 311 | **86.9% ±0.036** |

**389 hand-written labels bought 11.3 points**, not 61. Trained wins on all five
folds, so the lead is real - but it is a quarter of what the baseline comparison
implied.

### The first 160 labels bought nothing

Zero-shot is a flat line: it never sees our data, so more rows do not help it.
The trained model starts far below and climbs. Where they cross is the number of
tickets that had to be written before training was worth doing at all.

| Labels written | Trained accuracy | vs zero-shot |
|---:|---|---|
| 10 | 31.4% ±0.039 | behind |
| 20 | 51.4% ±0.037 | behind |
| 40 | 61.4% ±0.040 | behind |
| 80 | 70.2% ±0.062 | behind |
| **160** | **78.7% ±0.043** | **ahead** |
| 240 | 83.5% ±0.055 | ahead |
| 311 | 86.9% ±0.036 | ahead |

**Roughly 160 tickets before training overtakes doing nothing.** Anyone labelling
a hundred rows and stopping would have been better off with the pre-trained model
and no dataset at all.

### Wording moves accuracy 18 points, with no training

The label names are the only input you control when there is no training data,
and the model reads them as English. Four ways of describing the same four
categories:

| Scheme | Hypothesis sent to the model | Accuracy |
|---|---|---|
| `descriptive` | *This customer support ticket is about a charge, refund, invoice or payment problem.* | **57.3%** |
| `bare` | *This example is billing.* | 59.6% |
| `noun_phrase` | *This example is billing and payments.* | 74.0% |
| `domain_framed` | *This customer support ticket is about billing and payments.* | **75.6%** |

**The most detailed wording came last.** Longer hypotheses give the entailment
model more ways to be partially satisfied by the wrong ticket, so precision
falls. More explanation is not more signal.

This is the zero-shot equivalent of feature engineering - and an 18-point spread
from wording alone means **any zero-shot number quoted without its label scheme
is close to meaningless.**

> **The mistake this caught.** The head-to-head defaulted to `descriptive`,
> because spelling the categories out in full seemed obviously best. It is the
> worst of the four, so the first run compared the trained model against a
> deliberately weakened opponent and reported the labelling effort as worth 29.6
> points instead of 11.3. The script now picks the best scheme **by measurement**
> rather than by assumption - comparing against a badly configured alternative is
> the most common way to produce a flattering result, and it is rarely deliberate.

### Zero-shot wins the category the trained model is worst at

| Category | n | Trained | Zero-shot | Gap |
|---|---:|---|---|---|
| account | 96 | 84.4% | 51.0% | **+33.3** |
| shipping | 97 | 93.8% | 82.5% | +11.3 |
| billing | 100 | 85.0% | 76.0% | +9.0 |
| **technical** | 96 | 84.4% | **92.7%** | **−8.3** |

`technical` is the label this README already identifies as the weak one - CORS
errors, slow pages, failed exports and spelling mistakes, four unrelated problems
sharing a label with no consistent vocabulary. **TF-IDF counts words, so a class
with no shared words is exactly what it cannot learn.** A pre-trained model reads
meaning instead, and breadth costs it far less.

The mirror image is `account`, where zero-shot manages 51.0%. "Account" is a word
that appears in tickets of every category, so as a hypothesis it entails almost
anything. The trained model learns from context that "login", "password" and
"reset" are what actually mark the class.

> **Which suggests the real answer is neither.** Route `technical` to the
> pre-trained model, keep the trained model for `account`, and the combination
> beats both. That is a finding the baseline-versus-trained comparison could not
> have produced.

### Cost

| | Trained | Zero-shot |
|---|---|---|
| Inference | **2 ms**, CPU | ~140 ms, Apple silicon GPU |
| Model size | 4 MB | 1.6 GB |
| Setup | none | 2.5 GB of dependencies |
| Labelling | **389 tickets by hand** | none |
| Per-query cost | $0 | $0 |

Both are free to run. They cost different things: one costs a dataset, the other
costs 70&times; the latency and a GPU to keep it reasonable.

---

## Why no LLM

Every other project in this workspace calls a model. This one asks whether it
needs to.

For classification with clean labels, TF-IDF plus a linear model is often the
better answer: **2ms on CPU, no API key, no quota, no cost**, and you can name
the words that drove the decision. The interesting question in applied ML is not
"which model is strongest" but "what is the cheapest thing that is good enough".
