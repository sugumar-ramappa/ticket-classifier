# Interview reference — Ticket Classifier

**To perform it: [demo-script.md](demo-script.md)** — six minutes, in order, with the sentences to say.

The counterweight project. Every other application in this workspace calls a
language model; this one asks whether it needs to.

---

## The 60-second version

> Classifies support tickets by category, priority and sentiment using classical
> ML — TF-IDF into logistic regression. **No LLM, no API key, no network, no
> quota.** A prediction is a matrix multiply: about 2ms on CPU, and you can name
> the exact words that drove it.
>
> 389 hand-written tickets, four categories. Measured with **stratified 5-fold
> cross-validation**, never a single split: category **86.9% ±3.6%** against a
> 25.7% majority-class baseline.
>
> The number I would actually talk about is the ±3.6%. It was ±9.5% at 228 rows,
> and at that size a single train/test split landed ten points either side of the
> truth — which is how I ended up reporting 84.8% and having to walk it back.

---

## Why this project exists

For text classification with clean labels, an LLM is usually the wrong tool:

| | LLM call | TF-IDF + logistic regression |
|---|---|---|
| Latency | ~1–30s | **~2ms** |
| Cost | per call | **zero** |
| Determinism | sampled | **exact** |
| Explainable | "the model said so" | **the weighted words** |
| Offline | no | **yes** |

The interesting question in applied ML is not *"which model is strongest"* but
*"what is the cheapest thing that is good enough"* — and here the cheapest thing
runs on a laptop for nothing.

---

## The numbers, with their baselines

Stratified 5-fold cross-validation over all 389 rows. Every ticket is in the test
set exactly once.

| Target | Accuracy | Baseline | Margin |
|---|---|---|---|
| Category | **86.9% ±3.6%** | 25.7% | +61pp |
| Priority | **57.6% ±4.7%** | 42.2% | +15pp |
| Sentiment | **68.1% ±7.1%** | 51.7% | +16pp |

**Baseline means "always guess the most common class". Report it beside every
score.** The first sentiment model scored 58.7%, which reads as respectable until
you notice that answering "negative" to everything scored 56.5%. It had learned
nothing and was predicting the majority class — the desert weather forecaster who
says "no rain" every day and is right 95% of the time.

### How it got there

| | Category | Priority | Sentiment |
|---|---|---|---|
| 60 rows, RandomForest | 58.3% | 25.0% | crashed |
| 228 rows, LogisticRegression, tuned | 82.0% ±9.5% | 53.9% | 67.5% |
| 389 rows, retuned, cleaned | **86.9% ±3.6%** | **57.6%** | **68.1%** |

Three changes, in order of how much they mattered:

1. **More data.** 60 → 389 rows. Note that the error bar more than halved — extra
   rows did not only raise the score, they made the score *mean something*.
2. **The right algorithm.** TF-IDF produces a few hundred sparse features per
   ticket, mostly zeros. Trees split one feature at a time and cope badly with
   that shape; a linear model weighs every word at once. Measured across all
   three targets, every fold: linear won.
3. **Tuning**, mostly `C`. The default of 1.0 regularises far too hard on a few
   hundred rows — the model was being penalised into underfitting.

### The sentiment score that stayed flat, and why that is good

Sentiment read 67.5%, then 68.1%. Barely moved.

But the **baseline fell from 61.0% to 51.7%**, because the extra data brought the
negative class down from 61% of the set to 52%. Same accuracy against a harder
problem: the margin over guessing went from +6.5pp to +16pp.

The earlier number was partly a skewed dataset flattering the model.

---

## Benchmarked against real data — and the prediction was wrong

The 389 tickets are hand-written, and synthetic data is usually easier than
reality. `scripts/benchmark_real_data.py` runs the identical pipeline against
**banking77** — 13,083 genuine customer service queries, public, no API key:

```
OURS   389 rows, 4 broad classes            86.9% ±3.6%   baseline 25.7%

REAL   banking77, distinct intents          99.5% ±0.6%   baseline 25.0%
REAL   banking77, four kinds of fee         95.2% ±2.2%   baseline 25.0%
REAL   banking77, four kinds of failure     96.5% ±1.7%   baseline 25.0%
REAL   banking77, all 77 classes            84.7% ±0.7%   baseline  1.7%
```

Real data was expected to score **lower** and expose the synthetic set as too
clean. It scored higher — even on four intents chosen deliberately to overlap.

**The cause is class breadth, not provenance.** banking77's intents are narrow:
`card_arrival` is one question asked many ways. Our `technical` covers CORS
errors, slow pages, failed exports and spelling mistakes — four unrelated
problems under one label, with no consistent vocabulary to learn.

So the finding is the opposite of the hypothesis: **how the classes are defined
matters more than whether the text is real.** Splitting `technical` into `api`,
`performance` and `ui` would likely do more than another 200 rows.

And the error bars make the sample-size argument better than any explanation
could: ±3.6% on 389 rows against ±0.7% on 13,083.

---

## Confidence is a routing rule, not decoration

`predict_proba` returns a probability per class, and it is well calibrated:

| Confidence | Tickets | Accuracy |
|---|---|---|
| below 0.4 | 10 | 80% |
| 0.4 – 0.5 | 16 | 88% |
| 0.5 – 0.7 | 23 | 96% |
| above 0.7 | 29 | **100%** |

So a threshold becomes a real operating decision. On the 78 held-out tickets:

```
auto-accept >=0.5    52 of 78 automated (67%)   98% accurate   26 to a human
auto-accept >=0.6    42 of 78 automated (54%)  100% accurate   36 to a human
```

**At 0.6 it is right about every ticket it commits to.** The alternative is
automating all of them at 87% — which misfiles one in eight and cannot tell you
which.

Live example of it working:

```
"Do you deliver to Scotland"  →  shipping,  confidence 0.475   correct, but escalate
"I was charged twice"         →  billing,   confidence 0.95    auto-file
```

---

## The four mistakes — the strongest part of this project

Every one produced a number that looked like a result. **None of them threw an
error.**

### 1. 100% accuracy, scored on the training data

The teaching script trained and tested on the same rows. A model that memorises
scores perfectly and is useless on anything new. The honest figure was 58%.

### 2. A grid search that saw its own test set

Hyperparameter tuning ran over all 228 rows. Cross-validation then reported 83%
for category while the held-out score *fell* to 70% — the signature of a search
shown its own answers. Re-run on the training split alone, the same search chose
different parameters and scored 84.8%.

> **Any decision made by looking at the test set stops the test set being a
> test.**

### 3. A single split reported as a result

That 84.8% was a lucky draw. A *stratified* split of the same data gave 73.9%.
Only cross-validation over every row produced a number worth quoting: 86.9%.

### 4. Misaligned splits — caught by a score below baseline

Wiring in per-target stratified splits, I wrote:

```python
X_train, X_test, y_cat_train, y_cat_test = split_for(df, "category")
_, _, y_pri_train, y_pri_test = split_for(df, "priority")   # different shuffle
```

Stratifying on category produces a different shuffle than stratifying on
priority — so each ticket's text was paired with a different ticket's label.

Nothing errored. Category still scored 93.6%. **Priority came back at 35.9%,
below its own 42.2% majority baseline** — the only reason it was caught.

> **A score below the majority-class baseline is almost always a wiring bug, not
> a bad model.**

The uncomfortable detail: I had described this exact trap in prose, while
explaining why `split_data` splits every column together, and then committed it
in code two steps later.

---

## The code — 120 lines

| File | What it does |
|---|---|
| `src/data_loader.py` | load CSV; `split_for()` — one stratified split per target |
| `src/preprocessor.py` | `clean_text()`, TF-IDF configuration |
| `src/classifier.py` | `build_pipeline()`, train, evaluate, save, load |
| `scripts/train.py` | the offline run: split → fit → score → save |
| `api.py` | FastAPI `/predict`, `/health` |

### The two lines worth pointing at

**The Pipeline, because it makes a whole class of bug impossible:**

```python
Pipeline([("tfidf", TfidfVectorizer(...)), ("clf", LogisticRegression(...))])
```

The vectoriser holds **learned state** — the vocabulary built from training data.
Predicting later must use that same vocabulary; a freshly fitted one would give
word #17 a different meaning and the weights would be garbage. The Pipeline makes
that impossible to get wrong, and `joblib` serialises both halves together.

**Where cleaning happens:**

```python
TfidfVectorizer(preprocessor=clean_text, ...)
```

Inside the vectoriser, not in `load_data`. This way cleaning is part of the saved
pipeline and runs identically on an API request. Doing it in the loader would
clean the training data and leave live input untouched — **train/serve skew**, and
the kind that produces a model that scores well and behaves worse in production.

`clean_text` was dead code for most of this project's life: defined, unit-tested,
and never called by the pipeline. Measuring it showed +0.8% on priority and +1.3%
on sentiment, so it was wired in rather than deleted.

---

## What is still weak, and say it first

**Priority at 57.6%.** Fifteen points over baseline, and neither more data nor
tuning moved it much. Urgency is frequently not in the text — *"my invoice is
wrong"* is urgent or not depending on the amount, which the ticket does not
state. That is a finding, not a bug.

**Sentiment is the wrong tool for the signal.** Category lives in nouns — refund,
parcel, password — which is exactly what TF-IDF represents. Sentiment lives in
tone and negation, which word counts cannot capture. A small pre-trained
transformer would beat this, still locally and still free.

> **Category is TF-IDF because the signal is lexical; sentiment needs a language
> model because the signal is grammatical.** Knowing which problem needs which
> tool is the actual skill.

**Roughly one ticket in twenty is genuinely ambiguous.** *"Can I get a copy of the
receipt sent to a second email"* is billing or account depending on how you read
it. That is a ceiling near 95% that no amount of data removes.

**The dataset is hand-written.** 389 tickets from one person in one afternoon. The
banking77 benchmark exists precisely to test how much that flatters the numbers —
and the answer was "it doesn't", for a reason worth explaining.

---

## Running it

```bash
cd ticket-classifier
source .venv/bin/activate

python -m scripts.train                  # split, fit, score, save to models/
python -m pytest tests/ -q               # 6 tests, ~2s
python -m scripts.benchmark_real_data    # downloads banking77, ~20s
python -m uvicorn api:app --reload       # http://localhost:8000/docs
```

Everything runs offline except the benchmark, which fetches a public CSV.

---

## Four questions to have ready

**"Why not just use GPT for this?"**
> For classification with clean labels it is slower, costs money, is
> non-deterministic and cannot tell you why. This is 2ms, free, exact, and I can
> name the words that drove the decision. I would reach for a language model for
> the sentiment target, because tone is grammatical rather than lexical — and
> that is the one target here that does not work well.

**"Is 86.9% good?"**
> Against a 25.7% baseline, yes. Against banking77's 95%+ on narrow intents, my
> classes are broader and harder. The number I would defend is the ±3.6%, because
> at 228 rows it was ±9.5% and a single split was landing ten points off.

**"How do you know it works?"**
> Stratified 5-fold cross-validation, baselines reported beside every score, and
> a public real-world dataset run through the identical pipeline. I got the
> evaluation wrong four separate ways before those numbers were trustworthy.

**"What would you do next?"**
> Split `technical` into narrower classes — the benchmark says class breadth is
> costing more than sample size now. Then a pre-trained model for sentiment only.
> I would not chase priority; the signal is not in the text.
