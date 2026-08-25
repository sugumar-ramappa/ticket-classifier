# Demo script — 6 minutes

Everything runs locally. No API key, no network, no cost.

Say it in this order. The accuracy is unremarkable; the reasoning is the part
worth showing.

**Register note:** use the real terms — TF-IDF, cross-validation, baseline — and
gloss each one in a short clause the first time. Explaining a term as you use it
reads as fluency. Avoiding it reads as not knowing it.

---

## 0 — Open with why · 20 seconds

> Every other project I've built calls a language model. This one is the
> counter-argument.
>
> It classifies support tickets — category, priority, sentiment — using classical
> ML rather than an LLM. Two milliseconds per prediction, zero cost, and it can
> tell you exactly which words drove the decision.

That's the pitch. You aren't presenting a beginner ML project — you're presenting
a tool-selection argument, which is a more senior thing to be judged on.

---

## 1 — Lead with explainability · 90 seconds

Do this first. It's the capability nobody else in the loop will show.

```bash
python -m scripts.explain "my card was charged twice and the invoice is wrong"
```

```
-> billing   confidence 97%

word      in ticket   account   billing   shipping   technical
invoice        0.42     -0.40    +1.21      -0.47       -0.35
charged        0.43     -0.34    +1.07      -0.34       -0.39
card           0.45     -0.31    +0.94      -0.33       -0.29
twice          0.52     -0.17    +0.61      -0.26       -0.19
wrong          0.42     -0.24    -0.27      +0.44       +0.06
TOTAL                   -1.51    +3.53      -1.07       -0.95
```

**Say:**

> The model holds a weight for every term, per class. Classification is a dot
> product — multiply each term by its weight, sum, take the highest.
>
> Four terms carried this to billing. One pushed the other way: *wrong* leans
> shipping in my data, because it mostly appears in "wrong item delivered". It was
> outweighed.
>
> A language model would classify this correctly too. It couldn't produce this
> table. When this starts misclassifying, the cause is one command away — that's
> how I found "bank" in "bank holidays" was pulling deliveries toward billing.

That last detail matters. It shows the explainability isn't decorative — you used
it to diagnose something real.

---

## 2 — The scores, and what they're measured against · 60 seconds

```bash
python -m scripts.train
```

```
category    87%     baseline 26%
priority    58%     baseline 42%
sentiment   68%     baseline 52%
```

**Say:**

> An accuracy figure means nothing alone. It needs the baseline beside it — what
> you'd score ignoring the input entirely and always predicting the majority class.
>
> My tickets are mostly negative, so for sentiment the baseline is 57%.
>
> My first sentiment classifier scored 59%.

**Pause here.** Let them register that 59% sounds acceptable. Then:

> Two points above baseline. On that test set, one ticket. It had learned
> essentially nothing — and 59% still reads as respectable if nobody mentions the
> 57%.
>
> So I don't quote a score without the baseline next to it.

**They will ask: "what was your first classifier?"**

> Random Forest — the common default, and the wrong shape for text.
>
> TF-IDF gives me roughly 850 features, one per term. A single ticket activates
> four or five. So the matrix is about 99% zeros.
>
> A tree splits on one feature at a time — *does this contain "refund"?* — and for
> almost every ticket that's a no, so almost every split is uninformative. Random
> Forest compounds it by showing each split only a random subset of features.
>
> A linear model weights every term simultaneously, which is what the
> representation is built for. Swapping to logistic regression took sentiment from
> 59% to 67% with nothing else changed.

**Possibly: "so why did you start with the wrong one?"**

> It was the default and I hadn't reasoned about the data's shape yet. I found out
> by benchmarking both properly — five folds, all three targets. Linear won every
> fold.

**"How did you measure it?"**

> Stratified five-fold cross-validation. Five splits, every row tested exactly
> once, stratified so each fold keeps the same class balance.
>
> A single split on 389 rows is a lottery — I've seen the same model and data give
> 93% and 74% depending on which rows landed where.

**"What does the ± mean?"**

> Standard deviation across those folds. Mine is ±3.6. At 228 rows it was ±9.5,
> which made any single figure close to meaningless.

---

## 3 — The evaluation mistakes · 2 minutes · the strongest section

Pick two. They share a shape: **a plausible number, and nothing raised an error.**

### Hyperparameter search that saw the test set

> I ran a grid search over a few hundred configurations, but searched across the
> whole dataset — so the tuning could see the held-out rows. The settings it chose
> were the ones that happened to suit that particular split.
>
> Cross-validation reported 83% while the held-out score *fell* to 70%, which is
> the signature of exactly that. Re-run against the training split only, it chose
> different hyperparameters and the honest score was different again.
>
> **The rule: any decision made by looking at the test set stops it being a test.**

### Misaligned splits, caught by a score below baseline

> I moved to a stratified split per target — you can only stratify on one column,
> and I have three.
>
> Stratifying on category produces a different shuffle than stratifying on
> priority. So I ended up pairing one ticket's text with another ticket's label.
>
> Nothing threw. Category still scored 93%. Priority came back at 36% against a
> 42% baseline — **below chance**, which is the only reason I caught it.
>
> **The rule: a score below baseline is almost never a weak model. It's a wiring
> bug.**

If you're comfortable, add:

> I'd described that exact failure mode out loud a few minutes before I committed
> it.

Worth the cost. It says you audit your own work rather than trusting it.

### The other two, if asked

> I evaluated on the training set once and scored 100% — memorisation, not
> generalisation.
>
> And I quoted 85% from a single favourable split. A stratified split of the same
> data gave 74%.

---

## 4 — I tested my own assumption and was wrong · 60 seconds

```bash
python -m scripts.benchmark_real_data
```

```
MINE   389 hand-written tickets         87%
REAL   banking77, 13,000 real queries   95-99%
```

**Say:**

> I wrote those 389 tickets, so the obvious objection is that they're
> unrealistically clean.
>
> I took banking77 — thirteen thousand genuine customer queries, public — and ran
> the identical pipeline against it, expecting to be caught out.
>
> It scored higher.

**Pause.** Then:

> The difference is class granularity, not realism. Their intents are narrow: one
> label, one kind of question. My "technical" spans crashes, latency, failed
> exports and typos — four unrelated problems sharing a label, with no consistent
> vocabulary to learn.
>
> So my ceiling is the labelling scheme, not the data. Splitting "technical" would
> buy more than another two hundred rows.

The value is the shape of it: **hypothesis, test, refuted, diagnosed.** Worth more
than having been right.

---

## 5 — What doesn't work · 40 seconds · volunteer it

> Two of the three targets are weak, and I know why.

**Priority:**

> 58% against a 42% baseline, and more data didn't shift it. The urgency generally
> isn't in the text — "my invoice is wrong" is high or low depending on the amount,
> which the ticket doesn't state.
>
> That's a finding about the problem, not a bug I failed to fix.

**Sentiment:**

> Category is a lexical signal — refund, parcel, password. Bag-of-words captures
> that directly.
>
> Sentiment isn't. Tone lives in negation, sarcasm and construction — "not happy",
> a polite complaint — and term frequencies can't represent any of it. No amount of
> tuning moved it, which is consistent with that.
>
> For that target I'd drop bag-of-words entirely and use a pre-trained model.

Close with the line the project exists to make:

> **Bag-of-words works when the signal is lexical and fails when it's grammatical.
> Recognising which one you have is the actual decision.**

---

## Which tool for which job — name the model

*"I'd use a language model"* is weak. Naming one is not.

| Target | Tool | Reasoning |
|---|---|---|
| Category | TF-IDF + logistic regression | lexical signal, must be fast and free |
| Priority | none, honestly | the information isn't in the input |
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` | tone is grammatical; pre-trained on it, and returns the same three classes |
| Labelling new data | a frontier model, once | see below |

**On the sentiment choice:**

> It's a half-gigabyte download that then runs locally with no key and no cost —
> the same operating model as everything else here. I wouldn't train it; it's
> already trained on millions of sentences.
>
> I'm not arguing against language models. I'm arguing against paying per call for
> something a dot product already does in two milliseconds.

---

## Where a frontier model genuinely earns its place

Two patterns worth naming — this is what teams actually do.

**Use it to build the dataset, not to serve traffic.**

> The hard part here was having no labelled data, so I wrote 389 tickets by hand.
> In production you'd take five thousand real tickets, pay a frontier model once to
> label them, and train this classifier on the output.
>
> You pay for five thousand calls once, then serve millions of predictions at two
> milliseconds for nothing.
>
> **The LLM builds the training set. The cheap model serves the traffic.**

**Use it only for what the cheap model is unsure about.**

> Every prediction carries a confidence, and I measured how well calibrated it is.
> Above 0.6 it was correct on every held-out ticket.
>
> So: auto-file above 0.6, escalate the rest to a frontier model or a human. On my
> test set that resolved 54% with no errors and spent money only on the remainder.

**The economics, in one line:**

> A frontier model is a general tool with per-call cost. This is a specific tool
> with none. If you have labels and fixed classes, train the cheap one — and spend
> on the expensive one to create the labels, or to handle what the cheap one flags
> as uncertain.

---

## The four questions

**"Why not just use GPT for this?"**

> For classification with fixed labels it's slower, metered, non-deterministic and
> unexplainable. This is two milliseconds, free, identical every run, and auditable
> down to individual term weights.
>
> I would use one for sentiment — and sentiment is the one target here that doesn't
> work, which isn't a coincidence.

**"Is 87% good?"**

> Against a 26% baseline, yes. Against banking77's 95% on narrow intents, my classes
> are broader and harder. But the figure I'd defend is the ±3.6 — at 228 rows it was
> ±9.5, and a single evaluation could be ten points out.

**"How do you know it works?"**

> Stratified five-fold cross-validation, baselines reported alongside every score,
> and a public real-world dataset through the identical pipeline as a control.
>
> I got the evaluation wrong four separate ways before I trusted the numbers.

**"What would you do next?"**

> Split "technical" into narrower classes — the benchmark says granularity now costs
> more than sample size. Then swap sentiment to the pre-trained model. I'd leave
> priority; the signal isn't in the input.

---

## If they ask about your other projects

This pairing is the strongest thing across the portfolio:

> I've built a multi-agent system with a graph engine, MCP tool access and
> adversarial verification. And I've built this, which does its job in 120 lines
> with no model call at all.
>
> What I'd want to be assessed on is knowing which of those a problem needs.

---

## Terms, with the gloss to use

Say the term, then the clause. That's what fluency sounds like.

| Term | Gloss it like this |
|---|---|
| **TF-IDF** | "turns text into term weights — rare terms count for more" |
| **Logistic regression** | "a weight per term per class, summed" |
| **Baseline** | "what you'd score always predicting the majority class" |
| **Cross-validation** | "five splits, every row tested once, then averaged" |
| **Stratified** | "each fold keeps the same class balance as the whole set" |
| **Overfitting** | "memorising the training set instead of generalising" |
| **Calibration** | "whether the confidence score is actually trustworthy" |
