"""Run the same pipeline against a real, public dataset.

WHY
Our 389 tickets were hand-written. Synthetic data is usually easier than
reality - one person writing in one afternoon produces cleaner, more separable
text than real customers do - so the honest question is how much our 86.9%
is flattered by that.

THE DATASET
banking77: 13,083 genuine customer service queries to a banking app, labelled
with 77 fine-grained intents. Public, no API key, no auth.
https://github.com/PolyAI-LDN/task-specific-datasets

The text is visibly real:
    "I mistook my pin and now I am locked.  Can you unlock me?"
    "I removed cash from an ATM earlier but it shows up as pending in the app."

WHAT IT SHOWED, WHICH WAS NOT WHAT WAS EXPECTED
The prediction was that real data would score lower and expose the synthetic set
as too easy. It scored HIGHER - 95% to 99% on four-class subsets against our
86.9%, and that holds even when the four intents are deliberately chosen to
overlap.

The reason is class BREADTH, not data provenance. banking77's intents are
narrow: "card_arrival" is one question asked many ways. Our "technical" spans
CORS errors, slow pages, failed exports and spelling mistakes - four unrelated
problems under one label. A broad class has no consistent vocabulary to learn.

So the lesson is the opposite of the one being tested for: how you define the
classes matters more than whether the text is real. Our numbers are not inflated
by synthetic data; they are limited by a harder labelling scheme.
"""

import urllib.request
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.classifier import build_pipeline
from src.data_loader import load_data

warnings.filterwarnings("ignore")

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
CACHE = Path("/tmp")
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def fetch() -> pd.DataFrame:
    frames = []
    for name in ("train", "test"):
        path = CACHE / f"b77_{name}.csv"
        if not path.exists():
            urllib.request.urlretrieve(f"{BASE}/{name}.csv", path)
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def score(X, y, label: str) -> None:
    s = cross_val_score(build_pipeline("category"), X, y, cv=CV, scoring="accuracy")
    base = Counter(y).most_common(1)[0][1] / len(y)
    print(f"  {label:44} {s.mean():>6.1%} +/-{s.std():.1%}   baseline {base:>5.1%}")


def main() -> None:
    real = fetch()
    ours = load_data()

    print(f"\nOURS - hand written, broad categories")
    score(ours["text"], ours["category"], f"{len(ours)} rows, 4 classes")

    # Four intents with clearly different vocabulary.
    easy = ["card_arrival", "card_payment_fee_charged", "pin_blocked",
            "declined_card_payment"]
    # Four that describe the same kind of event, to remove the objection that
    # the easy set was cherry picked.
    fees = ["card_payment_fee_charged", "transfer_fee_charged",
            "cash_withdrawal_charge", "extra_charge_on_statement"]
    fails = ["declined_card_payment", "declined_cash_withdrawal",
             "declined_transfer", "pending_card_payment"]

    print(f"\nREAL - banking77, narrow intents, 100 rows each")
    for names, label in [(easy, "distinct intents"),
                         (fees, "four kinds of fee (confusable)"),
                         (fails, "four kinds of failure (confusable)")]:
        sub = real[real["category"].isin(names)].groupby("category").head(100)
        score(sub["text"], sub["category"], label)

    print(f"\nREAL - banking77, everything")
    score(real["text"], real["category"], f"{len(real)} rows, 77 classes")

    print("""
  Note the error bars. Ours is +/-3.8% on 389 rows; banking77 is +/-0.7% on
  13,083. That gap is sample size, not model quality - and it is why a single
  train/test split on a few hundred rows cannot tell you whether a change
  helped.
""")


if __name__ == "__main__":
    main()
