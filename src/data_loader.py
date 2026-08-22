"""Load and split the ticket dataset."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "tickets.csv"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load tickets CSV into a DataFrame."""
    df = pd.read_csv(path)
    df["text"] = df["text"].str.lower().str.strip()
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split into train/test sets."""
    X = df["text"]
    y_category = df["category"]
    y_priority = df["priority"]
    y_sentiment = df["sentiment"]

    X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test, y_sent_train, y_sent_test = train_test_split(
        X, y_category, y_priority, y_sentiment, test_size=test_size, random_state=random_state,
    )
    return X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test, y_sent_train, y_sent_test


def split_for(df: pd.DataFrame, target: str, test_size: float = 0.2,
              random_state: int = 42):
    """Split for ONE target, keeping its class proportions in both halves.

    WHY THIS EXISTS ALONGSIDE split_data
    train_test_split without `stratify` cuts randomly, so nothing guarantees the
    test set reflects the real class balance. With 29 positive sentiment tickets
    out of 228, a random 20% could take 2 or it could take 12 - and the accuracy
    swings accordingly for a reason that has nothing to do with the model.

    You can only stratify on ONE column, and split_data splits three targets at
    once. So the honest fix is a separate split per target: each model is
    independent anyway, and nothing is shared between them but the text.

    Returns X_train, X_test, y_train, y_test.
    """
    return train_test_split(
        df["text"], df[target],
        test_size=test_size,
        random_state=random_state,
        stratify=df[target],
    )
