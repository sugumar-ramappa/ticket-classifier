"""Train the ticket classifier models."""

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import sklearn
from rich.console import Console
from rich.table import Table

from src.classifier import (cross_validated_score, evaluate_model,
                            majority_baseline, save_model, train_model)
from src.data_loader import load_data, split_for

console = Console()
df_global = None


def report(target: str, single_split: float, y) -> dict:
    """Print both numbers, say which one to quote, and return them to be recorded.

    The single split figure is what just got computed, so it is printed - but on
    a few hundred rows it is a lottery. The same model and data gave 93.6% on one
    split and 73.9% on another. The cross-validated mean is the honest number and
    the baseline is what makes it mean anything.

    Returning them rather than only printing is what lets `main` write
    `results.json`. A number that exists only in terminal scrollback cannot be
    checked by anyone later, including you - which is the failure this project's
    sibling hit when its latency figures survived in a README and nowhere else.
    """
    mean, sd = cross_validated_score(df_global["text"], y, target)
    base = majority_baseline(y)

    console.print(f"  this split      {single_split:.1%}   <- one test, do not quote this")
    console.print(f"  [bold]5-fold average  {mean:.1%} +/-{sd:.1%}[/bold]   <- the honest number")
    console.print(f"  always guessing {base:.1%}   <- what beating it has to mean\n")

    return {
        "target": target,
        "single_split_accuracy": round(single_split, 4),
        "cross_validated_mean": round(mean, 4),
        "cross_validated_sd": round(sd, 4),
        "majority_baseline": round(base, 4),
        "quote": "cross_validated_mean",
    }


def main():
    console.print("\n[bold blue]Training Ticket Classifier[/bold blue]\n")

    # Load and split data
    global df_global
    df = load_data()
    df_global = df
    console.print(f"Loaded {len(df)} tickets")
    console.print(f"Categories: {df['category'].value_counts().to_dict()}")
    console.print(f"Priorities: {df['priority'].value_counts().to_dict()}")
    console.print(f"Sentiments: {df['sentiment'].value_counts().to_dict()}\n")

    # One stratified split per target, and each target keeps ITS OWN X.
    #
    # You can only stratify on one column, so each target needs its own cut -
    # and stratifying on category produces a different shuffle from stratifying
    # on priority. Reusing one X_train across all three would pair each ticket's
    # text with another ticket's label. Nothing errors; the scores simply become
    # meaningless, and priority landing BELOW its own majority-class baseline is
    # what gave it away.
    cat_X_train, cat_X_test, y_cat_train, y_cat_test = split_for(df, "category")
    pri_X_train, pri_X_test, y_pri_train, y_pri_test = split_for(df, "priority")
    sen_X_train, sen_X_test, y_sent_train, y_sent_test = split_for(df, "sentiment")
    console.print(f"Train: {len(cat_X_train)} | Test: {len(cat_X_test)}\n")

    results: list[dict] = []

    # Train category classifier
    console.print("[bold]Training category classifier...[/bold]")
    cat_model = train_model(cat_X_train, y_cat_train, "category")
    cat_metrics = evaluate_model(cat_model, cat_X_test, y_cat_test)
    save_model(cat_model, "category_classifier")
    results.append(report("category", cat_metrics["accuracy"], df["category"]))

    # Train priority classifier
    console.print("[bold]Training priority classifier...[/bold]")
    pri_model = train_model(pri_X_train, y_pri_train, "priority")
    pri_metrics = evaluate_model(pri_model, pri_X_test, y_pri_test)
    save_model(pri_model, "priority_classifier")
    results.append(report("priority", pri_metrics["accuracy"], df["priority"]))

    # Train sentiment classifier
    console.print("[bold]Training sentiment classifier...[/bold]")
    sent_model = train_model(sen_X_train, y_sent_train, "sentiment")
    sent_metrics = evaluate_model(sent_model, sen_X_test, y_sent_test)
    save_model(sent_model, "sentiment_classifier")
    results.append(report("sentiment", sent_metrics["accuracy"], df["sentiment"]))

    # Show results
    table = Table(title="Classification Report — Category")
    table.add_column("Class")
    table.add_column("Precision")
    table.add_column("Recall")
    table.add_column("F1")
    for label, metrics in cat_metrics["report"].items():
        if isinstance(metrics, dict) and "precision" in metrics:
            table.add_row(
                label,
                f"{metrics['precision']:.2f}",
                f"{metrics['recall']:.2f}",
                f"{metrics['f1-score']:.2f}",
            )
    console.print(table)

    console.print(f"\n[bold]Confusion Matrix (Category):[/bold] {cat_metrics['confusion_matrix']}")

    # Test predictions
    console.print("\n[bold]Sample Predictions:[/bold]")
    samples = [
        "My payment was charged twice",
        "I cannot login to my account",
        "App crashes on startup",
        "Where is my package",
    ]
    for text in samples:
        cat = cat_model.predict([text])[0]
        pri = pri_model.predict([text])[0]
        sent = sent_model.predict([text])[0]
        console.print(f"  '{text}' → category={cat}, priority={pri}, sentiment={sent}")

    write_results(results, df, len(cat_X_train), len(cat_X_test))

    console.print("\n[bold green]✓ Models saved to models/[/bold green]")


def write_results(results: list[dict], df, n_train: int, n_test: int) -> None:
    """Record the run so its numbers can be checked without re-reading a terminal.

    Everything here is seeded - `random_state=42` on the classifier, on the
    stratified folds and on the splits - so re-running reproduces this file
    exactly. That is what makes it a receipt rather than a snapshot: a reader who
    doubts a figure can run the script and diff.

    The library version is recorded for the case where it stops reproducing.
    scikit-learn has changed default solvers between minor versions, and a number
    that moved for that reason looks identical to one that moved because the data
    changed.
    """
    out = Path(__file__).resolve().parents[1] / "results.json"
    payload = {
        "recordedAt": datetime.now(timezone.utc).isoformat(),
        "tickets": len(df),
        "train": n_train,
        "test": n_test,
        "seed": 42,
        "folds": 5,
        "deterministic": True,
        "scikit_learn": sklearn.__version__,
        "python": platform.python_version(),
        "note": (
            "Quote cross_validated_mean, never single_split_accuracy. On a few "
            "hundred rows one split is a lottery - the same model and data gave "
            "93.6% and 73.9% on two different splits. majority_baseline is what "
            "makes the mean mean anything."
        ),
        "targets": results,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    console.print(f"[bold]Results written to {out.name}[/bold]")


if __name__ == "__main__":
    main()
