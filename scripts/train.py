"""Train the ticket classifier models."""

from rich.console import Console
from rich.table import Table

from src.classifier import (cross_validated_score, evaluate_model,
                            majority_baseline, save_model, train_model)
from src.data_loader import load_data, split_for

console = Console()
df_global = None


def report(target: str, single_split: float, y) -> None:
    """Print both numbers, and say which one to quote.

    The single split figure is what just got computed, so it is printed - but on
    a few hundred rows it is a lottery. The same model and data gave 93.6% on one
    split and 73.9% on another. The cross-validated mean is the honest number and
    the baseline is what makes it mean anything.
    """
    mean, sd = cross_validated_score(df_global["text"], y, target)
    base = majority_baseline(y)

    console.print(f"  this split      {single_split:.1%}   <- one test, do not quote this")
    console.print(f"  [bold]5-fold average  {mean:.1%} +/-{sd:.1%}[/bold]   <- the honest number")
    console.print(f"  always guessing {base:.1%}   <- what beating it has to mean\n")


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

    # Train category classifier
    console.print("[bold]Training category classifier...[/bold]")
    cat_model = train_model(cat_X_train, y_cat_train, "category")
    cat_metrics = evaluate_model(cat_model, cat_X_test, y_cat_test)
    save_model(cat_model, "category_classifier")
    report("category", cat_metrics["accuracy"], df["category"])

    # Train priority classifier
    console.print("[bold]Training priority classifier...[/bold]")
    pri_model = train_model(pri_X_train, y_pri_train, "priority")
    pri_metrics = evaluate_model(pri_model, pri_X_test, y_pri_test)
    save_model(pri_model, "priority_classifier")
    report("priority", pri_metrics["accuracy"], df["priority"])

    # Train sentiment classifier
    console.print("[bold]Training sentiment classifier...[/bold]")
    sent_model = train_model(sen_X_train, y_sent_train, "sentiment")
    sent_metrics = evaluate_model(sent_model, sen_X_test, y_sent_test)
    save_model(sent_model, "sentiment_classifier")
    report("sentiment", sent_metrics["accuracy"], df["sentiment"])

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

    console.print("\n[bold green]✓ Models saved to models/[/bold green]")


if __name__ == "__main__":
    main()
