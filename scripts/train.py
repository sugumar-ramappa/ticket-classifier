"""Train the ticket classifier models."""

from rich.console import Console
from rich.table import Table

from src.classifier import evaluate_model, save_model, train_model
from src.data_loader import load_data, split_for

console = Console()


def main():
    console.print("\n[bold blue]Training Ticket Classifier[/bold blue]\n")

    # Load and split data
    df = load_data()
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
    console.print(f"  Accuracy: {cat_metrics['accuracy']:.2%}\n")

    # Train priority classifier
    console.print("[bold]Training priority classifier...[/bold]")
    pri_model = train_model(pri_X_train, y_pri_train, "priority")
    pri_metrics = evaluate_model(pri_model, pri_X_test, y_pri_test)
    save_model(pri_model, "priority_classifier")
    console.print(f"  Accuracy: {pri_metrics['accuracy']:.2%}\n")

    # Train sentiment classifier
    console.print("[bold]Training sentiment classifier...[/bold]")
    sent_model = train_model(sen_X_train, y_sent_train, "sentiment")
    sent_metrics = evaluate_model(sent_model, sen_X_test, y_sent_test)
    save_model(sent_model, "sentiment_classifier")
    console.print(f"  Accuracy: {sent_metrics['accuracy']:.2%}\n")

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
