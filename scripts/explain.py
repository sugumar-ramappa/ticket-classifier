"""Show what the model learned, and why it decided a particular ticket.

    python -m scripts.explain                      # strongest words per class
    python -m scripts.explain "my card was charged twice"   # explain one ticket

WHY THIS IS POSSIBLE AT ALL
A linear model is a table of numbers: one weight per word, per class. Nothing is
hidden, so "why did it say billing?" has a literal arithmetic answer - these
words, these weights, this sum.

That is the property being traded away when a language model is used instead. An
LLM would likely classify these tickets well and could not show its working; here
the working IS the model.
"""

import sys

import numpy as np
from rich.console import Console
from rich.table import Table

from src.classifier import load_model

console = Console()


def strongest_words(pipeline, per_class: int = 12) -> None:
    """The words carrying most weight for each class."""
    vec, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    names = vec.get_feature_names_out()

    table = Table(title=f"Strongest signals per class  ({len(names)} words learned)")
    table.add_column("class")
    table.add_column("words, with their weights")

    for i, label in enumerate(clf.classes_):
        top = np.argsort(-clf.coef_[i])[:per_class]
        table.add_row(label, "  ".join(f"{names[j]}[{clf.coef_[i][j]:.1f}]" for j in top))
    console.print(table)

    console.print(
        "\n  A weight is how hard that word pushes towards that class. Every word "
        "\n  also has a NEGATIVE weight for the classes it argues against.\n")


def explain(pipeline, text: str) -> None:
    """Break one prediction into the words that produced it."""
    vec, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    names = vec.get_feature_names_out()

    row = vec.transform([text]).toarray()[0]
    present = np.nonzero(row)[0]

    if present.size == 0:
        # Every word was dropped as a stopword or never seen in training, so the
        # model is deciding on nothing but its intercept. Worth saying out loud:
        # the prediction is a prior, not a judgement.
        console.print("[yellow]No known words in this ticket - the model has "
                      "nothing to go on.[/yellow]")
        return

    scores = clf.coef_ @ row + clf.intercept_
    winner = clf.classes_[int(np.argmax(scores))]
    proba = pipeline.predict_proba([text])[0].max()

    console.print(f'\n[bold]"{text}"[/bold]')
    console.print(f"  -> [bold green]{winner}[/bold green]  confidence {proba:.0%}\n")

    table = Table(title="Contribution of each recognised word")
    table.add_column("word")
    table.add_column("in ticket", justify="right")
    for label in clf.classes_:
        table.add_column(label, justify="right")

    for j in sorted(present, key=lambda k: -row[k]):
        table.add_row(names[j], f"{row[j]:.2f}",
                      *[f"{clf.coef_[i][j] * row[j]:+.2f}" for i in range(len(clf.classes_))])

    table.add_row("[bold]TOTAL[/bold]", "",
                  *[f"[bold]{s:+.2f}[/bold]" for s in scores])
    console.print(table)

    ignored = [w for w in text.lower().split()
               if w not in names and w.isalpha()]
    if ignored:
        console.print(f"\n  ignored (stopword or never seen in training): "
                      f"{', '.join(ignored[:12])}")


def main() -> None:
    pipeline = load_model("category_classifier")

    if len(sys.argv) > 1:
        explain(pipeline, " ".join(sys.argv[1:]))
    else:
        strongest_words(pipeline)


if __name__ == "__main__":
    main()
