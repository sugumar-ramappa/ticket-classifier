"""What did labelling 389 tickets actually buy us?

The trained classifier scores 86.9% on category. Guessing scores 25.7%. Those
are the only two numbers we had, and the gap between them is not the value of
the labelling effort - because a model that has never seen a single label of
ours is not restricted to guessing.

This script measures the third number: a pre-trained model given nothing but the
four category names in English. Four experiments:

    1  head to head    trained vs zero-shot vs guessing, on identical folds
    2  label wording   the only input you control when there is no training
    3  learning curve  how many labels before training overtakes zero-shot
    4  per class       where each approach wins, and why

    python -m scripts.zero_shot              # all four
    python -m scripts.zero_shot --scheme descriptive
    python -m scripts.zero_shot --skip-curve

First run downloads ~1.6 GB and takes a few minutes. Predictions are cached, so
every run after that is seconds.
"""

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import StratifiedKFold

from src.classifier import build_pipeline, majority_baseline
from src.data_loader import load_data
from src.zero_shot import SCHEMES, ZeroShotClassifier

console = Console()

TARGET = "category"
# Identical to classifier.cross_validated_score, so the trained figures here are
# the same ones the README quotes - and, more importantly, so both approaches
# are scored on exactly the same held-out rows.
FOLDS = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def _progress(done: int, total: int) -> None:
    if done == 1 or done % 25 == 0 or done == total:
        console.print(f"  [dim]classifying {done}/{total}[/dim]", end="\r")


# ---------------------------------------------------------------- experiment 1

def head_to_head(texts, labels, zs_preds) -> dict:
    """Score both approaches on the same five folds, then compare per fold.

    WHY PAIRED, AND NOT TWO SEPARATE AVERAGES
    Some folds are simply harder than others. Comparing two independent averages
    mixes that fold difficulty into the difference between the models. Scoring
    both on identical test rows and subtracting per fold cancels it, leaving the
    part that is actually about the approaches.
    """
    trained_scores, zs_scores = [], []

    for train_idx, test_idx in FOLDS.split(texts, labels):
        model = build_pipeline(TARGET)
        model.fit(texts[train_idx], labels[train_idx])
        trained_scores.append(float((model.predict(texts[test_idx])
                                     == labels[test_idx]).mean()))
        # Zero-shot never trains, so it only ever sees the test rows - which is
        # exactly the point. Same rows, no labels.
        zs_scores.append(float((zs_preds[test_idx] == labels[test_idx]).mean()))

    per_fold = [t - z for t, z in zip(trained_scores, zs_scores)]
    return {
        "trained": (float(np.mean(trained_scores)), float(np.std(trained_scores))),
        "zero_shot": (float(np.mean(zs_scores)), float(np.std(zs_scores))),
        "gap": float(np.mean(per_fold)),
        "wins": sum(d > 0 for d in per_fold),
        "folds": len(per_fold),
    }


# ---------------------------------------------------------------- experiment 2

def wording_experiment(clf, texts, labels) -> dict:
    """Same model, same tickets, four ways of describing the categories."""
    results = {}
    for name, scheme in SCHEMES.items():
        console.print(f"  [dim]scheme: {name}[/dim]", end="\r")
        preds = np.array(clf.predict(list(texts), scheme, progress=_progress))
        results[name] = (float((preds == labels).mean()), scheme, preds)
    console.print(" " * 60, end="\r")
    return results


# ---------------------------------------------------------------- experiment 3

def learning_curve(texts, labels, zero_shot_acc: float) -> list[tuple]:
    """Accuracy of the trained model against how many labels it was given.

    THE QUESTION THIS ANSWERS
    Zero-shot is a flat line - it does not improve with more of our data,
    because it never sees any. The trained model starts far below it and climbs.
    Where the two cross is the number of labels that had to be written before
    training was worth doing at all.

    Every point is averaged over the same five folds, and each training subset is
    drawn stratified so a 20-row sample still contains all four categories.
    """
    sizes = [10, 20, 40, 80, 160, 240, 311]
    rng = np.random.default_rng(42)
    curve = []

    for n in sizes:
        fold_scores = []
        for train_idx, test_idx in FOLDS.split(texts, labels):
            if n > len(train_idx):
                continue
            # Stratified subsample: take a proportional slice of each class
            # rather than n rows at random, which at n=10 could miss a class
            # entirely and make the point meaningless.
            picked = []
            for cls in np.unique(labels):
                cls_idx = train_idx[labels[train_idx] == cls]
                take = max(1, round(n * len(cls_idx) / len(train_idx)))
                picked.extend(rng.choice(cls_idx, size=min(take, len(cls_idx)),
                                         replace=False))
            picked = np.array(picked)

            model = build_pipeline(TARGET)
            model.fit(texts[picked], labels[picked])
            fold_scores.append(float((model.predict(texts[test_idx])
                                      == labels[test_idx]).mean()))
        if fold_scores:
            curve.append((n, float(np.mean(fold_scores)), float(np.std(fold_scores)),
                          float(np.mean(fold_scores)) >= zero_shot_acc))
    return curve


# ---------------------------------------------------------------- experiment 4

def per_class(texts, labels, zs_preds) -> dict:
    """Recall per category for both approaches, on the same folds."""
    trained_preds = np.empty(len(labels), dtype=object)
    for train_idx, test_idx in FOLDS.split(texts, labels):
        model = build_pipeline(TARGET)
        model.fit(texts[train_idx], labels[train_idx])
        trained_preds[test_idx] = model.predict(texts[test_idx])

    out = {}
    for cls in sorted(set(labels)):
        mask = labels == cls
        out[cls] = (int(mask.sum()),
                    float((trained_preds[mask] == cls).mean()),
                    float((zs_preds[mask] == cls).mean()))
    return out


# ---------------------------------------------------------------- report

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Default "best", not a named scheme.
    #
    # This defaulted to "descriptive" on the assumption that spelling the
    # categories out in detail would help most. Measured, it is the WORST of the
    # four at 57.3% against domain_framed's 75.6% - so the head-to-head was
    # comparing the trained model against a deliberately weakened opponent, and
    # every downstream number inherited that.
    #
    # Comparing your approach against a badly configured version of the
    # alternative is the most common way to produce a flattering result, and it
    # is rarely deliberate. Picking the winner by measurement removes the
    # opportunity.
    parser.add_argument("--scheme", default="best",
                        choices=["best", *sorted(SCHEMES)],
                        help="label wording for the head-to-head (default: the "
                             "best-scoring scheme, chosen by measurement)")
    parser.add_argument("--skip-curve", action="store_true")
    args = parser.parse_args()

    console.print("\n[bold blue]Trained vs zero-shot — what did 389 labels buy?[/bold blue]\n")

    df = load_data()
    texts = df["text"].to_numpy()
    labels = df[TARGET].to_numpy()
    baseline = majority_baseline(labels)
    console.print(f"{len(texts)} tickets, {len(set(labels))} categories, "
                  f"guessing the most common scores {baseline:.1%}\n")

    clf = ZeroShotClassifier()
    console.print(f"[dim]model: {clf.model_name} on {clf.device}[/dim]\n")

    # ---- 2 first: it decides which wording the rest of the run uses
    console.print("[bold]1. Label wording — the only input you control[/bold]\n")
    wording = wording_experiment(clf, texts, labels)

    t = Table(box=None)
    t.add_column("scheme")
    t.add_column("hypothesis sent to the model")
    t.add_column("accuracy", justify="right")
    for name, (acc, scheme, _) in sorted(wording.items(), key=lambda kv: kv[1][0]):
        example = scheme.hypothesis.format(next(iter(scheme.candidates)))
        t.add_row(name, f"[dim]{example[:52]}[/dim]", f"{acc:.1%}")
    console.print(t)

    best = max(wording.items(), key=lambda kv: kv[1][0])
    worst = min(wording.items(), key=lambda kv: kv[1][0])
    spread = best[1][0] - worst[1][0]
    console.print(f"\n  Wording alone moves accuracy by [bold]{spread:.1%}[/bold] "
                  f"({worst[0]} {worst[1][0]:.1%} → {best[0]} {best[1][0]:.1%}).")
    console.print("  [dim]Same model, same tickets, zero training. This is the "
                  "zero-shot equivalent of feature engineering.[/dim]")

    console.print(f"\n  [yellow]Note the order.[/yellow] `descriptive` spells each "
                  f"category out in the most detail and comes [bold]last[/bold]. "
                  f"Longer\n  hypotheses give the entailment model more ways to be "
                  f"partially satisfied by\n  the wrong ticket, so precision drops. "
                  f"More explanation is not more signal.")

    chosen = best[0] if args.scheme == "best" else args.scheme
    zs_acc, _, zs_preds = wording[chosen]

    # ---- 1
    console.print(f"\n\n[bold]2. Head to head — identical folds, wording: {chosen}"
                  f"{' (best, chosen by measurement)' if args.scheme == 'best' else ''}[/bold]\n")
    h2h = head_to_head(texts, labels, zs_preds)

    t = Table(box=None)
    t.add_column("approach")
    t.add_column("labels it saw", justify="right")
    t.add_column("accuracy", justify="right")
    t.add_row("[dim]guess the most common[/dim]", "[dim]—[/dim]", f"[dim]{baseline:.1%}[/dim]")
    t.add_row("zero-shot (pre-trained)", "[bold]0[/bold]",
              f"[bold cyan]{h2h['zero_shot'][0]:.1%} ±{h2h['zero_shot'][1]:.3f}[/bold cyan]")
    t.add_row("trained (TF-IDF + logistic)", "311",
              f"[bold green]{h2h['trained'][0]:.1%} ±{h2h['trained'][1]:.3f}[/bold green]")
    console.print(t)

    console.print(f"\n  Trained wins on [bold]{h2h['wins']}/{h2h['folds']}[/bold] folds, "
                  f"by [bold]{h2h['gap']:+.1%}[/bold] on average.")
    console.print(f"  Zero-shot beats guessing by "
                  f"[bold]{h2h['zero_shot'][0] - baseline:+.1%}[/bold] "
                  f"having seen [bold]no labels at all[/bold].")

    # ---- 4
    console.print("\n\n[bold]3. Where each one wins[/bold]\n")
    t = Table(box=None)
    t.add_column("category")
    t.add_column("n", justify="right")
    t.add_column("trained", justify="right")
    t.add_column("zero-shot", justify="right")
    t.add_column("gap", justify="right")
    for cls, (n, tr, zs) in per_class(texts, labels, zs_preds).items():
        d = tr - zs
        colour = "green" if d > 0.05 else ("cyan" if d < -0.05 else "dim")
        t.add_row(cls, str(n), f"{tr:.1%}", f"{zs:.1%}", f"[{colour}]{d:+.1%}[/{colour}]")
    console.print(t)

    # ---- 3
    if not args.skip_curve:
        console.print("\n\n[bold]4. How many labels before training was worth it?[/bold]\n")
        curve = learning_curve(texts, labels, zs_acc)

        t = Table(box=None)
        t.add_column("labels written", justify="right")
        t.add_column("trained accuracy", justify="right")
        t.add_column("vs zero-shot", justify="right")
        for n, acc, sd, beats in curve:
            mark = "[green]ahead[/green]" if beats else "[red]behind[/red]"
            t.add_row(str(n), f"{acc:.1%} ±{sd:.3f}", mark)
        console.print(t)
        console.print(f"\n  [dim]zero-shot holds a flat {zs_acc:.1%} at every "
                      f"row count — it never sees our labels[/dim]")

        crossed = [n for n, _, _, beats in curve if beats]
        if crossed:
            first = crossed[0]
            console.print(f"\n  Training overtakes zero-shot at roughly "
                          f"[bold]{first} labelled tickets[/bold].")
            console.print(f"  The {389 - first} written after that bought the "
                          f"remaining [bold]{h2h['trained'][0] - zs_acc:+.1%}[/bold].")
        else:
            console.print("\n  [yellow]Training never overtakes zero-shot at any "
                          "size tested — the labelling effort did not pay for "
                          "itself on this task.[/yellow]")

    console.print("\n[bold green]✓ Done[/bold green]\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
