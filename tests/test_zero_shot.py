"""Tests for the zero-shot classifier.

None of these load the model. bart-large-mnli is 1.6 GB and inference needs a
GPU backend to be quick - a test suite that requires either is a test suite
nobody runs. What is testable without it is the part that actually broke things:
label schemes, the mapping back from the model's wording to our labels, and the
cache key.
"""

import json

import pytest

from src.zero_shot import SCHEMES, LabelScheme, ZeroShotClassifier


def test_every_scheme_covers_the_same_four_categories():
    """A scheme that omits a category silently makes it unpredictable."""
    expected = {"billing", "account", "technical", "shipping"}
    for name, scheme in SCHEMES.items():
        assert set(scheme.phrasings) == expected, f"{name} has the wrong labels"


def test_phrasings_are_distinct_within_a_scheme():
    """Two categories sharing a phrasing makes to_label() ambiguous.

    It would not raise - it would return whichever matched first, and quietly
    misattribute every prediction for the other category.
    """
    for name, scheme in SCHEMES.items():
        phrasings = list(scheme.phrasings.values())
        assert len(phrasings) == len(set(phrasings)), f"{name} repeats a phrasing"


def test_hypothesis_template_has_a_slot():
    """Without the placeholder the same sentence is scored for every label."""
    for name, scheme in SCHEMES.items():
        assert "{}" in scheme.hypothesis, f"{name} has no substitution slot"
        rendered = scheme.hypothesis.format(scheme.candidates[0])
        assert "{}" not in rendered


def test_to_label_round_trips():
    scheme = SCHEMES["descriptive"]
    for label, phrasing in scheme.phrasings.items():
        assert scheme.to_label(phrasing) == label


def test_to_label_rejects_an_unknown_candidate():
    """Better to fail loudly than to attribute a prediction to the wrong class."""
    with pytest.raises(KeyError):
        SCHEMES["bare"].to_label("something the scheme never offered")


def test_fingerprint_changes_with_wording():
    """The cache key must include the wording, or schemes collide.

    WHY THIS MATTERS
    Predictions are cached per scheme. If two schemes hashed the same, the
    second would be served the first's answers - and the wording experiment
    would report that wording makes no difference, which is the opposite of
    what it measures.
    """
    base = SCHEMES["bare"]
    reworded = LabelScheme(name="bare", phrasings={**base.phrasings,
                                                   "billing": "invoices"})
    assert base.fingerprint() != reworded.fingerprint()


def test_fingerprint_changes_with_the_hypothesis_template():
    base = SCHEMES["noun_phrase"]
    retemplated = LabelScheme(name=base.name, phrasings=base.phrasings,
                              hypothesis="A support ticket concerning {}.")
    assert base.fingerprint() != retemplated.fingerprint()


def test_fingerprint_is_stable_across_calls():
    scheme = SCHEMES["domain_framed"]
    assert scheme.fingerprint() == scheme.fingerprint()


def test_fingerprint_ignores_key_order():
    """Same wording declared in a different order is the same experiment.

    Guards the bug this project's sibling hit in Java: a cache key built by
    iterating a map, where iteration order varied between runs, so identical
    content produced different keys and the cache never hit once.
    """
    base = SCHEMES["bare"]
    reordered = LabelScheme(
        name=base.name,
        phrasings={k: base.phrasings[k] for k in reversed(list(base.phrasings))},
        hypothesis=base.hypothesis,
    )
    assert base.fingerprint() == reordered.fingerprint()


def test_cache_path_separates_models_and_schemes(tmp_path, monkeypatch):
    monkeypatch.setattr("src.zero_shot.CACHE_DIR", tmp_path)
    clf = ZeroShotClassifier(device="cpu")
    a = clf._cache_path(SCHEMES["bare"])
    b = clf._cache_path(SCHEMES["descriptive"])
    assert a != b
    assert "/" not in a.name, "model id must be slugified or it creates directories"


def test_predict_serves_from_cache_without_loading_the_model(tmp_path, monkeypatch):
    """The whole point of the cache: a repeat run must not touch the GPU.

    _pipeline() raises here, so if predict() tries to load the model this fails.
    """
    monkeypatch.setattr("src.zero_shot.CACHE_DIR", tmp_path)
    scheme = SCHEMES["bare"]
    clf = ZeroShotClassifier(device="cpu")

    texts = ["my card was charged twice", "where is my parcel"]
    clf._cache_path(scheme).write_text(json.dumps(
        {texts[0]: "billing", texts[1]: "shipping"}))

    def explode():
        raise AssertionError("model was loaded despite a complete cache")

    monkeypatch.setattr(clf, "_pipeline", explode)
    assert clf.predict(texts, scheme) == ["billing", "shipping"]


def test_predict_returns_results_in_the_order_asked_for(tmp_path, monkeypatch):
    """Cache is a dict; the caller gets a list aligned to their input."""
    monkeypatch.setattr("src.zero_shot.CACHE_DIR", tmp_path)
    scheme = SCHEMES["bare"]
    clf = ZeroShotClassifier(device="cpu")
    clf._cache_path(scheme).write_text(json.dumps(
        {"a": "billing", "b": "shipping", "c": "account"}))

    monkeypatch.setattr(clf, "_pipeline", lambda: (_ for _ in ()).throw(
        AssertionError("should not load")))
    assert clf.predict(["c", "a", "b"], scheme) == ["account", "billing", "shipping"]
