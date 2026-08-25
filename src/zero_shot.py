"""Classify tickets with a pre-trained model that has never seen our labels.

WHAT ZERO-SHOT MEANS HERE
The trained classifier in classifier.py learned from 389 hand-labelled tickets.
This one has learned nothing from them. It takes a pre-trained model and the
category names as plain English, and asks, for each name, "does this ticket
entail this description?"

    ticket:     "my payment was charged twice please refund"
    hypothesis: "This support ticket is about billing."   -> 0.54
                "This support ticket is about shipping."  -> 0.11

The model was trained on natural language inference - deciding whether one
sentence follows from another - and classification is that task in disguise.
Nobody labelled a support ticket to make this work.

WHY IT IS WORTH MEASURING
It answers a question the trained model cannot: what did labelling 389 tickets
actually buy us? Without a zero-shot number, "86.9%" has a baseline of 25.7%
(guessing) and nothing in between. The honest comparison is against the best
you could do having labelled nothing at all.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "zero_shot_cache"

# Roughly 1.6 GB, downloaded once and then cached under ~/.cache/huggingface.
# bart-large-mnli is the reference model for this task - large enough to be
# worth comparing against, small enough to run on a laptop.
DEFAULT_MODEL = "facebook/bart-large-mnli"


def _ensure_ca_bundle() -> None:
    """Point TLS verification at a bundle including any locally trusted roots.

    WHY THIS IS NEEDED
    Networks that perform TLS inspection re-sign HTTPS traffic with their own
    root certificate. Browsers trust it because it is installed in the system
    keychain; Python does not, because it verifies against certifi's bundle,
    which by design contains only public certificate authorities. The download
    then fails with

        CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate chain

    which reads as a network fault and is not one - it is a trust-store gap.

    Build the bundle once with:

        security find-certificate -a -p \\
            /System/Library/Keychains/SystemRootCertificates.keychain \\
            > ~/.certs/system-roots.pem
        security find-certificate -a -p /Library/Keychains/System.keychain \\
            >> ~/.certs/system-roots.pem
        python -c "import certifi,shutil;shutil.copy(certifi.where(),'/tmp/c.pem')"
        cat /tmp/c.pem ~/.certs/system-roots.pem > ~/.certs/combined-ca.pem

    setdefault, never assignment: an explicitly configured bundle is already
    correct and must not be overridden.
    """
    bundle = Path.home() / ".certs" / "combined-ca.pem"
    if bundle.exists():
        for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
            os.environ.setdefault(var, str(bundle))


def pick_device() -> str:
    """Best available backend. Apple silicon gains roughly 8x over CPU here."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass(frozen=True)
class LabelScheme:
    """One way of describing the categories to the model.

    WHY THIS IS A TYPE AND NOT A LIST OF STRINGS
    A zero-shot model reads the label names as English, so the wording is not
    cosmetic - it is the only input you control. "billing" and "a question about
    billing, payment or a refund" are different hypotheses and score
    differently on identical tickets.

    That makes label wording the zero-shot equivalent of feature engineering,
    and it deserves to be a named, comparable object rather than a literal
    buried in a call.
    """

    name: str
    phrasings: dict[str, str]        # our label -> the words shown to the model
    hypothesis: str = "This example is {}."

    @property
    def candidates(self) -> list[str]:
        return list(self.phrasings.values())

    def to_label(self, candidate: str) -> str:
        """Map the model's chosen wording back to our label."""
        for label, phrasing in self.phrasings.items():
            if phrasing == candidate:
                return label
        raise KeyError(f"no label for candidate {candidate!r}")

    def fingerprint(self) -> str:
        payload = json.dumps([self.phrasings, self.hypothesis], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


# The four schemes compared in scripts/zero_shot.py, from barest to most
# explicit. Ordered deliberately: the experiment is whether saying more helps.
SCHEMES = {
    "bare": LabelScheme(
        name="bare",
        phrasings={"billing": "billing", "account": "account",
                   "technical": "technical", "shipping": "shipping"},
    ),
    "noun_phrase": LabelScheme(
        name="noun_phrase",
        phrasings={"billing": "billing and payments",
                   "account": "account access and login",
                   "technical": "a technical fault or error",
                   "shipping": "shipping and delivery"},
    ),
    "domain_framed": LabelScheme(
        name="domain_framed",
        phrasings={"billing": "billing and payments",
                   "account": "account access and login",
                   "technical": "a technical fault or error",
                   "shipping": "shipping and delivery"},
        hypothesis="This customer support ticket is about {}.",
    ),
    "descriptive": LabelScheme(
        name="descriptive",
        phrasings={
            "billing": "a charge, refund, invoice or payment problem",
            "account": "logging in, passwords, or account settings",
            "technical": "a bug, error message, or something not working",
            "shipping": "delivery, tracking, or a package that has not arrived",
        },
        hypothesis="This customer support ticket is about {}.",
    ),
}


class ZeroShotClassifier:
    """A pre-trained entailment model used as a classifier.

    Predictions are cached on disk, keyed by model, label scheme and the exact
    text. Inference is ~0.14s per ticket on Apple silicon - about a minute for
    the whole dataset - which is cheap enough to run but not cheap enough to
    repeat four times per experiment while iterating on the report.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = device or pick_device()
        self._pipe = None

    def _pipeline(self):
        # Loaded lazily so that constructing the object, reading the cache, and
        # running the tests do not pull 1.6 GB into memory.
        if self._pipe is None:
            _ensure_ca_bundle()
            from transformers import pipeline

            self._pipe = pipeline("zero-shot-classification",
                                  model=self.model_name, device=self.device)
        return self._pipe

    def _cache_path(self, scheme: LabelScheme) -> Path:
        model_slug = self.model_name.replace("/", "__")
        return CACHE_DIR / f"{model_slug}.{scheme.name}.{scheme.fingerprint()}.json"

    def predict(self, texts: list[str], scheme: LabelScheme,
                progress=None) -> list[str]:
        """Predicted label per text, in order. Cached across runs."""
        path = self._cache_path(scheme)
        cache: dict[str, str] = {}
        if path.exists():
            cache = json.loads(path.read_text())

        missing = [t for t in texts if t not in cache]
        if missing:
            pipe = self._pipeline()
            for i, text in enumerate(missing):
                result = pipe(text, scheme.candidates,
                              hypothesis_template=scheme.hypothesis)
                cache[text] = scheme.to_label(result["labels"][0])
                if progress:
                    progress(i + 1, len(missing))
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache, indent=1))

        return [cache[t] for t in texts]
