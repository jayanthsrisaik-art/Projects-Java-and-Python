"""Transaction category classifier and baselines.

Features (all derived from a single raw row, so serving == training):
* char n-gram TF-IDF (2-4, word-boundary aware) on the cleaned merchant name,
* word TF-IDF on the cleaned merchant name,
* one-hot processor, amount sign, log-amount bucket, ``.99``-style price ending.

Model: multinomial logistic regression. Its probabilities feed an abstain threshold
that is picked on out-of-fold, merchant-grouped predictions from the training set.
"""
from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from .normalize import normalize


def row_features(descriptor: str, amount: float) -> dict:
    n = normalize(descriptor)
    cents = round(abs(amount) * 100) % 100
    return {
        "name": n.name,
        "key": n.key,
        "tab": {
            f"proc={n.processor}": 1.0,
            f"sign={'credit' if amount > 0 else 'debit'}": 1.0,
            f"amt_bin={min(int(math.log2(abs(amount) + 1)), 12)}": 1.0,
            f"cents={'99' if cents in (99, 49) else ('00' if cents == 0 else 'other')}": 1.0,
        },
    }


class TxnClassifier:
    """sklearn-style classifier over raw (descriptor, amount) pairs."""

    def __init__(self, C: float = 5.0, seed: int = 0):
        self.C = C
        self.seed = seed
        self.threshold = 0.0  # abstain below this max-probability

    def _fit_vectorizers(self, feats: list[dict]) -> None:
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
        self.word = TfidfVectorizer(analyzer="word", token_pattern=r"[A-Z&]+", lowercase=False, min_df=1)
        self.tab = DictVectorizer()
        names = [f["name"] for f in feats]
        self.char.fit(names)
        self.word.fit(names)
        self.tab.fit([f["tab"] for f in feats])

    def _matrix(self, feats: list[dict]):
        names = [f["name"] for f in feats]
        return hstack([self.char.transform(names), self.word.transform(names),
                       self.tab.transform([f["tab"] for f in feats])]).tocsr()

    def fit(self, descriptors, amounts, labels) -> "TxnClassifier":
        feats = [row_features(d, a) for d, a in zip(descriptors, amounts)]
        self._fit_vectorizers(feats)
        self.clf = LogisticRegression(C=self.C, max_iter=3000, random_state=self.seed)
        self.clf.fit(self._matrix(feats), list(labels))
        self.classes_ = list(self.clf.classes_)
        return self

    def predict_proba(self, descriptors, amounts) -> np.ndarray:
        feats = [row_features(d, a) for d, a in zip(descriptors, amounts)]
        return self.clf.predict_proba(self._matrix(feats))

    def predict(self, descriptors, amounts) -> list[str]:
        p = self.predict_proba(descriptors, amounts)
        return [self.classes_[i] for i in p.argmax(axis=1)]

    def predict_with_confidence(self, descriptors, amounts) -> list[tuple[str, float]]:
        p = self.predict_proba(descriptors, amounts)
        return [(self.classes_[i], float(p[r, i])) for r, i in enumerate(p.argmax(axis=1))]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str | Path) -> "TxnClassifier":
        # pickle executes code on load: only load artifacts you produced yourself.
        with open(path, "rb") as fh:
            return pickle.load(fh)


def pick_threshold(confidences, correct, target_precision: float = 0.95) -> float:
    """Smallest confidence threshold whose accepted set reaches ``target_precision``.

    Scans thresholds from high to low and keeps the lowest one at which precision on
    accepted rows is still >= target, so coverage is maximised subject to the target.
    """
    order = np.argsort(-np.asarray(confidences))
    conf = np.asarray(confidences)[order]
    ok = np.asarray(correct, dtype=float)[order]
    prec = np.cumsum(ok) / np.arange(1, len(ok) + 1)
    best = 1.0
    for i in range(len(conf)):
        # only cut between distinct confidence values
        if i + 1 < len(conf) and conf[i + 1] == conf[i]:
            continue
        if prec[i] >= target_precision:
            best = float(conf[i])
    return best


def fit_with_threshold(descriptors, amounts, labels, groups, target_precision=0.95,
                       folds: int = 4, C: float = 5.0) -> TxnClassifier:
    """Out-of-fold grouped predictions pick the threshold; then refit on all rows."""
    descriptors, amounts, labels = list(descriptors), list(amounts), list(labels)
    conf = np.zeros(len(labels))
    correct = np.zeros(len(labels), dtype=bool)
    for tr, va in GroupKFold(n_splits=folds).split(descriptors, labels, groups):
        m = TxnClassifier(C=C).fit([descriptors[i] for i in tr], [amounts[i] for i in tr],
                                   [labels[i] for i in tr])
        for i, (lab, c) in zip(va, m.predict_with_confidence([descriptors[i] for i in va],
                                                              [amounts[i] for i in va])):
            conf[i], correct[i] = c, lab == labels[i]
    model = TxnClassifier(C=C).fit(descriptors, amounts, labels)
    model.threshold = pick_threshold(conf, correct, target_precision)
    return model


# ---------------------------------------------------------------- baselines
class MajorityBaseline:
    def fit(self, descriptors, amounts, labels):
        self.label = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, descriptors, amounts):
        return [self.label] * len(descriptors)


class MerchantLookupBaseline:
    """Memorise merchant_key -> most common label; unseen merchants get the majority label.

    Shows how much a random (non-grouped) split rewards memorisation.
    """

    def fit(self, descriptors, amounts, labels):
        votes: dict[str, Counter] = defaultdict(Counter)
        for d, lab in zip(descriptors, labels):
            votes[normalize(d).key][lab] += 1
        self.table = {k: c.most_common(1)[0][0] for k, c in votes.items()}
        self.default = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, descriptors, amounts):
        return [self.table.get(normalize(d).key, self.default) for d in descriptors]


KEYWORD_RULES: list[tuple[str, str]] = [
    # ordered: first match wins, so specific phrases precede generic words
    ("GAS & ELECTRIC", "utilities"), ("WATER DISTRICT", "utilities"), ("SERVICE STATION", "fuel"),
    ("CAR RENTAL", "travel"), ("FOOD MART", "groceries"), ("NOODLE HOUSE", "dining"),
    ("BURGER BAR", "dining"),
    ("PAYROLL", "income"), ("DIR DEP", "income"), ("ACH CREDIT", "income"),
    ("MARKET", "groceries"), ("GROCER", "groceries"), ("FOODS", "groceries"), ("SUPERMARKET", "groceries"),
    ("PRODUCE", "groceries"), ("PANTRY", "groceries"),
    ("CAFE", "dining"), ("GRILL", "dining"), ("PIZZERIA", "dining"), ("TAQUERIA", "dining"),
    ("BISTRO", "dining"), ("DINER", "dining"), ("SUSHI", "dining"), ("BAKERY", "dining"),
    ("COFFEE", "dining"), ("KITCHEN", "dining"),
    ("FUEL", "fuel"), ("PETRO", "fuel"), ("GAS", "fuel"),
    ("RIDES", "transport"), ("TRANSIT", "transport"), ("CAB", "transport"), ("TAXI", "transport"),
    ("PARKING", "transport"), ("RAIL", "transport"), ("SCOOTERS", "transport"), ("SHUTTLE", "transport"),
    ("POWER", "utilities"), ("ELECTRIC", "utilities"), ("BROADBAND", "utilities"),
    ("TELECOM", "utilities"), ("WIRELESS", "utilities"), ("ENERGY", "utilities"),
    ("FLIX", "subscriptions"), ("MUSIC", "subscriptions"), ("STREAM", "subscriptions"), ("TV", "subscriptions"),
    ("PHARMACY", "health"), ("DRUGS", "health"), ("DENTAL", "health"), ("CLINIC", "health"),
    ("MEDICAL", "health"), ("OPTOMETRY", "health"), ("HEALTH", "health"),
    ("AIRLINES", "travel"), ("AIR", "travel"), ("HOTEL", "travel"), ("INN", "travel"),
    ("RESORTS", "travel"), ("SUITES", "travel"), ("LODGE", "travel"),
    ("OUTFITTERS", "shopping"), ("DEPOT", "shopping"), ("EMPORIUM", "shopping"), ("GOODS", "shopping"),
    ("STORE", "shopping"), ("MART", "shopping"), ("SUPPLY", "shopping"), ("BOUTIQUE", "shopping"),
]


class KeywordRuleBaseline:
    """Hand-written keyword rules over the cleaned name (word-boundary match).

    No learning; anything without a keyword falls back to the training majority label.
    """

    def fit(self, descriptors, amounts, labels):
        self.default = Counter(labels).most_common(1)[0][0]
        return self

    def predict_one(self, descriptor: str) -> str | None:
        text = " " + descriptor.upper() + " "
        name = " " + normalize(descriptor).name + " "
        for kw, cat in KEYWORD_RULES:
            if kw in ("PAYROLL", "DIR DEP", "ACH CREDIT"):
                if kw in text:
                    return cat
            elif f" {kw} " in name:
                return cat
        return None

    def predict(self, descriptors, amounts):
        return [self.predict_one(d) or self.default for d in descriptors]
