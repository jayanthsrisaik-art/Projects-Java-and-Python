"""End-to-end measured evaluation.

1. Normalisation quality: how many merchant keys per true merchant (fragmentation) and
   true merchants per key (collisions), vs grouping on the raw descriptor.
2. Categorisation: merchant-grouped K-fold CV (no merchant appears on both sides of a fold),
   model vs majority / keyword-rule / merchant-lookup baselines. The same models are also
   scored on a shuffled row-level K-fold to show how much merchant leakage inflates the numbers.
3. Selective prediction: coverage and accuracy of auto-accepted rows at the abstain
   threshold, which is chosen on out-of-fold training predictions only.
4. Recurring detection: precision/recall vs the generator's ground truth, with normalised
   keys vs raw descriptors.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .model import (KeywordRuleBaseline, MajorityBaseline, MerchantLookupBaseline,
                    TxnClassifier, fit_with_threshold)
from .normalize import normalize
from .recurring import detect
from .synth import CATEGORIES


def dedupe(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        if r["txn_id"] not in seen:
            seen.add(r["txn_id"])
            out.append(r)
    return out


def normalization_report(rows: list[dict]) -> dict:
    def stats(key_fn):
        keys_per_m, m_per_key = defaultdict(set), defaultdict(set)
        for r in rows:
            k = key_fn(r)
            keys_per_m[r["merchant_id"]].add(k)
            m_per_key[k].add(r["merchant_id"])
        return {
            "merchants": len(keys_per_m),
            "keys": len(m_per_key),
            "mean_keys_per_merchant": round(float(np.mean([len(v) for v in keys_per_m.values()])), 3),
            "merchants_with_one_key": sum(len(v) == 1 for v in keys_per_m.values()),
            "keys_with_collisions": sum(len(v) > 1 for v in m_per_key.values()),
        }
    return {"normalized_key": stats(lambda r: normalize(r["descriptor"]).key),
            "raw_descriptor": stats(lambda r: r["descriptor"])}


def _scores(y_true, y_pred) -> dict:
    return {"accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro", labels=CATEGORIES, zero_division=0), 4)}


def _run_models(rows, tr, te, groups, seed, with_threshold: bool):
    D = [r["descriptor"] for r in rows]
    A = [r["amount"] for r in rows]
    Y = [r["category"] for r in rows]           # noisy user labels: what you'd train on
    T = [r["true_category"] for r in rows]      # clean labels: report both
    pick = lambda xs, idx: [xs[i] for i in idx]
    out, model = {}, None
    for name, est in [("majority", MajorityBaseline()), ("keyword_rules", KeywordRuleBaseline()),
                      ("merchant_lookup", MerchantLookupBaseline())]:
        est.fit(pick(D, tr), pick(A, tr), pick(Y, tr))
        pred = est.predict(pick(D, te), pick(A, te))
        out[name] = {"vs_user_labels": _scores(pick(Y, te), pred), "vs_true_labels": _scores(pick(T, te), pred)}
    if with_threshold:
        model = fit_with_threshold(pick(D, tr), pick(A, tr), pick(Y, tr), pick(groups, tr))
    else:
        model = TxnClassifier().fit(pick(D, tr), pick(A, tr), pick(Y, tr))
    pc = model.predict_with_confidence(pick(D, te), pick(A, te))
    pred = [p for p, _ in pc]
    out["logreg"] = {"vs_user_labels": _scores(pick(Y, te), pred), "vs_true_labels": _scores(pick(T, te), pred)}
    return out, model, pc


def _mean_std(values: list[float]) -> dict:
    return {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4)}


def _aggregate(fold_results: list[dict]) -> dict:
    out = {}
    for model in fold_results[0]:
        out[model] = {lab: {m: _mean_std([f[model][lab][m] for f in fold_results])
                            for m in ("accuracy", "macro_f1")}
                      for lab in ("vs_user_labels", "vs_true_labels")}
    return out


def evaluate(rows: list[dict], seed: int = 0, folds: int = 5) -> dict:
    """Merchant-grouped K-fold (every merchant is tested exactly once, never seen in training)
    plus a shuffled row-level K-fold that leaks merchants, for comparison."""
    rows = dedupe(rows)
    groups = [normalize(r["descriptor"]).key for r in rows]
    idx = np.arange(len(rows))
    report: dict = {"rows": len(rows), "merchants": len(set(groups)), "seed": seed, "folds": folds,
                    "normalization": normalization_report(rows)}

    # shuffle group->fold assignment reproducibly (GroupKFold itself is deterministic by size)
    rng = np.random.default_rng(seed)
    uniq = sorted(set(groups))
    fold_of = {g: i % folds for i, g in enumerate(rng.permutation(uniq))}
    grouped, sel, all_true, all_pred = [], [], [], []
    for k in range(folds):
        te = [i for i in idx if fold_of[groups[i]] == k]
        tr = [i for i in idx if fold_of[groups[i]] != k]
        assert not ({groups[i] for i in tr} & {groups[i] for i in te}), "merchant leakage"
        res, model, pc = _run_models(rows, tr, te, groups, seed, with_threshold=True)
        grouped.append(res)
        y_true = [rows[i]["true_category"] for i in te]
        y_user = [rows[i]["category"] for i in te]
        acc = [c >= model.threshold for _, c in pc]
        n_acc = sum(acc)
        sel.append({
            "threshold": model.threshold,
            "coverage": n_acc / len(te),
            "accepted_acc_user": float(np.mean([pc[i][0] == y_user[i] for i in range(len(te)) if acc[i]])) if n_acc else float("nan"),
            "accepted_acc_true": float(np.mean([pc[i][0] == y_true[i] for i in range(len(te)) if acc[i]])) if n_acc else float("nan"),
            "rejected_acc_true": float(np.mean([pc[i][0] == y_true[i] for i in range(len(te)) if not acc[i]])) if n_acc < len(te) else float("nan"),
        })
        all_true += y_true
        all_pred += [p for p, _ in pc]
    per_class = f1_score(all_true, all_pred, average=None, labels=CATEGORIES, zero_division=0)

    rs = np.random.default_rng(seed).permutation(len(rows))
    leaky = []
    for k in range(folds):
        te = sorted(rs[k::folds].tolist())
        tes = set(te)
        tr = [i for i in idx if i not in tes]
        res, _, _ = _run_models(rows, tr, te, groups, seed, with_threshold=False)
        leaky.append(res)

    report["categorization"] = {
        "grouped_kfold": {
            "models": _aggregate(grouped),
            "logreg_per_class_f1_vs_true_pooled": {c: round(float(f), 4) for c, f in zip(CATEGORIES, per_class)},
            "selective": {"target_precision_on_oof_train": 0.95,
                          **{k: _mean_std([s[k] for s in sel]) for k in sel[0]}},
        },
        "random_kfold_LEAKY": {"models": _aggregate(leaky)},
    }
    report["recurring"] = recurring_report(rows)
    return report


def recurring_report(rows: list[dict]) -> dict:
    truth_by_series = {}
    for r in rows:
        truth_by_series[(r["account_id"], r["merchant_id"])] = r["recurring"]
    positives = {s for s, v in truth_by_series.items() if v}

    def score(key_fn, return_pred: bool = False):
        key_to_mid = {}
        for r in rows:
            key_to_mid.setdefault((r["account_id"], key_fn(r)), r["merchant_id"])
        found = detect(rows, key_fn=key_fn)
        pred = {(f.account_id, key_to_mid[(f.account_id, f.merchant_key)]) for f in found}
        if return_pred:
            return pred
        tp = len(pred & positives)
        return {"detected": len(pred), "true_positives": tp,
                "precision": round(tp / len(pred), 4) if pred else 0.0,
                "recall": round(tp / len(positives), 4) if positives else 0.0}

    # false-positive trap: same store on the same weekday every week, variable basket totals
    dates_by_series = defaultdict(list)
    for r in rows:
        dates_by_series[(r["account_id"], r["merchant_id"])].append(r["posted_date"])
    from datetime import date as _d
    traps = set()
    for s, ds in dates_by_series.items():
        ds = sorted(_d.fromisoformat(x) for x in set(ds))
        if not truth_by_series[s] and len(ds) >= 8 and all((b - a).days == 7 for a, b in zip(ds, ds[1:])):
            traps.add(s)
    norm_found = score(lambda r: normalize(r["descriptor"]).key, return_pred=True)
    return {"true_recurring_series": len(positives),
            "weekly_fixed_day_variable_amount_series": len(traps),
            "of_which_flagged_recurring": len(traps & norm_found),
            "normalized_key": score(lambda r: normalize(r["descriptor"]).key),
            "raw_descriptor_baseline": score(lambda r: r["descriptor"])}
