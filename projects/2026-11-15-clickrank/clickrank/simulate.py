"""Simulated search traffic under a biased production ranker, written as messy raw JSONL events.

Users follow a position-based click model: a result at rank k is examined with probability
(1/k)**eta and, if examined, clicked with a probability that grows with its hidden relevance.
A small share of impressions gets a *swap intervention* (rank 1 swapped with a random rank k),
which is what makes the examination curve identifiable from the logs.

The raw stream is deliberately dirty: retried (duplicate) events, double clicks, clicks that
report the wrong position, orphan clicks, truncated lines, out-of-order delivery and bot users.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

import numpy as np

from .catalog import Catalog
from .features import FEATURES, SearchIndex

SHOWN = 10
START_TS = 1793491200.0  # 2026-11-01T00:00:00Z

# The legacy production ranker: text match plus a heavy popularity boost.
PRODUCTION_WEIGHTS = np.array([1.0 if f == "bm25" else 0.6 if f == "log_popularity" else 0.0
                               for f in FEATURES])


def attractiveness(grade: int) -> float:
    return 0.05 + 0.95 * (2 ** grade - 1) / 7


def true_propensity(k: int, eta: float) -> float:
    return (1.0 / k) ** eta


@dataclass
class NoiseConfig:
    duplicate_rate: float = 0.02
    double_click_rate: float = 0.03
    wrong_position_rate: float = 0.01
    orphan_rate: float = 0.005
    malformed_rate: float = 0.002
    delivery_jitter_s: float = 5.0

    @classmethod
    def clean(cls) -> "NoiseConfig":
        return cls(0, 0, 0, 0, 0, 0)


def _messy_text(rng: random.Random, text: str) -> str:
    words = [w.upper() if rng.random() < 0.1 else w.capitalize() if rng.random() < 0.1 else w
             for w in text.split()]
    sep = "  " if rng.random() < 0.1 else " "
    return (" " if rng.random() < 0.05 else "") + sep.join(words)


def simulate(catalog: Catalog, index: SearchIndex, n_impressions: int, *, swap_rate: float = 0.1,
             eta: float = 1.0, seed: int = 0, n_users: int = 3000, n_bots: int = 3,
             bot_share: float = 0.01, policy_noise: float = 0.15,
             noise: NoiseConfig | None = None) -> tuple[list[str], dict]:
    noise = noise or NoiseConfig()
    rng = random.Random(seed)
    queries = catalog.queries
    weights = [q.weight for q in queries]
    bots = [f"u{n_users + i:05d}" for i in range(n_bots)]
    events: list[dict] = []
    eid = 0

    def next_id() -> str:
        nonlocal eid
        eid += 1
        return f"e{eid:07d}"

    ts = START_TS
    for i in range(n_impressions):
        ts += rng.expovariate(1 / 3.0)
        is_bot = n_bots > 0 and rng.random() < bot_share
        user = rng.choice(bots) if is_bot else f"u{rng.randrange(n_users):05d}"
        q = rng.choices(queries, weights)[0]
        cs = index.candidates(q.text)
        if not cs.product_ids:
            continue
        scores = cs.X @ PRODUCTION_WEIGHTS + np.array([rng.gauss(0, policy_noise) for _ in cs.product_ids])
        order = sorted(range(len(cs.product_ids)), key=lambda j: -scores[j])
        shown = [cs.product_ids[j] for j in order[:SHOWN]]
        intervention = None
        if rng.random() < swap_rate and len(shown) >= 2:
            k = rng.randint(2, len(shown))
            shown[0], shown[k - 1] = shown[k - 1], shown[0]
            intervention = {"kind": "swap", "k": k}
        rid = f"r{i:07d}"
        session = f"{user}-{int((ts - START_TS) // 86400)}"
        events.append({"type": "impression", "event_id": next_id(), "request_id": rid,
                       "session_id": session, "user_id": user, "ts": round(ts, 3),
                       "query": _messy_text(rng, q.text), "results": shown,
                       "intervention": intervention, "policy": "production"})
        grades = catalog.grades[q.text]
        t = ts
        for pos, pid in enumerate(shown, start=1):
            if is_bot:
                clicked = True
                t += 0.2
            else:
                clicked = (rng.random() < true_propensity(pos, eta)
                           and rng.random() < attractiveness(grades[pid]))
                t += rng.uniform(1.5, 6.0)
            if clicked:
                events.append({"type": "click", "event_id": next_id(), "request_id": rid,
                               "session_id": session, "user_id": user, "ts": round(t, 3),
                               "product_id": pid, "position": pos})

    # ---- dirt -------------------------------------------------------------------------
    extra: list[dict] = []
    clicks = [e for e in events if e["type"] == "click"]
    for c in clicks:
        if rng.random() < noise.wrong_position_rate:
            c["position"] += 1                      # client off-by-one bug
        if rng.random() < noise.double_click_rate:
            extra.append({**c, "event_id": next_id(), "ts": round(c["ts"] + 0.4, 3)})
    for _ in range(int(len(clicks) * noise.orphan_rate)):
        extra.append({"type": "click", "event_id": next_id(), "request_id": f"rX{rng.randrange(10**7):07d}",
                      "session_id": "?", "user_id": f"u{rng.randrange(n_users):05d}",
                      "ts": round(START_TS + rng.uniform(0, ts - START_TS), 3),
                      "product_id": rng.choice(catalog.products).id, "position": rng.randint(1, SHOWN)})
    for e in events:
        if rng.random() < noise.duplicate_rate:
            extra.append(dict(e))                   # at-least-once delivery retry
    events.extend(extra)
    keyed = sorted(events, key=lambda e: e["ts"] + rng.uniform(0, noise.delivery_jitter_s))
    lines = []
    for e in keyed:
        line = json.dumps(e, separators=(",", ":"))
        if rng.random() < noise.malformed_rate:
            line = line[: len(line) // 2]
        lines.append(line)

    truth = {"bots": bots, "eta": eta,
             "true_propensity": {k: true_propensity(k, eta) for k in range(1, SHOWN + 1)},
             "production_weights": dict(zip(FEATURES, PRODUCTION_WEIGHTS.tolist()))}
    return lines, truth
