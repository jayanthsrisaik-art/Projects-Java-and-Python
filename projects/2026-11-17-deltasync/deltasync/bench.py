"""Measured transfer-cost comparison on synthetic edit workloads.

Every method is a complete protocol: the receiver holds ``old``, the sender holds
``new``, and the receiver must end with exactly ``new``. Bytes are counted in both
directions and every reconstruction is checked byte-for-byte.

Methods
    whole-file      send ``new`` verbatim
    aligned-blocks  receiver sends a strong hash per block; sender resends blocks whose
                    hash differs *at the same offset* (no search for moved data)
    fixed-dedupe    receiver sends hashes of its fixed 8 KiB chunks; sender sends the
                    recipe (chunk hash list) plus bodies of chunks the receiver lacks
    cdc-dedupe      same protocol with FastCDC chunks (min 2 KiB / avg 8 KiB / max 64 KiB)
    rsync-delta     this project: signature up, delta down (rolling search, any offset)
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from hashlib import blake2b

from .cdc import chunk_hashes, fastcdc_chunks, fixed_chunks
from .delta import Delta, generate_delta
from .patch import apply_delta
from .signature import Signature, make_signature

CHUNK_HASH = 16
BLOCK_HASH = 8


# ---------------------------------------------------------------- synthetic data

_WORDS = ("order customer invoice shipped pending refund warehouse sku qty price total "
          "status region north south east west created updated id user session token "
          "error warn info debug latency_ms bytes host api v1 v2 GET POST 200 404 500").split()


def make_base(size: int, rng: random.Random) -> bytes:
    """Log/CSV-like text with occasional binary blobs, deterministic for a given rng."""
    out = bytearray()
    line_no = 0
    while len(out) < size:
        if rng.random() < 0.03:
            out += bytes(rng.getrandbits(8) for _ in range(rng.randint(64, 512)))
            continue
        words = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(4, 12)))
        out += f"{line_no:07d},{rng.randint(0, 10**6)},{words}\n".encode()
        line_no += 1
    return bytes(out[:size])


def _rand(rng: random.Random, n: int) -> bytes:
    return bytes(rng.getrandbits(8) for _ in range(n))


def mutate(base: bytes, kind: str, rng: random.Random) -> bytes:
    n = len(base)
    if kind == "identical":
        return base
    if kind == "append-4k":
        return base + make_base(4096, rng)
    if kind == "prepend-1b":
        return b"#" + base
    if kind == "insert-1k-middle":
        p = n // 2 + rng.randint(-1000, 1000)
        return base[:p] + make_base(1024, rng) + base[p:]
    if kind == "scattered-20-edits":
        buf = bytearray(base)
        for _ in range(20):
            p = rng.randrange(0, n - 16)
            ln = rng.randint(1, 16)
            buf[p:p + ln] = _rand(rng, ln)
        return bytes(buf)
    if kind == "delete-10k":
        p = rng.randrange(0, n - 10240)
        return base[:p] + base[p + 10240:]
    if kind == "move-32k-section":
        p = rng.randrange(0, n // 2)
        section = base[p:p + 32768]
        rest = base[:p] + base[p + 32768:]
        q = rng.randrange(len(rest) // 2, len(rest))
        return rest[:q] + section + rest[q:]
    if kind == "full-rewrite":
        return make_base(n, rng)
    raise ValueError(kind)


WORKLOADS = ["identical", "append-4k", "prepend-1b", "insert-1k-middle",
             "scattered-20-edits", "delete-10k", "move-32k-section", "full-rewrite"]


# ---------------------------------------------------------------- methods

@dataclass
class Result:
    up: int
    down: int
    seconds: float
    correct: bool

    @property
    def total(self) -> int:
        return self.up + self.down


def m_whole(old: bytes, new: bytes) -> Result:
    t = time.perf_counter()
    rebuilt = bytes(new)
    return Result(0, len(new), time.perf_counter() - t, rebuilt == new)


def m_aligned(old: bytes, new: bytes, bs: int = 2048) -> Result:
    t = time.perf_counter()
    old_h = [blake2b(old[i:i + bs], digest_size=BLOCK_HASH).digest() for i in range(0, len(old), bs)]
    up = len(old_h) * BLOCK_HASH
    down = 8  # new length
    out = bytearray()
    for j, i in enumerate(range(0, len(new), bs)):
        blk = new[i:i + bs]
        same = (j < len(old_h) and blake2b(blk, digest_size=BLOCK_HASH).digest() == old_h[j]
                and len(old[i:i + bs]) == len(blk))
        down += 1  # per-block flag
        if same:
            out += old[i:i + bs]
        else:
            down += len(blk)
            out += blk
    return Result(up, down, time.perf_counter() - t, bytes(out) == new)


def _dedupe(old: bytes, new: bytes, chunker) -> Result:
    t = time.perf_counter()
    oc = chunker(old)
    have = dict(zip(chunk_hashes(old, oc, CHUNK_HASH), oc))
    up = len(have) * CHUNK_HASH
    nc = chunker(new)
    nh = chunk_hashes(new, nc, CHUNK_HASH)
    down = len(nh) * CHUNK_HASH  # recipe
    sent: dict[bytes, bytes] = {}
    for h, (o, ln) in zip(nh, nc):
        if h not in have and h not in sent:
            sent[h] = new[o:o + ln]
            down += 4 + ln  # length prefix + body
    out = bytearray()
    for h in nh:
        if h in have:
            o, ln = have[h]
            out += old[o:o + ln]
        else:
            out += sent[h]
    return Result(up, down, time.perf_counter() - t, bytes(out) == new)


def m_fixed_dedupe(old: bytes, new: bytes) -> Result:
    return _dedupe(old, new, lambda d: fixed_chunks(d, 8192))


def m_cdc_dedupe(old: bytes, new: bytes) -> Result:
    return _dedupe(old, new, fastcdc_chunks)


def m_rsync(old: bytes, new: bytes, block_size: int | None = None) -> Result:
    t = time.perf_counter()
    sig_wire = make_signature(old, block_size).to_bytes()
    delta_wire = generate_delta(Signature.from_bytes(sig_wire), new).to_bytes()
    rebuilt = apply_delta(old, Delta.from_bytes(delta_wire))
    return Result(len(sig_wire), len(delta_wire), time.perf_counter() - t, rebuilt == new)


METHODS = {
    "whole-file": m_whole,
    "aligned-blocks": m_aligned,
    "fixed-dedupe": m_fixed_dedupe,
    "cdc-dedupe": m_cdc_dedupe,
    "rsync-delta": m_rsync,
}


# ---------------------------------------------------------------- runner

def run(size: int = 262144, seeds: tuple[int, ...] = (1, 2, 3),
        workloads: list[str] | None = None) -> dict:
    workloads = workloads or WORKLOADS
    table: dict[str, dict[str, dict]] = {}
    all_correct = True
    for wl in workloads:
        table[wl] = {}
        per_method: dict[str, list[Result]] = {m: [] for m in METHODS}
        for seed in seeds:
            rng = random.Random(seed)
            old = make_base(size, rng)
            new = mutate(old, wl, random.Random(seed * 1000 + 7))
            for name, fn in METHODS.items():
                per_method[name].append(fn(old, new))
        for name, results in per_method.items():
            correct = all(r.correct for r in results)
            all_correct &= correct
            k = len(results)
            table[wl][name] = {
                "mean_up": round(sum(r.up for r in results) / k),
                "mean_down": round(sum(r.down for r in results) / k),
                "mean_total": round(sum(r.total for r in results) / k),
                "mean_seconds": round(sum(r.seconds for r in results) / k, 4),
                "all_correct": correct,
            }
    return {"size": size, "seeds": list(seeds), "workloads": table, "all_correct": all_correct}


def block_size_sweep(size: int = 262144, seeds: tuple[int, ...] = (1, 2, 3),
                     sizes: tuple[int, ...] = (128, 256, 512, 700, 2048, 8192),
                     workload: str = "scattered-20-edits") -> dict:
    rows = {}
    for bs in sizes:
        res = []
        for seed in seeds:
            old = make_base(size, random.Random(seed))
            new = mutate(old, workload, random.Random(seed * 1000 + 7))
            res.append(m_rsync(old, new, bs))
        rows[str(bs)] = {"mean_up": round(sum(r.up for r in res) / len(res)),
                         "mean_down": round(sum(r.down for r in res) / len(res)),
                         "mean_total": round(sum(r.total for r in res) / len(res)),
                         "all_correct": all(r.correct for r in res)}
    return {"workload": workload, "size": size, "block_sizes": rows}


def to_markdown(report: dict, sweep: dict | None = None) -> str:
    methods = list(METHODS)
    lines = [f"# deltasync benchmark (file size {report['size']} bytes, seeds {report['seeds']})", "",
             "Mean total wire bytes (receiver->sender + sender->receiver). Lower is better.", "",
             "| workload | " + " | ".join(methods) + " |",
             "|---|" + "---:|" * len(methods)]
    for wl, row in report["workloads"].items():
        best = min(row[m]["mean_total"] for m in methods)
        cells = []
        for m in methods:
            v = row[m]["mean_total"]
            cells.append(f"**{v:,}**" if v == best else f"{v:,}")
        lines.append(f"| {wl} | " + " | ".join(cells) + " |")
    lines += ["", f"All reconstructions byte-identical: **{report['all_correct']}**", "",
              "Mean seconds per sync (pure Python, includes signature + delta + patch):", "",
              "| workload | " + " | ".join(methods) + " |", "|---|" + "---:|" * len(methods)]
    for wl, row in report["workloads"].items():
        lines.append(f"| {wl} | " + " | ".join(f"{row[m]['mean_seconds']:.3f}" for m in methods) + " |")
    if sweep:
        lines += ["", f"## rsync-delta block-size sweep ({sweep['workload']})", "",
                  "| block size | signature up | delta down | total |", "|---:|---:|---:|---:|"]
        for bs, r in sweep["block_sizes"].items():
            lines.append(f"| {bs} | {r['mean_up']:,} | {r['mean_down']:,} | {r['mean_total']:,} |")
    return "\n".join(lines) + "\n"


def main_bench(out_dir: str | None, size: int, seeds: tuple[int, ...]) -> dict:
    report = run(size, seeds)
    sweep = block_size_sweep(size, seeds)
    md = to_markdown(report, sweep)
    if out_dir:
        from pathlib import Path
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "bench.json").write_text(json.dumps({"comparison": report, "sweep": sweep}, indent=2) + "\n")
        (p / "bench.md").write_text(md)
    print(md, end="")
    return report
