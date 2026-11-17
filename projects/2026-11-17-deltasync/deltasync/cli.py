"""Command line: deltasync {signature,delta,patch,sync,inspect,chunks,bench}."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .cdc import chunk_hashes, fastcdc_chunks, fixed_chunks
from .delta import Copy, Delta, generate_delta
from .errors import DeltaError, FormatError
from .patch import apply_delta_stream
from .signature import Signature, make_signature
from .sync import sync_file


def _write(path: str, data: bytes) -> None:
    if path == "-":
        sys.stdout.buffer.write(data)
    else:
        Path(path).write_bytes(data)


def cmd_signature(a) -> int:
    sig = make_signature(Path(a.basis).read_bytes(), a.block_size, a.strong_len)
    wire = sig.to_bytes()
    _write(a.out, wire)
    print(f"signature: {sig.block_count} blocks of {sig.block_size} bytes, "
          f"{len(wire)} bytes", file=sys.stderr)
    return 0


def cmd_delta(a) -> int:
    sig = Signature.from_bytes(Path(a.signature).read_bytes())
    delta = generate_delta(sig, Path(a.target).read_bytes())
    wire = delta.to_bytes()
    _write(a.out, wire)
    print(f"delta: {len(delta.ops)} ops, copy {delta.copy_bytes} B, literal "
          f"{delta.literal_bytes} B, {len(wire)} bytes", file=sys.stderr)
    return 0


def cmd_patch(a) -> int:
    delta = Delta.from_bytes(Path(a.delta).read_bytes())
    basis = Path(a.basis)
    out = Path(a.out)
    tmp = out.with_name(f".{out.name}.partial")
    with basis.open("rb") as b, tmp.open("wb") as o:
        try:
            apply_delta_stream(b, basis.stat().st_size, delta, o)
        except DeltaError:
            o.close()
            tmp.unlink()
            raise
    tmp.replace(out)
    print(f"patched: wrote {delta.target_len} bytes to {out} (hash verified)", file=sys.stderr)
    return 0


def cmd_sync(a) -> int:
    report = sync_file(a.src, a.dst, a.block_size, a.strong_len, a.dry_run)
    print(json.dumps(report.as_dict(), indent=2))
    return 0


def inspect_text(delta: Delta, wire_len: int) -> str:
    lines = [f"target_len   {delta.target_len}",
             f"target_hash  {delta.target_hash.hex()[:16]}...",
             f"wire_bytes   {wire_len}",
             f"ops          {len(delta.ops)}",
             f"copy_bytes   {delta.copy_bytes}",
             f"literal_bytes {delta.literal_bytes}",
             "---"]
    pos = 0
    for op in delta.ops:
        if isinstance(op, Copy):
            lines.append(f"@{pos:>8}  COPY    basis[{op.offset}:{op.offset + op.length}]  ({op.length} B)")
        else:
            preview = op.data[:24].decode("ascii", "replace")
            lines.append(f"@{pos:>8}  LITERAL {op.length} B  {preview!r}")
        pos += op.length
    return "\n".join(lines) + "\n"


def cmd_inspect(a) -> int:
    wire = Path(a.delta).read_bytes()
    sys.stdout.write(inspect_text(Delta.from_bytes(wire), len(wire)))
    return 0


def cmd_chunks(a) -> int:
    old = Path(a.old).read_bytes()
    new = Path(a.new).read_bytes()
    out = {}
    for name, fn in (("fixed", lambda d: fixed_chunks(d, a.avg)),
                     ("fastcdc", lambda d: fastcdc_chunks(d, a.avg // 4, a.avg, a.avg * 8))):
        oc, nc = fn(old), fn(new)
        have = set(chunk_hashes(old, oc))
        nh = chunk_hashes(new, nc)
        reused = sum(ln for h, (_, ln) in zip(nh, nc) if h in have)
        out[name] = {"old_chunks": len(oc), "new_chunks": len(nc),
                     "reused_bytes": reused, "new_bytes": len(new) - reused,
                     "reuse_ratio": round(reused / len(new), 4) if new else 1.0}
    print(json.dumps(out, indent=2))
    return 0


def cmd_bench(a) -> int:
    from .bench import main_bench
    seeds = tuple(range(1, a.seeds + 1))
    report = main_bench(a.out_dir, a.size, seeds)
    return 0 if report["all_correct"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="deltasync", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def strength(sp):
        sp.add_argument("--block-size", type=int, default=None,
                        help="block size in bytes (default: rsync-style sqrt heuristic, min 700)")
        sp.add_argument("--strong-len", type=int, default=8, help="strong hash bytes (1-32)")

    s = sub.add_parser("signature", help="compute block signature of a basis file")
    s.add_argument("basis"); s.add_argument("out"); strength(s)
    s.set_defaults(fn=cmd_signature)

    s = sub.add_parser("delta", help="compute delta of TARGET against a signature")
    s.add_argument("signature"); s.add_argument("target"); s.add_argument("out")
    s.set_defaults(fn=cmd_delta)

    s = sub.add_parser("patch", help="apply delta to basis, verify, write OUT atomically")
    s.add_argument("basis"); s.add_argument("delta"); s.add_argument("out")
    s.set_defaults(fn=cmd_patch)

    s = sub.add_parser("sync", help="make DST identical to SRC via the delta protocol")
    s.add_argument("src"); s.add_argument("dst"); strength(s)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_sync)

    s = sub.add_parser("inspect", help="print the ops in a delta file")
    s.add_argument("delta"); s.set_defaults(fn=cmd_inspect)

    s = sub.add_parser("chunks", help="compare fixed vs FastCDC chunk reuse between two files")
    s.add_argument("old"); s.add_argument("new")
    s.add_argument("--avg", type=int, default=8192, help="average/fixed chunk size (power of 2)")
    s.set_defaults(fn=cmd_chunks)

    s = sub.add_parser("bench", help="run the synthetic transfer-cost benchmark")
    s.add_argument("--size", type=int, default=262144)
    s.add_argument("--seeds", type=int, default=3)
    s.add_argument("--out-dir", default=None)
    s.set_defaults(fn=cmd_bench)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (FormatError, DeltaError) as exc:
        print(f"deltasync: error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"deltasync: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
