"""Local end-to-end sync: update ``dst`` to equal ``src`` using the delta protocol.

Both ends run in one process here, but every step goes through the serialized wire
formats so the byte counts are the ones a real two-party exchange would send:

    receiver: signature(dst)  --sig bytes-->  sender
    sender:   delta(sig, src) --delta bytes-->  receiver
    receiver: patch(dst, delta) -> temp file -> fsync -> atomic rename over dst

If the patched output fails whole-file verification (a strong-hash collision, or dst
changed underneath us) the exchange is retried once with 32-byte strong hashes, the
same recovery idea rsync uses; if that also fails, the whole file is sent.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .delta import Delta, generate_delta
from .errors import DeltaError
from .patch import apply_delta
from .signature import Signature, make_signature


@dataclass
class SyncReport:
    method: str               # "delta", "delta-retry", "whole-file", "unchanged-noop"
    target_len: int
    signature_bytes: int = 0
    delta_bytes: int = 0
    literal_bytes: int = 0
    copy_bytes: int = 0
    attempts: int = 0

    @property
    def wire_bytes(self) -> int:
        if self.method == "whole-file":
            return self.signature_bytes + self.delta_bytes + self.target_len
        return self.signature_bytes + self.delta_bytes

    def as_dict(self) -> dict:
        d = asdict(self)
        d["wire_bytes"] = self.wire_bytes
        return d


def _write_atomic(dst: Path, data: bytes, mode_from: Path | None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=dst.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if mode_from is not None:
            shutil.copymode(mode_from, tmp)
        os.replace(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def exchange(basis: bytes, target: bytes, block_size: int | None,
             strong_len: int) -> tuple[bytes, bytes, Delta]:
    """Sender + wire half of one protocol round: returns (sig_wire, delta_wire, delta)."""
    sig_wire = make_signature(basis, block_size, strong_len).to_bytes()
    delta = generate_delta(Signature.from_bytes(sig_wire), target)
    return sig_wire, delta.to_bytes(), delta


def sync_file(src: str | os.PathLike, dst: str | os.PathLike, block_size: int | None = None,
              strong_len: int = 8, dry_run: bool = False) -> SyncReport:
    src, dst = Path(src), Path(dst)
    target = src.read_bytes()
    if not dst.exists():
        report = SyncReport("whole-file", len(target), attempts=0)
        if not dry_run:
            _write_atomic(dst, target, src)
        return report
    basis = dst.read_bytes()

    report = SyncReport("delta", len(target))
    strengths = (strong_len, 32) if strong_len < 32 else (32,)
    for attempt, slen in enumerate(strengths, start=1):
        report.attempts = attempt
        sig_wire, delta_wire, delta = exchange(basis, target, block_size, slen)
        report.signature_bytes += len(sig_wire)
        report.delta_bytes += len(delta_wire)
        try:
            result = apply_delta(basis, Delta.from_bytes(delta_wire))
        except DeltaError:
            continue
        report.literal_bytes = delta.literal_bytes
        report.copy_bytes = delta.copy_bytes
        if attempt > 1:
            report.method = "delta-retry"
        if result == basis:
            report.method = "unchanged-noop"
        elif not dry_run:
            _write_atomic(dst, result, src)
        return report

    report.method = "whole-file"
    if not dry_run:
        _write_atomic(dst, target, src)
    return report
