"""deltasync: rsync-style binary delta sync plus content-defined chunking.

Public API:
    weak_checksum, RollingChecksum           -- rolling weak checksum (rollsum)
    make_signature, Signature                -- basis-side block signatures
    generate_delta, Delta, Copy, Literal     -- target-side delta generation
    apply_delta                              -- reconstruct target from basis + delta
    sync_file                                -- end-to-end local file sync with verification
    fastcdc_chunks, fixed_chunks             -- chunkers used for dedupe comparisons
"""
from .errors import DeltaError, FormatError
from .rollsum import RollingChecksum, weak_checksum
from .signature import Signature, default_block_size, make_signature
from .delta import Copy, Delta, Literal, generate_delta
from .patch import apply_delta
from .sync import SyncReport, sync_file
from .cdc import fastcdc_chunks, fixed_chunks

__all__ = [
    "DeltaError", "FormatError", "RollingChecksum", "weak_checksum", "Signature",
    "default_block_size", "make_signature", "Copy", "Delta", "Literal", "generate_delta",
    "apply_delta", "SyncReport", "sync_file", "fastcdc_chunks", "fixed_chunks",
]
__version__ = "1.0.0"
