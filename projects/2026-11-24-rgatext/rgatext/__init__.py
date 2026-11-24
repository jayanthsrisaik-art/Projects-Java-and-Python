"""rgatext: a replicated text CRDT (RGA) with causal delivery and delta sync."""

from .doc import ROOT, CRDTError, ReceiveReport, Replica, validate_op
from .oracle import oracle_text

__all__ = ["ROOT", "CRDTError", "ReceiveReport", "Replica", "validate_op", "oracle_text"]
__version__ = "0.1.0"
