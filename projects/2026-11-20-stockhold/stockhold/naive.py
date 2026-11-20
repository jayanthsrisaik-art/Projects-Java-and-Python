"""Baseline with the same business rules but weaker transaction handling.

* ``CheckThenActStore`` runs each statement in autocommit mode: the stock check
  and the INSERT are separate transactions, so two requests can both see the
  last unit as available (the classic tutorial bug).

It exists only to be measured against :class:`stockhold.store.Store`.
"""
from __future__ import annotations

import contextlib
import sqlite3
from typing import Iterator

from .store import Store


class CheckThenActStore(Store):
    @contextlib.contextmanager
    def _write_tx(self) -> Iterator[sqlite3.Connection]:
        yield self.conn()  # isolation_level=None -> every statement autocommits


STRATEGIES: dict[str, type[Store]] = {
    "immediate": Store,
    "check-then-act": CheckThenActStore,
}
