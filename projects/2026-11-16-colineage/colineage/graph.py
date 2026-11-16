"""Project-wide column graph: end-to-end tracing to sources, impact analysis, explanations.

Edges come from ``ModelResult``: ``model.col`` has value edges to its direct upstream columns
(with a kind) and control edges to every column in the model's control set. Composition:

* value ∘ value = value, kind = max of the two kinds
* anything involving a control edge = control (changing the upstream column can change which
  rows exist upstream, hence this column's values)
"""

from __future__ import annotations

from dataclasses import dataclass

from .analyzer import KIND_NAMES
from .project import ModelResult


@dataclass(frozen=True)
class Dep:
    value: int | None  # strongest kind if there is a value path, else None
    control: bool

    def to_json(self) -> dict:
        return {"value": KIND_NAMES[self.value] if self.value is not None else None, "control": self.control}


def _combine(a: Dep | None, b: Dep) -> Dep:
    if a is None:
        return b
    v = b.value if a.value is None else a.value if b.value is None else max(a.value, b.value)
    return Dep(v, a.control or b.control)


class LineageGraph:
    def __init__(self, results: dict[str, ModelResult], sources: set[str]):
        self.results = results
        self.sources = sources  # source table names
        self._memo: dict[str, dict[str, Dep]] = {}

    # nodes ------------------------------------------------------------------------------------
    def model_columns(self) -> list[str]:
        return [f"{m}.{c}" for m, r in self.results.items() if r.status == "ok" for c in r.columns]

    def is_source(self, node: str) -> bool:
        return node.split(".", 1)[0] in self.sources

    def direct(self, node: str) -> dict[str, Dep]:
        model, _, col = node.partition(".")
        r = self.results.get(model)
        if r is not None and r.status == "ok" and col == "*":  # row existence of the model
            return {up: Dep(None, True) for up in r.control}
        if r is None or r.status != "ok" or col not in r.columns:
            raise KeyError(f"unknown model column: {node}")
        out: dict[str, Dep] = {}
        for up, kind in r.columns[col].items():
            out[up] = _combine(out.get(up), Dep(kind, False))
        for up in r.control | r.column_control.get(col, set()):
            out[up] = _combine(out.get(up), Dep(None, True))
        return out

    # upstream ---------------------------------------------------------------------------------
    def trace(self, node: str) -> dict[str, Dep]:
        """Source columns this model column depends on, with role and strongest kind."""
        if node in self._memo:
            return self._memo[node]
        out: dict[str, Dep] = {}
        for up, edge in self.direct(node).items():
            if self.is_source(up):
                out[up] = _combine(out.get(up), edge)
                continue
            for src, d in self.trace(up).items():
                if edge.value is not None:  # value edge: carry the upstream role through
                    v = None if d.value is None else max(edge.value, d.value)
                    out[src] = _combine(out.get(src), Dep(v, d.control))
                if edge.control:  # control edge: everything behind it only controls
                    out[src] = _combine(out.get(src), Dep(None, True))
        self._memo[node] = out
        return out

    # downstream -------------------------------------------------------------------------------
    def impact(self, source_col: str, value_only: bool = False) -> dict[str, Dep]:
        """Model columns whose values can change when ``source_col`` changes."""
        out = {}
        for node in self.model_columns():
            d = self.trace(node).get(source_col)
            if d is None or (value_only and d.value is None):
                continue
            out[node] = d
        return out

    def explain(self, node: str, source_col: str) -> list[list[tuple[str, str]]]:
        """All edge paths node -> ... -> source_col as [(upstream, role/kind), ...]."""
        paths: list[list[tuple[str, str]]] = []

        def walk(cur: str, acc: list[tuple[str, str]]) -> None:
            if len(paths) >= 20:
                return
            for up, e in self.direct(cur).items():
                labels = []
                if e.value is not None:
                    labels.append(KIND_NAMES[e.value])
                if e.control:
                    labels.append("control")
                step = acc + [(up, "+".join(labels))]
                if up == source_col:
                    paths.append(step)
                elif not self.is_source(up) and source_col in self.trace(up):
                    walk(up, step)

        if not self.is_source(node) and source_col in self.trace(node):
            walk(node, [])
        return paths
