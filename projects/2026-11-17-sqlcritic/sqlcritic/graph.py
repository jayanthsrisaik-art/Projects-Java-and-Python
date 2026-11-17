"""LangGraph orchestration of the writer and critic agents.

    plan ──▶ write ──▶ execute ──▶ critique ──┬─▶ answer      (critic approves)
               ▲                              ├─▶ write       (issues, attempts left)
               └──────────── feedback ────────┴─▶ abstain     (issues, budget spent)

``mode`` selects the ablation arm:
    single      one attempt, no checks: whatever executes is returned (writer-only baseline)
    exec_retry  retry only on execution errors (the common "run it and retry" pattern)
    critic      full critic: static + runtime + result-contract checks gate the answer
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .contract import Contract, infer_contract
from .critic import Critic, Issue
from .execute import ExecResult, run_sql
from .writers import Attempt, Writer, WriterError

MODES = ("single", "exec_retry", "critic")


class State(TypedDict, total=False):
    question: str
    contract: Contract
    history: list[Attempt]
    sql: str | None
    result: ExecResult | None
    issues: list[Issue]
    status: str


@dataclass
class Answer:
    question: str
    status: str                    # "answered" | "abstained"
    sql: str | None
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    contract: Contract | None = None
    attempts: list[Attempt] = field(default_factory=list)
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"question": self.question, "status": self.status, "sql": self.sql,
                "columns": self.columns, "rows": [list(r) for r in self.rows],
                "contract": self.contract.as_dict() if self.contract else None,
                "attempts": [{"sql": a.sql, "issues": a.issues} for a in self.attempts],
                "seconds": round(self.seconds, 4)}


def build_graph(writer: Writer, con: sqlite3.Connection, mode: str = "critic",
                max_attempts: int = 3, critic: Critic | None = None):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    critic = critic or Critic()
    budget = 1 if mode == "single" else max_attempts

    def plan(state: State) -> State:
        return {"contract": infer_contract(state["question"]), "history": []}

    def write(state: State) -> State:
        try:
            sql = writer.write(state["question"], state["history"])
        except WriterError as exc:
            return {"sql": None, "result": None, "issues": [Issue("writer_error", str(exc))]}
        return {"sql": sql, "issues": []}

    def execute(state: State) -> State:
        if state.get("sql") is None:
            return {}
        return {"result": run_sql(con, state["sql"])}

    def critique(state: State) -> State:
        if state.get("sql") is None:
            issues = state["issues"]
        elif mode == "critic":
            issues = critic.review(state["sql"], state["result"], state["contract"]).issues
        elif not state["result"].ok:
            issues = [Issue("exec_error", f"execution failed: {state['result'].error}")]
        else:
            issues = []
        history = state["history"] + [Attempt(state.get("sql") or "", [i.as_dict() for i in issues])]
        return {"issues": issues, "history": history}

    def route(state: State) -> str:
        if not state["issues"]:
            return "answer"
        if any(i.code == "writer_error" for i in state["issues"]):
            return "abstain"
        return "write" if len(state["history"]) < budget else "abstain"

    g = StateGraph(State)
    g.add_node("plan", plan)
    g.add_node("write", write)
    g.add_node("execute", execute)
    g.add_node("critique", critique)
    g.add_node("answer", lambda s: {"status": "answered"})
    g.add_node("abstain", lambda s: {"status": "abstained"})
    g.add_edge(START, "plan")
    g.add_edge("plan", "write")
    g.add_edge("write", "execute")
    g.add_edge("execute", "critique")
    g.add_conditional_edges("critique", route, {"answer": "answer", "write": "write",
                                                "abstain": "abstain"})
    g.add_edge("answer", END)
    g.add_edge("abstain", END)
    return g.compile()


class Pipeline:
    def __init__(self, writer: Writer, con: sqlite3.Connection, mode: str = "critic",
                 max_attempts: int = 3):
        self.mode = mode
        self.graph = build_graph(writer, con, mode, max_attempts)

    def ask(self, question: str) -> Answer:
        t = time.perf_counter()
        st = self.graph.invoke({"question": question}, {"recursion_limit": 50})
        res: ExecResult | None = st.get("result")
        answered = st["status"] == "answered"
        return Answer(question, st["status"], st.get("sql") if answered else None,
                      res.columns if answered and res else [], res.rows if answered and res else [],
                      st.get("contract"), st.get("history", []), time.perf_counter() - t)
