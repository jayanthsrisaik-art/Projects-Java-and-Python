"""The agent loop: model proposes tool calls, the guard decides, the
environment executes, and every result the model sees goes into the ledger.

The model interface is three methods, implemented by ``MockModel`` and
``ClaudeModel``::

    start(task)                       -> None
    next_actions()                    -> list[ToolCall] | None   (None = finished)
    observe(call, text, is_error)     -> None
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from .ledger import Ledger
from .policy import Decision, Guard, ToolCall
from .tools import Environment

Approver = Callable[[ToolCall, Decision], bool]
OutputFilter = Callable[[str], str]


class Model(Protocol):
    def start(self, task: str) -> None: ...
    def next_actions(self) -> list[ToolCall] | None: ...
    def observe(self, call: ToolCall, result: str, is_error: bool = False) -> None: ...


def deny_all(call: ToolCall, decision: Decision) -> bool:
    """Headless default: nothing that needs confirmation runs."""
    return False


@dataclass
class Step:
    call: ToolCall
    decision: Decision
    executed: bool
    approved: bool | None   # None when no confirmation was asked
    result: str


@dataclass
class Trace:
    task: str
    steps: list[Step] = field(default_factory=list)
    ledger: Ledger = field(default_factory=Ledger)

    def executed_calls(self) -> list[ToolCall]:
        return [s.call for s in self.steps if s.executed]

    def visible_tool_text(self) -> str:
        return "\n".join(seg.text for seg in self.ledger.segments if seg.source != "user")

    def to_dict(self) -> dict:
        return {"task": self.task, "steps": [
            {"tool": s.call.tool, "args": s.call.args, "origin": s.call.origin,
             "decision": s.decision.action, "reasons": s.decision.reasons,
             "approved": s.approved, "executed": s.executed,
             "result": s.result if len(s.result) <= 300 else s.result[:300] + "..."}
            for s in self.steps]}


def run_agent(task: str, model: Model, env: Environment, guard: Guard | None = None,
              output_filter: OutputFilter | None = None, approver: Approver = deny_all,
              max_calls: int = 25) -> Trace:
    trace = Trace(task)
    trace.ledger.add("user", "user", task)
    model.start(task)
    calls = 0
    while calls < max_calls:
        actions = model.next_actions()
        if not actions:
            break
        for call in actions:
            calls += 1
            decision = guard.check(call, trace.ledger) if guard else Decision()
            approved: bool | None = None
            if decision.action == "confirm":
                approved = bool(approver(call, decision))
            run = decision.action == "allow" or (decision.action == "confirm" and approved)
            if run:
                result = env.execute(call)
                shown = output_filter(result.text) if output_filter else result.text
                # The ledger records what the model actually saw. A withheld
                # result attests nothing; a secret one stays marked secret.
                trace.ledger.add(f"tool:{call.tool}", result.trust, shown, secret=result.secret)
                model.observe(call, shown, is_error=result.error)
                trace.steps.append(Step(call, decision, True, approved, shown))
            else:
                verdict = "blocked by policy" if decision.action == "block" else "not approved by the user"
                msg = f"error: call {verdict}: " + "; ".join(decision.reasons)
                model.observe(call, msg, is_error=True)
                trace.steps.append(Step(call, decision, False, approved, msg))
    return trace
