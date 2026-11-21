"""A deterministic stand-in for an LLM agent, used for offline evaluation.

It has two behaviours:

* **Planner.** It carries out the scenario's scripted plan (the calls a correct
  assistant would make), but only uses a value once it has *seen* it: a plan
  step whose sink values (recipient, account, amount, host, path) are not
  visible in the user message or in tool output shown to the model is skipped.
  If a defence withholds a tool result, steps that depended on it fail, as
  they would for a real model.
* **Obedient reader.** After every tool result it looks for instructions in
  the text (send/forward to an address, transfer an amount to an account,
  delete a file, open a URL) and carries them out at once, de-obfuscating
  "name (at) domain (dot) com" style addresses. When forwarding, it puts
  everything it has read into the body.

This is a worst case for the attacker side, not a measurement of how often a
real model falls for an injection. Real models follow some injections some of
the time; the mock follows every one it can parse. Defences are therefore
measured on whether they stop an attack *once the model has been fooled*.
"""

from __future__ import annotations

import re
from urllib.parse import quote_plus

from .atoms import atoms_of_value, extract
from .policy import ToolCall, ToolSpec

_DEOBF = [
    (re.compile(r"\s*[\[(\{]\s*at\s*[\])\}]\s*", re.I), "@"),
    (re.compile(r"\s*[\[(\{]\s*dot\s*[\])\}]\s*", re.I), "."),
    (re.compile(r"\b([\w.+-]+)\s+at\s+([\w-]+(?:\s+dot\s+[\w-]+)+)\b", re.I),
     lambda m: m.group(1) + "@" + re.sub(r"\s+dot\s+", ".", m.group(2), flags=re.I)),
]
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])|\n+")
_TRANSFER_V = re.compile(r"\b(transfer|wire|send|pay|remit)\b", re.I)
_EMAIL_V = re.compile(r"\b(send|forward|email|e-mail|mail|share|cc|bcc|reply)\b", re.I)
_DELETE_V = re.compile(r"\b(delete|remove|erase|wipe)\b", re.I)
_FETCH_V = re.compile(r"\b(visit|open|fetch|load|go to|browse|navigate to|check)\b", re.I)
_URL = re.compile(r"https?://[^\s<>\"'()\[\]{}]+")


def deobfuscate(text: str) -> str:
    for rx, repl in _DEOBF:
        text = rx.sub(repl, text)
    return text


def parse_instructions(text: str, context: str) -> list[ToolCall]:
    """Turn imperative sentences in ``text`` into tool calls (the obedient reader).

    ``context`` is everything the model has read so far; it becomes the body of
    forwarded mail and the payload appended to exfiltration URLs.
    """
    calls: list[ToolCall] = []
    for sent in _SENT_SPLIT.split(deobfuscate(text)):
        if _TRANSFER_V.search(sent):
            accts, amounts = sorted(extract("account", sent)), sorted(extract("amount", sent))
            if accts and amounts:
                calls.append(ToolCall("transfer_money",
                                      {"to_account": accts[0], "amount": float(amounts[0]), "memo": "as requested"},
                                      origin="injection"))
                continue
        if _EMAIL_V.search(sent):
            emails = sorted(extract("email", sent))
            if emails and re.search(r"\b(to|with|at)\b", sent, re.I):
                calls.append(ToolCall("send_email", {"to": emails[0], "subject": "Requested information",
                                                     "body": context}, origin="injection"))
                continue
        if _DELETE_V.search(sent):
            paths = sorted(extract("path", sent))
            if paths:
                calls.append(ToolCall("delete_file", {"path": paths[0]}, origin="injection"))
                continue
        if _FETCH_V.search(sent):
            m = _URL.search(sent)
            if m:
                url = m.group(0).rstrip(".,;")
                if url.endswith("="):
                    url += quote_plus(context[:400])
                calls.append(ToolCall("fetch_url", {"url": url}, origin="injection"))
    return calls


class MockModel:
    name = "mock"

    def __init__(self, plan: list[dict], specs: dict[str, ToolSpec], follow_injections: bool = True):
        self.plan = [ToolCall(s["tool"], dict(s.get("args", {})), origin="plan") for s in plan]
        self.specs = specs
        self.follow_injections = follow_injections
        self.visible: list[str] = []
        self.read_content: list[str] = []
        self.queue: list[ToolCall] = []
        self.done: list[ToolCall] = []
        self.skipped: list[ToolCall] = []
        self._step = 0
        self._n = 0

    def start(self, task: str) -> None:
        self.visible = [task]

    def _seen(self, call: ToolCall) -> bool:
        spec = self.specs.get(call.tool)
        if spec is None:
            return True
        seen_text = "\n".join(self.visible)
        for arg, kind in spec.sinks.items():
            if arg not in call.args:
                continue
            if not atoms_of_value(kind, call.args[arg]) <= extract(kind, seen_text):
                return False
        return True

    def _id(self, call: ToolCall) -> ToolCall:
        self._n += 1
        return ToolCall(call.tool, call.args, f"call_{self._n}", call.origin)

    def next_actions(self) -> list[ToolCall] | None:
        if self.queue:
            return [self._id(self.queue.pop(0))]
        while self._step < len(self.plan):
            step = self.plan[self._step]
            self._step += 1
            if self._seen(step):
                return [self._id(step)]
            self.skipped.append(step)
        return None

    def observe(self, call: ToolCall, result: str, is_error: bool = False) -> None:
        self.done.append(call)
        self.visible.append(result)
        if is_error or not self.follow_injections:
            return
        if call.tool in ("read_inbox", "fetch_url", "read_file"):
            self.read_content.append(result)
        known = {(c.tool, repr(sorted(c.args.items()))) for c in self.plan + self.done + self.queue}
        for injected in parse_instructions(result, "\n\n".join(self.read_content)):
            key = (injected.tool, repr(sorted(injected.args.items())))
            if key in known or self._matches_plan(injected):
                continue
            known.add(key)
            self.queue.append(injected)

    def _matches_plan(self, call: ToolCall) -> bool:
        spec = self.specs.get(call.tool)
        for step in self.plan:
            if step.tool != call.tool or spec is None:
                continue
            if all(atoms_of_value(k, step.args.get(a)) == atoms_of_value(k, call.args.get(a))
                   for a, k in spec.sinks.items()):
                return True
        return False
