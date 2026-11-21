"""Tool specifications and the provenance guard.

Rules, applied to every proposed call before it runs:

1. Read-only tools are allowed.
2. **Sink arguments** (recipient, account, amount, URL host, path) must be
   *attested*: every atom in the value must appear in the user's message or in
   a trusted tool result seen earlier.
   - appears only in untrusted content  -> ``confirm`` (likely injected)
   - appears nowhere / cannot be parsed -> ``confirm`` (fail closed: an
     obfuscated or model-composed value is never auto-approved)
3. **Secret egress**: if an egress argument (email body, URL) carries text from
   a secret segment (8+ consecutive words, after URL-decoding), the recipient
   must be named by the user. Recipient attested only by a trusted source
   -> ``confirm``; otherwise -> ``block``.

The strongest outcome wins: block > confirm > allow.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import unquote_plus

from .atoms import atoms_of_value
from .ledger import Ledger

ACTION_RANK = {"allow": 0, "confirm": 1, "block": 2}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: dict[str, str]                      # arg -> JSON type ("string", "number")
    side_effect: bool = False
    sinks: dict[str, str] = field(default_factory=dict)   # arg -> atom kind
    egress: tuple[str, ...] = ()                # args whose content leaves the system
    recipient: str | None = None                # sink arg naming where egress goes


@dataclass(frozen=True)
class ToolCall:
    tool: str
    args: dict
    call_id: str = ""
    origin: str = "plan"                        # mock-model bookkeeping only; guard ignores it


@dataclass
class Decision:
    action: str = "allow"
    reasons: list[str] = field(default_factory=list)

    def escalate(self, action: str, reason: str) -> None:
        self.reasons.append(reason)
        if ACTION_RANK[action] > ACTION_RANK[self.action]:
            self.action = action


_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(unquote_plus(text).lower())


def _shingles(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


class Guard:
    """Provenance guard. Stateless apart from configuration; all state is in the ledger."""

    def __init__(self, specs: dict[str, ToolSpec], secret_ngram: int = 8):
        self.specs = specs
        self.secret_ngram = secret_ngram

    def check(self, call: ToolCall, ledger: Ledger) -> Decision:
        decision = Decision()
        spec = self.specs.get(call.tool)
        if spec is None:
            decision.escalate("block", f"unknown tool {call.tool!r}")
            return decision
        if not spec.side_effect:
            return decision

        recipient_trust = None
        for arg, kind in spec.sinks.items():
            if arg not in call.args:
                continue
            value = call.args[arg]
            atoms = atoms_of_value(kind, value)
            if not atoms:
                decision.escalate("confirm", f"{arg}={value!r}: no recognisable {kind} (cannot verify origin)")
                if arg == spec.recipient:
                    recipient_trust = None
                continue
            worst = "user"
            for atom in sorted(atoms):
                origin = ledger.origin(kind, atom)
                if origin.trust in ("user", "trusted"):
                    if origin.trust == "trusted" and worst == "user":
                        worst = "trusted"
                    continue
                worst = None
                if origin.trust == "untrusted":
                    decision.escalate(
                        "confirm",
                        f"{arg}: {kind} {atom!r} appears only in untrusted content ({', '.join(origin.sources)})")
                else:
                    decision.escalate("confirm", f"{arg}: {kind} {atom!r} appears nowhere in the context")
            if arg == spec.recipient:
                recipient_trust = worst

        self._check_secret_egress(spec, call, ledger, recipient_trust, decision)
        return decision

    def _check_secret_egress(self, spec: ToolSpec, call: ToolCall, ledger: Ledger,
                             recipient_trust: str | None, decision: Decision) -> None:
        if not spec.egress:
            return
        secrets = ledger.secrets()
        if not secrets:
            return
        payload = " ".join(str(call.args.get(a, "")) for a in spec.egress)
        payload_sh = _shingles(_words(payload), self.secret_ngram)
        if not payload_sh:
            return
        leaked = [s.source for s in secrets if payload_sh & _shingles(_words(s.text), self.secret_ngram)]
        if not leaked:
            return
        where = ", ".join(dict.fromkeys(leaked))
        if recipient_trust == "user":
            return  # the user named this destination explicitly
        if recipient_trust == "trusted":
            decision.escalate("confirm", f"sends secret content from {where} to a recipient the user did not name")
        else:
            decision.escalate("block", f"sends secret content from {where} to an unverified recipient")
