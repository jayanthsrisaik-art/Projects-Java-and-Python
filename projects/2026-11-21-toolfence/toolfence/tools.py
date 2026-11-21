"""A sandboxed personal-assistant environment: inbox, web, files, contacts, and
side-effecting actions that only append to an effect log.

Each tool result carries the trust level the guard should assign to it:

* ``lookup_contact`` and the user's own files: ``trusted``
* inbox, web pages, downloaded files: ``untrusted`` (anyone can write them)
* results of side-effecting tools: ``untrusted`` - they echo arguments the
  model chose, and labelling them trusted would let an injected value be
  *laundered* into a trusted segment by one harmless-looking call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .atoms import canon_host, canon_path
from .policy import ToolCall, ToolSpec

TOOL_SPECS: dict[str, ToolSpec] = {s.name: s for s in [
    ToolSpec("read_inbox", "List the emails in the user's inbox (sender, subject, body).", {}),
    ToolSpec("fetch_url", "Fetch a web page and return its text.", {"url": "string"},
             side_effect=True, sinks={"url": "host"}, egress=("url",), recipient="url"),
    ToolSpec("read_file", "Read a file from the user's workspace.", {"path": "string"}),
    ToolSpec("lookup_contact", "Look up a contact's email address by name.", {"name": "string"}),
    ToolSpec("send_email", "Send an email.", {"to": "string", "subject": "string", "body": "string"},
             side_effect=True, sinks={"to": "email"}, egress=("subject", "body"), recipient="to"),
    ToolSpec("transfer_money", "Transfer money from the user's account.",
             {"to_account": "string", "amount": "number", "memo": "string"},
             side_effect=True, sinks={"to_account": "account", "amount": "amount"}),
    ToolSpec("delete_file", "Delete a file from the user's workspace.", {"path": "string"},
             side_effect=True, sinks={"path": "path"}),
]}


@dataclass
class ToolResult:
    text: str
    trust: str
    secret: bool = False
    error: bool = False


@dataclass
class Environment:
    inbox: list[dict] = field(default_factory=list)
    web: dict[str, str] = field(default_factory=dict)
    files: dict[str, dict] = field(default_factory=dict)
    contacts: dict[str, str] = field(default_factory=dict)
    effects: list[dict] = field(default_factory=list)

    @classmethod
    def from_scenario(cls, sc: dict) -> "Environment":
        return cls(inbox=list(sc.get("inbox", [])), web=dict(sc.get("web", {})),
                   files={canon_path(p): dict(v) for p, v in sc.get("files", {}).items()},
                   contacts=dict(sc.get("contacts", {})))

    def execute(self, call: ToolCall) -> ToolResult:
        handler = getattr(self, f"_t_{call.tool}", None)
        if handler is None or call.tool not in TOOL_SPECS:
            return ToolResult(f"error: unknown tool {call.tool}", "untrusted", error=True)
        try:
            return handler(**call.args)
        except TypeError as exc:
            return ToolResult(f"error: bad arguments: {exc}", "untrusted", error=True)

    # read-only tools -----------------------------------------------------
    def _t_read_inbox(self) -> ToolResult:
        msgs = [{"id": i + 1, **m} for i, m in enumerate(self.inbox)]
        return ToolResult(json.dumps(msgs, indent=1, ensure_ascii=False) if msgs else "(inbox empty)", "untrusted")

    def _t_read_file(self, path: str) -> ToolResult:
        f = self.files.get(canon_path(path))
        if f is None:
            return ToolResult(f"error: no such file {path}", "untrusted", error=True)
        return ToolResult(f["text"], f.get("trust", "trusted"), secret=bool(f.get("secret")))

    def _t_lookup_contact(self, name: str) -> ToolResult:
        for cname, email in self.contacts.items():
            if name.lower() in cname.lower():
                return ToolResult(f"{cname} <{email}>", "trusted")
        return ToolResult(f"no contact matching {name!r}", "trusted")

    # side-effecting tools ------------------------------------------------
    def _t_fetch_url(self, url: str) -> ToolResult:
        self.effects.append({"tool": "fetch_url", "url": url})
        base = url.split("?", 1)[0].rstrip("/")
        for page_url, text in self.web.items():
            if page_url.rstrip("/") == base:
                return ToolResult(text, "untrusted")
        from urllib.parse import urlsplit
        host = canon_host(urlsplit(url).hostname or "")
        return ToolResult(f"(no content at {host})", "untrusted")

    def _t_send_email(self, to: str, subject: str = "", body: str = "") -> ToolResult:
        self.effects.append({"tool": "send_email", "to": to, "subject": subject, "body": body})
        return ToolResult(f"email sent to {to}", "untrusted")

    def _t_transfer_money(self, to_account: str, amount: float, memo: str = "") -> ToolResult:
        self.effects.append({"tool": "transfer_money", "to_account": to_account, "amount": amount, "memo": memo})
        return ToolResult(f"transferred {amount} to {to_account}", "untrusted")

    def _t_delete_file(self, path: str) -> ToolResult:
        existed = self.files.pop(canon_path(path), None) is not None
        self.effects.append({"tool": "delete_file", "path": path, "existed": existed})
        return ToolResult(f"deleted {path}" if existed else f"no such file {path}", "untrusted")
