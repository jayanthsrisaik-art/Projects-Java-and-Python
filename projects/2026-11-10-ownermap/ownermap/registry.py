"""Who is who: users, team membership and e-mail aliases (a local JSON file).

Format::

    {
      "users": ["@alice", "@bob"],
      "teams": {"@acme/web": ["@alice", "@bob"]},
      "emails": {"docs@example.com": "@carol"}
    }

Every member of a team and every e-mail target is implicitly a known user.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class Registry:
    users: set[str] = field(default_factory=set)
    teams: dict[str, list[str]] = field(default_factory=dict)
    emails: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "Registry":
        teams = {k.lower(): list(v) for k, v in data.get("teams", {}).items()}
        emails = {k.lower(): v for k, v in data.get("emails", {}).items()}
        users = {u.lower() for u in data.get("users", [])}
        users |= {m.lower() for ms in teams.values() for m in ms}
        users |= {v.lower() for v in emails.values()}
        return cls(users, teams, emails)

    @classmethod
    def from_file(cls, path: str) -> "Registry":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    # GitHub handles and team slugs are case-insensitive.
    def is_team(self, owner: str) -> bool:
        return owner.lower() in self.teams

    def knows(self, owner: str) -> bool:
        o = owner.lower()
        return o in self.users or o in self.teams or o in self.emails

    def members(self, owner: str) -> list[str]:
        return self.teams.get(owner.lower(), [])

    def approvers(self, owner: str) -> set[str]:
        """People whose approval satisfies ``owner`` (lower-cased handles)."""
        o = owner.lower()
        if o in self.teams:
            return {m.lower() for m in self.teams[o]}
        if o in self.emails:
            return {self.emails[o].lower()}
        return {o}
