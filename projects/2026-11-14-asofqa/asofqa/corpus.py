"""Seeded synthetic corpus: versioned HR/finance policies plus questions with ground truth.

Each *policy family* (e.g. paid time off) has a chronology of versions:

* active versions that took effect one after another (the "truth" timeline),
* optionally a *scheduled* version: approved and active, but effective after the reference date,
* optionally a *withdrawn* version: had a planned effective date in the past but never took effect,
* optionally a *proposed* draft: not approved, planned for the future.

Documents are Markdown files with a small front-matter header. Later versions mention the value
they replace, so the old number appears in the new document — a deliberate trap for readers.

Ground truth is computed from the generator's own schedule, never from the pipeline.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

from .textutil import norm_number

DEFAULT_REF_DATE = date(2026, 11, 14)
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


@dataclass(frozen=True)
class Topic:
    key: str
    title: str
    ladder: tuple
    monotone: bool
    value_str: str  # how a value is written in questions, e.g. "{v} days"
    value_sentences: tuple
    purpose: str
    distractors: tuple  # each contains {n}, a unit-bearing number unrelated to the answer
    distractor_range: tuple
    dev_names: tuple
    test_names: tuple


def _money(v) -> str:
    return f"{v:,}" if isinstance(v, int) else f"{v}"


TOPICS: tuple[Topic, ...] = (
    Topic(
        "pto", "Paid Time Off Policy", (15, 18, 20, 22, 25, 27, 30), True, "{v} days",
        ("Full-time employees receive {v} days of paid time off per calendar year.",
         "Each full-time employee is granted {v} days of paid time off annually."),
        "This policy covers vacation and personal days for all staff.",
        ("Requests should be submitted at least {n} days in advance.",
         "No more than {n} days of paid time off may be taken consecutively without director approval."),
        (3, 14),
        ("paid time off allowance", "number of paid time off days"),
        ("vacation allowance", "annual vacation days for staff"),
    ),
    Topic(
        "remote", "Remote Work Policy", (1, 2, 3, 4, 5), False, "{v} days per week",
        ("Employees may work remotely up to {v} days per week.",
         "Staff are permitted to work remotely on as many as {v} days per week."),
        "This policy explains hybrid and work from home arrangements.",
        ("Requests for remote days must be agreed with your manager {n} days ahead.",
         "Equipment issues must be reported within {n} days."),
        (2, 9),
        ("remote work allowance", "number of remote work days per week"),
        ("work from home allowance", "hybrid days allowed each week"),
    ),
    Topic(
        "meal", "Business Travel Meal Policy", (40, 50, 55, 60, 65, 75, 90), True, "${v}",
        ("The daily meal allowance during business travel is ${v}.",
         "Travellers may claim a daily meal allowance of up to ${v} while on business travel."),
        "This policy sets the per diem for food on work trips.",
        ("Receipts must be submitted within {n} days of returning.",
         "Alcohol above ${n} per trip is never reimbursed."),
        (10, 30),
        ("daily meal allowance for business travel", "business travel meal allowance"),
        ("per diem for food on work trips", "food per diem when travelling"),
    ),
    Topic(
        "parental", "Parental Leave Policy", (8, 12, 14, 16, 18, 20, 26), True, "{v} weeks",
        ("Eligible employees receive {v} weeks of paid parental leave.",
         "Paid parental leave of {v} weeks is available to every eligible employee."),
        "This policy covers maternity and paternity leave for new parents.",
        ("Leave must be taken within {n} months of the birth or adoption.",
         "Employees should notify HR at least {n} weeks before leave begins."),
        (2, 7),
        ("paid parental leave", "length of paid parental leave"),
        ("maternity and paternity leave", "leave for new parents"),
    ),
    Topic(
        "equipment", "Home Office Equipment Policy", (300, 400, 500, 600, 750, 1000), True, "${v}",
        ("New hires receive a one-time home office equipment stipend of ${v}.",
         "A one-time home office equipment stipend of ${v} is provided to each new hire."),
        "This policy helps staff buy desks, chairs and monitors for their home setup.",
        ("Purchases must be claimed within {n} days of the start date.",
         "Items costing more than ${n} need a receipt."),
        (25, 90),
        ("home office equipment stipend", "equipment stipend for new hires"),
        ("home setup budget for desks and monitors", "home setup stipend"),
    ),
    Topic(
        "match", "Retirement Plan Match Policy", (3, 4, 5, 6, 7, 8), True, "{v}%",
        ("Employer matching of retirement plan contributions is capped at {v}% of base salary.",
         "The company matches retirement plan contributions up to {v}% of base salary."),
        "This policy describes the 401(k) employer match.",
        ("Matching contributions vest over {n} months.",
         "Contribution changes take effect within {n} days."),
        (10, 36),
        ("retirement plan match", "employer retirement contribution match"),
        ("401(k) matching percentage", "401(k) employer match"),
    ),
    Topic(
        "oncall", "On-Call Compensation Policy", (150, 200, 250, 300, 350, 400), True, "${v}",
        ("Engineers on the on-call rotation receive a stipend of ${v} per week on call.",
         "Each week on the on-call rotation is compensated with a stipend of ${v}."),
        "This policy covers pager duty pay for engineers.",
        ("Pages answered after midnight earn {n} days of time off in lieu per quarter at most.",
         "Shifts may be swapped with {n} days notice."),
        (1, 5),
        ("on-call stipend", "weekly on-call rotation stipend"),
        ("pager duty pay", "weekly compensation for pager duty"),
    ),
    Topic(
        "learning", "Learning and Development Policy", (500, 1000, 1500, 2000, 2500, 3000), True, "${v}",
        ("Each employee has an annual learning budget of ${v} for courses, books and conferences.",
         "An annual learning budget of ${v} per employee covers courses, books and conferences."),
        "This policy funds training and professional education.",
        ("Unused budget up to ${n} may roll over once.",
         "Course requests must be approved {n} days before enrolment."),
        (100, 400),
        ("annual learning budget", "learning budget for courses and conferences"),
        ("training budget", "professional education allowance"),
    ),
    Topic(
        "mileage", "Vehicle Mileage Reimbursement Policy",
        (0.545, 0.56, 0.585, 0.625, 0.655, 0.67, 0.7), True, "${v} per mile",
        ("Business use of a personal vehicle is reimbursed at ${v} per mile.",
         "Mileage driven in a personal vehicle for business is reimbursed at ${v} per mile."),
        "This policy sets the rate paid for driving a personal car on company business.",
        ("Mileage logs must be filed within {n} days.",
         "Parking fees above ${n} require a receipt."),
        (15, 45),
        ("mileage reimbursement rate", "personal vehicle mileage reimbursement"),
        ("rate paid for driving a personal car", "car reimbursement rate"),
    ),
    Topic(
        "sick", "Sick Leave Policy", (5, 6, 8, 10, 12, 15), True, "{v} days",
        ("Employees accrue {v} paid sick days per year.",
         "Every employee accrues {v} paid sick days each year."),
        "This policy covers absence for illness and medical appointments.",
        ("A doctor's note is required after {n} days of consecutive absence.",
         "Up to {n} days may be used to care for a family member."),
        (2, 5),
        ("paid sick day allowance", "number of paid sick days"),
        ("days off for illness", "annual sick day entitlement"),
    ),
)
TOPIC_BY_KEY = {t.key: t for t in TOPICS}


@dataclass
class Version:
    doc_id: str
    policy: str
    version: int
    status: str  # active | proposed | withdrawn
    effective: date
    value: object
    kind: str  # past | scheduled | proposed | withdrawn
    withdrawn_on: date | None = None


@dataclass
class Family:
    topic: Topic
    versions: list[Version] = field(default_factory=list)

    def active(self) -> list[Version]:
        return sorted((v for v in self.versions if v.status == "active"), key=lambda v: v.effective)


def fmt_value(topic: Topic, v) -> str:
    return topic.value_str.format(v=_money(v))


def _first_of_month(d: date) -> date:
    return date(d.year, d.month, 1)


def _add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    return date(d.year + y, m + 1, 1)


def build_family(topic: Topic, rng: random.Random, ref: date) -> Family:
    n_active = rng.randint(2, 4)
    has_sched = rng.random() < 0.35
    has_withdrawn = rng.random() < 0.30
    has_proposed = rng.random() < 0.30
    total = n_active + has_sched + has_withdrawn + has_proposed
    while total > len(topic.ladder):
        n_active -= 1
        total -= 1
    values = rng.sample(list(topic.ladder), total)
    timeline_vals = values[: n_active + has_sched]
    extra_vals = values[n_active + has_sched:]
    if topic.monotone:
        timeline_vals.sort()

    # Past effective dates, first-of-month, 8–26 months apart, all before ref - 6 weeks.
    latest_allowed = ref - timedelta(days=45)
    while True:
        start = _first_of_month(date(2020, 1, 1) + timedelta(days=rng.randint(0, 900)))
        dates = [start]
        for _ in range(n_active - 1):
            dates.append(_add_months(dates[-1], rng.randint(8, 26)))
        if dates[-1] <= latest_allowed:
            break
    if has_sched:
        dates.append(_add_months(_first_of_month(ref), rng.randint(2, 7)))

    fam = Family(topic)
    specs = []
    for i, (d, v) in enumerate(zip(dates, timeline_vals)):
        specs.append(dict(status="active", effective=d, value=v, kind="scheduled" if d > ref else "past"))
    ev = iter(extra_vals)
    if has_withdrawn:
        # Planned to take effect between two past active versions (or after the last past one).
        past = [s for s in specs if s["kind"] == "past"]
        i = rng.randrange(len(past))
        lo = past[i]["effective"]
        hi = past[i + 1]["effective"] if i + 1 < len(past) else latest_allowed
        span = (hi - lo).days
        if span > 70:
            eff = _first_of_month(lo + timedelta(days=rng.randint(40, span - 30)))
            if lo < eff < hi:
                specs.append(dict(status="withdrawn", effective=eff, value=next(ev), kind="withdrawn",
                                  withdrawn_on=eff - timedelta(days=rng.randint(5, 20))))
    if has_proposed:
        eff = _add_months(_first_of_month(ref), rng.randint(3, 10))
        specs.append(dict(status="proposed", effective=eff, value=next(ev, None), kind="proposed"))
    specs = [s for s in specs if s["value"] is not None]
    specs.sort(key=lambda s: s["effective"])
    for n, s in enumerate(specs, start=1):
        fam.versions.append(Version(doc_id=f"{topic.key}-v{n}", policy=topic.key, version=n, **s))
    return fam


def render_doc(fam: Family, ver: Version, rng: random.Random) -> str:
    t = fam.topic
    head = (
        "---\n"
        f"doc_id: {ver.doc_id}\npolicy: {t.key}\ntitle: {t.title}\nversion: {ver.version}\n"
        f"status: {ver.status}\neffective: {ver.effective.isoformat()}\n"
        "---\n"
    )
    paras = [f"# {t.title} (version {ver.version})\n\n{t.purpose} It applies to all employees of Northwind Labs."]
    value_s = rng.choice(t.value_sentences).format(v=_money(ver.value))
    lo, hi = t.distractor_range
    distract = rng.choice(t.distractors).format(n=rng.randint(lo, hi))
    body = [value_s, distract]
    rng.shuffle(body)
    paras.append(" ".join(body))
    # What did this version replace?  (The value in effect just before its effective date.)
    prior = [v for v in fam.active() if v.effective < ver.effective]
    if prior:
        p = prior[-1]
        paras.append(
            f"This version replaces the previous figure of {fmt_value(t, p.value)}, "
            f"in effect since {p.effective.isoformat()}. "
            + rng.choice(t.distractors).format(n=rng.randint(lo, hi))
        )
    if ver.kind == "past":
        appr = ver.effective - timedelta(days=30)
        paras.append(f"Approved by the People Team on {appr.isoformat()}. Effective {ver.effective.isoformat()}.")
    elif ver.kind == "scheduled":
        appr = min(ver.effective - timedelta(days=60), DEFAULT_REF_DATE - timedelta(days=10))
        paras.append(
            f"Approved on {appr.isoformat()}; this change takes effect on {ver.effective.isoformat()}. "
            "Until then the previous version continues to apply."
        )
    elif ver.kind == "proposed":
        paras.append(
            f"DRAFT. This proposal is under review and has not been approved. "
            f"If adopted it would take effect on {ver.effective.isoformat()}."
        )
    else:
        paras.append(
            f"WITHDRAWN. This proposal was withdrawn on {ver.withdrawn_on.isoformat()} "
            f"and never took effect. It had been planned for {ver.effective.isoformat()}."
        )
    return head + "\n\n".join(paras) + "\n"


# ---------------------------------------------------------------- ground truth

def values_in_range(fam: Family, start: date, end: date) -> list[tuple[Version, date, date | None]]:
    """Active versions whose validity interval [eff, next_eff) intersects [start, end]."""
    act = fam.active()
    out = []
    for i, v in enumerate(act):
        nxt = act[i + 1].effective if i + 1 < len(act) else None
        valid_to = (nxt - timedelta(days=1)) if nxt else None
        if v.effective <= end and (valid_to is None or valid_to >= start):
            out.append((v, v.effective, valid_to))
    return out


def gold_values(fam: Family, start: date, end: date) -> tuple[list[str], list[str]]:
    hits = values_in_range(fam, start, end)
    vals, ids = [], []
    for v, _, _ in hits:
        nv = norm_number(str(v.value))
        if not vals or vals[-1] != nv:
            vals.append(nv)
        ids.append(v.doc_id)
    return vals, ids


# ---------------------------------------------------------------- questions

def _iso(d: date) -> str:
    return d.isoformat()


def _long(d: date) -> str:
    return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"


def _dmy(d: date) -> str:
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def _us(d: date) -> str:
    return f"{d.month}/{d.day}/{d.year}"


NUM_WORDS = {2: "two", 3: "three", 4: "four"}

# Templates are split by phrasing.  The *test* phrasings were written together with the dev ones and
# frozen before the first evaluation run; the parser was never edited after looking at test results.
TEMPLATES = {
    "dev": {
        "current": ["What is the {name}?", "What is the current {name}?"],
        "point": [("What was the {name} on {d}?", _iso), ("As of {d}, what was the {name}?", _long)],
        "year": ["What was the {name} in {y}?"],
        "month": ["What was the {name} during {m} {y}?"],
        "quarter": ["What was the {name} in Q{q} {y}?"],
        "half": None,
        "last_year": ["What was the {name} last year?"],
        "years_ago": ["What was the {name} {nw} years ago?"],
        "future": [("What will the {name} be on {d}?", _iso)],
        "previous": ["What was the {name} before the most recent change?"],
        "when": ["When did the {name} become {vs}?"],
    },
    "test": {
        "current": ["These days, what {name} do employees get?", "What is the {name} at present?"],
        "point": [("What {name} was in effect on {d}?", _dmy), ("Looking back to {d}, what was the {name}?", _us)],
        "year": ["Throughout {y}, what was the {name}?"],
        "month": ["What was the {name} for {m} of {y}?"],
        "quarter": ["During Q{q} {y}, what {name} applied?"],
        "half": ["What {name} applied in the second half of {y}?"],
        "last_year": ["What {name} applied last year?"],
        "years_ago": ["What was the {name} a year ago?"],
        "future": [("Starting {d}, what will the {name} be?", _long)],
        "previous": ["What did the {name} used to be before it was last revised?"],
        "when": ["Since when has the {name} been {vs}?", "On what date did the {name} change to {vs}?"],
    },
}


def _rand_date(rng: random.Random, lo: date, hi: date) -> date:
    return lo + timedelta(days=rng.randint(0, max(0, (hi - lo).days)))


def make_questions(fams: list[Family], rng: random.Random, ref: date, split: str) -> list[dict]:
    T = TEMPLATES[split]
    qs: list[dict] = []

    def add(fam, qtype, text, kind, vals=None, ids=None, gdate=None):
        abstain = (kind == "values" and not vals) or (kind == "date" and gdate is None)
        qs.append({
            "qid": f"{split}-{fam.topic.key}-{len([q for q in qs if q['policy'] == fam.topic.key])}",
            "policy": fam.topic.key, "qtype": qtype, "question": text, "kind": kind,
            "gold_values": vals or [], "gold_date": gdate, "gold_docs": ids or [], "abstain": abstain,
        })

    for fam in fams:
        names = fam.topic.test_names if split == "test" else fam.topic.dev_names
        name = lambda: rng.choice(names)  # noqa: E731
        act = fam.active()
        past = [v for v in act if v.effective <= ref]

        # current
        vals, ids = gold_values(fam, ref, ref)
        add(fam, "current", rng.choice(T["current"]).format(name=name()), "values", vals, ids)

        # two explicit points: one inside a random past interval, one anywhere 2019..ref
        for d in (_rand_date(rng, past[0].effective, ref), _rand_date(rng, date(2019, 1, 1), ref)):
            tpl, f = rng.choice(T["point"])
            vals, ids = gold_values(fam, d, d)
            add(fam, "point", tpl.format(name=name(), d=f(d)), "values", vals, ids)

        # one period question
        kinds = ["year", "month", "quarter"] + (["half"] if T["half"] else [])
        pk = rng.choice(kinds)
        y = rng.randint(2020, ref.year - 1)
        if pk == "year":
            s, e, text = date(y, 1, 1), date(y, 12, 31), rng.choice(T["year"]).format(name=name(), y=y)
        elif pk == "month":
            m = rng.randint(1, 12)
            s, e = date(y, m, 1), _add_months(date(y, m, 1), 1) - timedelta(days=1)
            text = rng.choice(T["month"]).format(name=name(), m=MONTHS[m - 1], y=y)
        elif pk == "quarter":
            q = rng.randint(1, 4)
            s, e = date(y, 3 * q - 2, 1), _add_months(date(y, 3 * q - 2, 1), 3) - timedelta(days=1)
            text = rng.choice(T["quarter"]).format(name=name(), q=q, y=y)
        else:
            s, e = date(y, 7, 1), date(y, 12, 31)
            text = rng.choice(T["half"]).format(name=name(), y=y)
        vals, ids = gold_values(fam, s, e)
        add(fam, "range", text, "values", vals, ids)

        # relative
        if rng.random() < 0.5:
            s, e = date(ref.year - 1, 1, 1), date(ref.year - 1, 12, 31)
            text = rng.choice(T["last_year"]).format(name=name())
        else:
            n = rng.choice([2, 3]) if split == "dev" else 1
            s = e = date(ref.year - n, ref.month, ref.day)
            text = rng.choice(T["years_ago"]).format(name=name(), nw=NUM_WORDS.get(n, n))
        vals, ids = gold_values(fam, s, e)
        add(fam, "relative", text, "values", vals, ids)

        # future point (scheduled versions count, proposals do not)
        sched = [v for v in act if v.effective > ref]
        d = (sched[0].effective + timedelta(days=rng.randint(0, 40))) if sched else _rand_date(
            rng, ref + timedelta(days=30), ref + timedelta(days=300))
        tpl, f = rng.choice(T["future"])
        vals, ids = gold_values(fam, d, d)
        add(fam, "future", tpl.format(name=name(), d=f(d)), "values", vals, ids)

        # previous value (before the version in effect today)
        if len(past) >= 2:
            add(fam, "previous", rng.choice(T["previous"]).format(name=name()), "values",
                [norm_number(str(past[-2].value))], [past[-2].doc_id])

        # when did it become X?  X may be a withdrawn/proposed value -> must abstain.
        cand = [v for v in fam.versions if v.version > 1 or v.status != "active"]
        v = rng.choice(cand)
        gd = v.effective.isoformat() if v.status == "active" else None
        text = rng.choice(T["when"]).format(name=name(), vs=fmt_value(fam.topic, v.value))
        add(fam, "when", text, "date", gdate=gd, ids=[v.doc_id] if gd else [])
    return qs


def generate(seed: int, split: str, ref: date = DEFAULT_REF_DATE):
    rng = random.Random(seed)
    fams = [build_family(t, rng, ref) for t in TOPICS]
    docs = {v.doc_id: render_doc(f, v, rng) for f in fams for v in f.versions}
    qs = make_questions(fams, rng, ref, split)
    return fams, docs, qs


def write_corpus(out_dir: Path, seed: int, split: str, ref: date = DEFAULT_REF_DATE) -> tuple[int, int]:
    fams, docs, qs = generate(seed, split, ref)
    out_dir = Path(out_dir)
    (out_dir / "docs").mkdir(parents=True, exist_ok=True)
    for doc_id, text in docs.items():
        (out_dir / "docs" / f"{doc_id}.md").write_text(text, encoding="utf-8")
    with open(out_dir / "questions.jsonl", "w", encoding="utf-8") as fh:
        for q in qs:
            fh.write(json.dumps(q) + "\n")
    timeline = [
        {**{k: (val.isoformat() if isinstance(val, date) else val) for k, val in asdict(v).items()}}
        for f in fams for v in f.versions
    ]
    (out_dir / "timeline.json").write_text(json.dumps({"ref_date": ref.isoformat(), "versions": timeline}, indent=1))
    return len(docs), len(qs)
