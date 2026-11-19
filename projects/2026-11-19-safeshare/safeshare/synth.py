"""Synthetic clinic data + a synthetic public "voter roll" for linkage attacks.

Everything is generated from a seed; no real person is represented.  Zip codes
follow a skewed (Zipf-like) distribution inside four 3-digit prefixes and the
diagnosis depends on age and sex, so some small groups are homogeneous - the
situation where k-anonymity alone still discloses the sensitive value.
"""

from __future__ import annotations

import csv
import datetime as dt
import random
from pathlib import Path

REFERENCE_DATE = dt.date(2026, 1, 1)
PREFIXES = ("021", "024", "017", "019")
FIRST = ("Avery Blake Casey Dana Eden Finley Gray Harper Indigo Jordan Kai Logan "
         "Morgan Noel Oakley Parker Quinn Riley Sage Taylor Uma Val Wren Yael "
         "Zion Arden Bowie Cruz Devon Ellis Frankie Hollis Jules Lennox Marlow").split()
LAST = ("Abara Brennan Castellanos Dimitrov Eriksen Fairweather Galloway Haddad "
        "Ishikawa Jovanovic Kowalczyk Lindqvist Moreau Nakamura Okonkwo Petrakis "
        "Quintero Rasmussen Sorensen Takahashi Underwood Valdivia Whitlock Xiong "
        "Yamamoto Zielinski Achterberg Bellweather Cortland Delacroix").split()
PLANS = ("bronze", "silver", "gold", "medicaid", "medicare")
DIAGNOSES = ("asthma", "hypertension", "diabetes", "migraine", "anxiety",
             "arthritis", "copd", "pregnancy", "dementia")


def _diagnosis(rng: random.Random, age: int, sex: str) -> str:
    w = {"asthma": 3, "hypertension": 1 + age / 15, "diabetes": 1 + age / 25,
         "migraine": 3 if age < 50 else 1, "anxiety": 3 if age < 45 else 1.5,
         "arthritis": max(0.2, (age - 40) / 10), "copd": max(0.1, (age - 50) / 12),
         "pregnancy": 4 if sex == "F" and 18 <= age <= 42 else 0,
         "dementia": max(0.0, (age - 70) / 4)}
    names = list(w)
    return rng.choices(names, weights=[w[n] for n in names])[0]


def generate(out_dir: str | Path, patients: int = 2000, seed: int = 0,
             voter_fraction: float = 0.7, non_patient_voters: float = 0.5) -> dict:
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    zips = [p + f"{i:02d}" for p in PREFIXES for i in rng.sample(range(1, 99), 10)]
    zip_w = [1 / (r + 1) ** 0.9 for r in range(len(zips))]
    rng.shuffle(zip_w)

    def person():
        age = min(95, max(18, int(rng.gauss(48, 19))))
        dob = REFERENCE_DATE - dt.timedelta(days=age * 365 + rng.randrange(365))
        sex = "F" if rng.random() < 0.52 else "M"
        z = rng.choices(zips, weights=zip_w)[0]
        first, last = rng.choice(FIRST), rng.choice(LAST)
        return age, dob, sex, z, first, last

    prows, erows, vrows = [], [], []
    enc = 0
    for n in range(patients):
        age, dob, sex, z, first, last = person()
        pid = f"P{n + 1:06d}"
        diag = _diagnosis(rng, age, sex)
        email = f"{first}.{last}{rng.randrange(100)}@example.org".lower()
        phone = f"617-{rng.randrange(200, 999)}-{rng.randrange(10000):04d}"
        prows.append([pid, f"{first} {last}", email, phone, dob.isoformat(), sex, z,
                      rng.choice(PLANS), diag])
        for _ in range(max(1, int(rng.expovariate(1 / 3)))):
            enc += 1
            day = dt.date(2024, 1, 1) + dt.timedelta(days=rng.randrange(731))
            cost = round(rng.lognormvariate(5, 0.8), 2)
            note = f"Follow-up for {diag}; contact {first} at {phone}."
            erows.append([f"E{enc:07d}", pid, day.isoformat(), f"{cost:.2f}", note])
        if rng.random() < voter_fraction:
            vrows.append([f"{first} {last}", dob.isoformat(), sex, z, pid])
    for _ in range(int(patients * non_patient_voters)):
        age, dob, sex, z, first, last = person()
        vrows.append([f"{first} {last}", dob.isoformat(), sex, z, ""])
    rng.shuffle(vrows)

    def write(name, header, rows):
        with (out / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(header)
            w.writerows(rows)

    write("patients.csv", ["patient_id", "name", "email", "phone", "dob", "sex", "zip",
                           "plan", "diagnosis"], prows)
    write("encounters.csv", ["encounter_id", "patient_id", "visit_date", "cost",
                             "notes"], erows)
    # The voter roll is the attacker's background knowledge, kept *outside* the
    # release input.  `_truth_patient_id` exists only so the evaluator can score
    # the attack; an attacker would not have it.
    voters_dir = out.parent / (out.name + "_public")
    voters_dir.mkdir(parents=True, exist_ok=True)
    with (voters_dir / "voters.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["name", "dob", "sex", "zip", "_truth_patient_id"])
        w.writerows(vrows)
    return {"patients": len(prows), "encounters": len(erows), "voters": len(vrows),
            "voters_path": str(voters_dir / "voters.csv")}
