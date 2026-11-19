import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from safeshare.policy import load_policy  # noqa: E402
from safeshare.synth import generate  # noqa: E402

POLICY = ROOT / "samples" / "policy.json"


@pytest.fixture(scope="session")
def clinic(tmp_path_factory):
    d = tmp_path_factory.mktemp("data") / "clinic"
    info = generate(d, patients=600, seed=7)
    return d, Path(info["voters_path"])


@pytest.fixture
def policy():
    return load_policy(POLICY)


@pytest.fixture
def policy_doc():
    return json.loads(POLICY.read_text())


def random_records(n, seed, l_vals=4):
    from safeshare.qi import QISpec
    rng = random.Random(seed)
    specs = [QISpec("dob", "date"), QISpec("sex", "categorical"), QISpec("zip", "zip")]
    recs = []
    for _ in range(n):
        recs.append({"qi": (rng.randint(700000, 712000), rng.choice("FMX"),
                            rng.choice([2101, 2102, 2150, 2400, 1701, 1999])),
                     "sens": rng.choice("abcdefg"[:l_vals]), "key": None})
    for d, s in enumerate(specs):
        s.fit(r["qi"][d] for r in recs)
    return specs, recs
