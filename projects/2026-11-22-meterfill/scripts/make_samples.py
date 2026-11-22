"""Regenerate samples/: a small fleet with every fault type, and the expected outputs.

Expected outputs use the deterministic ``profile`` method so they do not depend on
the scikit-learn version. Run from the project root:  python scripts/make_samples.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meterfill.cli import _csv_rows  # noqa: E402
from meterfill.simulate import SimConfig, simulate, write_world  # noqa: E402
from meterfill.store import Store  # noqa: E402

SAMPLE_CONFIG = SimConfig(
    households=6, days=10, seed=1, rollover_frac=0.3, exchange_frac=0.2,
    misread_rate=1 / 250, outages_per_meter=1.5, trailing_frac=0.15,
)


def expected_outputs(samples: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        st = Store(Path(tmp) / "s.db")
        st.ingest_weather(_csv_rows(str(samples / "weather.csv"), ["hour_ts", "temp_c"]))
        ingest = st.ingest_readings(_csv_rows(str(samples / "readings.csv"), ["meter_id", "ts", "register_kwh"]))
        build = st.rebuild("profile")
        meters = st.meter_ids()
        out = {
            "ingest": ingest,
            "build": build,
            "issues": {m: st.issues(m) for m in meters},
            "daily_usage": {m: [{"start": r["start"], "kwh": r["kwh"], "hours": r["hours"]} for r in st.usage(m, None, None, "day")["rows"]] for m in meters},
        }
        st.close()
    return out


if __name__ == "__main__":
    samples = ROOT / "samples"
    world = simulate(SAMPLE_CONFIG)
    paths = write_world(world, samples, include_truth=True)
    (samples / "expected_outputs.json").write_text(json.dumps(expected_outputs(samples), indent=1) + "\n")
    print("wrote", sorted(Path(p).name for p in paths.values()), "and expected_outputs.json")
