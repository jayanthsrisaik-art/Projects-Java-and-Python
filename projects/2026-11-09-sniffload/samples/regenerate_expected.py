"""Regenerate samples/expected/* from the sample inputs. Review diffs before committing."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from sniffload import parse, write_clean_csv  # noqa: E402
from sniffload.cli import _describe  # noqa: E402

INPUTS = ["de_orders.csv", "us_payments.csv", "readings_noheader.psv", "ambiguous_dates.csv"]


def main() -> None:
    out = os.path.join(HERE, "expected")
    os.makedirs(out, exist_ok=True)
    for name in INPUTS:
        with open(os.path.join(HERE, name), "rb") as f:
            res = parse(f.read())
        stem = os.path.splitext(name)[0]
        with open(os.path.join(out, f"{stem}.schema.txt"), "w", encoding="utf-8") as f:
            f.write(_describe(res.inference) + "\n")
        write_clean_csv(res, os.path.join(out, f"{stem}.clean.csv"))
        with open(os.path.join(out, f"{stem}.rejects.json"), "w", encoding="utf-8") as f:
            json.dump([r.__dict__ for r in res.rejects], f, indent=2, ensure_ascii=False)
            f.write("\n")
        print("wrote", stem)


if __name__ == "__main__":
    main()
