"""Regenerate the sample files deterministically: python samples/make_samples.py"""
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from deltasync.bench import make_base  # noqa: E402


def main() -> None:
    rng = random.Random(2026)
    lines = [f"{i:05d},{rng.choice(['north','south','east','west'])},"
             f"{rng.randint(1, 500)},{rng.randint(100, 99999) / 100:.2f}\n" for i in range(600)]
    basis = "id,region,qty,price\n" + "".join(lines)
    # target: header changed, a row inserted near the top, a block of rows moved to the end,
    # one price edited, and rows appended
    t = lines[:]
    t.insert(10, "00010a,north,1,0.99\n")
    moved = t[200:260]
    del t[200:260]
    t += moved
    t[400] = t[400].rsplit(",", 1)[0] + ",12345.67\n"
    t += [f"{i:05d},east,{i % 7 + 1},1.00\n" for i in range(600, 640)]
    target = "id,region,qty,unit_price\n" + "".join(t)
    (HERE / "basis.csv").write_text(basis)
    (HERE / "target.csv").write_text(target)
    # larger mixed text/binary pair for the chunking comparison
    big = make_base(65536, random.Random(7))
    (HERE / "log_v1.bin").write_bytes(big)
    (HERE / "log_v2.bin").write_bytes(b"0000000,BOOT,inserted header line\n" + big[:40000] + big[41000:])


if __name__ == "__main__":
    main()
