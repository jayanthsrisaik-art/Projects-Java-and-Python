"""Command line: synth | scan | release | evaluate | bench."""

from __future__ import annotations

import argparse
import json
import sys

from . import anonymize, piiscan, synth
from .policy import PolicyError, load_policy
from .pseudonym import KeyError_, load_key
from .qi import QIError
from .release import METHODS, PIIFound, read_tables, run_release

EXIT_POLICY, EXIT_PII, EXIT_INFEASIBLE = 2, 3, 4


def _key(args) -> bytes:
    return load_key(getattr(args, "key_file", None), demo=getattr(args, "demo_key", False))


def _add_key(p):
    g = p.add_mutually_exclusive_group()
    g.add_argument("--key-file", help="file holding the HMAC secret (>= 16 bytes)")
    g.add_argument("--demo-key", action="store_true",
                   help="built-in key for synthetic data only")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="safeshare", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synth", help="generate synthetic clinic data + voter roll")
    p.add_argument("--out", required=True)
    p.add_argument("--patients", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("scan", help="PII-scan verbatim columns only")
    p.add_argument("--input", required=True)
    p.add_argument("--policy", required=True)

    p = sub.add_parser("release", help="build an anonymised extract")
    p.add_argument("--input", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--method", choices=METHODS, default="mondrian")
    p.add_argument("--k", type=int)
    p.add_argument("--l", type=int)
    _add_key(p)

    p = sub.add_parser("evaluate", help="linkage attack + query utility")
    p.add_argument("--original", required=True)
    p.add_argument("--release", required=True)
    p.add_argument("--voters", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--queries", type=int, default=400)
    _add_key(p)

    p = sub.add_parser("bench", help="multi-seed benchmark of all methods")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--patients", type=int, default=5000)
    p.add_argument("--work", default="work/bench")
    p.add_argument("--out", default="results")

    args = ap.parse_args(argv)
    try:
        if args.cmd == "synth":
            print(json.dumps(synth.generate(args.out, args.patients, args.seed), indent=2))
        elif args.cmd == "scan":
            pol = load_policy(args.policy)
            findings = piiscan.scan(read_tables(args.input, pol), pol)
            print(json.dumps(findings, indent=2))
            return EXIT_PII if findings else 0
        elif args.cmd == "release":
            pol = load_policy(args.policy, args.k, args.l)
            rep = run_release(args.input, pol, args.out, _key(args), args.method,
                              demo_key=args.demo_key)
            print(json.dumps({k: rep[k] for k in ("method", "subjects",
                              "suppressed_subjects", "verified", "avg_ncp")}, indent=2))
        elif args.cmd == "evaluate":
            from .evaluate import evaluate
            pol = load_policy(args.policy)
            print(json.dumps(evaluate(args.original, args.release, args.voters, pol,
                                      _key(args), args.queries), indent=2))
        elif args.cmd == "bench":
            from .bench import run_bench
            run_bench(args.seeds, args.patients, args.work, args.out)
    except PIIFound as exc:
        print(f"refusing to release: {exc}", file=sys.stderr)
        print(json.dumps(exc.findings, indent=2), file=sys.stderr)
        return EXIT_PII
    except (PolicyError, QIError, KeyError_) as exc:
        print(f"policy/input error: {exc}", file=sys.stderr)
        return EXIT_POLICY
    except anonymize.Infeasible as exc:
        print(f"infeasible: {exc}", file=sys.stderr)
        return EXIT_INFEASIBLE
    return 0


if __name__ == "__main__":
    sys.exit(main())
