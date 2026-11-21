"""Command line: ``python -m toolfence eval`` and ``python -m toolfence run``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import deny_all, run_agent
from .evaluate import DEFENCES, build, evaluate, format_table, load_scenarios, make_oracle, score
from .mock_model import MockModel
from .policy import Decision, ToolCall
from .tools import TOOL_SPECS, Environment

DEFAULT_SCENARIOS = Path(__file__).resolve().parent.parent / "scenarios"


def ask_user(call: ToolCall, decision: Decision) -> bool:
    print(f"\n  CONFIRM {call.tool}({json.dumps(call.args)[:200]})")
    for r in decision.reasons:
        print(f"    - {r}")
    return input("  approve? [y/N] ").strip().lower() == "y"


def cmd_eval(args: argparse.Namespace) -> int:
    scenarios = load_scenarios(args.scenarios)
    if args.split:
        scenarios = [s for s in scenarios if s.get("split") == args.split]
    report = evaluate(scenarios)
    print(format_table(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.json}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    scenarios = {s["id"]: s for s in load_scenarios(args.scenarios)}
    if args.scenario not in scenarios:
        print(f"unknown scenario {args.scenario!r}; available: {', '.join(sorted(scenarios))}", file=sys.stderr)
        return 2
    sc = scenarios[args.scenario]
    guard, filt = build(args.defence)
    approver = {"deny": deny_all, "oracle": make_oracle(sc), "ask": ask_user}[args.approve]
    if args.model == "claude":
        from .claude_model import ClaudeModel
        model = ClaudeModel(TOOL_SPECS, model=args.claude_model)
    else:
        model = MockModel(sc["plan"], TOOL_SPECS)
    env = Environment.from_scenario(sc)
    try:
        trace = run_agent(sc["user"], model, env, guard, filt, approver)
    except Exception as exc:  # live backend only: missing credentials, network, API errors
        if args.model != "claude":
            raise
        print(f"Claude backend failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    out = trace.to_dict()
    out["effects"] = env.effects
    out["score"] = score(sc, trace).__dict__
    if args.model == "claude":
        out["final_text"] = model.final_text
    print(json.dumps(out, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="toolfence", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", help="run all scenarios under every defence")
    e.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS))
    e.add_argument("--split", choices=["dev", "test"])
    e.add_argument("--json", help="write the full report here")
    e.set_defaults(func=cmd_eval)
    r = sub.add_parser("run", help="run one scenario and print the trace")
    r.add_argument("scenario")
    r.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS))
    r.add_argument("--defence", choices=DEFENCES, default="toolfence")
    r.add_argument("--approve", choices=["deny", "oracle", "ask"], default="deny")
    r.add_argument("--model", choices=["mock", "claude"], default="mock")
    r.add_argument("--claude-model", default="claude-opus-5-5")
    r.set_defaults(func=cmd_run)
    args = p.parse_args(argv)
    return args.func(args)
