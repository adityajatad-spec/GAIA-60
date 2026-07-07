import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .config import load_config
from .core import ResearchAgent, _provider_from_config
from .setup import interactive_setup


def _print_step(event: dict) -> None:
    data = event["data"]
    step = data["step"]

    if event["type"] == "assistant":
        print(f"\n─── Step {step} ───")
        text = data.get("content", "")
        if text:
            for line in text.strip().splitlines():
                print(f"  {line}")
        for tc in data.get("tool_calls", []):
            inp_str = json.dumps(tc["input"], ensure_ascii=False)
            print(f"  ▶ Tool: {tc['name']}({inp_str})")
        print()

    elif event["type"] == "tool_result":
        tool = data["tool"]
        inp = data["input"]
        out = data["output"]

        inp_str = json.dumps(inp, ensure_ascii=False)
        print(f"  ▶ Tool: {tool}({inp_str})")

        if out.startswith("ERROR:"):
            print(f"  ⚠ {out}")
        else:
            lines = out.strip().splitlines()
            display = lines[:6]
            if len(lines) > 6:
                display.append(f"  … ({len(lines) - 6} more lines)")
            for line in display:
                print(f"  └ {line}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="gaia-agent: a CLI research agent that answers complex multi-step questions."
    )
    parser.add_argument("question", nargs="*", help="The question to answer")
    parser.add_argument(
        "-f", "--files", nargs="*", default=[], help="Paths to attached files"
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Model name override (provider-dependent)",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=None,
        help="Max tool-use steps (default: 15)",
    )
    parser.add_argument(
        "--reconfigure",
        action="store_true",
        help="Re-run the setup wizard to change provider or keys",
    )
    parser.add_argument(
        "--verify-provider",
        default=None,
        help="Provider for self-verification (default: same as main provider)",
    )
    parser.add_argument(
        "--votes",
        type=int,
        default=None,
        help="Run N independent runs and pick majority answer (self-consistency voting)",
    )

    args = parser.parse_args()

    question = " ".join(args.question) if args.question else ""

    # ── Interactive setup (first-run or --reconfigure) ──────────────────
    interactive_setup(reconfigure=args.reconfigure)

    if args.reconfigure and not question:
        return

    if not question:
        parser.print_help()
        sys.exit(1)

    cfg = load_config()
    provider = _provider_from_config(cfg)

    verify_provider = None
    if args.verify_provider and args.verify_provider != cfg["provider"]:
        vcfg = load_config()
        vcfg["provider"] = args.verify_provider
        verify_provider = _provider_from_config(vcfg)

    agent = ResearchAgent(
        question=question,
        files=args.files,
        provider=provider,
        verify_provider=verify_provider,
    )
    if args.steps is not None:
        agent.step_budget = args.steps

    print(f"Question: {question}")
    if args.files:
        print(f"Files:    {args.files}")
    print(f"Provider: {cfg['provider']}")
    if verify_provider:
        print(f"Verify:   {args.verify_provider}")
    print(f"Model:    {provider.model}")
    print(f"Budget:   {agent.step_budget} steps")
    if args.votes:
        print(f"Votes:    {args.votes}")
    print("─" * 50)

    try:
        if args.votes:
            result = agent.run_with_voting(n=args.votes, step_callback=_print_step)
        else:
            result = agent.run(step_callback=_print_step)
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(130)

    print("─" * 50)
    print(f"\nFINAL ANSWER: {result['answer']}")
    print(f"(used {result['steps_used']} of {agent.step_budget} steps)")
    if args.votes:
        print(f"(votes: {result.get('votes', 'N/A')})")

    # Write trajectory to logs/<timestamp>.json
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{timestamp}.json"

    with open(log_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nTrajectory written to {log_path}")


if __name__ == "__main__":
    main()
