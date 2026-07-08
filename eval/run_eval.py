"""
GAIA evaluation harness.

Loads the GAIA validation set (JSONL), runs each question through a
ResearchAgent, scores via quasi-exact-match, and logs results.

Defaults to --provider anthropic (regardless of .env PROVIDER setting).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.core import ResearchAgent
from agent.providers.base import LLMProvider
from agent.providers.anthropic_provider import AnthropicProvider
from agent.providers.ollama_provider import OllamaProvider
from agent.providers.openai_provider import OpenAIProvider


# ── Provider factory (eval always defaults to anthropic) ───────────────────


def _build_provider(provider_name: str, model: str | None = None) -> LLMProvider:
    name = provider_name.lower().strip()
    if name == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not found in environment. "
                "Set it in .env or export it."
            )
        return AnthropicProvider(
            api_key=api_key,
            model=model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        )
    if name == "ollama":
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        resolved_model = model or os.getenv("OLLAMA_MODEL", "qwen3:8b-64k")
        return OllamaProvider(model=resolved_model, host=host)
    if name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not found in environment. "
                "Set it in .env or export it."
            )
        return OpenAIProvider(
            api_key=api_key,
            model=model or os.getenv("OPENAI_MODEL", "gpt-4o"),
        )
    raise ValueError(f"Unsupported provider: {provider_name}")


# ── GAIA quasi-exact-match scoring ─────────────────────────────────────────


def _normalize(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_number(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def _normalize_number(x: str) -> str:
    return f"{float(x):.10f}"


# ── Compound-match helpers ────────────────────────────────────────────


_BOOL_WORDS = {
    "yes": "yes", "no": "no",
    "true": "true", "false": "false",
    "even": "even", "odd": "odd",
}


def _extract_numbers(text: str) -> list[float]:
    """Extract all integers and floats in reading order."""
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]


def _find_bool_terms(text: str) -> set[str]:
    """Return the set of boolean-like terms found in *text*."""
    found: set[str] = set()
    for word in re.findall(r"[a-zA-Z]+", text.lower()):
        if word in _BOOL_WORDS:
            found.add(_BOOL_WORDS[word])
    return found


def _is_compound(expected: str) -> bool:
    """Heuristic: ground truth is compound when it has ≥2 numbers,
    or exactly 1 number plus a boolean-like term."""
    nums = _extract_numbers(expected)
    bools = _find_bool_terms(expected)
    return bool(len(nums) >= 2 or (len(nums) == 1 and bools))


def _nums_equal(a: float, b: float) -> bool:
    """Check numeric equality with relative tolerance (matching _quasi_exact_match single-number logic)."""
    if abs(a) < 1e-8:
        return abs(a - b) < 1e-8
    return abs(a - b) / max(abs(a), abs(b)) < 1e-4


def _is_number_subsequence(pred_nums: list[float], exp_nums: list[float]) -> bool:
    """Check if *exp_nums* appears as an ordered subsequence of *pred_nums* with tolerance."""
    it = iter(pred_nums)
    for en in exp_nums:
        for pn in it:
            if _nums_equal(pn, en):
                break
        else:
            return False
    return True


def _compound_match(predicted: str, expected: str) -> bool:
    """Compound match: all numbers (in order) plus all boolean terms match."""
    pred_nums = _extract_numbers(predicted)
    exp_nums = _extract_numbers(expected)

    if not exp_nums or not _is_number_subsequence(pred_nums, exp_nums):
        return False

    pred_bools = _find_bool_terms(predicted)
    exp_bools = _find_bool_terms(expected)

    if exp_bools and not exp_bools.issubset(pred_bools):
        return False

    return True


def _quasi_exact_match(predicted: str, expected: str) -> bool:
    pred = predicted.strip()
    exp = expected.strip()
    if not pred or not exp:
        return pred == exp

    # Pipe-separated alternatives
    alternatives = [a.strip() for a in exp.split("|")]
    if len(alternatives) > 1:
        return any(_quasi_exact_match(predicted, a) for a in alternatives)

    # Boolean variants (whole-string)
    bool_map = {
        "yes": "true", "no": "false",
        "y": "true", "n": "false",
        "true": "true", "false": "false",
    }
    p_bool = bool_map.get(pred.lower().strip())
    e_bool = bool_map.get(exp.lower().strip())
    if p_bool is not None and e_bool is not None:
        return p_bool == e_bool

    # Numeric match with tolerance (single number)
    if _is_number(pred) and _is_number(exp):
        pn = float(pred)
        en = float(exp)
        if abs(pn) < 1e-6:
            return abs(pn - en) < 1e-6
        if abs(pn - en) / max(abs(pn), abs(en)) < 1e-4:
            return True
        return _normalize_number(pred) == _normalize_number(exp)

    # Compound match (multi-value answers)
    if _is_compound(exp):
        if _compound_match(pred, exp):
            return True

    # Normalized string match
    p_norm = _normalize(pred)
    e_norm = _normalize(exp)
    if p_norm == e_norm:
        return True
    if p_norm in e_norm or e_norm in p_norm:
        return True
    return False


# ── Data loading ───────────────────────────────────────────────────────────


def load_gaia(path: str) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            questions.append(record)
    return questions


# ── Question runner ────────────────────────────────────────────────────────


def run_question(
    record: dict[str, Any],
    provider: LLMProvider,
    step_budget: int,
    votes: int = 1,
) -> dict[str, Any]:
    task_id = record.get("task_id", "unknown")
    question = record.get("Question", "")
    level = str(record.get("Level", "?"))
    ground_truth = record.get("Final answer", "")

    agent = ResearchAgent(
        question=question,
        provider=provider,
        verify_provider=None,
    )
    if step_budget:
        agent.step_budget = step_budget

    t0 = time.time()
    try:
        if votes > 1:
            result = agent.run_with_voting(n=votes)
        else:
            result = agent.run()
    except Exception as exc:
        elapsed = time.time() - t0
        return {
            "task_id": task_id,
            "level": level,
            "question": question,
            "predicted": f"ERROR: {exc}",
            "ground_truth": ground_truth,
            "correct": False,
            "trajectory": getattr(agent, "trajectory", []),
            "provider": provider.__class__.__name__.replace("Provider", "").lower(),
            "model": getattr(provider, "model", "?"),
            "elapsed_seconds": round(elapsed, 2),
            "usage": {},
            "error": str(exc),
        }

    elapsed = time.time() - t0
    predicted = result.get("answer", "")
    correct = _quasi_exact_match(predicted, ground_truth)

    return {
        "task_id": task_id,
        "level": level,
        "question": question,
        "predicted": predicted,
        "ground_truth": ground_truth,
        "correct": correct,
        "trajectory": result.get("trajectory", []),
        "provider": provider.__class__.__name__.replace("Provider", "").lower(),
        "model": getattr(provider, "model", "?"),
        "elapsed_seconds": round(elapsed, 2),
        "steps_used": result.get("steps_used"),
        "format_noncompliant": result.get("format_noncompliant", False),
        "empty_response_failure": result.get("empty_response_failure", False),
        "usage": result.get("usage", {}),
        "votes": result.get("votes"),
        "n_runs": result.get("n_runs"),
    }


# ── Summary printing ───────────────────────────────────────────────────────


def print_summary(results: list[dict[str, Any]]) -> None:
    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    failed = [r for r in results if not r.get("correct")]
    failed_ids = [r["task_id"] for r in failed]

    levels: dict[str, list[dict]] = {}
    for r in results:
        lev = r.get("level", "?")
        levels.setdefault(lev, []).append(r)

    print()
    print("═" * 50)
    print("  GAIA Evaluation Summary")
    print("═" * 50)
    pct = (correct / total * 100) if total else 0.0
    print(f"  Overall:  {correct}/{total} ({pct:.1f}%)")
    print()
    for lev in sorted(levels, key=lambda x: (len(x), x)):
        group = levels[lev]
        g_correct = sum(1 for r in group if r.get("correct"))
        g_pct = (g_correct / len(group) * 100) if group else 0.0
        print(f"  Level {lev}: {g_correct}/{len(group)} ({g_pct:.1f}%)")
    print()
    if failed_ids:
        print(f"  Failed ({len(failed_ids)}): {', '.join(failed_ids)}")
    else:
        print("  Failed: (none)")
    print("═" * 50)
    print()


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GAIA evaluation harness."
    )
    parser.add_argument(
        "--data",
        default=str(
            Path(__file__).resolve().parent / "data" / "gaia_validation.jsonl"
        ),
        help="Path to GAIA validation JSONL (default: eval/data/gaia_validation.jsonl)",
    )
    parser.add_argument(
        "--level",
        type=str,
        default=None,
        help="Filter by level (e.g., '1', '1,2'). Default: all levels.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N questions.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1).",
    )
    parser.add_argument(
        "--provider",
        default="anthropic",
        help="Provider to use (default: anthropic). Override with --provider ollama.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override (provider-dependent).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=15,
        help="Max tool-use steps per question (default: 15).",
    )
    parser.add_argument(
        "--votes",
        type=int,
        default=1,
        help="Run N agents per question and take majority answer (default: 1).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path override (default: eval/results/<timestamp>.jsonl).",
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help="Stop after cumulative spend exceeds this amount (cloud providers only).",
    )

    args = parser.parse_args()

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        print(f"Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    # Load and filter
    all_questions = load_gaia(str(data_path))
    print(f"Loaded {len(all_questions)} questions from {data_path}")

    if args.level:
        allowed = set(args.level.replace(" ", "").split(","))
        all_questions = [
            q for q in all_questions if str(q.get("Level", "")).strip() in allowed
        ]
        print(f"Filtered to level(s) {args.level}: {len(all_questions)} questions")

    if args.limit is not None:
        all_questions = all_questions[: args.limit]
        print(f"Limited to first {args.limit} questions")

    if not all_questions:
        print("No questions to process.", file=sys.stderr)
        sys.exit(0)

    # Build provider
    try:
        provider = _build_provider(args.provider, model=args.model)
    except EnvironmentError as e:
        print(f"Provider error: {e}", file=sys.stderr)
        sys.exit(1)

    # Prepare output path
    if args.output:
        out_path = Path(args.output)
    else:
        results_dir = Path(__file__).resolve().parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = results_dir / f"{ts}.jsonl"

    print(f"Provider: {args.provider}")
    print(f"Model:    {provider.model}")
    print(f"Workers:  {args.parallel}")
    print(f"Output:   {out_path}")
    print()

    # Run
    results: list[dict[str, Any]] = []
    cumulative_cost: float = 0.0
    if args.parallel == 1:
        for i, record in enumerate(all_questions, 1):
            if args.max_cost_usd is not None and cumulative_cost >= args.max_cost_usd:
                print(
                    f"  [cost cap] ${cumulative_cost:.4f} >= ${args.max_cost_usd}"
                    " — stopping"
                )
                break

            task_id = record.get("task_id", "unknown")
            level = record.get("Level", "?")
            print(
                f"[{i}/{len(all_questions)}] {task_id}"
                f"  (Level {level})  {record.get('Question', '')[:80]}..."
            )
            result = run_question(record, provider, args.steps, votes=args.votes)
            mark = "✓" if result.get("correct") else "✗"
            predicted = result.get("predicted", "")
            gt = result.get("ground_truth", "")
            elapsed = result.get("elapsed_seconds", 0)
            usage = result.get("usage", {})
            cost = usage.get("cost_usd", 0.0)
            cumulative_cost += cost
            cost_info = f", cost=${cost:.4f} (cumul=${cumulative_cost:.4f})" if cost > 0 else ""
            print(f"  {mark}  pred={predicted[:80]}  gt={gt[:80]}  ({elapsed}s{cost_info})")
            results.append(result)

            # Flush every question
            with open(out_path, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            fut_map = {
                pool.submit(
                    run_question, record, provider, args.steps, args.votes
                ): (i, record, record.get("task_id", "unknown"))
                for i, record in enumerate(all_questions, 1)
            }
            for future in as_completed(fut_map):
                i, record, task_id = fut_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "task_id": task_id,
                        "level": record.get("Level", "?"),
                        "question": record.get("Question", ""),
                        "predicted": f"ERROR: {exc}",
                        "ground_truth": record.get("Final answer", ""),
                        "correct": False,
                        "trajectory": [],
                        "provider": args.provider,
                        "model": provider.model,
                        "error": str(exc),
                    }
                mark = "✓" if result.get("correct") else "✗"
                cost = result.get("usage", {}).get("cost_usd", 0.0)
                cumulative_cost += cost
                cost_info = f"  cost=${cost:.4f} (cumul=${cumulative_cost:.4f})" if cost > 0 else ""
                print(
                    f"[{i}/{len(all_questions)}] {task_id}  {mark}"
                    f"  ({result.get('elapsed_seconds', '?')}s{cost_info})"
                )
                results.append(result)
                with open(out_path, "a") as f:
                    f.write(
                        json.dumps(result, ensure_ascii=False, default=str) + "\n"
                    )

    if args.max_cost_usd is not None and cumulative_cost > 0:
        print(
            f"\nTotal spend: ${cumulative_cost:.4f}"
            f"  (cap: ${args.max_cost_usd})"
        )

    # Summary
    print_summary(results)
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
