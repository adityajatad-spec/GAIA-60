from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

PLANNER_SYSTEM_PROMPT = """\
You are a precise task planner. Given a complex question, decompose it into a numbered sequence of atomic steps.

Each step must be:
- A single tool action (search, browse, code_exec, read_attached_file)
- Self-contained (does not depend on steps that come after it)
- Have a clear expected output

IMPORTANT: Output ONLY a valid JSON array. No explanation, no markdown formatting.

Example:
[
  {"step": 1, "action": "search", "description": "Find the year the first modern Olympic Games were held", "expected_output": "1896"},
  {"step": 2, "action": "search", "description": "Find the year the World Wide Web was invented", "expected_output": "1989"},
  {"step": 3, "action": "code_exec", "description": "Compute sum of the three years", "expected_output": "5788"},
  {"step": 4, "action": "code_exec", "description": "Check if the sum is even or odd", "expected_output": "even"}
]
"""


@dataclass
class PlanStep:
    step: int
    action: str
    description: str
    expected_output: str
    completed: bool = False
    result: str = ""


@dataclass
class Plan:
    steps: list[PlanStep] = field(default_factory=list)

    def all_complete(self) -> bool:
        return all(s.completed for s in self.steps)

    def incomplete_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if not s.completed]

    def mark_complete(self, step_num: int, result: str) -> None:
        for s in self.steps:
            if s.step == step_num:
                s.completed = True
                s.result = result[:200]
                return

    def mark_complete_by_output(self, text: str) -> list[str]:
        marked: list[str] = []
        for s in self.steps:
            if s.completed:
                continue
            exp = s.expected_output.strip().lower()
            if exp and exp in text.lower():
                s.completed = True
                s.result = exp
                marked.append(s.description)
        return marked

    def to_progress_string(self) -> str:
        if not self.steps:
            return ""
        lines = ["[Plan Progress]"]
        for s in self.steps:
            status = "✓" if s.completed else " "
            lines.append(f"[{status}] Step {s.step}: {s.description}")
            if s.result:
                lines.append(f"       \u2192 {s.result}")
        return "\n".join(lines)

    def to_plan_string(self) -> str:
        lines = ["[Plan]"]
        for s in self.steps:
            lines.append(f"Step {s.step}: {s.description}")
            lines.append(f"  Action: {s.action}")
            lines.append(f"  Expected: {s.expected_output}")
        return "\n".join(lines)

    def to_prompt_string(self) -> str:
        """Full plan + progress for injection into messages."""
        parts = [self.to_plan_string()]
        parts.append("")
        parts.append(self.to_progress_string())
        return "\n".join(parts)


def parse_plan_from_json(text: str) -> list[dict]:
    """Parse JSON plan from planner response, with text-fallback parsing."""
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    m = re.search(r"(\[[\s\S]*?\])", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    return []


def build_plan(planner_response: str) -> Plan:
    """Build a Plan from a planner provider response."""
    steps_data = parse_plan_from_json(planner_response)
    plan = Plan()
    for sd in steps_data:
        step_num = int(sd.get("step", len(plan.steps) + 1))
        plan.steps.append(
            PlanStep(
                step=step_num,
                action=sd.get("action", "search"),
                description=sd.get("description", ""),
                expected_output=sd.get("expected_output", ""),
            )
        )
    return plan


_MULTI_STEP_RE = re.compile(
    r"(?:Do the following in order|"
    r"in order.*then|"
    r"find.*find|"
    r"find.*compute|"
    r"compute.*determine|"
    r"Report all|"
    r"Do the following)",
    re.IGNORECASE,
)


def needs_planning(question: str) -> bool:
    """Heuristic: only generate a plan for multi-step questions."""
    return bool(_MULTI_STEP_RE.search(question))
