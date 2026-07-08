"""Tests for the planner module."""

from __future__ import annotations

from unittest.mock import MagicMock

from agent.planner import (
    Plan,
    PlanStep,
    build_plan,
    needs_planning,
    parse_plan_from_json,
)
from agent.providers.base import LLMProvider, ProviderResponse


class TestNeedsPlanning:
    def test_chain_question_detected(self):
        assert needs_planning(
            "Do the following in order: (1) Find the year... (2) Compute..."
        )

    def test_report_all_detected(self):
        assert needs_planning(
            "Report all of: the three years found, the sum, the average"
        )

    def test_find_find_detected(self):
        assert needs_planning(
            "Find the diameter of Earth. Find the diameter of Mars."
        )

    def test_simple_question_skipped(self):
        assert not needs_planning("What is the capital of France?")

    def test_fibonacci_question_skipped(self):
        assert not needs_planning(
            "Write a function to compute the nth Fibonacci number. "
            "What is the 20th Fibonacci number?"
        )


class TestParsePlanFromJson:
    def test_direct_json(self):
        text = '[{"step": 1, "action": "search", "description": "Find year", "expected_output": "1896"}]'
        data = parse_plan_from_json(text)
        assert len(data) == 1
        assert data[0]["step"] == 1
        assert data[0]["action"] == "search"

    def test_json_in_code_block(self):
        text = "Some text\n```json\n[{\"step\": 1, \"action\": \"code_exec\", \"description\": \"Compute\", \"expected_output\": \"5788\"}]\n```"
        data = parse_plan_from_json(text)
        assert len(data) == 1
        assert data[0]["action"] == "code_exec"

    def test_json_in_plain_brackets(self):
        text = "Here is the plan: [{\"step\": 1, \"action\": \"search\", \"description\": \"Find X\", \"expected_output\": \"100\"}]"
        data = parse_plan_from_json(text)
        assert len(data) == 1
        assert data[0]["step"] == 1

    def test_empty_response(self):
        assert parse_plan_from_json("") == []

    def test_malformed_json(self):
        assert parse_plan_from_json("{bad json}") == []


class TestBuildPlan:
    def test_build_from_valid_json(self):
        text = '[{"step": 1, "action": "search", "description": "Find X", "expected_output": "100"}, {"step": 2, "action": "code_exec", "description": "Compute Y", "expected_output": "200"}]'
        plan = build_plan(text)
        assert len(plan.steps) == 2
        assert plan.steps[0].step == 1
        assert plan.steps[0].action == "search"
        assert plan.steps[1].action == "code_exec"

    def test_all_complete_initially_false(self):
        text = '[{"step": 1, "action": "search", "description": "Find X", "expected_output": "100"}]'
        plan = build_plan(text)
        assert not plan.all_complete()
        assert len(plan.incomplete_steps()) == 1

    def test_empty_response_yields_empty_plan(self):
        plan = build_plan("")
        assert len(plan.steps) == 0


class TestPlanProgress:
    def test_mark_complete(self):
        plan = Plan(steps=[PlanStep(step=1, action="search", description="Find X", expected_output="100")])
        plan.mark_complete(1, "100")
        assert plan.steps[0].completed
        assert plan.steps[0].result == "100"
        assert plan.all_complete()

    def test_mark_complete_by_output(self):
        plan = Plan(steps=[
            PlanStep(step=1, action="search", description="Find X", expected_output="1896"),
            PlanStep(step=2, action="search", description="Find Y", expected_output="1989"),
        ])
        marked = plan.mark_complete_by_output("The year is 1896")
        assert len(marked) == 1
        assert plan.steps[0].completed
        assert not plan.steps[1].completed

    def test_mark_complete_by_output_multi(self):
        plan = Plan(steps=[
            PlanStep(step=1, action="search", description="Find X", expected_output="1896"),
            PlanStep(step=2, action="search", description="Find Y", expected_output="1989"),
        ])
        marked = plan.mark_complete_by_output("1896 and 1989")
        assert len(marked) == 2
        assert plan.all_complete()

    def test_progress_string_shows_status(self):
        plan = Plan(steps=[PlanStep(step=1, action="search", description="Find X", expected_output="100")])
        ps = plan.to_progress_string()
        assert "[ ]" in ps  # not started
        assert "Step 1" in ps

        plan.mark_complete(1, "100")
        ps = plan.to_progress_string()
        assert "[✓]" in ps
        assert "100" in ps

    def test_plan_string_shows_steps(self):
        plan = Plan(steps=[PlanStep(step=1, action="search", description="Find X", expected_output="100")])
        ps = plan.to_plan_string()
        assert "Step 1" in ps
        assert "Find X" in ps


class TestPlanIntegration:
    def test_plan_generated_in_run_for_multi_step(self):
        """When needs_planning is True, the agent should generate a plan
        and inject it into messages."""
        provider = MagicMock(spec=LLMProvider)
        provider.model = "test-model"

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: planning
                return ProviderResponse(
                    text='[{"step": 1, "action": "search", "description": "Find year", "expected_output": "1896"}]',
                    tool_calls=[],
                    stop_reason="end_turn",
                )
            # Subsequent calls: agent loop
            return ProviderResponse(
                text="FINAL ANSWER: 1896",
                tool_calls=[],
                stop_reason="end_turn",
            )

        provider.call.side_effect = side_effect

        from agent.core import ResearchAgent

        agent = ResearchAgent(
            question=(
                "Do the following in order: (1) Find the year the first "
                "modern Olympic Games were held."
            ),
            provider=provider,
        )
        result = agent.run()
        assert result["answer"] is not None
        # Planning call happened
        assert call_count[0] >= 2
        # Plan was created
        assert agent._plan is not None
        assert len(agent._plan.steps) == 1

    def test_plan_skipped_for_simple_question(self):
        """When needs_planning is False, no planning call should happen."""
        provider = MagicMock(spec=LLMProvider)
        provider.model = "test-model"
        provider.call.return_value = ProviderResponse(
            text="FINAL ANSWER: Paris",
            tool_calls=[],
            stop_reason="end_turn",
        )

        from agent.core import ResearchAgent

        agent = ResearchAgent(
            question="What is the capital of France?",
            provider=provider,
        )
        result = agent.run()
        assert result["answer"] == "Paris"
        # No plan was created
        assert agent._plan is None
