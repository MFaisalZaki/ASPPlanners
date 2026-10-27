"""Tests for the goal-support decision heuristic (aspplanners.plasp.heuristics).

Rintanen's planning-specific branching rule, as a clingo `decide` propagator
over the seq encoding. Pinned here: (1) it changes which model is found first
and never which horizon is satisfiable, so plans and horizons equal the
default's on every fixture and bundled domain; (2) it actually decides, and
yields to the solver when there is nothing to work back from (a numeric
goal); (3) its option surface and refusals.
"""

import os

import pytest

from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner
from unified_planning.engines import PlanGenerationResultStatus

import aspplanners  # noqa: F401
from aspplanners.plasp.heuristics import HEURISTICS, GoalSupportHeuristic
from aspplanners.plasp.planner import PLASPPlanner

from test_planner import (
    disjunction_problem, forall_problem, negative_precondition_problem,
    numeric_counter_problem, robot_line_problem, rover_recharge_problem,
)

PDDL_DIR = os.path.join(os.path.dirname(__file__), "pddl")


def pddl(domain, problem="problem.pddl"):
    return PDDLReader().parse_problem(os.path.join(PDDL_DIR, domain, "domain.pddl"),
                                      os.path.join(PDDL_DIR, domain, problem))


def solve(problem, **options):
    planner = PLASPPlanner(problem, **options)
    plan = planner.plan(timeout=120)
    assert planner.status == PlanGenerationResultStatus.SOLVED_SATISFICING, planner.logs
    return planner, plan


@pytest.mark.parametrize("make_problem", [
    robot_line_problem, negative_precondition_problem, disjunction_problem,
    lambda: forall_problem()[0], lambda: forall_problem(universal_effect=True)[0],
    lambda: numeric_counter_problem(4), rover_recharge_problem,
])
def test_the_same_horizon_and_a_valid_plan_on_the_fixtures(make_problem):
    default, default_plan = solve(make_problem())
    guided, guided_plan = solve(make_problem(), heuristic="goal-support")
    assert guided.solved_horizon == default.solved_horizon
    assert len(guided_plan.actions) == len(default_plan.actions)


@pytest.mark.parametrize("domain", ["alarm", "blocksworld", "gripper", "lift", "transport",
                                    "rover", "zenoflight", "tradepost", "briefcase"])
def test_the_same_horizon_and_a_valid_plan_on_the_bundled_domains(domain):
    for problem in sorted(f for f in os.listdir(os.path.join(PDDL_DIR, domain)) if f.startswith("problem")):
        default, _ = solve(pddl(domain, problem))
        guided, _ = solve(pddl(domain, problem), heuristic="goal-support")
        assert guided.solved_horizon == default.solved_horizon, (domain, problem)
        assert "goal-support heuristic" in guided.logs[-1]


def test_the_heuristic_decides_when_there_is_a_boolean_goal_and_yields_otherwise():
    guided, _ = solve(pddl("gripper"), heuristic="goal-support", heuristic_depth=None)
    assert guided._heuristic.decisions > 0
    assert guided._heuristic.decisions == guided._heuristic.calls   # every decision was its own
    numeric_goal, _ = solve(pddl("tradepost"), heuristic="goal-support", heuristic_depth=None)
    assert numeric_goal._heuristic.decisions == 0 and numeric_goal._heuristic.calls > 0


def test_the_depth_bounds_where_the_heuristic_speaks():
    """At depth 0 it never decides (the root level is below the first decision);
    unbounded it decides every time it is asked and finds something to do."""
    silent, _ = solve(pddl("gripper"), heuristic="goal-support", heuristic_depth=0)
    assert silent._heuristic.decisions == 0 and silent._heuristic.calls > 0
    shallow, _ = solve(pddl("gripper"), heuristic="goal-support", heuristic_depth=2)
    always, _ = solve(pddl("gripper"), heuristic="goal-support", heuristic_depth=None)
    assert 0 < shallow._heuristic.decisions <= always._heuristic.decisions
    assert shallow.solved_horizon == always.solved_horizon == 5


def test_the_heuristic_helps_find_a_model_at_a_satisfiable_horizon():
    """The one place it is for: a fixed horizon that has a plan. Pinned loosely
    (no slower than default, and deciding) rather than by the measured 3x."""
    import time
    problem = pddl("gripper")
    start = time.time(); PLASPPlanner(problem).plan(horizon=5); default = time.time() - start
    guided = PLASPPlanner(problem, heuristic="goal-support")
    start = time.time(); plan = guided.plan(horizon=5); with_heuristic = time.time() - start
    assert len(plan.actions) == 5 and guided._heuristic.decisions > 0
    assert with_heuristic < max(2 * default, 1.0)


def test_a_fixed_horizon_solve_and_an_unsolvable_task_behave_as_by_default():
    planner = PLASPPlanner(robot_line_problem(), heuristic="goal-support")
    assert len(planner.plan(horizon=3).actions) == 3
    blocked = PLASPPlanner(robot_line_problem(connected_line=False), heuristic="goal-support")
    blocked.plan(max_horizon=5)
    assert blocked.status == PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY


def test_the_heuristic_reads_goals_achievers_and_conditions_off_the_facts():
    planner = PLASPPlanner(pddl("gripper"), heuristic="goal-support")
    planner.plan(horizon=2)   # grounds and runs init
    h = planner._heuristic
    assert len(h.goals) == 2                       # both balls at room_b
    assert all(any('"drop"' in str(a) for a, _ in h.achievers[g]) for g in h.goals)
    drop = next(a for a, _ in h.achievers[h.goals[0]])
    assert len(h.conditions[drop]) == 2            # carry and at_robby
    assert any(t == 2 for (_a, t) in h.occurs) and any(t == 2 for (_v, _val, t) in h.holds)


def test_through_the_up_engine():
    with OneshotPlanner(name="PLASPPlanner", params={"heuristic": "goal-support"}) as planner:
        result = planner.solve(pddl("gripper"))
    assert result.status == PlanGenerationResultStatus.SOLVED_SATISFICING
    assert len(result.plan.actions) == 5


def test_bad_options_are_rejected():
    assert HEURISTICS == ("goal-support",)
    with pytest.raises(ValueError, match="heuristic must be"):
        PLASPPlanner(robot_line_problem(), heuristic="greedy")
    with pytest.raises(ValueError, match="available on the 'seq' encoding"):
        PLASPPlanner(pddl("gripper"), "lifted", heuristic="goal-support")


def test_the_default_search_is_unchanged():
    planner, plan = solve(pddl("gripper"))
    assert planner.heuristic is None and planner._heuristic is None
    assert not planner.logs and planner.solved_horizon == len(plan.actions) == 5
