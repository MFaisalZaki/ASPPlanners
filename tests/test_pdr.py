"""Tests for property-directed reachability (aspplanners.plasp.pdr and
encodings/seq/pdr.lp), the planner's ``search='pdr'`` mode.

PDR answers one-step queries against a single ground program whose state at
time 0 is fixed by assumptions, learns blocked cubes from unsatisfiable cores,
and proves unsolvability when a layer stops growing. Pinned here: (1) valid
plans on the classical fixtures and bundled domains; (2) the two things
deepening cannot do -- a proof of unsolvability, and a goal already true read
off the first query; (3) the query mechanics: a reason is a subset of the
state, never empty on a solvable task, and the noop never reaches a plan;
(4) the refusals and the option surface.
"""

import os

import clingo
import pytest

from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner
from unified_planning.engines import PlanGenerationResultStatus

import aspplanners  # noqa: F401
from aspplanners.plasp.pdr import NOOP, PDR
from aspplanners.plasp.planner import PLASPPlanner

from test_planner import (
    disjunction_problem, forall_problem, negative_precondition_problem,
    numeric_counter_problem, robot_line_problem,
)

PDDL_DIR = os.path.join(os.path.dirname(__file__), "pddl")


def pddl(domain, problem="problem.pddl"):
    return PDDLReader().parse_problem(os.path.join(PDDL_DIR, domain, "domain.pddl"),
                                      os.path.join(PDDL_DIR, domain, problem))


def pdr_plan(problem, **kwargs):
    planner = PLASPPlanner(problem)
    plan = planner.plan(search="pdr", timeout=120, **kwargs)
    return planner, plan


def pdr_instance(problem):
    planner = PLASPPlanner(problem)
    ctl = clingo.Control(["-n", "1"])
    for path in planner.encoding_paths:
        ctl.load(path)
    ctl.load(planner._pdr_path())
    ctl.add("base", [], planner._pdr_facts())
    return PDR(ctl)


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make_problem", [
    robot_line_problem,
    negative_precondition_problem,
    lambda: forall_problem()[0],
    lambda: forall_problem(universal_effect=True)[0],
    disjunction_problem,
])
def test_fixtures_are_solved_with_validated_plans(make_problem):
    planner, plan = pdr_plan(make_problem())
    assert planner.status == PlanGenerationResultStatus.SOLVED_SATISFICING, planner.logs
    assert len(plan.actions) == planner.solved_horizon >= 1
    assert "PDR: plan" in planner.logs[0]


@pytest.mark.parametrize("domain", ["alarm", "blocksworld", "gripper", "lift", "transport"])
def test_bundled_classical_domains_are_solved(domain):
    for problem in sorted(f for f in os.listdir(os.path.join(PDDL_DIR, domain)) if f.startswith("problem")):
        planner, plan = pdr_plan(pddl(domain, problem))
        assert planner.status == PlanGenerationResultStatus.SOLVED_SATISFICING, (domain, problem, planner.logs)
        deepening = PLASPPlanner(pddl(domain, problem))
        shortest = deepening.plan(timeout=120)
        # PDR does not promise the shortest plan, but on these it finds one.
        assert len(plan.actions) >= len(shortest.actions)


def test_a_plan_never_contains_the_noop():
    planner, plan = pdr_plan(pddl("gripper"))
    assert all(a.action.name != "pdrNoop" for a in plan.actions)
    assert {a.action.name for a in plan.actions} <= {"pick", "drop", "move"}


# ---------------------------------------------------------------------------
# What deepening cannot do
# ---------------------------------------------------------------------------

def test_an_unsolvable_task_is_proven_unsolvable():
    """Deepening can only run out of horizon; PDR's layers stop growing."""
    planner, plan = pdr_plan(robot_line_problem(connected_line=False))
    assert planner.status == PlanGenerationResultStatus.UNSOLVABLE_PROVEN
    assert len(plan.actions) == 0 and "stopped growing" in planner.logs[-1]
    deepening = PLASPPlanner(robot_line_problem(connected_line=False))
    deepening.plan(max_horizon=20)
    assert deepening.status == PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY


def test_a_goal_that_already_holds_is_the_empty_plan():
    problem = robot_line_problem()
    problem.clear_goals()
    problem.add_goal(problem.fluent("robot_at")(problem.object("loc_0")))
    planner, plan = pdr_plan(problem)
    assert planner.status == PlanGenerationResultStatus.SOLVED_SATISFICING
    assert len(plan.actions) == 0


def test_the_layer_bound_is_respected():
    planner, plan = pdr_plan(robot_line_problem(n_locations=6), max_horizon=2)
    assert planner.status == PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY
    assert "within 2 layers" in planner.logs[-1]


# ---------------------------------------------------------------------------
# Query mechanics
# ---------------------------------------------------------------------------

def test_the_initial_state_and_variables_are_read_off_the_program():
    pdr = pdr_instance(pddl("gripper"))
    assert len(pdr.variables) == 12          # 4 at + 2 at_robby + 2 free + 4 carry
    assert len(pdr.initial) == 5             # 2 at, at_robby(a), 2 free
    assert all(v in set(pdr.variables) for v in pdr.initial)


def test_a_reason_is_a_proper_subset_of_the_state_and_a_sat_query_yields_a_successor():
    pdr = pdr_instance(pddl("gripper"))
    unsat = pdr.extend(pdr.initial, 1)       # no single action reaches the goal
    assert unsat[0] == "unsat"
    cube = unsat[1]
    assert 0 < len(cube) < len(pdr.variables)
    assert all((v in pdr.initial) == positive for v, positive in cube)
    pdr._add_cube(cube, 1)
    sat = pdr.extend(pdr.initial, 2)         # some action leads outside the cube
    assert sat[0] == "sat" and sat[1] != NOOP
    assert not pdr._blocked(sat[2], 1)


def test_the_noop_lets_a_goal_state_pass_a_level_one_query():
    pdr = pdr_instance(pddl("gripper"))
    goal_state = frozenset(v for v in pdr.variables if '"at"' in str(v) and 'room_b' in str(v))
    result = pdr.extend(goal_state | frozenset(v for v in pdr.initial if '"at"' not in str(v)), 1)
    assert result[0] == "sat" and result[1] == NOOP


def test_the_pdr_program_renames_the_initial_state_and_is_standalone(tmp_path):
    planner = PLASPPlanner(pddl("gripper"))
    text = planner.pdr_program()
    assert "pdrInitialState(" in text and "\ninitialState(" not in text
    path = tmp_path / "pdr.lp"
    path.write_text(text)
    ctl = clingo.Control(["-n", "1"])
    ctl.load(str(path))
    ctl.ground([("base", []), ("step", [clingo.Number(1)]), ("check", [clingo.Number(1)])])
    assert ctl.solve().satisfiable


# ---------------------------------------------------------------------------
# Options and refusals
# ---------------------------------------------------------------------------

def test_through_the_up_engine():
    with OneshotPlanner(name="PLASPPlanner", params={"search": "pdr"}) as planner:
        result = planner.solve(pddl("gripper"))
    assert result.status == PlanGenerationResultStatus.SOLVED_SATISFICING
    # Validated by the planner; PDR does not promise the 5-step optimum.
    assert len(result.plan.actions) >= 5


@pytest.mark.parametrize("make_problem, reason", [
    (numeric_counter_problem, "numeric task"),
    (lambda: pddl("matchcellar"), "temporal task"),
])
def test_numeric_and_temporal_tasks_are_refused(make_problem, reason):
    with pytest.raises(ValueError, match=reason):
        PLASPPlanner(make_problem()).plan(search="pdr")


def test_bad_options_are_rejected():
    with pytest.raises(ValueError, match="no fixed horizon"):
        PLASPPlanner(robot_line_problem()).plan(search="pdr", horizon=3)
    with pytest.raises(ValueError, match="search must be"):
        PLASPPlanner(robot_line_problem()).plan(search="dfs")


def test_deepening_is_unchanged():
    planner = PLASPPlanner(pddl("gripper"))
    plan = planner.plan()
    assert planner.status == PlanGenerationResultStatus.SOLVED_SATISFICING
    assert planner.solved_horizon == len(plan.actions) == 5
    assert not planner.logs
