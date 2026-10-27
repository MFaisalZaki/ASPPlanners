"""Tests for property-directed reachability (aspplanners.plasp.pdr and
encodings/seq/pdr.lp), the planner's ``search='pdr'`` mode.

PDR answers one-step queries against a ground program whose boolean state at
time 0 is fixed by assumptions -- one program for all queries on a
propositional task, one per query on a numeric one, with the numeric values as
facts -- learns blocked cubes from unsatisfiable cores, and proves
unsolvability when a layer stops growing. Pinned here: (1) valid plans on the
classical and numeric fixtures and bundled domains; (2) the two things
deepening cannot do -- a proof of unsolvability, and a goal already true read
off the first query; (3) the query mechanics: a reason is a subset of the
state, never empty on a solvable task, numeric values ride along in cubes,
and the noop never reaches a plan; (4) the refusals and the option surface.
"""

import os

import clingo
import pytest

from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner
from unified_planning.engines import PlanGenerationResultStatus

import aspplanners  # noqa: F401
from aspplanners.plasp.pdr import NOOP, PDR, FreshBackend, IncrementalBackend
from aspplanners.plasp.planner import PLASPPlanner

from test_planner import (
    disjunction_problem, forall_problem, negative_precondition_problem,
    numeric_counter_problem, robot_line_problem, rover_recharge_problem,
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
    return PDR(planner._pdr_control, planner._pdr_static_fluents(), numeric="numeric" in planner.layers)


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


@pytest.mark.parametrize("make_problem", [
    lambda: numeric_counter_problem(3),
    lambda: numeric_counter_problem(12),
    rover_recharge_problem,
])
def test_numeric_fixtures_are_solved_with_validated_plans(make_problem):
    planner, plan = pdr_plan(make_problem())
    assert planner.status == PlanGenerationResultStatus.SOLVED_SATISFICING, planner.logs
    assert len(plan.actions) >= 1 and "groundings" in planner.logs[0]


@pytest.mark.parametrize("domain", ["alarm", "blocksworld", "gripper", "lift", "transport",
                                    "rover", "zenoflight", "tradepost", "briefcase"])
def test_bundled_classical_and_numeric_domains_are_solved(domain):
    for problem in sorted(f for f in os.listdir(os.path.join(PDDL_DIR, domain)) if f.startswith("problem")):
        planner, plan = pdr_plan(pddl(domain, problem))
        assert planner.status == PlanGenerationResultStatus.SOLVED_SATISFICING, (domain, problem, planner.logs)
        deepening = PLASPPlanner(pddl(domain, problem))
        shortest = deepening.plan(timeout=120)
        # PDR does not promise the shortest plan (rescheduled obligations keep
        # their path); it never beats deepening's, which is shortest.
        assert len(plan.actions) >= len(shortest.actions)
        assert planner.solved_horizon == len(plan.actions)


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


def unsolvable_counter(bound=None, step=1, target=5):
    """`tick` adds `step` to a counter (only below `bound`, when given); the
    goal wants the counter equal to `target`."""
    from unified_planning.shortcuts import Equals, Fluent, InstantaneousAction, IntType, LT, Problem
    counter = Fluent("counter", IntType())
    tick = InstantaneousAction("tick")
    if bound is not None:
        tick.add_precondition(LT(counter(), bound))
    tick.add_increase_effect(counter(), step)
    problem = Problem("unsolvable_counter")
    problem.add_fluent(counter, default_initial_value=0)
    problem.add_action(tick)
    problem.add_goal(Equals(counter(), target))
    return problem


def test_a_bounded_numeric_task_is_proven_unsolvable():
    """`tick` only below 4, goal needs exactly 5: the reachable values are
    finite, so the layers close and the numeric backend proves it."""
    planner, plan = pdr_plan(unsolvable_counter(bound=4, target=5))
    assert planner.status == PlanGenerationResultStatus.UNSOLVABLE_PROVEN, planner.logs


def test_an_unbounded_numeric_task_cannot_be_proven_unsolvable():
    """Ticking by 2 never hits 5, but every layer sees a new value, so the
    layers never close: PDR runs out of layers, which is the honest answer --
    and each layer terminates, because an obligation is rescheduled only
    while its path length plus its level stays within the layer bound."""
    planner, plan = pdr_plan(unsolvable_counter(step=2, target=5), max_horizon=6)
    assert planner.status == PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY, planner.logs
    assert "within 6 layers" in planner.logs[-1]


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
    assert isinstance(pdr.backend, IncrementalBackend)
    assert len(pdr.backend.variables) == 12   # 4 at + 2 at_robby + 2 free + 4 carry
    assert len(pdr.initial.bools) == 5        # 2 at, at_robby(a), 2 free
    assert pdr.initial.nums == frozenset()


def test_a_reason_is_a_proper_subset_of_the_state_and_a_sat_query_yields_a_successor():
    pdr = pdr_instance(pddl("gripper"))
    unsat = pdr.extend(pdr.initial, 1)       # no single action reaches the goal
    assert unsat[0] == "unsat"
    cube = unsat[1]
    assert 0 < len(cube) < len(pdr.backend.variables)
    assert cube.covers(pdr.initial)
    pdr._add_cube(cube, 1)
    sat = pdr.extend(pdr.initial, 2)         # some action leads outside the cube
    assert sat[0] == "sat" and sat[1] != NOOP
    assert not pdr._blocked(sat[2], 1)


def test_the_noop_lets_a_goal_state_pass_a_level_one_query():
    from aspplanners.plasp.pdr import State
    pdr = pdr_instance(pddl("gripper"))
    variables = pdr.backend.variables
    goal_bools = {v for v in variables if '"at"' in str(v) and 'room_b' in str(v)}
    goal_bools |= {v for v in pdr.initial.bools if '"at"' not in str(v)}
    result = pdr.extend(State(frozenset(goal_bools)), 1)
    assert result[0] == "sat" and result[1] == NOOP


def test_a_numeric_task_grounds_per_query_and_its_cubes_carry_the_numeric_values():
    pdr = pdr_instance(numeric_counter_problem(3))
    assert isinstance(pdr.backend, FreshBackend)
    assert len(pdr.initial.nums) == 1        # counter = 0
    unsat = pdr.extend(pdr.initial, 1)       # one tick does not reach 3
    assert unsat[0] == "unsat"
    assert unsat[1].nums == pdr.initial.nums  # the value rides along, never generalised
    before = pdr.backend.groundings
    pdr._add_cube(unsat[1], 1)
    sat = pdr.extend(pdr.initial, 2)
    assert sat[0] == "sat"
    assert sat[2].nums != pdr.initial.nums   # tick moved the counter
    assert pdr.backend.groundings == before + 1


def test_the_pdr_program_renames_the_initial_state_and_is_standalone(tmp_path):
    planner = PLASPPlanner(pddl("gripper"))
    text = planner.pdr_program()
    assert "pdrInitialState(" in text and "\ninitialState(" not in text
    numeric = PLASPPlanner(numeric_counter_problem()).pdr_program()
    assert "The first query's numeric state" in numeric and "holds(variable" in numeric
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
    assert len(result.plan.actions) >= 5    # validated; not promised shortest


def test_temporal_tasks_are_refused():
    with pytest.raises(ValueError, match="temporal task"):
        PLASPPlanner(pddl("matchcellar")).plan(search="pdr")


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
