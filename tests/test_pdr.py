"""Tests for property-directed reachability (aspplanners.plasp.pdr and
encodings/seq/pdr.lp), the planner's ``search='pdr'`` mode.

PDR answers one-step queries against a single ground program whose state at
time 0 is fixed by assumptions, learns blocked cubes from unsatisfiable cores,
and proves unsolvability when a layer stops growing. Classical tasks only.
Pinned here: (1) valid plans on the classical fixtures and bundled domains;
(2) the two things deepening cannot do -- a proof of unsolvability, and a goal
already true read off the first query; (3) the query mechanics: a reason is a
subset of the state, never empty on a solvable task, and the noop never
reaches a plan; (4) the refusals and the option surface.
"""

import os

import clingo
import pytest

from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner
from unified_planning.engines import PlanGenerationResultStatus

import aspplanners  # noqa: F401
from aspplanners.plasp.pdr import PDR
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
    return PDR(planner._pdr_control(), planner._pdr_static_fluents())


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


def test_static_fluents_are_pinned_and_never_in_a_reason():
    """transport's `link` is static: the encoder folds it into the action
    signatures, so its initial facts keep their name and PDR pins it."""
    pdr = pdr_instance(pddl("transport"))
    assert pdr.static and all('"link"' in str(v) for v in pdr.static)
    assert not any(v in pdr.static for v in pdr.free)
    unsat = pdr.extend(pdr.initial, 1)
    assert unsat[0] == "unsat" and not any(v in pdr.static for v, _ in unsat[1])


def test_a_reason_is_a_proper_subset_of_the_state_and_a_sat_query_yields_a_successor():
    pdr = pdr_instance(pddl("gripper"))
    unsat = pdr.extend(pdr.initial, 1)       # no single action reaches the goal
    assert unsat[0] == "unsat"
    cube = unsat[1]
    assert 0 < len(cube) < len(pdr.variables)
    assert all((v in pdr.initial) == positive for v, positive in cube)
    pdr._add_cube(cube, 1)
    sat = pdr.extend(pdr.initial, 2)         # some action leads outside the cube
    assert sat[0] == "sat" and sat[1] is not None
    assert not pdr._blocked(sat[2], 1)


def test_the_noop_lets_a_goal_state_pass_a_level_one_query():
    pdr = pdr_instance(pddl("gripper"))
    goal_state = frozenset(v for v in pdr.variables if '"at"' in str(v) and 'room_b' in str(v))
    result = pdr.extend(goal_state | frozenset(v for v in pdr.initial if '"at"' not in str(v)), 1)
    assert result[0] == "sat" and result[1] is None


def test_the_pdr_program_renames_the_initial_state_and_is_standalone(tmp_path):
    planner = PLASPPlanner(pddl("gripper"))
    text = planner.pdr_program()
    assert "pdrInitialState(" in text and "\ninitialState(" not in text
    transport = PLASPPlanner(pddl("transport")).pdr_program()
    assert '\ninitialState(variable(("link"' in transport     # statics keep their facts
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


@pytest.mark.parametrize("make_problem, reason", [
    (numeric_counter_problem, "classical tasks only"),
    (lambda: pddl("rover"), "classical tasks only"),
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


# ---------------------------------------------------------------------------
# pdr_first: PDR on the boolean abstraction before deepening
# ---------------------------------------------------------------------------

def fuel_gripper(balls, fuel):
    """Gripper with an `energy` counter every move consumes; `fuel` of it."""
    from unified_planning.shortcuts import (
        BoolType, Fluent, GE, InstantaneousAction, IntType, Object, Problem, UserType)
    Room, Ball, Gripper = UserType("room"), UserType("ball"), UserType("gripper")
    at_robby, at = Fluent("at_robby", BoolType(), r=Room), Fluent("at", BoolType(), b=Ball, r=Room)
    free, carry = Fluent("free", BoolType(), g=Gripper), Fluent("carry", BoolType(), b=Ball, g=Gripper)
    energy = Fluent("energy", IntType())
    move = InstantaneousAction("move", f=Room, t=Room)
    f, t = move.parameters
    move.add_precondition(at_robby(f)); move.add_precondition(GE(energy(), 1))
    move.add_effect(at_robby(f), False); move.add_effect(at_robby(t), True); move.add_decrease_effect(energy(), 1)
    pick = InstantaneousAction("pick", b=Ball, r=Room, g=Gripper)
    b, r, g = pick.parameters
    for pre in (at(b, r), at_robby(r), free(g)):
        pick.add_precondition(pre)
    pick.add_effect(carry(b, g), True); pick.add_effect(at(b, r), False); pick.add_effect(free(g), False)
    drop = InstantaneousAction("drop", b=Ball, r=Room, g=Gripper)
    b, r, g = drop.parameters
    for pre in (carry(b, g), at_robby(r)):
        drop.add_precondition(pre)
    drop.add_effect(at(b, r), True); drop.add_effect(carry(b, g), False); drop.add_effect(free(g), True)
    problem = Problem("fuel_gripper")
    for fl in (at_robby, at, free, carry):
        problem.add_fluent(fl, default_initial_value=False)
    problem.add_fluent(energy, default_initial_value=fuel)
    rooms = [Object("a", Room), Object("b", Room)]
    ball_objs = [Object(f"ball{i}", Ball) for i in range(balls)]
    grippers = [Object("left", Gripper), Object("right", Gripper)]
    problem.add_objects(rooms + ball_objs + grippers)
    problem.add_actions([move, pick, drop])
    problem.set_initial_value(at_robby(rooms[0]), True)
    for ball in ball_objs:
        problem.set_initial_value(at(ball, rooms[0]), True)
        problem.add_goal(at(ball, rooms[1]))
    for gr in grippers:
        problem.set_initial_value(free(gr), True)
    return problem


def test_pdr_first_returns_pdrs_plan_on_a_classical_task():
    planner = PLASPPlanner(pddl("gripper"))
    plan = planner.plan(pdr_first=30)
    assert planner.status == PlanGenerationResultStatus.SOLVED_SATISFICING
    assert "PDR found a plan" in planner.logs[0] and len(plan.actions) >= 5


def test_pdr_first_takes_a_valid_abstract_plan_on_a_numeric_task_with_slack():
    planner = PLASPPlanner(fuel_gripper(2, 20))
    plan = planner.plan(pdr_first=30)
    assert planner.status == PlanGenerationResultStatus.SOLVED_SATISFICING
    assert "on the boolean abstraction, valid for the task" in planner.logs[0]
    assert len(plan.actions) >= 5


def test_pdr_first_falls_back_to_deepening_when_the_abstract_plan_is_not_real():
    """Fuel for exactly one move: the abstraction's plan, unaware of fuel, is
    rejected by the validator and the deepening finds the real one."""
    planner = PLASPPlanner(fuel_gripper(2, 1))
    plan = planner.plan(pdr_first=30)
    assert planner.status == PlanGenerationResultStatus.SOLVED_SATISFICING
    assert "not valid for the task" in planner.logs[0]
    assert len(plan.actions) == 5 == planner.solved_horizon   # pick, pick, move, drop, drop


def test_pdr_first_carries_an_abstract_proof_of_unsolvability_over():
    """The boolean abstraction of an unsolvable robot line with a numeric
    fluent added is still unsolvable, and that proof holds for the task."""
    from unified_planning.shortcuts import Fluent, IntType
    problem = robot_line_problem(connected_line=False)
    problem.add_fluent(Fluent("odometer", IntType()), default_initial_value=0)
    problem.action("move").add_increase_effect(problem.fluent("odometer")(), 1)
    planner = PLASPPlanner(problem)
    assert "numeric" in planner.layers
    planner.plan(pdr_first=30)
    assert planner.status == PlanGenerationResultStatus.UNSOLVABLE_PROVEN
    assert "proved the task unsolvable" in planner.logs[0]


def test_pdr_first_is_skipped_on_a_temporal_task_and_the_deepening_still_runs():
    planner = PLASPPlanner(pddl("matchcellar"))
    planner.plan(pdr_first=5)
    assert planner.status == PlanGenerationResultStatus.SOLVED_SATISFICING
    assert "pdr_first skipped" in planner.logs[0]


def test_pdr_first_through_the_up_engine():
    with OneshotPlanner(name="PLASPPlanner", params={"pdr_first": 30}) as planner:
        result = planner.solve(fuel_gripper(2, 20))
    assert result.status == PlanGenerationResultStatus.SOLVED_SATISFICING


# ---------------------------------------------------------------------------
# PDR on the lifted family
# ---------------------------------------------------------------------------

def test_pdr_runs_on_the_lifted_family_as_on_seq():
    for domain in ("gripper", "transport", "blocksworld"):
        seq = PLASPPlanner(pddl(domain))
        lifted = PLASPPlanner(pddl(domain), "lifted")
        seq_plan, lifted_plan = seq.plan(search="pdr", timeout=120), lifted.plan(search="pdr", timeout=120)
        assert seq.status == lifted.status == PlanGenerationResultStatus.SOLVED_SATISFICING, domain
        assert len(lifted_plan.actions) == len(seq_plan.actions), domain
        assert all(a.action.name != "pdrNoop" for a in lifted_plan.actions)


def test_pdr_proves_unsolvability_on_the_lifted_family():
    planner = PLASPPlanner(robot_line_problem(connected_line=False), "lifted")
    planner.plan(search="pdr")
    assert planner.status == PlanGenerationResultStatus.UNSOLVABLE_PROVEN


def test_pdr_refuses_a_numeric_task_on_the_lifted_family_too():
    with pytest.raises(ValueError, match="classical tasks only"):
        PLASPPlanner(numeric_counter_problem(), "lifted").plan(search="pdr")


def test_pdr_first_on_the_lifted_family_leaves_the_numeric_step_rules_out():
    """The abstraction drops the numeric layer and the lifted encoder's
    numeric step rules alike: the slack fuel gripper yields a valid abstract
    plan, the tight one falls back to the (numeric) deepening."""
    slack = PLASPPlanner(fuel_gripper(2, 20), "lifted")
    plan = slack.plan(pdr_first=30)
    assert slack.status == PlanGenerationResultStatus.SOLVED_SATISFICING
    assert "on the boolean abstraction, valid for the task" in slack.logs[0]
    tight = PLASPPlanner(fuel_gripper(2, 1), "lifted")
    plan = tight.plan(pdr_first=30)
    assert tight.status == PlanGenerationResultStatus.SOLVED_SATISFICING
    assert "not valid for the task" in tight.logs[0] and len(plan.actions) == 5
