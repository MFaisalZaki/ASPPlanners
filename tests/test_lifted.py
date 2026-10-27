"""Tests for the `lifted` encoding family (aspplanners.plasp.lifted).

A step chooses a schema and one object per parameter position; every
precondition and effect, boolean or numeric, is a step rule joining only the
arguments its literal mentions, so no ground action is ever named. Pinned
here: (1) the same plans and horizons as seq on every classical and numeric
task the fixtures and bundled domains offer -- the two encodings are the same
semantics -- with validated plans; (2) the shape of the emitted rules, literal
for literal; (3) that the step program is smaller than seq's where action
arity exceeds predicate arity; (4) the refusals, with their reasons.
"""

import os

import clingo
import pytest

from unified_planning.io import PDDLReader
from unified_planning.shortcuts import (
    BoolType, Equals, Fluent, InstantaneousAction, Not, Object, OneshotPlanner, Problem,
    UserType, Variable, Forall,
)
from unified_planning.engines import PlanGenerationResultStatus

import aspplanners  # noqa: F401
from aspplanners.plasp.layers import LIFTED, SEQ, fact_predicates
from aspplanners.plasp.planner import PLASPPlanner

from test_planner import (
    disjunction_problem, forall_problem, negative_precondition_problem,
    numeric_counter_problem, robot_line_problem, rover_recharge_problem,
)

PDDL_DIR = os.path.join(os.path.dirname(__file__), "pddl")


def pddl(domain, problem="problem.pddl"):
    return PDDLReader().parse_problem(os.path.join(PDDL_DIR, domain, "domain.pddl"),
                                      os.path.join(PDDL_DIR, domain, problem))


def solve(problem, encoding):
    planner = PLASPPlanner(problem, encoding)
    plan = planner.plan(timeout=120)
    assert planner.status == PlanGenerationResultStatus.SOLVED_SATISFICING, planner.logs
    return planner, plan


def assert_same_as_seq(problem):
    seq, seq_plan = solve(problem, "seq")
    lifted, lifted_plan = solve(problem, "lifted")
    assert lifted.solved_horizon == seq.solved_horizon
    assert len(lifted_plan.actions) == len(seq_plan.actions)
    return lifted, lifted_plan


def conditional_effects_problem():
    """`paint(c)` colours every block `b` with `ready(b)` and un-readies it;
    `prepare(b)` readies a block unless it equals `special`; the goal wants all
    painted. Exercises a forall conditional effect with a negated literal in
    its condition, an equality precondition and a static precondition."""
    Block, Colour = UserType("Block"), UserType("Colour")
    ready, painted = Fluent("ready", BoolType(), b=Block), Fluent("painted", BoolType(), b=Block, c=Colour)
    allowed = Fluent("allowed", BoolType(), c=Colour)          # static
    blocks = [Object(f"b{i}", Block) for i in range(3)]
    special = Object("special", Block)
    red, blue = Object("red", Colour), Object("blue", Colour)
    prepare = InstantaneousAction("prepare", b=Block)
    b = prepare.parameter("b")
    prepare.add_precondition(Not(Equals(b, special)))
    prepare.add_precondition(Not(ready(b)))
    prepare.add_effect(ready(b), True)
    paint = InstantaneousAction("paint", c=Colour)
    c = paint.parameter("c")
    paint.add_precondition(allowed(c))
    x = Variable("x", Block)
    paint.add_effect(painted(x, c), True, condition=ready(x), forall=[x])
    paint.add_effect(ready(x), False, condition=ready(x), forall=[x])
    problem = Problem("conditional")
    for fluent in (ready, painted, allowed):
        problem.add_fluent(fluent, default_initial_value=False)
    problem.add_objects(blocks + [special, red, blue])
    problem.set_initial_value(allowed(red), True)
    problem.add_actions([prepare, paint])
    for block in blocks:
        problem.add_goal(painted(block, red))
    return problem


# ---------------------------------------------------------------------------
# Same semantics as seq
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make_problem", [
    robot_line_problem,
    negative_precondition_problem,
    lambda: forall_problem()[0],
    lambda: forall_problem(universal_effect=True)[0],
    conditional_effects_problem,
])
def test_fixtures_solve_as_under_seq(make_problem):
    assert_same_as_seq(make_problem())


def test_conditional_effects_fire_per_binding_and_the_equality_holds():
    _, plan = solve(conditional_effects_problem(), "lifted")
    names = [a.action.name for a in plan.actions]
    assert names.count("prepare") == 3 and names[-1] == "paint"
    assert all(str(a.actual_parameters[0]) != "special" for a in plan.actions if a.action.name == "prepare")


@pytest.mark.parametrize("make_problem", [
    lambda: numeric_counter_problem(3),
    lambda: numeric_counter_problem(7),
    rover_recharge_problem,
    lambda: rover_recharge_problem(with_bool_goal=True),
])
def test_numeric_fixtures_solve_as_under_seq(make_problem):
    planner, _ = assert_same_as_seq(make_problem())
    assert planner.layers == ("core", "numeric")


@pytest.mark.parametrize("domain", ["alarm", "blocksworld", "gripper", "transport",
                                    "rover", "zenoflight", "tradepost"])
def test_bundled_classical_and_numeric_domains_solve_as_under_seq(domain):
    for problem in sorted(f for f in os.listdir(os.path.join(PDDL_DIR, domain)) if f.startswith("problem")):
        assert_same_as_seq(pddl(domain, problem))


def test_through_the_up_engine():
    with OneshotPlanner(name="PLASPPlanner", params={"encoding": "lifted"}) as planner:
        result = planner.solve(pddl("gripper"))
    assert result.status == PlanGenerationResultStatus.SOLVED_SATISFICING
    assert len(result.plan.actions) == 5


def test_a_fixed_horizon_solve_and_an_unsolvable_task():
    planner = PLASPPlanner(robot_line_problem(), "lifted")
    assert len(planner.plan(horizon=3).actions) == 3
    blocked = PLASPPlanner(robot_line_problem(connected_line=False), "lifted")
    blocked.plan(max_horizon=5)
    assert blocked.status == PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY


# ---------------------------------------------------------------------------
# What is emitted
# ---------------------------------------------------------------------------

def test_no_ground_action_is_named_and_a_literal_joins_only_its_parameters():
    planner = PLASPPlanner(pddl("gripper"), "lifted")
    facts = fact_predicates(planner.compiled_task.fact_lines)
    assert "schema" in facts and "param" in facts
    assert not facts & {"action", "precondition", "postcondition", "occurs"}
    rules = planner.step_rules.splitlines()
    # pick(OBJ, ROOM, GRIPPER)'s `free(GRIPPER)` precondition mentions one parameter:
    free = [r for r in rules if r.startswith(':- schema("pick", t)') and '"free"' in r]
    assert len(free) == 1 and free[0].count("arg(") == 1 and 'arg("pick", 3, GRIPPER, t)' in free[0]
    # and its `carry(OBJ, GRIPPER)` effect joins exactly those two.
    carry = [r for r in rules if r.startswith('caused(variable(("carry"') and '"pick"' in r]
    assert len(carry) == 1 and carry[0].count("arg(") == 2


def test_numeric_conditions_and_effects_are_per_schema_step_rules():
    """rover: `recharge` needs energy below a bound and raises it; `move`
    spends it. The comparison is a #sum over the expression's terms under the
    schema's arguments, the effects apply a delta to the value at t-1, and the
    expressions are declared under the parameters' types, not under a ground
    action."""
    planner = PLASPPlanner(pddl("rover"), "lifted")
    numeric = planner.compiled_task.numeric_step_rules
    assert numeric and all(r.startswith((":- schema(", "caused(")) for r in numeric)
    assert any("#sum {" in r and "numval(NV, NX, t-1)" in r for r in numeric)
    assert any("holds(" in r and "NX +" in r for r in numeric)
    facts = planner.compiled_task.fact_lines
    assert any(l.startswith("numTerm(") and ":- has(" in l for l in facts) or \
        any(l.startswith("numTerm(") and ":-" not in l for l in facts)
    assert not any(l.startswith(("numPrecondition(", "numEffect(", "numAssign(")) for l in facts)
    assert not any("occurs(" in r for r in numeric)


def test_the_initial_state_carries_the_false_values_the_lifted_rules_read():
    """The seq compile emits a false initial value only for fluents some literal
    reads as false; the lifted rules read the same literals, so the same set."""
    for make_problem in (negative_precondition_problem, conditional_effects_problem):
        seq = PLASPPlanner(make_problem(), "seq").compiled_task.facts["_initial_state"]
        lifted = PLASPPlanner(make_problem(), "lifted").compiled_task.facts["_initial_state"]
        assert seq == lifted


def test_the_dumped_program_is_a_standalone_clingo_input(tmp_path):
    planner = PLASPPlanner(pddl("gripper"), "lifted")
    path = tmp_path / "lifted.lp"
    planner.dump_lp_program(path)
    ctl = clingo.Control(["-n", "1"])
    ctl.load(str(path))
    ctl.ground([("base", []), ("check", [clingo.Number(5)])] + [("step", [clingo.Number(t)]) for t in range(1, 6)])
    ctl.assign_external(clingo.Function("query", [clingo.Number(5)]), True)
    assert ctl.solve().satisfiable


# ---------------------------------------------------------------------------
# Size
# ---------------------------------------------------------------------------

def _step_atoms(problem, encoding, horizon=4):
    planner = PLASPPlanner(problem, encoding)
    ctl = clingo.Control()
    for path in planner.encoding_paths:
        ctl.load(path)
    ctl.add("base", [], planner.task_facts)
    if planner.step_rules:
        ctl.add("step", ["t"], planner.step_rules)
    ctl.ground([("base", []), ("check", [clingo.Number(horizon)])]
               + [("step", [clingo.Number(t)]) for t in range(1, horizon + 1)])
    return sum(1 for _ in ctl.symbolic_atoms)


def big_gripper(balls=12, grippers=4):
    Room, Ball, Gripper = UserType("room"), UserType("ball"), UserType("gripper")
    at_robby, at = Fluent("at_robby", BoolType(), r=Room), Fluent("at", BoolType(), b=Ball, r=Room)
    free, carry = Fluent("free", BoolType(), g=Gripper), Fluent("carry", BoolType(), b=Ball, g=Gripper)
    move = InstantaneousAction("move", f=Room, t=Room)
    f, t = move.parameters
    move.add_precondition(at_robby(f)); move.add_effect(at_robby(f), False); move.add_effect(at_robby(t), True)
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
    problem = Problem("big_gripper")
    for fluent in (at_robby, at, free, carry):
        problem.add_fluent(fluent, default_initial_value=False)
    rooms = [Object("a", Room), Object("b", Room)]
    ball_objs = [Object(f"ball{i}", Ball) for i in range(balls)]
    gripper_objs = [Object(f"g{i}", Gripper) for i in range(grippers)]
    problem.add_objects(rooms + ball_objs + gripper_objs)
    problem.add_actions([move, pick, drop])
    problem.set_initial_value(at_robby(rooms[0]), True)
    for ball in ball_objs:
        problem.set_initial_value(at(ball, rooms[0]), True)
        problem.add_goal(at(ball, rooms[1]))
    for gr in gripper_objs:
        problem.set_initial_value(free(gr), True)
    return problem


def test_the_lifted_step_program_is_smaller_where_action_arity_exceeds_predicate_arity():
    """pick/drop have three parameters over a two-place `at`/`carry`: seq
    replicates each step rule per ground action, lifted per ground atom."""
    seq, lifted = _step_atoms(big_gripper(), "seq"), _step_atoms(big_gripper(), "lifted")
    assert lifted < seq / 2, (seq, lifted)


# ---------------------------------------------------------------------------
# Registry and refusals
# ---------------------------------------------------------------------------

def test_the_family_has_its_own_files_and_the_numeric_layer_is_seqs_verbatim():
    assert LIFTED.layer_names == ("core", "numeric")
    assert LIFTED.directory != SEQ.directory
    for name in LIFTED.layer_names:
        assert os.path.isfile(LIFTED.path(name)), name
    assert LIFTED.by_name["core"].describes_instance == SEQ.by_name["core"].describes_instance

    def rules(path):
        text = open(path).read()
        return text[text.index("#program base."):]
    assert rules(LIFTED.path("numeric")) == rules(SEQ.path("numeric"))


@pytest.mark.parametrize("make_problem, feature", [
    (disjunction_problem, "disjunctive"),
    (lambda: pddl("lift"), "disjunctive"),
    (lambda: pddl("briefcase", "problem-carry.pddl"), "disjunctive"),
    (lambda: pddl("matchcellar"), "temporal"),
])
def test_tasks_needing_a_missing_layer_are_refused_with_the_reason(make_problem, feature):
    with pytest.raises(ValueError, match=f"has no {feature} layer"):
        PLASPPlanner(make_problem(), "lifted")


def test_a_conditional_numeric_effect_is_refused_with_the_reason():
    from unified_planning.shortcuts import Fluent, IntType
    problem = numeric_counter_problem(3)
    tick = problem.action("tick")
    flag = Fluent("boost", BoolType())
    problem.add_fluent(flag, default_initial_value=False)
    tick.add_increase_effect(problem.fluent("counter")(), 2, condition=flag())
    with pytest.raises(ValueError, match="conditional numeric effect"):
        PLASPPlanner(problem, "lifted")


def test_the_seq_family_is_unchanged():
    planner, plan = solve(pddl("gripper"), "seq")
    assert "occurs" in planner.lp_program() and not planner.step_rules
    assert planner.solved_horizon == len(plan.actions) == 5
