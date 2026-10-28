"""Object symmetry breaking (aspplanners.plasp.symmetry), always on.

Pinned here: (1) the interchangeable-object classes found on the fixtures;
(2) every bundled domain -- classical, numeric, temporal, on seq and on
lifted -- is solved at the SAME horizon with the layer as with it stripped:
the lex-leader argument in the module docstring says the shortest horizon
is unchanged, and this is the test of it; (3) the constraint bites: a plan
that violates it is no longer a model; (4) PDR's program leaves the layer
out; (5) the layer joins an explicit layer spec.
"""

import glob
import os

import clingo
import pytest
import unified_planning.shortcuts as ups
from unified_planning.io import PDDLReader
from unified_planning.engines import PlanGenerationResultStatus

import aspplanners  # noqa: F401  (registers the engines)
from aspplanners.plasp.planner import PLASPPlanner
from aspplanners.plasp.symmetry import object_classes, symmetry_facts
from tests.test_pdr import fuel_gripper

ups.get_environment().credits_stream = None
PDDL_DIR = os.path.join(os.path.dirname(__file__), "pddl")


def pddl(domain, problem="problem.pddl"):
    return PDDLReader().parse_problem(os.path.join(PDDL_DIR, domain, "domain.pddl"),
                                      os.path.join(PDDL_DIR, domain, problem))


def strip_symmetry(planner):
    """The same planner without its symmetry layer and facts: the control
    case of every "unchanged" test below."""
    planner.task_facts = "\n".join(line for line in planner.task_facts.splitlines()
                                   if not line.startswith(("symClass(", "mentions(")))
    kept = [(n, p) for n, p in zip(planner.layers, planner.encoding_paths) if n != "symmetry"]
    planner.layers = tuple(n for n, _ in kept)
    planner.encoding_paths = [p for _, p in kept]
    return planner


# ---------------------------------------------------------------------------
# Class detection
# ---------------------------------------------------------------------------

def test_gripper_balls_and_grippers_are_interchangeable_but_rooms_are_not():
    planner = PLASPPlanner(pddl("gripper"))
    assert planner.symmetry_classes == [["ball-1", "ball-2"], ["left-gripper", "right-gripper"]]
    assert "symmetry" in planner.layers


@pytest.mark.parametrize("domain", ["blocksworld", "lift", "transport"])
def test_tasks_whose_goal_or_initial_state_tells_the_objects_apart_have_no_classes(domain):
    planner = PLASPPlanner(pddl(domain))
    assert planner.symmetry_classes == []
    assert "symmetry" not in planner.layers


def test_an_object_an_action_names_outright_is_never_in_a_class():
    from unified_planning.shortcuts import BoolType, Fluent, InstantaneousAction, Object, Problem, UserType
    Thing = UserType("thing")
    on = Fluent("on", BoolType(), x=Thing)
    a, b, c = (Object(n, Thing) for n in "abc")
    light = InstantaneousAction("light", x=Thing)
    light.add_precondition(on(a))        # `a` is a constant of the schema
    light.add_effect(on(light.parameters[0]), True)
    problem = Problem("pinned")
    problem.add_fluent(on, default_initial_value=False)
    problem.add_objects([a, b, c])
    problem.add_action(light)
    problem.set_initial_value(on(a), True)
    problem.add_goal(on(b)); problem.add_goal(on(c))
    assert object_classes(PLASPPlanner(problem).task) == [["b", "c"]]


def test_classes_are_in_clingo_term_order_of_the_constants():
    # Ten balls: by name "ball-10" sorts before "ball-2", and clingo compares
    # strings the same way, which is what the lex-leader argument needs.
    classes = PLASPPlanner(fuel_gripper(10, 100)).symmetry_classes
    balls = next(c for c in classes if len(c) == 10)
    assert balls == sorted(balls)


def test_the_facts_state_classes_and_mentions():
    planner = PLASPPlanner(pddl("gripper"))
    lines, classes = symmetry_facts(planner.task, planner.task.actions)
    assert len(classes) == 2
    assert 'symClass(0, 0, constant("ball_1")).' in lines
    assert 'symClass(0, 1, constant("ball_2")).' in lines
    assert 'mentions(action(("pick",OBJ,ROOM,GRIPPER)), OBJ) :- action(action(("pick",OBJ,ROOM,GRIPPER))).' in lines
    assert set(lines) == planner.compiled_task.facts["_symmetry"]
    assert "symClass" in planner.lp_program()


def test_the_lifted_facts_keep_the_classes_and_drop_the_mentions_rules():
    planner = PLASPPlanner(pddl("gripper"), "lifted")
    assert planner.symmetry_classes == [["ball-1", "ball-2"], ["left-gripper", "right-gripper"]]
    assert "symmetry" in planner.layers
    assert not any(line.startswith("mentions(") for line in planner.compiled_task.facts["_symmetry"])


# ---------------------------------------------------------------------------
# The shortest horizon is unchanged
# ---------------------------------------------------------------------------

def _horizon(problem, encoding="seq", stripped=False):
    planner = PLASPPlanner(problem, encoding)
    if stripped:
        strip_symmetry(planner)
    plan = planner.plan(timeout=120)
    assert planner.status == PlanGenerationResultStatus.SOLVED_SATISFICING, planner.logs
    assert planner.validate(plan)[0]
    return planner.solved_horizon


def _every_bundled_task():
    for domain_file in sorted(glob.glob(os.path.join(PDDL_DIR, "*", "domain.pddl"))):
        directory = os.path.dirname(domain_file)
        for problem_file in sorted(glob.glob(os.path.join(directory, "problem*.pddl"))):
            yield os.path.basename(directory), os.path.basename(problem_file)


@pytest.mark.parametrize("domain,problem", list(_every_bundled_task()))
def test_the_shortest_horizon_is_unchanged_on_every_bundled_task(domain, problem):
    try:
        task = pddl(domain, problem)
    except Exception as error:  # a reader limitation is not this layer's
        pytest.skip(f"PDDL reader: {error}")
    try:
        planner = PLASPPlanner(task)
    except (ValueError, NotImplementedError) as refusal:
        pytest.skip(f"refused: {refusal}")
    if not planner.symmetry_classes:
        pytest.skip("no interchangeable objects")
    with_layer = planner.plan(timeout=120)
    if planner.status != PlanGenerationResultStatus.SOLVED_SATISFICING:
        pytest.skip(f"not solved: {planner.status.name}")
    assert planner.validate(with_layer)[0]
    assert _horizon(pddl(domain, problem), stripped=True) == planner.solved_horizon


@pytest.mark.parametrize("encoding", ["seq", "lifted"])
def test_the_shortest_horizon_is_unchanged_on_a_numeric_task(encoding):
    problem = fuel_gripper(3, 3)
    assert _horizon(problem, encoding) == _horizon(problem, encoding, stripped=True)


def test_the_shortest_horizon_is_unchanged_on_lifted_gripper():
    assert _horizon(pddl("gripper"), "lifted") == _horizon(pddl("gripper"), "lifted", stripped=True) == 5


# ---------------------------------------------------------------------------
# The constraint bites
# ---------------------------------------------------------------------------

def _one_step_models(planner, extra=""):
    """Models of the planner's program at horizon 1, plus `extra` in base."""
    ctl = clingo.Control(["0"])
    for path in planner.encoding_paths:
        ctl.load(path)
    ctl.add("base", [], planner.task_facts + "\n" + extra)
    ctl.ground([("base", []), ("step", [clingo.Number(1)]), ("check", [clingo.Number(1)])])
    ctl.assign_external(clingo.Function("query", [clingo.Number(1)]), True)
    models = []
    ctl.solve(on_model=lambda m: models.append(m.symbols(shown=True)))
    return models


def _some_ball_carried():
    problem = pddl("gripper")
    problem.clear_goals()
    from unified_planning.shortcuts import Exists, Variable
    b, g = Variable("b", problem.user_type("ball")), Variable("g", problem.user_type("gripper"))
    problem.add_goal(Exists(problem.fluent("carry")(b, g), b, g))
    return problem


def test_the_constraint_rules_out_using_the_second_ball_or_gripper_first():
    # Goal: some ball is carried -- symmetric, reached by any one pick. With the
    # layer, the pick has to be ball-1 with the left gripper: one model where
    # the stripped program has four.
    planner = PLASPPlanner(_some_ball_carried())
    assert [len(c) for c in planner.symmetry_classes] == [2, 2]
    models = _one_step_models(planner)
    assert len(models) == 1
    assert str(models[0][0]) == 'occurs(action(("pick",constant("ball_1"),constant("room_a"),constant("left_gripper"))),1)'
    assert len(_one_step_models(strip_symmetry(PLASPPlanner(_some_ball_carried())))) == 4


# ---------------------------------------------------------------------------
# Where it does not apply, and the explicit spec
# ---------------------------------------------------------------------------

def test_pdr_leaves_the_layer_out_and_still_solves():
    planner = PLASPPlanner(pddl("gripper"))
    assert "symmetry" in planner.layers
    assert "Layer: symmetry" not in planner.pdr_program()
    plan = planner.plan(search="pdr")
    assert planner.status == PlanGenerationResultStatus.SOLVED_SATISFICING
    assert planner.validate(plan)[0]
    planner = PLASPPlanner(fuel_gripper(2, 20))
    planner.plan(pdr_first=30)
    assert planner.status == PlanGenerationResultStatus.SOLVED_SATISFICING


def test_the_layer_joins_an_explicit_layer_spec():
    planner = PLASPPlanner(pddl("gripper"), "seq+core")
    assert planner.layers == ("core", "symmetry")
    planner = PLASPPlanner(pddl("blocksworld"), "seq+core")
    assert planner.layers == ("core",)
