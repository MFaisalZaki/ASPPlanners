"""Symmetry breaking on the seq encoding (aspplanners.plasp.symmetry).

Pinned here: (1) the interchangeable-object classes found on the fixtures;
(2) with either constraint on, every bundled classical domain and a numeric
task are solved at the SAME horizon as without -- the lex-leader argument in
the module docstring says the shortest horizon is unchanged, and this is the
test of it; (3) the constraints do bite: a plan that violates them is no
longer a model; (4) the option through the UP engine, and the refusals.
"""

import os

import pytest
import unified_planning.shortcuts as ups
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner
from unified_planning.engines import PlanGenerationResultStatus

import aspplanners  # noqa: F401  (registers the engines)
from aspplanners.plasp.planner import PLASPPlanner
from aspplanners.plasp.symmetry import object_classes, symmetry_program
from tests.test_pdr import fuel_gripper

ups.get_environment().credits_stream = None
PDDL_DIR = os.path.join(os.path.dirname(__file__), "pddl")


def pddl(domain, problem="problem.pddl"):
    return PDDLReader().parse_problem(os.path.join(PDDL_DIR, domain, "domain.pddl"),
                                      os.path.join(PDDL_DIR, domain, problem))


def test_gripper_balls_and_grippers_are_interchangeable_but_rooms_are_not():
    classes = object_classes(PLASPPlanner(pddl("gripper")).task)
    assert classes == [["ball-1", "ball-2"], ["left-gripper", "right-gripper"]]


@pytest.mark.parametrize("domain", ["blocksworld", "lift", "transport"])
def test_tasks_whose_goal_or_initial_state_tells_the_objects_apart_have_no_classes(domain):
    assert object_classes(PLASPPlanner(pddl(domain)).task) == []


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
    problem = fuel_gripper(10, 100)
    classes = object_classes(PLASPPlanner(problem).task)
    balls = next(c for c in classes if len(c) == 10)
    assert balls == sorted(balls)


def test_the_program_states_classes_mentions_and_the_order_switch():
    planner = PLASPPlanner(pddl("gripper"))
    program, classes = symmetry_program(planner.task, "all")
    assert len(classes) == 2
    assert 'symClass(0, 0, constant("ball_1")).' in program
    assert 'symClass(0, 1, constant("ball_2")).' in program
    assert "mentions(action((\"pick\",OBJ,ROOM,GRIPPER)), OBJ) :- action(action((\"pick\",OBJ,ROOM,GRIPPER)))." in program
    assert "symOrder." in program
    objects_only, _ = symmetry_program(planner.task, "objects")
    assert "symOrder." not in objects_only
    order_only, found = symmetry_program(planner.task, "order")
    assert "symClass" not in order_only and found == []


def _horizon(problem, **kwargs):
    planner = PLASPPlanner(problem)
    plan = planner.plan(timeout=120, **kwargs)
    assert planner.status == PlanGenerationResultStatus.SOLVED_SATISFICING, planner.logs
    assert planner.validate(plan)[0]
    return planner.solved_horizon


@pytest.mark.parametrize("mode", ["objects", "order", "all"])
@pytest.mark.parametrize("domain", ["alarm", "blocksworld", "gripper", "lift", "transport"])
def test_the_shortest_horizon_is_unchanged_on_the_classical_domains(domain, mode):
    problem = pddl(domain, "problem-trigger.pddl" if domain == "alarm" else "problem.pddl")
    assert _horizon(problem, symmetry=mode) == _horizon(problem)


@pytest.mark.parametrize("mode", ["objects", "order", "all"])
def test_the_shortest_horizon_is_unchanged_on_a_numeric_task(mode):
    problem = fuel_gripper(3, 3)
    assert _horizon(problem, symmetry=mode) == _horizon(problem)


def _one_step_models(planner, program, extra):
    """Models of the one-step seq program plus `program` and `extra` in base."""
    import clingo
    ctl = clingo.Control(["0"])
    for path in planner.encoding_paths:
        ctl.load(path)
    ctl.load(os.path.join(planner.family.directory, "symmetry.lp"))
    ctl.add("base", [], planner.task_facts + "\n" + program + "\n" + extra)
    ctl.ground([("base", []), ("step", [clingo.Number(1)]), ("check", [clingo.Number(1)])])
    ctl.assign_external(clingo.Function("query", [clingo.Number(1)]), True)
    models = []
    ctl.solve(on_model=lambda m: models.append(m.symbols(shown=True)))
    return models


def test_the_object_constraint_rules_out_using_the_second_ball_first():
    # Goal: some ball is carried -- symmetric, reached by any one pick. With the
    # constraint, the pick has to be ball-1 with the left gripper; forbidding
    # ball-1 leaves no model, where without the constraint two remain.
    problem = pddl("gripper")
    problem.clear_goals()
    from unified_planning.shortcuts import Exists, Variable
    b, g = Variable("b", problem.user_type("ball")), Variable("g", problem.user_type("gripper"))
    problem.add_goal(Exists(problem.fluent("carry")(b, g), b, g))
    planner = PLASPPlanner(problem)
    program, classes = symmetry_program(planner.task, "objects")
    assert [len(c) for c in classes] == [2, 2]
    forbid_ball_1 = ':- occurs(A, 1), mentions(A, constant("ball_1")).'
    assert _one_step_models(planner, program, forbid_ball_1) == []
    mentions_only = "\n".join(line for line in program.splitlines() if line.startswith("mentions"))
    assert len(_one_step_models(planner, mentions_only, forbid_ball_1)) == 2
    models = _one_step_models(planner, program, "")
    assert len(models) == 1
    assert str(models[0][0]) == 'occurs(action(("pick",constant("ball_1"),constant("room_a"),constant("left_gripper"))),1)'


def test_the_order_constraint_rules_out_a_descending_pair_of_independent_actions():
    # pick(ball-1, left) ; pick(ball-2, right) is the only ascending order of the
    # two picks; the same two picks the other way round are a plan of the task
    # but not a model.
    problem = pddl("gripper")
    problem.clear_goals()
    carry = problem.fluent("carry")
    problem.add_goal(carry(problem.object("ball-1"), problem.object("left-gripper")))
    problem.add_goal(carry(problem.object("ball-2"), problem.object("right-gripper")))
    planner = PLASPPlanner(problem)
    plan = planner.plan(horizon=2, symmetry="order")
    assert planner.status == PlanGenerationResultStatus.SOLVED_SATISFICING
    names = [str(a.action.name) + str(tuple(str(p) for p in a.actual_parameters)) for a in plan.actions]
    assert names[0].startswith("pick('ball-1'"), names


def test_a_temporal_task_skips_the_layer_with_a_log_line():
    planner = PLASPPlanner(pddl("matchcellar"))
    planner.plan(timeout=120, symmetry="all")
    assert planner.status == PlanGenerationResultStatus.SOLVED_SATISFICING
    assert any("Symmetry breaking skipped" in line for line in planner.logs)


def test_the_lifted_encoding_and_a_bad_mode_are_refused():
    with pytest.raises(ValueError, match="seq encoding only"):
        PLASPPlanner(pddl("gripper"), "lifted").plan(symmetry="all")
    with pytest.raises(ValueError, match="symmetry must be one of"):
        PLASPPlanner(pddl("gripper")).plan(symmetry="yes")


def test_through_the_up_engine():
    with OneshotPlanner(name="PLASPPlanner", params={"encoding": "seq", "symmetry": "all"}) as planner:
        result = planner.solve(pddl("gripper"))
    assert result.status == PlanGenerationResultStatus.SOLVED_SATISFICING
    assert len(result.plan.actions) == 5
    assert any("object class" in m.message for m in result.log_messages)
