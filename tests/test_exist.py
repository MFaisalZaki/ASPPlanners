"""Tests for the `exist` encoding family: exists-step parallel semantics.

The family is the seq encoding with a set of actions per horizon step, kept as
its own directory of .lp files (encodings/exist/). What is pinned here is
(1) the registry: its own files, the same fact ownership as seq, the numeric
and temporal vocabulary refused with a reason; (2) the two interference rules,
through tasks small enough that the horizon tells which steps were allowed;
and (3) that the sequential family is untouched.
"""

import os

import pytest

from unified_planning.io import PDDLReader
from unified_planning.shortcuts import (
    BoolType, Fluent, InstantaneousAction, OneshotPlanner, Problem,
)
from unified_planning.engines import PlanGenerationResultStatus

import aspplanners  # noqa: F401  (registers the UP engines)
from aspplanners.plasp.layers import EXIST, SEQ, fact_predicates
from aspplanners.plasp.planner import PLASPPlanner

from test_planner import numeric_counter_problem, robot_line_problem

PDDL_DIR = os.path.join(os.path.dirname(__file__), "pddl")


def pddl(domain, problem="problem.pddl"):
    return PDDLReader().parse_problem(os.path.join(PDDL_DIR, domain, "domain.pddl"),
                                      os.path.join(PDDL_DIR, domain, problem))


def solve(problem, spec):
    planner = PLASPPlanner(problem, spec)
    plan = planner.plan(timeout=60)
    assert planner.status == PlanGenerationResultStatus.SOLVED_SATISFICING, planner.logs
    return planner, plan


def two_actions(first, second, first_deletes_p=False, conflicting_writes=False):
    """Two actions, `first` and `second`, that both need `p` and each set their
    own `done` flag; the goal wants both flags.

    `first_deletes_p`: `first` also clears `p`, so it disables `second`.
    `conflicting_writes`: the two write `v` to opposite values.
    The names decide the fixed execution order -- clingo orders the action terms
    by their name strings -- which is what the tests below steer.
    """
    p, v = Fluent("p", BoolType()), Fluent("v", BoolType())
    done = {name: Fluent(f"done_{name}", BoolType()) for name in (first, second)}
    problem = Problem("two_actions")
    problem.add_fluent(p, default_initial_value=True)
    problem.add_fluent(v, default_initial_value=False)
    for flag in done.values():
        problem.add_fluent(flag, default_initial_value=False)
        problem.add_goal(flag())
    for name in (first, second):
        action = InstantaneousAction(name)
        action.add_precondition(p())
        action.add_effect(done[name](), True)
        if first_deletes_p and name == first:
            action.add_effect(p(), False)
        if conflicting_writes:
            action.add_effect(v(), name == first)
        problem.add_action(action)
    return problem


# ---------------------------------------------------------------------------
# The family's registry
# ---------------------------------------------------------------------------

def test_the_exist_family_has_its_own_files():
    assert EXIST.directory != SEQ.directory
    for name in EXIST.layer_names:
        assert os.path.isfile(EXIST.path(name)), name
    assert EXIST.layer_names == ("core", "disjunctive")
    assert EXIST.parallel and not SEQ.parallel


def test_the_exist_layers_own_what_their_seq_counterparts_own():
    """Same encoder, same facts: only the step rules differ, so the fact
    vocabulary each layer consumes must be seq's -- or selection would drift."""
    for layer in EXIST.layers:
        assert layer.owns == SEQ.by_name[layer.name].owns, layer.name


def test_the_exist_core_reads_every_fact_it_owns():
    """Same drift guard test_layers applies to seq: a claimed fact has to be
    read by a rule of the file that claims it."""
    from test_layers import _rule_predicates  # noqa: F401  (same logic, other family)
    import re
    from aspplanners.lp_io import parse_lp_file, ASPDirective
    for layer in EXIST.layers:
        names = set()
        for statement in parse_lp_file(EXIST.path(layer.name)):
            if not isinstance(statement, ASPDirective):
                names.update(re.findall(r"([a-z_][A-Za-z0-9_]*)\s*\(", str(statement)))
        assert sorted((layer.owns - layer.describes_instance) - names) == [], layer.name


def test_exist_selects_the_feature_layers_the_task_needs():
    assert PLASPPlanner(pddl("gripper"), "exist").layers == ("core",)
    assert PLASPPlanner(pddl("lift"), "exist").layers == ("core", "disjunctive")
    assert PLASPPlanner(pddl("gripper"), "exist+disjunctive").layers == ("core", "disjunctive")


def test_exist_never_loads_a_seq_file():
    planner = PLASPPlanner(pddl("lift"), "exist")
    assert all(os.sep + "exist" + os.sep in path for path in planner.encoding_paths)


@pytest.mark.parametrize("make_problem, feature", [
    (numeric_counter_problem, "numeric"),
    (lambda: pddl("matchcellar"), "temporal"),
])
def test_exist_refuses_numeric_and_temporal_tasks_with_a_reason(make_problem, feature):
    with pytest.raises(ValueError, match=f"has no {feature} layer"):
        PLASPPlanner(make_problem(), "exist")


# ---------------------------------------------------------------------------
# The semantics, on tasks whose horizon tells what was allowed to share a step
# ---------------------------------------------------------------------------

def test_independent_actions_share_a_step():
    """Gripper: pick with both grippers, move, drop both -- three steps for a
    five-action plan, and the serialised plan still validates."""
    sequential, _ = solve(pddl("gripper"), "seq")
    parallel, plan = solve(pddl("gripper"), "exist")
    assert sequential.solved_horizon == 5
    assert parallel.solved_horizon == 3
    assert len(plan.actions) == 5


def test_a_disabling_action_may_not_precede_what_it_disables():
    """`a_del` clears p, which `b_keep` needs. The fixed order runs `a_del`
    first, so the pair cannot share a step: two steps, `b_keep` before `a_del`."""
    planner, plan = solve(two_actions("a_del", "b_keep", first_deletes_p=True), "exist")
    assert planner.solved_horizon == 2
    assert [a.action.name for a in plan.actions] == ["b_keep", "a_del"]


def test_a_disabling_action_may_follow_what_it_disables_in_the_same_step():
    """Same pair, renamed so the fixed order runs the reader first: one step,
    serialised reader-then-deleter -- the order the step is legal under."""
    planner, plan = solve(two_actions("z_del", "b_keep", first_deletes_p=True), "exist")
    assert planner.solved_horizon == 1
    assert [a.action.name for a in plan.actions] == ["b_keep", "z_del"]


def test_two_actions_writing_one_variable_to_different_values_do_not_share_a_step():
    planner, _ = solve(two_actions("a", "b", conflicting_writes=True), "exist")
    assert planner.solved_horizon == 2


def test_two_actions_writing_one_variable_to_the_same_value_do_share_a_step():
    problem = two_actions("a", "b")
    for action in problem.actions:
        action.add_effect(problem.fluent("v")(), True)
    planner, _ = solve(problem, "exist")
    assert planner.solved_horizon == 1


def test_one_action_may_still_set_and_clear_a_fluent_through_conditional_effects():
    """Lift's `stop` boards and un-boards through two forall/when effects on one
    fluent; core's add-wins rule settles that, and the layer's conflicting-writes
    rule must not mistake it for two actions in conflict (it did, once)."""
    sequential, _ = solve(pddl("lift"), "seq")
    parallel, _ = solve(pddl("lift"), "exist")
    assert parallel.solved_horizon <= sequential.solved_horizon


@pytest.mark.parametrize("domain", ["alarm", "blocksworld", "gripper", "lift", "transport"])
def test_every_classical_bundled_domain_solves_under_exist(domain):
    problems = sorted(f for f in os.listdir(os.path.join(PDDL_DIR, domain)) if f.startswith("problem"))
    for problem in problems:
        sequential, _ = solve(pddl(domain, problem), "seq")
        parallel, _ = solve(pddl(domain, problem), "exist")
        # Never longer: every sequential plan is an exists-step plan.
        assert parallel.solved_horizon <= sequential.solved_horizon, (domain, problem)


def test_through_the_up_engine():
    with OneshotPlanner(name="PLASPPlanner", params={"encoding": "exist"}) as planner:
        result = planner.solve(pddl("gripper"))
    assert result.status == PlanGenerationResultStatus.SOLVED_SATISFICING
    assert len(result.plan.actions) == 5


# ---------------------------------------------------------------------------
# The sequential reading is unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make_problem", [
    robot_line_problem, lambda: pddl("gripper"), lambda: pddl("lift")])
def test_the_sequential_reading_still_puts_one_action_per_step(make_problem):
    planner, plan = solve(make_problem(), "seq")
    assert not planner.family.parallel
    assert planner.solved_horizon == len(plan.actions)
