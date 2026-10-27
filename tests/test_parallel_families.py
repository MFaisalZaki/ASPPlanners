"""Tests for the `forall` and `exists` encoding families and their propagators.

Each is the seq encoding, in its own directory of .lp files, with a horizon
step read as a set of actions and the mutexes generated on demand during
search by a propagator (aspplanners.plasp.propagators), after Abdelwahed,
Bofill, Espasa & Villaret (ICAPS 2026). Pinned here: (1) the registry: own
files, the same fact ownership as seq, temporal refused with a reason; (2) the
semantics, through tasks small enough that the horizon says which steps were
allowed, and through every bundled classical and numeric domain, whose plans
the UP validator checks; (3) the read/write analysis the propagator builds from
the facts; (4) that the seq family is unchanged.
"""

import os

import clingo
import pytest

from unified_planning.io import PDDLReader
from unified_planning.shortcuts import (
    BoolType, Fluent, InstantaneousAction, IntType, OneshotPlanner, Problem,
)
from unified_planning.engines import PlanGenerationResultStatus

import aspplanners  # noqa: F401  (registers the UP engines)
from aspplanners.plasp.layers import EXISTS, FORALL, SEQ
from aspplanners.plasp.planner import PLASPPlanner
from aspplanners.plasp.propagators import ANY, InterferenceGraph, ParallelStepPropagator

from test_planner import robot_line_problem

PDDL_DIR = os.path.join(os.path.dirname(__file__), "pddl")


def pddl(domain, problem="problem.pddl"):
    return PDDLReader().parse_problem(os.path.join(PDDL_DIR, domain, "domain.pddl"),
                                      os.path.join(PDDL_DIR, domain, problem))


def solve(problem, encoding="seq", **options):
    planner = PLASPPlanner(problem, encoding, **options)
    plan = planner.plan(timeout=120)
    assert planner.status == PlanGenerationResultStatus.SOLVED_SATISFICING, planner.logs
    return planner, plan


def two_actions(first, second, first_deletes_p=False, conflicting_writes=False, same_writes=False):
    """Two actions that need `p` and each set their own `done` flag; the goal
    wants both flags. The options make the first disable the second, or make
    their writes to `v` clash or agree."""
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
        if same_writes:
            action.add_effect(v(), True)
        problem.add_action(action)
    return problem


# ---------------------------------------------------------------------------
# The families' registry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", [FORALL, EXISTS], ids=lambda f: f.name)
def test_a_parallel_family_has_its_own_files_and_seqs_ownership(family):
    assert family.directory != SEQ.directory
    assert family.layer_names == ("core", "numeric", "disjunctive")
    for layer in family.layers:
        assert os.path.isfile(family.path(layer.name)), layer.name
        # Same encoder, same facts: only the step rules differ, so the fact
        # vocabulary each layer consumes must be seq's or selection would drift.
        assert layer.owns == SEQ.by_name[layer.name].owns, layer.name
        assert layer.kind_features == SEQ.by_name[layer.name].kind_features, layer.name
    assert family.propagator == family.name
    assert SEQ.propagator is None


@pytest.mark.parametrize("family", [FORALL, EXISTS], ids=lambda f: f.name)
def test_the_numeric_and_disjunctive_copies_are_verbatim(family):
    """Only the header differs: the rules are seq's, byte for byte."""
    def rules(path):
        text = open(path).read()
        return text[text.index("#program base."):]
    for name in ("numeric", "disjunctive"):
        assert rules(family.path(name)) == rules(SEQ.path(name)), name


@pytest.mark.parametrize("encoding", ["forall", "exists"])
def test_a_parallel_family_selects_the_feature_layers_the_task_needs(encoding):
    assert PLASPPlanner(pddl("gripper"), encoding).layers == ("core",)
    assert PLASPPlanner(pddl("lift"), encoding).layers == ("core", "disjunctive")
    assert PLASPPlanner(pddl("rover"), encoding).layers == ("core", "numeric")
    planner = PLASPPlanner(pddl("lift"), encoding)
    assert all(os.sep + encoding + os.sep in path for path in planner.encoding_paths)


@pytest.mark.parametrize("encoding", ["forall", "exists"])
def test_temporal_tasks_are_refused_with_a_reason(encoding):
    with pytest.raises(ValueError, match="has no temporal layer"):
        PLASPPlanner(pddl("matchcellar"), encoding)


def test_the_parallel_cores_state_a_set_of_actions_and_seqs_core_one():
    assert "1 {occurs(Action, t) : action(Action)} 1." in PLASPPlanner(pddl("gripper")).lp_program()
    for encoding in ("forall", "exists"):
        program = PLASPPlanner(pddl("gripper"), encoding).lp_program()
        assert "1 {occurs(Action, t) : action(Action)}." in program
        assert "} 1." not in program


# ---------------------------------------------------------------------------
# Semantics, on tasks whose horizon tells what was allowed to share a step
# ---------------------------------------------------------------------------

def test_independent_actions_share_a_step():
    """Gripper: pick with both grippers and move in one exists-step (the move
    disables the picks, so it is serialised last); three steps under forall."""
    sequential, _ = solve(pddl("gripper"))
    exists, plan = solve(pddl("gripper"), "exists")
    forall, _ = solve(pddl("gripper"), "forall")
    assert (sequential.solved_horizon, exists.solved_horizon, forall.solved_horizon) == (5, 2, 3)
    assert len(plan.actions) == 5


def test_a_disabling_pair_shares_an_exists_step_in_the_right_order_whatever_the_names():
    """`a` clears p, which `b` needs: under exists-step they share a step with b
    serialised first -- the cycle check is name-independent, unlike a fixed
    order would be. Under forall-step they interfere and cannot share one."""
    for names in (("a_del", "b_keep"), ("z_del", "b_keep")):
        problem = two_actions(*names, first_deletes_p=True)
        exists, plan = solve(problem, "exists")
        assert exists.solved_horizon == 1
        assert [a.action.name for a in plan.actions] == [names[1], names[0]]
        forall, _ = solve(problem, "forall")
        assert forall.solved_horizon == 2


def two_cycle_problem():
    """`a` needs p and clears q; `b` needs q and clears p; `restore` sets both.
    Each disables the other, so no order of {a, b} works: a 2-cycle. The goal
    wants both done, which takes a, restore, b -- three steps -- unless the
    cycle were wrongly allowed, in which case one step would do."""
    p, q = Fluent("p", BoolType()), Fluent("q", BoolType())
    done_a, done_b = Fluent("done_a", BoolType()), Fluent("done_b", BoolType())
    problem = Problem("two_cycle")
    for fluent in (p, q):
        problem.add_fluent(fluent, default_initial_value=True)
    for fluent in (done_a, done_b):
        problem.add_fluent(fluent, default_initial_value=False)
        problem.add_goal(fluent())
    a, b, restore = InstantaneousAction("a"), InstantaneousAction("b"), InstantaneousAction("restore")
    a.add_precondition(p()); a.add_effect(q(), False); a.add_effect(done_a(), True)
    b.add_precondition(q()); b.add_effect(p(), False); b.add_effect(done_b(), True)
    restore.add_effect(p(), True); restore.add_effect(q(), True)
    for action in (a, b, restore):
        problem.add_action(action)
    return problem


def test_a_two_cycle_cannot_share_a_step():
    sequential, _ = solve(two_cycle_problem())
    exists, plan = solve(two_cycle_problem(), "exists")
    assert sequential.solved_horizon == exists.solved_horizon == 3
    assert [x.action.name for x in plan.actions] in (["a", "restore", "b"], ["b", "restore", "a"])


def test_writes_to_different_values_do_not_share_a_step_but_equal_writes_do():
    clash, _ = solve(two_actions("a", "b", conflicting_writes=True), "exists")
    agree, _ = solve(two_actions("a", "b", same_writes=True), "exists")
    assert (clash.solved_horizon, agree.solved_horizon) == (2, 1)


def test_one_action_may_still_set_and_clear_a_fluent_through_conditional_effects():
    """Lift's `stop` boards and un-boards through two forall/when effects on one
    fluent; core's add-wins rule settles that within one action, and the
    propagator only ever judges pairs of distinct actions."""
    sequential, _ = solve(pddl("lift"))
    exists, _ = solve(pddl("lift"), "exists")
    assert exists.solved_horizon == 4 < sequential.solved_horizon == 5


@pytest.mark.parametrize("domain", ["alarm", "blocksworld", "gripper", "lift", "transport",
                                    "rover", "zenoflight", "tradepost", "briefcase"])
def test_every_bundled_classical_and_numeric_domain_solves_in_parallel(domain):
    """Validated plans under both semantics, never more steps than sequential
    (a sequential plan is a forall-step plan is an exists-step plan), and
    exists never more than forall."""
    problems = sorted(f for f in os.listdir(os.path.join(PDDL_DIR, domain)) if f.startswith("problem"))
    for problem in problems:
        sequential, _ = solve(pddl(domain, problem))
        forall, _ = solve(pddl(domain, problem), "forall")
        exists, _ = solve(pddl(domain, problem), "exists")
        assert exists.solved_horizon <= forall.solved_horizon <= sequential.solved_horizon, (domain, problem)


def test_eager_and_lazy_interference_agree():
    for domain in ("gripper", "lift", "rover"):
        lazy, _ = solve(pddl(domain), "exists", interference="lazy")
        eager, _ = solve(pddl(domain), "exists", interference="eager")
        assert lazy.solved_horizon == eager.solved_horizon, domain


def two_ticks_problem():
    """`tick_a` and `tick_b` each increment `counter`; `finish` needs it >= 2."""
    counter = Fluent("counter", IntType())
    done = Fluent("done", BoolType())
    problem = Problem("two_ticks")
    problem.add_fluent(counter, default_initial_value=0)
    problem.add_fluent(done, default_initial_value=False)
    for name in ("tick_a", "tick_b"):
        tick = InstantaneousAction(name)
        tick.add_increase_effect(counter(), 1)
        problem.add_action(tick)
    finish = InstantaneousAction("finish")
    finish.add_precondition(counter() >= 2)
    finish.add_effect(done(), True)
    problem.add_action(finish)
    problem.add_goal(done())
    return problem


def test_numeric_writers_of_one_fluent_do_not_share_a_step():
    """Two `tick`s would each state counter+1 as the new value -- two values, not
    a sum -- so the propagator keeps numeric writers of one fluent apart."""
    planner, plan = solve(two_ticks_problem(), "exists")
    assert planner.solved_horizon == 3 and len(plan.actions) == 3


def test_through_the_up_engine():
    with OneshotPlanner(name="PLASPPlanner", params={"encoding": "exists"}) as planner:
        result = planner.solve(pddl("gripper"))
    assert result.status == PlanGenerationResultStatus.SOLVED_SATISFICING
    assert len(result.plan.actions) == 5


# ---------------------------------------------------------------------------
# The read/write analysis
# ---------------------------------------------------------------------------

def _graph(problem, interference="lazy"):
    planner = PLASPPlanner(problem)
    ctl = clingo.Control()
    for path in planner.encoding_paths:
        ctl.load(path)
    ctl.add("base", [], planner.task_facts)
    ctl.ground([("base", [])])
    return InterferenceGraph(ctl.symbolic_atoms, interference)


def _action(graph, name):
    return next(a for a in set(graph.reads) | set(graph.writes) if name in str(a))


def test_disabling_is_judged_on_values_and_in_one_direction():
    graph = _graph(pddl("gripper"))
    move_ab = next(a for a in graph.writes if str(a).startswith('action(("move",constant("room_a"),constant("room_b")'))
    pick_a = next(a for a in graph.reads if str(a).startswith('action(("pick",constant("ball_1"),constant("room_a")'))
    drop_b = next(a for a in graph.reads if str(a).startswith('action(("drop",constant("ball_1"),constant("room_b")'))
    assert graph.disables(move_ab, pick_a)        # move clears at_robby(room_a), which pick reads true
    assert not graph.disables(pick_a, move_ab)    # pick writes nothing move reads
    assert not graph.disables(move_ab, drop_b)    # move sets at_robby(room_b) to the value drop reads


def test_numeric_fluents_are_value_less_reads_and_writes():
    graph = _graph(two_ticks_problem())
    tick_a, tick_b, finish = (_action(graph, n) for n in ('"tick_a"', '"tick_b"', '"finish"'))
    (counter, written), = graph.writes[tick_a].items()
    assert written == {ANY}
    assert graph.reads[finish][counter] == {ANY}
    # a numeric write disables every reader, in that direction only ...
    assert graph.disables(tick_a, finish) and not graph.disables(finish, tick_a)
    # ... and clashes with every other writer, both ways
    assert graph.disables(tick_a, tick_b) and graph.disables(tick_b, tick_a)


def test_eager_interference_judges_every_pair_up_front():
    lazy, eager = _graph(pddl("gripper")), _graph(pddl("gripper"), "eager")
    n = len(set(eager.reads) | set(eager.writes))
    assert (lazy.edges_computed, eager.edges_computed) == (0, n * (n - 1))


def test_serialise_orders_a_step_topologically_and_rejects_a_cycle():
    graph = _graph(two_actions("a", "b", first_deletes_p=True))
    a, b = _action(graph, '"a"'), _action(graph, '"b"')
    assert graph.serialise([a, b]) == [b, a] == graph.serialise([b, a])
    cyclic = _graph(two_cycle_problem())
    with pytest.raises(ValueError, match="cannot be serialised"):
        cyclic.serialise([_action(cyclic, '"a"'), _action(cyclic, '"b"')])


# ---------------------------------------------------------------------------
# Options and the seq family
# ---------------------------------------------------------------------------

def test_bad_options_are_rejected():
    with pytest.raises(ValueError, match="Unsupported encoding family"):
        PLASPPlanner(robot_line_problem(), "sometimes")
    with pytest.raises(ValueError, match="interference must be"):
        PLASPPlanner(robot_line_problem(), "exists", interference="psychic")
    with pytest.raises(ValueError, match="semantics must be"):
        ParallelStepPropagator("maybe")


@pytest.mark.parametrize("make_problem", [
    robot_line_problem, lambda: pddl("gripper"), lambda: pddl("lift")])
def test_the_sequential_reading_still_puts_one_action_per_step(make_problem):
    planner, plan = solve(make_problem())
    assert planner.parallel is None
    assert planner.solved_horizon == len(plan.actions)
