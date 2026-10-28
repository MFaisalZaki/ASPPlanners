"""Tests for the encoding-layer registry and its selection.

The encoding is split into one .lp per feature and only the layers a task needs
are handed to clingo. A layer that is loaded but not needed is inert; a layer
that is *needed* but not loaded is silently wrong -- clingo happily ignores facts
no rule reads -- so most of what is pinned here is the machinery that makes the
second case impossible.
"""

import glob
import os

import clingo
import pytest

from unified_planning.io import PDDLReader
from unified_planning.shortcuts import Equals, Or

from aspplanners.plasp.layers import SEQ, fact_predicates, parse_selection
from aspplanners.plasp.planner import PLASPPlanner

from test_planner import (
    counter_problem,
    disjunction_problem,
    numeric_counter_problem,
    robot_line_problem,
    rover_recharge_problem,
)

PDDL_DIR = os.path.join(os.path.dirname(__file__), "pddl")


def pddl(domain, problem="problem.pddl"):
    return PDDLReader().parse_problem(os.path.join(PDDL_DIR, domain, "domain.pddl"),
                                      os.path.join(PDDL_DIR, domain, problem))


# ---------------------------------------------------------------------------
# The registry itself
# ---------------------------------------------------------------------------

def test_layers_are_listed_in_dependency_order():
    """Load order is list order, so a layer must never precede one it requires.
    core is first because it declares the program parts and the query external."""
    seen = set()
    for layer in SEQ.layers:
        assert set(layer.requires) <= seen, f"{layer.name} precedes what it requires"
        seen.add(layer.name)
    assert SEQ.layer_names[0] == SEQ.base


def test_every_layer_file_exists():
    for name in SEQ.layer_names:
        assert os.path.isfile(SEQ.path(name)), name


def test_no_two_layers_claim_the_same_fact():
    """`owns` is what selection keys off; an overlap would make it ambiguous."""
    for i, a in enumerate(SEQ.layers):
        for b in SEQ.layers[i + 1:]:
            assert not (a.owns & b.owns), f"{a.name} and {b.name} both own {a.owns & b.owns}"


def test_fact_predicates_reads_the_head_of_a_guarded_fact():
    """Durative facts are rules, not plain facts -- only the head names the
    vocabulary, so the guard in the body must not be mistaken for one."""
    assert fact_predicates(["durativeAction(durative((\"a\"))) :- action((\"a_start\"))."]) \
        == {"durativeAction"}
    assert fact_predicates(["initialState(variable(x), value(x, true))."]) == {"initialState"}


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make_problem, expected", [
    (robot_line_problem,                       ("core",)),
    (numeric_counter_problem,                  ("core", "numeric")),
    (rover_recharge_problem,                   ("core", "numeric")),
    (disjunction_problem,                      ("core", "disjunctive")),
    (lambda: pddl("blocksworld"),              ("core",)),
    # matchcellar's matches are interchangeable, so it carries the symmetry layer
    (lambda: pddl("matchcellar"),              ("core", "symmetry", "temporal")),
    (lambda: pddl("arithdrive"),               ("core", "numeric", "temporal")),
])
def test_a_task_loads_exactly_the_layers_it_uses(make_problem, expected):
    assert PLASPPlanner(make_problem(), "seq").layers == expected


def test_a_classical_task_does_not_pay_for_the_numeric_or_temporal_layers():
    planner = PLASPPlanner(robot_line_problem(), "seq")
    assert planner.layers == ("core",)
    assert all(path.endswith("core.lp") for path in planner.encoding_paths)


def test_a_numeric_literal_in_a_disjunct_pulls_in_both_layers():
    """orDisjunctNum is a disjunctive fact evaluated against the numeric layer's
    numValue/3 -- a bridge rule, so its presence has to imply numeric as well."""
    problem = counter_problem(precondition=lambda c: Or(Equals(c, 0), Equals(c, 100)))
    assert PLASPPlanner(problem, "seq").layers == ("core", "numeric", "disjunctive")


# ---------------------------------------------------------------------------
# problem.kind as a second opinion
# ---------------------------------------------------------------------------

def test_kind_and_facts_agree_across_the_bundled_domains():
    """The kind mapping is hand-maintained and only advisory, but the two
    signals disagreeing means one of them is wrong; keep them honest."""
    for domain_file in sorted(glob.glob(os.path.join(PDDL_DIR, "*", "domain.pddl"))):
        directory = os.path.dirname(domain_file)
        for problem_file in sorted(glob.glob(os.path.join(directory, "problem*.pddl"))):
            problem = PDDLReader().parse_problem(domain_file, problem_file)
            planner = PLASPPlanner(problem, "seq")
            predicates = fact_predicates(planner.compiled_task.fact_lines)
            from_facts = SEQ.close(SEQ.layers_from_facts(predicates))
            from_kind = SEQ.close(SEQ.layers_from_kind(planner.task.kind))
            # The symmetry layer is the encoder's, not a feature ProblemKind
            # could know about: it is left out of the comparison.
            from_facts = tuple(name for name in from_facts if name != "symmetry")
            assert from_facts == from_kind, os.path.relpath(problem_file, PDDL_DIR)


def test_a_fluent_used_only_in_a_duration_still_selects_the_numeric_layer():
    """arithdrive's `distance`/`speed` appear in no condition and no effect, so
    ProblemKind reports no *_FLUENTS feature at all -- but the durative split
    turns those durations into numTerm facts only the numeric layer reads."""
    problem = pddl("arithdrive")
    planner = PLASPPlanner(problem, "seq")
    assert "REAL_FLUENTS" not in planner.task.kind.features
    assert "numeric" in SEQ.layers_from_kind(planner.task.kind)
    assert "numeric" in planner.layers


# ---------------------------------------------------------------------------
# Explicit specs
# ---------------------------------------------------------------------------

def test_parse_selection_splits_family_from_layers():
    assert parse_selection("seq") == ("seq", None)
    assert parse_selection("seq+numeric") == ("seq", ("numeric",))
    assert parse_selection("seq+temporal+numeric") == ("seq", ("temporal", "numeric"))


def test_an_explicit_spec_may_load_more_than_the_task_needs():
    """Useful for benchmarking a layer's grounding cost: the extra rules are
    inert, so the plan is unchanged."""
    planner = PLASPPlanner(robot_line_problem(), "seq+numeric+temporal+disjunctive")
    assert planner.layers == ("core", "numeric", "disjunctive", "temporal")
    assert len(planner.plan(max_horizon=6).actions) == 3


def test_an_explicit_spec_is_closed_under_requires():
    planner = PLASPPlanner(pddl("matchcellar"), "seq+temporal")
    assert planner.layers[0] == "core"


def test_an_explicit_spec_that_cannot_express_the_task_is_rejected():
    """The whole point of the coverage check: without it this program would
    solve, ignore every numeric precondition, and fail validation much later."""
    with pytest.raises(ValueError, match="do not cover the facts"):
        PLASPPlanner(rover_recharge_problem(), "seq+disjunctive")


def test_an_unknown_layer_is_rejected():
    with pytest.raises(ValueError, match="Unknown seq encoding layer"):
        PLASPPlanner(robot_line_problem(), "seq+quantum")


def test_an_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="Unsupported encoding family"):
        PLASPPlanner(robot_line_problem(), "parallel")


def test_the_registry_knows_every_fact_the_encoder_emits():
    """Drift guard: a new fact predicate in aspplanners.plasp.facts that no
    layer claims would otherwise be selected-for by nothing and silently
    ignored."""
    problems = [robot_line_problem(), numeric_counter_problem(), disjunction_problem(),
                rover_recharge_problem(), pddl("matchcellar"), pddl("arithdrive"),
                pddl("rover"), pddl("tempdrive"), pddl("transport")]
    for problem in problems:
        predicates = fact_predicates(PLASPPlanner(problem, "seq").compiled_task.fact_lines)
        assert SEQ.unregistered(predicates) == set(), problem.name


# ---------------------------------------------------------------------------
# ... and the other direction: the registry answers to the .lp files
# ---------------------------------------------------------------------------
#
# `test_the_registry_knows_every_fact_the_encoder_emits` checks facts against
# the registry. Nothing checked the registry against the *rules*, so a claim of
# ownership was self-certifying -- and that is the direction that can be
# silently wrong. If layer A claims a predicate whose consuming rule actually
# lives in layer B, emitting the fact selects A, B stays unloaded, and B's
# constraint quietly disappears from the program: exactly the failure the module
# docstring says the safety net exists to prevent.

def _rule_predicates(layer_name):
    """The predicate names the *rules* of one layer's .lp mention.

    Directives are the only thing skipped, and on purpose: `#defined p/1.`
    declares that `p` may be absent, which is not the same as reading it, and
    counting it would let a layer certify itself with a declaration. Everything
    else counts -- including the statements `lp_io` calls facts, because a
    choice rule with no body (`1 {occurs(A,t) : action(A)} 1.`) is one of those
    and it is where core reads `action`.
    """
    from aspplanners.lp_io import parse_lp_file, ASPDirective
    import re
    names = set()
    for statement in parse_lp_file(SEQ.path(layer_name)):
        if isinstance(statement, ASPDirective):
            continue
        names.update(re.findall(r"([a-z_][A-Za-z0-9_]*)\s*\(", str(statement)))
    return names


@pytest.mark.parametrize("layer", SEQ.layers, ids=lambda layer: layer.name)
def test_every_owned_predicate_is_read_by_a_rule_of_its_own_layer(layer):
    """A layer owns a fact because its rules consume it. The exception is the
    vocabulary that describes the instance itself, which is declared as such."""
    consumed = _rule_predicates(layer.name)
    claimed = layer.owns - layer.describes_instance
    unread = sorted(claimed - consumed)
    assert unread == [], (
        f"{layer.name} claims {unread}, but no rule in {layer.filename} reads them. "
        f"Either a rule moved to another layer -- in which case `owns` has to move "
        f"with it, or selecting this layer will not load the rule -- or the "
        f"predicate describes the instance and belongs in `describes_instance`.")


@pytest.mark.parametrize("layer", SEQ.layers, ids=lambda layer: layer.name)
def test_instance_vocabulary_is_a_subset_of_what_the_layer_owns(layer):
    """`describes_instance` annotates `owns`; it never adds to it. A predicate
    outside `owns` would escape `unregistered` and be silently ignored."""
    assert layer.describes_instance <= layer.owns


def test_the_instance_vocabulary_is_what_the_problem_states():
    """Pinned so that adding to this set is a deliberate act.

    These come from the problem instance -- its types, its objects, its fluents,
    the boolean domain -- rather than from the encoding, which is why no rule
    reads them and why that is not drift.
    """
    assert SEQ.by_name["core"].describes_instance == {
        "type", "constant", "has", "variable", "boolean"}
    assert all(not layer.describes_instance
               for layer in SEQ.layers if layer.name != "core")


# ---------------------------------------------------------------------------
# Loading the layers and concatenating them have to be the same program
# ---------------------------------------------------------------------------

def _ground(control, horizon):
    control.ground([("base", []), ("check", [clingo.Number(horizon)])]
                   + [("step", [clingo.Number(t)]) for t in range(1, horizon + 1)])
    return sorted(str(atom.symbol) for atom in control.symbolic_atoms)


def _statements_by_part(text):
    """Every statement of an .lp text paired with the #program part it lands in.

    clingo's parser opens each parse with an implicit `#program base.`, which is
    why loading N files resets the part N times but concatenating them into one
    text resets it once.
    """
    from clingo import ast as clingo_ast

    pairs, part = [], "base"

    def collect(node):
        nonlocal part
        rendered = str(node)
        if rendered.startswith("%"):
            return
        if rendered.startswith("#program"):
            part = rendered[len("#program"):].strip().rstrip(".").strip()
            return
        pairs.append((part, rendered))

    clingo_ast.parse_string(text, collect)
    return pairs


def test_concatenating_the_layers_preserves_every_program_part():
    """The invariant `lp_program()` rests on, and the reason every layer file
    opens with an explicit `#program base.`: without that pin the statements
    above a file's first `#program` would inherit the part its predecessor ended
    in, which is `check(t)` for all four layers."""
    texts = [open(SEQ.path(name)).read() for name in SEQ.layer_names]

    separate = [pair for text in texts for pair in _statements_by_part(text)]
    concatenated = _statements_by_part("\n".join(texts))

    assert sorted(separate) == sorted(concatenated)


@pytest.mark.parametrize("make_problem", [robot_line_problem, lambda: pddl("matchcellar")])
def test_loading_the_layers_equals_concatenating_them(make_problem, tmp_path):
    """The same invariant end to end: the ground program must not depend on
    whether clingo was handed the layers as files or as one concatenated text."""
    planner = PLASPPlanner(make_problem(), "seq")

    loaded = clingo.Control(["-n", "1"])
    for path in planner.encoding_paths:
        loaded.load(path)
    loaded.add("base", [], planner.task_facts)

    concatenated_file = tmp_path / "program.lp"
    planner.dump_lp_program(concatenated_file)
    concatenated = clingo.Control(["-n", "1"])
    concatenated.load(str(concatenated_file))

    assert _ground(loaded, 3) == _ground(concatenated, 3)


def test_the_dumped_program_is_still_a_standalone_clingo_input(tmp_path):
    """`clingo program.lp` on the dump has to stay a working debugging path."""
    planner = PLASPPlanner(robot_line_problem(), "seq")
    out = tmp_path / "program.lp"
    planner.dump_lp_program(out)

    control = clingo.Control(["-n", "1"])
    control.load(str(out))
    _ground(control, 3)
    control.assign_external(clingo.Function("query", [clingo.Number(3)]), True)

    models = []
    control.solve(on_model=lambda m: models.append(m.symbols(shown=True)))
    assert models, "the dumped program should solve on its own"
    assert sum(1 for s in models[-1] if s.match("occurs", 2)) == 3
