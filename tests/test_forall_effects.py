"""``forall`` effects: the ADL shape the encoding expands in the grounder.

A ``forall`` effect is not compiled away, on the same terms as a ``forall``
*condition*: its variables are emitted free and the grounder ranges them over
the universe. The load-bearing part is that a *conditional* one is one effect
per binding -- the encoding's ``caused/3`` fires an effect only when every
``precondition`` of its effect term holds, so the term has to carry the
bindings. Sharing one term across them would encode a different task: "fire
when every object satisfies the condition" instead of "fire for each object
that does".

The behavioural tests below are all stated as goals a wrong reading cannot
reach, and every plan is validated against the original problem by UP's
validator, which expands the quantifier independently of this encoder.
"""

import os

import pytest

import aspplanners  # noqa: F401 -- registers both engines
from unified_planning.engines import PlanGenerationResultStatus as Status
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import And, Equals, GE, LE, Not, OneshotPlanner, Or
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.model import ProblemKind

from aspplanners.plasp.facts import QUANTIFIED_PREFIX
from aspplanners.plasp.planner import PLASPPlanner
from aspplanners.up_engines import UPABAPlanner, UPPLASPPlanner
from test_planner import assert_plan_is_over_original_problem

PDDL_DIR = os.path.join(os.path.dirname(__file__), "pddl")


def parse_case(domain, problem_file):
    return PDDLReader().parse_problem(
        os.path.join(PDDL_DIR, domain, "domain.pddl"),
        os.path.join(PDDL_DIR, domain, problem_file),
    )


# (case id, problem file, optimal plan length, the action names it is made of)
CASES = [
    # Only `d` is put in, and `k` must stay home: the effect has to fire per
    # binding of ?o, not once for all of them.
    ("carry-only-what-is-inside", "problem-carry.pddl", 2,
     ["put-in", "move"]),
    # `k` travels because it is glued -- the disjunctive arm of the condition
    # under the quantifier, whose orGroup facts are keyed by the effect term.
    ("disjunctive-condition-under-forall", "problem-glued.pddl", 2,
     ["put-in", "move"]),
    # One `tax` raises every object's weight: an unconditional numeric forall.
    ("numeric-forall-effect", "problem-tax.pddl", 3,
     ["tax", "put-in", "move"]),
]


@pytest.mark.parametrize("case_id,problem_file,optimal_length,action_names",
                         CASES, ids=[c[0] for c in CASES])
def test_forall_effect_instances(case_id, problem_file, optimal_length, action_names):
    task = parse_case("briefcase", problem_file)
    assert task.kind.has_forall_effects(), f"{case_id}: fixture lost its forall effect"
    with OneshotPlanner(name="PLASPPlanner") as planner:
        result = planner.solve(task)
    assert result.status == Status.SOLVED_SATISFICING, (
        f"{case_id}: expected a plan, got {result.status} "
        f"({[str(m) for m in result.log_messages or []]})"
    )
    assert len(result.plan.actions) == optimal_length, (
        f"{case_id}: expected the {optimal_length}-step optimum, got "
        f"{[str(a) for a in result.plan.actions]}"
    )
    assert sorted(ai.action.name for ai in result.plan.actions) == sorted(action_names)
    assert_plan_is_over_original_problem(task, result.plan)


def test_both_engines_declare_forall_effects():
    """A task whose only extra feature is FORALL_EFFECTS is accepted up front.

    The ABA backend takes it through its pipeline's quantifiers remover rather
    than natively, so a *conditional* forall stays refused there -- for
    CONDITIONAL_EFFECTS, which is what it expands into.
    """
    kind = ProblemKind(version=LATEST_PROBLEM_KIND_VERSION)
    kind.set_problem_class("ACTION_BASED")
    kind.set_typing("FLAT_TYPING")
    kind.set_effects_kind("FORALL_EFFECTS")
    assert UPPLASPPlanner.supports(kind)
    assert UPABAPlanner.supports(kind)

    kind.set_effects_kind("CONDITIONAL_EFFECTS")
    assert UPPLASPPlanner.supports(kind)
    assert not UPABAPlanner.supports(kind)


def test_the_effect_term_carries_the_quantified_variable():
    """Each binding of the quantifier gets its own effect term, and every rule
    naming that term ranges the variable in its body (an unranged one is an
    unsafe variable, which clingo rejects outright)."""
    task = parse_case("briefcase", "problem-carry.pddl")
    facts = PLASPPlanner(task).task_facts.splitlines()

    conditional = [f for f in facts if 'effect((cond,"move"' in f]
    assert conditional, "the conditional forall effect emitted no facts"
    for fact in conditional:
        assert f'{QUANTIFIED_PREFIX}O)' in fact.split(":-")[0], (
            f"the effect term does not index by the quantified variable: {fact}")

    for fact in facts:
        if QUANTIFIED_PREFIX not in fact:
            continue
        head, _, body = fact.partition(":-")
        assert body, f"quantified variable left unbound (unsafe rule): {fact}"
        for name in {token for token in _tokens(head) if token.startswith(QUANTIFIED_PREFIX)}:
            assert f"has({name}," in body, (
                f"{name} is not ranged over the universe in the body of: {fact}")


def _tokens(text):
    token, tokens = "", []
    for char in text:
        if char.isalnum() or char == "_":
            token += char
        else:
            tokens.append(token)
            token = ""
    tokens.append(token)
    return [t for t in tokens if t]


def test_a_quantified_variable_only_in_the_condition():
    """``forall ?o . in(?o) -> alarm`` is ``(exists ?o . in(?o)) -> alarm``:
    the effect fires once any binding satisfies it, even though the fluent it
    assigns does not mention the variable at all."""
    task = parse_case("briefcase", "problem-carry.pddl")
    # (forall (?o) (when (in ?o) (at-b office))) -- a variable the effect's
    # fluent does not carry still has to be safe in the rule and still has to
    # tell the bindings' effect terms apart.
    move = task.action("move")
    quantified = next(e for e in move.effects if e.is_forall()).forall
    move.add_effect(task.fluent("at-b")(task.object("office")), True,
                    condition=task.fluent("in")(quantified[0]),
                    forall=quantified)
    with OneshotPlanner(name="PLASPPlanner") as planner:
        result = planner.solve(task)
    assert result.status == Status.SOLVED_SATISFICING
    assert_plan_is_over_original_problem(task, result.plan)


def test_a_forall_effect_on_a_durative_action():
    """The durative split copies effects into the snap actions with their
    quantifier intact, so a temporal task encodes the same way."""
    task = parse_case("tbriefcase", "problem.pddl")
    assert task.kind.has_forall_effects()
    with OneshotPlanner(name="PLASPPlanner") as planner:
        result = planner.solve(task)
    assert result.status == Status.SOLVED_SATISFICING, (
        f"{[str(m) for m in result.log_messages or []]}")
    assert [ai.action.name for _start, ai, _dur in result.plan.timed_actions] == \
        ["put-in", "move"]


def test_a_conditional_clear_survives_an_add_of_the_same_fluent():
    """The miconic shape: one action sets and clears `boarded` through two
    `forall`/`when` effects. Only an add that *always* fires shadows a delete,
    so the clear has to survive -- dropped, every passenger stays aboard and
    the lift can never move again, which reads back as an unsolvable task."""
    task = parse_case("lift", "problem.pddl")
    stop = task.action("stop")
    kept = PLASPPlanner(task).task.action("stop").effects
    assert len(kept) == len(stop.effects), (
        f"the delete-then-set pass dropped a conditional effect: "
        f"{[str(e) for e in stop.effects]} -> {[str(e) for e in kept]}")

    with OneshotPlanner(name="PLASPPlanner") as planner:
        result = planner.solve(task)
    assert result.status == Status.SOLVED_SATISFICING, (
        f"{[str(m) for m in result.log_messages or []]}")
    assert [ai.action.name for ai in result.plan.actions] == \
        ["stop", "up", "stop", "down", "stop"]
    assert_plan_is_over_original_problem(task, result.plan)


def test_an_add_wins_over_a_delete_at_the_same_step():
    """Both surviving effects can still fire together, and PDDL says the add
    wins. The encoding arbitrates it, so the fluent takes one value -- without
    that the step would hold true *and* false, which is not a state."""
    task = parse_case("lift", "problem.pddl")
    stop = task.action("stop")
    quantified = stop.effects[0].forall[0]
    # `served ?p := false` alongside the existing `served ?p := true`, under the
    # very same condition, so the two race for every binding that fires.
    stop.add_effect(task.fluent("served")(quantified), False,
                    condition=stop.effects[-1].condition, forall=(quantified,))
    with OneshotPlanner(name="PLASPPlanner") as planner:
        result = planner.solve(task)
    assert result.status == Status.SOLVED_SATISFICING, (
        "the add lost the race, so nothing was ever served: "
        f"{[str(m) for m in result.log_messages or []]}")
    assert_plan_is_over_original_problem(task, result.plan)


def test_an_equality_gates_the_postcondition_not_a_disjunct():
    """An equality in an effect's condition gates the effect by withholding its
    `postcondition`. It has to land on that rule even when a disjunction in the
    same condition was emitted first -- appending it to whatever rule came last
    would put it on an `orDisjunct`, over-restricting one disjunct instead."""
    task = parse_case("briefcase", "problem-carry.pddl")
    move = task.action("move")
    quantified = next(e for e in move.effects if e.is_forall()).forall[0]
    move.add_effect(
        task.fluent("at-o")(quantified, move.parameter("to")), True,
        condition=And(Or(task.fluent("in")(quantified), task.fluent("glued")(quantified)),
                      Not(Equals(quantified, task.object("d")))),
        forall=(quantified,))

    test = f'{QUANTIFIED_PREFIX}O != constant("d")'
    carrying = [f for f in PLASPPlanner(task).task_facts.splitlines() if test in f]
    assert len(carrying) == 1, f"the equality landed on {len(carrying)} rules, not 1"
    assert carrying[0].startswith("postcondition("), (
        f"the equality gates the wrong rule: {carrying[0]}")


def test_a_conditional_numeric_effect_is_encoded_as_a_gated_delta():
    """An increase carries its condition on the numeric layer's own vocabulary,
    not on ``postcondition``: the delta is collected when the effect term fires
    and applied to the fluent's previous value, so it is a *delta* rather than
    (as an assignment of it would be) the new value."""
    task = parse_case("briefcase", "problem-carry.pddl")
    tax = task.action("tax")
    tax.clear_effects()
    tax.add_increase_effect(task.fluent("weight")(task.object("d")), 2,
                            condition=task.fluent("in")(task.object("d")))
    planner = PLASPPlanner(task)
    facts = planner.task_facts
    assert 'numCondEffect(action(("tax")), effect((cond,"tax",0)), ' \
           'variable(("weight",constant("d"))), 2)' in facts
    # The condition rides on the effect term, exactly as a boolean one does.
    assert 'precondition(effect((cond,"tax",0)), variable(("in",constant("d"))), ' \
           'value(variable(("in",constant("d"))), true))' in facts
    assert planner.compilationlist == []


def test_a_forall_conditional_numeric_effect_fires_per_binding():
    """The hard shape twice over: a numeric delta, gated, under a quantifier.
    `tax` raises the weight of the objects that are *inside* the briefcase, so
    the effect has to fire once per binding that satisfies its condition -- and
    the delta has to be collected per binding, which is what indexing the effect
    term by the quantified variable buys on the numeric side as well."""
    task = parse_case("briefcase", "problem-tax.pddl")
    tax = task.action("tax")
    quantified = next(e for e in tax.effects if e.is_forall()).forall
    tax.clear_effects()
    tax.add_increase_effect(task.fluent("weight")(quantified[0]), 2,
                            condition=task.fluent("in")(quantified[0]),
                            forall=quantified)
    task.clear_goals()
    # `d` goes in, `k` never does, so one `tax` must move d's weight and not k's.
    task.add_goal(task.fluent("at-o")(task.object("d"), task.object("office")))
    task.add_goal(GE(task.fluent("weight")(task.object("d")), 2))
    task.add_goal(LE(task.fluent("weight")(task.object("k")), 0))

    planner = PLASPPlanner(task)
    assert any(line.startswith("numCondEffect(") and f"{QUANTIFIED_PREFIX}O)" in line
               for line in planner.task_facts.splitlines()), \
        "the delta is not indexed by the quantified variable"

    plan = planner.plan(max_horizon=6)
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert sorted(ai.action.name for ai in plan.actions) == ["move", "put-in", "tax"]
    assert_plan_is_over_original_problem(task, plan)


def test_two_equal_deltas_on_one_fluent_are_two_elements_of_the_sum():
    """`forall ?o . w(?o) += 2` alongside `w(d) += 2` is +4 on `d`, and the two
    are only distinguishable in the sum by the effect term they carry: keyed by
    the delta alone they would be one element and count once."""
    task = parse_case("briefcase", "problem-tax.pddl")
    tax = task.action("tax")
    weight, d = task.fluent("weight"), task.object("d")
    tax.add_increase_effect(weight(d), 2)
    # A conditional delta anywhere in the action is what routes all of its
    # deltas through the sum; without one they stay on numEffect.
    tax.add_increase_effect(weight(d), 0, condition=task.fluent("in")(d))
    task.clear_goals()
    task.add_goal(GE(weight(d), 4))

    planner = PLASPPlanner(task)
    plan = planner.plan(max_horizon=4)
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["tax"], (
        "one tax should be worth 4 on d")
    assert_plan_is_over_original_problem(task, plan)


def test_a_conditional_constant_assignment_is_still_encoded():
    """The numeric shape the postcondition path *can* state: a constant
    assignment travels on the same holds/3 chain a boolean value does."""
    task = parse_case("briefcase", "problem-carry.pddl")
    tax = task.action("tax")
    tax.clear_effects()
    tax.add_effect(task.fluent("weight")(task.object("d")), 5,
                   condition=task.fluent("in")(task.object("d")))
    facts = PLASPPlanner(task).task_facts
    assert 'postcondition(action(("tax")), effect((cond,"tax",0)), ' \
           'variable(("weight",constant("d"))), ' \
           'value(variable(("weight",constant("d"))), 5))' in facts


def test_a_universally_quantified_disjunct_over_an_empty_type_holds_vacuously():
    """elevators-00-full's `stop` has the precondition
    ``(or (forall (?p - vip) (served ?p)) (exists (?p - vip) ...))``; with no
    vip in the instance the first disjunct is vacuously true. It used to be
    declared under the vip binding and so vanished with the type, leaving the
    group without a disjunct: `stop` was never applicable and the sweep
    reported 21 solvable tasks as unsolvable."""
    import os
    from unified_planning.io import PDDLReader
    from unified_planning.engines import PlanGenerationResultStatus
    from aspplanners.plasp.planner import PLASPPlanner
    here = os.path.join(os.path.dirname(__file__), "pddl", "elevators-00-full")
    problem = PDDLReader().parse_problem(os.path.join(here, "domain.pddl"), os.path.join(here, "problem.pddl"))
    planner = PLASPPlanner(problem)
    plan = planner.plan(timeout=120)
    assert planner.status == PlanGenerationResultStatus.SOLVED_SATISFICING, planner.logs
    assert planner.solved_horizon == 4 and planner.validate(plan)[0]
    planner = PLASPPlanner(problem)
    plan = planner.plan(search="pdr", timeout=120)
    assert planner.status == PlanGenerationResultStatus.SOLVED_SATISFICING, planner.logs
    assert planner.validate(plan)[0]
