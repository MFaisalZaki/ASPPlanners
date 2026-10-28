"""Symmetry breaking for the seq encoding.

Two families of constraints, both stated as rules of
``encodings/seq/symmetry.lp`` over facts this module derives from the task,
and both *lex-leader* constraints: each symmetry class of plans keeps at
least the member that is smallest under clingo's own term order, so a task
keeps a plan at every horizon it had one at and the shortest horizon is
unchanged.

**Object symmetries** (Rintanen 2003, "Symmetry reduction for SAT
representations of transition systems"). Two objects of the same type are
interchangeable when swapping them everywhere leaves the initial state and
the goal unchanged and neither is named by an action schema as a constant.
The interchangeable objects of a type form classes on which the full
symmetric group acts (two transpositions that are automorphisms generate a
third), so any plan can be relabelled so that the objects of a class are
first used in a fixed order. The constraint says exactly that: object i+1
of a class may not be used before object i. The classes are
``symClass(Class, Index, constant("name"))`` facts, indexed in clingo's order
of the constant terms, and which objects an action mentions is stated per
schema as ``mentions(action(("pick", B, R, G)), B) :- action(...)`` rules.

**Plan permutation symmetries** (Long & Fox 2003). Two adjacent actions of
a sequential plan that are independent -- neither writes a variable the
other reads or writes -- can be swapped without changing the plan's effect,
so a plan is required to list such pairs in ascending term order. The
independence relation is computed by the layer itself from the action facts
(``interferes/2``, variable-level, which is coarser than the value-level
relation but needs no case analysis), and the constraint is a single rule
over ``occurs(A, t-1), occurs(B, t)``.

The two are sound together: of all the plans reachable from a given one by
relabelling and swapping, the lexicographically smallest satisfies both.
That argument needs the class index order to agree with clingo's term order
on the constants, which is why the classes are sorted by `clingo.Symbol`
comparison rather than by name.

Neither constraint is stated for a temporal task: a snap action's position
is coupled to its partner's by the duration, and a swap changes the
schedule, so the layer is not loaded there.
"""

from typing import Dict, FrozenSet, Iterable, List, Set, Tuple

import clingo
from unified_planning.model import Problem

from aspplanners.plasp.facts import asp_name, _head_term, _signature

MODES = ('off', 'objects', 'order', 'all')


def _canonical(f) -> tuple:
    """A hashable rendering of an FNode in which the arguments of commutative
    operators are sorted, so that substitution (which may reorder an `and`)
    compares equal to the original when it is the same formula."""
    if f.is_object_exp():
        return ('obj', f.object().name)
    if f.is_fluent_exp():
        return ('fl', f.fluent().name) + tuple(_canonical(a) for a in f.args)
    if f.is_constant():
        return ('c', str(f.constant_value()))
    if f.is_parameter_exp():
        return ('p', f.parameter().name)
    if f.is_variable_exp():
        return ('v', f.variable().name)
    args = tuple(_canonical(a) for a in f.args)
    if f.is_and() or f.is_or() or f.is_equals() or f.is_plus() or f.is_times() or f.is_iff():
        args = tuple(sorted(args))
    bound = tuple(v.name for v in f.variables()) if (f.is_forall() or f.is_exists()) else ()
    return (f.node_type.name, bound) + args


def _objects_in(f, found: Set[str]) -> None:
    """Collect the names of the objects an expression names as constants."""
    if f.is_object_exp():
        found.add(f.object().name)
    for a in f.args:
        _objects_in(a, found)


def _pinned_objects(problem: Problem) -> Set[str]:
    """Objects an action names outright (so no relabelling moves them)."""
    pinned: Set[str] = set()
    for action in problem.actions:
        conditions = getattr(action, 'conditions', None)
        if isinstance(conditions, dict):          # durative: interval -> conditions
            conditions = [c for group in conditions.values() for c in group]
        else:
            conditions = list(getattr(action, 'preconditions', []))
        for condition in conditions:
            _objects_in(condition, pinned)
        effects = action.effects
        if isinstance(effects, dict):             # durative: timing -> effects
            effects = [e for group in effects.values() for e in group]
        for eff in effects:
            _objects_in(eff.fluent, pinned)
            _objects_in(eff.value, pinned)
            _objects_in(eff.condition, pinned)
    return pinned


def _init_tuples(problem: Problem) -> Set[tuple]:
    """The initial state as (fluent, args, value) tuples with objects by name."""
    tuples = set()
    for fluent_exp, value in problem.initial_values.items():
        args = tuple(a.object().name if a.is_object_exp() else str(a) for a in fluent_exp.args)
        val = value.object().name if value.is_object_exp() else str(value)
        tuples.add((fluent_exp.fluent().name, args, val))
    return tuples


def _swap_tuple(tup, a: str, b: str) -> tuple:
    def s(x):
        return b if x == a else a if x == b else x
    fluent, args, val = tup
    return (fluent, tuple(s(x) for x in args), s(val))


def object_classes(problem: Problem) -> List[List[str]]:
    """The classes of interchangeable objects, each sorted in clingo's order
    of the ``constant("name")`` terms. Objects in no class are left out."""
    pinned = _pinned_objects(problem)
    objects = [o for o in problem.all_objects if o.name not in pinned]
    if len(objects) < 2:
        return []
    init = _init_tuples(problem)
    mentions: Dict[str, Set[tuple]] = {o.name: set() for o in objects}
    for tup in init:
        for name in set(tup[1]) | {tup[2]}:
            if name in mentions:
                mentions[name].add(tup)
    em = problem.environment.expression_manager
    sub = problem.environment.substituter
    goals = problem.goals
    goal_canon = frozenset(_canonical(g) for g in goals)

    def swapped_goals(a, b) -> FrozenSet[tuple]:
        subs = {em.ObjectExp(a): em.ObjectExp(b), em.ObjectExp(b): em.ObjectExp(a)}
        return frozenset(_canonical(sub.substitute(g, subs)) for g in goals)

    # A cheap signature first: the shape of an object's own initial facts with
    # itself blanked out. Objects whose signatures differ are never swappable;
    # ones whose signatures agree still get the exact check (`on(a, b)`).
    def signature(o):
        return (o.type.name if o.type is not None else None,
                frozenset(_swap_tuple(t, o.name, '*') for t in mentions[o.name]))
    buckets: Dict[tuple, List] = {}
    for o in objects:
        buckets.setdefault(signature(o), []).append(o)

    parent = {o.name: o.name for o in objects}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for members in buckets.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if find(a.name) == find(b.name):
                    continue
                affected = mentions[a.name] | mentions[b.name]
                if any(_swap_tuple(t, a.name, b.name) not in init for t in affected):
                    continue
                if swapped_goals(a, b) != goal_canon:
                    continue
                parent[find(a.name)] = find(b.name)

    classes: Dict[str, List[str]] = {}
    for o in objects:
        classes.setdefault(find(o.name), []).append(o.name)
    result = []
    for members in classes.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda n: clingo.Function('constant', [clingo.String(asp_name(n))]))
        result.append(members)
    result.sort(key=lambda m: m[0])
    return result


def mentions_rules(problem: Problem) -> List[str]:
    """``mentions(action(H), P) :- action(action(H)).`` per action parameter."""
    rules = []
    for action in problem.actions:
        signature = _signature(action.parameters)
        if not signature:
            continue
        head = f'action({_head_term(action.name, signature)})'
        for variable, _ in signature:
            rules.append(f'mentions({head}, {variable}) :- action({head}).')
    return rules


def symmetry_program(problem: Problem, mode: str) -> Tuple[str, List[List[str]]]:
    """The base-part program enabling the chosen constraints, and the object
    classes it states (empty when none were found or `mode` leaves them out)."""
    if mode not in MODES:
        raise ValueError(f"symmetry must be one of {MODES}, not {mode!r}")
    lines: List[str] = []
    classes: List[List[str]] = []
    if mode in ('objects', 'all'):
        classes = object_classes(problem)
        for c, members in enumerate(classes):
            for i, name in enumerate(members):
                lines.append(f'symClass({c}, {i}, constant("{asp_name(name)}")).')
        if classes:
            lines += mentions_rules(problem)
    if mode in ('order', 'all'):
        lines.append('symOrder.')
    return '\n'.join(lines), classes
