"""Object symmetry breaking (Rintanen 2003, "Symmetry reduction for SAT
representations of transition systems"), always on.

Two objects of the same type are interchangeable when swapping them everywhere
leaves the initial state and the goal unchanged and neither is named by an
action schema as a constant. The interchangeable objects of a type form
classes on which the full symmetric group acts (two transpositions that are
automorphisms generate a third), so any plan can be relabelled so that the
objects of a class are first used in a fixed order -- and the deepening then
proves each unsatisfiable horizon once instead of once per relabelling. The
constraint says exactly that: object i+1 of a class may not be used before
object i. It is a lex-leader constraint under clingo's own term order (the
classes are indexed in the order of their ``constant("name")`` terms), so the
term-smallest plan of every class survives and the shortest horizon is
unchanged.

The encoder states the classes as ``symClass(Class, Index, constant("name"))``
facts and, for the seq family, which objects an action mentions as one
``mentions(action(("pick", B, R, G)), B) :- action(...)`` rule per schema
parameter; the ``symmetry`` layer of each family (``encodings/*/symmetry.lp``)
reads them. The lifted family reads the objects off ``arg/4`` instead. A
temporal task is covered too: a relabelled schedule is a schedule with the
same happenings. PDR leaves the layer out, since its one-step program starts
from an arbitrary state where "first use" has no meaning.
"""

from typing import Dict, FrozenSet, Iterable, List, Set, Tuple

import clingo
from unified_planning.model import Problem

from aspplanners.plasp.facts import asp_name, _head_term, _signature

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


def mentions_rules(actions: Iterable) -> List[str]:
    """``mentions(action(H), P) :- action(action(H)).`` per parameter of each
    of `actions` -- the actions the facts declare, which on a temporal task
    are the snap actions rather than the problem's durative ones."""
    rules = []
    for action in actions:
        signature = _signature(action.parameters)
        if not signature:
            continue
        head = f'action({_head_term(action.name, signature)})'
        for variable, _ in signature:
            rules.append(f'mentions({head}, {variable}) :- action({head}).')
    return rules


def symmetry_facts(problem: Problem, actions: Iterable) -> Tuple[List[str], List[List[str]]]:
    """The fact lines of the symmetry layer for `problem` (classes, and the
    mentions rules over `actions`), and the classes themselves. Empty when
    no two objects are interchangeable."""
    classes = object_classes(problem)
    if not classes:
        return [], []
    lines: List[str] = []
    for c, members in enumerate(classes):
        for i, name in enumerate(members):
            lines.append(f'symClass({c}, {i}, constant("{asp_name(name)}")).')
    lines += mentions_rules(actions)
    return lines, classes
