"""Property-directed reachability over the seq encoding's one-step program.

See ``encodings/seq/pdr.lp`` for the program and the idea; this module is the
loop around it (after Suda, JAIR 2014, with IC3's obligation rescheduling).

Vocabulary. A *state* has a boolean part, the frozenset of boolean variables
(their ``V`` terms) that are true, and a numeric part, the frozenset of
``(V, X)`` pairs giving each numeric fluent's value. A *cube* is a partial
state: pairs ``(V, bool)`` over the booleans and ``(V, X)`` equalities over the
numerics; a state *satisfies* a cube when it agrees with every pair. A cube at
*level* i is blocked at every layer 1..i: no state satisfying it can reach the
goal within i steps. Layer B_i is therefore the set of cubes with level >= i,
which is what a query at level i blocks the successor with when it asks about
i-1.

One query, ``extend(s, i)``, asks: is there an action (doing nothing counts)
whose successor from s is a goal state (i = 1) or not blocked at i-1 (i > 1)?
SAT gives the action and the successor; UNSAT gives a reason, a cube that s
satisfies and that is blocked at i. The boolean part of the reason is clingo's
unsatisfiable core over the boolean state assumptions, shrunk further by trying
to drop each of its literals in turn; the numeric part is the state's numeric
values as they are.

Two backends answer the queries:

  * `IncrementalBackend`, for tasks without numeric fluents: ONE ground program
    for every query. The boolean state at time 0 is free and fixed by
    assumptions, each learned cube is a switch atom pinned on or off per query,
    and clingo keeps what it learned across queries.
  * `FreshBackend`, for numeric tasks: a program grounded PER QUERY. The
    numeric layer grounds a fluent's values by reachability from the atoms at
    time 0, so a query's numeric state has to be there as facts before the step
    is grounded; the booleans stay free and assumed, so boolean generalisation
    is unchanged and the reason's shrinking reuses the query's own program.
    The active cubes are plain constraints of that program. The price is one
    grounding per obligation extension and per push, and that numeric values
    never leave a reason: a cube pins every numeric fluent to the exact value
    the state had, so PDR generalises over the booleans only and enumerates
    numeric states. That is the general, no-bounds-needed version; how far it
    goes is what the benchmarks will say.

The main loop raises the target layer k = 1, 2, ... (bounded by the planner's
max_horizon). At each k it processes obligations from (initial state, k):
extend; on SAT keep the obligation and add (successor, i-1), on UNSAT block the
reason at i and, if i < k, re-add (s, i+1) so that a longer path from s is
still tried. A SAT answer at level 1 closes the plan: the chain of parent
obligations is the path. When the obligations run out, every cube is tried one
layer up ("push"); a layer left with no cube of its own level equals the next
one, which proves the task unsolvable: that layer is closed under successors,
contains the initial state and no goal state.

The rescheduling is what makes PDR fast here: without it every target layer
is a bounded search over plans of at most k steps that has to refute every
shorter length first, and the measured times go from seconds to timeouts.
Its price is twofold. Plans are not guaranteed shortest, since a rescheduled
obligation keeps the path it came with and the plan found at target layer k
can be longer than k. And on a finite state space the obligations of a layer
are finitely many, but with numeric values that never repeat (a counter that
only grows) they are not: so an obligation is only rescheduled while its path
length plus its new level stays within max_horizon, which bounds the length of
any plan explored and makes every layer terminate.
"""

import heapq
import itertools
import time
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import clingo

NOOP = clingo.Function('action', [clingo.String('pdrNoop')])
TRUE = clingo.Function('true')


def _fluent_name(variable: clingo.Symbol) -> str:
    """variable(("name", args...)) -> name."""
    inner = variable.arguments[0]
    return inner.string if inner.type == clingo.SymbolType.String else inner.arguments[0].string


class State:
    """A total state: the true booleans and the numeric values."""
    __slots__ = ('bools', 'nums')

    def __init__(self, bools: FrozenSet[clingo.Symbol], nums: FrozenSet[Tuple[clingo.Symbol, clingo.Symbol]] = frozenset()):
        self.bools, self.nums = bools, nums

    def __eq__(self, other):
        return self.bools == other.bools and self.nums == other.nums

    def __hash__(self):
        return hash((self.bools, self.nums))


class Cube:
    """A partial state: boolean literals and numeric equalities."""
    __slots__ = ('bools', 'nums')

    def __init__(self, bools: FrozenSet[Tuple[clingo.Symbol, bool]], nums: FrozenSet[Tuple[clingo.Symbol, clingo.Symbol]] = frozenset()):
        self.bools, self.nums = bools, nums

    def covers(self, state: State) -> bool:
        return all((v in state.bools) == positive for v, positive in self.bools) and self.nums <= state.nums

    def subsumes(self, other: 'Cube') -> bool:
        return self.bools <= other.bools and self.nums <= other.nums

    def body(self) -> str:
        """The cube's literals over the successor state at time 1."""
        parts = [(f'holds({v}, value({v}, true), 1)' if positive else f'not holds({v}, value({v}, true), 1)')
                 for v, positive in sorted(self.bools, key=lambda p: str(p[0]))]
        parts += [f'holds({v}, value({v}, {x}), 1)' for v, x in sorted(self.nums, key=lambda p: str(p[0]))]
        return ', '.join(parts)

    def __len__(self):
        return len(self.bools) + len(self.nums)


class _Obligation:
    __slots__ = ('level', 'state', 'parent', 'action', 'depth')

    def __init__(self, level, state, parent=None, action=None):
        self.level, self.state, self.parent, self.action = level, state, parent, action
        self.depth = 0 if parent is None else parent.depth + 1   # path length from the initial state

    def path(self) -> List[clingo.Symbol]:
        actions, node = [], self
        while node.parent is not None:
            actions.append(node.action)
            node = node.parent
        return list(reversed(actions))


# ----------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------

class _Backend:
    """What both backends share: reading the task off a ground program and
    turning a solve into ('sat', action, successor) | ('unsat', core) |
    ('timeout',)."""

    def __init__(self, make_control: Callable[[], clingo.Control], static_fluents: Set[str]):
        self.make_control = make_control
        self.static_fluents = static_fluents
        self.deadline: Optional[float] = None
        self.solves = 0
        self.groundings = 0
        self.query1 = clingo.Function('query', [clingo.Number(1)])
        ctl = self._ground(self.make_control())
        atoms = ctl.symbolic_atoms
        self.variables: List[clingo.Symbol] = sorted(
            (a.symbol.arguments[0] for a in atoms.by_signature('pdrVariable', 1)), key=str)
        self.numerics: List[clingo.Symbol] = sorted(
            (a.symbol.arguments[0] for a in atoms.by_signature('numVariable', 1)), key=str)
        known, numeric = set(self.variables), set(self.numerics)
        bools, nums = set(), set()
        for name in ('pdrInitialState', 'initialState'):
            for a in atoms.by_signature(name, 2):
                v, value = a.symbol.arguments
                if v in known and value.arguments[1] == TRUE:
                    bools.add(v)
                elif v in numeric:
                    nums.add((v, value.arguments[1]))
        self.initial = State(frozenset(bools), frozenset(nums))
        self.static: Set[clingo.Symbol] = {v for v in self.variables if _fluent_name(v) in static_fluents}
        self.free: List[clingo.Symbol] = [v for v in self.variables if v not in self.static]
        self._first = ctl

    def _ground(self, ctl: clingo.Control) -> clingo.Control:
        self.groundings += 1
        ctl.ground([('base', []), ('step', [clingo.Number(1)]), ('check', [clingo.Number(1)])])
        return ctl

    def _literals(self, ctl: clingo.Control) -> Dict[clingo.Symbol, int]:
        atoms = ctl.symbolic_atoms
        literals = {}
        for v in self.variables:
            atom = clingo.Function('holds', [v, clingo.Function('value', [v, TRUE]), clingo.Number(0)])
            literals[v] = atoms[atom].literal
        return literals

    def _run(self, ctl: clingo.Control, assumptions: Sequence[int], goal: bool, literals: Dict[clingo.Symbol, int]):
        self.solves += 1
        ctl.assign_external(self.query1, goal)
        numeric = set(self.numerics)
        found, core = {}, []

        def on_model(model):
            symbols = model.symbols(atoms=True)
            found['action'] = next(s.arguments[0] for s in symbols if s.match('occurs', 2))
            bools, nums = set(), set()
            for s in symbols:
                if s.match('holds', 3) and s.arguments[2].number == 1:
                    v, value = s.arguments[0], s.arguments[1].arguments[1]
                    if v in literals and value == TRUE:
                        bools.add(v)
                    elif v in numeric:
                        nums.add((v, value))
            found['successor'] = State(frozenset(bools), frozenset(nums))

        remaining = None if self.deadline is None else self.deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            return ('timeout',)
        with ctl.solve(assumptions=list(assumptions), on_model=on_model, async_=True) as handle:
            if not handle.wait(remaining):
                handle.cancel()
                return ('timeout',)
            result = handle.get()
            if not result.satisfiable:
                core = list(handle.core())   # on_core is not invoked for an asynchronous solve
        if result.satisfiable:
            return ('sat', found['action'], found['successor'])
        return ('unsat', core)

    @staticmethod
    def _bool_assumptions(literals: Dict[clingo.Symbol, int], pairs) -> List[int]:
        return [literals[v] if positive else -literals[v] for v, positive in pairs]

    def _static_pairs(self):
        return [(v, v in self.initial.bools) for v in self.static]

    def _reason(self, ctl, literals, pairs, nums, layer_assumptions, goal, core) -> Optional[Cube]:
        """Shrink the core over the boolean pairs by dropping literals while
        the query stays unsatisfiable, on the query's own program."""
        lit_pair = {literals[v]: (v, True) for v, _ in pairs}
        lit_pair.update({-literals[v]: (v, False) for v, _ in pairs})
        assumed = {literals[v] if positive else -literals[v] for v, positive in pairs}
        kept = [lit for lit in core if lit in assumed]
        statics = self._bool_assumptions(literals, self._static_pairs())
        i = 0
        while i < len(kept):
            trial = kept[:i] + kept[i + 1:]
            result = self._run(ctl, trial + statics + layer_assumptions, goal, literals)
            if result[0] == 'timeout':
                return None
            if result[0] == 'unsat':
                new_core = set(result[1])
                kept = [lit for lit in trial if lit in new_core] or trial
                i = min(i, len(kept))
            else:
                i += 1
        return Cube(frozenset(lit_pair[lit] for lit in kept), nums)

    # to implement
    def query(self, pairs, nums, layer: int, goal: bool, cubes: List[Cube], levels: List[int]):
        raise NotImplementedError

    def add_cube(self, cube: Cube) -> None:
        pass


class IncrementalBackend(_Backend):
    """One ground program; cubes are switch atoms pinned per query."""

    def __init__(self, make_control, static_fluents):
        super().__init__(make_control, static_fluents)
        self.ctl = self._first
        self.literals = self._literals(self.ctl)
        self.cube_literal: List[int] = []

    def add_cube(self, cube: Cube) -> None:
        index = len(self.cube_literal)
        # A free atom, not an external: an external is fixed false, and an
        # assumption cannot override that. Every query pins every switch.
        self.ctl.add(f'pdrcube{index}', [], f'cubeHolds({index}) :- {cube.body()}.\n'
                                            f'{{ cubeOn({index}) }}.\n'
                                            f':- cubeOn({index}), cubeHolds({index}).')
        self.ctl.ground([(f'pdrcube{index}', [])])
        self.cube_literal.append(self.ctl.symbolic_atoms[clingo.Function('cubeOn', [clingo.Number(index)])].literal)

    def query(self, pairs, nums, layer, goal, cubes, levels):
        switches = [lit if (layer >= 1 and lvl >= layer) else -lit
                    for lit, lvl in zip(self.cube_literal, levels)]
        assumptions = self._bool_assumptions(self.literals, pairs) \
            + self._bool_assumptions(self.literals, self._static_pairs()) + switches
        result = self._run(self.ctl, assumptions, goal, self.literals)
        if result[0] != 'unsat':
            return result
        reason = self._reason(self.ctl, self.literals, pairs, nums, switches, goal, result[1])
        return ('timeout',) if reason is None else ('unsat', reason)


class FreshBackend(_Backend):
    """A program per query: the numeric state as facts, the active cubes as
    constraints, the booleans free and assumed."""

    def query(self, pairs, nums, layer, goal, cubes, levels):
        ctl = self.make_control()
        facts = [f'holds({v}, value({v}, {x}), 0).' for v, x in nums]
        constraints = [f':- {cube.body()}.' for cube, lvl in zip(cubes, levels) if layer >= 1 and lvl >= layer]
        ctl.add('base', [], '\n'.join(facts + constraints))
        self._ground(ctl)
        literals = self._literals(ctl)
        assumptions = self._bool_assumptions(literals, pairs) + self._bool_assumptions(literals, self._static_pairs())
        result = self._run(ctl, assumptions, goal, literals)
        if result[0] != 'unsat':
            return result
        reason = self._reason(ctl, literals, pairs, nums, [], goal, result[1])
        return ('timeout',) if reason is None else ('unsat', reason)


# ----------------------------------------------------------------------
# The search
# ----------------------------------------------------------------------

class PDR:
    """The PDR search. `make_control` returns a Control with the task's layers
    and pdr.lp loaded and the task facts (initial state renamed) added, nothing
    grounded; it is called once for propositional tasks and once per query for
    numeric ones."""

    def __init__(self, make_control: Callable[[], clingo.Control], static_fluents: Set[str] = frozenset(),
                 numeric: bool = False):
        self.backend = (FreshBackend if numeric else IncrementalBackend)(make_control, static_fluents)
        self.initial = self.backend.initial
        self.cubes: List[Cube] = []
        self.levels: List[int] = []
        self.queries = 0

    # -- cubes ---------------------------------------------------------------

    def _add_cube(self, cube: Cube, level: int) -> None:
        for other, other_level in zip(self.cubes, self.levels):
            if other_level >= level and other.subsumes(cube):
                return
        self.cubes.append(cube)
        self.levels.append(level)
        self.backend.add_cube(cube)

    def _blocked(self, state: State, level: int) -> bool:
        return any(lvl >= level and cube.covers(state) for cube, lvl in zip(self.cubes, self.levels))

    # -- queries -------------------------------------------------------------

    def _query(self, pairs, nums, layer, goal):
        self.queries += 1
        return self.backend.query(pairs, nums, layer, goal, self.cubes, self.levels)

    def extend(self, state: State, level: int):
        """('sat', action, successor) | ('unsat', reason cube) | ('timeout',)."""
        pairs = [(v, v in state.bools) for v in self.backend.free]
        return self._query(pairs, state.nums, level - 1, level == 1)

    # -- the loop ------------------------------------------------------------

    def run(self, max_layers: int, deadline: Optional[float] = None):
        """('plan', [action symbols]) | ('unsolvable', layer) |
        ('exhausted', max_layers) | ('timeout', layer)."""
        self.backend.deadline = deadline
        first = self.extend(self.initial, 1)
        if first[0] == 'timeout':
            return ('timeout', 1)
        if first[0] == 'sat' and first[1] == NOOP:
            return ('plan', [])
        self.max_layers = max_layers
        for k in range(1, max_layers + 1):
            outcome = self._solve_layer(k)
            if outcome[0] != 'none':
                return outcome
            if self._push(k) == 'timeout':
                return ('timeout', k)
            for i in range(1, k + 1):
                if i not in self.levels:
                    return ('unsolvable', i)
        return ('exhausted', max_layers)

    def _solve_layer(self, k: int):
        counter = itertools.count()
        heap = [(k, next(counter), _Obligation(k, self.initial))]
        def reschedule(node, level):
            if level < k and node.depth + level + 1 <= self.max_layers:
                heapq.heappush(heap, (level + 1, next(counter), node))

        while heap:
            level, _, node = heapq.heappop(heap)
            if self._blocked(node.state, level):
                reschedule(node, level)
                continue
            result = self.extend(node.state, level)
            if result[0] == 'timeout':
                return ('timeout', k)
            if result[0] == 'sat':
                _, action, successor = result
                if level == 1:
                    return ('plan', [a for a in node.path() + [action] if a != NOOP])
                heapq.heappush(heap, (level, next(counter), node))
                heapq.heappush(heap, (level - 1, next(counter), _Obligation(level - 1, successor, node, action)))
                continue
            self._add_cube(result[1], level)
            reschedule(node, level)
        return ('none',)

    def _push(self, k: int) -> str:
        """Try every cube one layer up, lowest layers first. A cube's query
        fixes its own literals and leaves the rest free."""
        for level in range(1, k + 1):
            for index in range(len(self.cubes)):
                if self.levels[index] != level:
                    continue
                cube = self.cubes[index]
                result = self._query(list(cube.bools), cube.nums, level, False)
                if result[0] == 'timeout':
                    return 'timeout'
                if result[0] == 'unsat':
                    self.levels[index] = level + 1
        return 'done'

    def statistics(self) -> str:
        by_level: Dict[int, int] = {}
        for lvl in self.levels:
            by_level[lvl] = by_level.get(lvl, 0) + 1
        b = self.backend
        return (f"{self.queries} one-step queries, {b.solves} solves, {b.groundings} groundings, "
                f"{len(self.cubes)} cubes learned (per layer: {dict(sorted(by_level.items()))})")
