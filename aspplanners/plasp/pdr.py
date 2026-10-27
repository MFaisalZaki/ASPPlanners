"""Property-directed reachability over the seq encoding's one-step program.

See ``encodings/seq/pdr.lp`` for the program and the idea; this module is the
loop around it (after Suda, JAIR 2014, with IC3's obligation rescheduling).
Classical tasks only: see the planner's `_check_pdr` for why numeric fluents
are out.

Vocabulary. A *state* is the frozenset of boolean variables (their ``V``
terms) that are true. A *cube* is a partial state, a frozenset of
``(V, bool)`` pairs; a state *satisfies* a cube when it agrees with every pair.
A cube at *level* i is blocked at every layer 1..i: no state satisfying it can
reach the goal within i steps. Layer B_i is therefore the set of cubes with
level >= i, which is what a query at level i switches on when it asks whether
a successor is blocked at i-1.

One query, ``extend(s, i)``, asks: is there an action (doing nothing counts)
whose successor from s is a goal state (i = 1) or not blocked at i-1 (i > 1)?
It is one solve of the single ground program with the state fixed by
assumptions and the right cube switches pinned. SAT gives the action and the
successor; UNSAT gives clingo's core over the state assumptions, which is a
reason: any state agreeing with it is blocked at i. The reason is shrunk
further by trying to drop each of its literals in turn (each try is one more
cheap solve on the same program). Static fluents are pinned to their initial
value in every query and never part of a reason.

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
shorter length first, and the measured times go from seconds to timeouts. Its
price is that plans are not guaranteed shortest: a rescheduled obligation
keeps the path it came with, so the plan found at target layer k can be longer
than k. An obligation is rescheduled only while its path length plus its new
level stays within max_horizon, which bounds the length of any plan explored.
"""

import heapq
import itertools
import time
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import clingo

Cube = FrozenSet[Tuple[clingo.Symbol, bool]]
State = FrozenSet[clingo.Symbol]

NOOP_NAME = 'pdrNoop'
TRUE = clingo.Function('true')


def read_seq_action(symbols):
    """The action of a one-step model of the seq family, as a (name, objects...)
    tuple, or None for the do-nothing action."""
    occurs = next(s for s in symbols if s.match('occurs', 2))
    term = occurs.arguments[0].arguments[0]
    if term.type == clingo.SymbolType.String:
        return None if term.string == NOOP_NAME else (term.string,)
    name = term.arguments[0].string
    args = [a.arguments[0].string if a.type == clingo.SymbolType.Function and a.name == 'constant'
            else (a.string if a.type == clingo.SymbolType.String else str(a))
            for a in term.arguments[1:]]
    return (name, *args)


def read_lifted_action(symbols):
    """The action of a one-step model of the lifted family: the schema chosen
    at step 1 with its arguments by position, or None for the do-nothing one."""
    name, args = None, {}
    for s in symbols:
        if s.match('schema', 2):
            name = s.arguments[0].string
        elif s.match('arg', 4):
            args[s.arguments[1].number] = s.arguments[2].arguments[0].string
    if name == NOOP_NAME:
        return None
    return (name, *[args[i] for i in sorted(args)])


def _fluent_name(variable: clingo.Symbol) -> str:
    """variable(("name", args...)) -> name."""
    inner = variable.arguments[0]
    return inner.string if inner.type == clingo.SymbolType.String else inner.arguments[0].string


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


class PDR:
    """The PDR search over a planner's one-step program.

    `ctl` is a Control with the task's layers and pdr.lp loaded and the task
    facts (initial state renamed, statics kept) added, nothing grounded yet.
    `read_action` turns a one-step model into a (name, objects...) tuple, or
    None for the do-nothing action: `read_seq_action` or `read_lifted_action`.
    """

    def __init__(self, ctl: clingo.Control, static_fluents: Set[str] = frozenset(),
                 read_action: Callable = read_seq_action):
        self.ctl = ctl
        self.read_action = read_action
        ctl.ground([('base', []), ('step', [clingo.Number(1)]), ('check', [clingo.Number(1)])])
        atoms = ctl.symbolic_atoms
        self.variables: List[clingo.Symbol] = sorted(
            (a.symbol.arguments[0] for a in atoms.by_signature('pdrVariable', 1)), key=str)
        known = set(self.variables)
        self.initial: State = frozenset(
            a.symbol.arguments[0]
            for name in ('pdrInitialState', 'initialState')
            for a in atoms.by_signature(name, 2)
            if a.symbol.arguments[1].arguments[1] == TRUE and a.symbol.arguments[0] in known)
        # Static fluents never change: pinned to their initial value in every
        # query (so a cube's free variables never include them) and never part
        # of a reason.
        self.static: Set[clingo.Symbol] = {v for v in self.variables if _fluent_name(v) in static_fluents}
        self.free: List[clingo.Symbol] = [v for v in self.variables if v not in self.static]
        # V -> program literal of holds(V, value(V, true), 0); the assumption
        # for "V false" is its negation.
        self.state_literal: Dict[clingo.Symbol, int] = {}
        for v in self.variables:
            atom = clingo.Function('holds', [v, clingo.Function('value', [v, TRUE]), clingo.Number(0)])
            self.state_literal[v] = atoms[atom].literal
        self.literal_variable: Dict[int, clingo.Symbol] = {lit: v for v, lit in self.state_literal.items()}
        self.static_assumptions: List[int] = [
            self.state_literal[v] if v in self.initial else -self.state_literal[v] for v in self.static]
        self.query1 = clingo.Function('query', [clingo.Number(1)])
        self.cubes: List[Cube] = []
        self.levels: List[int] = []
        self.cube_literal: List[int] = []
        self.queries = 0
        self.solves = 0
        self.deadline: Optional[float] = None
        self.max_layers = 0

    # ------------------------------------------------------------------
    # Cubes
    # ------------------------------------------------------------------

    def _add_cube(self, cube: Cube, level: int) -> None:
        """Block `cube` at every layer up to `level`, unless a cube already
        blocked that high subsumes it."""
        for other, other_level in zip(self.cubes, self.levels):
            if other_level >= level and other <= cube:
                return
        index = len(self.cubes)
        body = ', '.join(
            (f'holds({v}, value({v}, true), 1)' if positive else f'not holds({v}, value({v}, true), 1)')
            for v, positive in sorted(cube, key=lambda p: str(p[0])))
        # cubeOn is a free atom, not an external: an external is fixed false,
        # and assuming it true would contradict that at decision level zero.
        # Every query pins every cube's switch by assumption, on or off.
        self.ctl.add(f'pdrcube{index}', [], f'cubeHolds({index}) :- {body}.\n'
                                            f'{{ cubeOn({index}) }}.\n'
                                            f':- cubeOn({index}), cubeHolds({index}).')
        self.ctl.ground([(f'pdrcube{index}', [])])
        self.cubes.append(cube)
        self.levels.append(level)
        self.cube_literal.append(self.ctl.symbolic_atoms[clingo.Function('cubeOn', [clingo.Number(index)])].literal)

    def _blocked(self, state: State, level: int) -> bool:
        return any(lvl >= level and all((v in state) == positive for v, positive in cube)
                   for cube, lvl in zip(self.cubes, self.levels))

    def _layer_assumptions(self, level: int) -> List[int]:
        """Every cube's switch: on for the cubes of layer `level` (level >= 1),
        off for the rest -- and off for all of them at level 0."""
        return [lit if (level >= 1 and lvl >= level) else -lit
                for lit, lvl in zip(self.cube_literal, self.levels)]

    # ------------------------------------------------------------------
    # The one-step query
    # ------------------------------------------------------------------

    def _solve(self, state_assumptions: Sequence[int], layer: int, goal: bool):
        """One solve: the state at 0 as given (a total state or a cube), the
        successor at 1 not blocked at `layer` (0 means no cubes) and, with
        `goal`, a goal state. Returns ('sat', action, successor) |
        ('unsat', core state literals) | ('timeout',)."""
        self.solves += 1
        self.ctl.assign_external(self.query1, goal)
        assumptions = list(state_assumptions) + self.static_assumptions + self._layer_assumptions(layer)
        found, core = {}, []

        def on_model(model):
            symbols = model.symbols(atoms=True)
            found['action'] = self.read_action(symbols)   # None for the do-nothing action
            found['successor'] = frozenset(
                s.arguments[0] for s in symbols
                if s.match('holds', 3) and s.arguments[2].number == 1
                and s.arguments[1].arguments[1] == TRUE and s.arguments[0] in self.state_literal)

        remaining = None if self.deadline is None else self.deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            return ('timeout',)
        with self.ctl.solve(assumptions=assumptions, on_model=on_model, async_=True) as handle:
            if not handle.wait(remaining):
                handle.cancel()
                return ('timeout',)
            result = handle.get()
            if not result.satisfiable:
                core = list(handle.core())   # on_core is not invoked for an asynchronous solve
        if result.satisfiable:
            return ('sat', found['action'], found['successor'])
        state_set = set(state_assumptions)
        return ('unsat', [lit for lit in core if lit in state_set])

    def _state_assumptions(self, state: State) -> List[int]:
        return [self.state_literal[v] if v in state else -self.state_literal[v] for v in self.free]

    def _cube_assumptions(self, cube: Cube) -> List[int]:
        return [self.state_literal[v] if positive else -self.state_literal[v] for v, positive in cube]

    def _as_cube(self, literals: Sequence[int]) -> Cube:
        return frozenset((self.literal_variable[abs(lit)], lit > 0) for lit in literals)

    def _minimise(self, literals: List[int], layer: int, goal: bool) -> Optional[List[int]]:
        """Drop reason literals while the query stays unsatisfiable."""
        kept = list(literals)
        i = 0
        while i < len(kept):
            trial = kept[:i] + kept[i + 1:]
            result = self._solve(trial, layer, goal)
            if result[0] == 'timeout':
                return None
            if result[0] == 'unsat':
                new_core = set(result[1])
                kept = [lit for lit in trial if lit in new_core] or trial
                i = min(i, len(kept))
            else:
                i += 1
        return kept

    def extend(self, state: State, level: int):
        """('sat', action, successor) | ('unsat', reason cube) | ('timeout',)."""
        self.queries += 1
        goal = level == 1
        result = self._solve(self._state_assumptions(state), level - 1, goal)
        if result[0] != 'unsat':
            return result
        reason = self._minimise(result[1], level - 1, goal)
        if reason is None:
            return ('timeout',)
        return ('unsat', self._as_cube(reason))

    # ------------------------------------------------------------------
    # The search
    # ------------------------------------------------------------------

    def run(self, max_layers: int, deadline: Optional[float] = None):
        """('plan', [(name, objects...) tuples]) | ('unsolvable', layer) |
        ('exhausted', max_layers) | ('timeout', layer)."""
        self.deadline = deadline
        self.max_layers = max_layers
        first = self.extend(self.initial, 1)
        if first[0] == 'timeout':
            return ('timeout', 1)
        if first[0] == 'sat' and first[1] is None:
            return ('plan', [])
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
                    return ('plan', [a for a in node.path() + [action] if a is not None])
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
                self.queries += 1
                result = self._solve(self._cube_assumptions(self.cubes[index]), level, False)
                if result[0] == 'timeout':
                    return 'timeout'
                if result[0] == 'unsat':
                    self.levels[index] = level + 1
        return 'done'

    def statistics(self) -> str:
        by_level: Dict[int, int] = {}
        for lvl in self.levels:
            by_level[lvl] = by_level.get(lvl, 0) + 1
        return (f"{self.queries} one-step queries, {self.solves} solves, {len(self.cubes)} cubes learned "
                f"(per layer: {dict(sorted(by_level.items()))})")
