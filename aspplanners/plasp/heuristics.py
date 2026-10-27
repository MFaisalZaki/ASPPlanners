"""Planning heuristics for the ASP search, as clingo decision heuristics.

The solver does not expand states, so a planning heuristic cannot order an
open list here; what it can do is choose the solver's next DECISION. clingo
lets a propagator do that through `decide`, which is called whenever
propagation reaches a fixed point and may return the literal to make true next
(or the solver's own choice, the `fallback`).

`GoalSupportHeuristic` is the variable-selection rule of Rintanen, "Planning
as satisfiability: Heuristics" (AIJ 2012), the ingredient behind Madagascar's
planning-specific branching, restated over the seq encoding's atoms. It works
backward from the goal through the current partial assignment:

    take a subgoal (V = Val at step t), starting from the goal atoms at the
    horizon;
    find the earliest step e from which the value could persist up to t
    (walking back while it is not false at the step before); if that reaches
    the initial state and the value holds there, it is supported;
    else if an action already decided at some step in [e, t] achieves it, its
    preconditions at the step before (and the conditions of the achieving
    effect) are the next subgoals;
    else branch: decide TRUE a still-free achieving action at the EARLIEST
    such step -- next to the initial state, whose propagation refutes a wrong
    choice at once; under seq's one action per step this also settles that
    step. (Achieving it at the latest step instead is kept as a variant for
    comparison, and measures far worse: long persistence chains whose
    conflict with the initial state surfaces late.)

The walk stops at the first branching decision, or hands the solver its own
choice when every subgoal is supported. It never changes what is a model, so
every plan it finds is one the default search could find, at the same horizon.

Where it helps, measured: FINDING a model on a satisfiable horizon -- a
6-ball gripper at its horizon, with and without a fuel counter, in a third of
the default's time. Where it hurts, measured: one-step deepening, whose time
goes into refuting the horizons below the optimum, where a branching rule
guides the search for something that is not there no better than the default
and costs its learning: 37s against 63s on the same gripper. So it is off by
default, and meant for the settings where the horizon is satisfiable: a known
or overshot horizon, a scheduler that tries several at once.

Everything the rule reads -- the goal atoms, each action's effects and the
preconditions of the action and of each conditional effect -- is read off the
ground facts in the solver's symbolic atoms (`goal/2`, `postcondition/4`,
`precondition/3`), and the atoms of each step are mapped to solver literals as
the deepening grounds them (`init` runs at every solve). Numeric conditions
and effects are not subgoals here: an action is still chosen for the boolean
atom it achieves, and the numeric layer's constraints decide whether it is
applicable, as they do under the default search.
"""

from typing import Dict, List, Optional, Set, Tuple

import clingo

HEURISTICS = ('goal-support',)
TRUE = clingo.Function('true')


class GoalSupportHeuristic:
    """A clingo propagator whose only business is `decide`."""

    def __init__(self, max_level: Optional[int] = 2):
        # `max_level`: speak only while the solver's decision level is at most
        # this deep; beyond it the solver's own heuristic decides. None: always.
        # Measured on a 6-ball gripper at its satisfiable horizon: default
        # search 7.5s; depth 2, 2.9s; depth 3, 4.2s; always on, over 76s --
        # a Python `decide` at every fixpoint is costly, and while it speaks
        # the solver's own activity-based learning is bypassed.
        self.max_level = max_level
        self.horizon = 0
        self.goals: List[Tuple[clingo.Symbol, clingo.Symbol]] = []
        # (V, Val) -> [(action, effect term)] achieving it
        self.achievers: Dict[Tuple[clingo.Symbol, clingo.Symbol], List[Tuple[clingo.Symbol, clingo.Symbol]]] = {}
        # owner (action or effect term) -> [(V, Val)] it requires
        self.conditions: Dict[clingo.Symbol, List[Tuple[clingo.Symbol, clingo.Symbol]]] = {}
        self.occurs: Dict[Tuple[clingo.Symbol, int], int] = {}             # (A, t) -> literal
        self.holds: Dict[Tuple[clingo.Symbol, clingo.Symbol, int], int] = {}  # (V, Val, t) -> literal
        self._loaded = False
        self._seen_steps: Set[int] = set()
        self.decisions = 0     # decisions this heuristic made
        self.calls = 0         # times decide was asked

    # -- clingo.Propagator ---------------------------------------------------

    def init(self, init) -> None:
        atoms = init.symbolic_atoms
        if not self._loaded:
            self._loaded = True
            for a in atoms.by_signature('goal', 2):
                self.goals.append((a.symbol.arguments[0], a.symbol.arguments[1]))
            for a in atoms.by_signature('postcondition', 4):
                action, effect, v, val = a.symbol.arguments
                self.achievers.setdefault((v, val), []).append((action, effect))
            for a in atoms.by_signature('precondition', 3):
                owner, v, val = a.symbol.arguments
                self.conditions.setdefault(owner, []).append((v, val))
        # The atoms of steps grounded since the last solve.
        for a in atoms.by_signature('occurs', 2):
            key = (a.symbol.arguments[0], a.symbol.arguments[1].number)
            if key not in self.occurs:
                self.occurs[key] = init.solver_literal(a.literal)
        for a in atoms.by_signature('holds', 3):
            v, value, t = a.symbol.arguments
            key = (v, value, t.number)
            if key not in self.holds:
                self.holds[key] = init.solver_literal(a.literal)
        init.check_mode = clingo.PropagatorCheckMode.Off

    def decide(self, thread_id, assignment, fallback) -> int:
        self.calls += 1
        if self.max_level is not None and assignment.decision_level > self.max_level:
            return fallback
        worklist = [(v, val, self.horizon) for v, val in self.goals]
        seen: Set[Tuple[clingo.Symbol, clingo.Symbol, int]] = set()
        while worklist:
            v, val, t = worklist.pop()
            if t <= 0 or (v, val, t) in seen:
                continue
            seen.add((v, val, t))
            lit = self.holds.get((v, val, t))
            if lit is None or assignment.is_false(lit):
                continue   # cannot hold at t: nothing to support
            decision = self._support(v, val, t, assignment, worklist)
            if decision:
                self.decisions += 1
                return decision
        return fallback

    def _support(self, v, val, t, assignment, worklist) -> int:
        """Support the subgoal V = Val at t: push the subgoals of whatever
        already supports it, or return the action literal to decide."""
        # The earliest step e from which the value could persist up to t: walk
        # back while it is not false at the step before.
        e = t
        while e > 0:
            prev = self.holds.get((v, val, e - 1))
            if prev is None or assignment.is_false(prev):
                break
            e -= 1
        if e == 0:
            # It can hold from the initial state on; if it does, nothing to do.
            at0 = self.holds.get((v, val, 0))
            if at0 is not None and assignment.is_true(at0):
                return 0
            e = 1
        # Earliest step first: a variant achieving the subgoal at the latest
        # step instead was measured and only ever did worse (it builds long
        # persistence chains whose conflict with the initial state surfaces
        # after many more decisions).
        steps = range(e, t + 1)
        # Already achieved by a decided action somewhere in [e, t]?
        for s in steps:
            for action, effect in self.achievers.get((v, val), ()):
                action_lit = self.occurs.get((action, s))
                if action_lit is not None and assignment.is_true(action_lit):
                    worklist.extend((pv, pval, s - 1) for pv, pval in self.conditions.get(action, ()))
                    worklist.extend((pv, pval, s - 1) for pv, pval in self.conditions.get(effect, ()))
                    return 0
        # Otherwise branch on a free achiever, in the preferred order of steps.
        for s in steps:
            for action, _effect in self.achievers.get((v, val), ()):
                action_lit = self.occurs.get((action, s))
                if action_lit is not None and assignment.is_free(action_lit):
                    return action_lit
        return 0

    def statistics(self) -> str:
        return f"goal-support heuristic: {self.decisions} of {self.calls} decisions"
