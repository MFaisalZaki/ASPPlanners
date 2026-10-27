"""On-demand mutex constraints for parallel steps, as clingo propagators.

Under the sequential reading a horizon step holds one action, so the horizon
the deepening search has to reach is the plan's length. A *parallel* reading
lets a step hold a set of actions that can still be serialised:

  * forall-step (Kautz & Selman 1996): no two actions of a step interfere, so
    they can be executed in any order;
  * exists-step (Dimopoulos, Nebel & Koehler 1997; Rintanen, Heljanko & Niemela
    2006): there exists *some* order in which the step can be executed, i.e. the
    disabling graph induced by the step's actions is acyclic.

Both need interference constraints between actions. Encoding them up front is
the classical approach, and the number of mutexes -- quadratic in the actions,
per step -- is what dominates the size of the program. This module instead
follows Abdelwahed, Bofill, Espasa & Villaret, "On-Demand Mutex Constraints for
Numeric Planning as SMT" (ICAPS 2026): the encoding states no interference at
all (core.lp's step choice becomes `1 {occurs(A, t)}`), and a propagator plugged
into the solver's search watches the action atoms, keeps the set of actions
active at each step, and reacts when the solver makes an action true:

  * the forall propagator raises a conflict as soon as the new action interferes
    with one already active at the same step (the paper's Algorithm 1);
  * the exists propagator builds the disabling graph induced by the step's
    active actions and raises a conflict when the new action closes a cycle, the
    cycle's actions being the conflict (Algorithm 2).

A conflict is handed to clingo as a nogood over the `occurs` literals involved.
It is learned like any other clause, so a mutex is paid for only once it has
been hit, and is deletable, so it is forgotten when it stops being useful --
the two properties the paper identifies as the source of its scalability.
Interference itself is computed on demand too, the first time two actions are
considered together, and memoised (the paper's *lazy* interference); `eager`
precomputes the whole disabling graph at the first `init` instead, for
comparison, which is the paper's eager baseline with the mutexes still added
lazily.

A third semantics, `sequential`, completes the matrix: one action per step, as
core's own cap states it, but enforced by the propagator instead -- any two
actions active in one step are a conflict, with no interference to judge. It
exists to measure the propagator's overhead against clasp's native cardinality
constraint, since the two admit exactly the same plans.

What the propagator knows about an action is read off the ground task facts in
the solver's symbolic atoms (`precondition/3`, `postcondition/4`, the
disjunctive layer's `orDisjunct/5`, and the numeric layer's vocabulary), so no
encoding layer has to be changed or added to use it. Z3's user-propagator API
the paper builds on and clingo's `Propagator` line up directly: `fixed` is
`propagate`, `push`/`pop` is `undo`, a conflict is `add_nogood`.

Interference, as this module judges it. An action *reads* a variable at a value
through its preconditions, the conditions of its conditional effects (judged at
t-1 by core's caused/3, so an earlier action changing one would change which
effects fire), and every literal of its disjunctive conditions (an earlier
action breaking the disjunct that held could leave none holding). It *writes*
a variable at a value through every postcondition, conditional or not, whether
or not the condition fires. Action `a` DISABLES `b` -- `a` cannot be executed
before `b` within a step -- when `a` writes a variable `b` reads to a value
other than the one `b` reads, or when the two write one variable to different
values (which is symmetric and so a 2-cycle: they cannot share a step at all).
Numeric fluents are value-less here: a numeric write disables every reader of
the fluent (a precondition or an effect expression evaluated at t-1 would see
another value) and conflicts with every other writer (the numeric layer's
unconditional effects do not sum, so two writes in one step are not a state).
Every one of these is syntactic and over-approximates -- the paper's "semantic"
interference check (Bofill, Espasa & Villaret 2021), which would let two
increases of one fluent commute, is the natural extension point: override
`InterferenceGraph.disables`.

Why the two conditions suffice for correctness: a step whose actions were all
applicable in the state before it (core checks that) and whose disabling graph
is acyclic can be serialised in a topological order; along it every variable an
action reads is either untouched since the step's start or already carrying the
value it reads, so its preconditions still hold and the same conditional effects
fire as core judged; and with no two actions writing a variable differently the
state after the serialisation is the one core computes by applying all effects
at once. The planner serialises each step with `serialise()` and the UP
validator then checks that order.
"""

from typing import Dict, FrozenSet, Hashable, Iterable, List, Optional, Sequence, Set, Tuple

import clingo

# A numeric fluent's "value" in the read/write sets: any write to it clashes
# with any read of it or write to it.
ANY = object()

SEMANTICS = ('exists', 'forall', 'sequential')
INTERFERENCE = ('lazy', 'eager')


class InterferenceGraph:
    """What each ground action reads and writes, and the disabling relation
    derived from it, computed on demand and memoised.

    Built from the ground facts in a `clingo.SymbolicAtoms` table (every fact
    the encoder emits is there once base has been grounded, so this is the task
    as the solver sees it). Keys are the action terms exactly as they appear in
    `occurs/2`.
    """

    def __init__(self, symbolic_atoms, interference: str = 'lazy'):
        if interference not in INTERFERENCE:
            raise ValueError(f"interference must be one of {INTERFERENCE}, not {interference!r}")
        self.interference = interference
        # action -> variable -> set of values (ANY for a numeric fluent)
        self.reads: Dict[clingo.Symbol, Dict[clingo.Symbol, Set]] = {}
        self.writes: Dict[clingo.Symbol, Dict[clingo.Symbol, Set]] = {}
        self._memo: Dict[Tuple[clingo.Symbol, clingo.Symbol], bool] = {}
        self.edges_computed = 0   # pairs judged (lazily or eagerly)
        self.edges = 0            # of which are disabling edges
        self._load(symbolic_atoms)
        if interference == 'eager':
            actions = list(set(self.reads) | set(self.writes))
            for a in actions:
                for b in actions:
                    if a is not b:
                        self.disables(a, b)

    # ------------------------------------------------------------------
    # Loading the read/write sets from the ground facts
    # ------------------------------------------------------------------

    @staticmethod
    def _facts(symbolic_atoms, name: str, arity: int) -> Iterable[Sequence[clingo.Symbol]]:
        for atom in symbolic_atoms.by_signature(name, arity):
            yield atom.symbol.arguments

    @staticmethod
    def _add(store, action, variable, value) -> None:
        store.setdefault(action, {}).setdefault(variable, set()).add(value)

    def _load(self, atoms) -> None:
        actions = {args[0] for args in self._facts(atoms, 'action', 1)}
        # Conditions keyed by a term: an action's own, or one of its effect
        # terms (postcondition/4's second argument). The effect term's
        # conditions are the conditional effect's, and count as reads of the
        # action that carries the effect.
        conditions: Dict[clingo.Symbol, List[Tuple[clingo.Symbol, object]]] = {}
        for owner, variable, value in self._facts(atoms, 'precondition', 3):
            conditions.setdefault(owner, []).append((variable, value))
        for owner, _group, _disjunct, variable, value in self._facts(atoms, 'orDisjunct', 5):
            conditions.setdefault(owner, []).append((variable, value))
        # Numeric: the fluents of an expression, for reads through comparisons
        # and effect expressions.
        terms: Dict[clingo.Symbol, Set[clingo.Symbol]] = {}
        for expression, _coefficient, fluent in self._facts(atoms, 'numTerm', 3):
            terms.setdefault(expression, set()).add(fluent)
        for owner, _op, e1, e2 in self._facts(atoms, 'numPrecondition', 4):
            for fluent in terms.get(e1, set()) | terms.get(e2, set()):
                conditions.setdefault(owner, []).append((fluent, ANY))
        for owner, _group, _disjunct, _op, e1, e2 in self._facts(atoms, 'orDisjunctNum', 6):
            for fluent in terms.get(e1, set()) | terms.get(e2, set()):
                conditions.setdefault(owner, []).append((fluent, ANY))

        for action in actions:
            for variable, value in conditions.get(action, ()):
                self._add(self.reads, action, variable, value)

        effect_owner: Dict[clingo.Symbol, Set[clingo.Symbol]] = {}
        for action, effect, variable, value in self._facts(atoms, 'postcondition', 4):
            self._add(self.writes, action, variable, value)
            effect_owner.setdefault(effect, set()).add(action)
        # Numeric effects: constant and expression-valued, unconditional and
        # conditional. Expression-valued ones also read the expression's fluents.
        for action, fluent, _delta in self._facts(atoms, 'numEffect', 3):
            self._add(self.writes, action, fluent, ANY)
        for action, fluent, _constant in self._facts(atoms, 'numAssign', 3):
            self._add(self.writes, action, fluent, ANY)
        for name in ('numEffectExpr', 'numAssignExpr'):
            for action, fluent, expression in self._facts(atoms, name, 3):
                self._add(self.writes, action, fluent, ANY)
                for read in terms.get(expression, ()):
                    self._add(self.reads, action, read, ANY)
        for action, effect, fluent, _delta in self._facts(atoms, 'numCondEffect', 4):
            self._add(self.writes, action, fluent, ANY)
            effect_owner.setdefault(effect, set()).add(action)
        for name in ('numCondEffectExpr', 'numCondAssignExpr'):
            for action, effect, fluent, expression in self._facts(atoms, name, 4):
                self._add(self.writes, action, fluent, ANY)
                effect_owner.setdefault(effect, set()).add(action)
                for read in terms.get(expression, ()):
                    self._add(self.reads, action, read, ANY)
        # A conditional effect's conditions are reads of the action carrying it.
        for effect, owners in effect_owner.items():
            for variable, value in conditions.get(effect, ()):
                for action in owners:
                    self._add(self.reads, action, variable, value)

    # ------------------------------------------------------------------
    # The disabling relation
    # ------------------------------------------------------------------

    def disables(self, a: clingo.Symbol, b: clingo.Symbol) -> bool:
        """`a` cannot be executed before `b` within one step.

        Override for a finer (semantic) notion of interference; the
        propagators only ever call this.
        """
        key = (a, b)
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        self.edges_computed += 1
        result = self._disables(a, b)
        self._memo[key] = result
        self.edges += result
        return result

    def _disables(self, a, b) -> bool:
        writes_a = self.writes.get(a, {})
        if not writes_a:
            return False
        reads_b = self.reads.get(b, {})
        for variable, written in writes_a.items():
            read = reads_b.get(variable)
            if read is None:
                continue
            if ANY in written or ANY in read:
                return True
            if any(w != r for w in written for r in read):
                return True
        writes_b = self.writes.get(b, {})
        for variable, written in writes_a.items():
            other = writes_b.get(variable)
            if other is None:
                continue
            if ANY in written or ANY in other or written != other:
                return True
        return False

    # ------------------------------------------------------------------
    # Serialising a step
    # ------------------------------------------------------------------

    def serialise(self, actions: Sequence[clingo.Symbol]) -> List[clingo.Symbol]:
        """A topological order of `actions` under the disabling relation: an
        action is placed once no action still waiting must precede it (i.e. it
        disables none of them). Raises if the induced graph has a cycle, which
        the exists propagator rules out for any model it lets through."""
        remaining, order = list(actions), []
        while remaining:
            for candidate in remaining:
                if not any(self.disables(candidate, other) for other in remaining if other is not candidate):
                    order.append(candidate)
                    remaining.remove(candidate)
                    break
            else:
                raise ValueError(
                    f"The step {[str(a) for a in remaining]} cannot be serialised: its "
                    "disabling graph has a cycle the propagator should have rejected.")
        return order


class ParallelStepPropagator:
    """clingo propagator enforcing forall-step or exists-step semantics over
    core.lp's `occurs/2` atoms, with mutexes generated on demand.

    Register it on a `clingo.Control` before grounding; `init` is called at
    every solve, so under the multi-shot deepening it picks up each new step's
    atoms as they are grounded. State is kept per solver thread.
    """

    def __init__(self, semantics: str = 'exists', interference: str = 'lazy'):
        if semantics not in SEMANTICS:
            raise ValueError(f"semantics must be one of {SEMANTICS}, not {semantics!r}")
        self.semantics = semantics
        self.interference = interference
        self.graph: Optional[InterferenceGraph] = None
        # solver literal -> the (action, step) atoms it stands for (normally one;
        # two occurs atoms can share a literal once the solver has equated them)
        self._literal_atoms: Dict[int, List[Tuple[clingo.Symbol, int]]] = {}
        # thread -> step -> action -> literal
        self._active: List[Dict[int, Dict[clingo.Symbol, int]]] = []
        self.conflicts = 0

    # -- clingo.Propagator interface -------------------------------------

    def init(self, init) -> None:
        if self.graph is None and self.semantics != 'sequential':
            self.graph = InterferenceGraph(init.symbolic_atoms, self.interference)
        for atom in init.symbolic_atoms.by_signature('occurs', 2):
            literal = init.solver_literal(atom.literal)
            key = (atom.symbol.arguments[0], atom.symbol.arguments[1].number)
            atoms = self._literal_atoms.setdefault(literal, [])
            if key in atoms:
                continue
            if not atoms:
                init.add_watch(literal)
            atoms.append(key)
        while len(self._active) < init.number_of_threads:
            self._active.append({})

    def propagate(self, control, changes) -> None:
        active = self._active[control.thread_id]
        for literal in changes:
            for action, step in self._literal_atoms[literal]:
                step_actions = active.setdefault(step, {})
                step_actions[action] = literal
                if self.semantics == 'sequential':
                    ok = self._check_sequential(control, action, literal, step_actions)
                elif self.semantics == 'forall':
                    ok = self._check_forall(control, action, literal, step_actions)
                else:
                    ok = self._check_exists(control, action, step_actions)
                if not ok:
                    return

    def undo(self, thread_id, assignment, changes) -> None:
        active = self._active[thread_id]
        for literal in changes:
            for action, step in self._literal_atoms[literal]:
                active.get(step, {}).pop(action, None)

    # -- the checks --------------------------------------------------------

    def _check_sequential(self, control, action, literal, step_actions) -> bool:
        """One action per step: every other active action is a conflict."""
        for other, other_literal in list(step_actions.items()):
            if other is action:
                continue
            self.conflicts += 1
            if not control.add_nogood([literal, other_literal]) or not control.propagate():
                return False
        return True

    def _check_forall(self, control, action, literal, step_actions) -> bool:
        """Algorithm 1: a conflict for every active action the new one
        interferes with, in either direction."""
        graph = self.graph
        for other, other_literal in list(step_actions.items()):
            if other is action:
                continue
            if graph.disables(action, other) or graph.disables(other, action):
                self.conflicts += 1
                if not control.add_nogood([literal, other_literal]) or not control.propagate():
                    return False
        return True

    def _check_exists(self, control, action, step_actions) -> bool:
        """Algorithm 2: a conflict when the new action closes a cycle in the
        disabling graph induced by the step's active actions."""
        cycle = self._cycle_through(action, step_actions)
        if cycle is None:
            return True
        self.conflicts += 1
        return control.add_nogood([step_actions[a] for a in cycle]) and control.propagate()

    def _cycle_through(self, start, step_actions) -> Optional[List[clingo.Symbol]]:
        """Depth-first search for a cycle through `start` along disabling
        edges among the step's actions. Only cycles through the newly added
        action can be new, so that is the only start considered."""
        graph = self.graph
        stack = [(start, [start])]
        seen = {start}
        while stack:
            node, path = stack.pop()
            for nxt in step_actions:
                if nxt is node or not graph.disables(node, nxt):
                    continue
                if nxt == start:
                    return path
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append((nxt, path + [nxt]))
        return None

    # -- for the planner ---------------------------------------------------

    def serialise(self, actions: Sequence[clingo.Symbol]) -> List[clingo.Symbol]:
        if self.semantics == 'sequential':
            if len(actions) > 1:
                raise ValueError(
                    f"A sequential step holds {[str(a) for a in actions]}; the propagator "
                    "should have rejected it.")
            return list(actions)
        return self.graph.serialise(actions)

    def statistics(self) -> str:
        if self.semantics == 'sequential':
            return f"sequential, cap enforced by the propagator: {self.conflicts} conflicts raised"
        graph = self.graph
        return (f"{self.semantics}-step, {self.interference} interference: "
                f"{self.conflicts} mutex conflicts raised, "
                f"{graph.edges} disabling edges among {graph.edges_computed} action pairs judged"
                if graph else f"{self.semantics}-step: not initialised")
