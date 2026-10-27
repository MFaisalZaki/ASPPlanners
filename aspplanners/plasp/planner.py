import os
import time
from fractions import Fraction
from typing import List, Optional, Tuple

import clingo

from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.plans import SequentialPlan, TimeTriggeredPlan, ActionInstance

from aspplanners.common.errors import refusal_from_clingo_error
from aspplanners.common.temporal import DEFAULT_TIME_SCALE
from aspplanners.plasp.encoder import PLASPEncoder
from aspplanners.plasp.facts import asp_name
from aspplanners.plasp.layers import FAMILIES, fact_predicates, parse_selection
from aspplanners.plasp.pdr import PDR
from aspplanners.lp_io import ASPStatement, parse_lp_file, dump_lp
from aspplanners.common.validation import validate_plan

# encoding family -> the encoder that produces its facts. Which *layers* of the
# family get loaded is decided per task; see aspplanners.plasp.layers.
ENCODERS = {
    'seq': PLASPEncoder,
}


class PLASPPlanner:
    """Horizon-based planner: compiles a UP problem to ASP facts and searches
    for a plan with clingo, deepening the horizon until a model is found.

    Registered with Unified Planning as the ``PLASPPlanner`` engine.

    `encoder_type` names the encoding family and, optionally, its layers:
    ``'seq'`` picks the layers from the task itself, ``'seq+numeric+temporal'``
    names them. Either way the resolved layers are in `self.layers` and the files
    they load from in `self.encoding_paths`; see `aspplanners.plasp.layers`.

    The solve status of the last `plan()` call is kept in `self.status`
    (a `PlanGenerationResultStatus`) and human-readable notes in `self.logs`.
    """

    def __init__(self, problem, encoder_type='seq', compilationlist: Optional[List[List[str]]] = None,
                 time_scale: int = DEFAULT_TIME_SCALE):
        family_name, explicit_layers = parse_selection(encoder_type)
        if family_name not in ENCODERS:
            raise ValueError(
                f"Unsupported encoder type: {encoder_type!r}; available: {sorted(ENCODERS)}")
        encoder_cls   = ENCODERS[family_name]
        self.family   = FAMILIES[family_name]
        self.problem       = problem
        # Kept because it is a decision about the task, not just a parameter: on
        # the default of None it is worked out from the task's own shape.
        self.compilationlist = self._check_compilationlist(problem, compilationlist)
        self.compiled_task = encoder_cls(time_scale=time_scale).compile(problem, self.compilationlist)
        self.task          = self.compiled_task.problem
        # The task facts never change across horizons: build the string once.
        self.task_facts    = '\n'.join(sorted(self.compiled_task.fact_lines))
        # Which layers of the encoding this task needs. Driven by the facts the
        # encoder actually emitted, widened by what the compiled problem's kind
        # suggests, and then checked for coverage -- an emitted fact that no
        # loaded layer reads is silently ignored by clingo, so it is rejected
        # here rather than surfacing as a failed plan validation later.
        self.layers = self._select_layers(explicit_layers)
        self.encoding_paths = self.family.paths(self.layers)
        # Model atoms carry ASP-rendered names ('-' -> '_'); map them back to
        # the compiled task's vocabulary (the encoder guarantees injectivity).
        self._actions_by_asp_name = {asp_name(a.name): a for a in self.task.actions}
        self._objects_by_asp_name = {asp_name(o.name): o for o in self.task.all_objects}
        self.logs: List[str] = []
        self.status: Optional[PlanGenerationResultStatus] = None
        # The horizon the last plan() call found its plan at; under PDR, the
        # plan's length (plans there are not guaranteed shortest).
        self.solved_horizon: Optional[int] = None

    def _select_layers(self, explicit_layers: Optional[Tuple[str, ...]]) -> Tuple[str, ...]:
        """Resolve the encoding layers for this task, explicit or inferred.

        `explicit_layers` comes from an ``'seq+numeric+temporal'``-style spec. It
        is not taken at face value: it is closed under each layer's requirements
        and then coverage-checked like an inferred set, so naming too few layers
        is an error rather than a program that quietly ignores half the task.
        """
        predicates = fact_predicates(self.compiled_task.fact_lines)
        if explicit_layers is None:
            layers = self.family.select(predicates, kind=self.task.kind)
        else:
            layers = self.family.close(explicit_layers)
        self.family.check_coverage(predicates, layers)
        return layers

    def _check_compilationlist(self, problem, compilationlist: Optional[List[List[str]]]) -> List[List[str]]:
        """The UP compilers to run before encoding: by default, none."""
        if compilationlist is not None:
            return compilationlist

        # Empty: the task goes to clingo as it stands.
        #
        # Nothing needs compiling away, because the encoding states all of it --
        # negative conditions (it is multi-valued, so `value(V, false)` is a
        # value like any other), `forall` (a conjunction over the universe,
        # emitted with its variable free for the grounder to range), and `or` /
        # `exists` (an orGroup/orDisjunct pair, the latter with its disjuncts
        # indexed by the quantified variable's binding).
        #
        # Nor is the task pre-ground. gringo grounds it either way, and the
        # lifted encoding gives it as much to prune with: action signature rules
        # bind parameters via has(_, type(...)) and fold static preconditions
        # into the rule body. On the shipped classical domains that reaches the
        # same program a reachability grounder does, byte for byte, for a
        # quarter of the build time.
        #
        # The exception is an action narrowed *only* by a dynamic precondition
        # -- `use(?x, ?m)` gated by `loaded(?x, ?m)`, where `loaded` is only ever
        # established for some pairs. Static folding cannot see that; reachability
        # analysis can, and the lifted program is quadratically bigger. For a
        # domain shaped like that, pass a compilationlist ending in
        #   [select_grounder(problem.kind, REACHABILITY_GROUNDERS), CompilationKind.GROUNDING]
        # (which is what common.compilation.select_grounder is there for).
        #
        # PDDL (:functions ...) parse as real-typed fluents; the fact builders
        # accept them as long as every constant is integral (clingo terms are
        # integers) and raise otherwise.
        #
        # A conditional *numeric* effect used to be the one exception, because
        # caused/3 carries a variable's new value: enough for a conditional
        # boolean effect or a conditional assignment of a constant, but not for a
        # conditional increase (whose new value is the old one plus a delta) nor
        # for a numeric condition (which caused/3 reads off precondition/3).
        # Those had to be compiled away, at the cost of 2^k action variants. The
        # numeric layer now states both directly -- the firing deltas are summed
        # and applied once, and a numeric condition is a one-disjunct group of
        # the effect term -- so nothing is left that needs the compiler.
        return []

    def validate(self, plan) -> Tuple[bool, Optional[str]]:
        """Validate a plan against the original problem with UP's sequential
        plan validator. Returns (is_valid, reason); reason is None/empty when
        the plan is valid."""
        return validate_plan(self.problem, plan)

    def lp_program(self) -> str:
        """The complete logic program: the compiled task's facts followed by
        the encoding rules. Useful for dumping to a file or feeding a clingo
        Control of your own.

        The selected layers are concatenated in dependency order, which is the
        same order `plan()` loads them in and produces the same program -- every
        layer file opens with an explicit `#program base.`, so no layer inherits
        the part its predecessor happened to end in.

        The encoding is multi-shot (`#program base/step(t)/check(t)`): ground
        it with parts [('base', []), ('step', [1..h]), ('check', [h])] and
        assign the external `query(h)` to true for a fixed-horizon solve.
        The facts come first so they belong to the implicit base part.
        """
        sections = [f"%% Task facts\n{self.task_facts}"]
        for name, path in zip(self.layers, self.encoding_paths):
            with open(path, 'r') as f:
                sections.append(f"%% Layer: {name}\n{f.read()}")
        return '\n\n'.join(sections)

    def encoding_terms(self) -> List[ASPStatement]:
        """The loaded encoding parsed into ASPTerm statements (facts, rules,
        constraints, directives) for programmatic inspection or rewriting;
        write a modified list back with `aspplanners.lp_io.dump_lp`.
        Unlike `lp_program()`, the rendering is clingo-normalized (comments
        dropped, whitespace normalized). Layers are concatenated in load order.
        """
        return [term for path in self.encoding_paths for term in parse_lp_file(path)]

    def dump_lp_program(self, destination) -> None:
        """Write the complete logic program (task facts + encoding, verbatim
        with comments) to a file path or file-like object."""
        program = self.lp_program()
        if hasattr(destination, 'write'):
            destination.write(program)
        else:
            with open(destination, 'w') as f:
                f.write(program)

    def _empty_plan(self):
        """The "no plan" value, of whichever plan class this task produces."""
        return TimeTriggeredPlan([]) if self.compiled_task.is_temporal else SequentialPlan([])

    def plan(self, horizon=None, max_horizon=1000, timeout=None, search='deepening',
             pdr_first: float = 0):
        """Iterative-deepening search over horizons 0..max_horizon, or a single
        solve at `horizon` when given, or -- with ``search='pdr'`` --
        property-directed reachability over the one-step program (see
        `aspplanners.plasp.pdr` and ``encodings/seq/pdr.lp``), with
        `max_horizon` bounding the number of layers. PDR can prove a task
        unsolvable (`UNSOLVABLE_PROVEN`), which deepening cannot, and takes
        no fixed `horizon`; it is available on the seq family for classical
        tasks (no numeric fluents, no durative actions).

        `pdr_first`, in seconds, runs PDR first on the BOOLEAN ABSTRACTION of
        the task (numeric layer left out) for at most that long, before the
        deepening. An abstract proof of unsolvability settles the task (it
        carries over); an abstract plan is validated against the real task
        and returned when it is a real one, which on a numeric task is the
        case whenever the numbers were slack; otherwise the deepening runs
        as usual. Skipped, with a log line, on a temporal task and on the
        one shape that does not abstract soundly (see `_pdr_first`).

        Deepening is multi-shot: one clingo Control instance grounds only the
        new step(t)/check(t) parts per horizon instead of regrounding the whole
        program each iteration.

        Returns the plan mapped back onto the original problem -- a
        `SequentialPlan`, or a `TimeTriggeredPlan` when the task has durative
        actions. The empty plan means "no plan found" (check `self.status` to
        distinguish an unsatisfiable/timed-out search from a goal that is
        trivially reached), except when the goal already holds in the initial
        state, in which case the empty plan IS the solution and `self.status`
        is SOLVED_SATISFICING.
        """
        deadline = time.monotonic() + timeout if timeout is not None else None
        if search == 'pdr':
            if horizon is not None:
                raise ValueError("search='pdr' has no fixed horizon; use max_horizon to bound its layers.")
            return self._plan_pdr(max_horizon, deadline)
        if search != 'deepening':
            raise ValueError(f"search must be 'deepening' or 'pdr', not {search!r}")

        if pdr_first > 0:
            settled = self._pdr_first(pdr_first, max_horizon, deadline)
            if settled is not None:
                return settled

        ctl = clingo.Control(arguments=['-n', '1'])
        for path in self.encoding_paths:
            ctl.load(path)
        ctl.add('base', [], self.task_facts)

        if horizon is not None:
            parts = [('base', []), ('check', [clingo.Number(horizon)])]
            parts += [('step', [clingo.Number(t)]) for t in range(1, horizon + 1)]
            self._ground(ctl, parts)
            ctl.assign_external(clingo.Function('query', [clingo.Number(horizon)]), True)
            outcome, symbols = self._solve(ctl, deadline)
            if outcome == 'unsat':
                self.status = PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY
                self.logs.append(f'No plan exists at the fixed horizon {horizon}.')
                return self._empty_plan()
            return self._conclude(outcome, symbols, horizon)

        self._ground(ctl, [('base', []), ('check', [clingo.Number(0)])])
        ctl.assign_external(clingo.Function('query', [clingo.Number(0)]), True)
        for t in range(0, max_horizon + 1):
            outcome, symbols = self._solve(ctl, deadline)
            if outcome != 'unsat':
                return self._conclude(outcome, symbols, t)
            if t == max_horizon:
                break
            # Retire the horizon-t goal test and extend the program by one step.
            ctl.release_external(clingo.Function('query', [clingo.Number(t)]))
            self._ground(ctl, [('step', [clingo.Number(t + 1)]), ('check', [clingo.Number(t + 1)])])
            ctl.assign_external(clingo.Function('query', [clingo.Number(t + 1)]), True)

        self.status = PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY
        self.logs.append(
            f'No plan found up to horizon {max_horizon} (the task may be solvable with a longer horizon).')
        return self._empty_plan()

    # ------------------------------------------------------------------
    # Solving
    # ------------------------------------------------------------------

    @staticmethod
    def _ground(ctl, parts) -> None:
        """`ctl.ground(parts)`, with a non-representable program reported as a
        refusal rather than as a crash.

        Grounding is where clingo finds out that the program the encoder built
        cannot be held in its 32-bit terms -- a `#sum` over the reachable values
        of an unbounded fluent, typically. The encoder cannot rule that out by
        inspection (see `aspplanners.common.errors`), so it is classified here,
        where it surfaces. Anything else clingo raises is left alone.
        """
        try:
            ctl.ground(parts)
        except RuntimeError as error:
            refusal = refusal_from_clingo_error(error)
            if refusal is None:
                raise
            raise refusal from error

    def _solve(self, ctl, deadline) -> Tuple[str, Optional[list]]:
        """Run one solve call against the current grounding.

        Returns ('sat', symbols) | ('unsat', None) | ('timeout', None).
        """
        time_left = None
        if deadline is not None:
            time_left = deadline - time.monotonic()
            if time_left <= 0:
                return 'timeout', None

        models = []
        with ctl.solve(on_model=lambda m: models.append(m.symbols(shown=True)),
                       async_=True) as handle:
            if not handle.wait(time_left):
                handle.cancel()
                return 'timeout', None
            result = handle.get()

        if result.satisfiable:
            return 'sat', models[-1]
        return 'unsat', None

    def _conclude(self, outcome, symbols, horizon):
        """Turn a non-unsat solve outcome into a plan + status."""
        if outcome == 'timeout':
            self.status = PlanGenerationResultStatus.TIMEOUT
            self.logs.append(f'Timed out while solving horizon {horizon}.')
            return self._empty_plan()

        _plan = self._extract_plan(symbols)
        is_valid, reason = self.validate(_plan)
        if not is_valid:
            self.status = PlanGenerationResultStatus.INTERNAL_ERROR
            self.logs.append(f'Plan validation failed: {reason}')
            return self._empty_plan()
        self.status = PlanGenerationResultStatus.SOLVED_SATISFICING
        self.solved_horizon = horizon
        return _plan

    # ------------------------------------------------------------------
    # Property-directed reachability
    # ------------------------------------------------------------------

    def pdr_program(self) -> str:
        """The one-step program PDR queries: the task facts with the initial
        state renamed to pdrInitialState/2 (the state at 0 is the query's;
        static fluents keep theirs), the family's layers, and pdr.lp."""
        self._check_pdr()
        sections = [f"%% Task facts (initial state renamed for PDR)\n{self._pdr_facts()}"]
        for name, path in zip(self.layers, self.encoding_paths):
            with open(path, 'r') as f:
                sections.append(f"%% Layer: {name}\n{f.read()}")
        with open(self._pdr_path(), 'r') as f:
            sections.append(f"%% PDR one-step query program\n{f.read()}")
        return '\n\n'.join(sections)

    def _pdr_path(self) -> str:
        return os.path.join(self.family.directory, 'pdr.lp')

    def _pdr_static_fluents(self) -> set:
        """Fluents no action writes. Their initial facts keep their name: the
        encoder folds static preconditions into the action signatures as
        `initialState(...)` body atoms, which would otherwise never hold and
        drop every such action from the program; and their value is pinned to
        the initial one in every PDR query rather than left free."""
        written = {eff.fluent._content.payload.name
                   for action in self.task.actions for eff in action.effects}
        # ASP-rendered names: that is how the fact lines and the atoms spell them.
        return {asp_name(f.name) for f in self.task.fluents if f.name not in written}

    def _pdr_facts(self) -> str:
        static = self._pdr_static_fluents()
        prefix = 'initialState(variable(("'

        def rename(line):
            if not line.startswith('initialState('):
                return line
            name = line[len(prefix):].split('"', 1)[0] if line.startswith(prefix) else None
            if name in static:
                return line
            return 'pdrInitialState(' + line[len('initialState('):]
        return '\n'.join(rename(line) for line in sorted(self.compiled_task.fact_lines))

    def _check_pdr(self) -> None:
        if self.family.name != 'seq':
            raise ValueError(f"search='pdr' is available on the 'seq' encoding, not {self.family.name!r}.")
        if self.compiled_task.is_temporal or 'temporal' in self.layers:
            raise ValueError("search='pdr' is not available on a temporal task.")
        if 'numeric' in self.layers:
            # Tried and measured: grounding a program per query with the
            # obligation's numeric values as facts is correct and general, but
            # a reason can never generalise over a numeric value, so the
            # layers grow with every value seen -- two orders of magnitude
            # slower than deepening on a gripper with a fuel counter. Numeric
            # PDR needs interval reasons, which is IC3 modulo theories.
            raise ValueError(
                "search='pdr' is available on classical tasks only: a reason cannot "
                "generalise over a numeric value, so PDR enumerates numeric states.")

    def _pdr_control(self) -> clingo.Control:
        """A Control with the layers and pdr.lp loaded and the task facts
        added, nothing grounded; PDR grounds it once and queries it for the
        whole search (see `aspplanners.plasp.pdr`)."""
        ctl = clingo.Control(arguments=['-n', '1'])
        for path in self.encoding_paths:
            ctl.load(path)
        ctl.load(self._pdr_path())
        ctl.add('base', [], self._pdr_facts())
        return ctl

    def _pdr_first(self, budget, max_layers, deadline):
        """Run PDR on the boolean abstraction for at most `budget` seconds.

        Returns the plan or proof that settles the task, or None when the
        deepening has to run: the budget ran out, the abstract plan is not a
        real one, or the abstraction does not apply -- a temporal task, or a
        boolean effect under a numeric condition, which the abstraction would
        make fire always rather than maybe (so a reason learned there would
        not hold for the real task).

        Why the abstraction is otherwise sound: without the numeric layer,
        numeric preconditions and goal conditions fall away, so more actions
        apply and more states count as goals. That over-approximates
        reachability, so "cannot reach the goal" there holds for the real
        task, and so does a proof of unsolvability. The cubes it learns hold
        too -- feeding them into the deepening as distance-to-goal
        constraints was tried and measured, and cost more per step than it
        saved on every instance, so only the plan and the proof are used.
        """
        if self.family.name != 'seq' or self.compiled_task.is_temporal or 'temporal' in self.layers:
            self.logs.append('pdr_first skipped: not a seq classical or numeric task.')
            return None
        numeric = 'numeric' in self.layers
        if numeric and any(line.startswith('orDisjunctNum(effect(') for line in self.compiled_task.fact_lines):
            self.logs.append('pdr_first skipped: a boolean effect is conditioned on a numeric '
                             'comparison, which the boolean abstraction cannot state soundly.')
            return None
        ctl = clingo.Control(arguments=['-n', '1'])
        for name, path in zip(self.layers, self.encoding_paths):
            if name != 'numeric':
                ctl.load(path)
        ctl.load(self._pdr_path())
        ctl.add('base', [], self._pdr_facts())
        start = time.monotonic()
        run_deadline = start + budget if deadline is None else min(start + budget, deadline)
        try:
            pdr = PDR(ctl, self._pdr_static_fluents())
            outcome = pdr.run(max_layers, run_deadline)
        except RuntimeError as error:
            refusal = refusal_from_clingo_error(error)
            if refusal is None:
                raise
            raise refusal from error
        elapsed = time.monotonic() - start
        kind = outcome[0]
        where = ' on the boolean abstraction' if numeric else ''
        if kind == 'unsolvable':
            self.status = PlanGenerationResultStatus.UNSOLVABLE_PROVEN
            self.logs.append(f'PDR{where} proved the task unsolvable in {elapsed:.2f}s ({pdr.statistics()}).')
            return self._empty_plan()
        if kind == 'plan':
            plan = self._pdr_plan_from(outcome[1])
            if plan is not None:
                self.logs.append(f'PDR found a plan in {elapsed:.2f}s{where}'
                                 f'{", valid for the task" if numeric else ""} ({pdr.statistics()}).')
                return plan
            self.logs.append(f"PDR's plan{where} is not valid for the task ({elapsed:.2f}s); deepening.")
            return None
        self.logs.append(f'pdr_first: {kind} after {elapsed:.2f}s ({pdr.statistics()}); deepening.')
        return None

    def _pdr_plan_from(self, actions):
        """A validated plan from PDR's action symbols, or None."""
        steps = [self._construct_action(self._action_tuple(clingo.Function('occurs', [a, clingo.Number(0)])))
                 for a in actions]
        _plan = SequentialPlan(steps).replace_action_instances(self.compiled_task.map_back_action_instance)
        is_valid, _reason = self.validate(_plan)
        if not is_valid:
            return None
        self.status = PlanGenerationResultStatus.SOLVED_SATISFICING
        self.solved_horizon = len(steps)
        return _plan

    def _plan_pdr(self, max_layers, deadline):
        self._check_pdr()
        try:
            pdr = PDR(self._pdr_control(), self._pdr_static_fluents())
            outcome = pdr.run(max_layers, deadline)
        except RuntimeError as error:
            refusal = refusal_from_clingo_error(error)
            if refusal is None:
                raise
            raise refusal from error
        kind = outcome[0]
        self.logs.append(f'PDR: {kind} ({pdr.statistics()}).')
        if kind == 'timeout':
            self.status = PlanGenerationResultStatus.TIMEOUT
            self.logs.append(f'Timed out while working on layer {outcome[1]}.')
            return self._empty_plan()
        if kind == 'unsolvable':
            self.status = PlanGenerationResultStatus.UNSOLVABLE_PROVEN
            self.logs.append(f'Layer {outcome[1]} stopped growing: no plan exists.')
            return self._empty_plan()
        if kind == 'exhausted':
            self.status = PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY
            self.logs.append(f'No plan found within {max_layers} layers.')
            return self._empty_plan()
        steps = [self._construct_action(self._action_tuple(clingo.Function('occurs', [a, clingo.Number(0)])))
                 for a in outcome[1]]
        _plan = SequentialPlan(steps).replace_action_instances(self.compiled_task.map_back_action_instance)
        is_valid, reason = self.validate(_plan)
        if not is_valid:
            self.status = PlanGenerationResultStatus.INTERNAL_ERROR
            self.logs.append(f'Plan validation failed: {reason}')
            return self._empty_plan()
        self.status = PlanGenerationResultStatus.SOLVED_SATISFICING
        self.solved_horizon = len(steps)
        return _plan

    # ------------------------------------------------------------------
    # Plan extraction
    # ------------------------------------------------------------------

    def _extract_plan(self, symbols):
        """Build a plan from the model's occurs/2 atoms and lift it back onto
        the original problem via the compiler pipeline's composed map-back.

        A temporal task yields a `TimeTriggeredPlan`, everything else a
        `SequentialPlan`.
        """
        if self.compiled_task.is_temporal:
            return self._extract_time_triggered_plan(symbols)
        occurs = sorted((s for s in symbols if s.match('occurs', 2)),
                        key=lambda s: s.arguments[1].number)
        steps = [self._construct_action(self._action_tuple(s)) for s in occurs]
        _plan = SequentialPlan(steps)
        return _plan.replace_action_instances(self.compiled_task.map_back_action_instance)

    def _extract_time_triggered_plan(self, symbols) -> TimeTriggeredPlan:
        """Read a schedule off the model's occurs/2 and delta/2 atoms.

        Happening t sits at scaled time delta(1) + ... + delta(t) -- the
        encoding leaves the absolute times implicit and reports only the gaps.
        One snap action fires per happening, and a durative action may not
        overlap itself, so its start is closed by the next end of the same
        action; the pair becomes one entry with the elapsed time as duration.
        Scaled times are multiplied by the encoding's time unit to land back on
        the task's own time scale.
        """
        unit = self.compiled_task.time_unit
        snaps = self.compiled_task.snap_actions
        gaps = {s.arguments[0].number: s.arguments[1].number
                for s in symbols if s.match('delta', 2)}
        clock, happening_time = 0, {}
        for step in sorted(gaps):
            clock += gaps[step]
            happening_time[step] = clock

        steps, open_starts = [], {}
        for symbol in sorted((s for s in symbols if s.match('occurs', 2)),
                             key=lambda s: s.arguments[1].number):
            action_tuple = self._action_tuple(symbol)
            at = Fraction(happening_time.get(symbol.arguments[1].number, 0)) * unit
            snap = snaps.get(action_tuple[0])
            if snap is None:
                steps.append([at, self._construct_action(action_tuple), None])
                continue
            durative_name, half = snap
            key = (durative_name, action_tuple[1:])
            if half == 'start':
                open_starts[key] = len(steps)
                steps.append([at, self._construct_action(action_tuple), None])
            else:
                start = steps[open_starts.pop(key)]
                start[2] = at - start[0]
        if open_starts:
            raise ValueError(
                f"The ASP model leaves {sorted(k[0] for k in open_starts)} running at the "
                "horizon; the encoding's goal test should have ruled that out.")

        _plan = TimeTriggeredPlan([tuple(step) for step in steps])
        return _plan.replace_action_instances(self.compiled_task.map_back_action_instance)

    @staticmethod
    def _action_tuple(occurs_symbol) -> tuple:
        """occurs(action(("name", constant("o1"), ...)), T) -> (name, o1, ...).

        Zero-parameter actions appear as action(("name")), which clingo
        represents as the plain string (a one-element parenthesis is not
        a tuple term).
        """
        term = occurs_symbol.arguments[0].arguments[0]
        if term.type == clingo.SymbolType.String:
            return (term.string,)
        name = term.arguments[0].string
        args = []
        for t in term.arguments[1:]:
            if t.type == clingo.SymbolType.Function and t.name == 'constant':
                args.append(t.arguments[0].string)
            elif t.type == clingo.SymbolType.String:
                args.append(t.string)
            else:
                args.append(str(t))
        return (name, *args)

    def _construct_action(self, action_tuple) -> ActionInstance:
        name, *arg_names = action_tuple
        # A snap action is not an action of the compiled task: it is half of the
        # durative action the plan step really is.
        snap = self.compiled_task.snap_actions.get(name)
        if snap is not None:
            name = asp_name(snap[0])
        up_action = self._actions_by_asp_name.get(name)
        if up_action is None:
            raise ValueError(f"Action {name!r} from the ASP model not found in the compiled task.")
        up_args = []
        for arg in arg_names:
            up_object = self._objects_by_asp_name.get(arg)
            if up_object is None:
                raise ValueError(
                    f"Object {arg!r} (argument of action {name!r}) not found in the compiled task.")
            up_args.append(up_object)
        return ActionInstance(up_action, up_args)
