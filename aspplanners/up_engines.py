"""Unified Planning engine adapters for both backends, in one place.

Two ``OneshotPlanner`` engines are registered on ``import aspplanners``:

  * ``PLASPPlanner`` -> :class:`UPPLASPPlanner`, the PLASP/clingo backend
    (:class:`aspplanners.plasp.planner.PLASPPlanner`); the default, numeric-capable.
  * ``ABAPlanner``   -> :class:`UPABAPlanner`, the STRIPS-to-ABA/aspforaba backend
    (:class:`aspplanners.abaplan.planner.ABAPlan`); optional, needs the ``aba`` extra.

The ABA backend's ``aspforaba`` dependency is imported lazily inside the planner,
so importing this module (and registering both engines) works without it.
"""

from typing import Callable, IO, Optional

import unified_planning as up
from unified_planning.engines import (OptimalityGuarantee, PlanGenerationResult,
                                      PlanGenerationResultStatus)
from unified_planning.engines.results import LogMessage, LogLevel
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION

from aspplanners.common.temporal import DEFAULT_TIME_SCALE
from aspplanners.plasp.planner import PLASPPlanner
from aspplanners.abaplan.planner import ABAPlan


def _unsupported_result(engine: str, refusal: NotImplementedError) -> PlanGenerationResult:
    """A refusal, stated as UP states it.

    `supported_kind` answers for the task's *declared* features and is
    deliberately broad; a backend also refuses shapes only visible once it reads
    the task, and clingo refuses programs it cannot represent. All three mean
    "outside this engine's fragment", which is what UNSUPPORTED_PROBLEM says --
    so a caller learns it from the result rather than by knowing which exception
    type to catch. The reason travels in the log messages.

    `NotImplementedError` rather than `UnsupportedTaskError` is caught because
    that was the convention before the refusal type existed and most raise sites
    still use it; `UnsupportedTaskError` subclasses it, so both land here.
    """
    return PlanGenerationResult(
        PlanGenerationResultStatus.UNSUPPORTED_PROBLEM,
        None,
        engine,
        log_messages=[LogMessage(LogLevel.INFO,
                                 f"{engine} cannot encode this task: {refusal}")])


class UPPLASPPlanner(up.engines.Engine, up.engines.mixins.OneshotPlannerMixin):
    """UP engine adapter for the PLASP/clingo backend (engine name ``PLASPPlanner``).

    Recognised options:
      - ``encoding``    (str, default ``"seq"``)
      - ``horizon``     (int, solve at one fixed horizon)
      - ``max_horizon`` (int, default 1000, bound for the deepening search)
      - ``time_scale``  (int, default 10, resolution of the temporal encoding's
        integer happening times, as a multiple of the durations' gcd)
      - ``compilationlist`` (list of ``[engine_name, CompilationKind]`` pairs;
        takes over the automatic selection, see
        :meth:`PLASPPlanner._check_compilationlist`)
      - ``parallel`` (``"exists"`` or ``"forall"``; default off) -- a set of
        actions per horizon step, with the interference constraints generated
        on demand by a propagator (see :mod:`aspplanners.plasp.propagators`);
        not available on temporal tasks
      - ``interference`` (``"lazy"`` default, or ``"eager"``) -- when the
        propagator judges whether two actions interfere
    """

    def __init__(self, **options):
        up.engines.Engine.__init__(self)
        up.engines.mixins.OneshotPlannerMixin.__init__(self)
        self.conf = options

    @property
    def name(self) -> str:
        return "PLASPPlanner"

    @staticmethod
    def supported_kind():
        # Classical planning plus LINEAR numeric planning: integer/real-valued
        # fluents, linear comparisons, and increase/decrease/assign effects
        # whose value is a linear expression over the state -- a constant, or
        # `k1*V1 + ... + C` evaluated against the step before (numEffectExpr /
        # numAssignExpr in encodings/seq/numeric.lp). Negative, quantified and
        # disjunctive conditions need no compiling -- the encoding states all
        # of them (see PLASPPlanner._check_compilationlist).
        # Reals are accepted because PDDL (:functions ...) parse as real-typed.
        # A task stating fractional values is rescaled to whole ones before it is
        # encoded (clingo terms are integers; see aspplanners.plasp.rescale),
        # which the plan is unaffected by -- it is a sequence of actions.
        # GENERAL_NUMERIC_PLANNING is declared because an effect that reads a
        # fluent already puts a task's kind there; the feature has no
        # linear/non-linear split, so the shapes past linear are raised at
        # encoding time instead, like everything else ProblemKind cannot say:
        # a product of two fluents or a division by one (facts._linear_form), a
        # fractional coefficient in an effect (facts._effect_expr, since scaling
        # moves values, not coefficients), and a *bounded* numeric type on a
        # task that needs rescaling, whose bound would not move with the values
        # it bounds.
        kind = up.model.ProblemKind(version=LATEST_PROBLEM_KIND_VERSION)
        kind.set_problem_class('ACTION_BASED')
        kind.set_problem_type('SIMPLE_NUMERIC_PLANNING')
        kind.set_problem_type('GENERAL_NUMERIC_PLANNING')
        kind.set_typing('FLAT_TYPING')
        kind.set_typing('HIERARCHICAL_TYPING')
        kind.set_numbers('BOUNDED_TYPES')
        kind.set_fluents_type('INT_FLUENTS')
        kind.set_fluents_type('REAL_FLUENTS')
        kind.set_conditions_kind('NEGATIVE_CONDITIONS')
        kind.set_conditions_kind('DISJUNCTIVE_CONDITIONS')
        kind.set_conditions_kind('EQUALITIES')
        kind.set_conditions_kind('EXISTENTIAL_CONDITIONS')
        kind.set_conditions_kind('UNIVERSAL_CONDITIONS')
        kind.set_effects_kind('CONDITIONAL_EFFECTS')
        # A `forall` effect is encoded, not compiled away: its variables are
        # emitted free and the grounder ranges them, the same expansion a
        # `forall` *condition* gets. A conditional one indexes its effect term
        # by them, so each binding fires on its own condition (facts.ASPAction).
        # A *numeric* conditional effect is the one shape left out, and is
        # raised at encoding time -- numEffect/numAssign hang off occurs/2 with
        # no room for a condition; a forall numeric effect is fine.
        kind.set_effects_kind('FORALL_EFFECTS')
        kind.set_effects_kind('INCREASE_EFFECTS')
        kind.set_effects_kind('DECREASE_EFFECTS')
        kind.set_effects_kind('STATIC_FLUENTS_IN_NUMERIC_ASSIGNMENTS')
        kind.set_effects_kind('FLUENTS_IN_NUMERIC_ASSIGNMENTS')
        # Temporal planning over the PDDL 2.1 durative-action fragment, encoded
        # as SMTPlan's happenings (see encodings/seq/temporal.lp). Left
        # out on purpose: SELF_OVERLAPPING (a durative action may not overlap
        # itself), INTERMEDIATE_CONDITIONS_AND_EFFECTS (conditions and effects
        # sit at the two snap actions only), TIMED_EFFECTS/TIMED_GOALS, and
        # PROCESSES/EVENTS (PDDL+ continuous change).
        kind.set_time('CONTINUOUS_TIME')
        kind.set_time('DURATION_INEQUALITIES')
        kind.set_expression_duration('INT_TYPE_DURATIONS')
        kind.set_expression_duration('REAL_TYPE_DURATIONS')
        kind.set_expression_duration('STATIC_FLUENTS_IN_DURATIONS')
        # A fluent with no entry in the initial state gets a default laid down
        # before the facts are emitted (bool -> false, numeric -> 0, i.e. PDDL's
        # own closed-world reading), so a task that leaves some out is accepted;
        # see common.compilation.initialize_fluent_defaults.
        kind.set_initial_state('UNDEFINED_INITIAL_NUMERIC')
        kind.set_initial_state('UNDEFINED_INITIAL_SYMBOLIC')
        # Quality metrics are ACCEPTED, not optimised: the search stops at the
        # first plan it finds and reports SOLVED_SATISFICING (see `satisfies`).
        # Declaring them is what lets the engine take the IPC benchmarks, nearly
        # all of which carry a `(:metric minimize (total-time))` or
        # `(total-cost)` that has nothing to do with whether a plan exists.
        # OVERSUBSCRIPTION is left out on purpose -- there the metric holds the
        # soft goals, so ignoring it would change which plans count as solutions.
        kind.set_quality_metrics('MAKESPAN')
        kind.set_quality_metrics('ACTIONS_COST')
        kind.set_quality_metrics('PLAN_LENGTH')
        kind.set_quality_metrics('FINAL_VALUE')
        kind.set_actions_cost_kind('STATIC_FLUENTS_IN_ACTIONS_COST')
        kind.set_actions_cost_kind('FLUENTS_IN_ACTIONS_COST')
        kind.set_actions_cost_kind('INT_NUMBERS_IN_ACTIONS_COST')
        kind.set_actions_cost_kind('REAL_NUMBERS_IN_ACTIONS_COST')
        return kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= UPPLASPPlanner.supported_kind()

    @staticmethod
    def satisfies(optimality_guarantee) -> bool:
        return optimality_guarantee == OptimalityGuarantee.SATISFICING

    def _solve(self, problem: 'up.model.Problem',
               callback: Optional[Callable[['up.engines.PlanGenerationResult'], None]] = None,
               timeout: Optional[float] = None,
               output_stream: Optional[IO[str]] = None) -> 'up.engines.PlanGenerationResult':
        encoding = self.conf.get('encoding', 'seq')
        horizon = self.conf.get('horizon')
        max_horizon = self.conf.get('max_horizon', 1000)
        time_scale = int(self.conf.get('time_scale', DEFAULT_TIME_SCALE))
        compilationlist = self.conf.get('compilationlist')
        parallel = self.conf.get('parallel')
        interference = self.conf.get('interference', 'lazy')

        try:
            planner = PLASPPlanner(problem, encoding, compilationlist=compilationlist,
                                   time_scale=time_scale, parallel=parallel,
                                   interference=interference)
            plan = planner.plan(horizon=horizon, max_horizon=max_horizon, timeout=timeout)
        except NotImplementedError as refusal:
            return _unsupported_result(self.name, refusal)
        status = planner.status
        solved = status == PlanGenerationResultStatus.SOLVED_SATISFICING
        logs = [LogMessage(LogLevel.INFO, message) for message in planner.logs]
        return PlanGenerationResult(status, plan if solved else None, self.name, log_messages=logs)

    def destroy(self):
        pass


class UPABAPlanner(up.engines.Engine, up.engines.mixins.OneshotPlannerMixin):
    """UP engine adapter for the ABA backend (engine name ``ABAPlanner``).

    Recognised options:
      - ``max_horizon`` (int, default 1000)
      - ``semantics``   (str, default ``"ST"``)
      - ``time_scale``  (int, default 10, resolution of the temporal encoding's
        integer happening times, as a multiple of the durations' gcd)
    """

    def __init__(self, **options):
        up.engines.Engine.__init__(self)
        up.engines.mixins.OneshotPlannerMixin.__init__(self)
        self.conf = options

    @property
    def name(self) -> str:
        return "ABAPlanner"

    @staticmethod
    def supported_kind():
        # Classical planning plus simple numeric planning (integer/real fluents
        # with increase/decrease/assign effects and linear comparisons, via
        # finite-domain propositionalisation). Quantified/disjunctive/negative
        # conditions are compiled away by the UP compilers in the pipeline.
        # Conditional effects are NOT supported by the ABA encoding, and neither
        # are numeric over-all conditions (the reduction turns an over-all
        # condition into a static test on the actions that could break it, which
        # only exists for boolean ones) -- ProblemKind has no feature for that
        # distinction, so it is raised at encoding time instead.
        kind = up.model.ProblemKind(version=LATEST_PROBLEM_KIND_VERSION)
        kind.set_problem_class("ACTION_BASED")
        kind.set_problem_type("SIMPLE_NUMERIC_PLANNING")
        kind.set_typing("FLAT_TYPING")
        kind.set_typing("HIERARCHICAL_TYPING")
        kind.set_numbers("BOUNDED_TYPES")
        kind.set_fluents_type("INT_FLUENTS")
        kind.set_fluents_type("REAL_FLUENTS")
        kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        kind.set_conditions_kind("DISJUNCTIVE_CONDITIONS")
        kind.set_conditions_kind("EQUALITIES")
        kind.set_conditions_kind("EXISTENTIAL_CONDITIONS")
        kind.set_conditions_kind("UNIVERSAL_CONDITIONS")
        kind.set_effects_kind("INCREASE_EFFECTS")
        kind.set_effects_kind("DECREASE_EFFECTS")
        # A `forall` effect needs nothing of this backend: the pipeline opens
        # with up_quantifiers_remover, which expands it over the objects along
        # with the quantified conditions (see abaplan.encoder._PRE_COMPILERS).
        # An effect whose forall carries a `when` expands into conditional
        # effects, which this encoding does not have -- so such a task is
        # refused for CONDITIONAL_EFFECTS, as it already was.
        kind.set_effects_kind("FORALL_EFFECTS")
        # Temporal planning over the same PDDL 2.1 durative-action fragment the
        # PLASP backend covers; see UPPLASPPlanner.supported_kind.
        kind.set_time("CONTINUOUS_TIME")
        kind.set_time("DURATION_INEQUALITIES")
        kind.set_expression_duration("INT_TYPE_DURATIONS")
        kind.set_expression_duration("REAL_TYPE_DURATIONS")
        kind.set_expression_duration("STATIC_FLUENTS_IN_DURATIONS")
        # Quality metrics are accepted but never optimised, on the same terms as
        # the PLASP backend; see UPPLASPPlanner.supported_kind.
        #
        # UNDEFINED_INITIAL_NUMERIC / _SYMBOLIC are NOT declared here, though the
        # PLASP backend takes them: this reduction is over ground STRIPS, and the
        # grounder has to run before the encoder can fill the gaps in. No
        # installed UP grounder accepts a task with an undefined initial value
        # (up_grounder is the only one that takes the rest of this kind, and it
        # declares neither), and filling them in first is not a way out either --
        # it would give every unreachable ground action a 0-length duration.
        kind.set_quality_metrics("MAKESPAN")
        kind.set_quality_metrics("ACTIONS_COST")
        kind.set_quality_metrics("PLAN_LENGTH")
        kind.set_quality_metrics("FINAL_VALUE")
        kind.set_actions_cost_kind("STATIC_FLUENTS_IN_ACTIONS_COST")
        kind.set_actions_cost_kind("FLUENTS_IN_ACTIONS_COST")
        kind.set_actions_cost_kind("INT_NUMBERS_IN_ACTIONS_COST")
        kind.set_actions_cost_kind("REAL_NUMBERS_IN_ACTIONS_COST")
        return kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= UPABAPlanner.supported_kind()

    @staticmethod
    def satisfies(optimality_guarantee) -> bool:
        return optimality_guarantee == OptimalityGuarantee.SATISFICING

    def _solve(self, problem: "up.model.Problem",
               callback: Optional[Callable[["up.engines.PlanGenerationResult"], None]] = None,
               timeout: Optional[float] = None,
               output_stream: Optional[IO[str]] = None) -> "up.engines.PlanGenerationResult":
        max_horizon = int(self.conf.get("max_horizon", 1000))
        semantics = self.conf.get("semantics", "ST")
        time_scale = int(self.conf.get("time_scale", DEFAULT_TIME_SCALE))

        try:
            planner = ABAPlan(problem, time_scale=time_scale)
            plan = planner.plan(max_horizon=max_horizon, semantics=semantics)
        except NotImplementedError as refusal:
            return _unsupported_result(self.name, refusal)
        status = planner.status
        solved = status == PlanGenerationResultStatus.SOLVED_SATISFICING
        logs = [LogMessage(LogLevel.INFO, message) for message in planner.logs]
        return PlanGenerationResult(status, plan if solved else None, self.name, log_messages=logs)

    def destroy(self):
        pass
