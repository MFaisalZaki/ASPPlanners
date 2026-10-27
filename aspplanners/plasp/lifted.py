"""The lifted encoding: action schemas and their arguments chosen per step.

The seq encoding instantiates every ground action once, in base
(`action(action(("drive",T,A,B))) :- has(T,...), has(A,...), ...`), and then
replicates each step rule per ground action: the step choice ranges over them,
and `precondition/3` and `postcondition/4` are facts keyed by the ground action
term that one generic rule per step consumes. The size of a step is therefore
proportional to the number of ground actions -- objects to the power of the
action's arity -- which is what makes a task like transport's
`drive(truck, from, to)` or zenotravel's five-parameter `fly` expensive before
a single step is searched.

The lifted encoding (after Holler & Behnke, "Encoding Lifted Classical Planning
in Propositional Logic", ICAPS 2022) never names a ground action. A step
chooses a schema, `schema(S, t)`, and one object per parameter position,
`arg(S, I, O, t)`; every precondition and effect is a rule of its own that
joins only the argument atoms of the parameters its literal mentions:

    :- schema("drive", t), arg("drive", 1, T, t), arg("drive", 2, A, t),
       not holds(variable(("at",T,A)), value(variable(("at",T,A)), true), t-1).
    caused(variable(("at",T,B)), value(variable(("at",T,B)), true), t) :-
       schema("drive", t), arg("drive", 1, T, t), arg("drive", 3, B, t).

so a literal costs as many ground instances per step as the predicate has
ground atoms -- objects to the power of the *predicate's* arity -- and the
step choice itself costs one atom per (schema, position, object). Everything
downstream of `caused/3` (add-wins arbitration, inertia, the goal test) is
seq's core unchanged, and so are the task facts other than the actions: the
encoder here runs the seq encoder's compilation and replaces the action facts
by schema facts plus these step rules.

What this module reads is the same vocabulary the seq action builder reads
(:mod:`aspplanners.plasp.facts`), through the same parsers, so the two agree
literal for literal: positive and negative conditions, `forall` conditions
(one constraint per binding), equalities, static preconditions (checked
against the initial state rather than the state chain), unconditional and
conditional effects, `forall` effects (one rule firing per binding), and
`forall` inside a condition (a conditional literal, which is gringo's
conjunction over the bindings). Not covered by this first version, and refused
at construction with the reason: disjunctive conditions, numeric fluents and
durative actions -- each would need its layer restated in this per-schema form.
"""

from typing import List, Set

from unified_planning.model import DurativeAction

from aspplanners.plasp.encoder import ASPEncodingResult, PLASPEncoder
from aspplanners.plasp.facts import (
    ASPDisjunction, ASPEquality, ASPExpr, ASPNumComparison, _effect_bindings,
    _variable_terms, asp_name, flatten_atoms, parseexpr,
)
from aspplanners.plasp.layers import fact_predicates


def _parameter_names(fnode) -> Set[str]:
    """The action parameters an expression mentions, by name."""
    names = set()
    stack = [fnode]
    while stack:
        node = stack.pop()
        if node.is_parameter_exp():
            names.add(node.parameter().name)
        stack.extend(node.args)
    return names


class LiftedAction:
    """One schema's base facts and step rules."""

    def __init__(self, action, static_fluents: Set[str]):
        if isinstance(action, DurativeAction):
            raise ValueError("The lifted encoding has no temporal layer.")
        self.up_action = action
        self.name = asp_name(action.name)
        self.term = f'"{self.name}"'
        self.index = {p.name: i + 1 for i, p in enumerate(action.parameters)}
        self.var = {p.name: asp_name(p.name).upper() for p in action.parameters}
        self.static_fluents = static_fluents
        self.negated_fluents: Set[str] = set()
        self.facts: List[str] = [f'schema({self.term}).']
        for p in action.parameters:
            self.facts.append(
                f'param({self.term}, {self.index[p.name]}, type("{asp_name(p.type.name)}")).')
        self.step_rules: List[str] = []
        self._preconditions(action)
        self._effects(action)

    # -- pieces ------------------------------------------------------------

    def _args(self, fnode) -> List[str]:
        """`arg(S, I, V, t)` for every parameter the expression mentions."""
        return [f'arg({self.term}, {self.index[n]}, {self.var[n]}, t)'
                for n in sorted(_parameter_names(fnode), key=self.index.get)]

    def _guard(self, *fnodes) -> List[str]:
        atoms = [f'schema({self.term}, t)']
        seen = set()
        for fnode in fnodes:
            for atom in self._args(fnode):
                if atom not in seen:
                    seen.add(atom)
                    atoms.append(atom)
        return atoms

    @staticmethod
    def _holds(expr: ASPExpr, step: str) -> str:
        term = str(expr)
        return f'holds({term}, value({term}, {expr.value}), {step})'

    def _literal_body(self, expr, step='t-1') -> str:
        """A condition literal as a body element: a `holds` atom at `step`,
        turned into a conditional literal over its quantified variables so a
        `forall` inside a condition reads as the conjunction it is."""
        if isinstance(expr, ASPEquality):
            atom = str(expr)
        else:
            atom = self._holds(expr, step)
        return f'{atom} : {", ".join(expr.bindings)}' if expr.bindings else atom

    def _note_negated(self, expr) -> None:
        if isinstance(expr, ASPExpr) and expr.up_expr.is_fluent_exp() and expr.value == 'false':
            self.negated_fluents.add(expr.up_expr._content.payload.name)

    @staticmethod
    def _refuse(what, where) -> None:
        raise ValueError(
            f"The lifted encoding cannot state {what} in {where}: disjunctive and numeric "
            "conditions are not in its first version. Use the 'seq' encoding for this task.")

    # -- preconditions -------------------------------------------------------

    def _preconditions(self, action) -> None:
        for precondition in action.preconditions:
            for literal in flatten_atoms(parseexpr(precondition)):
                if isinstance(literal, (ASPDisjunction, ASPNumComparison)):
                    self._refuse("a disjunctive or numeric condition", f"action {action.name}")
                body = self._guard(literal.up_expr)
                if isinstance(literal, ASPEquality):
                    # `=` holds unless the sides differ; `!=` unless they agree.
                    test = f'{literal.lhs} != {literal.rhs}' if literal.value == 'true' \
                        else f'{literal.lhs} = {literal.rhs}'
                    body += list(literal.bindings) + [test]
                    self.step_rules.append(f":- {', '.join(body)}.")
                    continue
                self._note_negated(literal)
                term = str(literal)
                fluent = literal.up_expr._content.payload.name if literal.up_expr.is_fluent_exp() else None
                body += list(literal.bindings)
                if fluent in self.static_fluents and literal.value == 'true':
                    # A static fluent is what the initial state says, at every
                    # step: no need to walk its holds/3 chain.
                    body.append(f'not initialState({term}, value({term}, true))')
                else:
                    body.append(f'not {self._holds(literal, "t-1")}')
                self.step_rules.append(f":- {', '.join(body)}.")

    # -- effects -------------------------------------------------------------

    def _effects(self, action) -> None:
        for eff in action.unconditional_effects:
            if not eff.fluent.type.is_bool_type():
                self._refuse("a numeric effect", f"action {action.name}")
            target = parseexpr(eff.fluent)
            body = self._guard(eff.fluent) + _effect_bindings(eff, target)
            term = str(target)
            value = str(eff.value).lower()
            self.step_rules.append(
                f"caused({term}, value({term}, {value}), t) :- {', '.join(body)}.")
        for eff in action.conditional_effects:
            if not eff.fluent.type.is_bool_type():
                self._refuse("a conditional numeric effect", f"action {action.name}")
            target = parseexpr(eff.fluent)
            _names, forall_bindings = _variable_terms(eff.forall)
            body = self._guard(eff.fluent, eff.condition) + forall_bindings + list(target.bindings)
            for literal in flatten_atoms(parseexpr(eff.condition)):
                if isinstance(literal, (ASPDisjunction, ASPNumComparison)):
                    self._refuse("a disjunctive or numeric effect condition", f"action {action.name}")
                self._note_negated(literal)
                if isinstance(literal, ASPEquality):
                    body += list(literal.bindings) + [str(literal)]
                else:
                    body.append(self._literal_body(literal))
            term = str(target)
            value = str(eff.value).lower()
            self.step_rules.append(
                f"caused({term}, value({term}, {value}), t) :- {', '.join(body)}.")


class LiftedEncoder(PLASPEncoder):
    """The seq encoder's compilation, with the action facts replaced by the
    lifted schema facts and per-schema step rules."""

    UNSUPPORTED = (
        ('numeric', 'its fluents would need the numeric layer restated per schema'),
        ('disjunctive', 'disjunctive conditions would need the disjunctive layer restated per schema'),
        ('temporal', 'happenings are built on one ground snap action per step'),
    )

    def compile(self, problem, compilationlist=None) -> ASPEncodingResult:
        result = super().compile(problem, compilationlist)
        self.refuse_unsupported(result)
        task = result.problem
        modified = set()
        for action in task.actions:
            for eff in action.effects:
                modified.add(eff.fluent._content.payload.name)
        static_fluents = {f.name for f in task.fluents if f.name not in modified}
        schemas = [LiftedAction(action, static_fluents) for action in task.actions]
        facts = dict(result.facts)
        facts['_actions'] = {line for schema in schemas for line in schema.facts}
        # The initial state the seq compile emitted already carries the false
        # value of every fluent some literal reads as false: the lifted rules
        # read exactly the same literals (same parsers, same `negated_fluents`
        # bookkeeping), so it is reused as is. Pinned by a test.
        return ASPEncodingResult(
            problem=task,
            facts=facts,
            map_back_action_instance=result.map_back_action_instance,
            snap_actions=result.snap_actions,
            time_unit=result.time_unit,
            numeric_scale=result.numeric_scale,
            is_temporal=result.is_temporal,
            step_rules=[rule for schema in schemas for rule in schema.step_rules],
            lifted=True,
        )

    @classmethod
    def refuse_unsupported(cls, result: ASPEncodingResult) -> None:
        """Raise, with the reason, when the task needs a layer the lifted
        family does not have. Judged on the seq compile's fact vocabulary, the
        same signal the registry's coverage check uses."""
        from aspplanners.plasp.layers import LIFTED
        LIFTED.refuse_unsupported(fact_predicates(result.fact_lines))
