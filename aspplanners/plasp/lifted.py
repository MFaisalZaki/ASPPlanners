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

Numeric fluents are stated the same way. The seq numeric layer keys a
comparison or an effect by the action term (`numPrecondition(A, ...)`,
`numEffect(A, V, D)`) and one generic rule per step consumes them; here each
becomes a per-schema step rule with the same body -- the comparison's
difference #sum, the delta applied to the value at t-1, the expression-valued
forms through numValue/3 -- under the argument atoms of the parameters it
mentions, and the numConst/numTerm declarations of its expressions are stated
under the parameters' type atoms rather than under a ground action. The parts
of the numeric layer keyed by expression or fluent (numval, numValue, the
delta sum, the numeric goals) are used as they are. Deltas on one fluent are
summed into one effect per schema, as the seq builder does.

What this module reads is the same vocabulary the seq action builder reads
(:mod:`aspplanners.plasp.facts`), through the same parsers, so the two agree
literal for literal: positive and negative conditions, `forall` conditions
(one constraint per binding), equalities, static preconditions (checked
against the initial state rather than the state chain), unconditional and
conditional boolean effects, `forall` effects (one rule firing per binding),
`forall` inside a condition (a conditional literal, which is gringo's
conjunction over the bindings), linear numeric preconditions and goals, and
constant and expression-valued increase, decrease and assign effects.

Disjunctive conditions use seq's disjunctive layer as it is. Its facts are
keyed by a ground action term and its one action-level constraint reads
`occurs(A, t)`, so a schema with a disjunctive precondition gets the same
orGroup/orDisjunct facts seq would build for it and one bridging step rule
deriving the key from the step's schema and arguments:

    occurs(disj(("stop",0,F)), t) :- schema("stop", t), arg("stop", 1, F, t).

The key names the schema, the disjunction and only the parameters the
condition mentions, so the facts range over those parameters' types rather
than the whole signature (seq ranges its over the action declaration, which
for a wide schema is objects to the power of its arity): one derived atom per
binding of those parameters per step, paid only by the schemas that have a
disjunction. The goal rules and the effect-condition
groups read the state alone and need no bridge. A conditional effect whose
condition is disjunctive carries the groups on an effect term of its own and
requires them in its `caused` rule the way core's does.

Not covered, and refused at construction with the reason: conditional numeric
effects, numeric conditions on a conditional effect, and durative actions.
"""

from fractions import Fraction
from typing import List, Set

from unified_planning.model import DurativeAction, EffectKind

from aspplanners.plasp.encoder import ASPEncodingResult, PLASPEncoder

from aspplanners.plasp.facts import (
    ASPDisjunction, ASPEquality, ASPExpr, ASPNumComparison, _dedup, _effect_bindings,
    _effect_constant, _effect_expr, _fold_fractional_statics, _lin_add,
    _lin_scale, _linear_form, _or_facts, _reads_fluents, _variable_terms,
    asp_name, flatten_atoms, parseexpr,
)
from aspplanners.plasp.numeric_terms import NO_FOLDING
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


# The violating side of each comparison, as the #sum of the sides' difference
# against the constants' gap: `S1 < S2` fails when T1 - T2 >= C2 - C1, and so on
# (the same table numeric.lp's four precondition rules spell out).
_VIOLATES = {'lt': '>=', 'le': '>', 'eq': '!=', 'neq': '='}


class LiftedAction:
    """One schema's base facts and step rules (boolean and numeric apart)."""

    def __init__(self, action, static_fluents: Set[str], context=NO_FOLDING):
        if isinstance(action, DurativeAction):
            raise ValueError("The lifted encoding has no temporal layer.")
        self.up_action = action
        self.context = context
        self.name = asp_name(action.name)
        self.term = f'"{self.name}"'
        self.index = {p.name: i + 1 for i, p in enumerate(action.parameters)}
        self.var = {p.name: asp_name(p.name).upper() for p in action.parameters}
        self.types = {p.name: f'has({self.var[p.name]}, type("{asp_name(p.type.name)}"))'
                      for p in action.parameters}
        self.static_fluents = static_fluents
        self.negated_fluents: Set[str] = set()
        self._group_index = 0
        # The schema's positive static preconditions, as (parameter names,
        # initialState atom): the narrowing seq folds into its action
        # declaration, applied here to the keys of the disjunction facts.
        self._static_guards: List[tuple] = []
        self.facts: List[str] = [f'schema({self.term}).']
        for p in action.parameters:
            self.facts.append(
                f'param({self.term}, {self.index[p.name]}, type("{asp_name(p.type.name)}")).')
        self.step_rules: List[str] = []
        self.numeric_step_rules: List[str] = []
        self._preconditions(action)
        self._effects(action)
        self._numeric_effects(action)

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

    def _type_atoms(self, *fnodes) -> List[str]:
        """`has(V, type(...))` for every parameter the expressions mention: the
        body under which an expression's declarations are stated in base, so
        they exist for every binding (where seq states them under the ground
        action)."""
        names = set()
        for fnode in fnodes:
            names |= _parameter_names(fnode)
        return [self.types[n] for n in sorted(names, key=self.index.get)]

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
            f"The lifted encoding cannot state {what} in {where}: numeric conditions on "
            "an effect are not in its first version. Use the 'seq' encoding for this task.")

    # -- preconditions -------------------------------------------------------

    def _key_body(self, names) -> List[str]:
        """The body ranging a key over parameters `names`: their type atoms,
        narrowed by every positive static precondition of the schema whose
        parameters are among them -- seq's own declaration folding, without
        which airport's `move(a, t, d1, s1, s2, d2)` ranges a key over every
        pair of segments where `move-dir(s1, s2, d2)` admits a few hundred."""
        names = set(names)
        return [self.types[n] for n in sorted(names, key=self.index.get)] + \
               [atom for guard_names, atom in self._static_guards if guard_names <= names]

    def _disjunction(self, disjunction: ASPDisjunction, fnode) -> None:
        """seq's orGroup/orDisjunct facts for one disjunctive precondition,
        keyed by a term of this schema and the parameters the condition
        mentions (`fnode`), ranged by their types and static preconditions --
        not by the whole signature, which for a wide schema is objects to the
        power of its arity. The layer's one constraint reads occurs(A, t) for
        the key, so a bridging step rule derives it from the step's schema and
        arguments."""
        names = sorted(_parameter_names(fnode), key=self.index.get)
        key = f'disj(({self.term},{self._group_index}' + ''.join(f',{self.var[n]}' for n in names) + '))'
        self.facts += _or_facts(disjunction, '0', owner=key, body_prefix=self._key_body(names))
        guard = [f'schema({self.term}, t)'] + [f'arg({self.term}, {self.index[n]}, {self.var[n]}, t)' for n in names]
        self.step_rules.append(f"occurs({key}, t) :- {', '.join(guard)}.")
        self.negated_fluents |= disjunction.negated_fluent_names
        self._group_index += 1

    def _preconditions(self, action) -> None:
        conjuncts = [c for pre in action.preconditions for c in (pre.args if pre.is_and() else [pre])]
        # The static guards first, so every disjunction can be narrowed by them.
        for precondition in conjuncts:
            for literal in flatten_atoms(parseexpr(precondition, context=self.context)):
                if isinstance(literal, ASPExpr) and literal.up_expr.is_fluent_exp() \
                        and literal.up_expr.fluent().name in self.static_fluents \
                        and literal.value == 'true' and not literal.bindings:
                    term = str(literal)
                    self._static_guards.append((frozenset(_parameter_names(literal.up_expr)),
                                                f'initialState({term}, value({term}, true))'))
        for precondition in conjuncts:
            for literal in flatten_atoms(parseexpr(precondition, context=self.context)):
                if isinstance(literal, ASPDisjunction):
                    self._disjunction(literal, precondition)
                    continue
                if isinstance(literal, ASPNumComparison):
                    self._numeric_precondition(literal)
                    continue
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

    def _numeric_precondition(self, cmp: ASPNumComparison) -> None:
        """numeric.lp's precondition rule, per schema: the two sides declared
        under the parameters' types, and the violating comparison of their
        difference #sum as a constraint under the schema's arguments."""
        decl_body = _dedup(self._type_atoms(cmp.up_expr) + list(cmp.bindings))
        self.facts += cmp.declaration_rules(decl_body)
        lhs, rhs = str(cmp.lhs), str(cmp.rhs)
        body = self._guard(cmp.up_expr) + list(cmp.bindings) + [
            f'numConst({lhs}, NC1)', f'numConst({rhs}, NC2)',
            f'#sum {{ NX * NK, NV, 1 : numTerm({lhs}, NK, NV), numval(NV, NX, t-1) ; '
            f'-(NX * NK), NV, 2 : numTerm({rhs}, NK, NV), numval(NV, NX, t-1) }} '
            f'{_VIOLATES[cmp.op]} NC2 - NC1']
        self.numeric_step_rules.append(f":- {', '.join(_dedup(body))}.")

    # -- effects -------------------------------------------------------------

    def _effects(self, action) -> None:
        for eff in action.unconditional_effects:
            if not eff.fluent.type.is_bool_type():
                continue   # numeric: see _numeric_effects
            target = parseexpr(eff.fluent)
            body = self._guard(eff.fluent) + _effect_bindings(eff, target)
            term = str(target)
            value = str(eff.value).lower()
            self.step_rules.append(
                f"caused({term}, value({term}, {value}), t) :- {', '.join(body)}.")
        for effect_index, eff in enumerate(action.conditional_effects):
            if not eff.fluent.type.is_bool_type():
                raise ValueError(
                    f"The lifted encoding cannot state the conditional numeric effect on "
                    f"{eff.fluent} in action {action.name}: conditional numeric effects are "
                    "not in its first version. Use the 'seq' encoding for this task.")
            target = parseexpr(eff.fluent)
            forall_names, forall_bindings = _variable_terms(eff.forall)
            body = self._guard(eff.fluent, eff.condition) + forall_bindings + list(target.bindings)
            # The effect's own term, for the groups of a disjunctive condition:
            # the parameters the condition mentions and the effect's forall
            # variables tell one binding's groups from another's.
            cond_names = sorted(_parameter_names(eff.condition), key=self.index.get)
            effect_vars = [self.var[n] for n in cond_names] + list(forall_names)
            effect_term = f'effect(({self.term},{effect_index}' + ''.join(f',{v}' for v in effect_vars) + '))'
            groups = 0
            for literal in flatten_atoms(parseexpr(eff.condition)):
                if isinstance(literal, ASPNumComparison):
                    self._refuse("a numeric effect condition", f"action {action.name}")
                if isinstance(literal, ASPDisjunction):
                    self.facts += _or_facts(literal, str(groups), owner=effect_term,
                                            body_prefix=_dedup(self._key_body(cond_names) + forall_bindings))
                    self.negated_fluents |= literal.negated_fluent_names
                    groups += 1
                    continue
                self._note_negated(literal)
                if isinstance(literal, ASPEquality):
                    body += list(literal.bindings) + [str(literal)]
                else:
                    body.append(self._literal_body(literal))
            if groups:
                # Every group of the effect term has to hold, as core's caused/3
                # demands of seq's effect terms; a group under a `forall` is one
                # per binding, which the conditional literal ranges over.
                body.append(f'orGroupHolds({effect_term}, G, t) : orGroup({effect_term}, G)')
            term = str(target)
            value = str(eff.value).lower()
            # Joined with ';', not ',': a condition literal over quantified
            # variables is a conditional literal, and after one a ','
            # continues its condition instead of starting the next body
            # literal -- a second `forall` condition then fails to parse, and a
            # plain atom after one is silently absorbed into the condition.
            self.step_rules.append(
                f"caused({term}, value({term}, {value}), t) :- {'; '.join(body)}.")


    def _numeric_effects(self, action) -> None:
        """numeric.lp's effect rules, per schema. Deltas on one fluent are
        summed into one linear form first (two increases of `x` are one update),
        as the seq builder does; an assignment is stated on its own."""
        context = self.context
        deltas = {}      # target term -> (summed form, scale, body atoms)
        for eff in action.unconditional_effects:
            if eff.fluent.type.is_bool_type():
                continue
            target = parseexpr(eff.fluent)
            term = str(target)
            scale = context.scale(eff.fluent.fluent().name)
            body = self._guard(eff.fluent, eff.value) + _effect_bindings(eff, target)
            if eff.kind in (EffectKind.INCREASE, EffectKind.DECREASE):
                form = _linear_form(eff.value, context)
                if eff.kind == EffectKind.DECREASE:
                    form = _lin_scale(form, Fraction(-1))
                previous = deltas.get(term)
                deltas[term] = (_lin_add(previous[0], form) if previous else form, scale,
                                _dedup((previous[2] if previous else []) + body),
                                [eff.fluent, eff.value] + (previous[3] if previous else []))
                continue
            # An assignment.
            form = _fold_fractional_statics(_linear_form(eff.value, context), context)
            if _reads_fluents(form):
                expr = _effect_expr(form, term, scale, context)
                self._declare_expression(expr, [eff.fluent, eff.value])
                rule_body = _dedup(body + expr.bindings + [f'numValue({str(expr)}, ND, t-1)'])
                self.numeric_step_rules.append(
                    f"caused({term}, value({term}, ND), t) :- {', '.join(rule_body)}.")
            else:
                value = _effect_constant(form, term, scale, context)
                rule_body = _dedup(body + list(context.body(form[1])))
                self.numeric_step_rules.append(
                    f"caused({term}, value({term}, {value}), t) :- {', '.join(rule_body)}.")
        for term, (form, scale, body, fnodes) in deltas.items():
            form = _fold_fractional_statics(form, context)
            if _reads_fluents(form):
                expr = _effect_expr(form, term, scale, context)
                self._declare_expression(expr, fnodes)
                rule_body = _dedup(body + expr.bindings + [
                    f'holds({term}, value({term}, NX), t-1)', f'numValue({str(expr)}, ND, t-1)'])
                self.numeric_step_rules.append(
                    f"caused({term}, value({term}, NX + ND), t) :- {', '.join(rule_body)}.")
                continue
            if form[1].is_zero():
                continue
            delta = _effect_constant(form, term, scale, context)
            rule_body = _dedup(body + list(context.body(form[1])) + [f'holds({term}, value({term}, NX), t-1)'])
            self.numeric_step_rules.append(
                f"caused({term}, value({term}, NX + {delta}), t) :- {', '.join(rule_body)}.")

    def _declare_expression(self, expr, fnodes) -> None:
        """An expression an effect evaluates through numValue/3: declared
        (numConst/numTerm) and marked numValueNeeded, under the parameters'
        types -- numeric.lp marks it off the numEffectExpr fact seq emits,
        which the lifted encoding has not."""
        body = _dedup(self._type_atoms(*fnodes) + list(expr.bindings))
        self.facts += expr.declaration_rules(body)
        tail = f" :- {', '.join(body)}." if body else "."
        self.facts.append(f"numValueNeeded({str(expr)}){tail}")


class LiftedEncoder(PLASPEncoder):
    """The seq encoder's compilation, with the action facts replaced by the
    lifted schema facts and per-schema step rules."""

    def compile(self, problem, compilationlist=None) -> ASPEncodingResult:
        result = super().compile(problem, compilationlist)
        self.refuse_unsupported(result)
        task = result.problem
        schemas = [LiftedAction(action, self.static_fluent_names, self.numeric_context)
                   for action in task.actions]
        facts = dict(result.facts)
        facts['_actions'] = {line for schema in schemas for line in schema.facts}
        # The symmetry classes carry over; the objects a step mentions are read
        # off arg/4 by lifted/symmetry.lp, so seq's mentions/2 rules do not.
        facts['_symmetry'] = {line for line in facts.get('_symmetry', ()) if not line.startswith('mentions(')}
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
            numeric_step_rules=[rule for schema in schemas for rule in schema.numeric_step_rules],
            lifted=True,
            symmetry_classes=result.symmetry_classes,
        )

    @classmethod
    def refuse_unsupported(cls, result: ASPEncodingResult) -> None:
        """Raise, with the reason, when the task needs a layer the lifted
        family does not have. Judged on the seq compile's fact vocabulary, the
        same signal the registry's coverage check uses."""
        from aspplanners.plasp.layers import LIFTED
        LIFTED.refuse_unsupported(fact_predicates(result.fact_lines))
