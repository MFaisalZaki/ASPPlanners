"""PLASP fact builders: render a compiled UP problem into the ASP fact
vocabulary (``variable``/``action``/``precondition``/``numGoal``/...) consumed
by the encodings in ``aspplanners/plasp/encodings``.

This is a recreation of the PLASP tool's translation. Every builder is an
:class:`~aspplanners.lp_io.ASPTerm`, so its identity is its rendered ASP text
and sets of facts deduplicate accordingly.
"""

from fractions import Fraction
from math import gcd

from unified_planning.shortcuts import FNode, EffectKind

from aspplanners.lp_io import ASPTerm
from aspplanners.plasp.numeric_terms import (
    Coeff,
    InexactCoefficient,
    NO_FOLDING,
    NumericContext,
)

# clingo terms are signed 32-bit and wrap around *silently*: `b(2147483648)`
# grounds to `b(-2147483648)` and `X = 2147483647*2` to `-2`, with no warning.
# A constant that does not fit would therefore yield a wrong plan rather than an
# error, so everything that reaches the fact text is range-checked.
CLINGO_INT_MAX = 2 ** 31 - 1


def asp_name(name):
    """ASP-side rendering of a UP name: PDDL hyphens become underscores.

    All names are emitted inside quoted clingo strings (where hyphens would
    be legal), but the underscore vocabulary is the contract downstream .lp
    encodings match on, and action parameters become ASP *variables* where
    hyphens are not legal. The planner maps model atoms back through this
    same function; the encoder rejects tasks where the mapping collides.
    """
    return name.replace('-', '_')

def _is_numeric_fnode(f):
    """Is this expression integer/real-valued (fluent, constant, or arithmetic)?"""
    return (f.is_int_constant() or f.is_real_constant() or f.is_plus() or f.is_minus()
            or (f.is_fluent_exp() and not f.type.is_bool_type()))

def flatten_atoms(parsed):
    """`parseexpr`'s result as a flat list; nested lists are conjunctions."""
    atoms = parsed if isinstance(parsed, list) else [parsed]
    while any(isinstance(a, list) for a in atoms):
        atoms = [e for a in atoms for e in (a if isinstance(a, list) else [a])]
    return atoms


class ASPDisjunct:
    """One disjunct of an :class:`ASPDisjunction`: everything in it has to hold.

    The parsed disjunct body splits three ways. `atoms` are the state literals,
    which become ``orDisjunct``/``orDisjunctNum`` facts. `tests` are equality
    constraints between non-fluent terms, which the grounder settles: they go in
    the *body* of the disjunct's rules, so a disjunct whose test fails is simply
    never declared and cannot satisfy its group. `subgroups` are nested
    disjunctions, which become groups of their own (see :func:`_or_facts`).

    `id_term` identifies the disjunct within its group, `own_vars` are the ASP
    variables an ``exists`` disjunct introduces -- they index its id and stay in
    scope for its subgroups -- and `bindings` are the ``has(_, type(...))`` atoms
    the disjunct's rule bodies need.
    """

    def __init__(self, id_term, parsed, own_vars=(), extra_bindings=()):
        self.id = id_term
        self.own_vars = list(own_vars)
        self.atoms, self.tests, self.subgroups = [], [], []
        for item in flatten_atoms(parsed):
            if isinstance(item, ASPDisjunction):
                self.subgroups.append(item)
            elif isinstance(item, ASPEquality):
                self.tests.append(item)
            else:
                self.atoms.append(item)
        self.bindings = sorted(set(extra_bindings) |
                               {b for item in self.atoms + self.tests for b in item.bindings})
        # The bindings the disjunct's *declaration* depends on: its own
        # `exists` variables and its tests'. An atom's bindings are not among
        # them -- a `forall` literal inside the disjunct ranges over them, and
        # over an empty type it is vacuously true, so the disjunct has to exist
        # (and hold) with no literal at all, not vanish and fail its group.
        self.own_bindings = sorted(set(extra_bindings) |
                                   {b for item in self.tests for b in item.bindings})


class ASPDisjunction:
    """A disjunctive condition: at least one of its disjuncts has to hold.

    Not an :class:`~aspplanners.lp_io.ASPTerm`, because it has no rendering of
    its own -- it is emitted as ``orGroup``/``orDisjunct`` facts by whoever owns
    the condition, which is the only place the action or goal it belongs to is
    known (see :func:`_or_facts`).

    An ``or`` numbers its disjuncts 0, 1, ...; an ``exists`` is the same
    structure with the disjuncts indexed by the quantified variable's binding,
    so the grounder enumerates them instead of a compiler. A disjunction under a
    ``forall`` additionally carries that quantifier's variables as `group_vars`
    (see :meth:`universally_bound`).
    """

    def __init__(self, disjuncts, group_vars=(), group_bindings=()):
        self.disjuncts = disjuncts
        self.group_vars = list(group_vars)
        self.group_bindings = list(group_bindings)

    def universally_bound(self, names, bindings):
        """Put this disjunction under a ``forall`` over `names`.

        The quantified variables end up in the *group's* term, so the grounder
        makes one group per binding and every one of them has to hold. Without
        that, all bindings would pool into the disjuncts of a single group,
        which states ``(forall x . p(x)) or (forall x . q(x))`` rather than the
        ``forall x . (p(x) or q(x))`` that was written.
        """
        self.group_vars = list(names) + self.group_vars
        self.group_bindings = list(bindings) + self.group_bindings
        return self

    @property
    def negated_fluent_names(self):
        # Numeric literals are read off numval, not off a `holds(V, false)`
        # chain, so they contribute nothing here (and carry no `value`).
        names = set()
        for disjunct in self.disjuncts:
            names |= {atom.up_expr._content.payload.name for atom in disjunct.atoms
                      if isinstance(atom, ASPExpr) and atom.up_expr.is_fluent_exp()
                      and atom.value == 'false'}
            for subgroup in disjunct.subgroups:
                names |= subgroup.negated_fluent_names
        return names


def _variable_terms(variables):
    """``(ASP variable names, has(...) bindings)`` for UP quantified variables.

    The bindings are what range a variable over the universe, so emitting a rule
    that carries them is what expands the quantifier -- in the grounder, not in a
    compiler. Both places a quantifier can appear share this: a ``forall``/
    ``exists`` in a condition (whose variables come off the FNode) and a
    ``forall`` *effect* (whose variables come off the UP ``Effect``).
    """
    variables = list(variables)
    names = [QUANTIFIED_PREFIX + asp_name(v.name).upper() for v in variables]
    bindings = [f'has({name}, {str(ASPType(v.type))})' for name, v in zip(names, variables)]
    return names, bindings


def _quantified_vars(f):
    """``(ASP variable names, has(...) bindings)`` for a quantifier's variables."""
    return _variable_terms(f.variables())


def _quantified_id(names):
    """The disjunct id an ``exists``'s variables index its disjuncts by."""
    return names[0] if len(names) == 1 else f"({','.join(names)})"


def _as_disjunct(f, t, index, context=NO_FOLDING):
    """The disjunct one argument of a disjunction contributes.

    A conjunctive argument is one disjunct numbered `index`; an ``exists`` is a
    family of them indexed by its variables' bindings, which is what lets an
    `or` and an `exists` share a vocabulary.
    """
    quantified = f.is_exists() if t != 'false' else f.is_forall()
    if quantified:
        names, bindings = _quantified_vars(f)
        return ASPDisjunct(f"({index},{_quantified_id(names)})", parseexpr(f.arg(0), t, context),
                           own_vars=names, extra_bindings=bindings)
    return ASPDisjunct(str(index), parseexpr(f, t, context))


def parseexpr(f, t=None, context=NO_FOLDING):
    """!
    given a FNode from UP representing a fluent and a possible timestep return a
    string representation of it.

    `context` is the task's :class:`~aspplanners.plasp.numeric_terms.NumericContext`
    -- the per-fluent value scales and the static fluents whose values the
    grounder may read out of the initial state. It only ever reaches a numeric
    comparison, and defaults to the no-folding context so that a caller with no
    task-level knowledge (a test, or the ABA side) gets the plain rational
    arithmetic this had before folding existed.
    """
    assert isinstance(f, FNode), f"Expected a FNode, got {type(f)}"
    if  f.is_fluent_exp(): # for fluents
        return ASPExpr(f, 'true' if t is None else t)
    if f.is_bool_constant():
        return ASPExpr(f, str(f).lower())
    if f.is_not():
        # A negation flips the polarity it is parsed under rather than forcing
        # it, so a double negation (which De Morgan on a nested `not (and ...)`
        # readily produces) cancels instead of staying negative.
        return parseexpr(f.args[0], 'true' if t == 'false' else 'false', context)
    if f.is_implies():
        # `a -> b` is `not a or b`, so restating it that way hands it to the
        # disjunction machinery below (and to De Morgan when it is negated),
        # rather than needing a compiler to have removed it beforehand.
        manager = f.environment.expression_manager
        return parseexpr(manager.Or(manager.Not(f.arg(0)), f.arg(1)), t, context)
    negated = t == 'false'
    # A conjunction is what precondition/goal facts *are*, so it just flattens --
    # and so does a `forall`, which is a conjunction over the universe, emitted
    # with its variable left free for the grounder to range (see _arg_term).
    # De Morgan swaps which is which under negation.
    if f.is_and() if not negated else f.is_or():
        return [parseexpr(arg, t, context) for arg in f.args]
    if f.is_forall() if not negated else f.is_exists():
        # The body's atoms keep the variable free, so every one of them is
        # emitted once per binding -- a conjunction over the universe. A
        # *disjunction* in the body cannot be left free that way, though: it has
        # to become one group per binding, which is what universally_bound
        # records for whoever emits it.
        parsed = parseexpr(f.arg(0), t, context)
        names, bindings = _quantified_vars(f)
        for item in flatten_atoms(parsed):
            if isinstance(item, ASPDisjunction):
                item.universally_bound(names, bindings)
        return parsed
    # A disjunction is not conjunctive, so it becomes an ASPDisjunction that the
    # owning action or goal emits as its own orGroup/orDisjunct facts. An
    # `exists` is one too, with the disjuncts indexed by its variables.
    if f.is_or() or f.is_and():
        return ASPDisjunction([_as_disjunct(arg, t, i, context) for i, arg in enumerate(f.args)])
    if f.is_exists() or f.is_forall():
        names, bindings = _quantified_vars(f)
        return ASPDisjunction([ASPDisjunct(_quantified_id(names), parseexpr(f.arg(0), t, context),
                                           own_vars=names, extra_bindings=bindings)])
    if f.is_lt() or f.is_le():
        cmp = ASPNumComparison(f, 'lt' if f.is_lt() else 'le', context)
        return cmp.negated() if t == 'false' else cmp
    if f.is_equals() and (_is_numeric_fnode(f.args[0]) or _is_numeric_fnode(f.args[1])):
        cmp = ASPNumComparison(f, 'eq', context)
        return cmp.negated() if t == 'false' else cmp
    if f.is_equals():
        lhs, rhs = f.args[0], f.args[1]
        # Object-fluent equality: encode as a normal precondition where the
        # fluent variable's value is the constant/parameter on the other side.
        if lhs.is_fluent_exp() or rhs.is_fluent_exp():
            if t == 'false':
                raise NotImplementedError(
                    "Negated object-fluent equality is not supported in the plasp encoding."
                )
            fluent_side, value_side = (lhs, rhs) if lhs.is_fluent_exp() else (rhs, lhs)
            if value_side.is_fluent_exp():
                raise NotImplementedError(
                    "Equality between two object-fluent expressions is not supported."
                )
            return ASPExpr(fluent_side, _equality_value_term(value_side))
        # Parameter/object equality: lifted-plasp folds this into the action
        # signature body so only matching bindings instantiate the action.
        return ASPEquality(f, 'true' if t is None else t)
    raise TypeError(f"Unsupported thing: {f} of type {type(f)}")

def is_numeric_comparison(f):
    """Is this FNode a numeric comparison (or the negation of one)?

    ``lt``/``le`` and numeric ``=`` are the linear comparisons the ASP
    encoding evaluates against ``numval``; UP folds ``GE``/``GT`` and their
    negations into these shapes. Used to route a *goal* onto the numeric
    path instead of the boolean/object state-goal path."""
    g = f.args[0] if f.is_not() else f
    if g.is_lt() or g.is_le():
        return True
    return g.is_equals() and (_is_numeric_fnode(g.args[0]) or _is_numeric_fnode(g.args[1]))


def _signature(parameters):
    """(ASP variable, ASPType) for each action/fluent parameter."""
    return [(asp_name(p.name).upper(), ASPType(p.type)) for p in parameters]


def _head_term(name, signature):
    """The ``("name", P1, ...)`` tuple term an action-like head is built from.

    A zero-parameter head is ``("name")``, which clingo reads as the plain
    string (a one-element parenthesis is not a tuple); the planner's plan
    extraction relies on that.
    """
    inner = f'"{asp_name(name)}"'
    if signature:
        inner += ',' + ','.join(variable for variable, _ in signature)
    return f'({inner})'


def action_declaration_atom(name, parameters) -> str:
    """The atom that declares an action, with each parameter as the ASP variable
    :class:`ASPAction` binds it to.

    Doubly wrapped -- ``action(action(("name", P1, ...)))`` -- because plasp tags
    a term with its declaring predicate, so the *term* is ``action((...))`` and
    the *fact* declaring it wraps that again. Useful as an :class:`ASPAction`
    ``signature_guard``, which is how one action's parameter bindings get
    restricted to another's.
    """
    return f'action(action({_head_term(name, _signature(parameters))}))'


def _equality_term(arg):
    """``(term, has(...) bindings)`` for one side of an equality.

    A ground object, an action parameter, or a quantified variable -- the same
    three things a fluent argument can be, rendered the same way (see
    :func:`_arg_term`), since an equality is a test between such terms.
    """
    try:
        term, asp_type, kind = _arg_term(arg)
    except TypeError:
        raise TypeError(f"Unsupported equality term: {arg} of type {type(arg)}") from None
    return term, ([f'has({term}, {str(asp_type)})'] if kind == 'quantified' else [])


def _equality_value_term(arg):
    return _equality_term(arg)[0]

# Prefix for the ASP variable a *quantified* variable renders as, keeping it
# clear of an action parameter that happens to share its name.
QUANTIFIED_PREFIX = 'Q_'


def _arg_term(arg):
    # Returns (term_str, asp_type, kind). Objects become `constant("name")`
    # ground terms. Parameters become uppercase ASP variables, already bound by
    # the action's signature rule. A quantified variable becomes one too, but
    # nothing binds it yet, so whoever emits the rule has to add the
    # `has(_, type(...))` atom that ranges it over the universe -- which is how
    # a `forall` gets expanded, by the grounder rather than by a compiler.
    if arg.is_object_exp():
        return (f'constant("{asp_name(arg.object().name)}")', ASPType(arg.type), 'object')
    if arg.is_parameter_exp():
        return (asp_name(arg.parameter().name).upper(), ASPType(arg.type), 'parameter')
    if arg.is_variable_exp():
        variable = arg.variable()
        return (QUANTIFIED_PREFIX + asp_name(variable.name).upper(),
                ASPType(variable.type), 'quantified')
    raise TypeError(f"Unsupported fluent argument: {arg} of type {type(arg)}")

def _in_clingo_range(value, context):
    """`value` as a Python int, or a raise if clingo could not hold it."""
    if isinstance(value, Fraction):
        if value.denominator != 1:
            raise NotImplementedError(
                f"Non-integral numeric constant is not supported by the ASP "
                f"encoding: {value} in {context}")
        value = int(value)
    if not -CLINGO_INT_MAX <= value <= CLINGO_INT_MAX:
        raise NotImplementedError(
            f"The numeric constant {value} in {context} does not fit in a clingo "
            f"term: terms are signed 32-bit and wrap around silently, so the "
            f"encoding would return a wrong plan rather than fail. State the task "
            f"in units that keep its numbers below {CLINGO_INT_MAX}.")
    return int(value)


def _int_value(f):
    """The FNode's constant as a Python int; PDDL functions are real-typed in
    UP, so integral reals are accepted (clingo terms are integers).

    Effect deltas and assignments change a fluent's *value*, so unlike the
    constants inside a comparison they cannot be normalized away locally -- a
    fractional one needs the whole task rescaled first (see
    :mod:`aspplanners.plasp.rescale`), and reaching here without that is an
    error rather than something to round.
    """
    if f.is_int_constant():
        return _in_clingo_range(f.constant_value(), f)
    if f.is_real_constant():
        frac = f.constant_value()
        if frac.denominator != 1:
            raise NotImplementedError(
                f"Non-integral numeric constant is not supported by the ASP encoding: {f}")
        return _in_clingo_range(int(frac), f)
    raise NotImplementedError(f"Expected a numeric constant, got: {f}")


def _constant_value(f):
    """`f`'s value as an exact Fraction, or None when it is not a constant."""
    if f.is_int_constant() or f.is_real_constant():
        return Fraction(f.constant_value())
    return None


# A linear form is `(terms, constant)`: `terms` maps a fluent's rendered ASP term
# to its `(ASPExpr, Coeff coefficient)` pair -- keyed by the rendering so the
# same fluent occurring twice accumulates rather than duplicating -- and
# `constant` is the additive rest. Coefficients are exact
# `aspplanners.plasp.numeric_terms.Coeff`s, which is a rational on all but the
# tasks that read a static fluent's value as a coefficient; they stay exact
# until ASPNumComparison multiplies the whole comparison out to integers.
#
# A form is stated in the task's *own* units, not in the units its values are
# stored in: a fluent whose values carry a factor (see
# `rescale.numeric_value_scales`) has that factor divided out of its coefficient
# here, and multiplied back in by whoever states a *value* -- an effect's delta
# or assignment. A comparison needs no such move, since multiplying both its
# sides through by the common denominator clears the factor exactly.

def _lin_add(left, right):
    terms = dict(left[0])
    for key, (expr, coefficient) in right[0].items():
        if key in terms:
            terms[key] = (terms[key][0], terms[key][1] + coefficient)
        else:
            terms[key] = (expr, coefficient)
    return terms, left[1] + right[1]


def _lin_scale(form, factor):
    """`form` times `factor`, which may be a Fraction/int or a :class:`Coeff`."""
    factor = factor if isinstance(factor, Coeff) else Coeff.rational(factor)
    return ({key: (expr, coefficient * factor)
             for key, (expr, coefficient) in form[0].items()},
            form[1] * factor)


def _fold_statics(form, context):
    """`form` restated with every fluent term folded into its constant, or None.

    A product is linear as soon as one of its factors reads nothing but *static*
    fluents, because a static fluent's value is whatever the initial state says
    and the grounder can look it up once per parameter binding -- zenotravel's
    ``(* (distance ?c1 ?c2) (slow-burn ?a))`` is a number by the time the rule
    grounds. Returns None when the form reads a fluent some action writes, which
    is the genuinely non-linear case.

    The lookup binds the fluent's *stored* value and the coefficient it replaces
    had already been divided by that fluent's factor, so the two meet in the
    task's own units and the fold introduces no factor of its own. Whether the
    denominator that leaves can be stated exactly is not decided here: a
    comparison multiplies it through and never has to, and an effect that does
    is refused by :meth:`Coeff.render` with the effect named.
    """
    constant = form[1]
    for expr, coefficient in form[0].values():
        if coefficient.is_zero():
            continue
        fluent = expr.up_expr.fluent()
        if not context.is_static(fluent.name):
            return None
        constant = constant + coefficient * context.fold(
            str(expr), str(expr), fluent.name, expr.bindings)
    return {}, constant


def _linear_form(f, context=NO_FOLDING):
    """Normalize a numeric expression to the linear form ``(terms, constant)``.

    Covers everything PDDL's simple-numeric fragment can state that an integer
    encoding represents exactly: constants, numeric fluents, sums, differences,
    and products/quotients where at most one factor is non-constant -- so
    ``(+ (x ?b) (y ?b))``, ``(- (y ?b) (x ?b))`` and ``(* 2 (x ?f))`` are all
    linear forms rather than the special cases they used to be.

    A product of two fluents is not linear *as an expression*, but it is linear
    in the state whenever one side reads only fluents no action writes: their
    values come out of the initial state, so the grounder settles that side into
    a number and what is left is an ordinary coefficient. `context` is what
    names those fluents; with the default no-folding context such a product is
    refused, exactly as it was before folding existed. What is left out is
    genuinely non-linear: a product of two fluents the task *writes*, or
    division by any fluent.
    """
    constant = _constant_value(f)
    if constant is not None:
        return {}, Coeff.rational(constant)
    if f.is_fluent_exp():
        expr = ASPExpr(f, None)
        # A fluent's stored values may carry a factor its own effects needed;
        # the coefficient is in the task's units, so that factor divides out.
        return ({str(expr): (expr, Coeff.rational(
            Fraction(1) / context.scale(f.fluent().name)))}, Coeff.ZERO)
    if f.is_plus():
        form = ({}, Coeff.ZERO)
        for arg in f.args:
            form = _lin_add(form, _linear_form(arg, context))
        return form
    if f.is_minus():
        return _lin_add(_linear_form(f.args[0], context),
                        _lin_scale(_linear_form(f.args[1], context), Fraction(-1)))
    if f.is_times():
        form = ({}, Coeff.ONE)
        for arg in f.args:
            factor = _linear_form(arg, context)
            if not form[0]:
                form = _lin_scale(factor, form[1])
            elif not factor[0]:
                form = _lin_scale(form, factor[1])
            else:
                # Both sides read state. If either of them reads only *static*
                # state the grounder resolves it into a number and the product
                # is linear after all; otherwise the task is genuinely
                # non-linear and no integer encoding states it. When both sides
                # are static the whole product is a number, so both are folded:
                # leaving one as a term would state it as a coefficient over a
                # value the encoding then has to divide back down, which is a
                # denominator it need never have carried.
                folded_factor = _fold_statics(factor, context)
                folded_form = _fold_statics(form, context)
                if folded_factor is not None and folded_form is not None:
                    form = _lin_scale(folded_form, folded_factor[1])
                elif folded_factor is not None:
                    form = _lin_scale(form, folded_factor[1])
                elif folded_form is not None:
                    form = _lin_scale(factor, folded_form[1])
                else:
                    raise NotImplementedError(
                        f"A product of two numeric fluents is not linear and is not "
                        f"supported by the ASP encoding: {f}")
        return form
    if f.is_div():
        divisor = _linear_form(f.args[1], context)
        if divisor[0]:
            raise NotImplementedError(
                f"Dividing by a numeric fluent is not linear and is not supported "
                f"by the ASP encoding: {f}")
        if not divisor[1].is_rational:
            raise NotImplementedError(
                f"Dividing by a static numeric fluent is not supported by the ASP "
                f"encoding: clingo's division truncates, so the quotient would be "
                f"rounded rather than exact: {f}")
        if divisor[1].value == 0:
            raise ZeroDivisionError(f"Division by zero in the numeric expression {f}.")
        return _lin_scale(_linear_form(f.args[0], context), Fraction(1) / divisor[1].value)
    raise NotImplementedError(f"Unsupported numeric term: {f} of type {f.node_type}")


def _lcm(a, b):
    return a * b // gcd(a, b)


def _common_denominator(*forms):
    """The least multiplier making every coefficient and constant of `forms` whole."""
    denominator = 1
    for terms, constant in forms:
        for _expr, coefficient in terms.values():
            denominator = _lcm(denominator, coefficient.denominator)
        denominator = _lcm(denominator, constant.denominator)
    return denominator


class ASPBooleanType(ASPTerm):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"boolean({str(self.value).lower()})"


class ASPType(ASPTerm):
    def __init__(self, t):
        self.up_type = t

    def __str__(self):
        return f"type(\"{asp_name(self.up_type.name)}\")"


class ASPConstant(ASPTerm):
    def __init__(self, c):
        self.up_constant = c

    def __str__(self):
        return f"constant(\"{asp_name(self.up_constant.name)}\")"


class ASPHasConstant(ASPTerm):
    def __init__(self, c):
        self.up_constant  = c
        self.asp_type     = ASPType(c.type)
        self.asp_constant = ASPConstant(c)
        # Walk the type-hierarchy ancestor chain so an object typed as a
        # leaf (e.g. depot1) also satisfies has(_, type("place")) when the
        # encoding's action / variable rules ask for the supertype. Without
        # this, supertype-typed parameters never ground and entire actions
        # disappear from the program.
        self.asp_ancestor_types = []
        t = c.type
        while t is not None and t.is_user_type():
            self.asp_ancestor_types.append(ASPType(t))
            t = t.father

    def __str__(self):
        return '\n'.join(
            f"has({str(self.asp_constant)}, {str(at)})."
            for at in self.asp_ancestor_types
        )


class ASPFluent(ASPTerm):
    def __init__(self, f):
        self.up_fluent = f
        name = asp_name(f.name)
        arity_types = [(asp_name(a.name).upper(), ASPType(a.type)) for a in f.signature]
        self._head = f"\"{name}\"," + ','.join(a[0] for a in arity_types) if len(arity_types) > 0 else f"\"{name}\""
        self._head = f"variable(({self._head}))"
        self._body = ', '.join(f'has({a}, {str(t)})' for a, t in arity_types)

    def __str__(self):
        return f"variable({self._head})." if len(self._body) == 0 else f"variable({self._head}) :- {self._body}."


class ASPNumFluent(ASPFluent):
    """Declares an int-valued fluent as numVariable(...) so the encoding's
    numval/3 projection (and only it) picks its holds/3 atoms up as numbers."""
    def __str__(self):
        head = f"numVariable({self._head})"
        return f"{head}." if len(self._body) == 0 else f"{head} :- {self._body}."


class ASPExpr(ASPTerm):
    def __init__(self, f, value):
        self.up_expr = f
        terms = [_arg_term(a) for a in f.args]
        name = asp_name(f._content.payload.name)
        head_args = ','.join(t for t, _, _ in terms)
        self._head = f"\"{name}\",{head_args}" if head_args else f"\"{name}\""
        self.value = value
        # `has(_, type(...))` atoms for the quantified variables in this
        # expression, which the rule emitting it has to carry in its body:
        # ranging them over the universe there is what expands the `forall`.
        self.bindings = [f'has({term}, {str(asp_type)})'
                         for term, asp_type, kind in terms if kind == 'quantified']

    def __str__(self):
        return f"variable(({self._head}))"


class ASPEquality(ASPTerm):
    """Lifted-plasp parameter equality between two non-fluent terms.

    The encoder hoists this into the action's signature rule body so the
    action only instantiates for parameter bindings that satisfy the
    constraint. ``value == 'true'`` means ``=`` and ``'false'`` means ``!=``.
    """
    def __init__(self, f, value):
        self.up_expr = f
        self.value = value
        self.lhs, lhs_bindings = _equality_term(f.args[0])
        self.rhs, rhs_bindings = _equality_term(f.args[1])
        # Neither side is a fluent, so the only thing to range is a quantified
        # variable -- a test is not what binds one (`Q != X` is unsafe on its
        # own), so its has(_, type(...)) atom travels with the test.
        self.bindings = lhs_bindings + rhs_bindings

    def __str__(self):
        op = '=' if self.value == 'true' else '!='
        return f'{self.lhs} {op} {self.rhs}'


class ASPLinearExpr(ASPTerm):
    """One side of a numeric comparison: ``k1*V1 + k2*V2 + ... + C``.

    Rendered as the term ``expr(Terms, C)``, which the encoding's ``numValue/3``
    aggregate evaluates. Two shapes are kept from the single-fluent encoding this
    generalises, so the facts a task used to produce are unchanged: a
    constant-only side is ``expr(none, C)`` and a lone fluent with coefficient 1
    is ``expr(V, C)``. Anything richer is ``expr(sum((V1,K1),(V2,K2),...), C)``.

    The terms themselves travel separately, in :meth:`declaration_heads`:
    ``numConst/2`` carries the additive constant and doubles as the expression's
    declaration (so a constant-only side still gets a ``numValue``), and one
    ``numTerm/3`` states each ``K*V``. Splitting them is what lets a side hold
    any number of fluents without the fact's arity depending on how many.
    """

    def __init__(self, terms, constant, where='a numeric comparison', context=NO_FOLDING):
        # A zero coefficient contributes nothing; the rest are sorted by their
        # rendering so the emitted fact text is stable from run to run. A
        # coefficient that reads a static fluent is a clingo *term* rather than
        # a number and cannot be range-checked here, because its value is not
        # known until the grounder evaluates it.
        self.context = context
        self.coefficients = [constant] + [coefficient for _expr, coefficient in terms
                                          if not coefficient.is_zero()]
        self.terms = sorted(
            ((expr, _render_coefficient(coefficient, expr, context))
             for expr, coefficient in terms if not coefficient.is_zero()),
            key=lambda term: str(term[0]))
        self.constant = _render_coefficient(constant, where, context)

    @property
    def bindings(self):
        """The atoms a rule stating this side has to carry in its body.

        The ``has(_, type(...))`` atoms of its quantified variables -- ranging
        them over the universe there is what expands a ``forall`` -- and the
        ``initialState`` lookups of any static fluent its coefficients read,
        which is what settles a folded product into a number (see
        :class:`~aspplanners.plasp.numeric_terms.NumericContext`).
        """
        return ([binding for expr, _coefficient in self.terms for binding in expr.bindings]
                + list(self.context.body(*self.coefficients)))

    def _terms_term(self):
        if not self.terms:
            return 'none'
        if len(self.terms) == 1 and self.terms[0][1] == '1':
            return str(self.terms[0][0])
        return 'sum(' + ','.join(f'({str(expr)},{coefficient})'
                                 for expr, coefficient in self.terms) + ')'

    def __str__(self):
        return f'expr({self._terms_term()}, {self.constant})'

    def declaration_heads(self):
        """The ``numConst``/``numTerm`` atoms declaring this side, bodies aside.

        Owners attach their own rule body, so on a lifted task these instantiate
        for exactly the parameter bindings their owner does. Two owners sharing
        an expression derive the same atoms, which is the point: the aggregate
        is then evaluated once for both.
        """
        return ([f"numConst({str(self)}, {self.constant})"]
                + [f"numTerm({str(self)}, {coefficient}, {str(expr)})"
                   for expr, coefficient in self.terms])

    def declaration_rules(self, body=()):
        """:meth:`declaration_heads`, each stated under `body`."""
        tail = f" :- {', '.join(body)}." if body else "."
        return [f"{head}{tail}" for head in self.declaration_heads()]


class ASPNumComparison(ASPTerm):
    """Arithmetic comparison ``lhs OP rhs`` between two linear expressions.

    Rendered as ``op, expr(...), expr(...)`` (wrapped by the caller into a
    ``numPrecondition``/``numGoal``/``numOverall``/``orDisjunctNum`` fact); the
    encoding evaluates each side with its ``numValue/3`` aggregate and compares
    the two. A constant-only side uses the pseudo-variable ``none``, whose
    aggregate is empty, so the constant carries the value.

    Both sides are multiplied through by the least common denominator of their
    coefficients and constants. That is exact, and it preserves the comparison
    because the factor is positive -- which is what lets a task state
    ``(* 1/2 (x ?f))`` or compare against ``3/2`` even though clingo has only
    integers, and what makes a *comparison* the one place a fluent's storage
    factor costs nothing (see :mod:`aspplanners.plasp.rescale`). Note this
    handles fractions within one comparison only: a fractional effect delta
    changes the fluent's value itself, and is cleared by scaling that fluent.
    """

    # Each comparison's complement, so a negated one is encoded as the opposite
    # test rather than needing a `not` the encoding has no place for.
    _NEGATION = {'lt': 'le', 'le': 'lt', 'eq': 'neq', 'neq': 'eq'}

    def __init__(self, f, op, context=NO_FOLDING):
        self.up_expr = f
        self.op = op
        lhs, rhs = _linear_form(f.args[0], context), _linear_form(f.args[1], context)
        whole = _common_denominator(lhs, rhs)
        self.lhs = _as_linear_expr(_lin_scale(lhs, whole), context=context)
        self.rhs = _as_linear_expr(_lin_scale(rhs, whole), context=context)

    @property
    def bindings(self):
        """``has(_, type(...))`` atoms for quantified variables on either side."""
        return self.lhs.bindings + self.rhs.bindings

    def declaration_heads(self):
        """The ``numConst``/``numTerm`` atoms declaring both sides."""
        return self.lhs.declaration_heads() + self.rhs.declaration_heads()

    def declaration_rules(self, body=()):
        """:meth:`declaration_heads`, each stated under `body`."""
        return self.lhs.declaration_rules(body) + self.rhs.declaration_rules(body)

    def negated(self):
        # not(a < b) == b <= a ; not(a <= b) == b < a ; not(a = b) == b != a
        neg = ASPNumComparison.__new__(ASPNumComparison)
        neg.up_expr = self.up_expr
        neg.op = self._NEGATION[self.op]
        neg.lhs, neg.rhs = self.rhs, self.lhs
        return neg

    def __str__(self):
        return f'{self.op}, {str(self.lhs)}, {str(self.rhs)}'


def _render_coefficient(coefficient, where, context):
    """One coefficient or additive constant of a linear form, as a clingo term.

    A plain rational is the integer it has to be by this point -- a comparison
    has been multiplied through and an effect multiplied by its target's factor
    -- and is range-checked like any other number the encoding emits. One that
    reads a static fluent is arithmetic the grounder does instead, so what is
    checked here is that the division taking the looked-up value back into the
    task's own units is exact; clingo's ``/`` truncates, and a rounded
    coefficient would return a wrong plan rather than fail.
    """
    if coefficient.is_rational:
        return str(_in_clingo_range(coefficient.value, where))
    try:
        return coefficient.render(1, context.divisors_for([coefficient]))
    except InexactCoefficient as inexact:
        raise NotImplementedError(
            f"{where} reads a static numeric fluent it cannot scale exactly "
            f"({inexact}); the ASP encoding cannot state it. State the task in units "
            f"that make the value whole.") from None


def _as_linear_expr(form, where='a numeric comparison', context=NO_FOLDING):
    terms, constant = form
    return ASPLinearExpr([(expr, coefficient) for expr, coefficient in terms.values()],
                         constant, where, context)


def _reads_fluents(form):
    """Does this linear form read any fluent (with a surviving coefficient)?"""
    return any(not coefficient.is_zero() for _expr, coefficient in form[0].values())


def _effect_form(form, term, target_scale, context):
    """An effect's linear form, in the units its target's values are stored in.

    A comparison clears a fractional coefficient by multiplying the whole
    comparison through, which preserves it exactly (see ASPNumComparison). An
    effect has no such move -- multiplying it through would change the fluent's
    value -- so the factor that clears it is the *target's own* storage factor,
    chosen for exactly this by
    :func:`~aspplanners.plasp.rescale.numeric_value_scales`. Multiplying by it
    here is what turns ``(increase (x ?b) (* 1.5 (v ?b)))`` into the whole
    coefficient 3 on a doubly-stored ``x``.

    A coefficient still fractional after that is one no scaling of the task
    reaches -- ``(increase (v) (* 0.9 (v)))`` demands a finer grid at every step
    -- and is refused here, ahead of ASPLinearExpr's own range check, for the
    sharper message.
    """
    scaled = _lin_scale(form, target_scale)
    for expr, coefficient in scaled[0].values():
        if coefficient.is_zero() or not coefficient.is_rational:
            continue
        if coefficient.value.denominator != 1:
            raise NotImplementedError(
                f"The numeric effect on {term} reads {expr} with the fractional "
                f"coefficient {coefficient.value}, which no scaling of the task can "
                f"make whole -- {expr}'s own values would have to be scaled with it, "
                f"and the effect asks for a finer grid than that leaves. The ASP "
                f"encoding cannot state it; state the task in units that make the "
                f"coefficient integral.")
    return scaled


def _fold_fractional_statics(form, context):
    """Resolve a form's *static* terms whose coefficient is not whole.

    Petrobras' ``(decrease (current_fuel ?sh) (/ (distance ?from ?to) 5))`` is
    the shape: ``distance`` is a table rather than state, so rather than store
    ``current_fuel`` five times finer just to make the 1/5 a whole coefficient,
    the grounder reads the distance out of the initial state and works the
    delta out itself, once per parameter binding. The effect is then a constant
    ``numEffect`` -- no reified ``numValue`` per reachable value -- which is the
    difference between grounding that domain and not.

    The scale of the effect's target is what makes the arithmetic exact: it was
    chosen so that ``S_t * coefficient / S_g`` is a whole number
    (:func:`~aspplanners.plasp.rescale.numeric_value_scales`), so the folded
    monomial multiplies the looked-up value by an integer and divides by
    nothing. Confined to the terms that are *not* whole, so a task whose static
    reads all have integer coefficients emits exactly the facts it did before.
    """
    terms, constant = {}, form[1]
    for key, (expr, coefficient) in form[0].items():
        fractional = (coefficient.is_rational
                      and coefficient.value.denominator != 1)
        if not fractional or not context.is_static(expr.up_expr.fluent().name):
            terms[key] = (expr, coefficient)
            continue
        name = expr.up_expr.fluent().name
        constant = constant + coefficient * context.fold(
            str(expr), str(expr), name, expr.bindings)
    return terms, constant


def _effect_expr(form, term, target_scale, context):
    """A numeric effect's delta or assigned value as a linear expression."""
    where = f'the numeric effect on {term}'
    return _as_linear_expr(_effect_form(form, term, target_scale, context),
                           where=where, context=context)


def _effect_constant(form, term, target_scale, context):
    """A numeric effect's delta or assigned value when it reads no state."""
    where = f'the numeric effect on {term}'
    return _render_coefficient(_effect_form(form, term, target_scale, context)[1],
                               where, context)


class ASPGroundedFluent(ASPTerm):
    def __init__(self, f):
        self.up_fluent = f
        self._arity = list(map(lambda e: ASPConstant(e._content.payload), f.args))
        self._head = f"\"{asp_name(f._content.payload.name)}\""

    def __str__(self):
        _ret_str = f"{self._head}," + ','.join(str(a) for a in self._arity) if len(self._arity) > 0 else f"{self._head}"
        return f'variable(({_ret_str}))'


def _effect_bindings(eff, target):
    """The ``has(_, type(...))`` atoms an effect's rules carry in their body.

    A ``forall`` effect is emitted with its variables free and expanded by the
    grounder, exactly as a ``forall`` *condition* is. Ranging the ones the
    target fluent happens to mention is not enough: a variable the quantifier
    binds but the fluent does not use still has to be safe in the rule (and, on
    a conditional effect, still tells that binding's effect term from the rest),
    so the quantifier's own atoms come first and the target's are added on top.
    """
    _names, bindings = _variable_terms(eff.forall)
    return _dedup(bindings + list(target.bindings))


def _dedup(items):
    """Order-preserving deduplication, for rule bodies assembled from parts."""
    seen, unique = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _or_facts(disjunction, gid, owner=None, body_prefix=(), scope_vars=(),
              scope_bindings=(), parent=None):
    """The facts stating one disjunctive condition, nested groups and all.

    `owner` is the term the condition belongs to -- an action, or the effect
    term of a conditional effect -- and is the first argument of every fact; a
    *goal* has no owner and gets the goalOr* vocabulary instead. `gid` numbers
    the group within its owner, `body_prefix` are the atoms every one of its
    rules needs (the owning action's declaration atom).

    Two things make a group more than a flat list of disjuncts:

      * a disjunction under a ``forall`` carries the quantified variables in its
        own group term (`ASPDisjunction.group_vars`), so the grounder makes one
        group per binding and the encoding's constraint demands all of them;
      * a disjunction *nested* inside a disjunct becomes a group of its own,
        declared without an ``orGroup`` fact -- nothing forces it on its own --
        and pulled in by the ``orDisjunctSub`` atom of the disjunct that owns
        it, whose failure is that disjunct's failure. `parent` is that owning
        disjunct, and is None for a top-level group.

    A subgroup's term repeats every variable in scope at its point of use, so
    two bindings of an enclosing quantifier never share (and thereby merge) a
    group.
    """
    prefix = 'or' if owner is not None else 'goalOr'
    owner_args = [owner] if owner is not None else []

    def fact(suffix, args, body):
        head = f"{prefix}{suffix}({', '.join(owner_args + args)})"
        return f"{head}." if not body else f"{head} :- {', '.join(body)}."

    variables = list(scope_vars) + disjunction.group_vars
    group = f"or({gid})" if not variables else f"or({gid},{','.join(variables)})"
    body = _dedup(list(body_prefix) + list(scope_bindings) + disjunction.group_bindings)

    if parent is None:
        facts = [fact('Group', [group], body)]
    else:
        parent_group, parent_id, parent_body = parent
        facts = [fact('DisjunctSub', [parent_group, parent_id, group],
                      _dedup(list(parent_body) + disjunction.group_bindings))]

    for index, disjunct in enumerate(disjunction.disjuncts):
        # An equality test goes in the body: a disjunct whose test fails is not
        # declared at all, which is exactly it failing.
        # The disjunct is declared under its own bindings only (see
        # ASPDisjunct.own_bindings); each literal adds the bindings of the
        # variables it ranges over, so a universally quantified literal over an
        # empty type contributes no instance and the disjunct holds vacuously.
        disjunct_body = _dedup(body + disjunct.own_bindings + [str(test) for test in disjunct.tests])
        facts.append(fact('DisjunctId', [group, disjunct.id], disjunct_body))
        for atom in disjunct.atoms:
            atom_body = _dedup(disjunct_body + atom.bindings)
            if isinstance(atom, ASPNumComparison):
                facts.append(fact('DisjunctNum', [group, disjunct.id, str(atom)], atom_body))
                # The sides are declared under the disjunct's own body, so a
                # disjunct the grounder never declares never declares them either.
                facts += atom.declaration_rules(atom_body)
            elif isinstance(atom, ASPExpr):
                facts.append(fact('Disjunct', [group, disjunct.id, str(atom),
                                               f'value({str(atom)}, {atom.value})'],
                                  atom_body))
            else:
                raise NotImplementedError(
                    f"Unsupported literal {atom} in a disjunctive condition of "
                    f"{owner if owner is not None else 'a goal'}.")
        for sub_index, subgroup in enumerate(disjunct.subgroups):
            facts += _or_facts(subgroup, f"({gid},{index},{sub_index})", owner=owner,
                               scope_vars=variables + disjunct.own_vars,
                               scope_bindings=disjunct_body,
                               parent=(group, disjunct.id, disjunct_body))
    return facts


class ASPAction(ASPTerm):
    def __init__(self, a, static_fluents=None, signature_guard=None, context=NO_FOLDING):
        # static_fluents: set of fluent names whose value is fixed by the
        # initial state (no action has them in its effects). Positive
        # preconditions on these are folded into the action signature body
        # so the grounder pre-filters parameter bindings by the static
        # relation, instead of cross-producing all type-typed bindings and
        # pruning at solve time. Without this, actions whose only
        # parameter-narrowing comes from a static precondition (e.g. TPP
        # `next/2`, Mystery `attacks/2`/`eats/2`) ground to |obj|^arity and
        # blow past clasp's 28-bit ID space.
        # signature_guard: an atom that already restricts every parameter, used
        # instead of the per-parameter has(_, type(...)) atoms. An end snap
        # passes its start snap's declaration atom, which is both tighter (the
        # start's folded static preconditions carry over) and sound -- an end
        # snap can only ever fire for a binding whose start exists.
        # context: the task's numeric facts -- the storage factor of each
        # fluent's values and which of them a product may resolve out of the
        # initial state (see aspplanners.plasp.numeric_terms.NumericContext).
        static_fluents = static_fluents or set()
        self.context = context
        self.up_action = a
        # Fluents this action reads as *false*, whose `holds(V, value(V,false))`
        # chain therefore has to exist from step 0 on (see the encoder).
        self.negated_fluents = set()
        self.signature = _signature(a.parameters)
        self._head = f"action({_head_term(a.name, self.signature)})"
        self._sig_body = signature_guard if signature_guard is not None else \
            ', '.join(f'has({p[0]}, {str(p[1])})' for p in self.signature)


        # iterate over the preconditions.
        self._preconditions = []
        equality_atoms = []
        static_atoms = []
        group_index = 0
        for precondition in a.preconditions:
            # And-within-And nesting yields nested lists — flatten them.
            variablelist = flatten_atoms(parseexpr(precondition, context=self.context))
            for variable in variablelist:
                if isinstance(variable, ASPDisjunction):
                    self._preconditions += self._disjunction_facts(variable, group_index)
                    self.negated_fluents |= variable.negated_fluent_names
                    group_index += 1
                    continue
                if isinstance(variable, ASPEquality):
                    equality_atoms.append(str(variable))
                    continue
                if isinstance(variable, ASPNumComparison):
                    body = _dedup([f'action({self._head})'] + variable.bindings)
                    head = f'numPrecondition({self._head}, {str(variable)})'
                    self._preconditions.append(f"{head} :- {', '.join(body)}.")
                    self._preconditions += variable.declaration_rules(body)
                    continue
                fluent_name = variable.up_expr._content.payload.name if isinstance(variable, ASPExpr) and variable.up_expr.is_fluent_exp() else None
                if fluent_name is not None and variable.value == 'false':
                    self.negated_fluents.add(fluent_name)
                # Only positive (true-valued) static preconditions are safe
                # to fold via initialState/2; closed-world handling for
                # `not initialState(...)` is non-trivial so we leave
                # negative cases on the runtime path.
                bindings = getattr(variable, 'bindings', [])
                if (fluent_name in static_fluents) and variable.value == 'true' and not bindings:
                    static_atoms.append(f'initialState({str(variable)}, value({str(variable)}, true))')
                    continue
                head = f'precondition({self._head}, {str(variable)}, value({str(variable)}, {variable.value}))'
                # action(...) ground-restricts every parameter via the action signature rule;
                # adding has(_, type(...)) per parameter forces the grounder to iterate
                # |objects|^arity body matches per rule with no semantic effect. A
                # *quantified* variable is the exception: nothing else binds it, and
                # ranging it over the universe here is what expands the forall.
                body = ', '.join([f'action({self._head})'] + bindings)
                self._preconditions.append(f"{head} :- {body}.")

        extra_body = equality_atoms + static_atoms
        if extra_body:
            joined = ', '.join(extra_body)
            self._sig_body = f'{self._sig_body}, {joined}' if self._sig_body else joined

        # iterate over the unconditional effects.
        self._postconditions = []
        # A fluent's increase/decrease deltas accumulate as one linear form, so
        # several effects on the same fluent are a single update. A form with no
        # fluent terms stays the numEffect/numAssign integer it always was; one
        # that reads fluents becomes a numEffectExpr/numAssignExpr, whose
        # expression the encoding evaluates against the step before.
        num_deltas = {}    # fluent term -> (summed linear-form delta, scale)
        num_bindings = {}  # fluent term -> the has(...) atoms its rules need
        # An unconditional delta cannot ride on numEffect while a conditional one
        # on the same fluent rides on the numeric layer's sum: they would be two
        # `caused` atoms carrying two different new values, which is not a state.
        # So once this action has any conditional delta, all of its deltas go
        # through the sum -- under an effect term with no condition, which fires
        # whenever the action does. An action with none is untouched.
        conditional_deltas = any(
            not e.fluent.type.is_bool_type() and e.kind in (EffectKind.INCREASE,
                                                            EffectKind.DECREASE)
            for e in a.conditional_effects)
        # Each of them gets a term of its own rather than one shared "no
        # condition" term, because the sum is keyed by (delta, term): two deltas
        # that happen to be equal -- `forall ?o . w(?o) += 2` alongside
        # `w(d) += 2` -- would otherwise be one element and count once.
        unconditional_terms = {}

        def unconditional_term(term):
            return unconditional_terms.setdefault(
                term, f'effect((uncond,"{asp_name(a.name)}",{len(unconditional_terms)}))')
        for eff in a.unconditional_effects:
            target = parseexpr(eff.fluent)
            quantified = _effect_bindings(eff, target)
            if eff.kind in (EffectKind.INCREASE, EffectKind.DECREASE):
                term = str(target)
                form = _linear_form(eff.value, self.context)
                if eff.kind == EffectKind.DECREASE:
                    form = _lin_scale(form, Fraction(-1))
                previous = num_deltas.get(term)
                num_deltas[term] = (_lin_add(previous[0], form) if previous else form,
                                    self.context.scale(eff.fluent.fluent().name))
                num_bindings[term] = _dedup(num_bindings.get(term, []) + quantified)
                continue
            if eff.kind == EffectKind.ASSIGN and not eff.fluent.type.is_bool_type():
                term = str(target)
                scale = self.context.scale(eff.fluent.fluent().name)
                form = _fold_fractional_statics(_linear_form(eff.value, self.context),
                                                self.context)
                if _reads_fluents(form):
                    expr = _effect_expr(form, term, scale, self.context)
                    body = _dedup([f'action({self._head})'] + quantified + expr.bindings)
                    head = f"numAssignExpr({self._head}, {term}, {str(expr)})"
                    self._postconditions.append(f"{head} :- {', '.join(body)}.")
                    self._postconditions += expr.declaration_rules(body)
                else:
                    value = _effect_constant(form, term, scale, self.context)
                    body = _dedup([f'action({self._head})'] + quantified
                                  + list(self.context.body(form[1])))
                    head = f"numAssign({self._head}, {term}, {value})"
                    self._postconditions.append(f"{head} :- {', '.join(body)}.")
                continue
            variable = target
            value    = str(eff.value).lower()
            head = f"postcondition({self._head}, effect(unconditional), {str(variable)}, value({str(variable)}, {value}))"
            # `forall ?x . f(?x) := v` needs ?x ranged the same way a forall
            # condition does; the caused/3 rule then fires once per binding.
            body = ', '.join(_dedup([f'action({self._head})'] + quantified))
            self._postconditions.append(f"{head} :- {body}.")
        for term, (form, scale) in num_deltas.items():
            quantified = num_bindings.get(term, [])
            form = _fold_fractional_statics(form, self.context)
            if _reads_fluents(form):
                expr = _effect_expr(form, term, scale, self.context)
                body = _dedup([f'action({self._head})'] + quantified + expr.bindings)
                head = (f"numCondEffectExpr({self._head}, {unconditional_term(term)}, "
                        f"{term}, {str(expr)})" if conditional_deltas
                        else f"numEffectExpr({self._head}, {term}, {str(expr)})")
                self._postconditions.append(f"{head} :- {', '.join(body)}.")
                self._postconditions += expr.declaration_rules(body)
                continue
            # A delta of a literal 0 is no effect at all; one that is only
            # *arithmetic* over a folded lookup is not known to be zero here.
            if form[1].is_zero():
                continue
            value = _effect_constant(form, term, scale, self.context)
            body = _dedup([f'action({self._head})'] + quantified
                          + list(self.context.body(form[1])))
            head = (f"numCondEffect({self._head}, {unconditional_term(term)}, {term}, "
                    f"{value})" if conditional_deltas
                    else f"numEffect({self._head}, {term}, {value})")
            self._postconditions.append(f"{head} :- {', '.join(body)}.")

        # Conditional effects (`when C then F := V`). The existential/sequential
        # encoding already supports them via:
        #   caused(Var,Val,T) :- occurs(A,T), postcondition(A,Eff,Var,Val),
        #                        holds(VP,VVP,T-1) : precondition(Eff,VP,VVP).
        # i.e. an effect fires when every precondition(Eff, ...) atom holds at T-1.
        # We assign each conditional effect a unique Effect term parameterised by
        # the action's parameters so distinct ground instances don't share
        # preconditions (which would mix bindings across instances and require
        # every binding's condition to hold simultaneously).
        params_tail = (',' + ','.join(p[0] for p in self.signature)) if self.signature else ''
        for idx, eff in enumerate(a.conditional_effects):
            variable = parseexpr(eff.fluent)
            # A `forall` conditional effect is one effect per binding of its
            # variables, so the bindings index the effect term. Sharing a single
            # term across them would be a different task: caused/3 fires an
            # effect only when *every* precondition of its term holds, so one
            # term for all bindings demands every binding's condition at once
            # instead of firing the effect once per binding that satisfies it.
            forall_names, forall_bindings = _variable_terms(eff.forall)
            effect_tail = params_tail + (',' + ','.join(forall_names) if forall_names else '')
            effect_term = f'effect((cond,"{asp_name(a.name)}",{idx}{effect_tail}))'
            # Every rule that names the effect term has to range the quantified
            # variables in it, whether or not this particular rule's own atoms
            # mention them -- they are what tells the bindings' effects apart.
            effect_body = _dedup([f'action({self._head})'] + forall_bindings)

            condition_atoms = flatten_atoms(parseexpr(eff.condition, context=self.context))
            # An equality in the condition gates the effect by not declaring its
            # postcondition at all, so it is folded into that one rule's body
            # (and only it: a precondition of an effect that has no
            # postcondition can never fire anything). It is collected before any
            # rule is emitted rather than patched onto the last one afterwards,
            # which would land on an orGroup fact when a disjunction came first.
            equalities = [ca for ca in condition_atoms if isinstance(ca, ASPEquality)]
            equality_body = [b for ca in equalities for b in ca.bindings] \
                + [str(ca) for ca in equalities]
            self._postconditions += self._conditional_effect_facts(
                a, eff, effect_term, effect_body, variable, equality_body)

            cond_group_index = 0
            for ca in condition_atoms:
                if isinstance(ca, ASPEquality):
                    continue
                if isinstance(ca, ASPDisjunction):
                    # The effect's own orGroup facts; the encoding's caused/3
                    # rule requires every group of the effect term to hold, the
                    # same way the action constraint does for an action's.
                    self._postconditions += _or_facts(
                        ca, str(cond_group_index), owner=effect_term,
                        body_prefix=effect_body)
                    self.negated_fluents |= ca.negated_fluent_names
                    cond_group_index += 1
                    continue
                if isinstance(ca, ASPNumComparison):
                    # An effect's conditions are read conjunctively off
                    # precondition/3, which is not numeric -- but a group of the
                    # effect term is required just as conjunctively, so a
                    # comparison is stated as the one-disjunct group holding it.
                    # No new rule anywhere: the disjunctive layer judges it with
                    # the same difference #sum it judges a disjunct's numeric
                    # literal with, and caused/3 already demands every group.
                    self._postconditions += _or_facts(
                        ASPDisjunction([ASPDisjunct('0', ca)]), str(cond_group_index),
                        owner=effect_term, body_prefix=effect_body)
                    cond_group_index += 1
                    continue
                if isinstance(ca, ASPExpr) and ca.up_expr.is_fluent_exp() and ca.value == 'false':
                    self.negated_fluents.add(ca.up_expr._content.payload.name)
                cond_head = (
                    f"precondition({effect_term}, {str(ca)}, "
                    f"value({str(ca)}, {ca.value}))"
                )
                # A quantified variable in the condition is ranged here, so the
                # effect gets one precondition atom per binding -- which is the
                # conjunction over the universe a `forall` condition asks for.
                # A variable of the effect's *own* `forall` is ranged too, but
                # its bindings are separate effect terms, so there it is one
                # condition each rather than a conjunction.
                body = ', '.join(_dedup(effect_body + list(ca.bindings)))
                self._postconditions.append(f"{cond_head} :- {body}.")

    def _conditional_effect_facts(self, a, eff, effect_term, effect_body, variable,
                                  equality_body):
        """The rules assigning one conditional effect's value, by its shape.

        A boolean effect and a numeric *assignment of a constant* state the
        variable's new value outright, so they ride on core's
        ``postcondition``/``caused`` and this is one rule. The rest cannot: a
        delta's new value is the old one plus something, and two deltas that
        both fire have to add rather than each state a state. They get the
        numeric layer's own conditional vocabulary, which collects the firing
        deltas and applies their sum once (see ``encodings/seq/numeric.lp``).
        """
        term = str(variable)
        body = _dedup(effect_body + list(variable.bindings) + equality_body)
        scale = self.context.scale(eff.fluent.fluent().name) \
            if not eff.fluent.type.is_bool_type() else 1

        if eff.fluent.type.is_bool_type() or eff.kind == EffectKind.ASSIGN:
            form = None if eff.fluent.type.is_bool_type() else \
                _fold_fractional_statics(_linear_form(eff.value, self.context), self.context)
            if form is None or not _reads_fluents(form):
                value = (str(eff.value).lower() if eff.fluent.type.is_bool_type()
                         else _effect_constant(form, term, scale, self.context))
                extra = [] if form is None else list(self.context.body(form[1]))
                head = (f"postcondition({self._head}, {effect_term}, "
                        f"{term}, value({term}, {value}))")
                return [f"{head} :- {', '.join(_dedup(body + extra))}."]
            expr = _effect_expr(form, term, scale, self.context)
            head = f"numCondAssignExpr({self._head}, {effect_term}, {term}, {str(expr)})"
            rule_body = _dedup(body + expr.bindings)
            return ([f"{head} :- {', '.join(rule_body)}."]
                    + expr.declaration_rules(rule_body))

        form = _linear_form(eff.value, self.context)
        if eff.kind == EffectKind.DECREASE:
            form = _lin_scale(form, Fraction(-1))
        form = _fold_fractional_statics(form, self.context)
        if _reads_fluents(form):
            expr = _effect_expr(form, term, scale, self.context)
            head = f"numCondEffectExpr({self._head}, {effect_term}, {term}, {str(expr)})"
            rule_body = _dedup(body + expr.bindings)
            return ([f"{head} :- {', '.join(rule_body)}."]
                    + expr.declaration_rules(rule_body))
        delta = _effect_constant(form, term, scale, self.context)
        rule_body = _dedup(body + list(self.context.body(form[1])))
        return [f"numCondEffect({self._head}, {effect_term}, {term}, {delta}) :- "
                f"{', '.join(rule_body)}."]

    def _disjunction_facts(self, disjunction, index):
        """orGroup/orDisjunct facts for one disjunctive precondition.

        The group is a plain fact of the action; each disjunct contributes one
        orDisjunct per literal, under whatever bindings that disjunct needs --
        which for an `exists` is what ranges it over the universe. A numeric
        literal goes to orDisjunctNum instead, since it is checked against
        numval rather than holds; orDisjunctId declares the disjunct itself, so
        a disjunct made only of numeric literals is still enumerated.
        """
        return _or_facts(disjunction, str(index), owner=self._head,
                         body_prefix=[f'action({self._head})'])

    def __str__(self):
        _sig = [
            f"action({self._head})." if len(self._sig_body) == 0 else f"action({self._head}) :- {self._sig_body}."
        ]
        _sig += self._preconditions
        _sig += self._postconditions
        return '\n'.join(_sig)


def _duration_arithmetic(expression, da, scales=None, counter=None, lookups=None):
    """``(numerator, denominator, lookups)`` for a deferred duration expression.

    The value is carried as an exact rational of two clingo integer terms rather
    than one divided term, because clingo's ``/`` truncates and a duration like
    ``(/ (distance ?y ?z) (speed ?x))`` has to survive the intermediate step. The
    caller divides once, at the end.

    `lookups` collects one ``initialState(...)`` body atom per fluent occurrence,
    binding it to a fresh ASP variable; that is what resolves the value per
    parameter binding at grounding time, so the task never has to be ground.

    `scales` is the per-fluent factor its initial values were multiplied by to
    become integers (see rescale.duration_fluent_scales). It goes into the
    denominator, which is what divides it back out of the duration.
    """
    scales = {} if scales is None else scales
    counter = [0] if counter is None else counter
    lookups = [] if lookups is None else lookups

    if expression.is_int_constant() or expression.is_real_constant():
        value = Fraction(expression.constant_value())
        return str(value.numerator), str(value.denominator), lookups

    if expression.is_fluent_exp():
        fluent = parseexpr(expression, None)
        variable = f"RAWDUR{counter[0]}"
        counter[0] += 1
        lookups.append(
            f"initialState({str(fluent)}, value({str(fluent)}, {variable}))")
        return variable, str(scales.get(expression.fluent().name, 1)), lookups

    operands = [_duration_arithmetic(arg, da, scales, counter, lookups)
                for arg in expression.args]
    numerators = [n for n, _d, _l in operands]
    denominators = [d for _n, d, _l in operands]

    if expression.is_times():
        return (' * '.join(f"({n})" for n in numerators),
                ' * '.join(f"({d})" for d in denominators), lookups)
    if expression.is_div():
        # a/b = (na*db) / (da*nb); with more than two operands this folds left,
        # which is the order UP itself reads the expression in.
        num, den = numerators[0], denominators[0]
        for n, d in zip(numerators[1:], denominators[1:]):
            num, den = f"({num}) * ({d})", f"({den}) * ({n})"
        return num, den, lookups
    if expression.is_plus() or expression.is_minus():
        sign = ' + ' if expression.is_plus() else ' - '
        # Over a common denominator, so the sum stays exact.
        common = ' * '.join(f"({d})" for d in denominators)
        terms = []
        for index, (n, _d) in enumerate(zip(numerators, denominators)):
            others = [f"({d})" for position, d in enumerate(denominators) if position != index]
            terms.append(f"({n})" + (' * ' + ' * '.join(others) if others else ''))
        return sign.join(terms), common, lookups

    raise NotImplementedError(
        f"Duration {expression} of durative action {da.name!r} is not arithmetic over "
        "static fluents and constants; the encoding evaluates a duration by looking its "
        "fluents up in the initial state.")


class ASPDurativeAction(ASPTerm):
    """The temporal facts tying a durative action to its two snap actions.

    A durative action is encoded as the pair of instantaneous *snap* actions
    PDDL 2.1 decomposes it into -- exactly what SMTPlan's happening encoder
    does. Their at-start / at-end conditions and effects travel as ordinary
    ``precondition``/``postcondition`` facts of the snap actions themselves
    (see :class:`ASPAction`), so all this builder emits is what the encoding's
    temporal layer needs to couple the two halves back together:

      * ``durativeAction/1``  -- the durative action term, which also switches
        the temporal layer on;
      * ``snap/3``            -- which ordinary action is its start and its end;
      * ``durationValue/2``   -- the admissible durations in scaled time units
        (an interval when the task states a duration *constraint* rather than a
        fixed duration);
      * ``overall/3`` / ``numOverall/4`` -- the over-all conditions, which
        belong to neither snap: the encoding checks them at every happening the
        action spans.

    Each fact is guarded by the snap action it belongs to, so on a lifted task
    they instantiate for exactly the parameter bindings the snap actions do.
    """

    def __init__(self, da, start_name, end_name, duration_bounds, overall_conditions,
                 duration_scales=None, context=NO_FOLDING):
        self.up_action = da
        signature = _signature(da.parameters)
        self._head  = f"durative({_head_term(da.name, signature)})"
        self._start = f"action({_head_term(start_name, signature)})"
        self._end   = f"action({_head_term(end_name, signature)})"

        # Numeric bounds go straight into the fact; a duration stated as a
        # static fluent becomes a lookup in the initial state instead, so it
        # resolves per parameter binding at grounding time and the task never
        # has to be ground first (see common.temporal.FluentDuration).
        self._duration_body = None
        if isinstance(duration_bounds, tuple):
            lower, upper = duration_bounds
            self._duration = str(lower) if lower == upper else f"{lower}..{upper}"
        else:
            numerator, denominator, lookups = _duration_arithmetic(
                duration_bounds.expression, da, duration_scales)
            # Kept as a single rational and divided once, because clingo's `/`
            # truncates: `(distance / speed) * mult / div` would floor 5/2 to 2
            # before the scaling ever ran. Dividing only at the end is exact,
            # since time_unit was built from every value the expression can take.
            term = (f"({numerator}) * {duration_bounds.multiplier} "
                    f"/ (({denominator}) * {duration_bounds.divisor})")
            # DURATION > 0 is what keeps a binding that computes no real duration
            # -- depots' `(distance ?y ?y)` of 0 -- from becoming an action: with
            # no durationValue fact the `1 {remaining(DA, D, t)} 1 :- starts(DA, t)`
            # choice has nothing to pick, so that durative action can never start.
            self._duration = "DURATION"
            self._duration_body = ', '.join(
                lookups + [f"DURATION = {term}", "DURATION > 0"])

        # Each over-all fact is `(head, extra body atoms)`: a numeric one whose
        # coefficients read a static fluent needs that lookup in its body, the
        # same way a numeric precondition does.
        self._overall = []
        for condition in overall_conditions:
            atoms = parseexpr(condition, context=context)
            atoms = [atoms] if not isinstance(atoms, list) else atoms
            while any(isinstance(a, list) for a in atoms):
                atoms = [e for a in atoms for e in (a if isinstance(a, list) else [a])]
            for atom in atoms:
                if isinstance(atom, ASPNumComparison):
                    # The comparison's sides are declared alongside it, guarded
                    # by the same start-snap atom the over-all fact itself is.
                    bindings = _dedup(atom.bindings)
                    self._overall.append((f"numOverall({self._head}, {str(atom)})", bindings))
                    self._overall += [(head, bindings) for head in atom.declaration_heads()]
                elif isinstance(atom, ASPExpr):
                    self._overall.append((
                        f"overall({self._head}, {str(atom)}, value({str(atom)}, {atom.value}))",
                        list(atom.bindings)))
                else:
                    raise NotImplementedError(
                        f"Unsupported over-all condition in durative action "
                        f"{da.name!r}: {condition}")

    def __str__(self):
        # `action(<term>)` is the declaration atom of the snap action (plasp
        # tags terms with their declaring predicate); guarding on it is what
        # instantiates these facts for exactly the bindings the snap has.
        started, ended = f"action({self._start})", f"action({self._end})"
        duration_body = started if self._duration_body is None \
            else f"{started}, {self._duration_body}"
        facts = [
            f"durativeAction({self._head}) :- {started}.",
            f"snap({self._head}, start, {self._start}) :- {started}.",
            f"snap({self._head}, end, {self._end}) :- {ended}.",
            f"durationValue({self._head}, {self._duration}) :- {duration_body}.",
        ]
        facts += [f"{atom} :- {', '.join([started] + bindings)}."
                  for atom, bindings in self._overall]
        return '\n'.join(facts)


class ASPStateVarVal(ASPTerm):
    def __init__(self, fluent, value):
        self.fluent = ASPGroundedFluent(fluent)
        # A numeric value goes through _int_value rather than str(): UP renders a
        # real constant as its Fraction, and `value(V, 3/2)` is not a rational to
        # clingo but *integer division*, which grounds to 1 -- a silently wrong
        # initial state rather than an error. Booleans and objects keep the plain
        # rendering; their str() is already the term the encoding expects.
        self.value  = (str(_int_value(value))
                       if value.is_int_constant() or value.is_real_constant()
                       else str(value).lower())

    def __str__(self):
        return f"{str(self.fluent)}, value({str(self.fluent)}, {self.value})"


class ASPInitialState(ASPStateVarVal):
    def __str__(self):
        return f"initialState({super().__str__()})."


class ASPGoalState(ASPTerm):
    """``goal(V, value(V, val))`` for one state goal.

    Built from a parsed :class:`ASPExpr` rather than a ground fluent, so a
    ``forall`` goal works the same way a ``forall`` precondition does: the
    quantified variable stays free in the head and its ``has(_, type(...))``
    binding goes in the body, leaving the expansion to the grounder.
    """

    def __init__(self, expr):
        if not isinstance(expr, ASPExpr):
            raise NotImplementedError(
                f"Unsupported state goal: {expr} of type {type(expr).__name__}.")
        self.expr = expr
        self.value = expr.value
        self.bindings = list(expr.bindings)

    @property
    def fluent_name(self):
        return self.expr.up_expr._content.payload.name

    def __str__(self):
        head = f"goal({str(self.expr)}, value({str(self.expr)}, {self.value}))"
        return f"{head}." if not self.bindings else f"{head} :- {', '.join(self.bindings)}."


class ASPGoalDisjunction(ASPTerm):
    """goalOrGroup/goalOrDisjunct facts for one disjunctive goal.

    The goal-time analogue of the orGroup/orDisjunct facts :class:`ASPAction`
    emits: same structure, checked against the state at the goal step rather
    than the one before an action.
    """

    def __init__(self, disjunction, index):
        self.disjunction = disjunction
        self._index = index

    @property
    def negated_fluent_names(self):
        return self.disjunction.negated_fluent_names

    def __str__(self):
        return '\n'.join(_or_facts(self.disjunction, str(self._index)))


class ASPNumGoal(ASPTerm):
    """A numeric goal ``lhs OP rhs`` checked against numval at the goal step.

    The goal-time analogue of ASPNumComparison/numPrecondition: it reuses that
    class's ``op, expr(V,C), expr(V,C)`` rendering (via parseexpr, which folds
    ``GE``/``GT`` and negations into ``lt``/``le``/``eq``) and wraps it as a
    standalone ``numGoal(...)`` fact. The encoding rejects any model whose
    numval at the queried timestep violates it.
    """
    def __init__(self, f, context=NO_FOLDING):
        self.cmp = f if isinstance(f, ASPNumComparison) else parseexpr(f, context=context)
        assert isinstance(self.cmp, ASPNumComparison), (
            f"ASPNumGoal expects a numeric comparison, got {type(self.cmp)} for {f}"
        )

    def __str__(self):
        # A goal is guarded by nothing, so its sides are plain facts -- unless a
        # coefficient reads a static fluent, whose value the grounder looks up.
        body = _dedup(self.cmp.bindings)
        head = f"numGoal({str(self.cmp)})"
        tail = f" :- {', '.join(body)}." if body else "."
        return '\n'.join([f"{head}{tail}"] + self.cmp.declaration_rules(body))

