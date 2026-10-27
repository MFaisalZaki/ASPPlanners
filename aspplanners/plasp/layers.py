"""Encoding layers: which ``.lp`` files make up the program for a given task.

The sequential-horizon encoding is split into one file per feature (see
``encodings/seq/``). clingo merges ``#program`` parts of the same name across the
files handed to it, so the program for a task is simply the concatenation of the
layers it needs; a layer that is left out is not an error, because layers refer
to each other's predicates only through ``#defined``.

That last property is also the hazard. Leaving out a layer a task *does* need is
silently wrong rather than loud: drop the numeric layer from a numeric task and
its ``numPrecondition`` facts sit there constraining nothing, and clingo returns
a plan that does not respect them. So selection is driven by what the encoder
actually emitted, and :meth:`EncodingFamily.check_coverage` turns "a fact nobody
consumes" into an exception at construction time instead of a mysterious
validation failure several horizons later.

Two independent signals pick the layers:

  * the **fact predicates** the encoder emitted (:func:`fact_predicates`). This
    is the authority. It is exact by construction -- a layer is needed iff some
    fact only it reads is present -- and it describes the task as encoded, after
    the compilation pipeline, TIM inference, numeric rescaling and the durative
    split have all had their say.
  * the **problem kind** (:meth:`EncodingFamily.layers_from_kind`), read off the
    *compiled* problem. This is advisory. It cannot be authoritative because it
    describes the task's declared features rather than the vocabulary the
    encoding consumes, and the two need not line up: a feature can be compiled
    away, and a feature can be absent while the encoder still emits the facts
    (a durative action's over-all comparison produces ``numOverall`` whether or
    not the task otherwise counts as numeric).

The default takes their union. Over-selecting costs a few unused rules that
ground to nothing; under-selecting costs correctness, so the two signals
disagreeing is resolved in favour of loading more.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

ENCODINGS_DIR = os.path.join(os.path.dirname(__file__), 'encodings')

# The head of a fact line: `p(...)` or a bare `p`, with the rule body (if any)
# already stripped off.
_HEAD_PREDICATE = re.compile(r'^([a-z_][A-Za-z0-9_]*)')


def fact_predicates(fact_lines: Iterable[str]) -> Set[str]:
    """The predicate names a set of encoder-emitted fact lines defines.

    Not every emitted line is a plain fact: a durative action's facts are guarded
    by their snap action's declaration atom (`durativeAction(...) :- action(...)`)
    so that a lifted task instantiates them for exactly the bindings the snap has.
    Only the head names the vocabulary, so the body is dropped.
    """
    predicates = set()
    for line in fact_lines:
        head = line.split(':-', 1)[0].strip()
        match = _HEAD_PREDICATE.match(head)
        if match:
            predicates.add(match.group(1))
    return predicates


@dataclass(frozen=True)
class Layer:
    """One ``.lp`` file of an encoding family.

    `owns` is the set of fact predicates this layer is the sole consumer of --
    the facts that mean "this layer is needed". Predicates read by several layers
    (``holds``, ``occurs``) belong to none of them; they are the encoding's
    internal interface, not part of the task's fact vocabulary.

    `requires` are the layers whose *rules* this one's rules read unconditionally.
    A dependency that only some of a layer's rules have -- the numeric
    comparisons inside a disjunct, say -- is not stated here but in
    :data:`PREDICATE_REQUIRES`, because it holds only when the corresponding fact
    is actually present.

    `describes_instance` is the subset of `owns` that states what the *task* is
    rather than driving a rule: the problem instance's types, its objects, its
    fluents, and the boolean value domain. The encoder emits them because the
    program describes the instance -- they are what makes a dumped program a
    standalone, readable clingo input -- and no rule reads them, which is
    correct rather than dead. They are called out here so that
    :func:`tests.test_layers` can hold every *other* owned predicate to the
    rule that consumes it; without the distinction, "claimed but read by
    nothing" would be indistinguishable from "claimed on the wrong layer",
    which is silently wrong (see the module docstring).

    `kind_features` are the ``ProblemKind.has_*`` predicates that imply this
    layer. They are consulted defensively (a name absent from the installed
    unified-planning is skipped) so that a UP upgrade cannot break selection.
    """
    name: str
    filename: str
    owns: FrozenSet[str] = frozenset()
    describes_instance: FrozenSet[str] = frozenset()
    requires: Tuple[str, ...] = ()
    kind_features: Tuple[str, ...] = ()
    summary: str = ''


# Facts whose presence pulls in a layer beyond the one that owns them. These are
# the *bridge* rules: rules that live in the more specific layer and read a more
# general one, each guarded in the .lp by its own `#defined` so the layer still
# loads without its bridge partner. `numOverall` is a temporal fact evaluated
# against the numeric layer's numValue/3; the two `*Num` predicates are the
# numeric literals of a disjunct.
PREDICATE_REQUIRES: Dict[str, Tuple[str, ...]] = {
    'numOverall': ('numeric',),
    'orDisjunctNum': ('numeric',),
    'goalOrDisjunctNum': ('numeric',),
    # A conditional numeric effect reads its condition through the same
    # orGroup/orGroupHolds pair core's caused/3 does, and a *numeric* condition
    # on one is stated as a group of its own, so the numeric layer's conditional
    # rules need the disjunctive layer whenever they are used at all.
    'numCondEffect': ('disjunctive',),
    'numCondEffectExpr': ('disjunctive',),
    'numCondAssignExpr': ('disjunctive',),
}


SEQ_LAYERS: Tuple[Layer, ...] = (
    Layer(
        name='core',
        filename='core.lp',
        summary='multi-valued STRIPS over the multi-shot horizon',
        owns=frozenset({
            'type', 'constant', 'has', 'variable', 'boolean', 'action',
            'precondition', 'postcondition', 'initialState', 'goal',
        }),
        # The instance's own vocabulary: its types, its objects (`constant`/`has`),
        # its fluents, and the two boolean values. The encoder states them because
        # the program describes the task; the rules work off `initialState`,
        # `action`, `precondition`, `postcondition` and `goal` instead. Note that
        # `variable` appears in core.lp only as the *term* `variable(V)` inside
        # `holds`, never as an atom -- the fact is not read either.
        describes_instance=frozenset({
            'type', 'constant', 'has', 'variable', 'boolean',
        }),
    ),
    Layer(
        name='numeric',
        filename='numeric.lp',
        summary='linear numeric fluents, comparisons and effects',
        requires=('core',),
        owns=frozenset({
            'numVariable', 'numConst', 'numTerm', 'numPrecondition',
            'numEffect', 'numAssign', 'numEffectExpr', 'numAssignExpr',
            'numGoal', 'numCondEffect', 'numCondEffectExpr', 'numCondAssignExpr',
        }),
        kind_features=(
            'has_int_fluents', 'has_real_fluents', 'has_numeric_fluents',
            'has_simple_numeric_planning', 'has_general_numeric_planning',
            'has_increase_effects', 'has_decrease_effects',
            'has_fluents_in_numeric_assignments',
            'has_static_fluents_in_numeric_assignments',
            # A fluent used *only* in a duration does not make the task numeric
            # as far as ProblemKind is concerned -- tests/pddl/arithdrive has
            # `distance`/`speed` but no *_FLUENTS feature, because neither ever
            # appears in a condition or an effect. The durative split turns those
            # durations into numVariable/numTerm facts all the same, so the
            # duration features have to imply this layer too.
            'has_fluents_in_durations', 'has_static_fluents_in_durations',
        ),
    ),
    Layer(
        name='disjunctive',
        filename='disjunctive.lp',
        summary='disjunctive / existential / nested conditions and goals',
        requires=('core',),
        owns=frozenset({
            'orGroup', 'orDisjunct', 'orDisjunctId', 'orDisjunctNum', 'orDisjunctSub',
            'goalOrGroup', 'goalOrDisjunct', 'goalOrDisjunctId', 'goalOrDisjunctNum',
            'goalOrDisjunctSub',
        }),
        kind_features=('has_disjunctive_conditions', 'has_existential_conditions'),
    ),
    Layer(
        name='temporal',
        filename='temporal.lp',
        summary="PDDL 2.1 durative actions as SMTPlan-style happenings",
        requires=('core',),
        owns=frozenset({
            'durativeAction', 'snap', 'durationValue', 'overall', 'numOverall',
            'minSeparation',
        }),
        kind_features=(
            'has_continuous_time', 'has_discrete_time', 'has_duration_inequalities',
            'has_expression_duration', 'has_int_type_durations',
            'has_real_type_durations',
        ),
    ),
)


@dataclass(frozen=True)
class EncodingFamily:
    """A horizon strategy and the feature layers that can be stacked on it.

    `layers` is in dependency order: a layer never precedes one it requires, and
    that order is also the load order, so ``core.lp`` -- which declares the
    program parts and the ``query(t)`` external -- is always first.

    `unsupported` names the fact vocabulary this family has no layer for, by
    the name of the feature it belongs to, with the reason. A task emitting one
    of those facts is refused with that reason at construction -- the generic
    coverage error would say the registry has drifted, when what is true is
    that the family does not take the task.
    """
    name: str
    directory: str
    layers: Tuple[Layer, ...]
    base: str = 'core'
    unsupported: Tuple[Tuple[str, FrozenSet[str], str], ...] = ()

    @property
    def by_name(self) -> Dict[str, Layer]:
        return {layer.name: layer for layer in self.layers}

    @property
    def layer_names(self) -> Tuple[str, ...]:
        return tuple(layer.name for layer in self.layers)

    def path(self, name: str) -> str:
        return os.path.join(self.directory, self.by_name[name].filename)

    def paths(self, names: Sequence[str]) -> List[str]:
        return [self.path(name) for name in names]

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def close(self, names: Iterable[str]) -> Tuple[str, ...]:
        """`names` plus everything they require, in this family's load order."""
        known = self.by_name
        wanted = {self.base}
        pending = list(names)
        while pending:
            name = pending.pop()
            if name in wanted:
                continue
            if name not in known:
                raise ValueError(
                    f"Unknown {self.name} encoding layer {name!r}; available: "
                    f"{', '.join(self.layer_names)}")
            wanted.add(name)
            pending.extend(known[name].requires)
        return tuple(n for n in self.layer_names if n in wanted)

    def layers_from_facts(self, predicates: Set[str]) -> Set[str]:
        """The layers the emitted fact vocabulary proves are needed."""
        needed = {self.base}
        for layer in self.layers:
            if layer.owns & predicates:
                needed.add(layer.name)
        for predicate in predicates & PREDICATE_REQUIRES.keys():
            needed.update(PREDICATE_REQUIRES[predicate])
        return needed

    def layers_from_kind(self, kind) -> Set[str]:
        """The layers a ``ProblemKind`` suggests.

        Advisory only -- see the module docstring. Read this off the *compiled*
        problem: the user's original kind describes a task that the compilation
        pipeline may have since rewritten.
        """
        needed = {self.base}
        for layer in self.layers:
            for feature in layer.kind_features:
                probe = getattr(kind, feature, None)
                if callable(probe) and probe():
                    needed.add(layer.name)
                    break
        return needed

    def select(self, predicates: Set[str], kind=None) -> Tuple[str, ...]:
        """The layers to load for a task, from its facts and (optionally) its kind."""
        needed = self.layers_from_facts(predicates)
        if kind is not None:
            needed |= self.layers_from_kind(kind)
        return self.close(needed)

    # ------------------------------------------------------------------
    # Safety net
    # ------------------------------------------------------------------

    def unowned(self, predicates: Set[str], names: Iterable[str]) -> Set[str]:
        """Emitted predicates that no selected layer consumes."""
        owned = set()
        for name in names:
            owned |= self.by_name[name].owns
        every = set().union(*(layer.owns for layer in self.layers))
        # Only judge predicates this family knows about at all. An unrecognized
        # one is a gap in the registry rather than a mis-selected layer, and is
        # reported separately by `unregistered`.
        return (predicates & every) - owned

    def unregistered(self, predicates: Set[str]) -> Set[str]:
        """Emitted predicates no layer of this family claims.

        A non-empty result means the encoder grew a fact the registry was never
        told about -- the `owns` sets have drifted from `aspplanners.plasp.facts`.
        """
        every = set().union(*(layer.owns for layer in self.layers))
        return predicates - every

    def refuse_unsupported(self, predicates: Set[str]) -> None:
        """Raise, with the reason, when the facts need a layer this family has not."""
        for feature, vocabulary, reason in self.unsupported:
            if predicates & vocabulary:
                raise ValueError(
                    f"The {self.name} encoding has no {feature} layer: {reason}. This task "
                    f"emits {sorted(predicates & vocabulary)}, so it cannot be solved with "
                    f"{self.name!r}; use the 'seq' encoding for it.")

    def check_coverage(self, predicates: Set[str], names: Sequence[str]) -> None:
        """Raise unless every emitted fact has a layer that reads it."""
        self.refuse_unsupported(predicates)
        missing = self.unowned(predicates, names)
        if missing:
            culprits = sorted(
                {layer.name for layer in self.layers if layer.owns & missing})
            raise ValueError(
                f"The {self.name} encoding layers {list(names)} do not cover the facts "
                f"this task emits: {sorted(missing)} would be ignored, and the plan "
                f"would not respect them. Add the layer(s) {culprits}.")
        stray = self.unregistered(predicates)
        if stray:
            raise ValueError(
                f"The {self.name} encoding registry does not know the fact predicate(s) "
                f"{sorted(stray)}; aspplanners.plasp.facts emits them but no layer in "
                f"aspplanners.plasp.layers claims them. Add them to a layer's `owns`.")


SEQ = EncodingFamily(
    name='seq',
    directory=os.path.join(ENCODINGS_DIR, 'seq'),
    layers=SEQ_LAYERS,
)

_SEQ_BY_NAME = {layer.name: layer for layer in SEQ_LAYERS}

# The lifted family (aspplanners.plasp.lifted): a schema and its arguments are
# chosen per step and every precondition and effect is a step rule of its own,
# so a model never names a ground action. One layer, since the step rules come
# from the encoder rather than from a file; it owns the schema vocabulary in
# place of seq's action facts, and declares what it has no layer for.
LIFTED_LAYERS: Tuple[Layer, ...] = (
    Layer(
        name='core',
        filename='core.lp',
        summary='multi-valued STRIPS with the action schema and its arguments '
                'chosen per step; preconditions and effects are per-schema step rules',
        owns=frozenset({
            'type', 'constant', 'has', 'variable', 'boolean', 'schema', 'param',
            'initialState', 'goal',
        }),
        describes_instance=_SEQ_BY_NAME['core'].describes_instance,
    ),
    # The numeric layer keyed by expression and fluent (numval, numValue, the
    # delta sum, the goals); the lifted encoder states a schema's comparisons
    # and effects as step rules, and declares expressions with numValueNeeded.
    Layer(
        name='numeric',
        filename='numeric.lp',
        summary=_SEQ_BY_NAME['numeric'].summary,
        requires=('core',),
        owns=frozenset({'numVariable', 'numConst', 'numTerm', 'numGoal', 'numValueNeeded'}),
        kind_features=_SEQ_BY_NAME['numeric'].kind_features,
    ),
)

LIFTED = EncodingFamily(
    name='lifted',
    directory=os.path.join(ENCODINGS_DIR, 'lifted'),
    layers=LIFTED_LAYERS,
    unsupported=(
        ('disjunctive', _SEQ_BY_NAME['disjunctive'].owns | frozenset({'orDisjunctNum', 'goalOrDisjunctNum'}),
         "disjunctive conditions would need the disjunctive layer restated per schema"),
        ('temporal', _SEQ_BY_NAME['temporal'].owns | frozenset({'numOverall'}),
         "happenings are built on one ground snap action per step"),
    ),
)

FAMILIES: Dict[str, EncodingFamily] = {SEQ.name: SEQ, LIFTED.name: LIFTED}


def parse_selection(spec: str) -> Tuple[str, Optional[Tuple[str, ...]]]:
    """Split an encoder spec into its family and its explicit layers.

    ``'seq'``                   -> ``('seq', None)``  (layers chosen automatically)
    ``'seq+numeric+temporal'``  -> ``('seq', ('numeric', 'temporal'))``
    ``'lifted'``                -> ``('lifted', None)``

    ``None`` for the layers means "decide from the task"; an explicit list is
    still closed under `requires` and still coverage-checked, so forcing a set
    that cannot express the task fails instead of quietly under-constraining it.
    """
    parts = [part.strip() for part in str(spec).split('+') if part.strip()]
    if not parts:
        raise ValueError(f"Empty encoder spec: {spec!r}")
    family, *layers = parts
    if family not in FAMILIES:
        raise ValueError(
            f"Unsupported encoding family {family!r}; available: {', '.join(sorted(FAMILIES))}")
    return family, (tuple(layers) if layers else None)
