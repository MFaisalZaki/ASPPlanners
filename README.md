# ASPPlanners

A lightweight planner that solves automated planning problems by compiling them to Answer Set Programming (ASP) and delegating search to [clingo](https://potassco.org/clingo/). ASPPlanners plugs into the [Unified Planning](https://github.com/aiplan4eu/unified-planning) (UP) framework as a `OneshotPlanner` engine, so any UP problem can be solved through a uniform interface.

Two backends share the same UP front-end (compilation pipeline, map-back, validation) and register as UP engines on import:

| engine | encoding | solver |
|---|---|---|
| `PLASPPlanner` (default) | PLASP-style multi-valued ASP, multi-shot over the horizon | [clingo](https://potassco.org/clingo/) |
| `ABAPlanner` (optional) | STRIPS-to-ABA (assumption-based argumentation) reduction | [aspforaba](https://bitbucket.org/coreo-group/aspforaba) |

Both are **satisficing** planners: iterative deepening stops at the first horizon that admits a plan, and reports `SOLVED_SATISFICING`. Because the horizon grows one step at a time, the plan returned is shortest *in number of steps*, but no quality metric (`total-cost`, `total-time`) is optimised — metrics are accepted and ignored.

---

# 1. Installation and quick usage

## Requirements

Python 3.10+. Runtime dependencies are installed automatically: `clingo>=5.6.0`, `unified-planning>=1.1.0`, `up_fast_downward>=0.5.2`.

## Install

```bash
git clone https://github.com/MFaisalZaki/ASPPlanners.git
cd ASPPlanners
pip install -e .
```

The `ABAPlanner` backend additionally needs `aspforaba`; install it with the optional `aba` extra:

```bash
pip install -e ".[aba]"
```

It is imported lazily, so `import aspplanners` and the default `PLASPPlanner` engine work without it.

## Quick usage

### Through the Unified Planning framework

Importing `aspplanners` registers the engine, so the standard UP entry points work out of the box:

```python
import aspplanners  # registers the PLASPPlanner and ABAPlanner engines
from unified_planning.shortcuts import OneshotPlanner

# `problem` is any unified_planning.model.Problem you constructed or parsed.
with OneshotPlanner(name="PLASPPlanner") as planner:
    result = planner.solve(problem, timeout=60)
    print(result.status)   # SOLVED_SATISFICING
    print(result.plan)     # move(l1, l2)
```

From PDDL files:

```python
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner
import aspplanners

problem = PDDLReader().parse_problem("gripper/domain.pddl", "gripper/prob01.pddl")
with OneshotPlanner(name="PLASPPlanner") as planner:
    print(planner.solve(problem, timeout=60).plan)
```

### Direct API

You can also drive the planner directly if you don't need the UP result wrapper:

```python
from aspplanners.plasp.planner import PLASPPlanner

planner = PLASPPlanner(problem, encoder_type="seq")
plan = planner.plan(max_horizon=100, timeout=60)   # or plan(horizon=10)
print(planner.status)   # PlanGenerationResultStatus of the last call
print(planner.logs)     # human-readable notes
print(planner.layers)   # ('core', 'numeric') — the encoding layers this task needs
```

`plan()` returns a `SequentialPlan` — or a `TimeTriggeredPlan` when the task has durative actions. It is empty when no plan was found (check `planner.status`) — or when the goal already holds in the initial state, in which case `status` is `SOLVED_SATISFICING`.

### Engine options

| option | engine | default | meaning |
|---|---|---|---|
| `max_horizon` | both | 1000 | upper bound on the iterative-deepening search |
| `horizon` | both | — | solve at one fixed horizon instead of deepening |
| `time_scale` | both | 10 | resolution of the temporal encoding, see [`time_scale`](#time_scale) |
| `encoder_type` | PLASP | `"seq"` | encoding family |
| `semantics` | ABA | `"ST"` | aspforaba extension semantics |
| `timeout` | PLASP only | — | honored by `PLASPPlanner`; **`ABAPlanner` ignores it**, so bound its search with `max_horizon` |

```python
with OneshotPlanner(name="PLASPPlanner", params={"max_horizon": 50}) as planner:
    ...
```

### Refusals

Both engines refuse tasks they cannot encode, and a refusal is part of the result rather than an exception that escapes it:

```python
result = planner.solve(problem)
if result.status == PlanGenerationResultStatus.UNSUPPORTED_PROBLEM:
    print(result.log_messages[0].message)   # ... cannot encode this task: <why>
```

`supported_kind()` answers for the task's *declared* features and is deliberately broad. Two things it cannot see also mean "outside this engine's fragment", and all three arrive the same way:

| refused | when | example |
|---|---|---|
| the `ProblemKind` is outside `supported_kind()` | before encoding | a feature the backend does not implement |
| the encoder reads the task and declines a shape | encoding time | a product of two fluents the task writes; an effect reading its own target with a fractional coefficient |
| clingo cannot represent the program the encoding built | grounding time | a `#sum` whose weights leave the signed 32-bit range |

The third is why this is a status and not a pre-flight check: the encoding states a numeric comparison as a difference `#sum` over the *reachable values* of its fluents, so on a task whose fluents declare no bounds there is nothing in the task to check against — the limit is only visible once clingo grounds it. `aspplanners.common.errors.UnsupportedTaskError` is the type these raise internally; it subclasses `NotImplementedError`, so code written against the older convention of catching that still works.

### Inspecting the generated logic program

```python
text = planner.lp_program()          # task facts + multi-shot encoding, verbatim
planner.dump_lp_program("task.lp")   # same, written to a file

terms = planner.encoding_terms()     # encoding parsed into ASPTerm statements
print(planner.encoding_paths)        # the .lp files loaded, in load order
```

`encoding_terms()` returns typed statements (`ASPFact`, `ASPRule`, `ASPConstraint`, `ASPWeakConstraint`, `ASPDirective`) wrapping clingo AST nodes — filter or rewrite them programmatically, then write them back out:

```python
from aspplanners.lp_io import parse_lp_file, dump_lp, ASPRule

terms = parse_lp_file("my-encoding.lp")
rules = [t for t in terms if isinstance(t, ASPRule)]
dump_lp(terms, "normalized.lp")   # also accepts fact-builder terms and plain strings
```

The encoding is split into `#program base / step(t) / check(t)` parts; ground `base` + `step(1..h)` + `check(h)` and set the external `query(h)` to true to solve at horizon `h`.

---

# 2. Supported PDDL features

## At a glance

| PDDL requirement | `PLASPPlanner` | `ABAPlanner` | notes |
|---|---|---|---|
| `:strips` | ✅ | ✅ | |
| `:typing` | ✅ | ✅ | flat and hierarchical |
| `:equality` | ✅ | ✅ | |
| `:negative-preconditions` | ✅ | ✅ | native in PLASP (multi-valued); compiled away for ABA |
| `:disjunctive-preconditions` | ✅ | ✅ | native in PLASP; compiled away for ABA |
| `:existential-preconditions` | ✅ | ✅ | native in PLASP; compiled away for ABA |
| `:universal-preconditions` | ✅ | ✅ | native in PLASP; compiled away for ABA |
| `:conditional-effects` | ✅ | ❌ | the ABA reduction has no encoding for them |
| `forall` **effects** | ✅ | ✅ unconditional only | native in PLASP (the quantifier is expanded by the grounder, like a `forall` condition); compiled away for ABA, so `(forall … (when …))` is still refused there as a conditional effect |
| `:fluents` / `:numeric-fluents` | ✅ linear | ✅ simple numeric | see [Numeric planning](#numeric-planning) |
| `:action-costs` | ⚠️ accepted | ⚠️ accepted | parsed and ignored; **not optimised** |
| `:durative-actions` | ✅ | ✅ | PDDL 2.1, see [Temporal planning](#temporal-planning) |
| `:duration-inequalities` | ✅ | ✅ | |
| `:timed-initial-literals` | ❌ | ❌ | `TIMED_EFFECTS` / `TIMED_GOALS` |
| `:continuous-effects`, PDDL+ `:processes` / `:events` | ❌ | ❌ | |
| `:derived-predicates` (axioms) | ❌ | ❌ | |
| `:preferences` / `:constraints` | ❌ | ❌ | rejected by the UP PDDL reader before the encoder sees them |

The authoritative list is `supported_kind()` in [aspplanners/up_engines.py](aspplanners/up_engines.py), in UP `ProblemKind` terms:

| `ProblemKind` feature | PLASP | ABA |
|---|---|---|
| `ACTION_BASED`, `FLAT_TYPING`, `HIERARCHICAL_TYPING`, `BOUNDED_TYPES` | ✅ | ✅ |
| `NEGATIVE_CONDITIONS`, `DISJUNCTIVE_CONDITIONS`, `EQUALITIES`, `EXISTENTIAL_CONDITIONS`, `UNIVERSAL_CONDITIONS` | ✅ | ✅ |
| `INT_FLUENTS`, `REAL_FLUENTS`, `INCREASE_EFFECTS`, `DECREASE_EFFECTS`, `SIMPLE_NUMERIC_PLANNING` | ✅ | ✅ |
| `GENERAL_NUMERIC_PLANNING`, `STATIC_FLUENTS_IN_NUMERIC_ASSIGNMENTS`, `FLUENTS_IN_NUMERIC_ASSIGNMENTS` | ✅ | ❌ |
| `CONDITIONAL_EFFECTS` | ✅ | ❌ |
| `FORALL_EFFECTS` | ✅ | ✅ (expanded by the pipeline's quantifier remover) |
| `UNDEFINED_INITIAL_NUMERIC`, `UNDEFINED_INITIAL_SYMBOLIC` | ✅ | ❌ |
| `CONTINUOUS_TIME`, `DURATION_INEQUALITIES`, `INT_TYPE_DURATIONS`, `REAL_TYPE_DURATIONS`, `STATIC_FLUENTS_IN_DURATIONS` | ✅ | ✅ |
| `MAKESPAN`, `ACTIONS_COST`, `PLAN_LENGTH`, `FINAL_VALUE` and the four `*_IN_ACTIONS_COST` kinds | ⚠️ accepted, never optimised | ⚠️ same |
| `TIMED_EFFECTS`, `TIMED_GOALS`, `PROCESSES`, `EVENTS`, `SELF_OVERLAPPING`, `INTERMEDIATE_CONDITIONS_AND_EFFECTS`, `OVERSUBSCRIPTION` | ❌ | ❌ |

A task outside the declared kind is refused up front with `UNSUPPORTED_PROBLEM` rather than silently mis-encoded. `ProblemKind` cannot express every distinction, so a few shapes are raised at *encoding* time instead, as `NotImplementedError`:

- a product of two numeric fluents the task **writes** (not linear in the state; a product with a *static* factor is linear and is encoded, see [Numeric planning](#numeric-planning)),
- division by a numeric fluent,
- an effect that reads **its own target** with a fractional coefficient (`(assign (v ?b) (+ … (* 0.9 (v ?b))))`) — no scaling reaches it, since the target's own factor cancels,
- a numeric value needing a scale factor past `MAX_NUMERIC_SCALE`,
- a bounded numeric type on a fluent that needs scaling,
- a numeric over-all condition in a durative action (**ABA backend only**).

`GENERAL_NUMERIC_PLANNING` is declared because an effect that merely *reads* a fluent already pushes a task's kind there, and the feature has no linear/non-linear split.

## Customizing the compilation pipeline

Pass `compilationlist` to take over the choice entirely. Each entry is a `[engine_name, CompilationKind]` pair applied in order, and the list is used verbatim — the automatic grounder selection is bypassed, so include or omit the grounder yourself:

```python
from unified_planning.shortcuts import CompilationKind
from aspplanners.plasp.planner import PLASPPlanner

# Solve on the lifted encoding: run the removers but skip the grounder.
pipeline = [
    ["up_quantifiers_remover",            CompilationKind.QUANTIFIERS_REMOVING],
    ["up_negative_conditions_remover",    CompilationKind.NEGATIVE_CONDITIONS_REMOVING],
    ["up_disjunctive_conditions_remover", CompilationKind.DISJUNCTIVE_CONDITIONS_REMOVING],
]
planner = PLASPPlanner(problem, encoder_type="seq", compilationlist=pipeline)
plan = planner.plan()
```

## Encoding layers

The encoding is one `.lp` file per feature, under [aspplanners/plasp/encodings/seq/](aspplanners/plasp/encodings/seq/):

| Layer | File | Covers |
|---|---|---|
| `core` | [core.lp](aspplanners/plasp/encodings/seq/core.lp) | multi-valued STRIPS over the horizon: the action choice, preconditions, (conditional) effects, add-wins arbitration, inertia, the goal test |
| `numeric` | [numeric.lp](aspplanners/plasp/encodings/seq/numeric.lp) | linear numeric fluents, comparisons, effects (conditional ones included) and goals |
| `disjunctive` | [disjunctive.lp](aspplanners/plasp/encodings/seq/disjunctive.lp) | disjunctive, existential and nested conditions and goals |
| `temporal` | [temporal.lp](aspplanners/plasp/encodings/seq/temporal.lp) | PDDL 2.1 durative actions as SMTPlan-style happenings, annotated with the SMTPlan constraint each rule mirrors |
| `existstep` | [existstep.lp](aspplanners/plasp/encodings/seq/existstep.lp) | *strategy layer, prototype*: ∃-step parallel semantics — a step carries a set of actions, see [Parallel steps](#parallel-steps-existstep) |

clingo merges `#program` parts of the same name across the files it is given, so the program for a task is the concatenation of the layers it needs, loaded in dependency order (`core` first — it declares the program parts and the `query(t)` external). Layers reference each other's predicates only through `#defined`, which is what lets one be left out without breaking the rest.

**Layers are chosen per task, and you do not normally pick them.** `encoder_type="seq"` infers them from the facts the encoder emitted, widened by what the compiled problem's `ProblemKind` reports:

```python
planner = PLASPPlanner(problem, encoder_type="seq")
planner.layers          # ('core', 'numeric', 'temporal')
planner.encoding_paths  # the .lp files that will be loaded, in load order
```

The emitted facts are the authority — they describe the task *as encoded*, after the compilation pipeline, TIM inference, numeric rescaling and the durative split. `ProblemKind` is a second opinion only: it reports declared features rather than the vocabulary the encoding consumes, and the two need not agree (a durative action whose duration reads a fluent produces numeric facts on a problem whose kind has no numeric feature at all). Where they differ, the union wins — an unneeded layer grounds to nothing, while a missing one would be silently ignored.

### Parallel steps (`existstep`)

Under the default reading a horizon step holds **one** action, so the horizon the deepening search has to reach is the plan's length, and every unsatisfiable horizon on the way is paid for in full. The `existstep` layer switches the horizon to **∃-step parallel semantics** (Rintanen, Heljanko & Niemelä, AIJ 2006): a step holds a *set* of actions, legal when there is an order in which they can be executed one after the other from the state before the step and that execution ends in the state core already computes by applying all their effects at once. Every sequential plan is still a plan, so nothing is lost; independent actions stop costing a horizon step each.

```python
planner = PLASPPlanner(problem, "seq+existstep")   # or params={"encoding": "seq+existstep"} through UP
plan = planner.plan()
planner.layers            # ('core', 'existstep') -- plus whatever feature layers the task needs
planner.solved_horizon    # the number of parallel steps; len(plan.actions) is still the plan length
```

The order is fixed up front rather than searched for: clingo's own order on the ground action terms. For a fixed order a set is legal iff no action *disables* a later one (writes a variable the later one reads, to another value than it reads) and no two actions write one variable to different values — the layer states the first as a chain per read (variable, value) and the second as a count of writers per variable, so both are linear in the step. "Reads" covers preconditions, the conditions of conditional effects and every literal of a disjunctive condition; "writes" covers every effect, conditional or not — both over-approximate, which only ever forbids a step that would have been fine. The planner serialises each step in that same order, so the sequential plan it validates and returns is the order the step was legal under.

`existstep` is a *strategy* layer: it owns no fact and no `ProblemKind` feature, so it is never inferred — you name it — and naming it adds it on top of the feature layers the task needs rather than replacing them (`seq+existstep` on a task with disjunctive conditions loads `core`, `disjunctive`, `existstep`). It is a prototype scoped to classical tasks: it **excludes `numeric`** (the numeric layer's effects and comparisons are not in its read/write analysis yet, so two increments of one fluent could share a step) and **`temporal`** (happenings are built on one snap action per step), and constructing a planner that would need either raises rather than searching with a program that is silently wrong. The benchmark configuration carries it as [plasp-existstep.json](benchmarks/planners/plasp-existstep.json) on the classical track.
