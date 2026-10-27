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
| `encoding` | PLASP | `"seq"` | encoding family |
| `parallel` | PLASP | off | `"exists"` or `"forall"`: a set of actions per horizon step, mutexes generated on demand, see [Parallel steps](#parallel-steps-on-demand-mutexes); `"sequential"`: one action per step with the cap enforced by the propagator instead of the encoding; not for temporal tasks |
| `interference` | PLASP | `"lazy"` | `"lazy"` judges whether two actions interfere the first time the solver puts them together, `"eager"` judges every pair up front |
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

clingo merges `#program` parts of the same name across the files it is given, so the program for a task is the concatenation of the layers it needs, loaded in dependency order (`core` first — it declares the program parts and the `query(t)` external). Layers reference each other's predicates only through `#defined`, which is what lets one be left out without breaking the rest.

**Layers are chosen per task, and you do not normally pick them.** `encoder_type="seq"` infers them from the facts the encoder emitted, widened by what the compiled problem's `ProblemKind` reports:

```python
planner = PLASPPlanner(problem, encoder_type="seq")
planner.layers          # ('core', 'numeric', 'temporal')
planner.encoding_paths  # the .lp files that will be loaded, in load order
```

The emitted facts are the authority — they describe the task *as encoded*, after the compilation pipeline, TIM inference, numeric rescaling and the durative split. `ProblemKind` is a second opinion only: it reports declared features rather than the vocabulary the encoding consumes, and the two need not agree (a durative action whose duration reads a fluent produces numeric facts on a problem whose kind has no numeric feature at all). Where they differ, the union wins — an unneeded layer grounds to nothing, while a missing one would be silently ignored.

## Parallel steps: on-demand mutexes

Under the default reading a horizon step holds **one** action, so the horizon the deepening search has to reach is the plan's length, and the unsatisfiable horizons on the way are paid for in full. `parallel="exists"` or `"forall"` lets a step hold a *set* of actions:

* **forall-step** (Kautz & Selman 1996): no two actions of a step interfere, so the step can be executed in any order;
* **exists-step** (Dimopoulos, Nebel & Koehler 1997; Rintanen, Heljanko & Niemelä 2006): the step can be executed in *some* order — the disabling graph induced by its actions is acyclic.

Either needs interference constraints between actions, and encoding them up front is quadratic in the actions per step. Instead, following Abdelwahed, Bofill, Espasa & Villaret, *On-Demand Mutex Constraints for Numeric Planning as SMT* (ICAPS 2026), the encoding states no interference at all — `core.lp`'s step choice drops to `1 {occurs(A, t)}` — and a clingo **propagator** ([aspplanners/plasp/propagators.py](aspplanners/plasp/propagators.py)) watches the action atoms during search: when the solver makes an action true, the forall propagator raises a conflict against any active action of the same step it interferes with, and the exists propagator raises one when the new action closes a cycle in the step's disabling graph, the cycle being the conflict. Conflicts are handed to clingo as nogoods, so a mutex is paid for only once it has been hit and is forgotten once it stops being useful. Interference is computed on first contact and memoised (`interference="lazy"`); `"eager"` precomputes the whole graph for comparison.

```python
planner = PLASPPlanner(problem, parallel="exists")   # or params={"parallel": "exists"} through UP
plan = planner.plan()
planner.solved_horizon   # the number of parallel steps; len(plan.actions) is still the plan length
planner.logs[-1]         # '... (exists-step, lazy interference: 379 mutex conflicts raised, 228 disabling edges among 1042 action pairs judged)'
```

What the propagator knows about an action is read off the ground facts in the solver — `precondition`, `postcondition`, the disjunctive layer's `orDisjunct`, the numeric layer's effects and comparisons — so no encoding layer changes. An action *reads* the variables of its preconditions, of its conditional effects' conditions and of every literal of its disjunctive conditions, and *writes* every effect, conditional or not; `a` disables `b` when it writes a variable `b` reads to another value than `b` reads, or when both write one variable to different values. Numeric fluents are value-less: a numeric write disables every reader and conflicts with every other writer (the numeric layer's unconditional effects do not sum). All of this is syntactic and over-approximates; the paper's *semantic* interference check, which would let two increases of one fluent commute, is the extension point (`InterferenceGraph.disables`). Each step of a model is serialised in a topological order of its disabling graph and the UP validator checks that order. Temporal tasks are refused: happenings are built on one snap action per step.

`parallel="sequential"` completes the matrix: the same one-action-per-step reading as the default, but with core's cardinality constraint dropped and the propagator rejecting any two actions in one step. It admits exactly the same plans and exists to measure the propagator's overhead against clasp's native cardinality constraint.

On the bundled gripper the exists-step reading needs 2 steps for a 5-action plan where the sequential reading needs 5; on a 6-ball gripper, 6 steps and 0.13s against 17 steps and 32s. The benchmark configuration carries both readings as [plasp-seq-exists.json](benchmarks/planners/plasp-seq-exists.json) and [plasp-seq-forall.json](benchmarks/planners/plasp-seq-forall.json).
