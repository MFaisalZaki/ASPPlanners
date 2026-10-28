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
| `search` | PLASP | `"deepening"` | `"pdr"`: property-directed reachability instead of horizon deepening, see [PDR](#property-directed-reachability); classical tasks on `seq` or `lifted`, `max_horizon` bounds its layers |
| `pdr_first` | PLASP | 0 | seconds of PDR on the task's boolean abstraction before deepening: a proof of unsolvability or a validated plan ends the search, see [PDR first](#pdr-first-on-the-boolean-abstraction); classical and numeric tasks |
| `symmetry` | PLASP | `"off"` | symmetry-breaking constraints on the deepening: `"objects"` (first-use order of interchangeable objects), `"order"` (adjacent independent actions in term order) or `"all"`, see [Symmetry breaking](#symmetry-breaking); `seq`, plans stay shortest |
| `time_scale` | both | 10 | resolution of the temporal encoding, see [`time_scale`](#time_scale) |
| `encoding` | PLASP | `"seq"` | encoding family: `"seq"` or `"lifted"`, see [Encoding families](#encoding-families) |
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

## Encoding families

Two encoding families, one directory each under [aspplanners/plasp/encodings/](aspplanners/plasp/encodings/), share the task facts, the multi-shot structure and the goal test, and differ in how an action is chosen at a step:

| Family | Directory | A step chooses | Layers |
|---|---|---|---|
| `seq` | [seq/](aspplanners/plasp/encodings/seq/) | one ground action | core, numeric, disjunctive, temporal |
| `lifted` | [lifted/](aspplanners/plasp/encodings/lifted/) | one action schema and one object per parameter position | core, numeric |

### The `lifted` family

`seq` instantiates every ground action once, in base, and replicates each step rule per ground action: the step choice ranges over them, and the preconditions and effects are facts keyed by the ground action term that one generic rule per step consumes. A step therefore costs as much as the task has ground actions — objects to the power of the action's arity — before anything is searched.

The `lifted` family (after Höller & Behnke, *Encoding Lifted Classical Planning in Propositional Logic*, ICAPS 2022; see [aspplanners/plasp/lifted.py](aspplanners/plasp/lifted.py)) never names a ground action. A step chooses a schema, `schema(S, t)`, and one object per parameter position, `arg(S, I, O, t)`, and the encoder emits every precondition and effect as a step rule of its own that joins only the argument atoms of the parameters its literal mentions:

```prolog
:- schema("pick", t), arg("pick", 3, GRIPPER, t),
   not holds(variable(("free",GRIPPER)), value(variable(("free",GRIPPER)), true), t-1).
caused(variable(("carry",OBJ,GRIPPER)), value(variable(("carry",OBJ,GRIPPER)), true), t) :-
   schema("pick", t), arg("pick", 1, OBJ, t), arg("pick", 3, GRIPPER, t).
```

A literal then costs as many ground instances per step as its predicate has ground atoms, and the step choice one atom per (schema, position, object). From `caused/3` on — add-wins arbitration, inertia, the goal test — [lifted/core.lp](aspplanners/plasp/encodings/lifted/core.lp) is seq's core unchanged, and the semantics is seq's: one action per step, the same plans at the same horizons. The rules come from the same parsers the seq action builder uses, so the two agree literal for literal: positive and negative conditions, `forall` conditions (one constraint per binding), equalities, static preconditions (checked against the initial state), unconditional and conditional effects, `forall` effects, and a `forall` inside a condition (a conditional literal, gringo's conjunction over the bindings).

```python
planner = PLASPPlanner(problem, "lifted")   # or params={"encoding": "lifted"} through UP
plan = planner.plan()
planner.step_rules    # the per-schema rules of step(t); lp_program() dumps them after the layers
```

Numeric fluents are stated the same way. seq keys a comparison or an effect by the action term (`numPrecondition(A, ...)`, `numEffect(A, V, D)`) and one generic rule per step consumes them; the lifted encoder emits each as a per-schema step rule with the same body — the comparison's difference `#sum`, the delta applied to the value at `t-1`, the expression-valued forms through `numValue/3` — under the argument atoms of the parameters it mentions, and declares the expressions' `numConst`/`numTerm` under the parameters' types. [lifted/numeric.lp](aspplanners/plasp/encodings/lifted/numeric.lp) is a verbatim copy of seq's, of which the parts keyed by expression or fluent (`numval`, `numValue`, the delta sum, the numeric goals) do the rest. Linear comparisons and constant or expression-valued increase, decrease and assign effects are covered; a conditional numeric effect, a disjunctive condition and a durative action are refused at construction with the reason. On a 12-ball, 4-gripper gripper the lifted step program grounds to less than half of seq's atoms; where every action has as many parameters as its predicates (blocksworld's `stack(x, y)` over `on(x, y)`) there is nothing to gain. The benchmark configuration carries it as [plasp-lifted.json](benchmarks/planners/plasp-lifted.json) on the classical and numeric tracks.

## Encoding layers

Within a family the encoding is one `.lp` file per feature; for `seq`, under [aspplanners/plasp/encodings/seq/](aspplanners/plasp/encodings/seq/):

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

## Property-directed reachability

Horizon deepening pays for every unsatisfiable horizon on the way to the first satisfiable one, and learns nothing durable from the goal side: everything clingo derives through the goal test of horizon *t* carries the external `query(t)` and dies when that external is released. `search="pdr"` replaces the deepening with property-directed reachability (Bradley 2011; for planning, Suda, *Property Directed Reachability for Automated Planning*, JAIR 2014), which never unrolls the horizon:

* it keeps **layers of blocked cubes** B₁, B₂, …, a cube in Bᵢ being a partial state none of whose completions can reach the goal within *i* steps;
* it works on **obligations** "state *s* must reach the goal in *i* steps": find an action from *s* whose successor is a goal state (*i* = 1) or not blocked at *i*−1, or, failing that, learn a **reason** — a subset of *s*'s literals that already rules every action out — and block it at *i*;
* layers are never thrown away, and when a layer stops growing the task is **proven unsolvable** (`UNSOLVABLE_PROVEN`), which deepening can never do.

Every PDR query is a question about one step, so one ground program answers all of them: [encodings/seq/pdr.lp](aspplanners/plasp/encodings/seq/pdr.lp) turns the state at time point 0 into a free choice that each query fixes through solver assumptions (the planner adds the task facts with `initialState` renamed to `pdrInitialState`, so core's own initial-state rule is inert), lets `step(1)` be the seq encoding's own step, adds a do-nothing action so that a goal state counts as reaching the goal, and gates the goal test at time 1 through `query(1)` and each learned cube through a switch atom the query pins on or off. The reason extraction — the hard part of any PDR — is clingo's unsatisfiable core over the state assumptions, shrunk further by dropping literals while the query stays unsatisfiable. The loop around it is [aspplanners/plasp/pdr.py](aspplanners/plasp/pdr.py).

```python
planner = PLASPPlanner(problem)             # seq encoding
plan = planner.plan(search="pdr")           # or params={"search": "pdr"} through UP
planner.status                              # SOLVED_SATISFICING | UNSOLVABLE_PROVEN | ...
planner.logs[0]                             # 'PDR: plan (41 one-step queries, 9 cubes learned (per layer: {1: 3, 2: 4, 3: 2}))'
planner.pdr_program()                       # the one-step program, standalone
```

Plans are not guaranteed shortest: a blocked obligation is re-tried one layer up with the path it came with (IC3's rescheduling, which is what makes PDR fast here — without it every layer has to refute every shorter length first, and the measured times go from seconds to timeouts), so the plan found at target layer *k* can be longer than *k*. An obligation is only re-tried while its path length plus its level stays within `max_horizon`, which bounds the length of any plan explored. `max_horizon` also bounds the number of layers.

PDR is **opt-in and classical-only**: `search="pdr"` on the `seq` or the `lifted` encoding ([lifted/pdr.lp](aspplanners/plasp/encodings/lifted/pdr.lp) is seq's with the do-nothing action as a schema), for tasks without numeric fluents or durative actions. Numeric support was built and measured — a program grounded per query with the obligation's numeric values as facts, which is correct and general — and dropped: a reason can never generalise over a numeric value, so the layers grow with every value seen, and on a gripper with a fuel counter it lost to deepening by two orders of magnitude. Numeric PDR needs interval reasons, which is IC3 modulo theories. The benchmark configuration carries PDR as [plasp-seq-pdr.json](benchmarks/planners/plasp-seq-pdr.json) on the classical track.

### PDR first, on the boolean abstraction

What does work on numeric tasks is running PDR on their **boolean abstraction**: leave the numeric layer out, so numeric preconditions and goal conditions fall away, more actions apply and more states count as goals. That over-approximates reachability, so a proof of unsolvability found there holds for the real task, and a plan found there is a candidate the validator checks against the real task — real whenever the numbers were slack enough. `pdr_first=<seconds>` does exactly that before the deepening (`params={"pdr_first": 60}` through UP), on `seq` and on `lifted` (whose numeric step rules the abstraction leaves out with the layer): a proof or a validated plan ends the search, anything else hands over to the deepening unchanged. On a classical task the abstraction is the task itself, so this is simply PDR with a budget and deepening as the fallback. The one shape it skips is a boolean effect conditioned on a numeric comparison, which the abstraction would make fire always rather than maybe.

On a 6-ball gripper with a fuel counter and fuel to spare, the abstract plan is valid and the task is solved in 3.3s where deepening needs 33s; with the fuel tight the abstract plan is rejected and the deepening does the work, so the cost is the budget. Feeding the learned cubes into the deepening as distance-to-goal constraints was also tried, and cost more per step than it saved on every instance, so it is not part of this. The benchmark configuration carries it as [plasp-seq-pdr-first.json](benchmarks/planners/plasp-seq-pdr-first.json) and [plasp-lifted-pdr-first.json](benchmarks/planners/plasp-lifted-pdr-first.json) on the classical and numeric tracks, and PDR alone as [plasp-seq-pdr.json](benchmarks/planners/plasp-seq-pdr.json) and [plasp-lifted-pdr.json](benchmarks/planners/plasp-lifted-pdr.json) on the classical track.

## Symmetry breaking

The deepening pays for every horizon below the shortest plan with a proof that no plan of that length exists, and on a task with interchangeable objects clasp proves it once per relabelling: a 6-ball gripper has 720 ways to name the balls and the proof at each unsatisfiable horizon is repeated for each. `symmetry="objects"` states the lex-leader constraint that collapses them (Rintanen 2003): two objects of a type are interchangeable when swapping them leaves the initial state and the goal unchanged and no action schema names either as a constant, the interchangeable objects form classes under the full symmetric group, and object *i+1* of a class may not be used by an action before object *i*. [symmetry.py](aspplanners/plasp/symmetry.py) finds the classes on the compiled task (an exact swap check per candidate pair, after bucketing by the shape of each object's own initial facts) and states them as `symClass(Class, Index, constant("name"))` facts, indexed in clingo's own term order of the constants; which objects an action mentions is one `mentions/2` rule per schema parameter. [seq/symmetry.lp](aspplanners/plasp/encodings/seq/symmetry.lp) then needs three rules: `used/2`, its monotone closure, and the constraint.

`symmetry="order"` adds the plan-permutation constraint (Long & Fox 2003): two adjacent actions neither of which writes a variable the other reads or writes can be swapped, so a plan has to list them in ascending term order. The independence relation is computed in the layer from the action facts at the level of variables, and an action whose shape the layer does not analyse (a numeric conditional effect, a numeric disjunct) interferes with everything. `symmetry="all"` is both. Each constraint keeps the term-smallest plan of every symmetry class, and the two are compatible because the class index order is clingo's term order, so the shortest horizon is unchanged; every bundled domain is solved at the same horizon with them as without ([tests/test_symmetry.py](tests/test_symmetry.py)). Temporal tasks skip the layer with a log line (a swap of snap actions changes the schedule), and the lifted encoding does not have it yet.

Measured on the seq deepening, 300s/600s limits:

| task | `off` | `objects` | `order` | `all` |
|---|---|---|---|---|
| gripper, 6 balls (h=17) | 28.4s | 0.41s | 26.7s | 0.47s |
| fuel gripper, 6 balls, numeric (h=17) | 30.8s | 0.62s | — | 0.71s |
| gripper, 12 balls (h=35) | horizon 19 reached at 613s, timeout | 728s (horizons 0–33 in 88s, 34 in 455s, 35 in 184s) | — | timeout at 600s, solving h=35 |

The object constraint is where the gain is: the refutation of each horizon below the optimum no longer repeats per relabelling. It does not change the shape of the curve, though: on the 12-ball gripper the time per horizon still grows about fourfold per step near the optimum, and the last unsatisfiable horizon alone costs 455s, so the constraint moves the wall by a few horizons rather than removing it. The order constraint on its own changes little, because learned nogoods already transfer between orderings of two independent actions, and it grounds a `swappable/2` pair per step for every independent pair of ground actions, so it is kept separate and is not the default. The classes are exact (a goal that names an object, or an initial state that tells two apart, keeps them out of any class), so a task without them pays only the detection, which is a pass over the initial state. The benchmark configuration carries it as [plasp-seq-symmetry.json](benchmarks/planners/plasp-seq-symmetry.json) on both tracks.

## Running the benchmarks

The benchmark configuration lives in [benchmarks/](benchmarks/) and is read by [pyPMTEvalToolkit](https://github.com/pyPMT/pyPMTEvalToolkit), which sweeps any planner reachable through the Unified Planning API over the IPC-style benchmark repositories it clones. Nothing in this repository is a harness; see [benchmarks/README.md](benchmarks/README.md) for the details and for how the toolkit labels tracks.

### What gets run

[`benchmarks/exp-details.json`](benchmarks/exp-details.json) sets the limits (30 minutes, 8 GB, plans validated) and the task selection (both tracks, every instance). Each file under [`benchmarks/planners/`](benchmarks/planners/) is one planner configuration — the engine name, the module that registers it, and the engine options from the [table above](#engine-options):

| file | engine options | tracks |
|---|---|---|
| [plasp-seq.json](benchmarks/planners/plasp-seq.json) | `encoding: seq` — horizon deepening | classical, numeric |
| [plasp-seq-symmetry.json](benchmarks/planners/plasp-seq-symmetry.json) | `encoding: seq`, `symmetry: all` | classical, numeric |
| [plasp-seq-pdr.json](benchmarks/planners/plasp-seq-pdr.json) | `encoding: seq`, `search: pdr` | classical |
| [plasp-seq-pdr-first.json](benchmarks/planners/plasp-seq-pdr-first.json) | `encoding: seq`, `pdr_first: 60` | classical, numeric |
| [plasp-lifted.json](benchmarks/planners/plasp-lifted.json) | `encoding: lifted` | classical, numeric |
| [plasp-lifted-pdr.json](benchmarks/planners/plasp-lifted-pdr.json) | `encoding: lifted`, `search: pdr` | classical |
| [plasp-lifted-pdr-first.json](benchmarks/planners/plasp-lifted-pdr-first.json) | `encoding: lifted`, `pdr_first: 60` | classical, numeric |
| [aba-st.json](benchmarks/planners/aba-st.json) | `ABAPlanner`, stable semantics | classical |

A `tracks` entry keeps a configuration off the tasks it refuses: PDR alone is classical-only, so its files do not list the numeric track. To sweep a subset of the planners, copy `benchmarks/` and delete the planner files you do not want; to narrow the tasks, edit `include-domains`, `exclude-domains`, `ipc-years` or `max-instances-per-domain` in the copy's `exp-details.json`.

### Running a sweep

```bash
pip install -e .                                   # the engines the planner files name
git clone https://github.com/pyPMT/pyPMTEvalToolkit.git
cd pyPMTEvalToolkit
./setup_benchmark.sh --config /path/to/ASPPlanners/benchmarks --yes
```

`setup_benchmark.sh` copies the configuration into a working experiment, creates a virtualenv, installs the toolkit and the engines, clones the benchmark repositories, generates the slurm arrays and prints the `sbatch` command. Before submitting, check that every planner file resolves to an engine — it is the fastest way to catch a missing install (`ABAPlanner` needs `pip install -e ".[aba]"`):

```bash
pypmtevalcli engines --exp-dir experiment
```

The stages can also be driven one by one, and everything but `solve` and `engines` is stdlib-only, so a sweep can be generated on a laptop and only the compute nodes carry the planners:

```
pypmtevalcli init | engines | discover | generate | solve | analyze | report
```

Results land as `results/<planner>/<task>.json` with `errors/` and `analysis/` beside them, and the two scripts in [results/](results/) read that layout:

```bash
python results/generate_results.py --sandbox-dir <sandbox>   # coverage and time tables
python results/analyze_errors.py <sandbox>/errors            # cluster the failures: parser vs planner
```

### A quick local comparison

To compare the configurations on one task without the toolkit, the engine options are the same ones the planner files carry:

```python
import time
import aspplanners
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner

problem = PDDLReader().parse_problem("tests/pddl/gripper/domain.pddl", "tests/pddl/gripper/problem.pddl")
for params in ({"encoding": "seq"}, {"encoding": "lifted"},
               {"encoding": "seq", "search": "pdr"}, {"encoding": "seq", "pdr_first": 60}):
    with OneshotPlanner(name="PLASPPlanner", params=params) as planner:
        start = time.time()
        result = planner.solve(problem, timeout=300)
    steps = len(result.plan.actions) if result.plan is not None else 0
    print(f"{params}: {result.status.name}, {steps} actions, {time.time() - start:.2f}s")
```

`result.log_messages` carries the planner's notes: which layers were loaded, and for PDR the number of queries and cubes.
