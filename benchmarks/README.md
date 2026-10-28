# Benchmark configuration

This directory is **not** a harness. It is the experiment configuration that
[pyPMTEvalToolkit](https://github.com/pyPMT/pyPMTEvalToolkit) reads to sweep
this repository's planners:

```
benchmarks/
├── exp-details.json             limits + task selection
└── planners/
    ├── plasp-seq.json               seq encoding, horizon deepening     (classical, numeric)
    ├── plasp-seq-symmetry.json      seq, symmetry=all                   (classical, numeric)
    ├── plasp-seq-pdr.json           seq, search=pdr                     (classical)
    ├── plasp-seq-pdr-first.json     seq, pdr_first=60 then deepening    (classical, numeric)
    ├── plasp-lifted.json            lifted encoding, horizon deepening  (classical, numeric)
    ├── plasp-lifted-pdr.json        lifted, search=pdr                  (classical)
    ├── plasp-lifted-pdr-first.json  lifted, pdr_first=60 then deepening (classical, numeric)
    └── aba-st.json                  ABAPlanner, stable semantics        (classical)
```

The toolkit runs **any planner reachable through the Unified Planning API**, so
nothing here is specific to ASPPlanners beyond the planner files. All of them
name `up-planner-module: "aspplanners"`, which is how the toolkit reaches an
engine that registers itself on import and ships no UP entry point — importing
`aspplanners` registers `PLASPPlanner` and `ABAPlanner` (see
[`aspplanners/up_engines.py`](../aspplanners/up_engines.py)). The
`planner-params` of a file are the engine options the top-level README's
[engine options table](../README.md#engine-options) lists; a file's `tracks`
keeps it off the tasks its configuration refuses (PDR alone is classical-only).

## Running a sweep

```bash
git clone https://github.com/pyPMT/pyPMTEvalToolkit.git
cd pyPMTEvalToolkit
./setup_benchmark.sh --config /path/to/ASPPlanners/benchmarks --yes
```

`setup_benchmark.sh` copies this configuration into a working experiment,
creates a virtualenv, installs the toolkit and the engines the planner files
name, clones the benchmark repositories, and generates the slurm arrays. It ends
by printing the `sbatch` command.

Check the engines resolve before submitting the sweep — the toolkit does this
for you, and it is the fastest way to catch a missing install:

```bash
pypmtevalcli engines --exp-dir experiment
```

`ABAPlanner` needs the optional `aba` extra (`pip install 'aspplanners[aba]'`);
without it that planner fails to resolve and `engines` exits non-zero.

The stages, if you would rather drive them yourself:

```
pypmtevalcli init | engines | discover | generate | solve | analyze | report
```

Everything except `solve` and `engines` is stdlib-only, so a sweep can be
generated on a laptop and only the compute nodes carry the planners.

## Reading the results

The toolkit writes the same sandbox layout the previous in-tree harness did —
`results/<planner>/<task>.json`, `errors/`, `analysis/` — with the same result
schema and the same status vocabulary, so both of this repository's own tools
keep working against it unchanged:

```bash
python results/generate_results.py --sandbox-dir <sandbox>   # the checked-in tables
python results/analyze_errors.py <sandbox>/errors            # cluster the crash logs
```

[`results/analyze_errors.py`](../results/analyze_errors.py) has no toolkit
equivalent, which is why it stayed: it separates failures raised while *parsing*
PDDL (a reader limitation) from failures raised while *planning* (ours), and
clusters the rest into distinct signatures. On the last sweep it is what showed
that 641 of 650 crashes were the UP PDDL reader rather than the planner.

## One thing that changed with the toolkit

**Track labels.** The toolkit decides a task's track by reading the domain file,
and a `(:functions ...)` block makes it numeric. The harness this replaced
applied a narrower rule: a fluent had to be *used as state* — compared in a
precondition, or assigned somewhere other than a cost accumulator — because
IPC's standard STRIPS encoding declares `(:functions (total-cost) - number)`
purely so plans can be scored.

Measured on the benchmark repositories this project sweeps:

| | previous harness | pyPMTEvalToolkit |
|---|---|---|
| classical | 4,831 | 3,330 |
| numeric | 1,817 | 3,354 |
| total | 6,648 | 6,684 |

**1,537 tasks across 69 domains** — `sokoban`, `woodworking`, `transport`,
`barman`, `elevators` and the rest of the cost-annotated IPC suite — are filed
under numeric by the toolkit and were filed under classical before. The set of
tasks actually *run* is almost unchanged, since both planners take both tracks;
what changes is the denominator of every per-track number.

So the per-track tables in [`results/`](../results/) and in
[§3 of the top-level README](../README.md#3-benchmark-results) describe a task
labelling that a new sweep will not reproduce. Compare totals across the two,
not per-track coverage. `--suite-tracks` does not restore the old split: it
*filters* a repository to some tracks rather than relabelling it, so
`classical-domains=classical` would drop those 1,537 tasks instead of moving
them back.
