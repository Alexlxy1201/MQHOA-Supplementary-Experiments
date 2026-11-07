# MQHOA supplementary experiments

This separate implementation adds five supplementary deliverables: a same-encoding PSO comparison, seven hyperparameter settings, fixed-scale and direct-generation controls, uncertainty/statistical analyses, and reproducible source release. The original manuscript, archives, figures and historical results are preserved.

## Status

Source preparation is complete. Execution status and results are recorded separately in `runs/` and the eventual results release. Code availability alone is not evidence that an experiment passed or a method improved.

## Model and methods

- One local resource and one edge resource, one task at a time per resource. Entry and exit stay local. Dependencies include cross-resource communication; transfers can overlap freely. Orders are level-sorted with independent within-level permutations, as in the manuscript.
- MQHOA implements the manuscript's Algorithm 2 in this supplementary package: persistent centres, Box–Muller index sampling, a shared pair of candidate pools, and the stated Hamming-triggered scale reduction.
- PSO: synchronous global-best PSO in the same two-index space. Positions use floor-and-clamp decoding; velocities are clipped to 20% of the index range and reset at a boundary. Population 20; four predeclared parameter settings are tuned separately.
- GA: supplementary index-coded GA with inverse-makespan roulette selection, copied parents, crossover, per-index mutation and one elite. It is labelled as a new implementation, not a rerun of the historical GA.
- HEFT: upward ranks using mean computation costs and the single inter-resource link's communication time; insertion scheduling with the same placement restrictions. Its processor orders are converted into an equivalent order for the common decoder. One shot, not an artificially repeated optimiser.
- NoHamming: MQHOA with fixed sigma=M/2. DirectRandom: fresh placements and legal within-level orders without pools. DirectRandom changes search dynamics as well as encoding, so it is a search-strategy control. RandomPool adds a uniform random-search diagnostic over exactly the shared pools.

No original script is imported or executed. Archived DAG constants are read with `ast.literal_eval`. All new reported schedules are independently checked. New measurements are kept separate from historical results and are never labelled proven optimal.

## Predeclared experiment matrix

| Block | Instances and repeats | Runs |
|---|---|---:|
| Tuning | 4 separate synthetic DAGs; PSO/GA, 4 settings each, 5 disjoint seeds, B=2,000 | 160 |
| Main, sensitivity, ablations | 2 archived DAGs + 12 new DAGs (n=10,50,100 × deep/wide × 2); 12 stochastic configurations × 20 seeds, B=10,000; HEFT once | 3,374 |
| Sequential runtime | Same 14 instances; 12 stochastic configurations × 5 seeds; HEFT 30 times | 1,260 |
| Total | Full MQHOA runs are shared across the three analysis blocks | 4,794 |

The standalone synthetic-only quick start has 2,892 quality runs, 1,080 timing runs and 160 tuning runs (4,132 total). The full matrix above additionally includes the two optional archived DAGs.

The seven MQHOA settings are baseline k=4,m=5,lambda=1.3 plus k=2/6, m=3/8 and lambda=1.2/1.5. Only one parameter changes at a time. No setting is selected after inspecting evaluation results.

Synthetic DAGs are generic workload supplements, not measured optical workloads or Alibaba traces. No external dataset is downloaded. Default pool sizes are M=2,000 for n<50 and M=20,000 otherwise, fixed across methods within an instance. Every fitness call, including initialisation and duplicate candidates, consumes the same evaluation budget.

## Run on an experiment computer

Python 3.10–3.12; no GPU is required. The public quick start below uses the self-contained synthetic configuration and needs no private archive. Use a dedicated virtual environment, then from this folder:

```bash
python -m pip install -r requirements.txt
python -m mqhoa_supp prepare --config configs/synthetic_only.json --out runs/five-v1
python -m mqhoa_supp plan --config configs/synthetic_only.json --out runs/five-v1
python -m mqhoa_supp preflight --config configs/synthetic_only.json --out runs/five-v1
```

Preflight runs the new implementation's tests and short 500-evaluation pilots, then reports a measured serial time estimate. Inspect that estimate before launching long stages. It never runs an archived script or enforces audit-derived optimum values.

```bash
python -m mqhoa_supp tune --config configs/synthetic_only.json --out runs/five-v1 --workers 2
python -m mqhoa_supp freeze --config configs/synthetic_only.json --out runs/five-v1
# Preserve frozen_defaults.json (including its hash/time) before evaluation.
python -m mqhoa_supp run --config configs/synthetic_only.json --out runs/five-v1 --workers 2
# Run timing separately with no competing experimental batch on the machine:
python -m mqhoa_supp timing --config configs/synthetic_only.json --out runs/five-v1 --workers 1
python -m mqhoa_supp analyze --config configs/synthetic_only.json --out runs/five-v1
```

For two computers, prepare and pass preflight on both, tune/freeze once, copy the identical `frozen_defaults.json` to the other computer, and run evaluation with `--shard-count 2 --shard-index 0` and `--shard-count 2 --shard-index 1`. Collect the two evaluation log files into the same results folder. Run all timing comparisons sequentially on just one machine. Analysis requires the complete unique matrix and a single shared frozen configuration.

The parent process writes append-only JSONL and preserves failure records. Resume skips matching completed run IDs; it refuses mixed code/configurations or corrupt/truncated logs. A failed test blocks a campaign. Diagnose a failed run before an intentional resume; there is no automatic retry.

`preflight.json` records the code hash. `frozen_defaults.json` records code, configuration, tuning-result hashes and choices. Every run records its source hash, instance hash, pool hash, seed, exact evaluation count, solution, validation result, trace, timing and host. Keep environment installation logs and `pip freeze` beside each machine's results.

## Statistical outputs

- Per-instance 20-run mean, SD, median, best observed and 95% seed-bootstrap CI. HEFT's identical schedule is not counted as 20 independent quality samples.
- Primary inference across the 12 independent synthetic DAGs: two-sided paired sign-flip randomisation on relative differences in instance means, with Holm adjustment within comparison/ablation families. Bootstrap CIs resample instances. Paired t and Wilcoxon results, effect sizes and Shapiro diagnostics are secondary, predeclared outputs; no test shopping.
- The two archived DAGs are descriptive cases, not a sample of independent workloads. Historical and new values are not pooled.
- Standalone PDF/PNG comparison, convergence, ablation, parameter and runtime figures; CSV summaries; a LaTeX table fragment; and draft reviewer responses based only on completed logs.

## Standalone public release

The public source package contains only the new code, tests, configurations and documentation. Original manuscript/reviewer documents remain in the private archive. To reproduce without the optional original archive, pass `--config configs/synthetic_only.json` to every command. The synthetic matrix then has 12 evaluation and 4 tuning instances. Alternatively place the original archive in the directory passed with `--repo`; the loader reads only literal DAG constants.

## Sources and software assistance

- Kennedy and Eberhart (1995), *Particle swarm optimization*, [doi:10.1109/ICNN.1995.488968](https://doi.org/10.1109/ICNN.1995.488968).
- Topcuoglu, Hariri and Wu (2002), *Performance-effective and low-complexity task scheduling for heterogeneous computing*, [doi:10.1109/71.993206](https://doi.org/10.1109/71.993206).
- SciPy 1.15.3: [paired t tests](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats.ttest_rel.html), [Wilcoxon signed-rank tests](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats.wilcoxon.html).
- Statistical workflow assistance used the `statistical-analysis` skill. Its procedural source is Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, [doi:10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065). This acknowledgement concerns workflow assistance, not scientific validation of MQHOA.

## Final presentation tools

After the completed campaign analysis, `python analysis_tools/finalize.py runs/five-v1` adds a Chinese summary and raw distribution figures. A completed quality matrix can also be summarised while the separate timing phase continues using `python analysis_tools/quality_preview.py runs/five-v1 --config configs/synthetic_only.json`; this writes a provisional quality-only folder. Both tools read recorded results and do not run optimisation.

For the fixed-scale control, the Hamming statistic remains a logged diagnostic; it does not drive sampling or scale updates. Runtime includes that common diagnostic instrumentation.
