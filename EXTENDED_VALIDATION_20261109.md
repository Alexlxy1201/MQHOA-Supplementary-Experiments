# Additional MQHOA validation code

This source-only release adds budget/encoding checks, exact small-instance optimization, executed angular-spectrum optics, shared-user/server simulation, online resource changes, modeled energy, packet retransmission and incremental task revelation. The previous root-level package is preserved. The additional export retains the private campaign's `supplement/` layout so its imports and scripts run unchanged.

## CPU environment and tests

Create a new environment and install `requirements-extended.txt` (Python 3.12 used). CUDA profiling additionally requires a compatible PyTorch/CUDA endpoint; the existing experiment used torch 2.12.1+cu130 on RTX4060. No CUDA package is installed by the CPU requirements.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-extended.txt
.venv/bin/python supplement/extended_validation/search/tests.py
PYTHONPATH=supplement .venv/bin/python -m unittest discover -s supplement/extended_validation/scenarios/tests -v
PYTHONPATH=supplement .venv/bin/python -m unittest discover -s supplement/extended_validation/online_lossy/tests -v
.venv/bin/python -m unittest discover -s supplement/extended_validation/provenance/tests -v
.venv/bin/python supplement/extended_validation/optical/test_workflow.py
```

See each subdirectory's protocol/config and the optical README for measurement commands. Run every campaign in a fresh output directory; preserve failed attempts. The search campaign reproducing the historical fourteen-case suite requires the prior `quality_dataset.zip` under `supplement/results/20261108_v1/raw/`; it is not included in this source-only release. That provenance bundle and complete private results are available through the authors' data-access process. New scenario generators and analytic optical inputs do not require it. Do not confuse passing synthetic/parser fixtures with recovery of an archived Alibaba job. The parser accepts only an explicit local trace path and never downloads a dataset.

Measurement and interpretation: distinguish model predictions, actual CPU/GPU execution, host optimizer time and modeled dynamic energy. Repeated algorithm seeds or timings are not independent application/scenario inputs. Codes implement the stated model adaptations; they do not presume that MQHOA outperforms its controls. Manuscripts, historical archives and raw run logs are not part of this public export.

For generated-scenario campaigns, use `PYTHONPATH=supplement python -m extended_validation.scenarios.campaign pilot --out NEW_OUT`, followed by `run --out NEW_OUT` and `verify --out NEW_OUT`. The analogous module `extended_validation.online_lossy.campaign` runs the lossy/incremental stage. Copy the matching script from `supplement/extended_validation/analysis_templates/` into that new results directory and execute it there to reproduce all tables and figures. Analysis scripts use their own directory as the raw-result root; keep them outside the frozen algorithm directory.


## New official-source trace workload

Thirty newly selected jobs have been evaluated: twenty primary single-entry/single-exit jobs (ten small, ten medium) and a separately declared ten-job large multi-terminal cohort. The originally planned primary large stratum remains empty. IDs, graph hashes and cohort labels are in `TRACE_SELECTED_JOBS_20261109.json`; this metadata contains no task vectors. The original archive10 job mapping remains unrecovered; new-source extraction does not identify it.

The standalone entries below require no private legacy evidence ZIP. Download only under the applicable source-access terms and keep raw data and outputs outside the checkout. The tested official `batch_task.tar.gz` URL is `https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/batch_task.tar.gz`; expected size is 130310549 bytes and SHA-256 is `7c4b32361bd1ec2083647a8f52a6854a03bc125ca5c202652316c499fbf978c6`. The extractor checks the checksum and streams the archive; local aggregation requires approximately 1 GB of disk space. It neither downloads nor commits anything.

```bash
.venv/bin/python supplement/extended_validation/extract_new_trace.py \
  --local /path/outside/repo/batch_task.tar.gz \
  --primary-out /path/outside/repo/new_primary \
  --secondary-out /path/outside/repo/new_secondary
.venv/bin/python supplement/extended_validation/reproduce_trace.py \
  --primary /path/outside/repo/new_primary/instances \
  --secondary /path/outside/repo/new_secondary/secondary_large_instances \
  --out /path/outside/repo/new_results
```

Use fresh paths. Add `--validate-only` to the replay command to check the input/task matrix without running optimization. Published global settings in `trace_parameters_20261109.json` produce exactly the 1,860 frozen task definitions; the unchanged `run_one` kernel performs search. The replay writes complete node-level logs locally. For redistribution, publish only whitelisted scalar outputs and aggregates, keeping raw source, per-task input vectors and complete schedules local. Wall-span durations are trace-derived proxies, edge speedup 3 and communication CCR1 are synthetic assumptions. Primary inference pairs twenty selected jobs, with repeated-topology sensitivity reported separately; it does not treat seeds as independent workloads.

Additional fixtures (no real archive/private legacy files needed):

```bash
.venv/bin/python supplement/extended_validation/run_trace_fixture_tests.py
.venv/bin/python supplement/extended_validation/tests_public_trace_entry.py
.venv/bin/python -m unittest discover -s supplement/extended_validation/provenance/tests -v
```

These add 6 trace, 2 standalone-entry and 14 parser/projection fixtures to the 21 search/scenario/reliability/optics fixtures. Original historical-suite replay still needs its archived input bundle; the new-source standalone entries above do not. Full experimental evidence and the manuscript are supplied separately from this source-only repository.
