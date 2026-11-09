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
