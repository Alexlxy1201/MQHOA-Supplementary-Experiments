# Reproduce the additive measured-optics experiment

This package leaves all earlier source, inputs and results unchanged. The frozen physical/task protocol is `PLAN.md` and `config.json`. Profiles are in seconds and actual payload bytes. The new repeated execution phase has a separate prospectively written secondary-control protocol in the result directory; it was added after inspecting the initial executions and is descriptive.

## Dependencies and endpoints

CPU profiling used Python 3.12 / NumPy 2.3.5 on an M1 Pro Mac. GPU profiling used the existing Windows Python 3.12 environment with PyTorch 2.12.1+cu130 and NumPy 2.2.6 on an RTX 4060 Laptop GPU. Analysis used an existing isolated environment with NumPy 2.2.6, SciPy 1.15.3 and Matplotlib 3.10.3. Exact environment records are embedded in the raw profiles; no optical package or raw external data is required.

Configure the existing SSH alias `amst-4060` and `REMOTE_PYTHON`/`REMOTE_ROOT` in `transport.py` for another machine. The authenticated worker serializes trusted NumPy arrays and commands using Python pickle; expose it only through the authenticated SSH process, never a public service. Do not alter active machine networking to reproduce this experiment. Copy `workflow.py`, `config.json` and `transport.py` to the remote root first. Existing old code modules `mqhoa_supp` and `recent_baseline` are imported without modification.

Use a **new empty output directory**; all science stages refuse an existing attempt. Preserve failures and use a new directory for any explicitly justified attempt. Commands below are examples with environment-appropriate `python`/`analysis-python`; they never download raw Alibaba data.

```bash
python supplement/extended_validation/optical/test_workflow.py
python supplement/extended_validation/optical/workflow.py pilot --device cpu --out NEW_OUT/mac_pilot.json
# On the CUDA endpoint, with the same source/config:
python workflow.py pilot --device cuda --out gpu_pilot.json
python workflow.py profile --device cuda --out gpu_profile.json
# Copy remote JSON profiles into NEW_OUT without changing their contents.
python supplement/extended_validation/optical/workflow.py profile --device cpu --out NEW_OUT/mac_profile.json
python supplement/extended_validation/optical/transport.py network --out NEW_OUT/network.json
python supplement/extended_validation/optical/campaign.py build --out NEW_OUT
python supplement/extended_validation/optical/campaign.py quality --out NEW_OUT
python supplement/extended_validation/optical/campaign.py actual --out NEW_OUT
python supplement/extended_validation/optical/verify.py --out NEW_OUT
analysis-python supplement/extended_validation/optical/campaign.py analyze --out NEW_OUT
python supplement/extended_validation/optical/all_local_control.py --out NEW_OUT
python supplement/extended_validation/optical/repeated_execution.py --out NEW_OUT
python supplement/extended_validation/optical/verify_repeated.py --out NEW_OUT
analysis-python supplement/extended_validation/optical/analyze_repeated.py --out NEW_OUT
```

`actual` preserves 12 fresh-process, fixed-seed schedule executions. The matched phase keeps those schedules, runs 15 warmups then 75 interleaved measured executions in a single persistent worker, and includes the identical-worker all-local control. It does not rerun or select a better optimizer seed. Planning and execution cost are separate; their composite is the sum of separately measured phases. Profiling, SSH startup, and endpoint reset are outside execution timing and are explicitly reported. The three designed optical workflows remain the scientific cases; algorithm seeds and timing repeats are conditional replication within a case.

`verify.py` independently checks stored profile arithmetic, bytes, numerical comparisons and all 126 returned schedules and 12 initial execution traces. `verify_repeated.py` independently checks all 75 matched executions and 15 warmups, output vectors, original unchanged schedules and planning costs. Source/config SHA-256 checks connect profiles to the executed optical code. The generated reports and figures are additions to the earlier evidence, with no new claim of global superiority.
