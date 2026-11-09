# Isolated FlexDO supplement

The frozen protocol is `../RECENT_BASELINE_PLAN.md`. Existing `mqhoa_supp`, its tests, configuration, 20-seed results and five-comparison analysis are not modified. Requires the original project's NumPy/SciPy/Matplotlib environment.

Run from `supplement/`, with absolute repository and new output paths:

```bash
python -m recent_baseline pilot --repo /path/to/MQHOA --out /new/output/flexdo-v1
# Read pilot_receipt.json: tests, source identity and measured time projection.
python -m recent_baseline quality --repo /path/to/MQHOA --out /new/output/flexdo-v1
# All 70 exact reproduction matches are required before any of the 42 FlexDO runs.
python -m recent_baseline timing --repo /path/to/MQHOA --out /new/output/flexdo-v1
python -m recent_baseline analyze --repo /path/to/MQHOA --out /new/output/flexdo-v1
```

Timing deliberately requires the recorded `DESKTOP-U6ORVRO` Windows host, CPU 2, no competing experiment, one worker and a stable sentinel ratio. It does not alter machine power settings or stop other processes. Each stage refuses a previous completed/partial attempt; preserve and diagnose it rather than silently retrying. The controller monitors process liveness, progress and timeout externally.

Every row carries original-engine and recent-package fingerprints. The original Windows hash includes backslashes in file names: Mac validation reconstructs exactly that archived rule, while Windows requires native equality too. This is documented platform-path normalization, not permission to change frozen source.

`FlexDO-N2` is the only new primary contrast. The final reports label the simulator/order adaptation and separately preserve the old five-comparison statistical family. No GPU, original Alibaba download or new tuning is used. Results and raw inputs remain private; this package does not publish them.
