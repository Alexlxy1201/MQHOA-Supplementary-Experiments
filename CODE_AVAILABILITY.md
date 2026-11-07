# Public supplementary code

Public repository: <https://github.com/Alexlxy1201/MQHOA-Supplementary-Experiments>

The repository was verified accessible without authentication (HTTP 200). It contains only the new supplementary implementation, tests, configurations and documentation under the package's MIT license. The manuscript, reviewer comments and original archives remain in the existing private repository.

The implementation matches the source tested remotely on the 3060 machine:

- Private source commit: `344dd9b3dd0f818bb90c5a1f88c0bfdd5ae40522`.
- Windows source-receipt hash: `d2f3ac61e4dee308eb669c0502e5f50e0cc85603b4e19f88f8faee28899341d4`.
- Python 3.11.9, NumPy 2.2.6, SciPy 1.15.3, Matplotlib 3.10.3.
- New-code tests: 14 passed, including agreement with the separate reference evaluator on 300 generated solutions.

The receipt hash includes relative filenames with the operating system's path separators; cross-platform byte comparisons should normalise that convention or compare individual file hashes. Both experiment hosts use Windows.

Public code availability and the unit tests do not stand in for the formal experiment results, which are stored and reported separately.
