# Recorded campaign validation

The five-task supplementary campaign completed on 2026-11-08. The selected matrix contains 160 tuning, 3,374 quality and 1,260 dedicated timing runs. Core implementation tests passed 14/14, including 300 reference-evaluator comparisons. Quality solutions and budgets were independently checked.

Final timing uses one logical CPU (CPU 2) of the 3060 host, with 60 seconds of warmup and seeded interleaving. All 14 fixed-input stability groups passed: maximum/minimum median time 1.089, below the 1.5 limit. Earlier 4060 timing and a failed stability recheck were retained separately.

This public repository contains the new source and reproduction procedures. Full report/data archives are retained by the authors in their existing project repository. The standalone synthetic configuration remains usable without the optional original archive.
