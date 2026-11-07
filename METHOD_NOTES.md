# Reading the supplementary results

The supplementary tables describe the new, shared-budget protocol. The original manuscript tables remain in their original files.

## Comparisons and uncertainty

The primary unit is an independently generated DAG. Each stochastic condition has 20 seeds within that instance and shares its fixed candidate pools with the other indexed methods. The 12 synthetic instances enter the paired cross-instance tests; the two archived instances have separate descriptive summaries. A single full-MQHOA run set serves the comparison, sensitivity and ablation analyses.

Relative differences use `(mean MQHOA − mean comparator) / mean comparator` within each instance, followed by an equal-weight average over instances. The comparison and ablation families have separately predeclared Holm corrections. Bootstrap confidence intervals describe seed variation for individual instances and instance variation for cross-instance summaries.

## Fixed-scale control

`NoHamming` disables the Hamming-triggered scale update and fixes sigma at M/2. The Hamming statistic remains logged in both variants. Its value never drives the fixed-scale variant's sampling or centre updates. Runtime therefore includes the same diagnostic distance calculation, and the ablation measures the effect of the adaptive rule on search.

## Direct-generation control

`DirectRandom` generates fresh placements and level-respecting orders. It changes the representation, the accessible candidate set and the sampling strategy. `RandomPool` is a complementary uniform-search control over exactly the same candidate pools as MQHOA and PSO. These contrasts address complete search strategies and the adaptive sampler, respectively.

## Timing boundary

Per-run solver timing starts after the common instance deserialisation and adjacency construction. It includes candidate-pool generation and hashing where applicable, the search or heuristic, fitness evaluation, and final schedule validation. Python startup, common input loading and log-file writing are outside this boundary. `t_pool + t_eval + t_search = t_total`, up to floating-point rounding.

Dedicated timing comparisons use a complete, single-host matrix with one Python worker. Quality data come from the 3060 computer; its parallel-batch times are retained in raw logs for provenance and are excluded from the dedicated timing summaries.

The initial 4060 timing block showed a speed transition on the first archived instance. A complete 90-condition warmed recheck failed its predeclared stability limit (ratio 2.034 versus limit 1.5). Both attempts are retained separately. The replacement protocol measures the entire timing matrix on the 3060 computer, fixes only the new Python process to logical CPU 2, warms it for 60 seconds, interleaves conditions with seed 20261108, and checks a fixed-input timing sentinel throughout. Final timing selection requires the complete matrix and a maximum/minimum sentinel ratio at most 1.5; individual fast rows are never selected across attempts.

## Scope

The workload set contains archived DAG constants and generic, seeded layered DAGs. Report best observed makespans under the recorded budgets. Exact optimality and measured optical workload performance require their respective evidence.
