# FlexDO supplementary baseline — frozen implementation and execution plan

## Material Passport

- Mode: code experiment / run. Authorization: user “执行”, 2026-11-08.
- Existing evidence: `results/20261108_v1/`; preserve every existing input, result, code module and frozen configuration.
- Primary source: de Queiroz, de Rezende and Barbosa, *A flexible algorithm to offload DAG applications for edge computing*, JNCA 222 (2024), 103791, DOI https://doi.org/10.1016/j.jnca.2023.103791; public preprint https://arxiv.org/pdf/2306.09458 (2023), Algorithm 1, Sections 3.1/4.2/6.
- This plan reconstructs the user-provided plan summary; the other computer's uncommitted `RECENT_BASELINE_PLAN.md` was not available locally. Any later recovered version is retained separately, never silently substituted.

## Question and boundaries

Compare a recent published DAG-offloading method with existing MQHOA under the same pinned-terminal, two-serial-resource, fixed-edge-delay makespan model. FlexDO-N² is fixed as the primary comparator; FlexDO-0 and FlexDO-N are descriptive variants, not alternatives selected after results. No tuning stage. The paper's CPU/bandwidth-sharing simulator is not reproduced: this is an explicitly documented adaptation of its candidate generation to our model.

## Fixed implementation choices

1. Let `q` be the number of offloadable tasks (total nodes minus pinned terminals), as N denotes offloadable tasks in the source paper. Phase-II bit targets are 0, `floor(log2(q))`, `floor(2*log2(q))`, clipped to `[0,q]`. At total N=10/50/100, N² has at most 64/2048/8192 phase-II candidates. These are upper bounds, not promises that every graph generates exactly that many.
2. Anchors are edges incident to a pinned terminal. For each non-anchor edge, subtract each incident anchor's original delay once. Negative adjusted scores are retained. Gain fallback is local duration minus edge duration minus all original incident communication delays (paper Eq. 6).
3. Start from all-local. Prefer edges with exactly one marked endpoint; otherwise select an edge with two unmarked endpoints. Use descending adjusted score and ascending `(source,target)` ties; discard exhausted edges. Mark both endpoints even if this overshoots the phase-I target by one. Fallback ranks unmarked tasks by descending gain, ascending task index. These deterministic choices are frozen before evaluation.
4. Retain every distinct phase-I placement and every phase-II placement in insertion order, as Algorithm 1's decision-set union does. Enumerate unmarked-task bit masks in ascending integer order. Deduplicate placements before calling the evaluator; separately record attempted/duplicate candidates. Keep the first minimum on a fitness tie.
5. For each placement, compute its own upward critical-path rank using placed duration and cross-resource edge delays. Select the highest-rank task from the currently ready set, breaking ties by ascending task index. This guarantees a topological order, including zero-duration ties. Evaluate with the unchanged common decoder, and independently validate the selected schedule. This order rule is an adaptation, not an equivalence claim about the paper's event simulator. FlexDO builds no indexed candidate pools.

All actual makespan calls, including all-local and retained greedy candidates, count toward a 10,000-call cap. Do not pad deterministic runs to 10,000 calls. Record actual calls and full time, including ranking/candidate generation/validation. Preserve an append-only best-so-far trace indexed by calls; no proven-optimum claims.

## Files and safeguards

Add a standalone `supplement/recent_baseline/` package, its own tests and JSON config. Import, but do not edit, `mqhoa_supp` or the existing tests/configuration. This leaves the old package source hash intact. The new package has a separate code hash, protocol hash and result directory. Old source/provenance gates remain active. New code and results are committed only to the private supplementary branch; public release of the reusable code can be handled separately from private evidence.

Six tests cover: anchor/gain arithmetic; retention of intermediate greedy placements; overshoot and distinct enumeration; placement-specific ready ordering and zero-duration precedence; deterministic output and budget enforcement; old-task/frozen-source invariance and meaningful integration against the existing decoder. All must pass before pilots.

## Execution sequence and stop gates

1. Commit this plan before formal runs; test on Mac.
2. On the original 3060 Windows host, use the recorded Python 3.11.9 / NumPy 2.2.6 environment and logical CPU 2. Check for competing work. Freeze source/config/input hashes and run bounded pilots; report a measured estimate before the full stages.
3. Reproduce MQHOA full at seeds 0–4 on all 14 instances (70 runs) using the original task definitions, pool seeds and parameters. Require exact makespan equality and identical input/pool hashes to archived rows. If any differ, stop; diagnose before continuing. These duplicates are not additional independent scientific samples.
4. Execute 3 deterministic variants × 14 instances = 42 quality runs. Total new quality/reproduction rows: 112. Require zero schedule violations and every actual call count ≤10,000.
5. Dedicated timing: per instance, 10 repetitions of each of the 3 deterministic variants plus MQHOA seeds 0–4 = 35; 14 × 35 = 490. Fixed CPU 2, single worker, 60-second warmup, seeded interleaving and archived-style stability sentinels (maximum/minimum group median ≤1.5). Preserve failed attempts; no automatic scientific rerun. Compare new paired timings directly; archived timings are secondary context only after host/environment boundaries are matched.

## Analysis and deliverables

- Report the 70-row reproduction gate; 42 deterministic results; per-instance means/95% CIs from the existing 20 MQHOA seeds; MQHOA win/tie/loss counts against FlexDO. Deterministic repeats are not independent quality samples.
- Primary inference uses the existing 12 independent synthetic DAGs and relative differences of instance means; archived graphs remain descriptive. Recompute the original five **raw** exact sign-flip p-values and require equality to archived values. Add the pre-fixed FlexDO-N² comparison, then recompute Holm over all six. Adjusted values may change. Preserve the original five-comparison table as an earlier analysis, and clearly label the expanded family. Do not insert new RNG consumption into the old bootstrap stream: reuse archived CIs for old rows and use a separate seeded stream for the new contrast. Descriptive variants are not additional uncorrected claims.
- Timing, 95% intervals, candidate counts and best-so-far curves go into a new results directory, together with raw JSONL, provenance, environment, manifests and an executable reproduction command. Preserve all raw attempts and previous reports.
- Update supplementary result documents / reviewer response for R2.9 based on actual outcomes and state the simulator/order adaptations. Source code and tests accompany the new evidence; the original paper/data are not rewritten or deleted. Manuscript integration can reference a separately saved addition rather than silently rewriting prior tables.

## Completion criterion

Tests and reproduction gate pass; 112 new quality/reproduction rows and 490 valid same-host timing rows are present; independent validators pass; original raw p-values reproduce; expanded-family statistics and figures are generated and checked; private branch contains code, frozen plan, evidence and bilingual result/reviewer additions. A launch, an estimate or an unexecuted plan is not completion.
