# Frozen online packet-loss and incremental-subtask experiment

## Material Passport

- User authorized all supplementary experiments; this separate campaign fills packet retransmissions and incremental tasks in an already running user's DAG.
- The completed 1,764-row scenarios-v3 campaign and its six-test inference family are sealed unchanged. This sibling package copies its frozen model/methods and extends only the new copy.
- Six independent generated scenarios, two or three users/servers and six or eight tasks per user; 20 seeds each for MQHOA/PSO, one deterministic HEFT-List/FIFO-Local row per scenario: 252 formal rows.

## Incremental information and execution

Each user's first n−2 nodes arrive at t=3u; its last two nodes and their incident edges are revealed five seconds later while earlier work can be executing. Newly revealed nodes can depend on completed or running nodes of that same application. The planner sees only the currently released nodes and edges and a partial per-user task list, never the unreleased node list or eventual DAG size. Outputs of finished tasks remain available on the resource where they executed. Sources and the eventual sink are pinned to the user's own device, and another user's local device is never admissible. Already-started compute or input transfers freeze placement.

The same compute-rate events (server 0 at t=4 and t=12) and link-rate events (t=7 and t=15) as v3 act on ongoing work. A fixed public ten-decision-slot allocation uses 100 calls per observed decision, independent of the actual future event list: two-user scenarios have eight epochs and use 800 calls; three-user scenarios have ten and use 1,000. Calls are not redistributed from unobserved future epochs. Completion and retry events dispatch the latest plan without extra optimization calls. All methods share the same simulator and current-rate forecast. Actual planning wall time is recorded separately and excluded from virtual time.

## Loss and retransmission

Every cross-resource DAG edge is one payload packet. Each completed full transmission attempt independently fails with probability 0.15. A reproducible uniform draw is keyed by scenario seed, directed semantic edge and attempt index, not by algorithm or run order; these common exogenous outcomes pair the methods whenever they attempt the same edge. Different algorithms can induce different transfer sets. Failed attempts consume their full actual airtime, shared-link capacity and both endpoints' dynamic energy, but deliver no payload. Stop-and-wait retransmission begins immediately and repeats until success, with task placement frozen across every attempt. The link rate may change mid-attempt. All attempted and failed transmissions, random uniform draws, attempt work, useful payload, airtime and energy are archived. There is no deadline-based dropping or hidden censoring.

The planner knows the declared current loss probability but receives neither future draws nor the loss seed. It uses expected remaining transmission work c/(1−p), with the corresponding current-rate energy, as a forecast. This expectation is not substituted for actual realized execution outcomes. CPU dynamic energy, network power, ownership, candidates, MQHOA/PSO parameters and heuristic adaptations follow v3. Losses apply to the simulated edge packets, not a calibrated real wireless packet-size/process model.

## Validation, analysis and stopping

Tests must cover: failure delivers no data; failed airtime and energy count; deterministic common draws; link-rate integration across a retry; incremental visibility before/after reveal; unchanged first decisions if future tasks/loss draws change; immutable started placements; actual per-condition evaluation budgets. Full row checking verifies every successful edge delivers one payload, attempt-work equals payload-work times attempts, failed attempts never count as delivered, and all shared-link/compute calendars, releases, precedence, energy, work and hashes are consistent.

Primary endpoint is mean full-application response time, measured from each user's initial release to completion of its eventual full DAG. Compare MQHOA with PSO and HEFT-List using six paired scenario means; 10,000-resample bootstrap intervals and exact sign flips, Holm over **these two new contrasts only**. Previous six contrasts are not silently expanded or overwritten. All other outcomes are descriptive. Conditional seed intervals do not turn algorithm repetitions into independent scenarios. No claim of optimality, real-time deployment or measured electrical savings is made. Stop on a test/validator failure or the predeclared one-hour hard timeout; preserve every attempt.

## Sources

The ARQ mechanism is a declared independent Bernoulli stop-and-wait simulator, not an empirical channel model. CPU/network energy formulas follow the same sources as v3: https://arxiv.org/abs/1605.05488 and https://doi.org/10.1186/s13677-022-00342-1. No new hardware parameter is taken from either work.
