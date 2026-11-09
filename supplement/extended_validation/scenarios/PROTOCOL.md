# Frozen additive scenario protocol

## Material Passport

- Mode: code experiment / simulation; authorized by user “执行所有补充”, 2026-11-09.
- Scope: new multi-user/multi-server static, genuinely online event-driven, and delay/energy experiments. No old source, result, manuscript or data is modified.
- Freeze: this protocol and config.json are written before tests, pilots or formal results. Subsequent changes must be disclosed in a new attempt.

## Experimental units and simulator

Six independently seeded scenarios have (users, shared servers, tasks per user) equal to (2,2,6), (2,3,6), (3,2,6), (3,3,6), (2,2,8), (3,3,8). Each user has its own local device and one fork/join DAG generated from its scenario RNG; task work is 3–9 billion cycles, edge payload work 0.2–0.8 baseline-link seconds, source/sink stay on their owner's device. A task can use its own local device or any shared server, never another user's local device. Local frequencies are 1+0.1*user GHz, server frequencies 2+0.5*server GHz. Each compute resource and the single shared half-duplex network execute at most one operation at a time. Communication can overlap computation. Inputs are transferred only after all parents complete; each cross-resource edge is transferred once, with no cache/broadcast assumption. Placement is frozen once a task starts an input transfer or computation. Started operations are never migrated, restarted or cancelled.

The event simulator advances remaining compute work and transfer work at the currently active rates. Running tasks and transfers therefore respond to rate changes mid-operation. Started/finished tasks are frozen across re-planning. Planning uses a non-insertion list-schedule forecast with current rates held constant; actual event-dispatch outcomes are measured separately. This is a new generalized adaptation of the algorithms, not an identity with the old two-resource decoder or original HEFT insertion algorithm.

## Three predeclared groups

1. **Static shared system:** all users arrive at t=0; rates constant. Primary endpoint: makespan.
2. **Online dynamic system:** user u arrives at t=3u; server 0 multiplier changes to 0.5 at t=4 and 1.25 at t=12; link multiplier changes to 0.5 at t=7 and 1.4 at t=15. The simulator stores these future events, but the optimizer receives only currently released DAGs, current rates, completed operations and remaining work. It re-plans on arrivals and rate changes, with no future-event or unreleased-job access. Completions dispatch according to the latest plan without spending new evaluations. Primary endpoint: mean per-user response time; makespan, dynamic energy, and completion fairness are secondary.
3. **Delay–energy trade-off:** static shared system at fixed weights 0, 0.1, 0.3, 1, 3. Objective is residual predicted makespan divided by current visible all-local serial work plus weight times incremental predicted dynamic energy divided by visible all-local dynamic energy. Both denominators use only visible information and are fixed within a planning epoch. Report observed nondominated solutions, not proof of a globally optimal Pareto front.

## Energy model and sources

Compute dynamic power is P=κf³ and incremental energy is its time integral (equivalently κ*C*f² at constant f). Transfer energy is network dynamic power times occupied link time. κ=5e-28 SI and aggregate network dynamic power=2W are illustrative simulation parameters, not hardware measurements. All local devices and servers are included; idle/baseline power and optimization CPU energy are excluded. Frequency multipliers change dynamic power as well as processing rate. This experiment does not validate battery lifetime or measured electrical savings.

Primary sources for the formulas and online-information distinction: Mao, Zhang and Letaief, *Dynamic Computation Offloading for Mobile-Edge Computing with Energy Harvesting Devices*, https://arxiv.org/abs/1605.05488; Liu and Liu, *Energy-efficient allocation for multiple tasks in mobile edge computing* (2022), https://doi.org/10.1186/s13677-022-00342-1. Their algorithms and energy-harvesting systems are not reproduced. HEFT reference: Topcuoglu et al. (2002), DOI https://doi.org/10.1109/71.993206. No primary-source numerical parameters are claimed.

## Methods and budgets

- MQHOA adaptation and PSO adaptation use the same 512-entry placement pool and 512-entry level-consistent order pool in each observed problem. Multi-valued placements replace binary placements; original MQHOA k=4,m=5,lambda=1.3 and declared PSO settings are frozen without tuning.
- HEFT-List uses upward ranks and greedy earliest finish / incremental-energy choices in the same forecast. It is explicitly a shared-link list adaptation, not the old insertion HEFT implementation. FIFO-Local is an arrival/task-ID priority baseline with all computations on the user's local device.
- MQHOA and PSO each use exactly 1,000 forecast evaluations per run, including initialization and every re-planning epoch. The budget is evenly allocated to the maximum possible arrival/rate decision epochs, with integer remainder assigned to early epochs. A final no-pending epoch consumes no calls; unused calls remain unused and are reported. Deterministic heuristics use one forecast per epoch and are not padded to 1,000.
- Stochastic methods have seeds 0–19; deterministic methods run once for quality. Same scenario inputs and observed environment events are used for every method. Algorithm repetitions are not independent scenario samples.

## Tests, pilot, execution and analysis

Meaningful tests cover shared-link serialization and compute overlap, ownership/pinning, release/precedence, mid-task rate changes, energy arithmetic, future-information isolation, started-placement freezing, and MQHOA/PSO exact budget accounting. A pilot measures runtime before full execution. All raw attempts are retained. Formal rows record complete task and transfer schedules, event/rate segments, observations and evaluations. Every final schedule is checked by an independent constraint/energy checker. Tests or validators failing stop the campaign.

Means and 95% bootstrap intervals are reported per scenario over 20 algorithm seeds (conditional algorithm variation). Cross-scenario effects use the six scenario means as independent units, exact paired sign flips and Holm correction across the two group-specific MQHOA comparisons (PSO, HEFT-List); three groups give six primary tests. Energy-weight fronts are descriptive, with no selected-best-weight significance claim. Report every condition, failures and adverse outcomes. Existing experiments/statistics are kept intact.
