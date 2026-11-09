# Measured computational-optics supplement — frozen before profiling

## Material Passport

- Mode: code experiment / run. User authorization: execute all supplements, 2026-11-09.
- Status at creation: protocol fixed; no measured data or comparative outcomes observed.
- Original manuscript, inputs, code and all old runs remain unchanged.

## Workload and physical meaning

Execute a scalar angular-spectrum, multi-plane beam-propagation and focus-assessment workflow. A sampled circular Gaussian aperture with a deterministic phase aberration is analytically generated; it is not a camera/laboratory acquisition. Compute a zero-padded Fourier spectrum, then branch into 3/5/8 propagation distances. Each branch multiplies by the Rayleigh–Sommerfeld transfer function, takes the inverse FFT, crops the field, computes intensity and reports integrated intensity, peak intensity, encircled energy and a spatial second moment. A final task collects all plane scores and identifies the smallest second moment. Actual operators create actual arrays and edges; this is not a hand-weighted proxy graph.

The transfer function is H(kx,ky)=exp(i*z*sqrt(k²-kx²-ky²)), Im(sqrt)>=0, with k=2π/λ. Numerical parameters and repeat counts are in config.json. Use complex128/float64, a 2× padded FFT grid, wavelengths 532 nm, pitch 8 µm, distances 3,6,…,24 mm. These are coherent scalar computational optics instances, not a demonstration of vector Maxwell accuracy or optical hardware acquisition. Plane count and grid size are fixed before timing.

Primary method sources inspected: ETH Zurich *Angular Spectrum Representation*, equations 7.5–7.7, https://ethz.ch/content/dam/ethz/special-interest/itet/photonics-dam/documents/lectures/EandM/AngularSpectrumRepresentation.pdf ; TorchOptics author documentation, https://torchoptics.readthedocs.io/en/latest/user-guide/propagation.html (ASM formula and padding). This code independently implements the formula; it does not claim to reproduce the TorchOptics package.

## Measurement and tests

Mac CPU: NumPy FFT and array operators; edge endpoint: available 4060 CUDA PyTorch, with synchronization around each timed task. Inspect occupancy first and do not disturb other jobs. Try the 3060 existing SSH route, record failure if unavailable; do not modify network/security state. Warm each operator twice and record all seven timed repetitions; use median durations. Build the DAG from executed task dependencies and record bytes actually output by each task. Include per-task CPU/GPU full-array numerical comparison on the GPU host plus cross-host sampled output and final complete metric-vector checks. Test z=0 identity, plane-wave phase, finite outputs, graph/output dependency consistency and independent schedule validity. Require full-array relative L2 and absolute errors within 1e-8 where relevant.

Measure actual array-payload transfer over the existing Mac↔4060 SSH route using persistent binary framing. For each distinct edge byte size, record three acknowledged uploads and three downloads and validate content SHA-256. Both direction wall times include protocol/SSH costs; use their arithmetic-average median as the shared symmetric edge-delay input. This is a deliberately documented symmetric-delay approximation from real transport measurements; it is not a direct one-way-clock measurement and not a LAN-only claim. Record route, raw times and any failure. No unmeasured delay will be called measured.

Run a small pilot first and report its measured projected cost. On the frozen measured profiles compare unchanged MQHOA/PSO settings and existing HEFT and adapted FlexDO-N². Twenty stochastic seeds, 10,000 actual fitness calls, fixed pool seeds from the existing rule, all entry/final tasks pinned to Mac. HEFT/FlexDO are deterministic and evaluated once per workload. Independently check every returned schedule. No tuning on these workloads. Treat three workflows as three designed cases, not 60 independent real-world applications; seed intervals are conditional on a fixed workload/profile. Descriptive per-case contrasts only; no population-general significance claim from three cases.

After model comparisons, attempt a persistent two-worker execution of seed-0 schedules for each of the four methods, measuring observed end-to-end time and comparing predicted values. All executable outputs must match all-local reference. Preserve actual queue/transfer and CPU/GPU overhead. If end-to-end execution cannot be validated, explicitly limit this stage to a measured-profile scheduling experiment rather than presenting model predictions as observed execution speedups.

## Deliverables

New source/tests/config, all timing repetitions, environment/availability records, task/output hashes, complete final metric vectors, measured-byte transport records, immutable profile-driven instances, per-seed solution/trace logs, analysis CSVs and PDF/PNG figures, bilingual supplementary section and reviewer addition, independent verification receipt and SHA-256 manifest. Never overwrite a different result or erase a failed attempt. No Git commit from this worker; parent reviews and publishes only authorized artifacts.
