# Source-readiness amendment, 2026-11-09

This is an engineering readiness amendment before any raw trace was downloaded or any actual trace extraction was run. The archived research inputs and completed supplementary scientific runs remain unchanged.

- Extraction now reports explicit planned, selected and missing counts for all three frozen strata, including zero-coverage cells. `selection_complete` requires ten valid jobs in every stratum. The lexicographic selection rule and filtering criteria are unchanged.
- An explicitly permitted nonofficial local CSV is labeled `unverified-local-csv`, with `official_archive_verified=false`. Fixtures and unverified input are not labeled as a checksum-verified official dataset.
- Job identifiers are encoded safely in output filenames; the original job name remains unchanged in metadata.
- Nine tests cover the official task-name example, invalid graphs, aggregation, all three strata with excess candidates and rejected rows, full directed structural mapping, preserved historical-identity uncertainty, output reuse refusal and rejection of an incorrect official archive hash before any output is created.

The tests use deterministic generated CSV/archive fixtures. They establish parser behavior, not recovery of an actual Alibaba job. Actual raw-source processing still waits for the previously reserved download confirmation or an already-authorized local source file.
