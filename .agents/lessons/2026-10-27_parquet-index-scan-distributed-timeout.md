---
date: 2026-10-27
category: infrastructure
severity: important
---

# Build large immutable Parquet indices before distributed GPU startup

## What happened

After changing to an immutable subset, the new data root had no
`map_style_cache/file_info.pkl`. Rank0 serially opened over100k Parquet footers
on NFS while other ranks waited at the loader's world-group barrier. The
30-minute Gloo timeout expired before optimization started.

## Workaround

Validate actual file metadata using bounded CPU concurrency before launching
torchrun. Write the exact native loader cache format: sorted absolute paths
inside the NEW data root, paired with their actual per-file row counts.
Bind the index hash to the frozen dataset proof and verify exact path coverage.
Do not reuse the parent cache containing old paths or restore obsolete loader
indices after deleting examples. Preserve model/optimizer/global-step resume.

## Validation

The CPU regressions in `tests/local_tests/test_cosmos_qc_resume_20261027.py`
exercise the actual loader functions for rank0 and rank7, with a prebuilt new
cache and no metadata-scanning fallback. Full distributed startup still needs
live verification. Increasing timeout alone does not address the serial I/O
bottleneck, and GPU allocation does not prove optimization has started.
