---
date: 2026-05-09
repo: isoster
branch: qa-model-eval-info-2026-05-08
tags:
  - journal
  - exhausted-benchmark
  - qa
  - model-evaluation
---

## Progress

- Ported focused sga_isoster v1.1-style benchmark metrics and flags into `benchmarks/exhausted/analysis`.
- Added contour-gated residual metrics, azimuthal diagnostics, model-evaluation helpers, and profile I/O helpers.
- Wired per-arm QA, cross-arm overlays, and cross-tool comparison plots through the exhausted benchmark fitters and orchestrator.
- Added optional arcsinh surface-brightness profile scaling while keeping `log10` as the default.
- Added an arcsinh `I=0` dashed reference line for calibrated and uncalibrated SB profiles.
- Documented QA/model-evaluation behavior in `docs/04-architecture.md`, `docs/06-qa-functions.md`, and `docs/09-exhausted-benchmark.md`.
- Generated fresh smoke-local asinh demo outputs under `outputs/benchmark_exhausted_asinh_demo`.

## Lessons Learned

- QA plots should use the same profile/model artifacts that feed metrics, not independent plotting-only measurements.
- Cross-arm and cross-tool comparisons need contour-gated residual zones so scoring does not reward fits outside the meaningful galaxy footprint.
- Flag semantics should stay close to sga_isoster unless a local mismatch is explicitly documented.
- Arcsinh SB profiles are useful for low-S/N and negative-intensity regions, but calibrated high-S/N points should remain consistent with log10 magnitudes.
- Visual demos should be run with `skip_existing: false` when validating new plotting behavior.

## Key Issues

- Full-repo Ruff still has unrelated benchmark-script failures; touched-file Ruff passes for the implemented changes.
- AutoProf was available for the smoke-local demo and completed successfully in this session.
- Focused validation passed: `tests/unit/test_comparison_qa.py`, `tests/integration/test_exhausted_smoke.py`, touched-file Ruff, `mkdocs build --strict`, and `git diff --check`.
- Next review should inspect the generated QA PNGs and decide whether `sb_profile_scale: asinh` should remain opt-in or become a benchmark preset.
