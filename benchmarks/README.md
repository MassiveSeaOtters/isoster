# Benchmarks

This folder contains performance and comparison benchmarks for isoster.

## Scope

- `performance/`: runtime and throughput comparisons (including method-vs-method).
- `timing/`: frozen four-stage Isoster/photutils/AutoProf timing campaign; the
  completed Stage 4 record is in
  `docs/publication/three-way-timing-benchmark-reference.md`.
- `profiling/`: hotspot and profiler-oriented diagnostics.
- `baselines/`: threshold configs and CI regression gate scripts.
- `exhausted/`: 39-configuration exhaustive sweep on any galaxy image.
- `benchmark_baseline/`: baseline fits on real galaxies with isoster/photutils/autoprof comparison.
- `robustness/`: sensitivity of the fit to initial `sma0` and isophotal geometry (`eps`, `pa`, `x0`, `y0`).
- `utils/`: shared benchmark helpers and adapters.

## Goals

- Quantify speed and scaling behavior.
- Compare isoster against reference methods under controlled scenarios.
- Generate reproducible machine-readable outputs (JSON/CSV) and optional figures.

---

## Script Census

| Script | Category | Motivation | Input Data | Output Artifacts | Status |
|--------|----------|-----------|-----------|-----------------|--------|
| `performance/bench_vs_photutils.py` | Performance | Speed + accuracy vs photutils | Synthetic Sérsic | CSV / JSON / figures | Active |
| `performance/bench_vs_autoprof.py` | Performance | Speed + accuracy vs AutoProf | IC3370, eso243-49, ngc3610 | Reports, figures | Active |
| `performance/bench_efficiency.py` | Performance | Standardized efficiency gate | Synthetic Sérsic | JSON timing | Active |
| `performance/bench_numba_speedup.py` | Performance | Numba JIT impact | Synthetic Sérsic | JSON timing | Active |
| `exhausted/run_benchmark.py` | Config sweep | 39-config exhaustive sweep on any galaxy image | Any 2D FITS (default: IC3370_mock2) | Per-config JSON + figures | Active |
| `baselines/run_benchmark_gate.py` | CI gate | Regression gate (efficiency + QA) | Synthetic + Huang2013 | Pass/fail + artifacts | Active |
| `baselines/collect_phase4_profile_baseline.py` | Baseline | Collect profile residual metrics for 3 Sérsic scenarios | Synthetic Sérsic | JSON | Active |
| `baselines/lock_phase4_thresholds.py` | Baseline | Lock profile thresholds from baseline metrics | JSON from above | `phase4_profile_thresholds.json` | Active |
| `baselines/lock_efficiency_thresholds.py` | Baseline | Lock efficiency thresholds from bench_efficiency output | JSON from bench_efficiency | `efficiency_thresholds.json` | Active |
| `profiling/profile_hotpaths.py` | Profiling | cProfile hotspot analysis | Synthetic Sérsic | `.prof` + JSON | Active |
| `profiling/profile_isophote.py` | Profiling | Per-isophote profiling | Synthetic Sérsic | `.prof` + JSON | Active |
| `profiling/profile_numba_flagged_cases.py` | Profiling | Numba flagged-case timing | Synthetic Sérsic | JSON timing | Active |
| `utils/mockgal_adapter.py` | Utility | External mockgal adapter (libprofit) | Presets / config | Mock FITS images | Active |
| `utils/scaffold_models_config_batch_templates.py` | Utility | Scaffold batch template files | None | Template YAML/JSON | Active |
| `benchmark_baseline/run_baseline.py` | Baseline | Multi-method baseline (isoster/photutils/autoprof) on real galaxies | `data/` FITS | profiles, models, fit_configs.json, comparison QA | Active |
| `robustness/run_sweep.py` | Robustness | 1-D perturbation sweep on initial `sma0` + geometry; characterizes fit capture radius | Mocks + Huang2013 (IC2597 mocks) + eso243-49 + ngc3610 + HSC edge cases (tiered) | `results.json`, `REPORT.md` | Active (all 4 tiers wired) |

See `FRAMEWORK.md` for guidance on adding new benchmarks.

---

## Threshold Config Files (baselines/)

These JSON files are committed to the repo as they define the acceptance criteria for the gate:

| File | Used By | Notes |
|------|---------|-------|
| `baselines/efficiency_thresholds.json` | `run_benchmark_gate.py`, `lock_efficiency_thresholds.py` | Full 9-case efficiency thresholds (active) |
| `baselines/efficiency_thresholds_quick.json` | `run_benchmark_gate.py --quick` | 3-case subset for fast CI checks |
| `baselines/phase4_profile_thresholds.json` | `run_benchmark_gate.py`, `lock_phase4_thresholds.py` | 1-D profile residual thresholds (active) |
| `baselines/run_benchmark_gate_defaults.json` | `run_benchmark_gate.py` | Default QA/timing parameters for the gate |

Threshold files are regenerated with `lock_efficiency_thresholds.py` and `lock_phase4_thresholds.py`
after a significant code change or hardware upgrade. See the full workflow below.

---

## Reproducible Commands

```bash
# Efficiency benchmark
uv run python benchmarks/performance/bench_efficiency.py

# Numba speedup benchmark
uv run python benchmarks/performance/bench_numba_speedup.py

# Numba diagnostic sweep (more stable timing + scaled workload)
uv run python benchmarks/performance/bench_numba_speedup.py --n-runs 4 --scale-factor 1.0
uv run python benchmarks/performance/bench_numba_speedup.py --n-runs 3 --scale-factor 2.0 --output outputs/benchmark_performance/bench_numba_speedup_scale2

# Flagged-case numba profiling drills (defaults target scale1+scale2 JSON outputs)
uv run python benchmarks/profiling/profile_numba_flagged_cases.py --timing-runs 3 --profile-runs 1

# Method comparison benchmark: isoster vs photutils (quick mode)
uv run python benchmarks/performance/bench_vs_photutils.py --quick

# Method comparison benchmark: isoster vs AutoProf (quick mode — IC3370 only)
uv run python benchmarks/performance/bench_vs_autoprof.py --quick --plots

# AutoProf full benchmark (all galaxies)
uv run python benchmarks/performance/bench_vs_autoprof.py --plots

# AutoProf benchmark — isoster only (skip AutoProf, replot from cache)
uv run python benchmarks/performance/bench_vs_autoprof.py --skip-autoprof --plots

# AutoProf benchmark — specific galaxies
uv run python benchmarks/performance/bench_vs_autoprof.py --galaxies IC3370_mock2 ngc3610 --plots

# Profiling artifacts (.prof + summaries)
uv run python benchmarks/profiling/profile_hotpaths.py --multi 1 --top-n 20
uv run python benchmarks/profiling/profile_isophote.py --repetitions 1 --top-n 20

# Method comparison with explicit output root
uv run python benchmarks/performance/bench_vs_photutils.py --quick --output outputs/benchmark_performance/manual_run

# Collect profile baseline metrics
uv run python benchmarks/baselines/collect_phase4_profile_baseline.py

# Lock profile thresholds from baseline metrics
uv run python benchmarks/baselines/lock_phase4_thresholds.py

# Lock efficiency thresholds from measured efficiency baselines
uv run python benchmarks/baselines/lock_efficiency_thresholds.py

# Run baseline-locked benchmark gate (quick: efficiency + profile QA)
# Defaults loaded from benchmarks/baselines/run_benchmark_gate_defaults.json
uv run python benchmarks/baselines/run_benchmark_gate.py --quick --n-runs 1 --efficiency-lock benchmarks/baselines/efficiency_thresholds_quick.json

# Override defaults file (optional)
uv run python benchmarks/baselines/run_benchmark_gate.py --quick --n-runs 1 --gate-defaults benchmarks/baselines/run_benchmark_gate_defaults.json --efficiency-lock benchmarks/baselines/efficiency_thresholds_quick.json

# Full threshold-refresh + full gate workflow (run before large batch campaigns)
uv run python benchmarks/performance/bench_efficiency.py --n-runs 3 --output outputs/benchmark_performance/bench_efficiency_refresh
uv run python benchmarks/baselines/lock_efficiency_thresholds.py --input outputs/benchmark_performance/bench_efficiency_refresh/efficiency_benchmark_results.json --output benchmarks/baselines/efficiency_thresholds.json
uv run python benchmarks/baselines/collect_phase4_profile_baseline.py --output outputs/tests_integration/baseline_metrics_refresh
uv run python benchmarks/baselines/lock_phase4_thresholds.py --input outputs/tests_integration/baseline_metrics_refresh/phase4_profile_baseline_metrics.json --output benchmarks/baselines/phase4_profile_thresholds.json
# Run this full gate command sequentially (do not run concurrent benchmark commands).
uv run python benchmarks/baselines/run_benchmark_gate.py --n-runs 3 --efficiency-lock benchmarks/baselines/efficiency_thresholds.json --profile-lock benchmarks/baselines/phase4_profile_thresholds.json --require-all-locked-cases

# Gate with Huang2013 basic-QA artifacts
uv run python benchmarks/baselines/run_benchmark_gate.py --quick --n-runs 1 --efficiency-lock benchmarks/baselines/efficiency_thresholds_quick.json --qa-output-subdir qa_figures --qa-overlay-step 10 --qa-dpi 180

# Optional external mockgal adapter (safe dry-run)
uv run python benchmarks/utils/mockgal_adapter.py --dry-run

# List and use science-ready mockgal presets
uv run python benchmarks/utils/mockgal_adapter.py --list-presets
uv run python benchmarks/utils/mockgal_adapter.py --preset single_psf_noise_sblimit --dry-run
uv run python benchmarks/utils/mockgal_adapter.py --preset models_config_batch --dry-run
uv run python benchmarks/utils/mockgal_adapter.py --preset models_config_batch --preset-value models_file=my_models.yaml --preset-value config_file=my_config.yaml --preset-value workers=2 --dry-run

# Baseline benchmark (all galaxies)
uv run python benchmarks/benchmark_baseline/run_baseline.py

# Baseline benchmark (quick smoke test — IC3370 only)
uv run python benchmarks/benchmark_baseline/run_baseline.py --quick

# Baseline benchmark (single galaxy)
uv run python benchmarks/benchmark_baseline/run_baseline.py --galaxy eso243-49

# Scaffold reusable models_config_batch template files into a new outputs/ run directory
uv run python benchmarks/utils/scaffold_models_config_batch_templates.py

# Robustness benchmark — sub-minute smoke test (mocks tier, bare arm, 3 sma0 factors)
uv run python benchmarks/robustness/run_sweep.py --quick

# Robustness benchmark — full 1-D sweep on the mocks tier
uv run python benchmarks/robustness/run_sweep.py --tiers mocks

# Robustness benchmark — full 1-D sweep on the huang2013 tier
# (IC2597 libprofit mocks; set HUANG2013_DATA_ROOT to override the data path)
uv run python benchmarks/robustness/run_sweep.py --tiers huang2013

# Robustness benchmark — full 1-D sweep on the hsc edge-case tier
uv run python benchmarks/robustness/run_sweep.py --tiers hsc
```

---

## Output Policy

Benchmark outputs go under `outputs/` with source-specific folder names.
Override the output root with `ISOSTER_OUTPUT_ROOT`.

Naming rule: use **singular** prefix — `benchmark_`, not `benchmarks_`.

Recommended naming:
- `outputs/benchmark_performance/<run_id>/...`
- `outputs/benchmark_profiling/<run_id>/...`

Performance and profiling scripts should emit:
- machine-readable summaries (`.json`, `.csv`)
- plot artifacts (`.png`/`.pdf`) for summary or failing/borderline cases
- run metadata (git SHA, package versions, platform info)
- profiler artifacts (`.prof`) for profiling scripts

See `FRAMEWORK.md` for complete naming and content requirements.
