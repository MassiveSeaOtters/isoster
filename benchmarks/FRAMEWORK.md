# Benchmark Framework

Rules and conventions for adding new benchmarks to isoster.

---

## 1. Naming

**Script location**: `benchmarks/<category>/<verb>_<subject>.py`

Valid categories:

| Category | Purpose |
|----------|---------|
| `performance/` | Speed, throughput, method comparisons |
| `profiling/` | Hotspot analysis, `cProfile`, flame graphs |
| `baselines/` | Threshold locking, CI regression gates |
| `exhausted/` | 39-config sweep on any galaxy image |
| New subdirectory | Add when a new class of benchmark warrants its own folder |

**Output folder prefix**: always singular.
- Correct: `outputs/benchmark_performance/`, `outputs/benchmark_profiling/`
- Wrong: `outputs/benchmarks_performance/`, `outputs/benchmarks_profiling/`

---

## 2. Required Artifacts Per Benchmark Run

Every benchmark must emit:

| Artifact | Format | Notes |
|----------|--------|-------|
| `results.json` | JSON | Machine-readable summary — timings, metrics, config |
| `REPORT.md` | Markdown | Human-readable summary with key findings and interpretation |
| `figures/` | PNG, ≥150 DPI | QA figures: profiles, residuals, speedup bars |

Use `benchmarks/utils/run_metadata.py` to generate the environment block (git SHA, platform,
package versions) included in `results.json`.

---

## 3. Shared Utilities

| Module | Purpose |
|--------|---------|
| `benchmarks/utils/sersic_model.py` | Synthetic Sérsic image generation |
| `benchmarks/utils/run_metadata.py` | Environment metadata + JSON write |
| `benchmarks/utils/autoprof_adapter.py` | AutoProf subprocess adapter (needs `AUTOPROF_PYTHON` env var) |
| `benchmarks/utils/mockgal_adapter.py` | External mockgal.py + libprofit adapter for high-fidelity mock generation |
| `benchmarks/utils/scaffold_models_config_batch_templates.py` | Copy YAML templates for `models_config_batch` runs |

Future planned: `benchmarks/utils/report.py` — shared markdown report builder.

---

## 4. Data

- **Shared FITS files**: `data/` at project root.
  - `data/IC3370_mock2.fits` — Huang2013 mock
  - `data/eso243-49.fits` — edge-on S0
  - `data/ngc3610.fits` — boxy-bulge elliptical
  - `data/m51/M51.fits` — M51 spiral
- **External Huang2013 data**: not tracked in the repo; the path is machine-specific and
  supplied per run via `--huang-root` (see `examples/example_huang2013/`).
- **Synthetic data**: generate at runtime using `benchmarks/utils/sersic_model.py` or
  `benchmarks/utils/mockgal_adapter.py`.

---

## 5. CLI Requirements

Every benchmark script must:

1. Accept `--help` and print a concise usage description.
2. Accept a `--quick` flag that runs a fast smoke test (single galaxy or single config) in
   under 60 seconds.
3. Accept `--output <dir>` to override the default output directory.

---

## 6. AutoProf Benchmark Notes

`bench_vs_autoprof.py` uses a subprocess-based adapter because AutoProf requires `numpy < 2`,
which conflicts with isoster's environment. The adapter spawns a separate Python interpreter
pointed to by the `AUTOPROF_PYTHON` environment variable.

### Setting AUTOPROF_PYTHON

`AUTOPROF_PYTHON` must point to a Python binary in an environment where AutoProf is installed
with `numpy < 2`. The compiled-in default is machine-specific and will not exist on your
system, so set the variable rather than relying on it. For the canonical install recipe —
including why the venv must not live in `/tmp` — see
[`benchmarks/exhausted/README.md`](exhausted/README.md); `docs/09-exhausted-benchmark.md` §0
reproduces it.

```bash
# Check what python is currently set (or defaulted to)
echo ${AUTOPROF_PYTHON:-<machine-specific default; see autoprof_adapter.py>}

# Override for your environment
export AUTOPROF_PYTHON=/path/to/autoprof-env/bin/python3

# Quick sanity check
$AUTOPROF_PYTHON -c "import autoprof; import numpy; print(numpy.__version__)"
```

To create a compatible AutoProf environment with conda/mamba:

```bash
conda create -n autoprof python=3.9 numpy="<2" && conda activate autoprof
pip install autoprof
```

Once set, the benchmark can be verified with:

```bash
uv run python benchmarks/performance/bench_vs_autoprof.py --quick --plots
```

---

## 7. Adding a New Benchmark — Checklist

> Obsolete scripts (`convergence_diagnostic.py`, `huang2013_convergence_benchmark.py`,
> `ngc1209_convergence_benchmark.py`, `bench_isofit_overhead.py`) were deleted in the
> 2026-03-02 housekeeping session.

Before submitting a new benchmark script:

- [ ] Script lives under `benchmarks/<category>/`
- [ ] Script name follows `<verb>_<subject>.py`
- [ ] `--help`, `--quick`, `--output` flags implemented
- [ ] Emits `results.json` and `REPORT.md`
- [ ] Uses `benchmarks/utils/run_metadata.py` for environment block
- [ ] Output folder uses singular prefix (`benchmark_`, not `benchmarks_`)
- [ ] Data paths use `data/` at project root
- [ ] Script added to census table in `benchmarks/README.md`
