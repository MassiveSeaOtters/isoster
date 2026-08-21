# Exhausted Benchmark Campaign

Declarative, multi-tool, multi-arm, multi-galaxy, multi-dataset
benchmark for `isoster`, `photutils.isophote`, and `autoprof`. One
campaign YAML drives the whole run: arms, galaxies, tools, QA, and
aggregated statistics.

Full reference: [`docs/09-exhausted-benchmark.md`](../../docs/09-exhausted-benchmark.md).

## Quick start

### 1. (Optional) Install AutoProf in its own venv

AutoProf 1.3.4 pins `numpy<2` and `photutils==1.5` and cannot coexist
with the main `uv` env. The fitter runs it via subprocess. When the
venv is absent, autoprof arms skip cleanly with a regeneration hint.

Use a stable location (not `/tmp`, which macOS prunes periodically and
will silently strip individual `.py` files mid-campaign):

```bash
mkdir -p ~/.venvs && python -m venv ~/.venvs/autoprof_venv
~/.venvs/autoprof_venv/bin/pip install --upgrade pip
~/.venvs/autoprof_venv/bin/pip install 'autoprof==1.3.4'
~/.venvs/autoprof_venv/bin/python -c "from autoprof.Pipeline import Isophote_Pipeline; print('ok')"
```

Override the path in the campaign YAML with `tools.autoprof.venv_python`.
This recipe is the canonical one; `docs/09-exhausted-benchmark.md` §0 and
`benchmarks/FRAMEWORK.md` §6 both defer to it.

### 2. Dry-run the fit matrix

```bash
uv run python -m benchmarks.exhausted.orchestrator.cli dry-run \
    benchmarks/exhausted/configs/campaign.example.yaml
```

Reports the planned `(dataset × galaxy × tool × arm)` count and
adapter-resolved galaxy lists without running any fits.

### 3. Run the campaign

```bash
uv run python -m benchmarks.exhausted.orchestrator.cli run \
    benchmarks/exhausted/configs/campaign.smoke_local.yaml
```

Outputs land under `<output_root>/<campaign_name>/`. Cached arms
(`profile.fits` exists) short-circuit; their inventory rows are
rebuilt from `run_record.json`. Delete an arm directory to force a
re-fit.

### 4. Read the results

For each dataset:

- `<dataset>/cross_tool_table.md` — one row per (galaxy, tool) at each
  tool's default arm, with `cross_tool_score` (three-zone residual +
  runtime, tool-neutral) and the original intra-tool `composite_score`.
- `<dataset>/<tool>/cross_arm_summary.md` — median metrics across
  galaxies for every arm of one tool.
- `<galaxy>/<tool>/cross_arm_table.md` + `cross_arm_overlay.png` — arm
  ranking and 6-panel overlay for one galaxy.
- `<galaxy>/cross/cross_tool_comparison.png` — per-galaxy 3-way tool
  comparison (fires when ≥2 tools are enabled).
- `<galaxy>/<tool>/arms/<arm>/{profile.fits, model.fits, qa.png,
  config.yaml, run_record.json}` — raw per-arm artifacts.

## Layout

```
benchmarks/exhausted/
├── configs/       Arm rosters + campaign YAML templates
├── adapters/      Dataset loaders (huang2013, local_fits, ...)
├── fitters/       Per-tool fit drivers (isoster / photutils / autoprof)
├── analysis/      Metrics, residual zones, quality flags, inventory
├── plotting/      Per-arm QA, cross-arm overlay, cross-tool figure
├── orchestrator/  CLI, runner, config loader, stats writers
└── legacy/        Archived single-galaxy 39-config sweep (pre-v2)
```

## Ready-to-run configs

- `configs/campaign.example.yaml` — full template with all three tools
  and the Huang2013 adapter. Enable tools/datasets as needed.
- `configs/campaign.smoke_local.yaml` — two local FITS files in `data/`
  exercising all three tools. Used by the integration smoke test and
  for quick iteration.

## Arm roster reference

| File | Tool | Arms |
|---|---|---|
| `configs/isoster_arms.yaml` | isoster | 22 declared + `harm_higher_orders::*` expansion (one arm per entry in `isoster_harmonic_sweeps`) |
| `configs/photutils_arms.yaml` | photutils | 5 |
| `configs/autoprof_arms.yaml` | autoprof | 5 (including `fix_center`, which pins the center to the isoster `ref_default` intensity-weighted centroid) |

Arm sentinels resolved at fit time:

- `_use_Re`, `_use_2Re` — isoster, replaced by the adapter's
  `effective_Re_pix`. Arm is skipped when the adapter has no Re.
- `_special: drop_variance` — isoster, forces OLS. Arm is skipped when
  the adapter has no variance map (nothing to drop).
- `_fix_center_from: isoster_weighted` — autoprof, sets `ap_set_center`
  from the companion isoster `ref_default` profile. Requires isoster
  to run first for the same galaxy.

## Scoring

- `composite_score` — intra-tool ranking (residual RMS + centroid
  drift + stop-code health + completeness + runtime).
- `cross_tool_score` — tool-neutral ranking used in the cross-tool
  table; residuals + runtime only. AutoProf forces a single shared
  center and emits no photutils-style stop codes, so the unfair axes
  are dropped.
- `cross_tool_score_simple` — outer-zone residual + runtime only;
  retained as a publication-friendly sanity check.

Lower is better everywhere. Non-ok and severity-error rows get a flat
`1_000_000` penalty so pathological fits never rank best.

## Parallelism

Set `execution.max_parallel_galaxies > 1` to run galaxies concurrently
via `concurrent.futures.ProcessPoolExecutor` with the `spawn` start
method. Tools and arms stay serial inside one galaxy. `fail_fast=true`
cancels pending futures after the first worker exception (running
workers cannot be killed cleanly).

## Caveats

- `skip_existing` checks only `profile.fits` existence. If you
  regenerate the isoster `ref_default` profile but leave autoprof's
  `fix_center` profile untouched, the autoprof arm will keep using
  the stale center until its directory is deleted.
- The `summary_grids: true` toggle emits one thumbnail grid per
  `(tool, arm)` and does not scale past ~20 galaxies. Off by default;
  raw outputs in `profile.fits` + `inventory.fits` + `run_record.json`
  are enough to reconstruct any multi-galaxy figure offline.

## Integration smoke test

```bash
uv run pytest tests/integration/test_exhausted_smoke.py -v
```

Creates a synthetic Sérsic galaxy in `tmp_path`, runs a 1-galaxy ×
2-arm × 1-tool campaign, and verifies inventory schema, profile
monotonicity, QA rendering, and `skip_existing` short-circuit
behavior. Runtime ≈ 3 s.

## See also

- [`docs/09-exhausted-benchmark.md`](../../docs/09-exhausted-benchmark.md)
  — full YAML schema, arm sentinels, output layout, composite + cross-
  tool score derivations, adapter recipe, AutoProf venv details,
  parallel execution semantics.
- `docs/agent/journal/2026-04-21-phase-e-hardening.md` — Phase E
  (parallelism, integration test, env-var fix, docs) development notes.
