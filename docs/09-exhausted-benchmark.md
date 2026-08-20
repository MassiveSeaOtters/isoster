# Exhausted Benchmark Campaign

`benchmarks/exhausted/` is a declarative, multi-tool, multi-galaxy
campaign framework for stress-testing `isoster` against alternatives
(`photutils.isophote`, `autoprof`) across a curated knob roster
("arms"). One campaign YAML drives the whole run: arms, galaxies,
tools, QA, and aggregated statistics.

This document is the reference for operators and the recipe for new
contributions (new tools, new datasets, new arms).

## 0. AutoProf setup (one-time, optional)

AutoProf 1.3.4 pins `numpy<2` and `photutils==1.5`, so it cannot share the
main `uv` environment. Install it in its own venv at a **stable** location —
not `/tmp`, which macOS prunes periodically and will silently strip individual
`.py` files mid-campaign:

```bash
mkdir -p ~/.venvs && python -m venv ~/.venvs/autoprof_venv
~/.venvs/autoprof_venv/bin/pip install --upgrade pip
~/.venvs/autoprof_venv/bin/pip install 'autoprof==1.3.4'
~/.venvs/autoprof_venv/bin/python -c "from autoprof.Pipeline import Isophote_Pipeline; print('ok')"
```

Point the code at it by whichever entry point applies:

| Consumer | How to point it at the venv |
|---|---|
| Exhausted campaigns (`benchmarks/exhausted/`) | `tools.autoprof.venv_python` in the campaign YAML |
| `bench_vs_autoprof.py` (`benchmarks/utils/autoprof_adapter.py`) | `AUTOPROF_PYTHON` environment variable |

Both carry machine-specific defaults that will not exist on your system; set
one of the two above rather than relying on them. When the venv is absent,
AutoProf arms skip cleanly with a regeneration hint.

## 1. Quick Start

```bash
# (one-time) set up the AutoProf venv if you want that tool active.
# See "AutoProf setup" below -- use a stable path, never /tmp.
mkdir -p ~/.venvs && python -m venv ~/.venvs/autoprof_venv
~/.venvs/autoprof_venv/bin/pip install 'autoprof==1.3.4'

# Dry-run the planned fit matrix.
uv run python -m benchmarks.exhausted.orchestrator.cli dry-run \
    benchmarks/exhausted/configs/campaign.example.yaml

# Execute it. Outputs land under `output_root/campaign_name/`.
uv run python -m benchmarks.exhausted.orchestrator.cli run \
    benchmarks/exhausted/configs/campaign.example.yaml
```

Cached runs short-circuit: any arm whose `profile.fits` already exists
under the campaign output is skipped and its inventory row is rebuilt
from the on-disk `run_record.json`. Delete the arm directory to force a
re-fit.

## 2. Campaign YAML Schema

The campaign YAML is validated at load time by
`benchmarks/exhausted/orchestrator/config_loader.py`. Top-level keys:

| Key | Type | Notes |
|---|---|---|
| `campaign_name` | str | Used as output subdirectory name. |
| `output_root` | path | Repo-relative (under `outputs/`), absolute, or `~`-prefixed. |
| `tools` | map | One entry per tool (`isoster`, `photutils`, `autoprof`). |
| `isoster_harmonic_sweeps` | list\[list\[int]] | Expands into `harm_higher_orders::*` isoster arms. Explicit `[]` disables the expansion; omitting the key falls back to `[[5, 6]]`. |
| `datasets` | map | One entry per dataset. `adapter` is the module name under `adapters/`. |
| `qa` | map | Per-figure toggles (see below). |
| `execution` | map | `max_parallel_galaxies`, `skip_existing`, `fail_fast`, `dry_run`. |
| `summary` | map | Includes the composite-score weights. |

The canonical example lives at
`benchmarks/exhausted/configs/campaign.example.yaml`; the smoke
variant `campaign.smoke_local.yaml` is what the integration smoke test
and hands-on debugging run against.

Per-tool entry:

```yaml
tools:
  isoster:
    enabled: true
    arms_file: benchmarks/exhausted/configs/isoster_arms.yaml
  photutils:
    enabled: true
    arms_file: benchmarks/exhausted/configs/photutils_arms.yaml
  autoprof:
    enabled: true
    arms_file: benchmarks/exhausted/configs/autoprof_arms.yaml
    venv_python: "~/.venvs/autoprof_venv/bin/python"   # stable path; see §0
    timeout: 300        # seconds per arm
```

Per-dataset entry:

```yaml
datasets:
  huang2013:
    enabled: true
    adapter: huang2013
    root: "~/isophote_test/output/huang2013"
    select:             # optional allowlist
      - "NGC_0596/mock001"
```

`qa` toggles:

| Key | Default | Effect |
|---|---|---|
| `per_galaxy_qa` | true | Writes `arms/<arm>/qa.png`. |
| `cross_arm_overlay` | true | Writes `<tool>/cross_arm_overlay.png` (6-panel all-arms overlay). |
| `cross_tool_comparison` | true | Writes `<galaxy>/cross/cross_tool_comparison.png` (effective only when ≥2 tools are enabled and produce rows). |
| `summary_grids` | false | Per-(tool, arm) multi-galaxy thumbnail grids. Off by default: the format does not scale past ~20 galaxies. |
| `residual_models` | true | Writes `arms/<arm>/model.fits` with MODEL + RESIDUAL HDUs. |
| `sb_profile_scale` | `log10` | 1D SB profile scale for QA figures: `log10` or `asinh` / `arcsinh`. |
| `sb_asinh_softening` | null | Optional positive asinh softening scale; in calibrated QA it is flux per arcsec². |

`execution`:

| Key | Default | Effect |
|---|---|---|
| `max_parallel_galaxies` | 1 | `>1` activates a `ProcessPoolExecutor` with `mp_context=spawn`. Tools and arms always run serially within one galaxy. |
| `skip_existing` | true | Skip any arm whose `profile.fits` exists; rebuild the inventory row from `run_record.json`. |
| `fail_fast` | false | On the first worker exception, cancel pending galaxies and raise. Already-running workers cannot be killed cleanly. |
| `dry_run` | false | Reserved; use the `dry-run` subcommand for matrix inspection. |

## 3. Arm File Format

Each arm is a minimal *delta* from the tool's baseline configuration.
The runner merges the delta into the tool's default config before
invoking the fitter. Arms live in:

- `configs/isoster_arms.yaml`   — 22 declared arms + `harm_higher_orders::*` expansion
- `configs/photutils_arms.yaml` — 5 arms
- `configs/autoprof_arms.yaml`  — 4 arms

Top-level schema:

```yaml
arms:
  ref_default: {}                # empty delta = use the tool's baseline
  sclip_off:
    nclip: 0                     # isoster knob override
```

### Sentinels

Certain values are resolved at fit time against the loaded galaxy
rather than the static YAML. The runner routes through
`fitters/<tool>_fitter.py` to substitute these before the config is
built.

| Sentinel | Scope | Behavior |
|---|---|---|
| `_use_Re` | isoster (integrator knobs) | Replaced by `adapter.effective_Re_pix`. If the adapter supplies no Re, the arm is skipped with `status="skipped"` and a clear `error_msg`. |
| `_use_2Re` | isoster (integrator knobs) | As above, multiplied by 2. |
| `_special: drop_variance` | isoster (top-level key) | Forces `variance_map=None` (OLS path). If the adapter did not supply a variance map, the arm is skipped (nothing to drop). |
| `_fix_center_from: isoster_weighted` | autoprof | Hard-fixes AutoProf's center via `ap_set_center` to the intensity-weighted (x0, y0) over the first 10 stop_code==0 isophotes of the companion isoster `ref_default` profile. Requires isoster to run first on the same galaxy; otherwise the arm is skipped with a clear `error_msg`. |

A galaxy whose adapter metadata does not carry `effective_Re_pix` or
whose bundle has no variance map will see these arms silently
`status="skipped"` rather than failing — this is what lets a generic
campaign run across heterogeneous datasets.

### `harm_higher_orders::*` expansion

The campaign YAML's `isoster_harmonic_sweeps` is a list of integer
lists. Each entry expands into one isoster arm with id
`harm_higher_orders::<entries joined by "_">` and the delta
`harmonic_orders: <entry>`. Example:

```yaml
isoster_harmonic_sweeps:
  - [5, 6]
  - [5, 6, 7, 8]
# -> arms: harm_higher_orders::5_6, harm_higher_orders::5_6_7_8
```

## 4. Output Layout

```
<output_root>/<campaign_name>/
├── campaign.yaml                    # frozen snapshot of the input YAML
├── environment.json                 # python + deps + git sha at run start
└── <dataset>/                       # one per enabled dataset
    ├── <galaxy_id>/
    │   ├── MANIFEST.json            # bundle metadata + image_sigma
    │   ├── cross/
    │   │   └── cross_tool_comparison.png
    │   └── <tool>/
    │       ├── inventory.fits       # additive per-arm inventory schema
    │       ├── cross_arm_overlay.png
    │       ├── cross_arm_table.csv
    │       ├── cross_arm_table.md
    │       └── arms/<arm>/
    │           ├── profile.fits     # canonical per-arm profile
    │           ├── model.fits       # MODEL + RESIDUAL HDUs
    │           ├── qa.png
    │           ├── config.yaml      # effective config for this arm
    │           └── run_record.json  # timings + metrics + flags
    ├── <tool>/cross_arm_summary.{csv,md}   # per-tool medians across galaxies
    └── cross_tool_table.{csv,md}           # dataset-level tool comparison
```

Auxiliary AutoProf artifacts (`.prof`, `.aux`, `*_genmodel.fits`) land
in `arms/<arm>/raw/` inside the per-arm directory.

## 5. Composite Score

### Evaluation workflow standard

The exhausted benchmark follows the same QA principle as the plotting
API: every score should be traceable back to the same profile, model,
mask, and metadata artifacts shown in the QA figures. A campaign run
should therefore be read in this order:

1. Confirm each arm's `status`, `flags`, and `flag_severity_max`.
2. Inspect the per-arm `qa.png` for image/model/residual/profile
   agreement.
3. Use `cross_arm_overlay.png` to compare arms within one tool.
4. Use `cross_tool_comparison.png` to compare the best available
   method products for one galaxy.
5. Use `inventory.fits`, cross-arm tables, and cross-tool tables for
   ranking after the visual failure modes are understood.

The score is a decision aid, not a replacement for QA inspection. A
low score should mean "this run deserves to rank near the top among
runs with comparable status," not "the fit is scientifically correct."

### Artifact contract

Each method/arm writes a small set of artifacts that serve both the QA
figures and the metric code:

- `profile.fits`: the profile table used for SB, geometry, and
  completeness diagnostics.
- `model.fits`: the rendered model and residual image used for visual
  residual inspection and model-evaluation metrics.
- `config.yaml`: the effective method configuration after arm deltas
  and sentinels are resolved.
- `run_record.json`: runtime, metrics, flags, and status for the arm.
- `inventory.fits`: the additive per-tool table used by cross-arm and
  cross-tool summaries.

Do not rank a method using one residual definition while showing a
different residual image in QA. If an adapter or fitter cannot write a
model, the missing model should be visible as missing QA context and
the row should not be silently promoted by a partial metric set.

### Region gating

Residual metrics are contour/geometry gated so that all tools are
measured on the same galaxy footprint. This avoids rewarding a method
for fitting blank sky or penalizing it for pixels outside the
meaningful galaxy region. The residual zones are anchored to the
adapter's initial geometry rather than the fit-derived geometry, so a
method cannot move the evaluation aperture to make its own residuals
look better.

When comparing campaigns across datasets, confirm that `R_ref_used_pix`,
`r_inner_floor_pix`, `image_sigma_adu`, and `sigma_method` are
reasonable for the dataset. These fields document the scale and noise
normalization that make the residual zones interpretable.

### v1.1 model-evaluation metrics

Each successful arm writes the verified v1.1 metric contract into both
`run_record.json["metrics"]` and the additive inventory schema. The
residual zones are anchored to the adapter's initial geometry, not the
fit-derived geometry, so different tools are measured on the same
elliptical coordinate system.

Zone definitions use `R_ref_used_pix`:

- `inner`: `r_inner_floor_pix <= r < 0.5 * R_ref_used_pix`
- `mid`: `0.5 * R_ref_used_pix <= r < 1.5 * R_ref_used_pix`
- `outer`: `1.5 * R_ref_used_pix <= r < min(maxsma, 3 * R_ref_used_pix)`

The inventory preserves the older `resid_rms_{inner,mid,outer}` and
`resid_median_{inner,mid,outer}` names, but they now use the v1.1 zone
definitions. Additional amplitude metrics are:

- `F_ref`, `R_ref_used_pix`, `r_inner_floor_pix`
- `sigma_{inner,mid,outer}` using `1.4826 * MAD` per zone
- `abs_resid_over_ref_{inner,mid,outer}`
- `abs_resid_over_sigma_{inner,mid,outer}`

Azimuthal pattern metrics are:

- `azim_A1_{zone}`, `azim_A2_{zone}`
- `azim_struct_ratio_{zone}`, `azim_phase1_deg_{zone}`
- `quadrant_imbalance_{zone}`

Quality flags use the v1.1 portable names and thresholds:
`FEW_ISOPHOTES`, `FIRST_ISOPHOTE_RETRY`,
`HIGH_NONZERO_STOP_FRAC`, `LARGE_DRIFT`, `LARGE_DPA`,
`LARGE_DEPS`, `INNER_RESID_LARGE`, `OUTER_RESID_LARGE`,
`FAILED`, and `SKIPPED`.

Every inventory row is annotated with `composite_score` at write time.
Lower is better. Rows with non-`ok` status or a severity-error quality
flag get a flat penalty of `1_000_000` so pathological fits never rank
best.

All terms are dimension-normalized so no single axis dominates by
amplitude alone:

```
score =
    # FIDELITY (residual RMS per zone, normalized by image sigma)
    w_inner * resid_rms_inner / image_sigma
  + w_mid   * resid_rms_mid   / image_sigma
  + w_outer * resid_rms_outer / image_sigma

    # CENTER STABILITY (centroid wander only)
  + w_centroid * combined_drift_pix / centroid_tol_pix

    # CONVERGENCE HEALTH
  + w_stop_m1 * n_stop_m1
  + w_frac_stop * frac_stop_nonzero

    # COMPLETENESS (penalize short fits vs. best arm on this galaxy)
  + w_completeness * max(0, 1 - n_iso / n_iso_ref)

    # SPEED (tie-breaker only)
  + w_wall_time * wall_time_fit_s
```

The reference `n_iso_ref` is the max `n_iso` across `status in {ok,
cached}` rows with no severity-error flag; it is rebuilt per galaxy
per tool and stamped into `image_sigma_adu` / `n_iso_ref_used` columns
of the inventory for audit.

### What is explicitly not scored

- `max_dpa_deg`, `max_deps` (PA / ellipticity drift): both can be
  genuinely astrophysical in elliptical BCGs (isophote twist, ε
  gradient). They are reported in the cross-arm tables for inspection
  but do not enter the score.

Defaults (from `stats.DEFAULT_WEIGHTS`):

```yaml
summary:
  composite_score_weights:
    resid_inner: 1.0
    resid_mid: 1.0
    resid_outer: 2.0     # LSB is where arms actually differ
    centroid_drift: 1.0
    centroid_tol_pix: 2.0  # divisor; scale with PSF FWHM on real data
    n_stop_m1: 2.0
    frac_stop_nonzero: 5.0
    n_iso_completeness: 3.0
    wall_time: 0.1
```

### Cross-tool score (tool-neutral)

`composite_score` is only fair *within one tool*: different fitters do
not populate the same axes uniformly. AutoProf forces a single shared
center for the whole profile and emits no photutils-style stop codes,
so `combined_drift_pix`, `n_stop_m1`, and `frac_stop_nonzero` are zero
by construction and would hand AutoProf an unfair advantage in any
cross-tool ranking.

The cross-tool table therefore emits two additional columns alongside
`composite_score`:

- `cross_tool_score` (Option B, primary)

  ```
  cross_tool_score =
      w_in  * resid_rms_inner / image_sigma
    + w_mid * resid_rms_mid   / image_sigma
    + w_out * resid_rms_outer / image_sigma
    + w_t   * wall_time_fit_s
  ```

- `cross_tool_score_simple` (Option A, publication-friendly sanity
  check) — outer-zone residual only, plus runtime:

  ```
  cross_tool_score_simple =
      w_out * resid_rms_outer / image_sigma
    + w_t   * wall_time_fit_s
  ```

Both drop the unfair axes (`centroid_drift`, stop-code counts,
completeness). Weights are independent of the intra-tool weights;
defaults preserve the familiar 1 / 1 / 2 residual-zone emphasis:

```yaml
summary:
  cross_tool_score_weights:
    resid_inner: 1.0
    resid_mid: 1.0
    resid_outer: 2.0
    wall_time: 0.1
```

Non-ok and severity-error rows sink to the same `1_000_000` penalty as
the composite score, so catastrophic fits never win either ranking.

### Reading comparison outputs

Use `composite_score` for cross-arm selection inside one tool. It can
include tool-specific health terms such as centroid drift, stop-code
counts, and profile completeness because all arms share the same method
contract.

Use `cross_tool_score` for method comparison across tools. It is
restricted to tool-neutral residual zones and runtime because methods
do not expose equivalent convergence and geometry-health signals.
Keep the excluded diagnostics in view: a method with a good
tool-neutral residual score can still be operationally risky if its QA
figure shows unstable geometry, poor coverage, or a severe flag.

The `cross_tool_score_simple` column is intended as a sanity check and
communication aid. It emphasizes the outer residual floor plus runtime,
so it is easy to explain, but it should not replace the full
zone-based score during detailed debugging.

### Surface-brightness profile scale in benchmark QA

Benchmark QA defaults to `sb_profile_scale: log10`. This keeps the
historical log/magnitude profile behavior and is usually the right
choice for high-S/N inner-region comparisons.

Use `sb_profile_scale: asinh` when the outer profile approaches zero,
has negative samples, or needs to show low-S/N behavior without
dropping finite points. If the adapter provides both `sb_zeropoint` and
`pixel_scale_arcsec`, the asinh profile is calibrated so high-S/N
points match the log10 mag/arcsec² profile; the QA panel draws a dashed
`I = 0` reference line to mark the zero-intensity level.

When validating a new QA option, set `execution.skip_existing: false`
or delete the target arm directories. Reusing cached `profile.fits`
files can validate table reconstruction, but it does not prove that the
new figure option was rendered.

## 6. Writing a New Dataset Adapter

A dataset adapter is a module under `benchmarks/exhausted/adapters/`
that exposes `ADAPTER_CLASS` pointing at a class implementing the
`DatasetAdapter` protocol (`adapters/base.py`):

```python
class MyAdapter:
    dataset_name = "my_dataset"

    def __init__(self, root, **kwargs):
        ...

    def list_galaxies(self) -> list[str]:
        """Stable-ordered galaxy_id strings found under self.root."""

    def load_galaxy(self, galaxy_id: str) -> GalaxyBundle:
        """Return image + variance + mask + initial_geometry."""


ADAPTER_CLASS = MyAdapter
```

Requirements:

- The adapter instance must be picklable: the orchestrator passes it
  to spawn workers when `execution.max_parallel_galaxies > 1`. Keep
  state in plain attributes (paths, primitive dicts); avoid live file
  handles or database connections.
- `list_galaxies` is called once at run start, so it can be
  O(filesystem-walk). `load_galaxy` runs inside the worker and should
  be cheap (it fires once per arm-group per galaxy).
- Initial geometry keys: `x0, y0, eps, pa, sma0, maxsma`. `pa` is in
  radians (IsosterConfig convention). Supply `effective_Re_pix` in the
  returned `GalaxyMetadata` when the dataset knows the half-light
  radius; `_use_Re` / `_use_2Re` arms are skipped otherwise.

Reference implementations:

- `adapters/huang2013.py` — multi-Sersic mock directory layout
  (`<galaxy>_mockNNN.fits` per directory).
- `adapters/huang2013_scenarios.py` — scenario grid layout
  (`<galaxy>_{clean|wide|deep}_z{005|020|035|050}.fits`). Accepts
  optional `depths` and `redshift_tags` constructor kwargs to narrow
  the enumeration at the YAML level. Emits `galaxy_id` as
  `<galaxy>/<depth>_z<zzz>` so `safe_galaxy_id` produces one flat
  output directory per scenario.
- `adapters/s4g_scenarios.py` — same layout as the Huang2013 scenario
  adapter but emits `dataset_name="s4g"`. Thin subclass; no new logic.
- `adapters/local_fits.py` — explicit FITS paths listed in the YAML.

Register in the campaign YAML by setting `adapter: my_dataset`.

### Scenario mock driver

`benchmarks/exhausted/campaigns/run_mock_campaigns.py` wraps the
orchestrator with a compact CLI for the depth x redshift grids. It
generates a campaign YAML from flag values and hands it to
`run_campaign` (the YAML is still snapshotted to the output directory,
so every run remains reproducible from disk):

```bash
# 2-galaxy smoke, dry-run only.
uv run python -m benchmarks.exhausted.campaigns.run_mock_campaigns \
    --dataset huang2013 --depth wide --redshift 020 \
    --select IC1459 NGC1600 --tools isoster,autoprof \
    --max-parallel 2 --dry-run

# Real run. All scenarios for one galaxy, three tools.
uv run python -m benchmarks.exhausted.campaigns.run_mock_campaigns \
    --dataset s4g --depth all --redshift all \
    --select ESO026-001 --tools all --max-parallel 4
```

Default roots: `/Volumes/galaxy/isophote/huang2013` and
`/Volumes/galaxy/isophote/s4g_mock`. Override with `--root-huang2013`
/ `--root-s4g`. Pass `--write-yaml PATH` to also persist the generated
YAML for later reuse via the plain `orchestrator.cli` entry point.

## 7. AutoProf Setup

AutoProf 1.3.4 requires `numpy < 2` and `photutils == 1.5`, which
cannot coexist in `isoster`'s `uv` environment (numpy 2.x). The
campaign runs it via subprocess against a dedicated venv.

See §0 for the install recipe. In short: put the venv at a stable path such
as `~/.venvs/autoprof_venv`, never under `/tmp`, which macOS prunes and which
will strip individual `.py` files out from under a running campaign.

Point `tools.autoprof.venv_python` at that venv's Python. The compiled-in
default is `/tmp/autoprof_venv/bin/python`, which is exactly the location this
document advises against — set the YAML key rather than relying on it. If the
path is missing or the import
fails, every autoprof arm is reported as `status="skipped"` with a
clear regeneration hint — the rest of the campaign continues.

## 8. Parallel Execution

Set `execution.max_parallel_galaxies > 1` to run galaxies concurrently
via `concurrent.futures.ProcessPoolExecutor` with the `spawn` start
method (deterministic across platforms). Each worker:

- Reloads the campaign plan from pickled state.
- Loads the galaxy bundle, estimates `image_sigma`, writes `MANIFEST.json`.
- Runs every enabled tool × arm serially for that galaxy.
- Writes per-galaxy cross-tool comparison figure.

Dataset-level aggregation (per-tool cross-arm summary, cross-tool
pivot table, optional multi-galaxy grids) runs in the main process
once every galaxy future resolves.

Caveats:

- Workers cannot share cached matplotlib state. `plotting/__init__.py`
  calls `matplotlib.use("Agg")` and `_process_one_galaxy` re-asserts
  it on entry, so figure rendering is headless and process-safe.
- `fail_fast=true` cancels pending futures after the first worker
  exception. Already-running workers continue to completion — Python
  cannot kill them cleanly. The final `RuntimeError` chains the
  original exception for debugging.
- Startup cost: macOS `spawn` pays ~1–2 s per worker to reimport
  modules. Negligible against real fits (seconds to minutes each) but
  noticeable on all-cached re-runs.

## 9. Command Reference

```bash
# Dry-run the fit matrix (adapters loaded, no fits performed).
uv run python -m benchmarks.exhausted.orchestrator.cli dry-run <campaign.yaml>

# Run the campaign. Exit code 0 on zero failed arms, 1 otherwise.
uv run python -m benchmarks.exhausted.orchestrator.cli run <campaign.yaml>
```

The one-galaxy × two-arm integration smoke at
`tests/integration/test_exhausted_smoke.py` is a self-contained
executable example and should be kept passing as part of the regular
test suite.
