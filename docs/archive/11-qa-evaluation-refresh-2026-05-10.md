# QA Evaluation Refresh 2026-05-10

This document records the completed refresh of the exhausted benchmark QA and
model-evaluation products.

The refresh updated existing campaign products from the already-written image,
profile, and model files. It did **not** refit the arms.

## Scope

The refresh covered both benchmark campaign trees:

- Huang2013 mock campaigns:
  `/Volumes/galaxy/isophote/huang2013/_campaigns/`
- S4G mock campaigns:
  `/Volumes/galaxy/isophote/s4g_mock/_campaigns/`

The refreshed products include:

- per-arm `run_record.json`
- per-tool `inventory.fits`
- per-tool `cross_arm_table.csv` and `cross_arm_table.md`
- per-scenario `cross_tool_table.csv` and `cross_tool_table.md`
- per-arm `qa.png`
- per-tool `cross_arm_overlay.png`
- per-galaxy `cross_tool_comparison.png`
- aggregate reports under each `_analysis/` directory

## Code Added

The refresh command added in this branch is:

- `benchmarks/exhausted/campaigns/refresh_model_evaluation.py`

It recomputes v1.1 model-evaluation metrics from existing campaign artifacts,
then rewrites the JSON records, FITS inventories, and score tables.

The focused test file is:

- `tests/unit/test_refresh_model_evaluation.py`

Verification command:

```bash
uv run --extra dev pytest -q tests/unit/test_refresh_model_evaluation.py
```

Result:

```text
3 passed
```

## Metric Refresh

Huang2013 command:

```bash
uv run --extra dev python -m benchmarks.exhausted.campaigns.refresh_model_evaluation \
  --campaign-root /Volumes/galaxy/isophote/huang2013/_campaigns \
  --dataset huang2013 \
  --write
```

Huang2013 result:

- 837 galaxies processed
- 30,132 of 30,969 arms refreshed
- 837 arms skipped
- 0 arms failed
- 2,511 `inventory.fits` files written
- 9 `cross_tool_table` files written

S4G command:

```bash
uv run --extra dev python -m benchmarks.exhausted.campaigns.refresh_model_evaluation \
  --campaign-root /Volumes/galaxy/isophote/s4g_mock/_campaigns \
  --dataset s4g \
  --write \
  --max-parallel 6 \
  --progress-every 25
```

S4G result:

- 1,500 galaxies processed
- 53,977 of 55,500 arms refreshed
- 1,523 arms skipped
- 0 arms failed
- 4,500 `inventory.fits` files written
- 5 `cross_tool_table` files written

## QA Figure Redraw

Huang2013 command:

```bash
uv run --extra dev python -m benchmarks.exhausted.campaigns.rerender_qa \
  --campaign-root /Volumes/galaxy/isophote/huang2013/_campaigns \
  --dataset huang2013 \
  --max-parallel 8 \
  --progress-every 25 \
  > outputs/qa-evaluation-refresh-huang2013-rerender.log 2>&1
```

Huang2013 result:

- 30,132 `qa.png` files redrawn
- 2,511 `cross_arm_overlay.png` files redrawn
- 837 `cross_tool_comparison.png` files redrawn
- 0 redraw errors

S4G command:

```bash
uv run --extra dev python -m benchmarks.exhausted.campaigns.rerender_qa \
  --campaign-root /Volumes/galaxy/isophote/s4g_mock/_campaigns \
  --dataset s4g \
  --max-parallel 8 \
  --progress-every 25 \
  > outputs/qa-evaluation-refresh-s4g-rerender.log 2>&1
```

S4G result:

- 53,977 `qa.png` files redrawn
- 4,500 `cross_arm_overlay.png` files redrawn
- 1,500 `cross_tool_comparison.png` files redrawn
- 0 redraw errors

No new `.err.txt` files were found during the final check.

## Aggregate Analysis Refresh

Aggregate reports were regenerated after the metric refresh and QA redraw.

Huang2013 aggregate output root:

```text
/Volumes/galaxy/isophote/huang2013/_campaigns/_analysis/
```

Huang2013 row counts:

- `clean_z005_audit`: 2,511 rows
- `cross_scenario_audit_isoster`: 22,599 rows
- `cross_scenario_audit_photutils`: 4,185 rows
- `cross_scenario_audit_autoprof`: 4,185 rows
- `cross_tool_extended_metrics`: 36 rows

S4G aggregate output root:

```text
/Volumes/galaxy/isophote/s4g_mock/_campaigns/_analysis/
```

S4G row counts:

- `cross_scenario_audit_isoster`: 40,500 rows
- `cross_scenario_audit_photutils`: 7,500 rows
- `cross_scenario_audit_autoprof`: 7,500 rows
- `cross_tool_extended_metrics`: 36 rows

The refreshed aggregate folders include:

- `cross_scenario_audit_isoster/`
- `cross_scenario_audit_photutils/`
- `cross_scenario_audit_autoprof/`
- `cross_tool_composite/`
- `cross_tool_extended_metrics/`

Huang2013 also includes:

- `clean_z005_audit/`

## Useful Logs

Refresh and analysis logs were written under `outputs/`:

```text
outputs/qa-evaluation-refresh-huang2013-rerender.log
outputs/qa-evaluation-refresh-s4g-rerender.log
outputs/qa-evaluation-refresh-huang2013-clean-z005-audit.log
outputs/qa-evaluation-refresh-huang2013-cross-scenario-isoster.log
outputs/qa-evaluation-refresh-huang2013-cross-scenario-photutils.log
outputs/qa-evaluation-refresh-huang2013-cross-scenario-autoprof.log
outputs/qa-evaluation-refresh-huang2013-cross-tool-composite.log
outputs/qa-evaluation-refresh-huang2013-cross-tool-extended.log
outputs/qa-evaluation-refresh-s4g-cross-scenario-isoster.log
outputs/qa-evaluation-refresh-s4g-cross-scenario-photutils.log
outputs/qa-evaluation-refresh-s4g-cross-scenario-autoprof.log
outputs/qa-evaluation-refresh-s4g-cross-tool-composite.log
outputs/qa-evaluation-refresh-s4g-cross-tool-extended.log
```

## Representative QA Figures

Huang2013:

```text
/Volumes/galaxy/isophote/huang2013/_campaigns/huang2013_clean_z005/huang2013/IC1459__clean_z005/isoster/arms/ref_default/qa.png
/Volumes/galaxy/isophote/huang2013/_campaigns/huang2013_clean_z005/huang2013/IC1459__clean_z005/cross/cross_tool_comparison.png
/Volumes/galaxy/isophote/huang2013/_campaigns/huang2013_wide_z050/huang2013/IC1459__wide_z050/isoster/arms/ref_default/qa.png
```

S4G:

```text
/Volumes/galaxy/isophote/s4g_mock/_campaigns/s4g_clean_z005/s4g/NGC1433__clean_z005/isoster/arms/ref_default/qa.png
/Volumes/galaxy/isophote/s4g_mock/_campaigns/s4g_wide_z010/s4g/ESO012-010__wide_z010/isoster/arms/ref_default/qa.png
/Volumes/galaxy/isophote/s4g_mock/_campaigns/s4g_deep_z010/s4g/ESO012-010__deep_z010/cross/cross_tool_comparison.png
```

## Known Remaining Issue

The known S4G `wide_z005` cross-tool table issue remains unchanged:

- `/Volumes/galaxy/isophote/s4g_mock/_campaigns/s4g_wide_z005/s4g/cross_tool_table.csv`
- current row count: 899
- expected row count: 900

This issue existed before the refresh and was not introduced by this work.

