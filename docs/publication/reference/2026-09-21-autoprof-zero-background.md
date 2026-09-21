# Huang2013 AutoProf zero-background replacement and analysis

## Scope and current status

The user approved fixing AutoProf's background to the known injected value,
`ap_set_background=0.0`, and repeating the full Huang2013 scientific analysis.
This compares extraction from prepared images, not an automatic sky-estimation
pipeline. S4G will receive the same policy in a later campaign.

The small fitting and analysis-environment gates passed. The full Huang2013
campaign has been launched; no full-sample result is claimed yet.

Replacement means explicit selection of the new AutoProf campaign. All old
images, fitted profiles, auxiliary files and analyses remain unchanged.
Isoster and Photutils are not refitted. The fixed-center AutoProf arm reads
copies of the accepted Isoster reference profiles.

## Conditions and reproducibility

The campaign retains the previous 93 galaxies, nine scenarios, four AutoProf
arms (`baseline`, `deep`, `high_regularization`, `fix_center`) and corrected
PA conversion: 837 images and 3,348 requested fits. All saved fitting options
other than background and filesystem paths must equal their PA-corrected
predecessors. The noise-estimation algorithm is unchanged; its measured
output is not required to equal the old estimate.

Code is isolated from Dropbox in
`/Users/shuang/code/isoster-autoprof-zero-background-20260921-remote`, branch
`benchmark/autoprof-zero-background`, implementation commit `6a4099a`.
The independent uv environment is `/Users/shuang/.venvs/isoster-zero-background`
(Python 3.12.11); AutoProf still uses
`/Users/shuang/.venvs/autoprof_venv/bin/python`. Eight galaxy workers are used,
with BLAS, OpenMP, MKL, Accelerate and NumExpr threads each limited to one.
`caffeinate -i` prevents idle sleep. These are throughput runs, not a repeat
of the controlled Stage 4 timing experiment.

All directories below are children of
`/Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28`:

- Source: `fits/publication_autoprof_pa_huang_2026_09_11`.
- Gate: `fits/publication_autoprof_zero_gate_2026_09_21`.
- Gate figures: `analysis/autoprof_zero_gate_qa_2026_09_21`.
- Full replacement: `fits/publication_autoprof_zero_huang_2026_09_21`.
- Environment-only analysis check (old PA selection, not new science):
  `analysis/huang2013_zero_background_analysis_environment_gate_2026_09_21`.

Invoke the existing correction entry point with the tracked new configuration:

```bash
uv run python -u -m benchmarks.exhausted.campaigns.run_autoprof_pa_correction \
  benchmarks/exhausted/configs/campaign.publication_autoprof_zero_huang_2026_09_21.yaml \
  --zero-background \
  --source-campaign /Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28/fits/publication_autoprof_pa_huang_2026_09_11
```

Set the environment and thread limits described above before invocation.
The runner rejects an existing destination. Its `correction_audit` directory
records resolved options, source hashes, dependency centers, environments,
accepted outcomes and completion verdicts. Scientific failures are retained;
no best-of-run selection is allowed.

## Gate results

All four arms succeeded for IC2597/wide_z005, NGC1209/noiseless_z005 and
NGC4742/deep_z050: 12/12 successful. All saved and reported backgrounds are
zero, reference centers match exactly, all other fitting options match, and
source hashes are unchanged. The recorded fitting/audit phase took 55.564 s,
excluding source-selection startup. Forty-four focused regression tests passed.

The three PNG/PDF before/after diagnostics reuse the existing PA-correction
QA function with accurate background labels, shared residual scales and a
common no-harmonic reconstruction aperture. Visual inspection found no new
execution pathology. The compact high-redshift case retains large ellipticity
changes, not silently discarded as bad data. Zero background is the known
input condition, not a promise that every individual residual must improve.

The installed AutoProf `Background_Mode` estimates noise from pixels below
the adopted background. If that estimate is non-finite, its existing fallback
uses half the 16th--84th percentile span of sampled intensities. Thus a
positive noiseless galaxy with zero adopted sky still receives a nonzero
internal fitting scale (0.012 ADU for the NGC1209 gate), not an injected noise
measurement. We did not add noise, change that fallback or set
`ap_set_background_noise`. This remains a noiseless-fitting caveat.

The frozen generator was recloned outside Dropbox and checked out at
`a6a90a07dc3aedd95465928ee2e93258c8ccb40a`. A one-galaxy environment check
processed 153 NGC1209 records successfully before the full new analysis.

## Analysis caveats

Recompute all selected metrics, matched primary comparisons, paired Isoster
contrasts, truth/seed checks, source hashes and figures in new directories.
Because the common aperture intersects every successful arm's finite model,
changing AutoProf can also change the metrics of unchanged Isoster/Photutils
fits. Report this explicitly in old/new comparisons. The no-harmonic renderer,
one noise draw per galaxy/scenario, auxiliary-center rounding, descriptive
harmonics and non-comparable AutoProf stop codes remain limitations.
