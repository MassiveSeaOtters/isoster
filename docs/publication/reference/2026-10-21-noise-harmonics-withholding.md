# Withholding noise harmonics: mock truth and SGA-2020 residuals

Experiment of 2026-10-21, isoster 1.1.2, branch `experiment/withhold-noise-harmonics`.
Question left open by the 1.1.2 fix (failed rings no longer carry harmonics): converged
rings can still carry harmonics that are noise, either because the ring's gradient was not
measurable (geometry locked or the gradient test bypassed) or because the harmonic itself
is not significant. Does the model improve if they are not drawn?

## Method

Rules act on a ring's stored a3, b3, a4, b4 before `build_isoster_model` (whose >= 1
amplitude guard stays on); nothing is refit for the mocks, and the fitter never reads
stored harmonics back.

| rule | harmonics drawn |
|---|---|
| `current` | isoster 1.1.2: every ring with stop code 0 or 2 |
| `off` | none |
| `gradX` | `current`, minus rings whose relative gradient error exceeds X or is undefined (X = 0.5 is the default `maxgerr`, the fitter's own test) |
| `snrK` | `current`, minus each order n whose amplitude S/N, sqrt(a_n² + b_n²) / sqrt((σ_a² + σ_b²) / 2), is below K |
| `grad0.5+snrK` | both cuts |

- **Mocks** (`benchmarks/exhausted/analysis/round2_harmonic_withhold.py`): all 837
  Huang2013 and 1,800 S4G inputs of the round-2 campaigns, six isoster arms, each profile
  drawn under every rule with the linear (native) and log-linear radial interpolation and
  scored by the truth-relative RMS per zone (zones of the round-2 records: inner < 0.5 Re,
  mid 0.5-2 Re, outer >= 2 Re). The truth is the PSF-convolved image cached by the round-2
  analysis. Check: the harmonics-off renders reproduce the archived `linear_off` and
  `log_off` RMS exactly on identical pixel support; the harmonics-on renders reproduce
  the archived `native` RMS to 0.003 percentage points, the documented size of the 1.1.1
  odd-harmonic sign fix.
- **SGA-2020** (`sga_isoster/scripts/onetime/_qa_harmonic_withhold.py`): the campaign
  profiles keep neither gradient errors nor harmonic errors, so 600 random bm1 round-2
  galaxies were refit with the frozen arm settings (six arms). Every refit reproduces the
  stored profile, and the `current` rule reproduces the campaign's stored residual
  exactly. Metric: mean |data - model| / sqrt(variance) per zone (campaign renderer and
  zones); win / loss beyond the Phase 4 null thresholds (inner 10.0 %, mid 5.2 %, outer
  1.0 %).

## How much each rule removes

Mean share of converged ring-orders withheld (mocks, all arms): `grad0.5` 1.5-2.3 %,
`grad0.3` 1.5-5.5 %, `snr2` 48-58 %, `snr3` 60-66 %. Most harmonics on these smooth
mocks are not significant; on noiseless inputs the S/N is set by the ring's residual
scatter, and removing them there changes nothing (ratio 1.000 in every zone).

## Results on the mocks (truth)

Median of (rule RMS / current RMS); percent of models better / worse by more than 2 %.
Log-linear renderer, `samp_cubic_ea` and `ref_default` (the other arms read the same; full
tables in the output folder). `off` and `grad0.5`: all inputs; the combined rules: noisy
inputs (deep and wide), since on noiseless inputs they change nothing.

| sample, zone | `off` | `grad0.5` | `grad0.5+snr2` | `grad0.5+snr3` |
|---|---|---|---|---|
| Huang2013 inner, `samp_cubic_ea` | 0.944; 53/42 | 1.000; 0/0 | 0.992; 26/1 | 0.977; 53/3 |
| Huang2013 outer, `samp_cubic_ea` | 0.755; 80/14 | 1.000; 2/0 | 0.904; 94/0 | 0.795; 98/0 |
| S4G inner, `samp_cubic_ea` | 1.775; 7/88 | 1.000; 0/0 | 0.997; 21/1 | 0.997; 27/1 |
| S4G mid, `samp_cubic_ea` | 1.343; 15/76 | 1.000; 0/0 | 0.993; 38/0 | 0.990; 44/0 |
| S4G outer, `samp_cubic_ea` | 0.721; 62/12 | 1.000; 1/0 | 0.890; 94/0 | 0.745; 96/0 |
| S4G outer, `ref_default` | 0.775; 61/12 | 1.000; 0/0 | 0.929; 86/1 | 0.811; 92/1 |

- Turning harmonics off is wrong on disks: the S4G bars live in the inner and mid
  harmonics (`off` worse on 76-88 % of models there).
- The gradient cut alone is a no-op on these arms: their converged rings almost all pass
  the fitter's own gradient test.
- The S/N cut keeps the bars (S4G inner and mid unchanged or slightly better) and removes
  the noise drawn into the outskirts: outer RMS 6-11 % lower at S/N 2, 16-26 % lower at
  S/N 3, worse on at most 1 % of models. Pooled over the six arms, noisy outer median
  6.04 → 5.42 % (Huang2013) and 1.61 → 1.45 % (S4G); whole-aperture 1.248 → 1.215 % and
  1.167 → 1.154 %.
- Cost of S/N 3 over 2: inner losses grow to 3-8 % of models on Huang2013 and 10 % for
  `harm_simul_ea` on S4G.
- isoster 1.1.2's failed-ring rule (`stored` → `current`) improves the in-loop arm
  `harm_simul_ea` on 31 % (Huang2013) and 4 % (S4G) of models over the full aperture,
  outer zone 39 % and 8 %; no other arm changes.

## Results on SGA-2020 (data residual)

Rule versus `current`, same arm (better / worse beyond the null threshold, % of 600):

| arm | `grad0.5` outer | `snr3` outer | `grad0.5+snr2` inner / mid / outer | `off` inner |
|---|---|---|---|---|
| `ref_default` | 0.2 / 0.0 | 0.7 / 0.2 | 0.0/0.3, 0.0/0.0, 0.5/0.2 | 0.0 / 64.2 |
| `permissive_on` | 12.6 / 0.0 | 12.8 / 0.3 | 0.0/0.2, 1.2/0.0, 12.8/0.2 | 0.0 / 62.5 |
| `stack_all` | 11.9 / 0.2 | 11.9 / 1.0 | 0.0/0.0, 0.8/0.0, 11.9/0.2 | 0.5 / 47.8 |

Arm versus `ref_default`, both under the same rule (the comparison of the round-2 record):

| arm | outer, `current` | outer, `grad0.5+snr2` |
|---|---|---|
| `permissive_on` | 0.7 / 14.5 | 0.7 / 1.8 |
| `stack_all` | 1.7 / 33.6 | 1.8 / 22.4 |
| `harm_isofit_inloop`, `lsb_autolock`, `samp_cubic_ea` | 2.7/2.4, 0.7/2.4, 3.5/3.4 | 3.0/2.4, 0.5/2.5, 3.5/3.2 |

- On real data the gradient cut is what matters: the outer-zone loss of `permissive_on`
  against the default is noise harmonics on rings whose gradient fails the fitter's own
  test, and it drops to the null level. `stack_all` keeps a 22 % loss, which matches
  the outer regularization's own (24.4 % for `reg_outer_damp`, round-2 record section 3).
- In the mid and outer zones the S/N cut does not move the SGA residual of the default
  arms beyond the null (≤ 0.8 % of galaxies): in the noise-dominated outskirts the data noise sets |data - model| / sigma.
  S/N 3 costs 2-3 % of galaxies in the inner zone; S/N 2 costs 0.2-0.3 %.
- The feature comparisons of the round-2 record (`samp_cubic_ea`, LSB lock, in-loop
  ISOFIT) are unchanged by the rule.

## Conclusion

Both kinds of noise harmonic exist and both are worth withholding, at different places:

1. **Unmeasurable gradient: in the fitter.** A converged ring whose relative gradient
   error exceeds `maxgerr` (or is undefined) carries a Bender-normalized harmonic divided
   by a gradient the fit could not measure; store zero, as 1.1.2 does for failed rings.
   It must be the fitter: the gradient error is stored only when `debug=True`. Effect:
   none on default fits; removes the outer-zone loss of permissive fits on real data.
2. **Insignificant harmonic: in the renderer.** Draw order n only when its amplitude S/N
   is at least 2, using the stored errors; profiles keep every measurement with its error
   for science use. Effect: outer truth error 6-11 % lower on noisy mocks, nothing worse
   anywhere beyond 1-2 % of models, SGA residuals unchanged, bars kept. S/N 3 removes more
   outer error at a small inner cost.

Both change the harmonics-on models, so the mock rescore and the SGA residuals should be
redone once, after this, issue 2 and the log-linear harmonics-on renderer. The SGA
rescore needs refits (the campaign profiles lack the error columns; the 600-galaxy,
six-arm refit above took 10 minutes on 8 workers), and the sga_isoster profile writer
should keep `grad_r_error` and the harmonic errors.

Outputs: `outputs/benchmark_exhausted/harmonic_withhold_2026_10_21/` and
`..._combined_2026_10_21/` (isoster; `summary.md`, `withhold_metrics.csv.gz`);
`/Volumes/galaxy/isophote/sga2020/output/campaign/bm1r2_phase4/harmonic_withhold*_2026_10_21/`
and `sga_isoster/output/qa/bm1r2_phase4/harmonic_withhold.md`.

## Adopted in isoster 1.2.0

Both rules are the default from 1.2.0 (decided 2026-10-21):

- **Fitter** (`fit_isophote`): a ring keeps a_n, b_n and their errors only if its stop code
  is 0 or 2 and the relative gradient error of its stored iteration (the value `debug=True`
  reports as `grad_r_error`) is within `maxgerr`; otherwise it stores zeros. The harmonics
  are still fitted; only what is stored changes, and no other column does. No separate
  switch: the threshold is `maxgerr`. Zeroed rings are recognizable by `a_n_err = 0`.
  Template forced photometry keeps its own rule (harmonics only where the gradient is
  measured) and the multi-band fitter is unchanged.
- **Renderer** (`build_isoster_model(..., harmonic_min_snr=2.0)`): order n of a ring is
  drawn only where its amplitude S/N reaches the threshold; 0 draws all. Orders without
  stored errors are drawn uncut with a warning.

Checks:

- 144 mock refits (24 random Huang2013 and S4G inputs, six arms) with 1.2.0 reproduce the
  stored profile exactly, and their harmonics equal the stored ones under `grad0.5`.
- 144 SGA-2020 refits (24 galaxies, six arms) with 1.2.0: the residual of the default
  render equals the experiment's `grad0.5+snr2` render exactly (difference 0.0 in every
  zone), and refits reproduce the stored intensities.
- Inside the 2 px inner cut that both experiments excluded (the gradient error is often
  undefined on the innermost rings of a cuspy profile): among the 180 mock models (60 random
  galaxies per sample, all scenarios, six arms) whose render changes under the gradient rule, the truth RMS within 2 px and 4 px
  is unchanged (median ratio 1.000; none better or worse by more than 2 %).
- Cost, on one S4G mock (513 x 513, 66 rings): the gradient rule takes 0.007 ms against a
  0.115 s fit; the render takes 0.059 s with the S/N cut and 0.063 s without (timing noise).
