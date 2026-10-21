# Renderer radius solve on twisting isophotes (issue 2)

Investigation of 2026-10-21, isoster 1.2.0, branch `analysis/issue2-renderer-spiral-cores`;
fixed in 1.2.1. Issue 2 of the SGA-2020 round-2 record
(`2026-10-13-sga2020-bm1-round2.md`): the same photutils fit read 2.05 sigma in the inner
zone of random spirals drawn by `build_isoster_model` and 1.45 drawn by photutils'
`build_ellipse_model`; ellipticals read 0.70 either way.

## Mechanism

`build_isoster_model` gives each pixel the intensity of the radius r that solves
r = R(r), R being the pixel's elliptical radius in the geometry (x0, y0, eps, pa)
interpolated at r. It solved this with three fixed-point steps starting from the outer
ring's geometry. Where the geometry twists quickly with radius (bars, bulge-to-disk
transitions) the steps oscillate and stop at a radius that no ring passes through.

Decomposition (`sga_isoster/scripts/onetime/_qa_issue2_render_decomposition.py`, 60 random
spirals and 60 random ellipticals of bm1 round 2; each variant changes one ingredient;
the photutils render reproduces the stored campaign models to 5e-7):

- photutils' "smoothing" spline (`LSQUnivariateSpline`, knots `sma[2:-2]`) passes through
  every ring (to 1e-13 of the column range on 30 profiles): it is a cubic interpolation.
  Resampling isoster's input with it moves the spiral inner zone by at most 0.05 sigma.
- Harmonics help both renderers by similar amounts.
- With harmonics off the gap stays (photutils fit: 1.819 painted, 2.568 isoster), so it
  is the pixel-to-ring assignment. On 20 spirals, 20 % of the inner-zone pixels were left
  more than 0.5 px from a consistent radius; they carried 99.7 % (median) of isoster's
  excess residual, while the other 80 % agreed with photutils to 0.001 sigma. The excess
  sits beyond 15 px, not at the centre (inside 5 px isoster was better).
- More fixed-point steps do not fix it: at 100 steps 10 % of the inner-zone pixels still do
  not converge, and the residual is unchanged (1.82 → 1.85 on the 20 spirals).

## Fix (1.2.1)

A consistent radius always exists below the last ring: g(s) = R(s) - s is positive at
s = 0 and not positive at s = sma_max for a pixel inside it. Pixels the fixed-point steps
leave inconsistent by more than 0.1 px are bisected on [0, sma_max] to 0.01 px. Every
other pixel is unchanged bit for bit (checked on eight SGA spirals: unsolved pixels are
4-22 % of each image, and no other pixel changes). Where rings cross, several radii are
consistent and bisection returns one; photutils' painting averages them.

Cost: the render takes 1.5-2.8 times as long on those eight spirals (median 2.0; e.g.
0.030 → 0.061 s on a 365² SGA image, 0.168 → 0.335 s on 811²). No published timing includes the renderer.

## Effect

SGA-2020, random spirals, inner zone (mean |data - model| / sigma, median of 60):

| | before | after | photutils' renderer |
|---|---:|---:|---:|
| photutils fit, harmonics on | 2.053 | 1.491 | 1.447 |
| photutils fit, harmonics off | 2.568 | 1.788 | 1.819 |
| isoster fit (`ref_default`), harmonics on | 1.929 | 1.275 | 1.447 (photutils fit) |

Mid zone of the isoster fit 0.803 → 0.773; ellipticals' inner zone 0.666 → 0.653.

Round-2 mocks, truth-relative RMS, every stored profile of six arms drawn by both solvers
on one pixel set (harmonics on, no S/N cut, log-linear interpolation; the linear one
gives similar numbers, e.g. S4G whole aperture 48 / 1.1, 1.155 → 0.981). Median ratio, percent of models better / worse by more than 2 %, and
median RMS before → after:

| sample, zone | better / worse | median RMS (%) |
|---|---|---|
| S4G, whole aperture | 49 / 0.8 | 1.125 → 0.936 |
| S4G, inner (< 0.5 Re) | 33 / 0.5 | 0.911 → 0.814 |
| S4G, mid (0.5-2 Re) | 43 / 1.7 | 1.065 → 0.902 |
| S4G, outer | 5 / 1.1 | 0.979 → 0.979 |
| Huang2013, whole aperture | 6 / 0.7 | 1.150 → 1.147 |
| Huang2013, outer | 13 / 3.4 | 5.068 → 4.996 |

The barred and twisting S4G disks gain most (`samp_cubic_ea` whole-aperture median ratio
0.863); the mostly smooth Huang2013 inputs barely move. Every harmonics-on and
harmonics-off render with twisting geometry changes, so the round-2 mock records and the
SGA-2020 residuals are rescored once, with the other pending renderer changes.

Outputs: `outputs/benchmark_exhausted/issue2_bisect_mocks_2026_10_21/paired_old_new.csv.gz`;
`/Volumes/galaxy/isophote/sga2020/output/campaign/bm1r2_phase4/issue2_render_decomposition{,_bisect}_2026_10_21/`.
