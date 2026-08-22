# Review dossier — Part A, three-way benchmark comparison

Date: 2026-08-22
Branch: `benchmarks/three-way-comparison`, 29 commits ahead of `main`, not
merged, no PR.
Anchor commit: `2d91217`. This is a **point-in-time snapshot**, not a live
document: the figures below were true of that commit and are not bound to the
archives by the prose gate, unlike those in the design spec. If the archives
have since moved, trust the gates and the spec over this file, and check
`git log benchmarks/harmonic_scale/`.

Purpose: **independent review before Part B begins.** Part A is complete and
its results are archived and gated. Part B (controlled three-way timing) has a
specification but no implementation, deliberately.

This document exists because the design spec
(`docs/specs/2026-08-22-three-way-benchmark-comparison-design.md`) is a
chronological record — pre-registration written before the measurements,
results appended after — and a reviewer should not have to reconstruct from it
what is worth attacking. Everything here points at that spec, the archives, or
the code; nothing is restated as fact that is not checkable from them.

---

## 1. What is being claimed

Part A asks one question: **do isoster, `photutils.isophote` and AutoProf
report harmonic deviation coefficients on the same scale?** If not, the
exhausted-benchmark campaign's cross-tool harmonic score compares different
quantities under one column name.

Five claims are made. Each is archived, and each is gated.

**C1 — The three tools agree on raw harmonic amplitudes.** Measured against
integrated analytic truth at matched apertures, agreement is 0.1–0.3% once the
tools sample comparably.

**C2 — AutoProf's apparent 13–25% harmonic excess is nearest-pixel sampling,
not a scale difference.** Controlled by `ap_iso_interpolate_start`. Claimed as
*causal*, on three independent grounds: `mode_matched_spread = 0` on both
campaigns; two different option pairs reaching the same sampling threshold give
**bit-identical** ratios; and holding `interpolate_start` fixed while changing
only `ap_set_psf` moves one ring from 16.3% to 0.56%. The aliasing lands in
`m = 4` specifically, which a square pixel grid's four-fold symmetry predicts.

**C3 — `photutils` does not converge at large radius**, running 1.7–2.4% flat
to rising where the other two fall below 0.2%. Attributed to order leakage: its
ring samples are not evenly spaced in polar angle (1.4× variation), so the
harmonic basis is not orthogonal there.

**C4 — AutoProf's eccentric-anomaly basis is unconvertible by a same-order
rotation.** 12% at `eps = 0.3`, 63–68% at `eps = 0.6`.

**C5 — Track 2 (Bender-normalized coefficients) is licensed, narrowly.**
Reconstructing AutoProf's radial gradient by a matched secant over isoster's
own interval reproduces isoster's gradient far better than a point derivative
does. Licensed **only on the reference configuration of both fixtures**.

A sixth finding is claimed as publishable in its own right: **Bender amplitudes
are convention-dependent at ~13% through the gradient definition alone**
(forward secant versus point derivative), reproducing on two galaxies of
different concentration — which makes it a property of the definition, not the
profile.

---

## 2. Where the evidence is

| Artifact | Path |
|---|---|
| Design spec, pre-registration and results | `docs/specs/2026-08-22-three-way-benchmark-comparison-design.md` |
| A3 archive, compact n=2 | `benchmarks/harmonic_scale/reference_harmonic_scale.json` |
| A3 archive, extended n=4 | `benchmarks/harmonic_scale/reference_harmonic_scale_n4.json` |
| Track 2 archive, n=2 | `benchmarks/harmonic_scale/reference_gradient_reconstruction_sersic_n2_compact.json` |
| Track 2 archive, n=4 | `benchmarks/harmonic_scale/reference_gradient_reconstruction_sersic_n4_extended.json` |
| Frozen tolerances | `benchmarks/harmonic_scale/frozen_tolerances*.json` (4 files) |
| A3 gate | `benchmarks/harmonic_scale/check_harmonic_scale.py` |
| Track 2 gate | `benchmarks/harmonic_scale/check_gradient_reconstruction.py` |
| Measurement runners | `benchmarks/harmonic_scale/run_harmonic_scale.py`, `run_gradient_reconstruction.py` |
| Tool adapters | `benchmarks/harmonic_scale/adapters.py`, `autoprof_worker.py`, `conventions.py` |
| CI wiring | `.github/workflows/docs.yml` |

Scale: 42 files changed against `main`. Roughly 4,300 lines of measurement and
gate code, ~58,000 lines of archived JSON, 158 tests in the six Part A test
files, within a suite of 1130.

### Verifying independently

```bash
uv run pytest tests/ -q
uv run python benchmarks/harmonic_scale/check_harmonic_scale.py
uv run python benchmarks/harmonic_scale/check_gradient_reconstruction.py
uv run python benchmarks/harmonic_scale/check_harmonic_scale.py --self-test
uv run python benchmarks/harmonic_scale/check_gradient_reconstruction.py --self-test
uv run mkdocs build --strict
```

The gates read only committed JSON and committed prose — **no AutoProf, no
fitting** — which is why they can run in CI, where AutoProf does not exist.
Re-*measuring* needs `~/.venvs/autoprof_venv` (Python 3.10, AutoProf 1.3.4), a
clean working tree and a quiet machine.

Expected: 1130 passed / 1 skipped / 5 deselected; all four gates green; both
self-tests green.

---

## 3. The procedure the claims rest on

Every archived number is judged against a **pre-registration**, not fitted
after the fact. A reviewer should test whether that is true in substance and
not only in form.

1. **Tolerances are frozen from a pilot run on one noise-seed block, committed,
   and only then is the validation run made on a disjoint block.** Seed blocks
   in use, all disjoint: 900000/20260822 (A3 n=2), 700000/30260822 (A3 n=4),
   500000/40260822 (T2 n=2), 300000/50260822 (T2 n=4).
2. **Acceptance criteria are threshold-free comparisons**, because a criterion
   checked against a tolerance derived from its own pilot value cannot fail.
   Criterion 1 weighs two candidate reconstructions against one target;
   criterion 2 weighs a normalized quantity against the unnormalized one it is
   built from.
3. **A fixture fingerprint** covers galaxy, radii, planted modes, gradient step
   and grid. A changed fingerprint is reported as a redefinition, not drift.
4. **A claim-definition fingerprint** covers the claims themselves — added this
   session, see §5.
5. **Archiving refuses a dirty working tree**, checked both when the run was
   made and when the archive is written.
6. **Prose is bound to the archive.** Any document sentence stating a guarded
   number must state the archived value. Whole clauses, never bare numbers;
   stems carry none of the value they guard; checks that fired nowhere are
   reported rather than counted as passes.
7. **Every gate has a `--self-test`** that corrupts the archive and requires
   the corruption to be caught. Currently 36/36 and 37/37 claims, 7/7 and 8/8
   A3 prose clauses, 12/12 Track 2 prose clauses.

---

## 4. Decisions that could have gone the other way

These are the review's highest-value targets: each was a judgment call, and
each is defensible but not forced.

**D1 — `ap_iso_interpolate_start` is a grid axis, not a setting.** It decides
whether the headline reads "0.1% agreement" or "13–25% excess". Crossing it
was the decision that turned an apparent scale disagreement into a sampling
explanation. *Attack:* is treating a tool's default configuration as one point
on an axis fair, or does it flatter AutoProf by reporting its best case?

**D2 — Track 2 matches isoster's convention, not the analytic truth.** isoster
divides by a forward secant sitting ~13% from the point derivative. Using an
*accurate* denominator would have manufactured a 13% cross-tool disagreement
out of a definitional difference. *Attack:* this makes Track 2 a test of
convention-matching rather than of correctness. Is that the right question?

**D3 — Licensing is per regime, and the hard regimes are excluded.**
`eps_high` and `noise_snr30` pass criterion 2 on the n=4 galaxy and fail on the
n=2 one. A verdict that flips with the choice of galaxy was judged not to be a
licence, so those regimes are excluded rather than reported as marginal.
*Attack:* two galaxies is a very small basis for declaring a regime
irreproducible. The opposite reading — that criterion 2 is too crude at these
magnitudes — is also available.

**D4 — Criterion 2's two sides are reduced separately, not paired.** It
compares `max(bender)` against `max(raw) + gradient`, where the maxima may
come from different rings. The *paired* form (ring-by-ring, order-by-order)
was measured and **fails everywhere, including the reference cases, by
numerical margins** (+0.0003, +0.0001). That is recorded, and the
pre-registered unpaired form was kept rather than redefined after seeing the
data. *Attack:* an unpaired budget comparison is statistically loose. Keeping
it was the conservative choice with respect to pre-registration, but not with
respect to rigour.

**D5 — The mask axis was dropped.** Reviving it would mean composing a mask
step into AutoProf's forced pipeline — that is, modifying the tool being
benchmarked. Left as an open fairness call for the user, not decided.

**D6 — Track 2's licence does not extend to the exhausted-benchmark
campaign.** Track 2 differences `b0` between a ring and its partner at
`sma·1.1`, obtained from a fixed aperture; the campaign's AutoProf arm is a
free fit, so in general no such ring exists. Interpolating `b0` would put an
unmeasured quantity in the denominator of every harmonic. A5 keeps its NaN.

---

## 5. What changed in the most recent session, and why it matters to a reviewer

One gate failed reproducibility and was fixed. The fix is the part most worth
checking, because it touched a claim *definition*.

**The failure.** `gradient_agreement_pct_noise_snr100` on the n=4 fixture moved
0.142 between seed blocks against a 0.118 tolerance. The claim was a **max over
five rings** of a median over realizations; its tolerance was three times the
largest **single-ring** standard error. That proxy is exact only when the claim
*is* one ring's median. A max additionally varies through which ring wins, so
its true spread is strictly larger. The gate failed a validation run whose
measurement was fine. This was the fourth instance of one root cause.

**What was measured before choosing a fix** (from the archived pilot and
validation runs, no re-measurement):

| reduction | pilot | validation | moved | bootstrap tolerance |
|---|---|---|---|---|
| max over rings | 0.197 | 0.339 | 0.142 | 0.195 |
| median over rings | 0.110 | 0.114 | 0.0036 | 0.0655 |

The median is ~40× more reproducible. **But a blanket switch to the median
flips criterion 2 on the n=2 reference case** (Bender 0.116 against a 0.093
budget) and would have unlicensed Track 2 on that fixture.

**The resolution.** Every claim is now archived under **both** reductions —
`worst_ring_*` (max) and `typical_ring_*` (median) — and **the licensing
criteria read the pre-registered `worst_ring` family and nothing else.**
Choosing a reduction after seeing which verdict it produces is what
pre-registration forbids. The typical-ring family carries no verdict; it is
what constrains the gate.

**And every tolerance is now measured, not estimated**: each is derived by
bootstrapping that claim's own definition — the same reduction over the same
columns — across resampled realizations of the pilot. Because a tolerance is
now tied to a definition, the frozen file carries a `claims_fingerprint` and
the gate fails when the definitions move. Both failure paths were exercised.

**No archive was re-measured, and this is a deliberate call a reviewer should
test.** The archived `summary` blocks are untouched; claims are recomputed from
them at check time, so pilot and validation remain judged under one definition
— which is the purpose of re-validating. A one-case end-to-end run reproduced
the archived reference case exactly. *Attack:* the instruction given was
"re-freeze and re-validate", and only the first half was done. If the reviewer
judges a full re-archive necessary, it is ~6 minutes per fixture and the tree
must be committed between the two.

---

## 6. Known limitations, stated rather than found

- **Two galaxies is not a survey.** Compact n=2 (R_e = 25, 241 px) and
  extended n=4 (R_e = 40, 321 px).
- **Neither fixture is PSF-convolved.** AutoProf does not measure the PSF
  either — its `psf` step is `PSF_Assumed`, hardcoded at 4.0 px — so
  `ap_set_psf` is the only independent route to the sampling threshold.
- **The reference configuration is noiseless.** Noise enters only as
  one-factor cases.
- **Five Track 2 claims on the n=2 fixture and four on the n=4 fixture carry
  a measured tolerance larger than half their own value** and so constrain
  little; the gate names them individually. All are noise-bearing claims. This
  is reported rather than hidden, but it means those claims are weak evidence.
  The gate reports them separately from claims that merely sit under the
  deterministic floor, which reproduce exactly and are not weak in the same
  sense.
- **A3 covers the polar-resampled path only.** The eccentric-anomaly path is
  measured (C4) but never converted.
- **26 pre-existing `I001` import-order errors in `examples/`**, untouched by
  this work. `examples/**` ignores E/F/W but not I, and pre-commit only sees
  changed files, so they have never surfaced.

---

## 7. Specific questions for the reviewer

1. Is the causal claim in **C2** carried by its evidence, or is it three
   correlational observations presented as a mechanism?
2. Is **D3** — excluding a regime because two galaxies disagree — the right
   call, or is it discarding a real signal about criterion 2's crudeness?
3. Is **D4** defensible? Keeping a loose unpaired comparison because it was
   pre-registered trades rigour for procedure. Which should win here?
4. Does the **`worst_ring` / `typical_ring` split** (§5) genuinely avoid
   choosing a statistic to obtain a verdict, or does gating both simply
   postpone the choice?
5. Is **not re-archiving** (§5) sound, given the claim definitions changed?
6. Are the **acceptance criteria able to fail in practice**, not only in
   principle? Criterion 1 passes by 101× and 162× against a bar of 10.
7. **Part B specification** — section "B2–B4 settled with the reviewer" in the
   design spec. Timing comparisons are where benchmarks most easily become
   unfair. Are the five decisions there sufficient, and is the accuracy gate
   (Part A's fixtures and truth, bar frozen from a pilot) the right instrument?

---

## 8. What Part B will do, and what is not yet decided

Specified, not implemented. Five decisions taken with the reviewer before any
code: the Python 3.10-versus-3.12 interpreter gap gets its own calibration line
and is never subtracted; the accuracy gate is Part A's planted fixtures and
truth; AutoProf's harness cost (subprocess, FITS round-trip, IPC) is timed
apart from the fit; few fixtures with many repetitions, so numbers carry
uncertainty intervals; harmonics are a grid axis.

Still to be fixed by measurement, frozen from a pilot first: the accuracy bar
per tool and fixture; session and repetition counts; a **dispersion abort
rule** — a laptop under thermal load produces numbers that look like
measurements — and success/failure accounting with partial-profile treatment.
