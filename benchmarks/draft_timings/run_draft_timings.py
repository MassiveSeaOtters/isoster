"""Reproduce every timing quoted in the publication technical draft.

Each block backs one table or number in ``docs/publication/draft``. The point is
that a reader on different hardware can re-run this and get their own numbers
rather than trusting ours: absolute times are machine-specific, so the draft
quotes ratios wherever a ratio is what the argument needs, and this script
reports a spread alongside every median so a reader can see when a ratio is too
noisy to lean on.

Measurement protocol
--------------------
Sessions are separate *processes*: ``--sessions N`` fans out N interpreters,
each with its own arm-ordering seed, because repeating the blocks inside one
process would share the Python imports, module state, warmed allocators and arm
ordering that make two measurements agree. Note what this does *not* isolate:
the operating system's filesystem page cache is machine-wide, as is the
machine's thermal and load state, so sessions are independent of each other's
*process* state only. Within a session, short operations are **batched** to a
floor duration before timing, because a
sub-millisecond call timed one at a time is dominated by clock and dispatch
overhead. Repetitions of the configurations being compared are **interleaved**
rather than run in blocks, so that a machine that warms up or throttles part-way
through does not bias one arm relative to another. Every raw repetition is
retained in the JSON output, and the median is reported with an interquartile
range.

Blocks
------
``lazy_gradient``
    Section 1.3.2: gradient-evaluation count, wall time and the geometry
    difference between the cached and uncached paths, in units of each
    isophote's own reported uncertainty.
``snr_sweep``
    Section 1.3.2: the same cached-versus-uncached geometry difference at three
    noise levels, which is what the accuracy table in that section quotes. The
    quantity is a difference between two converged fits rather than a time, so
    each session draws its own noise realisation --- see ``bench_snr_sweep``.
``ea_isofit``
    Section 1.4.1.4: the 2x2 of eccentric-anomaly sampling and ISOFIT.
``joint_solve_bands``
    Section 1.6.5: one joint multi-band solve against band count, for **both**
    parameterisations (the supported geometry-parameterised default and the
    legacy shared-amplitude form) and in both OLS and WLS.
``maxsma``
    Section 1.6.5: whole-fit cost against ``maxsma``, with the ring count, the
    *instrumented* sampling-call and sampled-point totals, and a stop-code and
    iteration summary. The last is essential: a fit whose outer rings fail after
    two iterations is cheap for the wrong reason, and reading its wall time as
    evidence about sampling cost would be an error.
``cold_start``
    Sections 1.3.2 and 1.6.5: package import, first dispatch against a *warm*
    Numba cache, and first dispatch against an explicitly **empty** cache, which
    is the only one of the three that measures compilation.
``numba_fallback``
    Section 1.6.5: the steady-state cost of the pure-NumPy fallback against the
    compiled kernels. Combined with ``cold_start``'s compilation figure this
    gives the fit count at which compiling starts to pay, which is what the
    draft's advice about the fallback turns on.

Usage::

    uv run python benchmarks/draft_timings/run_draft_timings.py
    uv run python benchmarks/draft_timings/run_draft_timings.py --only maxsma

Results go to ``outputs/benchmark_draft_timings/timings.json`` together with the
environment needed to interpret them.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isoster import fit_image  # noqa: E402
from isoster.config import IsosterConfig  # noqa: E402
from isoster.output_paths import resolve_output_directory  # noqa: E402

# Two fixtures, because one cannot serve both purposes. The compact fixture is
# the reference for the whole-fit comparisons. The wide fixture exists solely so
# that the maxsma block can reach 400 px with every ring still converging --- on
# the compact fixture the outer rings fail with stop_code=-1 after two or three
# iterations, which makes the fit cheap for a reason that has nothing to do with
# the scaling question being asked.
FIXTURE_COMPACT = dict(R_e=30.0, n=2.0, I_e=100.0, eps=0.4, pa=0.6, noise_snr=50.0, seed=7)
FIXTURE_WIDE = dict(R_e=80.0, n=2.0, I_e=100.0, eps=0.4, pa=0.6, noise_snr=200.0, seed=7)

# Noise levels for the accuracy table in section 1.3.2. The middle one is the
# compact fixture's own S/N, so the sweep's centre row and the ``lazy_gradient``
# block describe the same galaxy at the same noise level -- though not the same
# noise *realisation*, since the sweep redraws one per session (see
# ``bench_snr_sweep``) and ``lazy_gradient`` keeps the fixture's fixed seed.
SNR_SWEEP_LEVELS = (200.0, 50.0, 10.0)

# Outer limit for the cached-versus-uncached comparison, shared by the
# ``lazy_gradient`` and ``snr_sweep`` blocks so their S/N = 50 numbers refer to
# the same fit rather than to two fits that merely resemble each other.
LAZY_PAIR_MAXSMA = 120.0

REPEATS = 9
MIN_BATCH_SECONDS = 0.02
ORDER_SEED = 20260816
COLD_START_PAIRS = 3

# The committed archive backs the manuscript's numbers, so promoting a run to
# it is a deliberate act. A partial or single-session run must not be able to
# replace it by accident.
ARCHIVE_MIN_SESSIONS = 18

# Set once from ``--seed``. A session's arm-visiting order derives from it, so
# two sessions sharing a seed visit the arms identically and cannot reveal
# order-dependent variability.
ACTIVE_SEED = ORDER_SEED


def _summarize(samples_ms: Sequence[float]) -> Dict[str, object]:
    """Median with an interquartile range, plus the raw repetitions."""
    ordered = sorted(samples_ms)
    median = statistics.median(ordered)
    if len(ordered) >= 4:
        q1, q3 = statistics.quantiles(ordered, n=4)[0], statistics.quantiles(ordered, n=4)[2]
    else:
        q1, q3 = ordered[0], ordered[-1]
    return {
        "median_ms": round(median, 4),
        "iqr_ms": [round(q1, 4), round(q3, 4)],
        "spread_pct": round(100.0 * (q3 - q1) / median, 1) if median else None,
        "raw_ms": [round(value, 4) for value in ordered],
    }


def _batch_size(call: Callable[[], object]) -> int:
    """How many calls to run per timed batch so the batch clears the clock.

    Timing a sub-millisecond call individually measures the timer as much as the
    work; this grows the batch until it takes at least ``MIN_BATCH_SECONDS``.
    """
    call()
    start = time.perf_counter()
    call()
    single = time.perf_counter() - start
    if single <= 0:
        return 1000
    return max(1, int(MIN_BATCH_SECONDS / single) + 1)


def _time_interleaved(
    calls: Dict[str, Callable[[], object]],
    repeats: int = REPEATS,
    seed: int | None = None,
) -> Dict[str, object]:
    """Time several configurations, interleaving *and reordering* repetitions.

    Running arm A's repetitions and then arm B's would let a thermal or
    background-load trend masquerade as a difference between the arms.
    Interleaving alone is not enough either: if every repetition visits the arms
    in the same order, whichever arm always runs last absorbs any
    within-repetition drift. That biases exactly the comparisons this script
    exists to make, since the arms sort by band count and the largest always
    ran last. Each repetition therefore visits the arms in a freshly shuffled
    order, drawn from a seeded generator so the sequence is reproducible.
    """
    labels = list(calls)
    batches = {label: _batch_size(calls[label]) for label in labels}
    samples: Dict[str, List[float]] = {label: [] for label in labels}
    rng = np.random.default_rng(ACTIVE_SEED if seed is None else seed)

    for _ in range(repeats):
        for label in rng.permutation(labels):
            call, batch = calls[label], batches[label]
            start = time.perf_counter()
            for _ in range(batch):
                call()
            elapsed = time.perf_counter() - start
            samples[label].append(elapsed / batch * 1e3)

    out: Dict[str, object] = {}
    for label in labels:
        entry = _summarize(samples[label])
        entry["calls_per_batch"] = batches[label]
        out[label] = entry
    return out


def _ratio(numerator: Dict[str, object], denominator: Dict[str, object]) -> Dict[str, object]:
    """Ratio of medians, with the range implied by the two interquartile ranges."""
    num_med = float(numerator["median_ms"])  # type: ignore[arg-type]
    den_med = float(denominator["median_ms"])  # type: ignore[arg-type]
    num_iqr = numerator["iqr_ms"]  # type: ignore[index]
    den_iqr = denominator["iqr_ms"]  # type: ignore[index]
    return {
        "median": round(num_med / den_med, 3),
        "plausible_range": [
            round(float(num_iqr[0]) / float(den_iqr[1]), 3),
            round(float(num_iqr[1]) / float(den_iqr[0]), 3),
        ],
    }


def _build_fixture(spec: Dict[str, object]):
    from tests.fixtures.sersic_factory import create_sersic_model

    image, _, _ = create_sersic_model(**spec)  # type: ignore[arg-type]
    return image, float(image.shape[0] // 2)


def _config(centre: float, **overrides) -> IsosterConfig:
    base = dict(
        sma0=10.0,
        x0=centre,
        y0=centre,
        eps=0.3,
        pa=0.5,
        compute_deviations=True,
        harmonic_orders=[3, 4],
    )
    base.update(overrides)
    return IsosterConfig(**base)


def _lazy_pair_configs(centre: float) -> Dict[str, IsosterConfig]:
    """The cached and uncached configurations that §1.3.2 compares."""
    return {
        "lazy": _config(centre, maxsma=LAZY_PAIR_MAXSMA, use_lazy_gradient=True),
        "classical": _config(centre, maxsma=LAZY_PAIR_MAXSMA, use_lazy_gradient=False),
    }


def _pa_difference(lazy_pa: float, exact_pa: float) -> float:
    """Absolute position-angle difference, taken modulo pi.

    An ellipse's position angle is defined only up to a half turn, so a raw
    subtraction can report a difference of nearly pi for two orientations that
    are in fact identical. This folds the difference into [-pi/2, pi/2] before
    taking its size.
    """
    delta = (lazy_pa - exact_pa + np.pi / 2.0) % np.pi - np.pi / 2.0
    return abs(delta)


def _geometry_difference(lazy_profile, classical_profile) -> Dict[str, object]:
    """Summarise how far the cached-gradient geometry lands from the exact one.

    Returns the four quantities the §1.3.2 accuracy table quotes: the median and
    the maximum disagreement in units of each isophote's *own* reported
    uncertainty --- pooled over ``x0``, ``y0``, ``eps`` and PA, because the
    normalisation makes them commensurable --- plus the largest absolute centre
    and position-angle shifts, which are the two a reader can picture directly.
    """
    # The two profiles are compared ring by ring, so they have to *be* the same
    # rings. Nothing guarantees that: the two modes run independent fits, and a
    # divergent stopping point would leave ``zip`` silently truncating one
    # profile and pairing rings at different radii from there on. That failure
    # would look like a large approximation error rather than like a bug, so it
    # is checked rather than assumed.
    if len(lazy_profile) != len(classical_profile):
        raise RuntimeError(
            f"cached and uncached fits returned different ring counts "
            f"({len(lazy_profile)} vs {len(classical_profile)}); the geometry "
            "comparison assumes one shared ring sequence"
        )

    in_sigma: List[float] = []
    centre_shift: List[float] = []
    pa_shift: List[float] = []

    for index, (lazy_iso, exact_iso) in enumerate(zip(lazy_profile, classical_profile)):
        lazy_sma = float(lazy_iso.get("sma") or 0.0)
        exact_sma = float(exact_iso.get("sma") or 0.0)
        # The SMA grid is set before fitting and is not one of the fitted
        # parameters, so the two modes should agree on it exactly. A loose
        # tolerance still catches a misalignment without asserting bit equality
        # of a value neither mode computes.
        if abs(lazy_sma - exact_sma) > 1e-9 * max(1.0, abs(exact_sma)):
            raise RuntimeError(
                f"cached and uncached fits disagree on ring {index}'s semi-major "
                f"axis ({lazy_sma} vs {exact_sma}); the comparison would be "
                "between different radii"
            )
        if not lazy_sma:
            continue
        for key, err_key in (("x0", "x0_err"), ("y0", "y0_err"), ("eps", "eps_err"), ("pa", "pa_err")):
            if key == "pa":
                delta = _pa_difference(float(lazy_iso[key]), float(exact_iso[key]))
                pa_shift.append(delta)
            else:
                delta = abs(float(lazy_iso[key]) - float(exact_iso[key]))
                if key == "x0":
                    centre_shift.append(delta)
            sigma = float(exact_iso.get(err_key) or 0.0)
            if sigma > 0:
                in_sigma.append(delta / sigma)

    return {
        "median_abs_delta_sigma": float(np.median(in_sigma)),
        "max_abs_delta_sigma": float(np.max(in_sigma)),
        "max_abs_dx0_px": float(np.max(centre_shift)),
        "max_abs_dpa_rad": float(np.max(pa_shift)),
        "n_comparisons": len(in_sigma),
        "n_isophotes": len(centre_shift),
    }


def bench_lazy_gradient() -> Dict[str, object]:
    """Section 1.3.2: what the gradient cache saves, and what it costs."""
    import isoster.fitting as fitting_module

    image, centre = _build_fixture(FIXTURE_COMPACT)
    configs = _lazy_pair_configs(centre)

    original = fitting_module.compute_gradient
    counter = {"n": 0}

    def counted(*args, **kwargs):
        counter["n"] += 1
        return original(*args, **kwargs)

    out: Dict[str, object] = {}
    profiles = {}
    for label, cfg in configs.items():
        fitting_module.compute_gradient = counted
        counter["n"] = 0
        profiles[label] = fit_image(image, config=cfg)["isophotes"]
        gradient_calls = counter["n"]
        fitting_module.compute_gradient = original
        out[label] = {"gradient_evaluations": gradient_calls}

    timings = _time_interleaved({label: (lambda c=cfg: fit_image(image, config=c)) for label, cfg in configs.items()})
    for label in configs:
        out[label].update(timings[label])  # type: ignore[union-attr]
    # A ratio of runtimes, not a fractional saving: 0.75 means the lazy path
    # takes three-quarters of the time, i.e. saves a quarter.
    out["runtime_ratio_vs_classical"] = _ratio(timings["lazy"], timings["classical"])
    out["geometry_difference"] = _geometry_difference(profiles["lazy"], profiles["classical"])
    return out


def bench_snr_sweep() -> Dict[str, object]:
    """Section 1.3.2: the cached-gradient accuracy cost against noise level.

    The same compact fixture and the same pair of configurations as
    ``lazy_gradient``, at three signal-to-noise ratios, because the size of the
    approximation is the thing that depends on noise: a cached gradient is least
    representative exactly where the gradient is poorly measured.

    Unlike every other block here the measured quantity is not a time but the
    difference between two converged fits, which is fully determined by the
    image. Repeating it across sessions with one fixed noise realisation would
    therefore return the identical number eighteen times and report an
    interquartile range of exactly zero --- a spurious claim of precision. Each
    session instead draws its own noise realisation from its session seed, so
    the across-session spread is a spread over noise realisations, which is the
    variability a reader of the table actually needs to know about.
    """
    out: Dict[str, object] = {"noise_seed": ACTIVE_SEED, "levels": list(SNR_SWEEP_LEVELS)}
    for snr in SNR_SWEEP_LEVELS:
        image, centre = _build_fixture(dict(FIXTURE_COMPACT, noise_snr=snr, seed=ACTIVE_SEED))
        profiles = {
            label: fit_image(image, config=cfg)["isophotes"] for label, cfg in _lazy_pair_configs(centre).items()
        }
        out[f"snr={snr:g}"] = _geometry_difference(profiles["lazy"], profiles["classical"])
    return out


def bench_ea_isofit() -> Dict[str, object]:
    """Section 1.4.1.4: eccentric anomaly and ISOFIT are separable costs."""
    image, centre = _build_fixture(FIXTURE_COMPACT)
    configs = {
        "default_phi_posthoc": _config(centre, maxsma=120.0),
        "eccentric_anomaly_only": _config(centre, maxsma=120.0, use_eccentric_anomaly=True),
        "isofit_in_loop_only": _config(centre, maxsma=120.0, simultaneous_harmonics=True, isofit_mode="in_loop"),
        "eccentric_anomaly_plus_isofit": _config(
            centre, maxsma=120.0, use_eccentric_anomaly=True, simultaneous_harmonics=True, isofit_mode="in_loop"
        ),
    }
    timings = _time_interleaved({label: (lambda c=cfg: fit_image(image, config=c)) for label, cfg in configs.items()})
    baseline = timings["default_phi_posthoc"]
    out: Dict[str, object] = {}
    for label, entry in timings.items():
        entry = dict(entry)
        entry["ratio_to_default"] = _ratio(timings[label], baseline)
        out[label] = entry
    return out


def bench_joint_solve_bands() -> Dict[str, object]:
    """Section 1.6.5: one joint solve against band count.

    Both parameterisations are timed. The geometry-parameterised form is the
    supported default (``geometry_parameterized_solve=True``); the shared-
    amplitude form is what the default falls back to on an isophote's first
    iteration, and what ``False`` restores. Benchmarking only the latter --- as
    the first version of this script did --- would describe a path most users
    never take after iteration zero.
    """
    from isoster.multiband.fitting_mb import fit_first_and_second_harmonics_joint

    n_samples = 700
    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    rng = np.random.default_rng(0)

    out: Dict[str, object] = {"n_samples_per_band": n_samples}
    for mode in ("ols", "wls"):
        calls: Dict[str, Callable[[], object]] = {}
        for n_bands in (3, 6, 12, 24):
            intens = rng.normal(size=(n_bands, n_samples))
            weights = np.ones(n_bands)
            gradients = np.full(n_bands, -1.5)
            variances = np.full((n_bands, n_samples), 0.25) if mode == "wls" else None
            calls[f"amplitude_B={n_bands}"] = lambda a=angles, i=intens, w=weights, v=variances: (
                fit_first_and_second_harmonics_joint(a, i, w, v)
            )
            # The production entry point with per-band gradients supplied --
            # the call the fitting loop makes from an isophote's second
            # iteration onward. Timing the ``_geometry`` helper directly would
            # measure a function only the tests reach.
            calls[f"geometry_B={n_bands}"] = lambda a=angles, i=intens, w=weights, g=gradients, v=variances: (
                fit_first_and_second_harmonics_joint(a, i, w, v, per_band_gradients=g)
            )
        timings = _time_interleaved(calls, repeats=7)

        block: Dict[str, object] = {}
        for form in ("amplitude", "geometry"):
            previous = None
            for n_bands in (3, 6, 12, 24):
                entry = dict(timings[f"{form}_B={n_bands}"])
                if previous is not None:
                    entry["ratio_to_previous"] = _ratio(timings[f"{form}_B={n_bands}"], previous)
                block[f"{form}_B={n_bands}"] = entry
                previous = timings[f"{form}_B={n_bands}"]
        out[mode] = block
    return out


def bench_maxsma() -> Dict[str, object]:
    """Section 1.6.5: rings grow logarithmically; sampling work does not.

    Uses the wide fixture so that every ring out to 400 px still converges. The
    stop-code and iteration summary is reported alongside, because a block of
    outer rings failing early would make the fit cheap for a reason unrelated to
    the scaling question. The sampling totals are *instrumented* --- counted from
    the actual calls, which include the extra passes for gradient evaluation and
    every iteration --- rather than inferred from one nominal pass per output
    ring, which undercounts by a large factor.
    """
    import isoster.fitting as fitting_module

    image, centre = _build_fixture(FIXTURE_WIDE)
    configs = {f"maxsma={value:g}": _config(centre, maxsma=value) for value in (100.0, 200.0, 400.0)}

    original = fitting_module.extract_isophote_data
    tally = {"calls": 0, "points": 0}

    def counted(*args, **kwargs):
        data = original(*args, **kwargs)
        tally["calls"] += 1
        tally["points"] += int(np.size(data.intens))
        return data

    out: Dict[str, object] = {"fixture": "wide"}
    for label, cfg in configs.items():
        fitting_module.extract_isophote_data = counted
        tally["calls"] = tally["points"] = 0
        result = fit_image(image, config=cfg)
        sampling = dict(tally)
        fitting_module.extract_isophote_data = original

        isophotes = [iso for iso in result["isophotes"] if iso.get("sma")]
        iterations = [int(iso.get("niter") or 0) for iso in isophotes]
        out[label] = {
            "n_isophotes": len(isophotes),
            "stop_codes": dict(Counter(int(iso["stop_code"]) for iso in isophotes)),
            "median_iterations": statistics.median(iterations) if iterations else None,
            "sampling_calls": sampling["calls"],
            "sampled_points": sampling["points"],
        }

    timings = _time_interleaved({label: (lambda c=cfg: fit_image(image, config=c)) for label, cfg in configs.items()})
    for label in configs:
        out[label].update(timings[label])  # type: ignore[union-attr]
    return out


_COLD_START_CHILD = r"""
import json, sys, time
sys.path.insert(0, {repo!r})
t0 = time.perf_counter()
import numpy as np
from isoster import fit_image
from isoster.config import IsosterConfig
import_s = time.perf_counter() - t0

rng = np.random.default_rng(0)
yy, xx = np.indices((200, 200))
image = 100.0 * np.exp(-np.hypot(yy - 100, xx - 100) / 30.0) + rng.normal(0, 0.1, (200, 200))
cfg = IsosterConfig(sma0=10.0, maxsma=60.0, x0=100.0, y0=100.0)

t = time.perf_counter(); fit_image(image, config=cfg); first_s = time.perf_counter() - t
t = time.perf_counter(); fit_image(image, config=cfg); second_s = time.perf_counter() - t
print(json.dumps({{"import_s": import_s, "first_fit_s": first_s, "second_fit_s": second_s}}))
"""


def bench_cold_start() -> Dict[str, object]:
    """Sections 1.3.2 / 1.6.5: import, cache load, and actual compilation.

    Three different costs get conflated as "the JIT cost", and they call for
    different responses. Running the child process against an empty
    ``NUMBA_CACHE_DIR`` is the only way to observe genuine compilation: with a
    populated cache, first dispatch merely loads objects from disk, which is
    much cheaper and is what a user sees on every run *after* the first.

    Unlike the in-process blocks this one cannot be batched — each observation
    costs a whole interpreter start — so it takes independent empty/populated
    *pairs* instead, and reports the same median-and-IQR summary over them.
    ``COLD_START_PAIRS`` is small, so treat these as one-significant-figure
    quantities.
    """
    child = _COLD_START_CHILD.format(repo=str(REPO_ROOT))

    def run(cache_dir: str) -> Dict[str, float]:
        env = dict(os.environ)
        env["NUMBA_CACHE_DIR"] = cache_dir
        proc = subprocess.run(
            [sys.executable, "-c", child], capture_output=True, text=True, cwd=str(REPO_ROOT), env=env
        )
        if proc.returncode != 0:
            raise RuntimeError(f"cold-start child failed: {proc.stderr[-2000:]}")
        return json.loads(proc.stdout.strip().splitlines()[-1])

    imports_ms: List[float] = []
    compile_ms: List[float] = []
    cache_load_ms: List[float] = []
    pairs: List[Dict[str, object]] = []

    for _ in range(COLD_START_PAIRS):
        with tempfile.TemporaryDirectory() as cache_dir:
            fresh = run(cache_dir)  # empty cache: compiles
            warm = run(cache_dir)  # same dir, now populated: loads
        pairs.append({"fresh": fresh, "warm": warm})
        imports_ms.append(warm["import_s"] * 1e3)
        compile_ms.append((fresh["first_fit_s"] - warm["first_fit_s"]) * 1e3)
        cache_load_ms.append((warm["first_fit_s"] - warm["second_fit_s"]) * 1e3)

    return {
        "import": _summarize(imports_ms),
        "compilation": _summarize(compile_ms),
        "cache_load": _summarize(cache_load_ms),
        "pairs": pairs,
        "note": (
            "compilation is the extra first-fit cost against an empty cache; "
            "cache_load is the extra first-fit cost once the cache exists. Only "
            "the former is compilation, and it recurs whenever the cache is "
            "invalidated (a Numba or ISOSTER upgrade), not only at first install."
        ),
    }


_NUMBA_FALLBACK_CHILD = r"""
import json, statistics, sys, time
sys.path.insert(0, {repo!r})
import numpy as np
from isoster import fit_image
from isoster.config import IsosterConfig
from isoster.numba_kernels import check_numba_available

rng = np.random.default_rng(0)
yy, xx = np.indices((200, 200))
image = 100.0 * np.exp(-np.hypot(yy - 100, xx - 100) / 30.0) + rng.normal(0, 0.1, (200, 200))
cfg = IsosterConfig(sma0=10.0, maxsma=60.0, x0=100.0, y0=100.0)

fit_image(image, config=cfg)  # warm up: compile or first-touch, not measured
samples = []
for _ in range({repeats}):
    t = time.perf_counter(); fit_image(image, config=cfg); samples.append(time.perf_counter() - t)
print(json.dumps({{"numba_available": bool(check_numba_available()),
                  "median_fit_s": statistics.median(samples),
                  "raw_s": samples}}))
"""


def bench_numba_fallback() -> Dict[str, object]:
    """Section 1.6.5: when the pure-NumPy fallback is the cheaper choice.

    Every Numba kernel has a NumPy fallback, reachable by setting
    ``NUMBA_DISABLE_JIT=1``. It skips compilation but runs the hot kernels
    slower, so which is cheaper depends on how many fits the process performs.
    The draft advises on that trade-off, and the advice needs the break-even
    point rather than an intuition about it.

    Both arms are *steady-state* medians, taken after a warm-up fit, so this
    measures the ongoing penalty rather than the start-up cost --- the start-up
    cost is what ``cold_start`` measures, and the break-even below combines the
    two. Both run in child processes because the fallback is selected from the
    environment at import time.

    ``numba_available`` is reported for each arm and checked: if the fallback
    arm were to keep using the compiled kernels the penalty would come out at
    zero, and a zero penalty is indistinguishable from a fast fallback.
    """
    repeats = 7

    def run(disable_jit: bool) -> Dict[str, object]:
        env = dict(os.environ)
        env["NUMBA_DISABLE_JIT"] = "1" if disable_jit else "0"
        child = _NUMBA_FALLBACK_CHILD.format(repo=str(REPO_ROOT), repeats=repeats)
        proc = subprocess.run(
            [sys.executable, "-c", child], capture_output=True, text=True, cwd=str(REPO_ROOT), env=env
        )
        if proc.returncode != 0:
            raise RuntimeError(f"numba-fallback child failed: {proc.stderr[-2000:]}")
        return json.loads(proc.stdout.strip().splitlines()[-1])

    compiled = run(disable_jit=False)
    fallback = run(disable_jit=True)

    if not compiled["numba_available"]:
        raise RuntimeError("the compiled arm reported Numba unavailable; there is nothing to compare against")
    if fallback["numba_available"]:
        raise RuntimeError("NUMBA_DISABLE_JIT=1 did not select the NumPy fallback; the penalty would read as zero")

    compiled_s = float(compiled["median_fit_s"])
    fallback_s = float(fallback["median_fit_s"])
    penalty_s = fallback_s - compiled_s

    return {
        "compiled_fit_s": round(compiled_s, 6),
        "fallback_fit_s": round(fallback_s, 6),
        "fallback_penalty_s": round(penalty_s, 6),
        "fallback_slowdown": round(fallback_s / compiled_s, 3) if compiled_s else None,
        "repeats": repeats,
        "raw": {"compiled_s": compiled["raw_s"], "fallback_s": fallback["raw_s"]},
        "note": (
            "steady-state medians after a warm-up fit, on the same 200x200 "
            "fixture the cold_start block uses. The break-even fit count is "
            "derived across sessions, since it divides this penalty by the "
            "separately measured compilation cost."
        ),
    }


BLOCKS: Dict[str, Callable[[], Dict[str, object]]] = {
    "lazy_gradient": bench_lazy_gradient,
    "snr_sweep": bench_snr_sweep,
    "ea_isofit": bench_ea_isofit,
    "joint_solve_bands": bench_joint_solve_bands,
    "maxsma": bench_maxsma,
    "cold_start": bench_cold_start,
    "numba_fallback": bench_numba_fallback,
}


def _environment() -> Dict[str, object]:
    try:
        import numba

        numba_version = numba.__version__
    except ImportError:
        numba_version = None

    try:
        config = np.__config__.show(mode="dicts")  # type: ignore[call-arg]
        blas = {name: info.get("name") for name, info in config.get("Build Dependencies", {}).items()}
    except Exception:
        blas = {"note": "numpy build configuration unavailable"}

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "numba": numba_version,
        "blas": blas,
        "repeats": REPEATS,
        "min_batch_seconds": MIN_BATCH_SECONDS,
        "fixture_compact": FIXTURE_COMPACT,
        "fixture_wide": FIXTURE_WIDE,
    }


def _collect_ratio(sessions: List[Dict[str, object]], path: Sequence[str]) -> Dict[str, object] | None:
    """Pull one ratio's median out of every session and summarise the spread."""
    values: List[float] = []
    for session in sessions:
        node: object = session
        for key in path:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if isinstance(node, (int, float)):
            values.append(float(node))
    if len(values) < 2:
        return None
    ordered = sorted(values)
    if len(ordered) >= 4:
        quartiles = statistics.quantiles(ordered, n=4)
        q1, q3 = quartiles[0], quartiles[2]
    else:
        q1, q3 = ordered[0], ordered[-1]
    return {
        "n_sessions": len(ordered),
        "min": round(ordered[0], 4),
        "q1": round(q1, 4),
        "median": round(statistics.median(ordered), 4),
        "q3": round(q3, 4),
        "max": round(ordered[-1], 4),
        "values": [round(v, 4) for v in ordered],
    }


def _run_session_subprocess(seed: int, selected: Sequence[str]) -> Dict[str, object]:
    """Run one session in a fresh interpreter, with its own arm ordering.

    Repeating the blocks inside one process would share the imported modules,
    the module-level state and every warmed-up allocator, which is much of what
    makes two runs agree. This does not buy a cold *page* cache — that is
    machine-wide — nor a reset thermal state, so the isolation is of process
    state only.
    """
    argv = [sys.executable, str(Path(__file__).resolve()), "--json-only", "--seed", str(seed)]
    if len(selected) == 1:
        argv += ["--only", selected[0]]
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"session (seed={seed}) failed: {proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    global ACTIVE_SEED

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(BLOCKS), help="Run a single block.")
    parser.add_argument(
        "--sessions",
        type=int,
        default=1,
        help=(
            "Run this many sessions, each in its own interpreter and with its "
            "own arm-ordering seed. Ratios that vary between runs can only be "
            "characterised this way: repeating blocks inside one process shares "
            "the Python imports, module state, warmed allocators and arm "
            "ordering that make two measurements agree. Use enough sessions to "
            "report a median and an interquartile range -- three is too few, and "
            "produces an interval that moves between attempts."
        ),
    )
    parser.add_argument("--seed", type=int, default=ORDER_SEED, help="Arm-ordering seed for this session.")
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Emit only the results JSON on stdout (used for child sessions).",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help=(
            "Promote this run to reference_timings.json, the committed archive "
            "the manuscript's numbers are drawn from. Requires all blocks and at "
            "least "
            f"{ARCHIVE_MIN_SESSIONS} sessions, so a partial or single-session run "
            "cannot replace the canonical evidence by accident."
        ),
    )
    parser.add_argument(
        "--reaggregate",
        type=Path,
        help=(
            "Recompute the across-session summaries from an existing archive's "
            "stored per-session results and rewrite it. Adds new aggregates "
            "without re-timing anything."
        ),
    )
    args = parser.parse_args()

    if args.reaggregate is not None:
        payload = json.loads(args.reaggregate.read_text())
        payload["across_sessions"] = _aggregate(payload.get("sessions", []))
        args.reaggregate.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload["across_sessions"], indent=2))
        print(f"\n[draft-timings] reaggregated {args.reaggregate}")
        return

    ACTIVE_SEED = args.seed
    selected = [args.only] if args.only else list(BLOCKS)

    if args.archive:
        problems = []
        if args.only:
            problems.append(f"--only {args.only} runs one block; the archive must contain all of them")
        if args.sessions < ARCHIVE_MIN_SESSIONS:
            problems.append(
                f"--sessions {args.sessions} is below the {ARCHIVE_MIN_SESSIONS} the archive requires; "
                "fewer sessions do not pin the medians and quartiles the draft quotes"
            )
        if problems:
            raise SystemExit(
                "refusing to overwrite the committed archive:\n  - "
                + "\n  - ".join(problems)
                + f"\n\nRun without --archive to write only {resolve_output_directory('benchmark_draft_timings')}."
            )

    # Child session, or a plain single-session run.
    if args.sessions <= 1:
        session: Dict[str, object] = {"seed": args.seed}
        for name in selected:
            if not args.json_only:
                print(f"[draft-timings] {name} ...", flush=True)
            session[name] = BLOCKS[name]()
        if args.json_only:
            print(json.dumps(session))
            return
        results: Dict[str, object] = {"environment": _environment(), "sessions": [session]}
        results.update(session)
        _write(results, archive=args.archive)
        return

    # Parent: fan out one interpreter per session, each with a distinct seed.
    sessions: List[Dict[str, object]] = []
    for index in range(args.sessions):
        seed = args.seed + index
        print(f"[draft-timings] session {index + 1}/{args.sessions} (seed={seed}) ...", flush=True)
        sessions.append(_run_session_subprocess(seed, selected))

    results = {"environment": _environment(), "sessions": sessions}
    results.update(sessions[0])
    across = _aggregate(sessions)
    results["across_sessions"] = across
    print(json.dumps(across, indent=2))
    _write(results, archive=args.archive)


def _maxsma_step_ratios(sessions: Sequence[Dict[str, object]], lo: str, hi: str) -> List[float]:
    """Per-session wall-time ratio between two ``maxsma`` settings."""
    out: List[float] = []
    for session in sessions:
        block = session.get("maxsma")
        if not isinstance(block, dict):
            continue
        try:
            out.append(float(block[hi]["median_ms"]) / float(block[lo]["median_ms"]))
        except (KeyError, TypeError, ZeroDivisionError):
            continue
    return out


def _summarize_values(values: Sequence[float], digits: int = 4) -> Dict[str, object] | None:
    """Median and quartiles across sessions.

    ``digits`` exists because the default rounding is chosen for ratios of order
    one; the accuracy sweep reports quantities down to ~1e-4, which four decimal
    places would flatten to one digit or to zero.
    """
    if len(values) < 2:
        return None
    ordered = sorted(values)
    if len(ordered) >= 4:
        quartiles = statistics.quantiles(ordered, n=4)
        q1, q3 = quartiles[0], quartiles[2]
    else:
        q1, q3 = ordered[0], ordered[-1]
    return {
        "n_sessions": len(ordered),
        "min": round(ordered[0], digits),
        "q1": round(q1, digits),
        "median": round(statistics.median(ordered), digits),
        "q3": round(q3, digits),
        "max": round(ordered[-1], digits),
        "values": [round(v, digits) for v in ordered],
    }


def _aggregate(sessions: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Across-session summaries for every ratio the draft quotes."""
    across = {
        "lazy_gradient_runtime_ratio": _collect_ratio(
            sessions, ("lazy_gradient", "runtime_ratio_vs_classical", "median")
        ),
        "ea_isofit_ratio": _collect_ratio(
            sessions, ("ea_isofit", "eccentric_anomaly_plus_isofit", "ratio_to_default", "median")
        ),
        "joint_geometry_B24_over_B12": _collect_ratio(
            sessions, ("joint_solve_bands", "ols", "geometry_B=24", "ratio_to_previous", "median")
        ),
        # Wall-time growth per maxsma doubling. Quoting per-session extremes of
        # these was how a stale three-session range survived into the draft;
        # the median and quartiles are the reproducible summary.
        "maxsma_ratio_100_to_200": _summarize_values(_maxsma_step_ratios(sessions, "maxsma=100", "maxsma=200")),
        "maxsma_ratio_200_to_400": _summarize_values(_maxsma_step_ratios(sessions, "maxsma=200", "maxsma=400")),
        # The two step ratios come from the *same* sessions, so the paired
        # difference -- not the gap between their marginal spreads -- is the
        # quantity with any power. Stored so the draft's paired claim is
        # archive-derived rather than recomputed by hand.
        "maxsma_paired_difference": _paired_difference(sessions),
        # Cold start is measured per session; these pool all of them so the
        # quoted brackets come from an across-session statistic rather than
        # from one session's median plus a guessed margin.
        "cold_start_import_s": _cold_start_across(sessions, "import"),
        "cold_start_compilation_s": _cold_start_across(sessions, "compilation"),
        "cold_start_cache_load_s": _cold_start_across(sessions, "cache_load"),
        # Every cell of the §1.3.2 accuracy table, pooled over the sessions'
        # independent noise realisations.
        "snr_sweep": _snr_sweep_across(sessions),
        # The NumPy fallback's steady-state cost, and the fit count at which
        # paying for compilation starts to win.
        "numba_fallback_slowdown": _collect_ratio(sessions, ("numba_fallback", "fallback_slowdown")),
        "numba_fallback_penalty_s": _collect_ratio(sessions, ("numba_fallback", "fallback_penalty_s")),
        "numba_fallback_break_even_fits": _break_even_fits(sessions),
    }
    return {key: value for key, value in across.items() if value is not None}


SNR_SWEEP_COLUMNS = (
    "median_abs_delta_sigma",
    "max_abs_delta_sigma",
    "max_abs_dx0_px",
    "max_abs_dpa_rad",
)


def _snr_sweep_across(sessions: Sequence[Dict[str, object]]) -> Dict[str, object] | None:
    """Pool each accuracy-table cell across sessions, one summary per cell."""
    out: Dict[str, object] = {}
    for snr in SNR_SWEEP_LEVELS:
        row_key = f"snr={snr:g}"
        row: Dict[str, object] = {}
        for column in SNR_SWEEP_COLUMNS:
            values: List[float] = []
            for session in sessions:
                block = session.get("snr_sweep")
                if isinstance(block, dict) and isinstance(block.get(row_key), dict):
                    values.append(float(block[row_key][column]))  # type: ignore[index]
            summary = _summarize_values(values, digits=8)
            if summary is not None:
                row[column] = summary
        if row:
            out[row_key] = row
    return out or None


def _paired_difference(sessions: Sequence[Dict[str, object]]) -> Dict[str, object] | None:
    """Per-session (200→400 ratio) − (100→200 ratio), with a sign count."""
    first = _maxsma_step_ratios(sessions, "maxsma=100", "maxsma=200")
    second = _maxsma_step_ratios(sessions, "maxsma=200", "maxsma=400")
    if len(first) != len(second) or len(first) < 2:
        return None
    differences = [b - a for a, b in zip(first, second)]
    summary = _summarize_values(differences)
    if summary is None:
        return None
    summary["n_positive"] = sum(1 for value in differences if value > 0)
    return summary


def _break_even_fits(sessions: Sequence[Dict[str, object]]) -> Dict[str, object] | None:
    """How many fits a process must run before compiling beats the fallback.

    Compiling costs a one-off ``C`` and then saves ``P`` on every fit relative
    to the NumPy fallback, so after ``n`` fits it is ahead once ``n > C / P``.
    Both terms are measured in the *same* session --- the compilation cost by
    ``cold_start``, the per-fit penalty by ``numba_fallback`` --- because they
    are the two halves of one comparison and mixing sessions would pair a fast
    machine's compile with a slow machine's fit.
    """
    values: List[float] = []
    for session in sessions:
        cold = session.get("cold_start")
        fallback = session.get("numba_fallback")
        if not isinstance(cold, dict) or not isinstance(fallback, dict):
            continue
        try:
            compilation_s = float(cold["compilation"]["median_ms"]) / 1000.0
            penalty_s = float(fallback["fallback_penalty_s"])
        except (KeyError, TypeError):
            continue
        # A non-positive penalty means the fallback was not slower in that
        # session, so there is no crossing point to report rather than an
        # infinite or negative one.
        if penalty_s > 0 and compilation_s > 0:
            values.append(compilation_s / penalty_s)
    return _summarize_values(values, digits=1)


def _cold_start_across(sessions: Sequence[Dict[str, object]], component: str) -> Dict[str, object] | None:
    """Pool one cold-start component's per-session medians, in seconds."""
    values: List[float] = []
    for session in sessions:
        block = session.get("cold_start")
        if isinstance(block, dict) and component in block:
            values.append(float(block[component]["median_ms"]) / 1000.0)
    return _summarize_values(values)


ARCHIVE_PATH = Path(__file__).resolve().parent / "reference_timings.json"


def _write(results: Dict[str, object], archive: bool = False) -> None:
    payload = json.dumps(results, indent=2)
    out_path = Path(resolve_output_directory("benchmark_draft_timings")) / "timings.json"
    out_path.write_text(payload)
    print(f"\n[draft-timings] wrote {out_path}")
    if archive:
        ARCHIVE_PATH.write_text(payload)
        print(f"[draft-timings] archived {ARCHIVE_PATH}")


if __name__ == "__main__":
    main()
