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

REPEATS = 9
MIN_BATCH_SECONDS = 0.02
ORDER_SEED = 20260816
COLD_START_PAIRS = 3

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


def bench_lazy_gradient() -> Dict[str, object]:
    """Section 1.3.2: what the gradient cache saves, and what it costs."""
    import isoster.fitting as fitting_module

    image, centre = _build_fixture(FIXTURE_COMPACT)
    configs = {
        "lazy": _config(centre, maxsma=120.0, use_lazy_gradient=True),
        "classical": _config(centre, maxsma=120.0, use_lazy_gradient=False),
    }

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

    ratios: List[float] = []
    for lazy_iso, exact_iso in zip(profiles["lazy"], profiles["classical"]):
        if not lazy_iso.get("sma"):
            continue
        for key, err_key in (("x0", "x0_err"), ("y0", "y0_err"), ("eps", "eps_err"), ("pa", "pa_err")):
            sigma = float(exact_iso.get(err_key) or 0.0)
            if sigma > 0:
                ratios.append(abs(float(lazy_iso[key]) - float(exact_iso[key])) / sigma)
    out["geometry_difference_in_sigma"] = {
        "median": round(float(np.median(ratios)), 5),
        "max": round(float(np.max(ratios)), 3),
        "n_comparisons": len(ratios),
    }
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


BLOCKS: Dict[str, Callable[[], Dict[str, object]]] = {
    "lazy_gradient": bench_lazy_gradient,
    "ea_isofit": bench_ea_isofit,
    "joint_solve_bands": bench_joint_solve_bands,
    "maxsma": bench_maxsma,
    "cold_start": bench_cold_start,
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
            "Also write the result next to this script as reference_timings.json, "
            "which is committed. The manuscript's numbers should come from an "
            "archived run so they have a stable provenance; the outputs/ copy is "
            "overwritten by every invocation."
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


def _summarize_values(values: Sequence[float]) -> Dict[str, object] | None:
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
    }
    return {key: value for key, value in across.items() if value is not None}


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
