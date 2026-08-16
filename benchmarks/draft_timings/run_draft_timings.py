"""Reproduce every timing quoted in the publication technical draft.

Each block here backs one table or number in ``docs/publication/draft``. The
point is that a reader (or a reviewer on different hardware) can re-run this and
get their own numbers rather than having to trust ours: absolute times are
machine-specific, so the draft quotes ratios wherever a ratio is what the
argument needs.

Blocks
------
``lazy_gradient``
    Section 1.3.2: gradient-evaluation count and wall time with and without the
    lazy-gradient cache, plus the geometry differences between the two, in units
    of each isophote's own reported uncertainty.
``ea_isofit``
    Section 1.4.1.4: the 2x2 of eccentric-anomaly sampling and ISOFIT.
``joint_solve_bands``
    Section 1.6.5: one joint multi-band solve as a function of band count.
``maxsma``
    Section 1.6.5: whole-fit cost against ``maxsma``, with the ring count and
    the aggregate sample count, which scale differently.

Usage::

    uv run python benchmarks/draft_timings/run_draft_timings.py
    uv run python benchmarks/draft_timings/run_draft_timings.py --only joint_solve_bands

Results are written to ``outputs/benchmark_draft_timings/timings.json`` together
with the environment (Python, NumPy, threading and BLAS configuration, whether
Numba was importable) needed to interpret them.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isoster import fit_image  # noqa: E402
from isoster.config import IsosterConfig  # noqa: E402
from isoster.output_paths import resolve_output_directory  # noqa: E402

# One fixture for every whole-fit block, so the blocks are comparable.
FIXTURE = dict(R_e=30.0, n=2.0, I_e=100.0, eps=0.4, pa=0.6, noise_snr=50.0, seed=7)
REPEATS = 7


def _timeit(call: Callable[[], object], repeats: int = REPEATS) -> float:
    """Median wall time in ms, after one warm-up call.

    The median rather than the minimum: we are describing what a user typically
    waits, not the best case the machine can be coaxed into.
    """
    call()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e3


def _build_fixture():
    from tests.fixtures.sersic_factory import create_sersic_model

    image, _, _ = create_sersic_model(**FIXTURE)
    centre = float(image.shape[0] // 2)
    return image, centre


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

    image, centre = _build_fixture()
    original = fitting_module.compute_gradient
    counter = {"n": 0}

    def counted(*args, **kwargs):
        counter["n"] += 1
        return original(*args, **kwargs)

    out: Dict[str, object] = {}
    profiles = {}
    for lazy in (True, False):
        cfg = _config(centre, maxsma=120.0, use_lazy_gradient=lazy)
        fitting_module.compute_gradient = counted
        counter["n"] = 0
        fit_image(image, config=cfg)
        gradient_calls = counter["n"]
        fitting_module.compute_gradient = original

        wall_ms = _timeit(lambda: fit_image(image, config=cfg))
        profiles[lazy] = fit_image(image, config=cfg)["isophotes"]
        out["lazy" if lazy else "classical"] = {
            "gradient_evaluations": gradient_calls,
            "wall_ms": round(wall_ms, 2),
        }

    # Geometry difference, expressed in each isophote's own reported sigma. That
    # is the only scale on which "how much does the approximation cost" has an
    # answer a reader can act on.
    ratios: List[float] = []
    for lazy_iso, exact_iso in zip(profiles[True], profiles[False]):
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
    image, centre = _build_fixture()
    out: Dict[str, object] = {}
    for label, overrides in (
        ("default_phi_posthoc", {}),
        ("eccentric_anomaly_only", {"use_eccentric_anomaly": True}),
        ("isofit_in_loop_only", {"simultaneous_harmonics": True, "isofit_mode": "in_loop"}),
        (
            "eccentric_anomaly_plus_isofit",
            {"use_eccentric_anomaly": True, "simultaneous_harmonics": True, "isofit_mode": "in_loop"},
        ),
    ):
        cfg = _config(centre, maxsma=120.0, **overrides)
        out[label] = {"wall_ms": round(_timeit(lambda c=cfg: fit_image(image, config=c)), 2)}
    baseline = out["default_phi_posthoc"]["wall_ms"]
    for value in out.values():
        value["ratio_to_default"] = round(value["wall_ms"] / baseline, 3)
    return out


def bench_joint_solve_bands() -> Dict[str, object]:
    """Section 1.6.5: one joint solve against band count, at fixed ring size.

    Isolating the solve (rather than a whole multi-band fit) is deliberate: it is
    the term whose scaling the draft makes a claim about.
    """
    from isoster.multiband.fitting_mb import fit_first_and_second_harmonics_joint

    n_samples = 700
    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    rng = np.random.default_rng(0)

    out: Dict[str, object] = {"n_samples_per_band": n_samples}
    previous = None
    for n_bands in (3, 6, 12, 24):
        intens = rng.normal(size=(n_bands, n_samples))
        weights = np.ones(n_bands)
        wall_ms = _timeit(
            lambda a=angles, i=intens, w=weights: fit_first_and_second_harmonics_joint(a, i, w),
            repeats=41,
        )
        entry = {"wall_ms": round(wall_ms, 4)}
        if previous is not None:
            entry["ratio_to_previous"] = round(wall_ms / previous, 2)
        out[f"B={n_bands}"] = entry
        previous = wall_ms
    return out


def bench_maxsma() -> Dict[str, object]:
    """Section 1.6.5: rings grow logarithmically, sampling work does not.

    Both are recorded because they scale differently and the draft's claim turns
    on the distinction: under geometric stepping the ring *count* grows like
    log(maxsma), but each ring carries ``max(64, 2*pi*a)`` samples, so the
    aggregate sample count still grows roughly linearly in ``maxsma``.
    """
    image, centre = _build_fixture()
    out: Dict[str, object] = {}
    for maxsma in (100.0, 200.0, 400.0):
        cfg = _config(centre, maxsma=maxsma)
        result = fit_image(image, config=cfg)
        isophotes = result["isophotes"]
        total_samples = sum(max(64, int(2 * np.pi * float(iso["sma"]))) for iso in isophotes if iso.get("sma"))
        out[f"maxsma={maxsma:g}"] = {
            "wall_ms": round(_timeit(lambda c=cfg: fit_image(image, config=c)), 2),
            "n_isophotes": len(isophotes),
            "total_ring_samples": total_samples,
        }
    return out


BLOCKS: Dict[str, Callable[[], Dict[str, object]]] = {
    "lazy_gradient": bench_lazy_gradient,
    "ea_isofit": bench_ea_isofit,
    "joint_solve_bands": bench_joint_solve_bands,
    "maxsma": bench_maxsma,
}


def _environment() -> Dict[str, object]:
    """Everything needed to interpret an absolute timing."""
    try:
        import numba

        numba_version = numba.__version__
    except ImportError:
        numba_version = None

    blas = {}
    try:
        config = np.__config__.show(mode="dicts")  # type: ignore[call-arg]
        blas = {name: info.get("name") for name, info in config.get("Build Dependencies", {}).items()}
    except Exception:
        blas = {"note": "numpy build configuration unavailable"}

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "numba": numba_version,
        "blas": blas,
        "repeats": REPEATS,
        "fixture": FIXTURE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(BLOCKS), help="Run a single block.")
    args = parser.parse_args()

    selected = [args.only] if args.only else list(BLOCKS)
    results: Dict[str, object] = {"environment": _environment()}
    for name in selected:
        print(f"[draft-timings] {name} ...", flush=True)
        results[name] = BLOCKS[name]()

    out_dir = resolve_output_directory("benchmark_draft_timings")
    out_path = Path(out_dir) / "timings.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\n[draft-timings] wrote {out_path}")


if __name__ == "__main__":
    main()
