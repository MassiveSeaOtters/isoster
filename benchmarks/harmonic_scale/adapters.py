"""Fixed-aperture measurement, one adapter per tool, all on the same rings.

Raw harmonic amplitudes depend on the exact ellipse they were measured on, so a
convention comparison run at free geometry measures geometry rather than
convention. That failure mode is measured, not hypothetical: on one image
scaled by ten, AutoProf's fitted geometry moved by up to 0.047 in ellipticity
and 8.8 degrees in position angle, and none of 58 rings kept identical
geometry.

Each adapter therefore takes an explicit list of ring requests, imposes them,
and reports back the geometry it actually used so the caller can verify rather
than trust. :func:`assert_rings_match_request` does that check; nothing here
calls it implicitly, because a silent check is one nobody reads.

Every adapter returns the same row schema::

    tool, sma, x0, y0, eps, pa,
    s<n>_raw, c<n>_raw          raw sine/cosine amplitudes, intensity units
    a<n>_bender, b<n>_bender    the same, divided by sma * |dI/da|
    gradient, mean_intensity,
    status                      "measured", or why not

``status`` is the row's own provenance. A row that could not be measured still
appears, carrying NaN amplitudes and a reason, because a dropped row and a
measured-null row are indistinguishable once they leave the adapter.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .conventions import bender_from_raw, raw_from_bender

#: Tolerances for the post-run geometry check. Generous enough for float
#: round-trips through a CSV, far tighter than any drift that would matter.
GEOMETRY_TOLERANCES = {"sma": 1e-6, "x0": 1e-6, "y0": 1e-6, "eps": 1e-6, "pa": 1e-6}


class GeometryMismatch(AssertionError):
    """A tool returned a ring that is not the ring it was asked for."""


def assert_rings_match_request(
    rows: Sequence[dict],
    request: Sequence[dict],
    tolerances: dict | None = None,
) -> None:
    """Verify ring by ring that the imposed geometry survived the round trip.

    Raises :class:`GeometryMismatch` naming the first field that moved. This is
    load-bearing rather than defensive: a silently re-fitted ring invalidates
    every comparison built on it, and the failure is invisible in the output.
    """
    tolerances = tolerances or GEOMETRY_TOLERANCES
    if len(rows) != len(request):
        raise GeometryMismatch(f"expected {len(request)} rings, got {len(rows)}")
    for index, (row, wanted) in enumerate(zip(rows, request)):
        for field, tolerance in tolerances.items():
            if field not in wanted:
                continue
            actual = row.get(field)
            if actual is None or not np.isfinite(actual):
                raise GeometryMismatch(f"ring {index}: {field} is {actual!r}")
            if abs(float(actual) - float(wanted[field])) > tolerance:
                raise GeometryMismatch(
                    f"ring {index}: {field} drifted from {wanted[field]!r} to {actual!r} (tolerance {tolerance})"
                )


def _empty_row(tool: str, wanted: dict, orders: Iterable[int], status: str) -> dict:
    row = {
        "tool": tool,
        "sma": float(wanted["sma"]),
        "x0": float(wanted["x0"]),
        "y0": float(wanted["y0"]),
        "eps": float(wanted["eps"]),
        "pa": float(wanted["pa"]),
        "gradient": float("nan"),
        "mean_intensity": float("nan"),
        "status": status,
    }
    for order in orders:
        for key in (f"s{order}_raw", f"c{order}_raw", f"a{order}_bender", f"b{order}_bender"):
            row[key] = float("nan")
    return row


def measure_isoster_fixed(
    image: np.ndarray,
    request: Sequence[dict],
    orders: Iterable[int],
    mask: np.ndarray | None = None,
) -> list[dict]:
    """Measure the requested rings with isoster's template-based forced mode.

    ``fit_image(..., template=...)`` imposes the geometry and skips the
    geometry solve. ``debug=True`` is required, not optional: the gradient is
    what turns a Bender coefficient back into a raw amplitude, and it is only
    exposed under debug.

    Depends on the forced-harmonic fix on this branch. Before it, this path
    returned ``0.0`` for every harmonic and would have contributed fabricated
    zeros to the comparison.
    """
    from isoster import fit_image
    from isoster.config import IsosterConfig

    orders = list(orders)
    config = IsosterConfig(
        compute_deviations=True,
        harmonic_orders=orders,
        debug=True,
        use_eccentric_anomaly=False,
    )
    template = [
        {
            "sma": float(r["sma"]),
            "x0": float(r["x0"]),
            "y0": float(r["y0"]),
            "eps": float(r["eps"]),
            "pa": float(r["pa"]),
        }
        for r in request
    ]
    isophotes = fit_image(image, mask=mask, config=config, template=template)["isophotes"]

    rows = []
    for wanted, iso in zip(request, isophotes):
        gradient = iso.get("grad", float("nan"))
        if not np.isfinite(gradient) or gradient == 0.0:
            rows.append(_empty_row("isoster", wanted, orders, "gradient_unavailable"))
            continue
        row = {
            "tool": "isoster",
            "sma": float(iso["sma"]),
            "x0": float(iso["x0"]),
            "y0": float(iso["y0"]),
            "eps": float(iso["eps"]),
            "pa": float(iso["pa"]),
            "gradient": float(gradient),
            "mean_intensity": float(iso["intens"]),
            "status": "measured",
        }
        for order in orders:
            a_n, b_n = iso.get(f"a{order}", np.nan), iso.get(f"b{order}", np.nan)
            if not np.isfinite(a_n) or not np.isfinite(b_n):
                rows.append(_empty_row("isoster", wanted, orders, "harmonic_unmeasured"))
                break
            s_raw, c_raw = raw_from_bender(a_n, b_n, row["sma"], gradient)
            row[f"s{order}_raw"], row[f"c{order}_raw"] = s_raw, c_raw
            row[f"a{order}_bender"], row[f"b{order}_bender"] = float(a_n), float(b_n)
        else:
            rows.append(row)
    return rows


def measure_photutils_fixed(
    image: np.ndarray,
    request: Sequence[dict],
    orders: Iterable[int],
    mask: np.ndarray | None = None,
) -> list[dict]:
    """Measure the requested rings with photutils, one ``EllipseSample`` each.

    ``Ellipse.fit_image(fix_center=..., fix_pa=..., fix_eps=...)`` fixes the
    *shape* but still chooses its own radial grid, so it cannot guarantee
    matched radii. Building the sample per requested ``sma`` can.

    ``sample.update(...)`` is required before constructing the ``Isophote``:
    without it the sample has extracted no intensities and computed no
    gradient, and the constructor raises ``TypeError: 'NoneType' object is not
    subscriptable``.
    """
    from photutils.isophote import EllipseGeometry, EllipseSample, Isophote

    orders = list(orders)
    masked_image = image
    if mask is not None:
        masked_image = np.where(np.asarray(mask, dtype=bool), np.nan, image)

    rows = []
    for wanted in request:
        geometry = EllipseGeometry(
            x0=float(wanted["x0"]),
            y0=float(wanted["y0"]),
            sma=float(wanted["sma"]),
            eps=float(wanted["eps"]),
            pa=float(wanted["pa"]),
        )
        try:
            sample = EllipseSample(masked_image, sma=float(wanted["sma"]), geometry=geometry)
            sample.update(fixed_parameters=np.ones(4, dtype=bool))
            isophote = Isophote(sample, niter=0, valid=True, stop_code=0)
        except Exception:  # noqa: BLE001 - any failure means this ring is unmeasured
            rows.append(_empty_row("photutils", wanted, orders, "sample_failed"))
            continue

        gradient = isophote.grad
        if gradient is None or not np.isfinite(gradient) or gradient == 0.0:
            rows.append(_empty_row("photutils", wanted, orders, "gradient_unavailable"))
            continue

        row = {
            "tool": "photutils",
            "sma": float(isophote.sma),
            "x0": float(isophote.x0),
            "y0": float(isophote.y0),
            "eps": float(isophote.eps),
            "pa": float(isophote.pa),
            "gradient": float(gradient),
            "mean_intensity": float(isophote.intens),
            "status": "measured",
        }
        # photutils computes a3/b3/a4/b4 as attributes; anything else has to be
        # solved here with its own fitter, on the same samples it used.
        from photutils.isophote.harmonics import fit_upper_harmonic

        failed = False
        for order in orders:
            try:
                coefficients, _ = fit_upper_harmonic(sample.values[0], sample.values[2], order)
            except Exception:  # noqa: BLE001
                failed = True
                break
            s_raw, c_raw = float(coefficients[1]), float(coefficients[2])
            row[f"s{order}_raw"], row[f"c{order}_raw"] = s_raw, c_raw
            a_n, b_n = bender_from_raw(s_raw, c_raw, row["sma"], gradient)
            row[f"a{order}_bender"], row[f"b{order}_bender"] = a_n, b_n
        if failed:
            rows.append(_empty_row("photutils", wanted, orders, "harmonic_solve_failed"))
        else:
            rows.append(row)
    return rows


def measure_autoprof_fixed(
    image: np.ndarray,
    request: Sequence[dict],
    orders: Iterable[int],
    workspace: str,
    pixel_scale: float = 1.0,
    zeropoint: float = 22.5,
    isoclip: bool = True,
    interpolate_start: float = 5.0,
    set_psf: float | None = None,
    venv_python: str | None = None,
    timeout: int = 600,
    extra_options: dict | None = None,
) -> tuple[list[dict], dict]:
    """Measure the requested rings with AutoProf's forced pipeline.

    Runs :mod:`benchmarks.harmonic_scale.autoprof_worker` in the isolated venv,
    because AutoProf pins ``numpy<2``. See that module for why the forced
    pipeline needs an explicit ``UpdatePipeline``, a companion ``.aux`` even
    with ``ap_set_center``, and per-ring sampling-mode instrumentation.

    Returns ``(rows, provenance)``. Rows carry the native AutoProf
    coefficients alongside the reconstructed raw and Bender values, and every
    row records which angle basis produced it -- the conversion is only valid
    for the polar-resampled path.

    The Bender values are *not* filled here. AutoProf reports no radial
    gradient, and inventing one from its own profile is a separate decision
    that A4's measurement has to license first. Raw amplitudes are exact and
    are the primary track.

    ``interpolate_start`` is the campaign's largest single effect and is a
    grid axis rather than a setting to inherit -- 5.0 is AutoProf's own
    default, reproduced here so that omitting the argument reproduces stock
    behaviour rather than a quietly better one. It multiplies
    ``results["psf fwhm"]``, which on this pipeline is the assumed 4.0 px
    rather than a measured value. Each returned row still carries the mode its
    ring actually got (``harmonic_sampling_mode``), observed rather than
    inferred from the setting.
    """
    import json
    import subprocess
    from pathlib import Path

    from benchmarks.autoprof_env import resolve_autoprof_python

    from .conventions import raw_from_autoprof, rotate_raw_to_major_axis

    orders = list(orders)
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    rings = [
        {
            "sma_pix": float(r["sma"]),
            "eps": float(r["eps"]),
            # AutoProf takes astronomical degrees; isoster stores math radians.
            "pa_deg_astro": (np.degrees(float(r["pa"])) - 90.0) % 180.0,
        }
        for r in request
    ]
    job = {
        "name": "harmonic_scale",
        "output_dir": str(workspace),
        "result_path": str(workspace / "result.json"),
        "rings": rings,
        "orders": orders,
        "pixel_scale": pixel_scale,
        "zeropoint": zeropoint,
        "isoclip": bool(isoclip),
        "interpolate_start": float(interpolate_start),
        # None leaves AutoProf's assumed 4.0 px in place, which is stock
        # behaviour; a value sets ap_set_psf and moves the switch radius.
        "set_psf": None if set_psf is None else float(set_psf),
        "x0": float(request[0]["x0"]),
        "y0": float(request[0]["y0"]),
        "image": np.asarray(image, dtype=np.float64).tolist(),
        "extra_options": dict(extra_options or {}),
    }
    job_path = workspace / "job.json"
    job_path.write_text(json.dumps(job))

    worker = Path(__file__).with_name("autoprof_worker.py")
    completed = subprocess.run(
        [resolve_autoprof_python(venv_python), str(worker), str(job_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"autoprof worker failed (exit {completed.returncode}):\n{completed.stderr[-2000:]}")

    payload = json.loads((workspace / "result.json").read_text())
    provenance = dict(payload["provenance"])
    provenance["sampling_mode"] = payload["sampling_mode"]

    # The angle basis is decided by the same flag that chooses the resampling,
    # and it decides whether the PA rotation applies at all.
    basis = "polar_from_image_x_axis" if isoclip else "eccentric_anomaly"
    provenance["harmonic_basis"] = basis

    rows = []
    for wanted, entry in zip(request, payload["rows"]):
        pa_rad = float(wanted["pa"])
        row = {
            "tool": "autoprof",
            "sma": entry["sma_pix"],
            "x0": float(wanted["x0"]),
            "y0": float(wanted["y0"]),
            "eps": entry["eps"],
            "pa": pa_rad,
            "gradient": float("nan"),
            "mean_intensity": entry["b0"],
            "status": "measured",
            "harmonic_basis": basis,
            # Per ring, observed rather than derived from the setting. The
            # threshold is predictable on this pipeline (AutoProf assumes a
            # 4.0 px PSF), but the per-ring mode is the measurand, and
            # recomputing it here would check our arithmetic rather than
            # AutoProf's behaviour. ``None`` means the probe could not
            # attribute the call to this ring, which is not "nearest pixel".
            "harmonic_sampling_mode": (
                "line_nearest_pixel"
                if entry.get("interpolated") is False
                else "line_interpolated"
                if entry.get("interpolated") is True
                else "unknown"
            ),
            "rad_interp_pix": entry.get("rad_interp_pix"),
            "autoprof_b0": entry["b0"],
            "autoprof_a0": entry["a0"],
            "autoprof_median_flux": entry.get("median_flux", float("nan")),
        }
        for order in orders:
            native_a, native_b = entry[f"a{order}"], entry[f"b{order}"]
            row[f"autoprof_a{order}_native"] = native_a
            row[f"autoprof_b{order}_native"] = native_b
            s_sky, c_sky = raw_from_autoprof(native_a, native_b, entry["b0"])
            if basis == "polar_from_image_x_axis":
                s_raw, c_raw = rotate_raw_to_major_axis(s_sky, c_sky, order=order, pa_rad=pa_rad)
            else:
                # A basis change, not a rotation: it mixes harmonic orders, so
                # no same-order two-component transform can express it. Leave
                # the sky-frame values and let the caller decide.
                s_raw, c_raw = s_sky, c_sky
            row[f"s{order}_raw"], row[f"c{order}_raw"] = s_raw, c_raw
            row[f"a{order}_bender"] = float("nan")
            row[f"b{order}_bender"] = float("nan")
        rows.append(row)
    return rows, provenance
