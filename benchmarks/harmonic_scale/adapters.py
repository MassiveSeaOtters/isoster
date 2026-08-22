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
