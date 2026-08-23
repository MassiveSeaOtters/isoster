"""Evaluate natural end-to-end profiles on one frozen radial grid.

The three programs choose different radii. Comparing every returned ring would
therefore give the program with fewer rings fewer opportunities to fail. This
module converts each successfully returned profile to the five pre-registered
evaluation radii, without extrapolation.

Geometry and scalar ring quantities are linear in log semi-major axis. Position
angle is first unwrapped with its pi periodicity. Harmonics need one extra step:
their reported components are relative to the moving major axis, so they are
rotated into a fixed sky frame before interpolation and rotated back on the
interpolated aperture afterwards.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from benchmarks.harmonic_scale.conventions import rotate_raw_to_major_axis

HARMONIC_ORDERS = (3, 4)
SCALAR_COLUMNS = ("x0", "y0", "eps", "ring_mean")


def canonicalize_pa_and_harmonics(
    pa: float,
    components: Mapping[str, float],
    orders: Iterable[int] = HARMONIC_ORDERS,
) -> tuple[float, dict[str, float]]:
    """Fold PA into ``[0, pi)`` while preserving the represented ring signal.

    A half-turn describes the same ellipse but reverses the angular origin.
    Odd harmonic orders therefore change sign; even orders do not. Folding PA
    without carrying that sign turns a coordinate convention into an apparent
    scientific disagreement.
    """
    pa_value = float(pa)
    if not math.isfinite(pa_value):
        raise ValueError(f"pa must be finite, got {pa!r}")
    canonical_pa = pa_value % math.pi
    half_turns = int(round((pa_value - canonical_pa) / math.pi))
    result = dict(components)
    for order in orders:
        sign = -1.0 if (order * half_turns) % 2 else 1.0
        for prefix in ("s", "c"):
            name = f"{prefix}{order}_raw_major"
            if name not in result:
                raise ValueError(f"harmonic profile is missing {name}")
            value = float(result[name])
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {result[name]!r}")
            result[name] = sign * value
    return canonical_pa, result


def _major_to_sky(s_major: float, c_major: float, order: int, pa: float) -> tuple[float, float]:
    """Inverse of ``rotate_raw_to_major_axis``."""
    return rotate_raw_to_major_axis(s_major, c_major, order=order, pa_rad=-pa)


def _validate_rows(rows: Sequence[Mapping[str, float]], orders: tuple[int, ...]) -> list[dict[str, float]]:
    if len(rows) < 2:
        raise ValueError("profile interpolation requires at least two returned rings")
    required = {"sma", "pa", *SCALAR_COLUMNS}
    required.update(f"{prefix}{order}_raw_major" for order in orders for prefix in ("s", "c"))
    checked = []
    for index, row in enumerate(rows):
        if orders and row.get("harmonic_conversion_valid") is not True:
            raise ValueError(
                f"profile row {index} has no observed-valid harmonic conversion; "
                "invalid or unattributed rows cannot be interpolated"
            )
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"profile row {index} is missing {', '.join(missing)}")
        converted = {name: float(row[name]) for name in required}
        if not all(math.isfinite(value) for value in converted.values()):
            raise ValueError(f"profile row {index} must contain only finite values")
        if converted["sma"] <= 0.0:
            raise ValueError(f"profile row {index} has non-positive sma={converted['sma']!r}")
        if not 0.0 <= converted["eps"] < 1.0:
            raise ValueError(f"profile row {index} has invalid eps={converted['eps']!r}")
        checked.append(converted)
    checked.sort(key=lambda row: row["sma"])
    smas = [row["sma"] for row in checked]
    if len(set(smas)) != len(smas):
        raise ValueError("returned profile radii must be unique")
    return checked


def interpolate_profile_to_evaluation_radii(
    rows: Sequence[Mapping[str, float]],
    evaluation_radii: Sequence[float],
    orders: Iterable[int] = HARMONIC_ORDERS,
) -> list[dict[str, float]]:
    """Interpolate one returned profile to fixed radii, never extrapolating.

    A target outside the returned radial span raises. Stage 2 records the
    original run and its achieved coverage, but it cannot declare the arm's
    fixed-grid accuracy complete.
    """
    order_tuple = tuple(int(order) for order in orders)
    checked = _validate_rows(rows, order_tuple)
    targets = np.asarray(evaluation_radii, dtype=np.float64)
    if targets.ndim != 1 or targets.size == 0 or not np.all(np.isfinite(targets)) or np.any(targets <= 0.0):
        raise ValueError("evaluation radii must be a non-empty sequence of finite positive values")
    if len(set(float(value) for value in targets)) != len(targets):
        raise ValueError("evaluation radii must be unique")

    smas = np.asarray([row["sma"] for row in checked], dtype=np.float64)
    tolerance = 1e-12
    below = float(np.min(targets)) < smas[0] and not math.isclose(
        float(np.min(targets)), float(smas[0]), rel_tol=tolerance, abs_tol=tolerance
    )
    above = float(np.max(targets)) > smas[-1] and not math.isclose(
        float(np.max(targets)), float(smas[-1]), rel_tol=tolerance, abs_tol=tolerance
    )
    if below or above:
        raise ValueError(
            f"evaluation radii [{float(np.min(targets)):g}, {float(np.max(targets)):g}] "
            f"are not bracketed by the returned profile [{smas[0]:g}, {smas[-1]:g}]"
        )
    log_sma = np.log(smas)
    # A formatter may move a boundary by a few ulps. Treat that as the same
    # radius, but never permit scientifically meaningful extrapolation.
    clipped_targets = np.clip(targets, smas[0], smas[-1])
    log_targets = np.log(clipped_targets)

    pa_raw = np.asarray([row["pa"] for row in checked], dtype=np.float64)
    pa_unwrapped = np.unwrap(pa_raw, period=math.pi)
    pa_at = np.interp(log_targets, log_sma, pa_unwrapped)

    scalar_at = {name: np.interp(log_targets, log_sma, [row[name] for row in checked]) for name in SCALAR_COLUMNS}

    sky_at: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for order in order_tuple:
        sky = np.asarray(
            [
                _major_to_sky(row[f"s{order}_raw_major"], row[f"c{order}_raw_major"], order, row["pa"])
                for row in checked
            ],
            dtype=np.float64,
        )
        sky_at[order] = (
            np.interp(log_targets, log_sma, sky[:, 0]),
            np.interp(log_targets, log_sma, sky[:, 1]),
        )

    output = []
    for index, target in enumerate(targets):
        canonical_pa = float(pa_at[index] % math.pi)
        row = {"sma": float(target), "pa": canonical_pa}
        row.update({name: float(values[index]) for name, values in scalar_at.items()})
        for order in order_tuple:
            s_major, c_major = rotate_raw_to_major_axis(
                float(sky_at[order][0][index]),
                float(sky_at[order][1][index]),
                order=order,
                pa_rad=canonical_pa,
            )
            row[f"s{order}_raw_major"] = s_major
            row[f"c{order}_raw_major"] = c_major
        output.append(row)
    return output
